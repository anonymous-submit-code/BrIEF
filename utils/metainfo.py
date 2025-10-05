import os
import re
import json
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Tuple

IDENTIFIER_KEY_PATTERNS = [
    "model", "asin", "part number", "item number", "sku", "upc", "ean", "gtin", "isbn"
]

MOVIES_STOPLIST = {
    "Movies & TV",
    "Featured Categories",
    "Studio Specials",
    "Genre for Featured Categories",
}
CDS_STOPLABEL = "CDs & Vinyl"


def _normalize_ws(s: str) -> str:
    """Collapse excessive whitespace and strip."""
    return re.sub(r"\s+", " ", s or "").strip()


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Deduplicate strings while preserving order (after whitespace normalization)."""
    seen = set()
    out = []
    for it in items:
        norm = _normalize_ws(it)
        if not norm:
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _is_identifier_key(key: str) -> bool:
    kl = key.lower()
    return any(pat in kl for pat in IDENTIFIER_KEY_PATTERNS)


def _format_best_sellers_rank(v: Any) -> Optional[str]:
    """
    v is usually a dict mapping {category: rank_int}. We want the lowest rank and its category.
    Returns like '#9107 in Electric Guitar Parts' or None if not usable.
    """
    if isinstance(v, dict) and v:
        pairs: List[Tuple[str, int]] = []
        for cat, rank in v.items():
            try:
                r = int(str(rank).replace(",", ""))
                pairs.append((cat, r))
            except Exception:
                continue
        if pairs:
            cat, r = min(pairs, key=lambda x: x[1])
            return f"#{r} in {cat}"
    return None


def _prefer_color_name(details: Dict[str, Any]) -> Dict[str, Any]:
    """If both 'Color Name' and 'Color' exist, drop 'Color' and keep 'Color Name'."""
    keys_lower = {k.lower(): k for k in details.keys()}
    if "color name" in keys_lower and "color" in keys_lower:
        details = {k: v for k, v in details.items() if k.lower() != "color"}
    return details


def _format_details(details: Dict[str, Any]) -> str:
    """
    Build 'key: value' pairs separated by '; ' with special rules:
      - Skip identifier-type keys (e.g., Item model number, ASIN, etc.)
      - Prefer Color Name over Color when both exist
      - Best Sellers Rank: show lowest rank as '#N in Category'
      - Skip empty/null values
    """
    if not isinstance(details, dict) or not details:
        return ""

    details = _prefer_color_name(details)

    parts: List[str] = []
    for key in sorted(details.keys()):
        val = details[key]
        if val is None:
            continue
        if _is_identifier_key(key):
            continue

        if key == "Best Sellers Rank":
            bsr = _format_best_sellers_rank(val)
            if bsr:
                parts.append(f"{key}: {bsr}")
            continue

        if isinstance(val, (list, tuple)):
            vals = [_normalize_ws(str(x)) for x in val if _normalize_ws(str(x))]
            if not vals:
                continue
            v_str = ", ".join(vals)
        elif isinstance(val, dict):
            v_str = _normalize_ws(json.dumps(val, ensure_ascii=False, separators=(",", ":")))
        else:
            v_str = _normalize_ws(str(val))

        if not v_str:
            continue

        parts.append(f"{key}: {v_str}")

    return "; ".join(parts)


def _format_features(features: List[str], max_chars: int = 3000) -> str:
    """
    Numbered points, each on its own line, but cap the *features block* to <= max_chars.
    (Only counting the lines themselves; headers the caller adds are not counted here.)
    """
    if not isinstance(features, list) or not features:
        return ""
    lines: List[str] = []
    used = 0
    n = 1
    for feat in features:
        f = _normalize_ws(str(feat))
        if not f:
            continue
        line = f"{n}. {f}"
        extra = (1 if lines else 0) + len(line)
        if used + extra > max_chars:
            break
        if lines:
            lines.append("\n" + line)
            used += 1 + len(line)
        else:
            lines.append(line)
            used += len(line)
        n += 1
    return "".join(lines)


def _format_description(desc_list: List[str], max_chars: int = 2000) -> str:
    """Deduplicate and whitespace-normalize, join with spaces, then truncate."""
    if not isinstance(desc_list, list) or not desc_list:
        return ""
    deduped = _dedupe_preserve_order([str(x) for x in desc_list])
    if not deduped:
        return ""
    paragraph = _normalize_ws(" ".join(deduped))
    if len(paragraph) > max_chars:
        paragraph = paragraph[:max_chars].rstrip()
    return paragraph


def _stoplist_for_file(path: str) -> set:
    """Decide which category labels to filter out based on the filename."""
    name = os.path.basename(path).lower()
    if ("movie" in name) or ("tv" in name):
        return set(MOVIES_STOPLIST)
    if ("cds" in name) or ("vinyl" in name):
        return {CDS_STOPLABEL}
    return set()


def _format_categories(cats: List[str], stoplist: set) -> str:
    """Filter out unwanted category labels and join the rest with ' > '."""
    if not isinstance(cats, list) or not cats:
        return ""
    filtered: List[str] = []
    for c in cats:
        if isinstance(c, str):
            s = c.strip()
            if s and s not in stoplist:
                filtered.append(s)
    return " > ".join(filtered)


def _extract_author(obj: Dict[str, Any]) -> str:
    """Return a normalized 'author' string if available."""
    a = obj.get("author", None)
    if isinstance(a, dict):
        name = a.get("name")
        if name:
            s = _normalize_ws(str(name))
            if s:
                return s
    elif isinstance(a, str):
        s = _normalize_ws(a)
        if s:
            return s
    elif isinstance(a, (list, tuple)):
        vals = [_normalize_ws(str(x)) for x in a if _normalize_ws(str(x))]
        if vals:
            return ", ".join(vals)
    det = obj.get("details") or {}
    for k in ("Artists", "Artist", "Composer", "Actors", "Director"):
        if k in det:
            v = det[k]
            if isinstance(v, (list, tuple)):
                vals = [_normalize_ws(str(x)) for x in v if _normalize_ws(str(x))]
                if vals:
                    return ", ".join(vals)
            else:
                s = _normalize_ws(str(v))
                if s:
                    return s
    return ""


def build_item_texts(jsonl_path: str, start_idx: int = 0) -> pd.DataFrame:
    stoplist = _stoplist_for_file(jsonl_path)

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        idx_counter = start_idx
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj: Dict[str, Any] = json.loads(line)

            parent_asin = obj.get("parent_asin")
            parts: List[str] = []
            parts.append(f"Item ID: {idx_counter}")

            # Title (include subtitle if present; omit line if title empty).
            title = _normalize_ws(obj.get("title", ""))
            subtitle = _normalize_ws(obj.get("subtitle", "")) if obj.get("subtitle") is not None else ""
            if title:
                title_full = f"{title}: {subtitle}" if subtitle else title
                parts.append(f"Title: {title_full}")

            # Author.
            author = _extract_author(obj)
            if author:
                parts.append(f"Author: {author}")

            # Average rating and rating number.
            avg = obj.get("average_rating")
            num = obj.get("rating_number")
            avg_str = f"{float(avg):.1f}" if isinstance(avg, (int, float)) else _normalize_ws(str(avg or ""))
            num_str = f"{int(num)}" if isinstance(num, (int, float)) else _normalize_ws(str(num or ""))
            parts.append(f"Average rating: {avg_str}; rating number: {num_str}")

            # Categories (filtered by filename-based stoplist).
            cats_str = _format_categories(obj.get("categories"), stoplist)
            if cats_str:
                parts.append(f"Categories: {cats_str}")

            # Features (numbered, capped to 3000 chars).
            feat_str = _format_features(obj.get("features"), max_chars=800)
            if feat_str:
                parts.append("Features:")
                parts.append(feat_str)

            # Description (deduped, normalized, capped to 3000 chars).
            desc_str = _format_description(obj.get("description"), max_chars=800)
            if desc_str:
                parts.append(f"Description: {desc_str}")

            # Details.
            details_str = _format_details(obj.get("details") or {})
            if details_str:
                parts.append(f"Details: {details_str}")

            texts = "\n".join(parts)

            rows.append({
                "parent_asin": parent_asin,
                "idx": idx_counter,
                "texts": texts
            })
            idx_counter += 1

    return pd.DataFrame(rows, columns=["parent_asin", "idx", "texts"])