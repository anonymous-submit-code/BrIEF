import pandas as pd
from typing import Any, Dict, Iterable, List, Tuple

def _normalize_ws(s: str) -> str:
    return " ".join(str(s).split()).strip()

def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        t = _normalize_ws(it)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

def _parse_history_asins(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        toks = [t for t in val.strip().split() if t]
        return _dedupe_preserve_order(toks)
    if isinstance(val, (list, tuple)):
        return _dedupe_preserve_order([str(x) for x in val])
    return _dedupe_preserve_order([str(val)])

def _parse_history_ratings(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [t for t in val.strip().split() if t]
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    return [str(val).strip()]

def _list_to_block(lines: List[str]) -> str:
    return "\n\n".join(lines)

def _format_item_block(idx: Any, texts: str) -> str:
    return f"[ID {idx}]\n{texts}"

def _get_item_block(asin: str, meta_map: Dict[str, Tuple[int, str]]) -> str:
    data = meta_map.get(asin)
    if data is None:
        return f"[ID MISSING]\nASIN: {asin}\n<metadata unavailable>"
    midx, texts = data
    return _format_item_block(midx, texts)

def _build_intro(domain_name: str) -> str:
    return (
        f"You are a judgment assistant for a recommender system in the **{domain_name}** domain. "
        "The recommender suggests items based on the user's past interactions **within this category only**. "
        "User item ratings are on a **1-5** scale (higher is better)."
    )

def _build_task() -> str:
    return (
        "**TASK:** Given the user's item interaction history, decide whether the recommended item is **relevant** or **not relevant**.\n"
        "If the recommendation is relevant, justify your decision by listing **evidence item IDs** taken **exactly** "
        "from the provided history (use the integer IDs shown as `[ID N]`).\n"
        "**You must list evidence IDs if and only if the recommendation is relevant.**\n"
        "Provide exactly two fields:\n"
        "- **evidence** - a Python list of **integer IDs** from the history that support your decision. "
        "Leave this list **empty** if the recommendation is irrelevant.\n"
        "- **is_relevant** - your decision, either `\"YES\"` or `\"NO\"`."
    )

def _build_output_spec() -> str:
    return (
        "### Strict output format — return **only** this JSON object (no extra text):\n"
        "{\n"
        "  \"evidence\": [<id-1>, <...>],\n"
        "  \"is_relevant\": \"<YES|NO>\"\n"
        "}\n"
    )

def _build_category_guidance_general() -> str:
    categories: List[Tuple[str, str]] = [
        ("SAME_BRAND_OR_SERIES",
         "Matches brands/series/collections the user frequently engaged with (e.g., same maker, label, studio, or series)."),
        ("FUNCTION_OR_STYLE_MATCH",
         "Strong resemblance in primary function, genre, style, or use-case compared to items in history."),
        ("ACCESSORY_OR_COMPAT",
         "Complements owned items (e.g., compatible accessory/cable/part; correct media format/region for the user's items)."),
        ("UPGRADE_OR_COMPLEMENT",
         "A sensible upgrade or add-on that extends the user's existing use patterns."),
        ("REDUNDANT_DUPLICATE",
        "Too similar to what the user already owns and **not necessary or helpful at all** "
        "given the user's behavior pattern or the item's properties. Mark as redundant **only** when the additional item appears unnecessary."),
        ("NO_RELATION",
         "No clear, defensible link to any provided input; probably a weak/spurious recommendation."),
        ("NOISE",
         "The apparent correlations arise from noisy history and do **not** truly align with the user's genuine interests."),
    ]
    lines = [
        "### You can use the category guidance below to help your judgment (do **not** output any category):",
        "<relevance_categories>",
    ]
    lines += [f"{name:<20} - {desc}" for name, desc in categories]
    lines += [
        "</relevance_categories>",
        "",
        "Decision rules:",
        "- If your best match is **REDUNDANT_DUPLICATE**, **NOISE**, or **NO_RELATION**, set `\"is_relevant\"` to `\"NO\"` and return an empty `\"evidence\"` list.",
        "- Otherwise, set `\"is_relevant\"` to `\"YES\"` and list **integer** IDs from the history in `\"evidence\"`.",
        "- Do **not** invent IDs; use only the `[ID N]` values shown in the history block.",
    ]
    return "\n".join(lines)

def _build_input_section(history_block: str, candidate_block: str) -> str:
    parts = ["──────────────── INPUT"]
    if history_block.strip():
        parts += ["<history_items>", history_block, "</history_items>\n"]
    parts += ["<candidate_item>", candidate_block, "</candidate_item>", "────────────────"]
    return "\n".join(parts)


def build_brief_prompts(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    data_category: str,
    asin_col: str = "parent_asin",
    history_col: str = "history"
) -> pd.DataFrame:
    meta_map: Dict[str, Tuple[int, str]] = {}
    for _, r in meta_df.iterrows():
        pa = str(r["parent_asin"])
        meta_map[pa] = (int(r["idx"]), str(r["texts"]))

    domain_name = {
        "CDs_and_Vinyl": "CDs & Vinyl",
        "Movies_and_TV": "Movies & TV",
    }.get(data_category, _normalize_ws(data_category or "This Category"))

    prompts: List[str] = []

    intro = _build_intro(domain_name)
    task = _build_task()
    guide = _build_category_guidance_general()
    outspec = _build_output_spec()

    for _, row in df.iterrows():
        cand_asin = _normalize_ws(row.get(asin_col, ""))
        history_asins = _parse_history_asins(row.get(history_col))
        history_ratings = _parse_history_ratings(row.get("history_ratings"))
        history_blocks = []
        for j, a in enumerate(history_asins):
            base_block = _get_item_block(a, meta_map)
            if j < len(history_ratings):
                r = _normalize_ws(history_ratings[j])
                if r:
                    try:
                        rv = float(r)
                        r_disp = str(int(rv)) if rv.is_integer() else f"{rv:.1f}"
                    except Exception:
                        r_disp = r
                    base_block = f"{base_block}\nUser rating (1-5): {r_disp}"
            history_blocks.append(base_block)
        history_block = _list_to_block(history_blocks)

        cand_block = _get_item_block(cand_asin, meta_map)

        input_section = _build_input_section(history_block, cand_block)

        parts = [
            intro,
            task,
            guide,
            outspec,
            input_section,
            "**Output:** (return only the JSON object above)",
        ]
        prompts.append("\n\n".join(p for p in parts if p))

    out_df = df.copy()
    out_df["prompt"] = prompts
    return out_df