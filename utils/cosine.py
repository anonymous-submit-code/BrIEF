import json
import numpy as np
from typing import Optional, Dict

def add_cosine_scores(
    df,
    emb_json_path: str,
    factor: float = 2.0,
    top_k: int = 2,
):
    with open(emb_json_path, "r") as f:
        raw: Dict[str, list] = json.load(f)

    emb_norm: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        if n > 0 and np.isfinite(n):
            emb_norm[k] = a / n

    df_out = df.copy()
    n = len(df_out)
    scores = np.full(n, np.nan, dtype=np.float32)

    parent_vals = df_out["parent_asin"].astype(str).values
    hist_vals = df_out["history"].astype(str).values

    cand_cache: Dict[str, Optional[np.ndarray]] = {}

    for i, (cand_id, hist_str) in enumerate(zip(parent_vals, hist_vals)):
        if cand_id in cand_cache:
            c = cand_cache[cand_id]
        else:
            c = emb_norm.get(cand_id)
            cand_cache[cand_id] = c

        if c is None:
            continue

        hist_ids = hist_str.split()
        H = [emb_norm[h] for h in hist_ids if h in emb_norm]
        if not H:
            continue

        if len(H) == 1:
            scores[i] = float(np.dot(H[0], c))
            continue

        Hm = np.vstack(H)
        sims = Hm @ c

        m = sims.shape[0]
        if m <= top_k:
            scores[i] = float(np.mean(sims))
        else:
            idx = np.partition(sims, m - top_k)[m - top_k :]
            scores[i] = float(np.mean(idx))
        
        denominator = np.exp(factor) - 1
        if denominator >= 1e-9:
            scaled_abs = (np.exp(factor * abs(scores[i])) - 1) / denominator
            scores[i] = np.sign(scores[i]) * scaled_abs

    df_out["score"] = scores
    return df_out