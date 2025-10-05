import json
import torch
import numpy as np
import torch.nn as nn
from typing import Dict, Optional

class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hid_dim: int,
        max_len: int,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_items = num_items
        self.hid_dim = hid_dim
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, hid_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hid_dim)
        self.emb_layernorm = nn.LayerNorm(hid_dim)
        self.emb_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=num_heads,
            dim_feedforward=hid_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def _causal_mask(self, L: int, device):
        m = torch.full((L, L), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, seqs: torch.LongTensor, lengths: torch.LongTensor = None) -> torch.Tensor:
        device = seqs.device
        B, L = seqs.size()
        if lengths is None:
            lengths = (seqs != 0).sum(dim=1).clamp(min=1)

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seqs) + self.pos_emb(pos_ids)
        x = self.emb_layernorm(x)
        x = self.emb_dropout(x)

        x = self.encoder(
            x,
            mask=self._causal_mask(L, device),
            src_key_padding_mask=(seqs == 0),
        )

        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, x.size(-1))
        return x.gather(dim=1, index=idx).squeeze(1)

    def score_item(self, seq_repr: torch.Tensor, item_idx: int) -> torch.Tensor:
        e = self.item_emb.weight[item_idx]
        return (seq_repr * e).sum(dim=-1)


def add_sas_scores(
    df,
    model_ckpt_path: str,
    item2idx_path: str,
    factor: float = 2.0,
    batch_size: int = 128,
    num_heads: int = 1,
    device: Optional[str] = None,
):
    df_out = df.copy()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with open(item2idx_path, "r") as f:
        item2idx: Dict[str, int] = json.load(f)

    state = torch.load(model_ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    def _infer_hparams_from_state(sd: Dict[str, torch.Tensor]):
        ni_plus1, d = sd["item_emb.weight"].shape
        max_len, d2 = sd["pos_emb.weight"].shape
        assert d == d2, "Inconsistent hidden dims in checkpoint."
        import re
        layer_ids = set()
        pat = re.compile(r"^encoder\.layers\.(\d+)\.")
        for k in sd.keys():
            m = pat.match(k)
            if m:
                layer_ids.add(int(m.group(1)))
        num_layers = (max(layer_ids) + 1) if layer_ids else 2
        return ni_plus1 - 1, d, max_len, num_layers

    num_items, hid_dim, max_len, num_layers = _infer_hparams_from_state(state)

    model = SASRec(
        num_items=num_items,
        hid_dim=hid_dim,
        max_len=max_len,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _pad_left(seq_ids, L):
        if len(seq_ids) >= L:
            seq_ids = seq_ids[-L:]
        return [0] * (L - len(seq_ids)) + seq_ids, min(len(seq_ids), L)

    n = len(df_out)
    scores = np.full(n, np.nan, dtype=np.float32)

    hist_buf, len_buf, cand_buf, valid_rows = [], [], [], []

    for i, (asin, hist_str) in enumerate(
        zip(df_out["parent_asin"].astype(str), df_out["history"].astype(str))
    ):
        cidx = item2idx.get(asin)
        if cidx is None:
            continue

        hist_tokens = hist_str.split()
        hist_ids = [item2idx[t] for t in hist_tokens if t in item2idx]
        if not hist_ids:
            continue

        padded, true_len = _pad_left(hist_ids, max_len)
        hist_buf.append(padded)
        len_buf.append(true_len)
        cand_buf.append(cidx)
        valid_rows.append(i)

    if not hist_buf:
        df_out["score"] = scores
        return df_out

    hist_np = np.asarray(hist_buf, dtype=np.int64)
    len_np  = np.asarray(len_buf,  dtype=np.int64)
    cand_np = np.asarray(cand_buf, dtype=np.int64)
    idx_np  = np.asarray(valid_rows, dtype=np.int64)

    N = hist_np.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        seqs_t = torch.from_numpy(hist_np[start:end])
        lens_t = torch.from_numpy(len_np[start:end])
        cand_t = torch.from_numpy(cand_np[start:end])

        if device.type == "cuda":
            seqs_t = seqs_t.pin_memory().to(device, non_blocking=True)
            lens_t = lens_t.pin_memory().to(device, non_blocking=True)
            cand_t = cand_t.pin_memory().to(device, non_blocking=True)
        else:
            seqs_t = seqs_t.to(device)
            lens_t = lens_t.to(device)
            cand_t = cand_t.to(device)

        q = model(seqs_t, lens_t)
        e_cand = model.item_emb.weight[cand_t]
        raw = (q * e_cand).sum(dim=-1)
        prob = (2 * torch.sigmoid(raw).float().cpu().numpy()) - 1

        denominator = np.exp(factor) - 1
        if denominator >= 1e-9:
            scaled_abs = (np.exp(factor * abs(prob)) - 1) / denominator
            prob = np.sign(prob) * scaled_abs

        scores[idx_np[start:end]] = prob

    df_out["score"] = scores
    return df_out