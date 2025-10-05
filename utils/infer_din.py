import json
import torch
import numpy as np
import torch.nn as nn
from typing import List, Dict

class Dice(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        ps = torch.sigmoid((x - mean) / torch.sqrt(var + self.eps))
        return ps * x + (1 - ps) * self.alpha * x


class LocalActivationUnit(nn.Module):
    def __init__(self, emb_dim: int, hidden_units = [80, 40], use_dice: bool = True, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = emb_dim * 4
        for hu in hidden_units:
            layers.append(nn.Linear(in_dim, hu))
            layers.append(Dice(hu) if use_dice else nn.PReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hu
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
    def forward(self, hist_emb: torch.Tensor, tgt_emb: torch.Tensor):
        B, L, D = hist_emb.size()
        v = tgt_emb.unsqueeze(1).expand(B, L, D)
        x = torch.cat([hist_emb, v, hist_emb - v, hist_emb * v], dim=-1)
        return self.mlp(x).squeeze(-1)


class DIN(nn.Module):
    def __init__(self, num_items: int, emb_dim: int = 64, max_len: int = 50,
                 att_hidden = [80, 40], dnn_hidden = [200, 80],
                 att_dropout: float = 0.1, dnn_dropout: float = 0.2, use_dice: bool = True,
                 att_weight_normalization: bool = True):
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.att_unit = LocalActivationUnit(emb_dim, hidden_units=att_hidden, use_dice=use_dice, dropout=att_dropout)
        self.att_weight_normalization = att_weight_normalization
        layers = []
        in_dim = emb_dim * 4
        for hu in dnn_hidden:
            layers.append(nn.Linear(in_dim, hu))
            layers.append(Dice(hu) if use_dice else nn.PReLU())
            layers.append(nn.Dropout(dnn_dropout))
            in_dim = hu
        layers.append(nn.Linear(in_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, hist_seq: torch.LongTensor, tgt_idx: torch.LongTensor):
        B, L = hist_seq.size()
        hist_emb = self.item_emb(hist_seq)
        tgt_emb = self.item_emb(tgt_idx)
        mask = (hist_seq != 0)
        att_scores = self.att_unit(hist_emb, tgt_emb)
        att_scores = att_scores.masked_fill(~mask, float("-inf"))
        if self.att_weight_normalization:
            att_w = torch.softmax(att_scores, dim=1)
        else:
            att_scores = torch.relu(att_scores)
            att_w = att_scores * mask.float()
            denom = att_w.sum(dim=1, keepdim=True) + 1e-9
            att_w = att_w / denom
        interest_vec = torch.bmm(att_w.unsqueeze(1), hist_emb).squeeze(1)
        x = torch.cat([interest_vec, tgt_emb, interest_vec * tgt_emb, interest_vec - tgt_emb], dim=-1)
        return self.dnn(x).squeeze(-1)  # logits


def add_din_scores(
    df,
    model_ckpt_path: str,
    item2idx_path: str,
    factor: float = 2.0,
    max_len: int = 50,
    emb_dim: int = 64,
    batch_size: int = 128,
    device: str | None = None,
    use_dice: bool = False,
):
    df_out = df.copy()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with open(item2idx_path, "r") as f:
        item2idx: Dict[str, int] = json.load(f)
    num_items = len(item2idx)

    model = DIN(num_items=num_items, emb_dim=emb_dim, max_len=max_len, use_dice=use_dice).to(device)
    state = torch.load(model_ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True


    def _pad_left(seq_ids: list[int], L: int):
        if len(seq_ids) >= L:
            seq_ids = seq_ids[-L:]
        return [0] * (L - len(seq_ids)) + seq_ids


    n = len(df_out)
    cand_idx = np.full(n, -1, dtype=np.int64)
    has_hist = np.zeros(n, dtype=bool)

    hist_buf = []
    cand_buf = []
    valid_rows = []

    for i, (asin, hist_str) in enumerate(zip(df_out["parent_asin"].astype(str),
                                             df_out["history"].astype(str))):

        cidx = item2idx.get(asin, None)
        if cidx is not None:
            cand_idx[i] = cidx

        hist_ids = [item2idx[x] for x in hist_str.split() if x in item2idx]

        if hist_ids and cidx is not None:
            has_hist[i] = True
            hist_buf.append(_pad_left(hist_ids, max_len))
            cand_buf.append(cidx)
            valid_rows.append(i)

    scores = np.full(n, np.nan, dtype=np.float32)

    if not hist_buf:
        df_out["score"] = scores
        return df_out

    hist_np = np.asarray(hist_buf, dtype=np.int64)
    cand_np = np.asarray(cand_buf, dtype=np.int64)

    N_valid = hist_np.shape[0]
    L = hist_np.shape[1]

    for start in range(0, N_valid, batch_size):
        end = min(start + batch_size, N_valid)
        hist_t = torch.from_numpy(hist_np[start:end]).to(device, non_blocking=True)
        cand_t = torch.from_numpy(cand_np[start:end]).to(device, non_blocking=True)

        logits = model(hist_t, cand_t)
        probs = (2 * torch.sigmoid(logits).float().cpu().numpy()) - 1

        denominator = np.exp(factor) - 1
        if denominator >= 1e-9:
            scaled_abs = (np.exp(factor * abs(probs)) - 1) / denominator
            probs = np.sign(probs) * scaled_abs

        scores[np.array(valid_rows[start:end], dtype=np.int64)] = probs

    df_out["score"] = scores
    return df_out