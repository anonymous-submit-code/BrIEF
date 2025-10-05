import re
import torch
import numpy as np
import pandas as pd
from typing import List

def parse_prompt(prompt: str):
    def extract_block(tag: str) -> str:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", prompt, flags=re.S)
        if not m:
            raise ValueError(f"Missing <{tag}> block.")
        return m.group(1)

    def extract_first_id(block: str) -> int:
        m = re.search(r"Item ID:\s*(\d+)", block)
        if m:
            return int(m.group(1))
        m = re.search(r"\[\s*ID\s+(\d+)\s*\]", block)
        if m:
            return int(m.group(1))
        raise ValueError("Could not parse candidate Item ID.")

    def extract_all_ids(block: str) -> List[int]:
        ids = re.findall(r"Item ID:\s*(\d+)", block)
        if ids:
            return [int(x) for x in ids]
        ids = re.findall(r"\[\s*ID\s+(\d+)\s*\]", block)
        if ids:
            return [int(x) for x in ids]
        return []

    cand_block = extract_block("candidate_item")
    hist_block = extract_block("history_items")
    candidate = extract_first_id(cand_block)
    history = extract_all_ids(hist_block)
    return candidate, history

def lookup_score(idx_map: pd.DataFrame, scores: pd.DataFrame, candidate: int, history: List[int]):
    idx_to_asin = idx_map.set_index('idx')['parent_asin']
    try:
        cand_asin = str(idx_to_asin.at[candidate])
    except KeyError:
        return float('nan')

    hist_asins = idx_to_asin.reindex(history)
    if hist_asins.isna().any():
        return float('nan')
    hist_key = " ".join(hist_asins.astype(str).tolist())

    s = scores.set_index(['parent_asin', 'history'])['score']
    try:
        val = s.loc[(cand_asin, hist_key)]
        val = val.iloc[0] if isinstance(val, pd.Series) else val
        return float(val)
    except KeyError:
        return float('nan')


class V0Processor:
    def __init__(self, tokenizer, idx_map, scores, beta=5.0, use_entropy=True):
        print("Initializing V0 Processor...")
        self.tok = tokenizer
        self.idx_map = idx_map
        self.scores = scores
        
        self.beta = beta
        self.use_entropy = use_entropy
        print("Hyperparameters set:")
        print(f"  - Beta (Base Strength): {self.beta}")
        print(f"  - Use Entropy Scaling: {self.use_entropy}")

        self.trigger_tokens = self.tok.encode('vidence":', add_special_tokens=False)
        self.yea_target_token_id = self.tok.encode(' [', add_special_tokens=False)[0]
        self.nay_target_token_id = self.tok.encode(' [],\n', add_special_tokens=False)[0]
        self.target_2_token_id = self.tok.encode(' [\n', add_special_tokens=False)[0]
        self.max_entropy = np.log(3.0)

        self.idx_to_asin = dict(zip(idx_map['idx'].astype(int),
                                    idx_map['parent_asin'].astype(str)))
        sc = scores[['parent_asin', 'history', 'score']].copy()
        sc['parent_asin'] = sc['parent_asin'].astype(str)
        sc['history'] = sc['history'].astype(str)
        sc = sc.drop_duplicates(['parent_asin', 'history'], keep='last')
        self.score_lookup = {
            (a, h): float(s) for a, h, s in sc.itertuples(index=False)
        }

        print("Finished setting up V0 Processor.")

    def _lookup(self, candidate: int, history: List[int]) -> float:
        cand_asin = self.idx_to_asin.get(int(candidate))
        if cand_asin is None:
            return 0.0
        hist_asins = [self.idx_to_asin.get(int(h)) for h in history]
        if any(a is None for a in hist_asins):
            return 0.0
        hist_key = " ".join(hist_asins)
        return self.score_lookup.get((cand_asin, hist_key), 0.0)

    def __call__(self, prompt_ids, prev_ids, logits):
        if len(prev_ids) < len(self.trigger_tokens) or \
           list(prev_ids[-len(self.trigger_tokens):]) != self.trigger_tokens:
            return logits

        prompt_text = self.tok.decode(prompt_ids, skip_special_tokens=False)
        candidate, history = parse_prompt(prompt_text)
        scaled_signal = self._lookup(candidate, history)

        adaptive_strength = self.beta
        normalized_entropy = -1.0 # Default value if not used.
        if self.use_entropy:
            decision_logits = torch.tensor([
                logits[self.yea_target_token_id],
                logits[self.nay_target_token_id],
                logits[self.target_2_token_id]
            ])
            probs = torch.softmax(decision_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()
            normalized_entropy = entropy / self.max_entropy
            adaptive_strength = self.beta * (1 + normalized_entropy)
        
        delta = adaptive_strength * scaled_signal
        logits[self.yea_target_token_id] += delta
        logits[self.nay_target_token_id] -= delta
        return logits