import os
import torch
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from vllm import LLM, SamplingParams
from utils.processors import V0Processor
from utils.cosine import add_cosine_scores
from utils.infer_din import add_din_scores
from utils.infer_sas import add_sas_scores
from utils.metainfo import build_item_texts
from utils.helpers import safe_parse_response
from utils.template import build_brief_prompts

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    print("Better call Saul!")

    # Prepare textual description for each item. <parent_asin, idx, texts>
    meta_df = build_item_texts(f"./data/{args.data_cat}/{args.data_cat}_meta.jsonl",
                               start_idx=0)

    # Prepare a list of prompts and collaborative signals.
    pairs = pd.read_csv(f"./data/{args.task}/{args.data_cat}.csv")
    print(f"There are {len(pairs)} history-candidate pairs.")

    if args.set == "evidence":
        df = build_brief_prompts(pairs, meta_df, args.data_cat)
    else:
        df = build_brief_prompts(pairs, meta_df, args.data_cat)
        if args.set == "din-dir":
            model_ckpt_path = f"./outputs/ori_din/{args.data_cat}/din_model.pth"
            item2idx_path = f"./outputs/ori_din/{args.data_cat}/item2idx.json"
            score_df = add_din_scores(pairs, model_ckpt_path, item2idx_path)
        elif args.set == "sas-dir":
            model_ckpt_path = f"./outputs/ori_sasrec/{args.data_cat}/sasrec_model.pth"
            item2idx_path = f"./outputs/ori_sasrec/{args.data_cat}/item2idx.json"
            score_df = add_sas_scores(pairs, model_ckpt_path, item2idx_path)
        elif args.set == "din-emb":
            emb_path = f"./outputs/ori_din/{args.data_cat}/item_embedding_dict.json"
            score_df = add_cosine_scores(pairs, emb_path)
        else:
            emb_path = f"./outputs/ori_sasrec/{args.data_cat}/item_embedding_dict.json"
            score_df = add_cosine_scores(pairs, emb_path)

    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True
    )
    tokenizer = llm.get_tokenizer()

    chat_strs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for p in df["prompt"].tolist()
    ]

    logits_processors = []
    if args.set in ["din-dir", "sas-dir", "din-emb", "sas-emb"]:
        print("Logits processor ENABLED.")
        proc = V0Processor(
            tokenizer, 
            meta_df[['parent_asin', 'idx']],
            score_df, 
            beta=args.beta,
            use_entropy=args.use_entropy
        )
        logits_processors.append(proc)
    else:
        print("Logits processor DISABLED.")

    params = SamplingParams(
        max_tokens=16384,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        logits_processors=logits_processors,
    )

    print("Starting generation...")
    all_text_outputs = []
    for i in range(0, len(chat_strs), args.batch_size):
        batch = chat_strs[i : i + args.batch_size]
        batch_outs = llm.generate(batch, params)
        for o in batch_outs:
            all_text_outputs.append(o.outputs[0].text.strip())

    evidence_col, isrel_col = [], []
    for txt in all_text_outputs:
        evidence, is_rel = safe_parse_response(txt)
        evidence_col.append(evidence)
        isrel_col.append(is_rel)

    df["evidence"] = evidence_col
    df["is_relevant"] = isrel_col

    print(f"Size of the dataframe before filtering parse errors: {len(df)}")
    df = df[df['is_relevant'].isin(['YES', 'NO'])].copy()
    print(f"Size of the dataframe after filtering: {len(df)}")

    marker = 'ada'
    if not args.use_entropy:
        marker = 'rigid'
    destination_path = f"./outputs/{args.task}/{args.data_cat}/{args.set}/rewards_{marker}.csv"
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    df.to_csv(destination_path, index=False)
    print(f"Data saved to: {destination_path}")
    print(f"Number of positive rewards: {len(df.loc[df['is_relevant'] == 'YES'])}")

    # Calculate task specific metrics.
    if args.task == 'ratings':
        is_relevant_proportions = df.groupby('rating')['is_relevant'].apply(lambda x: (x == 'YES').mean())
        props = []
        for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
            props.append(float(is_relevant_proportions[rating]))
        print(props)
        bases = {
            'ratings': [1.0, 2.0, 3.0, 4.0, 5.0],
            'props':   props
        }
        df_stat = pd.DataFrame(bases)
        spearman_corr, _ = stats.spearmanr(df_stat['ratings'], df_stat['props'])
        kendall_corr, _ = stats.kendalltau(df_stat['ratings'], df_stat['props'])
        print(f"Spearman's Rho: {spearman_corr}")
        print(f"Kendall's Tau: {kendall_corr}")
    else:
        old_train = pd.read_csv(f"./data/{args.data_cat}/{args.data_cat}_train.csv")
        print(f"Number of data instances before augmentation: {len(old_train)}")

        pos_feedback = df.loc[df['is_relevant'].eq('YES'),
                              ['user_id', 'parent_asin', 'history', 'history_ratings']].copy()
        h = pos_feedback['history'].astype(str).str.strip()
        p = pos_feedback['parent_asin'].astype(str).str.strip()
        pos_feedback['asin_seqs'] = np.where(h.eq(''), p, h + ' ' + p)
        pos_feedback = pos_feedback.rename(columns={'history_ratings': 'ratings'})[['user_id', 'asin_seqs', 'ratings']]
        new_train = pd.concat([old_train, pos_feedback], ignore_index=True)
        print(f"Number of data instances after augmentation: {len(new_train)}")
        aug_path = f"./data/enriched/{args.set}_{marker}/{args.data_cat}_train.csv"
        os.makedirs(os.path.dirname(aug_path), exist_ok=True)
        new_train.to_csv(aug_path, index=False)
        print(f"Augmentation saved to: {aug_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrIEF")
    parser.add_argument("--task", type=str, required=True, help="The task (ratings & augment).")
    parser.add_argument("--set", type=str, required=True, help="The variant of BrIEF to use.")
    parser.add_argument("--data_cat", type=str, required=True, help="CDs_and_Vinyl, Movies_and_TV.")
    parser.add_argument("--use_entropy", action='store_true', help="Flag to enable entropy-based adaptive biasing.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B", help="Name of the LLM.")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--beta", type=float, default=25.0, help="Base strength of logit biasing.")
    
    args = parser.parse_args()
    print(vars(args))
    main(args)