"""SFT 어댑터를 로드해 멀티턴 rollout이 제대로 도는지 검증.

검증 포인트: </search>에서 멈춰 실제 retrieve된 <information>이 주입되는가,
            2-hop 검색 후 <answer>가 나오는가, reward 파싱이 맞는가.

실행:
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.smoke_rollout --adapter models/sol_a/sft/best --n 5
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hotpot_data import load_examples
from .reward_a import compute_reward
from .search_env import RolloutConfig, rollout_once
from .format_utils import Retriever

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/sol_a/sft/best")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model.eval()

    retriever = Retriever(device=device)
    cfg = RolloutConfig(do_sample=not args.greedy)

    examples = load_examples(args.split, n=args.n, start=args.start)
    f1s, ems, n_searches = [], [], []
    for ex in examples:
        r = rollout_once(model, tok, retriever, ex.question, ex.candidates, cfg, device)
        rb = compute_reward(r.gen_text, ex.answer)
        f1s.append(rb.f1); ems.append(rb.em); n_searches.append(r.n_search)
        print("=" * 90)
        print(f"Q: {ex.question}")
        print(f"gold: {ex.answer} | pred: {r.answer} | F1={rb.f1:.2f} EM={rb.em:.0f} "
              f"| searches={r.n_search} turns={r.n_turns} stop={r.stop_reason}")
        print(f"queries: {r.queries}")
        print("--- gen_text ---")
        print(r.gen_text[:1200])
    n = len(f1s)
    print("\n" + "=" * 90)
    print(f"[smoke summary] n={n} meanF1={sum(f1s)/n:.3f} EM={sum(ems)/n:.3f} "
          f"avg_search={sum(n_searches)/n:.2f}")


if __name__ == "__main__":
    main()
