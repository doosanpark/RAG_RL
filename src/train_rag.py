"""HotpotQA에서 REINFORCE+Baseline으로 Step-wise RL 학습.

마스터 컨텍스트 Phase 4. 학습 setup:
  - 학습 데이터: HotpotQA train 중 ~2000 샘플 (--n-train으로 조정)
  - Optimizer: Adam, lr=1e-4 (spec)
  - Discount γ = 0.95 (RAG에선 더 빨리 감소)
  - Batch: episode-by-episode
  - Gradient clipping max_norm=1.0
  - 3 seed 예정 (별도 호출)

W&B 로깅 (매 N step):
  - episode_reward (running mean)
  - answer_f1 (running mean)
  - policy_loss / value_loss / entropy
  - gold_keep_rate (정답 단락 keep 비율)
  - action_distribution

실행:
    python -m src.train_rag --seed 42 --n-episodes 5000 --use-llm
    python -m src.train_rag --seed 42 --n-episodes 50 --no-wandb --use-mock-llm  # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset

from .agent import REINFORCEAgent, Transition
from .env import RAGEnv, make_oracle_answerer_by_indices
from .rewards import GAMMA
from .rl_types import ActionKind
from .state_encoder import (
    StateEncoder,
    encode_sample,
    expected_state_dim,
    state_to_vector,
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_answerer(args, env: RAGEnv):
    """args에 따라 LLM, mock, oracle 중 선택."""
    if args.use_mock_llm:
        # 항상 빈 문자열 (F1=0)
        return lambda q, kt: ""
    if args.use_oracle:
        # passage에 gold가 있으면 정답을 반환 (학습 디버깅용)
        # env.passages가 reset 후에야 채워지므로 reset 직후 호출자가 갱신해야 함
        return None  # 매 reset마다 main loop에서 새로 만든다
    if args.use_llm:
        from .llm import QwenAnswerer
        return QwenAnswerer(device=args.device)
    # 기본: mock (빈 문자열)
    return lambda q, kt: ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=2000,
                        help="HotpotQA train에서 사용할 샘플 수")
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--lr-value", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-llm", action="store_true",
                        help="Qwen2.5-0.5B 답변 생성기 사용")
    parser.add_argument("--use-mock-llm", action="store_true",
                        help="LLM 안 쓰고 빈 답 반환 (smoke용)")
    parser.add_argument("--use-oracle", action="store_true",
                        help="kept에 gold 있으면 정답 반환 (학습 디버깅용)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="rag-rl")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--log-every", type=int, default=20,
                        help="콘솔 로그 / W&B 로그 간격 (episode 단위)")
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--use-step-reward", action="store_true", default=True,
                        help="step-wise reward 사용 (default True). Phase 5 sparse용으로 --no-step-reward")
    parser.add_argument("--no-step-reward", dest="use_step_reward", action="store_false")
    args = parser.parse_args()

    set_seed(args.seed)

    # ---------- W&B ----------
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            mode=args.wandb_mode,
            config=vars(args),
            name=f"step{'_off' if not args.use_step_reward else ''}_s{args.seed}_{int(time.time())}",
        )
    else:
        run = None

    # ---------- 데이터 ----------
    print(f"[data] HotpotQA train load ({args.n_train} samples)...")
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    train = ds["train"].shuffle(seed=args.seed).select(range(args.n_train))
    print(f"[data] loaded {len(train)} samples")

    # ---------- env / encoder / answerer ----------
    print(f"[encoder] loading sentence-transformers (device={args.device})...")
    encoder = StateEncoder(device=args.device)
    state_dim = expected_state_dim(n_candidates=10, emb_dim=encoder.emb_dim)
    n_actions = 2 * 10 + 1
    print(f"[env] state_dim={state_dim}, n_actions={n_actions}")

    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    answerer = build_answerer(args, env)

    # ---------- agent ----------
    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        hidden_dims=(256, 128),
        grad_clip=1.0,
        normalize_advantage=True,
        device=args.device,
    )

    # ---------- 학습 ----------
    window_R = deque(maxlen=100)
    window_F1 = deque(maxlen=100)
    window_gold_keep = deque(maxlen=100)
    action_counts = np.zeros(n_actions, dtype=np.int64)
    log_history: List[Dict[str, Any]] = []

    rng = random.Random(args.seed)
    t_start = time.time()

    for ep in range(1, args.n_episodes + 1):
        sample = train[rng.randrange(len(train))]
        state = env.reset(sample)

        # oracle/llm을 매 sample 기준으로 다시 만들어야 하는 경우
        if args.use_oracle:
            env.answerer = make_oracle_answerer_by_indices(env.passages, sample["answer"])
        else:
            env.answerer = answerer

        # 사전 인코딩 (1회)
        encoded = encode_sample(encoder, env.passages, env.question, n_candidates_max=10)
        # step reward 옵션
        force_zero_step = not args.use_step_reward

        transitions: List[Transition] = []
        ep_reward = 0.0
        ep_gold_keep = 0   # 정답 단락을 keep한 횟수
        n_gold_passages = sum(1 for p in env.passages if p.is_gold)
        done = False
        while not done:
            sv = state_to_vector(state, encoded, max_steps=env.max_steps)
            state_t = torch.from_numpy(sv).float()
            mask_t = torch.tensor(state.valid_actions_mask, dtype=torch.bool)
            action_idx, log_prob = agent.select_action(state_t, mask_t)
            action_counts[action_idx] += 1

            next_state, reward, done, info = env.step(action_idx)
            if force_zero_step:
                # Sparse RL ablation: 종료 step의 final만 남기고 step reward 제거
                # info["step_reward"]를 빼면 final만 남음
                step_r = info.get("step_reward", 0.0)
                reward -= step_r
            ep_reward += reward
            if info["action_kind"] == "keep" and info.get("is_gold"):
                ep_gold_keep += 1
            transitions.append(
                Transition(state=state_t, action=action_idx, log_prob=log_prob,
                           reward=float(reward), mask=mask_t)
            )
            state = next_state

        # update
        update_info = agent.update(transitions)

        # logging
        f1 = info.get("answer_f1", 0.0) if info else 0.0
        gold_keep_rate = ep_gold_keep / max(1, n_gold_passages)
        window_R.append(ep_reward)
        window_F1.append(f1)
        window_gold_keep.append(gold_keep_rate)

        if ep % args.log_every == 0 or ep == 1:
            avg_R = sum(window_R) / len(window_R)
            avg_F1 = sum(window_F1) / len(window_F1)
            avg_gold = sum(window_gold_keep) / len(window_gold_keep)
            elapsed = time.time() - t_start
            ips = ep / max(1e-6, elapsed)
            print(
                f"[ep {ep:5d}/{args.n_episodes}] R={ep_reward:+.2f} avg={avg_R:+.2f} "
                f"F1={f1:.2f} avgF1={avg_F1:.2f} gold_keep={avg_gold:.2f} "
                f"p_loss={update_info['policy_loss']:+.3f} v_loss={update_info['value_loss']:.2f} "
                f"ent={update_info['entropy']:.2f} ips={ips:.1f}"
            )
            log_history.append(
                {
                    "episode": ep, "R": ep_reward, "avg_R": avg_R,
                    "F1": f1, "avg_F1": avg_F1, "gold_keep_rate": avg_gold,
                    **update_info,
                }
            )
            if run is not None:
                run.log(
                    {
                        "episode": ep,
                        "episode_reward": ep_reward,
                        "avg100/reward": avg_R,
                        "answer_f1": f1,
                        "avg100/f1": avg_F1,
                        "avg100/gold_keep_rate": avg_gold,
                        "policy_loss": update_info["policy_loss"],
                        "value_loss": update_info["value_loss"],
                        "entropy": update_info["entropy"],
                        "mean_return": update_info["mean_return"],
                        "elapsed_sec": elapsed,
                    },
                    step=ep,
                )

        # checkpoint
        if ep % args.ckpt_every == 0:
            tag = "step" if args.use_step_reward else "sparse"
            ckpt = MODELS_DIR / f"{tag}_seed{args.seed}_ep{ep}.pt"
            torch.save(
                {
                    "policy": agent.policy.state_dict(),
                    "value": agent.value.state_dict(),
                    "args": vars(args),
                    "ep": ep,
                },
                ckpt,
            )
            print(f"  [ckpt] {ckpt}")

    # 최종 저장
    tag = "step" if args.use_step_reward else "sparse"
    final_path = MODELS_DIR / f"{tag}_seed{args.seed}_final.pt"
    torch.save(
        {
            "policy": agent.policy.state_dict(),
            "value": agent.value.state_dict(),
            "args": vars(args),
            "ep": args.n_episodes,
            "action_counts": action_counts.tolist(),
        },
        final_path,
    )
    log_path = RESULTS_DIR / f"train_{tag}_seed{args.seed}.json"
    log_path.write_text(json.dumps(log_history, indent=2), encoding="utf-8")
    print(f"[done] final ckpt: {final_path}")
    print(f"[done] log: {log_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
