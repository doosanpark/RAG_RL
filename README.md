# RAG-RL: Multi-hop QA를 강화학습으로 — 두 막의 정직한 분석

**서강대학교 강화학습 프로젝트** · 박두산 (팀장) · A72058 신세정 (팀원)
**GitHub**: https://github.com/doosanpark/RAG_RL · **발표 슬라이드**: [RL_Project_Presentation_v2.pptx](RL_Project_Presentation_v2.pptx)

> Multi-hop QA에서 강화학습이 무엇을 할 수 있는가? 두 막에 걸쳐 정직하게 답한다.
>
> - **1막 — selection-only RL** (REINFORCE+baseline). frozen LLM 옆에 작은 selector를 학습. cosine 휴리스틱을 못 넘고(0.355<0.370), transfer −24% 폭락. 진단: **"추론을 담을 그릇의 부재"**. → [report.md](report.md)
> - **2막 — Solution A** (Qwen LoRA로 SFT→GRPO, Search-R1 방식). 진단을 그대로 공략 — 추론을 LLM 안에. **in-domain 돌파(0.469)**, 단 도메인 전이는 과적합. → [report_solution_a.md](report_solution_a.md) · [docx](report_solution_a.docx)

## 결과 한눈에

### In-domain (HotpotQA validation, 3 seed mean±std)
| 방법 | answer F1 | 비고 |
|---|---:|---|
| Oracle (상한) | 0.557 | frozen Qwen2.5-0.5B의 천장 |
| Naive RAG (cosine top-3) | 0.370 | 학습 없는 휴리스틱 — 1막이 넘지 못한 벽 |
| **1막: Step-wise RL** | 0.355 ± 0.012 | H1/H2 기각 — cosine 못 넘음 |
| 2막: SFT search (Search-R1) | 0.434 | 추론을 LLM에 → 휴리스틱 돌파 |
| **2막: SFT+GRPO RL** | **0.469 ± 0.007** | RL 추가이득 +0.035 (3-seed 견고) |
| frozen-base (cold-start) | 0.006 | SFT warmup이 필수임을 정량 입증 |

### Transfer (스포츠 룰북 350문항)
| 방법 | in-domain | sports | 변화 |
|---|---:|---:|---:|
| Naive (cosine) | 0.370 | **0.386** | +0.016 (견고) |
| 1막: Step-wise RL | 0.355 | 0.270 ± 0.038 | **−24%** (H3 기각) |
| 2막: SFT+GRPO RL | 0.469 | 0.313 ± 0.023 | −33% (과적합, 단 RL > SFT 유지) |

→ **두 막을 합친 핵심 통찰**: 학습 없는 cosine은 in-domain 천장이 낮지만 OOD에 견고하다. 학습된 정책은 in-domain↑이지만 OOD에 취약하다. **"학습효과 vs OOD 강건성"의 트레이드오프**가 두 막 모두에서 정량으로 드러난다.

## 가설 검증 요약
**1막 (selection-only RL)**
- **H1** (Step-wise ≫ Sparse) → **기각** (3 seed에서 answer F1 동률 0.355 vs 0.354; 단일 seed 우위는 noise)
- **H2** (RL > cosine 휴리스틱) → **기각** (0.355 < 0.370; 단 keep 단락 2.0개로 cosine 3.0개와 동률 — precision 이점)
- **H3** (HotpotQA → 새 도메인 transfer) → **기각** (sports -24%, random 수준까지 폭락)

**2막 (Solution A)**
- **HA-1** (추론을 LLM 안에 두면 selection-RL·cosine을 넘는다) → **강하게 지지** (0.355 → 0.434 → 0.469, cold-start 0.006이 SFT 기여 증명)
- **HA-2** (RL이 SFT 위에 추가이득) → **부분 지지** (+0.035, std 0.007로 견고; 메인 held-out에선 comparison +0.14에 집중, bridge는 작고 분산 큰 신호)
- **HA-3** (파인튜닝된 search 정책의 도메인 전이) → **음성/혼합** (sports 0.313 < cosine 0.386 — 단 RL > SFT, comparison 전이는 강함 0.507)

---

## 1막 — Selection-only RL (`src/`)
1 에피소드 = HotpotQA 1 샘플. 후보 단락 N개(distractor 10)에 대해 **keep / drop / stop_and_answer**를 순차 결정하는 MDP를 정식화. REINFORCE+baseline으로 selector를 학습한다. LLM(Qwen2.5-0.5B)은 freeze, 답 생성에만 사용.

- **MDP**: State = (질문, 누적 keep 단락, 후보 단락, step), Action = {keep pᵢ, drop pᵢ, stop_and_answer} (2N+1)
- **Reward (v4.2)**: 비대칭 step reward + 연속 F1 final reward
  - keep 정답 +0.2 / drop 정답 −0.3 (정답 drop 최대 페널티 → recall 보존) / keep 노이즈 −0.1 / drop 노이즈 +0.05
  - R_final = 2.0 × answer_F1 − 0.1·t (cliff 없는 연속 보상)
- **Policy**: 2-layer MLP(~200K params) hidden (256, 128). LLM freeze.
- **Encoder**: sentence-transformers MiniLM-L6-v2 (384d)
- **Algorithm**: REINFORCE + learned baseline (강의 06 범위). CartPole-v1 3-seed sanity 통과.
- **핵심 설계**:
  - **BC warmup**: cold-start 시 "즉시 stop" reward hacking으로 수렴(빈 컨텍스트 F1=0.149 → reward +0.20 정확 일치). supporting_facts 기반 expert로 1000 샘플 모방학습 → 회피.
  - **Lean state (32d)**: raw 임베딩(4639d)은 train 과적합(dev F1 0.19 < cosine 0.37). 유사도 기반 lean state로 일반화 회복(dev 0.35).

자세한 결과·분석 → **[report.md](report.md)**, 아래 "Phase 4 결과" 표, 그리고 PPT slide 1-14.

## 2막 — Solution A: 검색·추론을 LLM 안에 (`src/sol_a/`)
1막의 진단("추론 그릇 부재")을 직접 공략 — **Qwen2.5-0.5B를 LoRA로 SFT warmup → GRPO RL** 파인튜닝(Search-R1 방식). 한 assistant 턴 안에서 `<think>` → `<search>` → env가 `<information>` 주입 → ... → `<answer>` 프로토콜을 학습. "검색"은 외부 인덱스가 아니라 후보 풀(distractor 10) 내 MiniLM top-2 retrieve로 한정 — 검색서버 엔지니어링 제거.

| 모델 | in-domain F1 | bridge | comparison | sports |
|---|---:|---:|---:|---:|
| frozen-base (cold-start) | 0.006 | 0.005 | 0.007 | 0.005 |
| SFT search | 0.434 | 0.435 | 0.428 | 0.299 |
| **RL (3-seed)** | **0.469 ± 0.007** | 0.445 ± 0.010 | **0.568 ± 0.024** | 0.313 ± 0.023 |

- **학습 동역학**: improve → peak → drift 하락. seed7은 step100에 dev 0.354·search 1.45로 포맷 붕괴 → **dev-best 체크포인트가 step80 peak 보존** (1막 교훈 재확인)
- **GRPO 위치**: GRPO = REINFORCE + **group baseline**(질문당 G=5 rollout의 mean) + **KL 정규화** — 강의 06의 baseline·variance reduction 개념의 자연스러운 확장
- **정성 사례 (보고서 §3.3)**: 같은 검색결과를 받고도 RL이 더 정확한 답 추출. 예: "Q is for Quarry 저자의 부친" → SFT는 등장인물 "Kinsey Millhone", **RL은 정답 "C. W. Grafton"**. RL이 새로 가르친 것은 retrieval이 아니라 **"질문 의도 → 답 토큰 정렬"의 정교화**
- 학습곡선: [results/sol_a_learning_curves.png](results/sol_a_learning_curves.png)
- 자세한 분석 → **[report_solution_a.md](report_solution_a.md)** ([docx](report_solution_a.docx))

### 2막 실행
```powershell
# SFT warmup (4000 trace, 3 epoch, ~30분)
python -m src.sol_a.sft_train --epochs 3

# RL 3 seed (lr 3e-5, KL 0.01, 100 step — 각 ~5h, 끊김 시 --resume)
foreach ($s in 42,123,7) {
  python -m src.sol_a.grpo_train --steps 100 --lr 3e-5 --kl-coef 0.01 --seed $s `
    --out models/sol_a/rl_s$s
}

# 최종 평가 (held-out: val[200:400], dev[0:64]와 disjoint) + 3-seed 집계
python -m src.sol_a.eval_a --adapter models/sol_a/rl_s42/best --dataset hotpot --n 200 --start 200
python -m src.sol_a.eval_a --adapter models/sol_a/rl_s42/best --dataset sports
python -m src.sol_a.aggregate_a
```
중단 시 이어하기: `--resume models/sol_a/rl_s42/ckpt` (adapter + optimizer + step + RNG 복원)

## 학습된 모델 (어댑터)
1막·2막 모두 LoRA 어댑터로 저장됨 (각 ~8.6 MB). 현재 로컬 보관, **HF Hub 업로드 스크립트 준비됨**.

| 단계 | 경로 | 크기 |
|---|---|---|
| 2막 SFT | `models/sol_a/sft/best` | 8.6 MB |
| 2막 RL seed 42 | `models/sol_a/rl_s42_v2/best` | 8.6 MB |
| 2막 RL seed 123 | `models/sol_a/rl_s123_v2/best` | 8.6 MB |
| 2막 RL seed 7 | `models/sol_a/rl_s7_v2/best` | 8.6 MB |
| 1막 step-wise | `models/step_seed{42,123,7}_best.pt` | — |

HF Hub 업로드: `huggingface-cli login` 후 `python -m src.sol_a.push_hub` (→ `huggingface.co/doosanpark/rag-rl-sol-a-{sft,rl-s42,rl-s123,rl-s7}`).

## Setup

### 0. 사전 요구
- Windows / Linux, NVIDIA GPU (8GB+), CUDA 12.1 드라이버
- Python 3.11 (3.14는 PyTorch wheel 미지원이라 사용 X)

### 1. venv 생성 + 활성화
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. PyTorch (CUDA 12.1) 설치
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. 나머지 의존성
```powershell
pip install -r requirements.txt
```

> **중요 — 항상 venv의 Python을 쓸 것.** 시스템 `python`이 3.14로 잡히면
> `datasets`/`dill` pickle 충돌(`Pickler._batch_setitems() takes 2 positional...`)이 난다.
> VS Code 통합 터미널은 [.vscode/settings.json](.vscode/settings.json) 덕분에 자동으로
> venv를 활성화한다 (새 터미널 열면 프롬프트에 `(.venv)` 표시). 일반 PowerShell 창에서는
> 매 세션 한 번 `.\.venv\Scripts\Activate.ps1` 또는 명령마다 `.\.venv\Scripts\python.exe`를 쓴다.
>
> 현재 어떤 Python이 잡히는지 확인: `python -c "import sys; print(sys.version)"` → `3.11.0` 이어야 정상.

### 4. 환경 검증
```powershell
python -m src.verify_env
```
출력에서 `CUDA avail. : True` 확인.

### 5. HotpotQA 다운로드 + 구조 확인
```powershell
python -m src.download_data
```

> 의존성은 2026.05 기준 검증 조합으로 핀되어 있다 (numpy 1.26 / pyarrow 17 /
> pandas 2.3 / datasets 2.21). numpy 2.x·pyarrow 24·pandas 3.x 조합은 Windows에서
> sentence-transformers import 시 access violation(0xC0000005)을 일으켜 회피한다.

## 사용법 (직접 실행해보기)

> 아래 명령은 모두 venv가 활성화된 상태(`(.venv)` 프롬프트)를 가정한다.
> 활성화가 안 됐다면 `python` 대신 `.\.venv\Scripts\python.exe`로 바꿔 친다.

### A. 대화형 질문 — `src.ask`
프롬프트에 질문을 입력하면 검색 → 답변 → (정답이 있으면) F1까지 보여준다.
```powershell
# hotpot 모드: 질문만 입력하면 가장 비슷한 HotpotQA 샘플의 단락을 검색 풀로 사용
python -m src.ask

python -m src.ask --k 5                 # top-5 검색
python -m src.ask --pool-size 1000      # 질문 매칭 풀 확대 (매칭 정확도↑)

# passages 모드: 질문 + 단락을 직접 입력 (자기 도메인 텍스트 테스트용)
python -m src.ask --mode passages

# 학습된 RL 정책으로 단락 selection (학습 완료 후)
python -m src.ask --policy rl --ckpt models/step_seed42_final.pt
```
종료: 빈 입력 / `q` / `quit` / `exit` / Ctrl+C.

### B. 한 샘플 자세히 보기 — `src.demo`
한 HotpotQA 샘플에 대해 후보 단락, 검색 결과, 답변, F1을 한 화면에 보여준다.
```powershell
python -m src.demo                       # 랜덤 1 샘플
python -m src.demo --index 3             # validation[3]
python -m src.demo --index 3 --variant use_all   # 같은 샘플, 단락 전부 사용
python -m src.demo --index 3 --variant oracle    # 같은 샘플, 정답 단락만 (상한)
# 직접 입력
python -m src.demo --question "..." --passages "단락1" "단락2" "단락3"
```
같은 `--index`로 `top_k_sim` / `use_all` / `oracle`을 비교하면 검색 차이가 답변에
어떻게 영향을 주는지 직접 확인할 수 있다.

### C. 정량 평가 — `src.run_eval`
검증셋 N개에 대해 평균 F1·EM·support_F1을 계산한다. 모든 변형이 동일 인터페이스라 1:1 비교된다.
```powershell
# 학습 없는 baseline
python -m src.run_eval --variant top_k_sim --k 3 --n 200   # = Naive RAG
python -m src.run_eval --variant use_all   --n 200
python -m src.run_eval --variant oracle    --n 200          # 상한
python -m src.run_eval --variant random    --k 3 --n 200    # 하한

# 학습된 RL 정책
python -m src.run_eval --variant rl --ckpt models/step_seed42_final.pt --policy greedy --n 200
```
`--n`은 평가 샘플 수(클수록 신뢰도↑/느림). 빠른 확인은 20, 표준은 200, 보고서용은 500~1000.

## 디렉토리 구조
```
src/
  rl_types.py        Action / StepRecord / Trajectory 타입
  rewards.py         v4.2 reward + answer F1
  env.py             RAGEnv (gym 스타일 MDP)
  agent.py           REINFORCE + learned baseline
  state_encoder.py   sentence-transformers + state→벡터
  llm.py             Qwen2.5-0.5B answerer
  train_cartpole.py  CartPole sanity check
  train_rag.py       HotpotQA 본 학습 (step / sparse)
  evaluate.py        범용 평가기
  run_eval.py        통합 평가 entrypoint (baseline + RL)
  demo.py            한 샘플 데모
  ask.py             대화형 질문 REPL
  baselines/
    naive_rag.py     use_all / top_k_sim / random / oracle
data/
  raw/         원본 HotpotQA 등 (gitignored)
  processed/   인코딩된 임베딩 / 토큰
  eval/        보드게임 + 근로기준법 평가셋
models/      체크포인트 (gitignored)
results/     Table 1, learning curves, trajectory dump
logs/        설치 / 실행 로그, W&B export
tests/       unit test (env, reward) — 24개
.vscode/     통합 터미널 venv 자동 활성화 설정
```

## 진행 상황
**1막 (Selection-only RL)**
- [x] Phase 1: 환경 셋업 + 데이터 다운로드
- [x] Phase 2: RAGEnv class + reward 함수 + unit test (24/24 green)
- [x] Phase 3: REINFORCE+Baseline + CartPole sanity check (3 seeds 통과)
- [x] Phase 4: HotpotQA 본 학습 (3 seed, step + sparse) — BC warmup + lean state
- [x] Phase 5: baseline — Naive RAG 4종 + Sparse RL (Classification은 범위 제외)
- [x] Phase 6: Table 1 + 학습곡선 + 스포츠 transfer + **보고서([report.md](report.md))**

**2막 (Solution A — Search-R1 SFT→GRPO)**
- [x] A1: SFT trace 자동 생성 (supporting_facts → 멀티홉 trace 4000+400)
- [x] A2: LoRA SFT warmup (val_loss 0.113 수렴)
- [x] A3: 멀티턴 search env (`</search>` 정지 + retrieve 주입) + reward (F1+format)
- [x] A4: GRPO smoke (OOM 없음, --resume 검증)
- [x] A5: 3-seed RL 학습 (lr 3e-5, KL 0.01, dev-best — seed7 step100 붕괴 방어)
- [x] A6: held-out 3-seed 집계 + 정성 사례 분석 ([report_solution_a.md](report_solution_a.md))
- [x] A7: 보고서 (md + docx) + README 두 막 통합 + PPT v2

## 1막 상세 결과 (3 seeds: 42/123/7, validation n=200)

| 방법 | answer_F1 | support_F1 | EM | avg_kept | 비고 |
|:---|---:|---:|---:|---:|:---|
| Oracle (정답 단락만) | 0.557 | 1.000 | 0.420 | 2.0 | 상한 |
| Naive RAG (cosine top-3) | 0.370 | 0.582 | 0.280 | 3.0 | 학습 없는 강한 베이스라인 |
| use_all | 0.367 | 0.340 | 0.275 | 9.9 | |
| **Step-wise RL (제안)** | 0.355 ± 0.012 | 0.533 ± 0.005 | 0.262 | 2.0 | |
| Sparse RL | 0.354 ± 0.002 | 0.512 ± 0.027 | 0.267 | 2.0 | ablation |
| random | 0.275 | 0.258 | 0.200 | 3.0 | 하한 |

곡선(3-seed mean±std): [results/learning_curves.png](results/learning_curves.png), 표: [results/table1_3seed.json](results/table1_3seed.json)

**핵심 발견 (정직한 분석)**
1. **일반화가 표현에 좌우됨**: 정책 입력에 raw 임베딩(4639-d)을 쓰면 train 과적합(dev F1 0.19 < cosine 0.37). cosine 유사도 기반 **lean state(32-d)**로 바꾸자 dev 일반화가 살아남(BC만으로 0.19→0.29, 학습 후 0.31~0.35).
2. **H1 (step vs sparse) — 지지되지 않음**: 3 seed에서 answer_F1 **동률**(0.355 vs 0.354). 단일 seed(42)에서 보였던 step 우위는 noise였고 3 seed로 사라짐. 유일하게 견고한 관찰: **support_F1의 seed 간 분산이 step에서 훨씬 작음**(±0.005 vs ±0.027) — dense reward가 선택 품질을 더 *일관적*으로 만듦.
3. **H2 (RL vs Naive)**: RL은 cosine 휴리스틱을 못 넘음(0.355 vs 0.370). 단 RL은 단락 2.0개만 keep해 cosine(3.0개)보다 **간결**하게 비슷한 F1 — precision 측면 이점.
4. **reward hacking 진단**: BC warmup 없이 학습 시 정책이 "즉시 stop"으로 수렴(빈 컨텍스트 F1=0.149 → reward +0.20 정확 일치). BC warmup이 이 국소최적을 회피.
5. **천장은 LLM**: oracle도 0.557 — frozen Qwen2.5-0.5B가 병목. selection 개선이 answer F1로 전이되지 않음.
6. **REINFORCE 불안정성**: 학습이 dev F1에서 plateau 후 후반 drift → dev-best 체크포인트로 peak 보존.

### Phase 4 재현 (3 seed)
```powershell
foreach ($s in 42,123,7) {
  python -m src.train_rag --seed $s --n-episodes 2500 --use-llm --no-wandb `
    --lr-policy 1e-4 --gamma 0.99 --bc-warmup-samples 1000 --batch-episodes 8 `
    --dev-eval-every 250 --weight-decay 1e-4                       # step-wise
  python -m src.train_rag --seed $s --no-step-reward ...같은옵션...  # sparse
  python -m src.run_eval --variant rl --ckpt models/step_seed$s`_best.pt --n 200
  python -m src.run_eval --variant rl --ckpt models/sparse_seed$s`_best.pt --n 200
}
python -m src.aggregate_results   # table1_3seed.json + learning_curves.png (mean±std)
```

## 재현

### CartPole sanity check (Phase 3)
```powershell
python -m src.train_cartpole --seed 42 --max-episodes 500
python -m src.train_cartpole --seed 123 --max-episodes 500
python -m src.train_cartpole --seed 7   --max-episodes 500
python -m src.plot_cartpole
```
결과: 3 seed 모두 `avg(100ep) >= 195` 통과.

| seed | solved episode | final avg100 |
|-----:|---------------:|-------------:|
| 42   |            242 |       196.66 |
| 123  |            185 |       196.56 |
| 7    |            157 |       195.47 |

곡선: [results/cartpole_curve.png](results/cartpole_curve.png).

**통과한 hyperparam**:
- `lr_policy=1e-3`, `lr_value=1e-3` (Adam)
- `gamma=0.99`
- `hidden_dims=(256, 128)` (policy/value 공통)
- `grad_clip=1.0`
- `normalize_advantage=True` (trajectory 내 z-score)
- baseline: 학습 baseline (raw return 타깃으로 MSE)

### 학습 전 baseline (HotpotQA validation, n=200)
```powershell
python -m src.run_eval --variant oracle    --n 200
python -m src.run_eval --variant top_k_sim --k 3 --n 200
python -m src.run_eval --variant use_all   --n 200
python -m src.run_eval --variant random    --k 3 --n 200
python -m src.summarize_baselines          # 표 + results/baseline_table.csv
```

| variant | answer_F1 | EM | support_F1 | avg_kept | 의미 |
|:---|---:|---:|---:|---:|:---|
| oracle | 0.557 | 0.420 | 1.000 | 2.0 | 정답 단락만 — RL 상한 |
| top_k_sim (k=3) | 0.370 | 0.280 | 0.582 | 3.0 | **Naive RAG** (학습 후 넘어야 할 기준) |
| use_all | 0.367 | 0.275 | 0.340 | 9.9 | 후보 전부 |
| random (k=3) | 0.275 | 0.200 | 0.258 | 3.0 | 무작위 — 하한 |

- selection의 잠재 이득(상한−하한): **0.282**.
- RL이 메워야 할 마진(상한−Naive): **0.187**.
- `comparison` 타입은 격차가 작고, `bridge`(다중 hop) 타입에서 RL 이득이 클 것으로 예상.

## 라이센스 / 데이터 출처
- HotpotQA distractor: CC BY-SA 4.0
- 스포츠 룰북 350문항 (자체 구축): 배구·탁구·배드민턴·미식축구·축구·농구·야구·하키 각 종목 공식 룰 텍스트를 HotpotQA 포맷으로 라벨링 (transfer 평가용)
