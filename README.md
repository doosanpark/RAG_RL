# RAG-RL: Step-wise Reward로 학습하는 Multi-hop RAG

> RL 클래스 프로젝트 — Multi-hop QA 검색·답변 과정을 REINFORCE+Baseline으로 최적화.
> Sparse reward 학습 비효율을 step-wise intermediate reward로 해소하고
> cross-domain transfer 능력을 측정한다.

## 핵심 가설
- **H1**: Step-wise reward ≫ Sparse reward (sample efficiency)
- **H2**: Step-wise RL > Classification baseline (왜 RL이 필요한가)
- **H3**: HotpotQA로 학습한 정책이 보드게임 / 근로기준법으로 transfer 가능

## 알고리즘
- **MDP**: State = (질문, 누적 keep 단락, 후보 단락, step), Action = {keep p_i, drop p_i, stop_and_answer}
- **Reward (v4.2)**: 비대칭 step reward + 연속 F1 final reward
  - keep 정답 +0.2 / drop 정답 -0.3 / keep 노이즈 -0.1 / drop 노이즈 +0.05
  - R_final = 2.0 × answer_F1 − 0.1·t
- **Policy**: 2-layer MLP (~200K params), LLM은 freeze (Qwen2.5-0.5B/1.5B)
- **Encoder**: sentence-transformers MiniLM-L6-v2 (384d)
- **Algorithm**: REINFORCE + learned baseline (강의 06 범위)

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
- [x] Phase 1: 환경 셋업 + 데이터 다운로드
- [x] Phase 2: RAGEnv class + reward 함수 + unit test (24/24 green)
- [x] Phase 3: REINFORCE+Baseline + CartPole sanity check (3 seeds 통과)
- [x] Phase 4: HotpotQA 본 학습 (3 seed, step + sparse) — BC warmup + lean state
- [x] Phase 5: baseline — Naive RAG 4종 + Sparse RL (Classification은 범위 제외)
- [x] Phase 6: Table 1 + 학습곡선 + 스포츠 transfer + **보고서([report.md](report.md))**

## Phase 4 결과 (3 seeds: 42/123/7, validation n=200)

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

### Phase 4 (RL 본 학습) 명령은 학습 완료 후 채워나간다.
```powershell
# (예정) python -m src.train_rag --seed 42 --n-episodes 5000 --use-llm
```

## 라이센스 / 데이터 출처
- HotpotQA: CC BY-SA 4.0
- KLRI 근로기준법 영문본: elaw.klri.re.kr 공개 번역
- 보드게임 룰북: 각 제작사 공식 PDF (Catan, Monopoly, Ticket to Ride)
