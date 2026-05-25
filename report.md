# Step-wise Intermediate Reward로 학습하는 Multi-hop RAG Passage Selection
### — 강화학습 기반 검색 단락 선택의 가능성과 한계에 대한 정직한 분석

**RL 클래스 프로젝트 보고서**

---

## 초록 (Abstract)

Multi-hop 질의응답에서 검색된 후보 단락 중 무엇을 LLM에 넘길지(passage selection)를
강화학습으로 최적화하는 문제를 다룬다. `supporting_facts` 라벨로부터 자동 생성되는
**step-wise intermediate reward**를 설계하고, freeze된 소형 LLM(Qwen2.5-0.5B) 위에서
작은 정책망을 REINFORCE+baseline으로 학습했다. 핵심 결과는 **정직한 negative/nuanced
result**다: (H1) step-wise reward는 sparse reward 대비 최종 answer F1에서 유의한 차이가
없었고(3 seed), (H2) 학습된 정책은 학습 없는 cosine 유사도 휴리스틱을 넘지 못했으며
(in-domain 0.355 vs 0.370), (H3) 새 도메인(스포츠 룰북)으로의 transfer에서 학습 정책은
24% 하락(0.355→0.270)한 반면 cosine은 견고했다(0.370→0.386). 본 보고서의 기여는
높은 점수의 성능이 아니라 **"왜 selection-only RL이 한계인가"를 정량적으로 규명**한 데 있다:
(1) reward hacking을 정량 포착하고 BC warmup으로 해소, (2) 입력 표현이 일반화를 좌우함을
입증, (3) 단일 seed의 착시를 3 seed로 반증, (4) frozen LLM + 추론 없는 selector라는
구조가 multi-hop을 담지 못함을 bridge 타입 분석으로 규명.

---

## 1. Introduction

### 1.1 동기

Retrieval-Augmented Generation(RAG)에서 검색기는 보통 질문-단락 유사도로 top-k를
고른다. 그러나 multi-hop 질문(예: "X를 작곡한 사람이 태어난 도시는?")은 2차 단락
(bridge)이 원질문과 직접 닮지 않아 단순 유사도로는 놓치기 쉽다. 본 프로젝트는 이
**선택 과정을 순차적 의사결정(MDP)으로 보고 강화학습으로 최적화**할 수 있는지 묻는다.

### 1.2 도메인 선택 이유

- **학습/평가(in-domain): HotpotQA distractor split.** `supporting_facts` 라벨이
  step reward의 자동 ground truth가 되어 추가 라벨링이 불필요하다. multi-hop QA의
  표준 벤치마크다.
- **Transfer(out-of-domain): 스포츠 룰북 QA(자체 구축 350문항).** 배구 등 공식 룰북을
  HotpotQA 포맷으로 라벨링했다. 도메인 전이 능력(H3)을 측정하기 위함이며, 학습에
  쓰지 않은 완전히 다른 분포다.

### 1.3 기여 (Contributions)

1. **Passage selection의 MDP 정식화 + step-wise reward 설계**, 그리고 강의 06 범위의
   REINFORCE+baseline 구현(CartPole-v1 3 seed sanity check 통과).
2. **체계적 진단 분석**: reward hacking 정량 포착, 입력 표현-일반화 관계 규명, 단일 seed
   착시의 3-seed 반증, REINFORCE 불안정성 대응(dev-best checkpoint).
3. **정직한 한계 규명**: selection-only RL(frozen LLM + 소형 정책)은 in-domain에서
   cosine과 동급, transfer에서 열세이며, 그 원인이 "추론을 담을 그릇의 부재"임을
   bridge 타입 분석과 Search-R1 대비로 설명.

---

## 2. Methodology

### 2.1 MDP 정식화

1 에피소드 = HotpotQA 1 샘플. 후보 단락 N개(distractor=10, 스포츠=8)가 주어진다.

- **State** $s_t$: 질문, 현재까지 keep한 단락 집합, 후보별 상태, step $t$.
- **Action** $a_t \in \{\text{keep } p_i,\ \text{drop } p_i,\ \text{stop\&answer}\}$.
  고정 크기 $2N+1$의 action space, 이미 처리한 단락은 마스킹.
- **Transition**: keep/drop은 해당 단락을 처리됨으로 표시, stop은 종료.
- **Horizon**: 최대 $T=10$ step. 종료 시 keep된 단락을 LLM에 넘겨 답 생성.

### 2.2 Reward 설계 (v4.2 — 비대칭 + 연속 F1)

- **Step reward** (즉시, `supporting_facts` 기반):
  정답 keep +0.2 / 정답 drop −0.3(가장 큰 페널티) / 노이즈 keep −0.1 / 노이즈 drop +0.05 / stop 0.
- **Final reward**: $R_{\text{final}} = 2.0 \cdot F_1(\text{answer}) - 0.1\cdot t$ (연속, cliff 없음).
- **Return**: $G_0 = \sum_k \gamma^k r_k + \gamma^T R_{\text{final}}$, $\gamma=0.99$.

비대칭(정답 drop을 무겁게)은 recall 보존을, 연속 final은 정답 품질을 유도한다.

### 2.3 정책 / 학습 알고리즘

- **REINFORCE + learned baseline**: reward-to-go $G_t$, advantage $A_t = G_t - V_\phi(s_t)$,
  $\nabla_\theta J = \sum \nabla \log \pi_\theta(a_t|s_t) A_t$, value는 $\text{MSE}(V_\phi, G_t)$.
- **분산 감소**: baseline, reward-to-go, advantage 정규화(에피소드 배치 전체 기준),
  gradient clipping(max_norm=1.0).
- **정책망**: 2-layer MLP. LLM(Qwen2.5-0.5B-Instruct)은 **freeze**, 답 생성에만 사용.

### 2.4 두 가지 핵심 설계 — BC warmup, lean state

학습 과정에서 발견한 두 문제를 해결하기 위해 도입했다(§5 진단 참조).

- **Behavior Cloning warmup**: `supporting_facts`로 만든 expert("정답 keep → 노이즈
  drop → stop")를 cross-entropy로 모방 후 RL 시작. cold-start의 "즉시 stop" 국소최적을 회피.
- **Lean state (32차원)**: 정책 입력을 raw 임베딩 대신 **일반화되는 per-candidate 신호**
  로 구성 — 질문-후보 cosine 유사도(q_sim), 누적 keep과의 유사도(kept_sim), 처리 여부,
  step. raw 임베딩(4639차원)은 train 과적합을 일으켜 일반화를 망쳤다.

### 2.5 Sparse RL ablation

H1 검증을 위해 동일 setup에서 **step reward를 모두 0으로** 만든 Sparse RL을 학습한다
(final reward만 학습 신호). step-wise reward의 sample efficiency·최종 성능 기여를 분리 측정.

---

## 3. Experiments

### 3.1 데이터셋

| 용도 | 데이터 | 규모 |
|:---|:---|:---|
| 학습 | HotpotQA distractor (train) | 2,000 샘플 |
| in-domain 평가 | HotpotQA distractor (validation) | n=200 |
| transfer 평가 | 스포츠 룰북 QA (자체) | n=350 (bridge 305 / comparison 45, 전부 hard) |

### 3.2 비교 대상 (baselines)

모두 **동일 frozen Qwen2.5-0.5B** 위에서 비교(공정 비교).

- **Oracle**: `supporting_facts`의 정답 단락만 사용 → RL의 상한.
- **Naive RAG (top_k_sim)**: sentence-transformers(MiniLM-L6-v2) 유사도 top-3 → 학습 없는 강한 베이스라인.
- **use_all**: 후보 전부 LLM에 입력.
- **random**: 무작위 3개 keep → 하한.
- **Sparse RL / Step-wise RL(제안)**: §2, 각 3 seed(42/123/7).

### 3.3 지표

- **answer_F1** (SQuAD 토큰 F1, 핵심), **exact_match**, **support_F1**(keep한 단락 vs
  정답 단락의 title F1 = 선택 품질), **avg_n_kept**.

### 3.4 하이퍼파라미터

REINFORCE+baseline, Adam lr=1e-4, $\gamma$=0.99, hidden=(256,128), grad_clip=1.0,
advantage 정규화(배치), weight_decay=1e-4, episode batch=8, BC warmup(1000 샘플,
bc_lr=1e-3, 30 epoch), lean state 32차원, dev-best checkpoint, 2,500 episode.
인코더 MiniLM-L6-v2(384d, freeze), LLM Qwen2.5-0.5B-Instruct(freeze, greedy, 32 tok).
하드웨어 RTX 4060 Ti 8GB. (학습 전 CartPole-v1 sanity check: 3 seed 모두 avg(100ep)≥195
통과 — 각각 242/185/157 episode.)

---

## 4. Results

### 4.1 Table 1 — In-domain (HotpotQA, n=200)

| 방법 | answer_F1 | EM | support_F1 | avg_kept |
|:---|---:|---:|---:|---:|
| Oracle (상한) | 0.557 | 0.420 | 1.000 | 2.0 |
| Naive RAG (cosine top-3) | 0.370 | 0.280 | 0.582 | 3.0 |
| use_all | 0.367 | 0.275 | 0.340 | 9.9 |
| **Step-wise RL (제안)** | **0.355 ± 0.012** | 0.262 | **0.533 ± 0.005** | 2.0 |
| Sparse RL | 0.354 ± 0.002 | 0.267 | 0.512 ± 0.027 | 2.0 |
| random (하한) | 0.275 | 0.200 | 0.258 | 3.0 |

### 4.2 Table 2 — Transfer (스포츠 룰북, n=350)

| 방법 | answer_F1 | support_F1 |
|:---|---:|---:|
| Oracle (상한) | 0.452 | 1.000 |
| Naive RAG (cosine top-3) | 0.386 | 0.666 |
| use_all | 0.369 | 0.400 |
| **Step-wise RL** | 0.270 ± 0.038 | 0.461 |
| Sparse RL | 0.267 ± 0.034 | 0.456 |
| random | 0.263 | 0.305 |

### 4.3 Figure 1 — 학습곡선 (dev F1, 3 seed mean±std)

`results/learning_curves.png`. step-wise·sparse 모두 BC 출발점(~0.29)에서 학습 후
dev best ~0.31–0.35로 상승하나, 두 곡선의 std 음영이 학습 내내 겹친다(유의차 없음).
후반부 drift가 관찰되어 dev-best checkpoint로 peak를 보존했다.

### 4.4 In-domain vs Transfer (핵심 비교)

| 방법 | HotpotQA | Sports | 변화 |
|:---|---:|---:|---:|
| Naive (cosine) | 0.370 | 0.386 | **+0.016 ↑** |
| Step-wise RL | 0.355 | 0.270 | **−0.085 (−24%) ↓** |
| Oracle | 0.557 | 0.452 | −0.105 |

### 4.5 Hop 타입별 분석 (in-domain, seed 42 대표)

| 방법 | bridge (multi-hop) | comparison |
|:---|---:|---:|
| Naive (cosine) | 0.35 | 0.43 |
| Step-wise RL | **0.316** | 0.448 |

RL은 cosine이 강한 comparison에서 약간 앞서나, **이겨야 할 bridge(multi-hop)에서
오히려 cosine보다 낮다.**

---

## 5. Discussion

### 5.1 가설 검증

- **H1 (step-wise ≫ sparse) — 지지되지 않음.** 3 seed에서 answer_F1 동률(0.355 vs 0.354),
  학습곡선 std 음영도 내내 겹친다. 단일 seed(42)에서 step이 우세해 보였으나 3 seed로
  사라졌다. 유일하게 견고한 차이는 step의 support_F1 분산이 작다는 점(±0.005 vs ±0.027)
  — dense reward가 직접 보상하는 선택 품질을 더 *일관적*으로 만든다.
- **H2 (RL > Naive) — 지지되지 않음.** in-domain에서 RL(0.355) < cosine(0.370). RL은
  단락 2개만 keep해 cosine(3개)과 비슷한 F1을 내는 precision 이점은 있으나, 절대 성능은
  못 넘었다.
- **H3 (transfer 가능) — 부정적.** 학습 정책은 새 도메인에서 24% 하락, cosine은 오히려
  소폭 상승. **학습된 MLP는 HotpotQA 분포에 특화되어 transfer가 안 되고, cosine은
  파라미터 없는 보편 신호라 transfer가 잘 된다.**

### 5.2 예상과 달랐던 점 — 그리고 그 원인

**왜 RL은 cosine을 못 넘는가?** RL이 이길 수 있는 유일한 지점은 cosine이 실패하는
bridge(2차 단락은 원질문과 안 닮음)인데, §4.5에서 RL은 정확히 bridge에서 cosine보다
낮았다. 즉 **정책이 multi-hop 추론(1차 단락을 보고 연결되는 2차 단락 찾기)을 학습하지
못했다.** 우리 정책은 유사도 feature 위의 작은 MLP라 그런 추론 용량이 없다.

이는 Search-R1(2025) 같은 성공 사례와 대비된다: 그들은 **LLM 자체를 RL로 학습**해
추론을 LLM 안에 둔다. 우리는 LLM을 freeze하고 추론 없는 selector를 붙였기에 multi-hop을
담을 그릇이 구조적으로 없다. 이는 8GB GPU·기간 제약에서 내린 단순화이며, 그 단순화가
곧 성능 천장이 되었다.

**또 하나의 천장은 LLM 자체다.** oracle(정답만 줘도) F1이 in-domain 0.557 / transfer
0.452에 그친다. 즉 단락을 완벽히 골라도 frozen Qwen2.5-0.5B가 그 이상 답하지 못하므로,
selection 개선이 answer F1로 전이될 여지 자체가 작다(in-domain selection 여지 = oracle−use_all
= 0.19, transfer는 0.08로 더 작음).

### 5.3 진단으로 얻은 부수적 발견 (방법론적 기여)

1. **Reward hacking 정량 포착.** BC warmup 없이 학습 시 정책이 "아무것도 keep하지 않고
   즉시 stop"으로 수렴했다. frozen LLM이 빈 컨텍스트에서도 parametric 지식으로 F1=0.149를
   내므로, lazy-stop의 기대 reward = $2\times0.149 - 0.1\times1 = +0.198$이 관측된 평균
   reward(+0.20)와 정확히 일치했다. BC warmup이 이 국소최적을 회피하게 했다.
2. **입력 표현이 일반화를 좌우.** raw 임베딩(4639d) 입력은 train 과적합으로 dev F1 0.19
   (cosine 0.37보다 낮음). 유사도 기반 lean state(32d)로 바꾸자 dev 0.29→0.35로 일반화 회복.
3. **단일 seed 착시의 반증.** seed 42 단독으로는 step-wise가 sparse보다 빠르고 우세해
   보였으나, 3 seed 집계 시 차이가 사라졌다. 다중 seed의 필요성을 실증.
4. **REINFORCE 불안정성.** dev F1이 plateau 후 후반 drift → dev-best checkpoint로 대응.

### 5.4 Future Work

multi-hop을 실제로 풀려면 추론을 정책에 내재화해야 한다. 가장 원칙적인 길은
**LLM을 LoRA로 RL 파인튜닝(Search-R1 방식)**하여 "검색 질의 생성 + 추론"을 학습하는
것이다. 다만 단일 8GB GPU에서 run당 1–3일 × (조건×seed) 규모로, 본 프로젝트 기간을
초과한다. 중간 대안으로 frozen LLM의 zero-shot 질의 재구성(query reformulation)으로
2차 단락을 후보에 끌어오는 **반복 검색(iterative retrieval) MDP**가 있으며, 이는 LLM
학습 없이 bridge를 다룰 그릇을 제공한다(향후 과제).

---

## 6. Limitations

- **단일 base LM(Qwen2.5-0.5B, freeze)**: 성능 천장이 LLM에 묶여 selection 개선이
  answer F1로 충분히 전이되지 않는다. 더 큰 모델(1.5B+)은 천장을 올리나 8GB 메모리·
  selection 여지 축소의 trade-off가 있다.
- **영어 단일 언어, transfer 도메인 1종(스포츠)**: 근로기준법 등 추가 도메인은 범위에서
  제외했다. transfer 일반화 주장은 도메인 1개에 기반한다.
- **평가 규모**: in-domain n=200, transfer n=350. 더 큰 n에서 안정성↑.
- **정책 표현 한계**: 유사도 feature 위 소형 MLP는 multi-hop 추론 용량이 없다(설계상 선택).
- **REINFORCE의 high variance**: 학습이 dev F1에서 불안정하며 dev-best 선택에 의존한다.

---

## 7. 결론

본 프로젝트는 "step-wise reward로 RL이 RAG selection을 개선한다"를 입증하지 **못했다**.
대신 **selection-only RL(frozen 소형 LLM + 추론 없는 정책)의 한계를 정량적으로
규명**했다: in-domain에서 학습 없는 cosine과 동급에 그치고, multi-hop(bridge)에서
오히려 뒤지며, 새 도메인으로 transfer되지 않는다. 그 원인은 (1) LLM이 천장이고,
(2) 정책에 multi-hop 추론 용량이 없으며, (3) 학습 정책이 도메인 특화되어 보편적
cosine 신호를 못 이긴다는 데 있다. 성능 향상이라는 결과보다, **재현 가능한 실험·정직한
다중 seed 검증·명확한 인과 규명**에 본 보고서의 가치가 있다.

---

## 부록 — 재현

코드/데이터/가중치: 본 저장소. 상세 명령은 `README.md`의 "재현" 절 참조.
- 환경 검증: `python -m src.verify_env` (venv Python 3.11)
- CartPole sanity: `python -m src.train_cartpole --seed {42,123,7}`
- 본 학습: `python -m src.train_rag --seed S --use-llm ...` (step / `--no-step-reward` sparse)
- 평가: `python -m src.run_eval --variant {oracle,top_k_sim,use_all,random,rl} [--eval-file data/eval/sports.json]`
- 집계: `python -m src.aggregate_results`, `python -m src.build_results`
- 결과물: `results/table1_3seed.json`, `results/transfer_sports.csv`, `results/learning_curves.png`
