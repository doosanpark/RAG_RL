# 해결책 A — 검색·추론을 LLM 안에 두기: Qwen LoRA를 SFT→RL로 파인튜닝
### selection-only RL의 한계를 넘기 위한 업그레이드와 그 정직한 평가

**RL 클래스 프로젝트 — 후속(업그레이드) 보고서**
(기반 프로젝트: [report.md](report.md) — step-wise selection RL)

---

## 초록 (Abstract)

기반 프로젝트는 **frozen LLM + 추론 없는 passage selector**를 REINFORCE로 학습했으나,
학습 없는 cosine 휴리스틱을 못 넘고(in-domain 0.355 vs 0.370) 도메인 전이도 실패했다.
원인은 "multi-hop **추론을 담을 그릇의 부재**"였다. 본 후속 연구는 그 한계를 정면으로
공략한다: 오픈 LLM(Qwen2.5-0.5B-Instruct)을 **LoRA로 파인튜닝**하여 검색+추론을 LLM
**안에서** 수행하게 한다(Search-R1 방식). `supporting_facts`로 생성한 멀티홉 추론 trace로
**SFT warmup** 후, answer-F1 보상으로 **GRPO**(group-normalized REINFORCE) RL을 얹었다.

핵심 결과(HotpotQA distractor, val held-out n=200; 3 seed):
- **Cold-start 완전 실패**(frozen-base F1 0.006) → SFT warmup이 필수임을 정량 입증.
- **In-domain: SFT 0.434 > 기존 cosine 0.370**, 그리고 **RL 0.469±0.007 > SFT**(+0.035).
  RL 이득은 **seed 표준편차 0.007로 견고**(단일 seed 착시 아님)하며 **거의 전부
  comparison 타입**(+0.14, 0.428→0.568)에서 나오고 bridge는 미미(+0.01)하다.
- **Transfer(스포츠 룰북 350문항): nuanced negative.** Solution A(0.312±0.023)는
  **기존 cosine(0.386)보다 낮다** — HotpotQA에 파인튜닝된 search 정책이 먼 도메인엔
  휴리스틱보다 못 전이한다. 단 **RL은 transfer에서도 SFT(0.299)보다 낫고**(+0.013),
  comparison 전이는 강하다(0.507).

기여는 높은 절대 점수가 아니라, **"추론을 LLM 안에 두면 in-domain에서 휴리스틱·selection-RL을
명확히 넘지만, 그 이득은 추론 신호가 선명한 comparison에 집중되고, 도메인 전이에서는
오히려 과적합 대가를 치른다"**는 점을 3-seed로 정직하게 규명한 데 있다.

---

## 1. 동기와 가설

기반 프로젝트의 결론(H2: RL이 cosine 못 넘음, 천장=frozen 0.5B)은 **구조적 한계**로
해석됐다. selector는 단락을 고를 뿐 추론을 못 하고, LLM은 freeze되어 학습이 안 된다.
이를 검증·해소하기 위한 가설:

- **HA-1 (그릇 가설):** 추론을 LLM 안에 두면(SFT search), 추론 없는 selector·cosine
  휴리스틱을 in-domain에서 넘는다.
- **HA-2 (RL 추가이득):** outcome F1 보상의 RL이 SFT 위에 추가 이득을 준다. 어떤 hop
  타입에서?
- **HA-3 (전이):** 파인튜닝된 search 정책의 도메인 전이는? 휴리스틱 대비 우열은?

---

## 2. 방법 (Method)

### 2.1 프로토콜 (Search-R1 스타일)
한 assistant 턴 안에서 도구 결과가 inline 주입된다:
```
<think>추론</think><search>질의</search>
   → env가 검색 후 <information>...</information> 주입
<think>추론</think><search>질의2</search> → <information>...</information>
<think>추론</think><answer>최종 답</answer>
```
"검색"은 외부 인덱스가 아니라 **주어진 후보 풀(distractor 10개) 내**에서 질의-기반
retrieve(MiniLM all-MiniLM-L6-v2, top-2 passage, 100단어 컷)로 한정한다 — 무거운
검색서버 엔지니어링을 제거하고 추론 학습에 집중하기 위함. env는 `</search>`에서 생성을
멈춰 실제 검색결과를 주입하고 재개한다.

### 2.2 SFT warmup (cold-start 제거)
`supporting_facts`의 gold title을 질문 단어겹침 순으로 정렬해 멀티홉 trace를 자동 생성한다
(예: "Oberoi family 검색 → The Oberoi Group 발견 → 그것을 검색 → head office=Delhi → 답").
주입된 `<information>`은 loss 마스킹하고 모델 생성 토큰(think/search/answer)만 학습한다.
train 4,000 / val 400 trace, LoRA(r=16, q/k/v/o), bf16, gradient checkpointing(8GB 대응).
**val loss 0.113 수렴**.

### 2.3 RL (GRPO)
질문당 G=5 rollout을 샘플링하고 **group-normalized advantage**(value network 불필요 →
8GB 친화)로 정책경사. 보상 = answer **F1** + 0.1·format(검색≥1회 & 답 형식). 모델이
생성한 action 토큰만 정책경사, **KL(참조=SFT 초기정책) 페널티**로 포맷붕괴 방어,
gradient clipping. dev-best 체크포인트(val[0:64] greedy F1)로 peak 보존.

### 2.4 하드웨어·환경
로컬 RTX 4060 Ti **8GB**(vLLM은 Windows 미지원이라 미사용, rollout은 HF generate).
0.5B+LoRA는 ~1GB VRAM. 기존 venv에 peft만 추가(폴백 환경 보존).

---

## 3. 결과 (Results)

### 3.1 메인 표 — in-domain & transfer (held-out, 3 seed mean±std)

| 모델 | in-domain F1 | EM | bridge | comparison | **sports(transfer) F1** |
|---|---|---|---|---|---|
| frozen-base (cold-start) | 0.006 | 0.000 | 0.005 | 0.007 | 0.005 |
| 기존 frozen+cosine (폴백) | 0.370 | – | – | – | **0.386** |
| **SFT search** | 0.434 | 0.340 | 0.435 | 0.428 | 0.299 |
| **RL (3-seed)** | **0.469±0.007** | 0.372±0.002 | 0.445±0.010 | **0.568±0.024** | **0.313±0.023** |

(in-domain = HotpotQA val[200:400] n=200, dev와 disjoint. sports = 350문항 전체.
per-seed in-domain: 0.466/0.479/0.463, sports: 0.282/0.338/0.317.)

### 3.2 학습 동역학
- **보수적 RL(lr 1e-5, KL 0.05)은 평평**(dev 0.418≈SFT 0.415, KL~0.01로 정책이 거의
  안 움직임). **공격적 RL(lr 3e-5, KL 0.01)에서 비로소 이득** 발생 → RL 하이퍼파라미터가
  관건임을 보여줌.
- dev F1은 **개선→정점→drift 하락** 패턴(seed42 peak 0.507@step80, step100엔 0.470;
  seed7은 step100에 dev 0.354·search 1.45로 **포맷 붕괴**). **dev-best 체크포인트가 붕괴
  직전 peak를 보존** — 기반 프로젝트 교훈(RL 불안정성)의 재확인.
- 전 구간 format 안정(has_answer≈1.0, 평균 2회 검색). reward hacking(보상↑·F1평평) 미관측.

---

## 4. 분석 (Discussion)

**HA-1 지지(강).** 추론을 LLM 안에 두자 in-domain이 0.370(cosine)·0.355(selection-RL)에서
0.434(SFT)→0.469(RL)로 올랐다. cold-start가 0.006인 점은 "포맷·검색·추론 절차"를 SFT가
실제로 가르쳤음을 보여준다(=기반 프로젝트의 "그릇 부재" 진단이 옳았다).

**HA-2 부분 지지(견고하나 국소적).** RL은 SFT 대비 +0.035를 **3 seed에서 std 0.007로
일관되게** 얻었다(단일 seed 착시 아님). 그러나 이득은 **comparison(+0.14)에 집중**되고
bridge(+0.01)는 거의 없다. 해석: comparison 질문(yes/no·둘 중 택1)은 정답이 선명해
F1 보상이 강한 학습 신호를 주지만, bridge의 개방형 답은 신호가 약하고 SFT가 이미 retrieve를
잘 해 헤드룸이 작다. RL이 고친 것은 주로 **비교 추론**이다(SFT가 양쪽 정보를 받고도
"Scott Derrickson vs Ed Wood 국적"을 틀리던 류).

**HA-3 — 정직한 음성/혼합 결과.** Solution A의 sports 전이(0.312)는 학습 없는 cosine(0.386)
**보다 낮다**. HotpotQA에 파인튜닝된 search/format이 룰북 도메인(번호 매겨진 규정 텍스트,
"six players and 2.43 m"식 복합 답)엔 과적합한 것. 다만 (i) RL은 전이에서도 SFT보다 낫고
(+0.013), (ii) comparison 전이는 강하다(0.507). 즉 **in-domain 성능과 OOD 강건성의 트레이드오프**가
드러난다 — 기반 프로젝트에서 "cosine은 전이에 견고"하던 사실과 정확히 대칭되는 통찰.

**기반 프로젝트와의 종합.** 두 보고서를 합치면 일관된 서사가 된다: *selection-only RL은
추론 그릇이 없어 휴리스틱을 못 넘었고(폴백), 추론을 LLM 안에 넣으면 in-domain은 명확히
넘지만(본 연구), 그 대가로 도메인 과적합이 생겨 전이 강건성은 오히려 휴리스틱에 진다.*

---

## 5. 한계와 향후

- **0.5B 천장·전이 과적합**: 1.5B+ 모델, 도메인 혼합 SFT, open-retrieval로 전이 강건성
  재검토. 보상에 format 외 검색-품질/근거-일치 신호 추가.
- **RL 비용**: vLLM 부재로 rollout이 순차 생성(~7s/개). batched rollout 엔진으로 가속하면
  더 긴 RL·더 많은 seed 가능.
- **SFT trace 품질**: 템플릿 기반이라 추론이 정형적. 교사 LLM trace나 rejection sampling으로
  품질을 올리면 bridge 헤드룸을 더 열 수 있다.

---

## 부록 — 재현
- 코드: `src/sol_a/` (build_sft_data, sft_train, search_env, reward_a, grpo_train, eval_a, aggregate_a)
- SFT: `python -m src.sol_a.sft_train --epochs 3`
- RL: `python -m src.sol_a.grpo_train --steps 100 --lr 3e-5 --kl-coef 0.01 --seed {42,123,7}`
  (중단 시 `--resume <out>/ckpt`로 이어하기)
- 평가/집계: `python -m src.sol_a.eval_a ...` → `python -m src.sol_a.aggregate_a`
- 결과 원본: `results/sol_a_*.json`
