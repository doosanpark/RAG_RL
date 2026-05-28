# 새 세션용 마스터 프롬프트 — 해결책 A (LLM을 RL로 파인튜닝)

> 아래 전체를 새 Claude 대화창 첫 메시지로 붙여넣으세요.
> 기존 selection-RL 프로젝트는 이미 완성·제출 가능 상태이고, 이건 그 위에 얹는
> **업그레이드(해결책 A)** 입니다. 실패해도 기존 결과로 폴백 가능합니다.

---

```
저는 RL 클래스 프로젝트의 후속 단계를 진행합니다. 기존에 "passage selection을
작은 MLP로 RL 학습"하는 버전을 완성했는데, 그게 학습 없는 cosine 휴리스틱을 못 넘었고
(in-domain 0.355 vs 0.370) 새 도메인 transfer도 실패(-24%)했습니다. 원인은 명확합니다:
LLM을 freeze하고 추론 없는 selector를 붙여서 multi-hop 추론을 담을 그릇이 없었던 것.
이번엔 그 한계를 해결하는 **해결책 A: 오픈 LLM(Qwen)을 LoRA로 RL 파인튜닝하여
검색+추론을 LLM 안에서 학습(Search-R1 방식)** 을 구현·디버깅·평가하도록 도와주세요.

[목표]
- Qwen2.5-0.5B(8GB) 또는 1.5B(16GB)-Instruct를 LoRA로 RL 파인튜닝.
- LLM이 "추론 → <search>질의</search> → 검색결과(<information>) → 추론 →
  (필요시 2차 <search>) → <answer>"를 생성하도록 학습. 추론을 LLM 안에 둠.
- 보상: outcome 기반 rule reward = answer EM/F1 (+ 선택적 format reward).
- 핵심 질문: "LLM을 RL 파인튜닝하면, 기존 frozen+cosine 파이프라인(answer_F1 0.370)과
  multi-hop(bridge)에서 그걸 넘는가? transfer는?"

[알고리즘 — 강의 06 범위 유지]
- REINFORCE + baseline을 LLM 토큰 생성에 적용(정책경사를 생성 시퀀스에 대해).
- 실무 표준인 GRPO/PPO는 옵션으로 검토하되, 기본은 REINFORCE+baseline로 범위 안에서.
- variance 감소: baseline, reward 정규화, gradient clipping. KL(참조정책) 페널티로
  포맷 붕괴 방지.

[압축 설계 — 기간 단축(이게 핵심)]
1. HotpotQA **distractor 세팅 유지** → 후보 10개가 주어지므로 외부 벡터DB/검색서버
   불필요(가장 무거운 엔지니어링 제거). "search"는 주어진 후보 풀 안에서의
   질의-기반 retrieve로 한정. (open-retrieval은 future work.)
2. **SFT warmup → 짧은 RL**: supporting_facts + 정답으로 expert 추론 trace를 템플릿
   생성("X 찾자 → [gold1] X=A → A의 Y 필요 → [gold2] Y=Z → 답:Z") → 이걸로 LoRA SFT
   (teacher-forced, 빠름) → 그 위에 짧은 RL로 다듬기. cold-start 탐색 비용 제거.
3. **HuggingFace TRL**(GRPOTrainer/PPOTrainer) 또는 verl로 RL 루프 직접 구현 회피.
4. **vLLM**로 배치 rollout 가속(5-10배).
5. run 수 최소화: 1방법 × 2 seed + frozen baseline 먼저, 시간 남으면 3 seed.
6. **자주 checkpoint 저장**(Colab 세션 끊김 대비, 매 N step).

[비교/baseline — 기존 결과 재사용]
- frozen LLM(파인튜닝 X) + cosine top-k = 기존 프로젝트 수치(answer_F1 0.370, oracle 0.557,
  use_all 0.367, random 0.275 / 스포츠 transfer cosine 0.386).
- 새 SFT+RL LLM이 이들을, 특히 bridge(multi-hop)·transfer에서 넘는지가 성패.

[지표] answer_F1(SQuAD 토큰 F1, 핵심), EM, support/joint F1(가능시), hop 타입별(bridge/comparison).

[데이터·환경 — 기존 자산]
- 저장소: c:\Users\erid3\Documents\workspace\RAG_RL (env.py, rewards.py의 compute_answer_f1,
  evaluate.py 재사용 가능; data/eval/sports.json = 스포츠 transfer셋 350문항).
- HotpotQA distractor 다운로드됨. supporting_facts 포맷: dict(title=list, sent_id=list).
- **venv Python 3.11 필수** (시스템 python은 3.14라 datasets/dill 깨짐). 실행은 항상
  `.\.venv\Scripts\python.exe`. 단, TRL/vLLM/peft가 최신 의존성을 요구하면 **별도 새 venv**를
  만들어야 할 수 있음(기존 핀: numpy1.26/pyarrow17/pandas2.3/datasets2.21와 충돌 가능).
- 하드웨어: 로컬 RTX 4060 Ti 8GB(연속가동 가능) 또는 Colab T4 16GB(세션제한, 체크포인트 필수).
  Colab Pro로 L4 24GB/A100 잡히면 적극 사용(학습 대폭 단축). 8GB면 0.5B+LoRA, 16GB면 1.5B 고려.
- 제출 기한: 6월 12일. 기존 프로젝트는 이미 제출 가능하므로 A는 실패 시 폴백 가능한 업그레이드.

[꼭 가져갈 교훈 — 같은 실수 반복 금지]
1. reward hacking 경계: reward↑인데 F1 평평 = 즉시 점검(과거 "빈 컨텍스트 즉시 stop"으로
   수렴, 빈 컨텍스트 F1 0.149가 reward 0.20과 일치). LLM판에선 "포맷 붕괴/parametric
   답변/검색 안 하고 바로 답" 등으로 나타남 → format+검색 사용 보상/패널티로 방어.
2. warmup(SFT/BC) 없이는 cold-start 국소최적에 빠짐 → SFT 먼저.
3. **단일 seed 믿지 말 것**: 과거 seed42 단독에서 가짜 우위가 3 seed로 사라짐.
4. RL은 불안정 → **dev-best checkpoint**로 peak 보존(끝점 아님).
5. LLM이 천장: 0.5B는 oracle도 0.557. 절대 수치 기대 낮추거나 1.5B로 천장↑.
6. **정직한 결과 우선**: "negative result + 좋은 분석 ≥ positive + 부실 분석". 안 되면
   왜 안 되는지 정량 규명이 곧 점수.

[작업 원칙]
1. 매 단계 작동 검증(특히 reward 파싱, rollout 포맷, SFT loss 수렴).
2. 학습 전 작은 smoke(수 샘플)로 파이프라인 확인 후 본 학습.
3. 막히면 디버깅 가설 먼저 제시 후 수정. 발산/포맷붕괴/OOM 미리 방어.
4. 8GB OOM 대비: LoRA + gradient checkpointing + 작은 batch + vLLM 분리.
5. 진행을 README/commit에 누적 기록. 백그라운드 학습은 Monitor로 dev 지표 추적,
   중단 시 `Get-Process python | Stop-Process -Force`(TaskStop만으론 tee의 python 안 죽음).

[권장 진행 순서]
0) 환경: TRL/peft/vllm 설치 가능한 새 venv 구성 + GPU/메모리 확인.
1) 데이터: supporting_facts로 SFT용 multi-hop 추론 trace 생성 + 포맷 정의(<search>/<information>/<answer>).
2) SFT warmup: Qwen+LoRA를 trace로 지도학습, loss 수렴·생성 포맷 확인.
3) RL 환경: distractor 후보 풀 기반 search/read + 보상(answer F1 + format) reward 함수.
4) smoke RL(수십 step) → reward hacking·포맷 점검.
5) 본 RL(짧게) 2 seed + dev-best checkpoint, vLLM rollout.
6) 평가: HotpotQA in-domain + sports transfer를 기존 evaluate.py로 frozen baseline과 비교
   (특히 bridge 타입). 3 seed 집계.
7) 보고서/슬라이드 업데이트.

준비됐으면 현재 저장소 상태와 GPU 환경부터 점검하고, "어디서부터 시작할지" 제안해주세요.
```

---

## 붙여넣은 뒤 새 세션에서 가장 먼저 시킬 것
1. `python -m src.verify_env`로 환경 확인(또는 TRL용 새 venv 구성)
2. GPU/VRAM 확인(`nvidia-smi`) → 0.5B vs 1.5B 결정
3. SFT trace 생성 설계부터 착수

## 기억할 폴백
해결책 A가 기한 내 안 되면, **기존 selection-RL 프로젝트(report.md 완성본)** 로 제출하면 됩니다.
A는 어디까지나 "더 강한 결과를 노리는 업그레이드"이고 critical path에 두지 마세요.
