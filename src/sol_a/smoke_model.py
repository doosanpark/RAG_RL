"""Smoke test: Qwen2.5-0.5B-Instruct + LoRA 로드/생성/VRAM 점검 (8GB 가용성 확인).

실행:
    .\.venv\Scripts\python.exe -m src.sol_a.smoke_model
"""

from __future__ import annotations

import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def vram(tag: str) -> None:
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [VRAM] {tag}: allocated={used:.2f}GB peak={peak:.2f}GB")


def main() -> None:
    assert torch.cuda.is_available(), "CUDA 필요"
    dev = "cuda"
    print(f"[1/4] tokenizer/model 로드: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    vram("base model loaded")

    print("[2/4] LoRA 부착")
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    vram("lora attached")

    print("[3/4] chat 템플릿 생성")
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Answer the question. Reason step by step, then give the final answer "
                "inside <answer></answer>.\n\nQuestion: What is the capital of France?"
            ),
        },
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(dev)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)
    dt = time.time() - t0
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[4/4] 생성 ({dt:.2f}s, {out.shape[1]-inputs['input_ids'].shape[1]} new tokens):")
    print("-" * 60)
    print(gen)
    print("-" * 60)
    vram("after generate")
    print("\nOK: 8GB에서 0.5B+LoRA 로드/생성 정상.")


if __name__ == "__main__":
    main()
