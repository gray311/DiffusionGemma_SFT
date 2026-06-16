"""Test JSON CoT + trajectory generation from the CoT-finetuned checkpoint.

For a few examples (train + held-out), build the inference prompt (1 image +
instruction, matching training), block-diffusion generate, decode, and check
whether the output is the expected JSON CoT format with a trajectory.

    python gen_cot.py <ckpt_dir|base> [n] [gpu] [first_train_idx]
"""
import json
import re
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

MODEL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
COT = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_training_cot.json"
WAYMO = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/waymo/"
INFER_TPL = open("/weka/home/ext-yingzima/DiffusionGemma_SFT/chat_templates/diffusion_gemma.jinja").read()


def to_user_text(ex):
    conv = ex["conversations"]
    last = max(i for i, m in enumerate(conv) if m["from"] in ("gpt", "assistant"))
    user = "\n".join(m["value"] for m in conv[:last] if m["from"] in ("human", "user"))
    return user.replace("<image>", "").strip(), conv[last]["value"]


def looks_like_cot_json(s):
    try:
        obj = json.loads(s)
        has_traj = bool(re.search(r"\[\s*[-+]?\d", s)) or "waypoint" in s.lower() or "trajectory" in s.lower()
        return True, (isinstance(obj, dict) and len(obj) > 0), has_traj
    except Exception:
        # partial: at least starts like the structured answer
        return False, s.strip().startswith("{"), bool(re.search(r"\[\s*[-+]?\d", s))


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "base"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    gpu = sys.argv[3] if len(sys.argv) > 3 else "0"
    first_train = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    dev = f"cuda:{gpu}"

    proc = AutoProcessor.from_pretrained(MODEL)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to(dev).eval()
    if ckpt != "base":
        import os
        sys.path.insert(0, "/weka/home/ext-yingzima/DiffusionGemma_SFT")
        from moe_lora import apply_moe_and_decoder_lora, load_lora_state
        moe_pt = ckpt if ckpt.endswith(".pt") else os.path.join(ckpt, "moe_lora.pt")
        apply_moe_and_decoder_lora(model, r=16, alpha=32, moe=True, linears="official", train_router=True)
        nt = load_lora_state(model, moe_pt)
        model = model.eval()
        print(f"loaded {nt} lora tensors from {moe_pt}", flush=True)

    data = json.load(open(COT))
    # mix: a few TRAIN examples (memorization) + a few HELD-OUT (generalization)
    idxs = list(range(first_train, first_train + n)) + list(range(500, 500 + n))
    ok_json = 0
    for tag, i in [("train", j) for j in idxs[:n]] + [("held-out", j) for j in idxs[n:]]:
        ex = data[i]
        user_text, gold = to_user_text(ex)
        img = Image.open(WAYMO + ex["image"][0]).convert("RGB")   # 1 image, matching training
        content = [{"type": "image", "image": img}, {"type": "text", "text": user_text}]
        inputs = proc.apply_chat_template(
            [{"role": "user", "content": content}], chat_template=INFER_TPL,
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(dev)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=512)
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        pred = proc.decode(seq[0][n_in:], skip_special_tokens=True)
        valid, structured, has_traj = looks_like_cot_json(pred)
        ok_json += int(valid)
        print(f"\n========== [{tag} idx {i}]  valid_json={valid}  structured={structured}  has_traj={has_traj} ==========")
        print(f"GOLD ({len(gold)} chars): {gold[:300]}")
        print(f"PRED ({len(pred)} chars): {pred[:400]}")
    print(f"\n=== valid JSON: {ok_json}/{2*n} ===", flush=True)


if __name__ == "__main__":
    main()
