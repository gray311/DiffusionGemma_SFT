"""Evaluate a (LoRA-)SFT'd DiffusionGemma on the multimodal test set.

For each example: build the image+text prompt, model.generate, decode the answer,
and exact-match against the gold answer (normalized). Reports per-category and
overall accuracy.

Single GPU (all examples):
    python eval_multimodal.py <ADAPTER_DIR_or_'base'> --gpu 0
Sharded across 2 GPUs (run both, then merge):
    python eval_multimodal.py <ADAPTER> --gpu 0 --shard 0 --nshards 2 --out /tmp/ev0.json &
    python eval_multimodal.py <ADAPTER> --gpu 1 --shard 1 --nshards 2 --out /tmp/ev1.json &
    # merge: python eval_multimodal.py --merge /tmp/ev0.json /tmp/ev1.json
"""
import json
import re
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

MODEL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
DATA = "/weka/home/ext-yingzima/NVIDIA_Internship/data/test.json"
FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")


def norm(s):
    s = s.strip().lower()
    s = re.sub(r"[.\s]+$", "", s)          # trailing periods/space
    s = re.sub(r"\s+", " ", s)
    return s


def _merge(files):
    per_cat = {}
    for f in files:
        for c, vals in json.load(open(f)).items():
            per_cat.setdefault(c, [0, 0, 0])
            for j in range(3):
                per_cat[c][j] += vals[j]
    print("=== MERGED per-category accuracy — strict / lenient ===")
    ts = tl = tn = 0
    for c, (s, l, n_) in sorted(per_cat.items()):
        ts += s; tl += l; tn += n_
        print(f"  {c:14} strict {100*s/n_:5.1f}%  lenient {100*l/n_:5.1f}%  ({n_})")
    accs = 100 * ts / max(tn, 1)
    accl = 100 * tl / max(tn, 1)
    print(f"\nOVERALL: strict-EM {ts}/{tn} = {accs:.1f}%   lenient {tl}/{tn} = {accl:.1f}%")
    print("RESULT(strict):", "LEARNED (>=90%)" if accs >= 90 else f"below 90% ({accs:.1f}%)")


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    if "--merge" in sys.argv:
        _merge(sys.argv[sys.argv.index("--merge") + 1:])
        return
    adapter = sys.argv[1] if len(sys.argv) > 1 else "base"
    n = int(_arg("--n", "0"))               # 0 = all
    gpu = _arg("--gpu", "0")
    shard = int(_arg("--shard", "0"))
    nshards = int(_arg("--nshards", "1"))
    out = _arg("--out", None)
    data_path = _arg("--data", DATA)
    dev = f"cuda:{gpu}"

    proc = AutoProcessor.from_pretrained(MODEL)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, dtype=torch.bfloat16,
    ).to(dev).eval()
    if adapter != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).to(dev).eval()
        print(f"loaded adapter: {adapter}", flush=True)

    data = json.load(open(data_path))
    if n:
        data = data[:n]
    data = list(enumerate(data))[shard::nshards]    # strided shard

    fix = lambda p: p.replace(FIX[0], FIX[1])
    per_cat = {}        # cat -> [correct, total]
    t0 = time.time()
    for i, ex in data:
        if ex["eval"] != "exact_match":
            continue    # speed_path_micro handled separately; skip from EM acc
        imgs = [Image.open(fix(p)).convert("RGB") for p in ex["image"]]
        qtext = ex["question"].replace("<image>", "").strip()
        content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": qtext}]
        inputs = proc.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(dev)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        n_in = inputs["input_ids"].shape[1]
        gen_kw = {"max_new_tokens": 64}
        _steps = _arg("--steps", None)
        if _steps is not None:
            gen_kw["max_denoising_steps"] = int(_steps)
        with torch.no_grad():
            gen = model.generate(**inputs, **gen_kw)   # NB: not `out` (that's the filename)
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        pred = proc.decode(seq[0][n_in:], skip_special_tokens=True)
        np_, ng = norm(pred), norm(ex["answer"])
        strict = np_ == ng
        lenient = bool(ng) and (ng in np_)            # gold answer appears in prediction
        c = ex["category"]
        per_cat.setdefault(c, [0, 0, 0])              # [strict, lenient, total]
        per_cat[c][0] += int(strict)
        per_cat[c][1] += int(lenient)
        per_cat[c][2] += 1
        if i < 20 or i % 100 == 0:
            print(f"[{i}] {c:13} gold={ex['answer'][:22]!r:24} pred={pred[:30]!r:32} "
                  f"{'S' if strict else '-'}{'L' if lenient else '-'}", flush=True)

    if out:
        json.dump(per_cat, open(out, "w"))
        print(f"shard {shard}: wrote {out}", flush=True)
    print("\n=== per-category accuracy (this shard) — strict / lenient ===", flush=True)
    ts = tl = tn = 0
    for c, (s, l, n_) in sorted(per_cat.items()):
        ts += s; tl += l; tn += n_
        print(f"  {c:14} strict {100*s/n_:5.1f}%  lenient {100*l/n_:5.1f}%  ({n_})")
    print(f"\nOVERALL (shard {shard}): strict {100*ts/max(tn,1):.1f}%  lenient {100*tl/max(tn,1):.1f}%"
          f"  n={tn}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
