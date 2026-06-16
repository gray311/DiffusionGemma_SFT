"""Fair eval using the user's THINKING-mode pipeline (eval/loaders/diffusiongemma.py),
optionally with a LoRA adapter. Lets us compare zero-shot vs SFT on equal footing.

    python eval_thinking.py <adapter_or_base> [N] [gpu]
"""
import json
import re
import sys

import torch

sys.path.insert(0, "/weka/home/ext-yingzima/NVIDIA_Internship")
from eval.loaders import diffusiongemma as L

DATA = "/weka/home/ext-yingzima/NVIDIA_Internship/data/test.json"


def norm(s):
    s = s.strip().lower()
    s = re.sub(r"[.\s]+$", "", s)
    return re.sub(r"\s+", " ", s)


def main():
    adapter = sys.argv[1] if len(sys.argv) > 1 else "base"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    gpu = sys.argv[3] if len(sys.argv) > 3 else "0"
    dev = f"cuda:{gpu}"

    bundle = L.load(device=dev)
    if adapter != "base":
        from peft import PeftModel
        bundle["model"] = PeftModel.from_pretrained(bundle["model"], adapter).to(dev).eval()
        print(f"adapter: {adapter}", flush=True)

    data = json.load(open(DATA))
    idxs = [i for i, e in enumerate(data) if e["eval"] == "exact_match"][:n]
    per_cat = {}
    for k, i in enumerate(idxs):
        ex = data[i]
        pred, _ = L.generate(bundle, ex["image"], ex["question"])
        ok = norm(pred) == norm(ex["answer"])
        c = ex["category"]
        per_cat.setdefault(c, [0, 0])
        per_cat[c][0] += int(ok); per_cat[c][1] += 1
        if k < 12:
            print(f"[{i}] {c:13} gold={ex['answer'][:18]!r:20} pred={pred[:30]!r:32} {'OK' if ok else 'XX'}", flush=True)

    print(f"\n=== thinking-mode eval ({adapter.split('/')[-1]}) ===")
    tc = tn = 0
    for c, (cor, n_) in sorted(per_cat.items()):
        tc += cor; tn += n_
        print(f"  {c:14} {100*cor/n_:5.1f}%  ({n_})")
    print(f"\nOVERALL exact-match: {tc}/{tn} = {100*tc/max(tn,1):.1f}%")


if __name__ == "__main__":
    main()
