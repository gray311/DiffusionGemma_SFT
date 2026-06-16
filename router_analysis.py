"""Measure the MoE decoder-router expert-usage distribution to check the
load-balancing loss works. For N test examples, teacher-force the gold answer
canvas, hook the 30 decoder routers, and accumulate per-expert top-k selection
counts + mean router probability (only at real answer positions).

    # analyze a checkpoint (dir with moe_lora.pt, or "none" for the base model)
    python router_analysis.py analyze <ckpt|none> <out.json> [N] [gpu]
    # plot two distributions side by side
    python router_analysis.py plot a.json:no-aux b.json:with-aux out.png
"""
import json
import os
import sys

import torch

MODEL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
DATA = "/weka/home/ext-yingzima/NVIDIA_Internship/data/test.json"


def analyze(ckpt, out_json, n, gpu):
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    sys.path.insert(0, "/weka/home/ext-yingzima/DiffusionGemma_SFT")
    from train_diffusiongemma_sft import MultimodalSFTDataset, MultimodalCollator
    from moe_lora import apply_moe_and_decoder_lora, load_lora_state

    dev = f"cuda:{gpu}"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    model.config.use_cache = False
    cfg = model.config.text_config
    ne, tk = cfg.num_experts, cfg.top_k_experts

    if ckpt != "none":
        moe_pt = ckpt if ckpt.endswith(".pt") else os.path.join(ckpt, "moe_lora.pt")
        apply_moe_and_decoder_lora(model, r=16, alpha=32, moe=True, decoder_attn=True,
                                   train_router=True)
        nt = load_lora_state(model, moe_pt)
        print(f"loaded {nt} tensors from {moe_pt}", flush=True)

    # hook decoder routers: capture (router_probs, _, top_k_index)
    cap = []
    def hook(mod, inp, out):
        cap.append((out[0].detach(), out[2].detach()))   # probs (T,E), idx (T,K)
    for name, mod in model.named_modules():
        if "decoder" in name and type(mod).__name__ == "DiffusionGemmaTextRouter":
            mod.register_forward_hook(hook)

    ds = MultimodalSFTDataset(DATA, proc, "/weka/home/xliu316/", "/weka/home/ext-yingzima/", 1024)
    coll = MultimodalCollator(canvas_length=model.config.canvas_length, pad_token_id=0)
    data = json.load(open(DATA))
    idxs = [i for i, e in enumerate(data) if e["eval"] == "exact_match"][:n]

    sel_counts = torch.zeros(ne, dtype=torch.float64)   # top-k selections per expert
    prob_sum = torch.zeros(ne, dtype=torch.float64)     # summed router prob per expert
    total_tokens = 0
    for k, i in enumerate(idxs):
        b = {kk: (v.to(dev) if torch.is_tensor(v) else v) for kk, v in coll([ds[i]]).items()}
        mask = b["canvas_loss_mask"][0]                  # (256,) real answer positions
        am = b["attention_mask"]; am = None if bool((am == 1).all()) else am
        cap.clear()
        with torch.no_grad():
            model(input_ids=b["input_ids"], attention_mask=am,
                  decoder_input_ids=b["canvas_input_ids"],          # teacher-forced clean canvas
                  decoder_position_ids=b["decoder_position_ids"],
                  decoder_attention_mask=b["decoder_attention_mask"],
                  pixel_values=b["pixel_values"].to(model.dtype),
                  image_position_ids=b["image_position_ids"],
                  mm_token_type_ids=b["mm_token_type_ids"])
        for probs, idx in cap:                            # 30 decoder layers
            probs = probs.reshape(-1, ne)[mask]           # (A, E) real positions only
            idx = idx.reshape(-1, tk)[mask]               # (A, K)
            prob_sum += probs.sum(dim=0).double().cpu()
            sel_counts += torch.bincount(idx.reshape(-1), minlength=ne).double().cpu()
            total_tokens += int(mask.sum())
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(idxs)} examples", flush=True)

    f = (sel_counts / sel_counts.sum()).tolist()          # selection fraction per expert
    P = (prob_sum / total_tokens).tolist()                # mean prob per expert
    tokens_per_expert = sel_counts / total_tokens         # sums to top_k
    aux = float(ne * torch.sum(tokens_per_expert * (prob_sum / total_tokens)))
    fr = torch.tensor(f)
    cv = float(fr.std(unbiased=False) / fr.mean())        # 0 = perfectly uniform
    dead = int((sel_counts == 0).sum())
    res = dict(ckpt=ckpt, n=len(idxs), num_experts=ne, top_k=tk,
               total_tokens=total_tokens, sel_fraction=f, mean_prob=P,
               aux_loss=aux, cv=cv, dead_experts=dead,
               max_frac=float(fr.max()), min_frac=float(fr.min()),
               uniform=1.0 / ne)
    json.dump(res, open(out_json, "w"))
    print(f"\n[{ckpt}] aux_loss={aux:.3f} (balanced floor={tk})  CV={cv:.3f} (0=uniform)  "
          f"dead={dead}/{ne}  max/min load={fr.max()/fr.max().clamp(min=1e-9):.0f}x"
          f"  -> {out_json}", flush=True)
    print(f"  max expert load={100*fr.max():.2f}%  min={100*fr.min():.2f}%  uniform={100/ne:.2f}%", flush=True)


def plot(specs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loaded = []
    for s in specs:
        path, label = s.split(":", 1)
        loaded.append((label, json.load(open(path))))
    fig, axes = plt.subplots(1, len(loaded), figsize=(7 * len(loaded), 4.2), squeeze=False)
    for ax, (label, r) in zip(axes[0], loaded):
        ne = r["num_experts"]
        frac = sorted([100 * x for x in r["sel_fraction"]], reverse=True)
        ax.bar(range(ne), frac, width=1.0, color="#3b7dd8")
        ax.axhline(100.0 / ne, color="red", ls="--", lw=1.2, label=f"uniform ({100/ne:.2f}%)")
        ax.set_title(f"{label}\nCV={r['cv']:.3f}  aux={r['aux_loss']:.2f}  dead={r['dead_experts']}/{ne}")
        ax.set_xlabel("expert rank (sorted by load)")
        ax.set_ylabel("% of top-k routing slots")
        ax.legend()
    fig.suptitle("Decoder-router expert load distribution (100 test examples)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        ckpt = sys.argv[2]
        out_json = sys.argv[3]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        gpu = sys.argv[5] if len(sys.argv) > 5 else "0"
        analyze(ckpt, out_json, n, gpu)
    elif sys.argv[1] == "plot":
        plot(sys.argv[2:-1], sys.argv[-1])
