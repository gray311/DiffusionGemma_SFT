"""Diagnostic: does the trained denoiser recover the answer (teacher-forced),
or did it never learn it? Separates 'didn't learn' from 'learned but generation
sampler fails'. For each example, corrupt the clean answer canvas at noise level
t and check if the model's argmax recovers the answer tokens (and exact text).

    python diagnose_recon.py <ADAPTER_or_base> [N] [gpu]
"""
import json
import sys

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

sys.path.insert(0, "/weka/home/ext-yingzima/DiffusionGemma_SFT")
from train_diffusiongemma_sft import MultimodalSFTDataset, MultimodalCollator

MODEL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
DATA = "/weka/home/ext-yingzima/NVIDIA_Internship/data/test.json"


def main():
    adapter = sys.argv[1] if len(sys.argv) > 1 else "base"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    gpu = sys.argv[3] if len(sys.argv) > 3 else "0"
    data_path = sys.argv[4] if len(sys.argv) > 4 else DATA
    dev = f"cuda:{gpu}"
    T_LEVELS = [0.2, 0.5, 0.8, 1.0]

    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    if adapter != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).to(dev).eval()
    vocab = model.config.text_config.vocab_size

    ds = MultimodalSFTDataset(data_path, proc, "/weka/home/xliu316/", "/weka/home/ext-yingzima/", 1024)
    coll = MultimodalCollator(canvas_length=model.config.canvas_length, pad_token_id=0)

    torch.manual_seed(0)
    # token-level recovery (argmax==x0 on answer positions) + exact-text recovery
    tok_hit = {t: [0, 0] for t in T_LEVELS}      # [correct_tokens, total_tokens]
    txt_hit = {t: 0 for t in T_LEVELS}
    n_done = 0
    data = json.load(open(data_path))
    idxs = [i for i, e in enumerate(data) if e["eval"] == "exact_match"][:n]
    for i in idxs:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll([ds[i]]).items()}
        x0 = b["canvas_input_ids"]
        cmask = b["canvas_loss_mask"]
        gold = tok.decode(x0[0][cmask[0]], skip_special_tokens=True)
        for t in T_LEVELS:
            tt = torch.full_like(x0, 0, dtype=torch.float).fill_(t)
            corrupt = (torch.rand_like(x0, dtype=torch.float) < t) & cmask
            x_t = torch.where(corrupt, torch.randint(0, vocab, x0.shape, device=dev), x0)
            am = b["attention_mask"]
            am = None if bool((am == 1).all()) else am
            with torch.no_grad():
                out = model(input_ids=b["input_ids"], attention_mask=am,
                            decoder_input_ids=x_t,
                            decoder_position_ids=b["decoder_position_ids"],
                            decoder_attention_mask=b["decoder_attention_mask"],
                            pixel_values=b["pixel_values"].to(model.dtype),
                            image_position_ids=b["image_position_ids"],
                            mm_token_type_ids=b["mm_token_type_ids"])
            pred = out.logits.argmax(-1)
            hit = ((pred == x0) & cmask).sum().item()
            tot = cmask.sum().item()
            tok_hit[t][0] += hit; tok_hit[t][1] += tot
            pred_txt = tok.decode(pred[0][cmask[0]], skip_special_tokens=True)
            txt_hit[t] += int(pred_txt.strip().lower() == gold.strip().lower())
        n_done += 1
        if n_done <= 3:
            print(f"[{i}] gold={gold!r}", flush=True)

    print(f"\n=== teacher-forced denoiser recovery ({n_done} examples, adapter={adapter.split('/')[-1]}) ===")
    print("  noise t |  token-recovery |  exact-text recovery")
    for t in T_LEVELS:
        c, tot = tok_hit[t]
        print(f"   {t:.1f}    |   {100*c/max(tot,1):5.1f}%       |   {100*txt_hit[t]/max(n_done,1):5.1f}%")
    print("\nIf recovery is HIGH at low t but generation acc is low -> sampler/gen gap.")
    print("If recovery is LOW even at t=0.2 -> the denoiser never learned the mapping.")


if __name__ == "__main__":
    main()
