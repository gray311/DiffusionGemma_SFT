"""Validate the DiffusionGemma SFT loss WITHOUT the 52GB weights / H100.

Builds a SMALL random-init model with the *same* architecture, then overfits a
single fixed batch and checks the denoising loss decreases normally. This proves
the collator + uniform-noise corruption + forward args + loss + backward are all
wired correctly (the real run just swaps in the pretrained 26B on the H100).

    conda activate dgemma
    python validate_loss.py
"""
import copy
import json

import torch
from transformers import AutoConfig, DiffusionGemmaForBlockDiffusion

from train_diffusiongemma_sft import BlockDiffusionCollator, compute_diffusion_loss

REAL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
DEV = "cuda:0"
torch.manual_seed(0)


def small_config():
    cfg = AutoConfig.from_pretrained(REAL)
    tc = cfg.text_config
    tc.hidden_size = 256
    tc.num_hidden_layers = 4
    # mirror the real model: mostly sliding + one full (this exercises the
    # sliding-window mask path that broke in the released transformers).
    tc.layer_types = ["sliding_attention", "sliding_attention",
                      "sliding_attention", "full_attention"]
    tc.sliding_window = 64
    tc.num_attention_heads = 4
    tc.head_dim = 64
    tc.global_head_dim = 64
    tc.num_key_value_heads = 2
    tc.num_global_key_value_heads = 1
    tc.intermediate_size = 512
    tc.moe_intermediate_size = 256
    tc.num_experts = 8
    tc.top_k_experts = 2
    tc.vocab_size = 2048
    tc.max_position_embeddings = 4096
    # vision tower: shrink (unused here, but built in __init__)
    vc = cfg.vision_config
    for k, v in dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=256).items():
        if hasattr(vc, k):
            setattr(vc, k, v)
    cfg.canvas_length = 16
    # keep special token ids inside the shrunk vocab
    cfg.image_token_id = 3
    cfg.boi_token_id = 4
    cfg.eoi_token_id = 5
    cfg.tie_word_embeddings = True
    return cfg


def fixed_batch(vocab, canvas_len, pad_id):
    # two (prompt, response) examples with small random ids
    g = torch.Generator().manual_seed(1)
    def rid(n, lo=10):
        return torch.randint(lo, vocab, (n,), generator=g).tolist()
    # single example (batch=1, no padding) — mirrors the per_device_batch=1
    # DeepSpeed setup and the encoder's no-padding mask path.
    examples = [
        {"prompt_ids": rid(12), "response_ids": rid(canvas_len)},
    ]
    collator = BlockDiffusionCollator(canvas_length=canvas_len, pad_token_id=pad_id,
                                      max_context_len=512)
    batch = collator(examples)
    return {k: v.to(DEV) for k, v in batch.items()}


def main():
    cfg = small_config()
    vocab = cfg.text_config.vocab_size
    print(f"small model: hidden={cfg.text_config.hidden_size} layers={cfg.text_config.num_hidden_layers} "
          f"experts={cfg.text_config.num_experts} vocab={vocab} canvas={cfg.canvas_length}", flush=True)

    model = DiffusionGemmaForBlockDiffusion(cfg).to(DEV).to(torch.float32)
    model.config.use_cache = False
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.1f}M", flush=True)

    batch = fixed_batch(vocab, cfg.canvas_length, pad_id=0)
    print("batch shapes:", {k: tuple(v.shape) for k, v in batch.items()}, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(0)
    print(f"\nrandom-guess CE ~ ln(vocab) = {torch.log(torch.tensor(float(vocab))):.3f}\n", flush=True)
    losses = []
    for step in range(200):
        loss = compute_diffusion_loss(model, batch, vocab_size=vocab, eps_t=1e-3)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == 199:
            print(f"step {step:3d}  loss {loss.item():.4f}  grad_norm {gnorm:.3f}", flush=True)

    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    print(f"\nmean loss  first5={first:.4f}  last5={last:.4f}  drop={first-last:.4f}", flush=True)
    print("RESULT:", "PASS — loss decreases" if last < first - 0.5 else "FAIL — loss not decreasing")


if __name__ == "__main__":
    main()
