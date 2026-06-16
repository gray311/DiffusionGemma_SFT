# DiffusionGemma SFT (HuggingFace Trainer + DeepSpeed)

SFT for `google/diffusiongemma-26B-A4B-it` — a **block-diffusion** MoE model
(encoder prefills the prompt into a KV cache; a bidirectional decoder denoises a
256-token "canvas"). The model's `forward` returns **only logits**; the
diffusion loss is computed in the trainer (per HF PRs #46568 / #46572).

## Files
```
train_diffusiongemma_sft.py   Trainer + collator + diffusion loss (the deliverable)
validate_loss.py              overfit a SMALL same-arch model -> proves loss decreases
ds_config_zero3.json          DeepSpeed ZeRO-3 (param-shards the 52GB base)
run_sft_deepspeed.sh          multi-GPU launcher (ZeRO-3)
run_sft.sh                    single-GPU launcher (80GB H100, model fits)
data/example_sft.jsonl        toy chat data
```

## Training objective (uniform discrete diffusion)
The model's sampler initializes a canvas with **random vocab tokens** and renoises
rejected positions with random tokens — there is **no [MASK] token**. So SFT is:
1. one (prompt, response) pair → canvas = first 256 response tokens (`--single_block`),
   context = prompt (encoder).
2. sample t~U(eps,1); replace each canvas token with a uniform-random token w.p. t → x_t.
3. `logits = model(input_ids=context, decoder_input_ids=x_t, ...)`.
4. loss = cross-entropy(logits, x0) over the canvas (the denoiser predicts x0 at
   every position; uniform corruption means it can't tell which are clean).
Optional: `--weight_by_t` (1/t ELBO weighting), `--self_cond_prob`,
`--encoder_ar_loss_weight`.

## Status (validated ✅ / blocked ⚠️)

**✅ Loss decreases normally** — `python validate_loss.py` overfits a small
same-architecture model (sliding-attention layers, batch=1): loss 6.8 → ~0.01.
This validates the collator + uniform-noise corruption + forward args + loss +
backward.

**✅ DeepSpeed ZeRO-3 shards the real 26B** across 4×L40S (46GB) with no OOM /
no meta-tensor errors; LoRA applied (3.98M trainable), training loop starts.

**⚠️ The real-26B training FORWARD is blocked by upstream transformers bugs** in
DiffusionGemma's brand-new (2026-06) training path — present in both the 5.12.1
release and `main` (5.13.0.dev0), independent of DeepSpeed/hardware:
  1. Encoder vision-bidirectional mask → 5D tensor in `sdpa_mask` (worked around
     by disabling `use_bidirectional_attention` for text-only — a no-op when
     there are no image tokens).
  2. `eager` attention path → `repeat_kv` "too many values to unpack".
  3. Encoder↔decoder **KV-cache shape mismatch** (`torch.cat` of 5D vs 4D) in the
     combined encoder→decoder training forward for this config. **Not yet worked
     around** — needs an upstream fix (or faithfully replicating the generation
     KV/mask construction, ~800 lines).

The model's *generation* works (inference is fine); only the training forward is
affected. The "Make trainable" PR was validated on a small 1.26B test config; the
full 26B config (global attention + sliding + MoE) trips paths the small configs
don't — our own small-model proxy passes for the same reason.

## How to run

Single GPU (80GB H100, model fits — no DeepSpeed):
```bash
conda activate dgemma
bash run_sft.sh
```

Multi-GPU (ZeRO-3, e.g. 4×L40S):
```bash
conda activate dgemma
bash run_sft_deepspeed.sh        # or: NUM_GPUS=8 bash run_sft_deepspeed.sh
```

Validate the loss decreases (no big weights / GPU needed beyond one L40S):
```bash
python validate_loss.py
```

## Environment
`dgemma` conda env: transformers @ git main (5.13.0.dev0 — needed for the
DiffusionGemma trainability fixes), torch 2.12+cu130, deepspeed 0.19, peft 0.19,
torchvision. Model + HF cache live under
`/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/`.

## Notes
- LoRA targets the attention projections' inner linears
  (`q_proj.linear,k_proj.linear,v_proj.linear,o_proj.linear`); the MoE experts
  are raw `nn.Parameter`s (not LoRA-able). Full FT of 26B needs ZeRO-3 + offload.
- `per_device_train_batch_size=1` (no intra-batch padding) — the encoder's mask
  path doesn't handle naive right-padding; multi-example batches need
  length-packing.
- Multimodal (image) SFT: route `pixel_values` through the encoder; the canvas
  stays text. Left as an extension (text-only here).
