"""Measure per-example train step time for a target data spec:
4 images @ 512x446, prompt ~500 tokens, output ~300 tokens (canvas)."""
import sys, time
import torch
from PIL import Image
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
sys.path.insert(0, "/weka/home/ext-yingzima/DiffusionGemma_SFT")
from moe_lora import apply_moe_and_decoder_lora
from train_diffusiongemma_sft import compute_diffusion_loss

MODEL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
dev = "cuda:0"

proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").to(dev)
model.config.use_cache = False
vocab = model.config.text_config.vocab_size
apply_moe_and_decoder_lora(model, r=16, alpha=32)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

# build a prompt with 4 images (512x446) + ~500 text tokens
imgs = [Image.new("RGB", (512, 446), (i*40 % 255, 80, 160)) for i in range(4)]
words = "the ego vehicle approaches an intersection with pedestrians and traffic " * 40
content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": words}]
enc = proc.apply_chat_template([{"role": "user", "content": content}],
                               tokenize=True, add_generation_prompt=True,
                               return_dict=True, return_tensors="pt").to(dev)
ctx_len = enc["input_ids"].shape[1]
CANVAS = model.config.canvas_length          # native block (256)
OUT_LEN = 300                                 # target output length
n_blocks = (OUT_LEN + CANVAS - 1) // CANVAS   # blocks needed to cover 300 out tokens

# decoder canvas: one native block (the model trains block-by-block)
x0 = torch.randint(0, vocab, (1, CANVAS), device=dev)
batch = {
    "input_ids": enc["input_ids"],
    "attention_mask": torch.ones_like(enc["input_ids"]),
    "canvas_input_ids": x0,
    "canvas_loss_mask": torch.ones(1, CANVAS, dtype=torch.bool, device=dev),
    "decoder_position_ids": torch.arange(ctx_len, ctx_len + CANVAS, device=dev).unsqueeze(0),
    "decoder_attention_mask": torch.cat([torch.ones(1, ctx_len, device=dev),
                                         torch.ones(1, CANVAS, device=dev)], 1),
    "pixel_values": enc["pixel_values"],
    "image_position_ids": enc.get("image_position_ids"),
    "mm_token_type_ids": enc.get("mm_token_type_ids"),
}
print(f"ctx_len(prompt+4img)={ctx_len}  canvas={CANVAS}  blocks_for_300out={n_blocks}", flush=True)

# time forward+backward+step over a few iters (1 block = 1 step)
torch.cuda.synchronize(); times = []
for it in range(6):
    t0 = time.time()
    opt.zero_grad()
    loss = compute_diffusion_loss(model, batch, vocab_size=vocab, eps_t=1e-3)
    loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    if it >= 1: times.append(dt)            # drop warmup
    print(f"  iter {it}: {dt:.2f}s  peak={torch.cuda.max_memory_allocated(dev)/1e9:.1f}GB", flush=True)
per_block = sum(times) / len(times)
print(f"\nper-block(256) step: {per_block:.2f}s  | per-example(~{n_blocks} blocks for 300 out): {per_block*n_blocks:.2f}s")
