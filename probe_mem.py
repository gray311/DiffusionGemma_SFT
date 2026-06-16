"""Measure peak GPU memory for MoE-LoRA training on a SINGLE A100: model +
forward + backward + AdamW optimizer step, batch=1, over a few real batches."""
import sys
import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

sys.path.insert(0, "/weka/home/ext-yingzima/DiffusionGemma_SFT")
from train_diffusiongemma_sft import MultimodalSFTDataset, MultimodalCollator, compute_diffusion_loss
from moe_lora import apply_moe_and_decoder_lora

MODEL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
DATA = "/weka/home/ext-yingzima/NVIDIA_Internship/data/test.json"
dev = "cuda:0"

proc = AutoProcessor.from_pretrained(MODEL)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
vocab = model.config.text_config.vocab_size
apply_moe_and_decoder_lora(model, r=16, alpha=32)
model.train()

trainables = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainables, lr=1e-4)

ds = MultimodalSFTDataset(DATA, proc, "/weka/home/xliu316/", "/weka/home/ext-yingzima/", 1024)
coll = MultimodalCollator(canvas_length=model.config.canvas_length, pad_token_id=0)

# pick a 3-image example (largest context) to stress memory
import json
data = json.load(open(DATA))
idx3 = next(i for i, e in enumerate(data) if len(e["image"]) == 3)
print("after load:", torch.cuda.memory_allocated(dev)/1e9, "GB")

for step, i in enumerate([idx3, 0, idx3]):
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll([ds[i]]).items()}
    opt.zero_grad()
    loss = compute_diffusion_loss(model, batch, vocab_size=vocab, eps_t=1e-3)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print(f"step {step} (ex {i}, ctx={batch['input_ids'].shape[1]}): loss={loss.item():.4f} "
          f"peak={torch.cuda.max_memory_allocated(dev)/1e9:.1f}GB", flush=True)
print("PEAK:", torch.cuda.max_memory_allocated(dev)/1e9, "GB")
