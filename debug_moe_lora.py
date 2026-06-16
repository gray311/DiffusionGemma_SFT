"""Debug MoE-LoRA: load model on one A100, apply MoE+decoder-attn LoRA, run a
forward+loss+backward on one multimodal batch, verify grouped LoRA forward is
correct and gradients flow to the MoE-LoRA params (not the frozen base)."""
import sys

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

sys.path.insert(0, "/weka/home/ext-yingzima/DiffusionGemma_SFT")
from train_diffusiongemma_sft import (MultimodalSFTDataset, MultimodalCollator,
                                       compute_diffusion_loss)
from moe_lora import apply_moe_and_decoder_lora, save_lora_state, load_lora_state

MODEL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
DATA = "/weka/home/ext-yingzima/NVIDIA_Internship/data/test.json"
dev = "cuda:0"

proc = AutoProcessor.from_pretrained(MODEL)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
vocab = model.config.text_config.vocab_size

# --- sanity: a single experts module, compare base-only vs LoRA(=0) forward ---
experts = None
for _, mod in model.named_modules():
    if type(mod).__name__ == "DiffusionGemmaTextExperts":
        experts = mod
        break
print("found experts:", type(experts).__name__,
      "gate_up", tuple(experts.gate_up_proj.shape),
      "down", tuple(experts.down_proj.shape),
      "is_transposed", experts.is_transposed, "has_gate", experts.has_gate)

# capture the stock forward output on a random token batch BEFORE patching
torch.manual_seed(0)
S = 17
hs = torch.randn(S, experts.hidden_dim, device=dev, dtype=torch.bfloat16)
# fabricate a top_k_index/top_k_weights like the router would produce
num_top_k = 8
tki = torch.randint(0, experts.num_experts, (S, num_top_k), device=dev)
tkw = torch.softmax(torch.randn(S, num_top_k, device=dev), dim=-1).to(torch.bfloat16)
with torch.no_grad():
    base_out = experts(hs, tki, tkw).clone()

apply_moe_and_decoder_lora(model, r=16, alpha=32, moe=True, decoder_attn=True)

# B==0 so LoRA delta is exactly 0 -> patched forward must match stock forward
with torch.no_grad():
    lora0_out = experts(hs, tki, tkw)
diff = (base_out.float() - lora0_out.float()).abs().max().item()
print(f"[forward-equiv @ B=0] max|base - lora|: {diff:.3e}  (should be ~0)")

# now perturb a B so delta is nonzero, confirm it changes the output
with torch.no_grad():
    experts.lora_gate_up_B.add_(0.01)
    lorad_out = experts(hs, tki, tkw)
print(f"[forward delta @ B!=0] max change: "
      f"{(lorad_out.float()-base_out.float()).abs().max().item():.3e}  (should be >0)")
with torch.no_grad():
    experts.lora_gate_up_B.add_(-0.01)   # restore

# --- full forward + loss + backward on a real multimodal batch ---
ds = MultimodalSFTDataset(DATA, proc, "/weka/home/xliu316/", "/weka/home/ext-yingzima/", 1024)
coll = MultimodalCollator(canvas_length=model.config.canvas_length, pad_token_id=0)
batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll([ds[0]]).items()}

model.train()
loss = compute_diffusion_loss(model, batch, vocab_size=vocab, eps_t=1e-3)
print(f"[loss] {loss.item():.4f}")
loss.backward()

# verify grads: lora params should have nonzero grad, base experts should be None
g_lora_moe = g_lora_attn = n_moe = n_attn = 0
base_has_grad = 0
for name, p in model.named_parameters():
    if ".lora_" in name or "_lora_" in name:
        if "down_proj" in name or "gate_up" in name or "lora_gate_up" in name or "lora_down" in name:
            n_moe += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                g_lora_moe += 1
        else:
            n_attn += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                g_lora_attn += 1
    elif p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
        base_has_grad += 1

print(f"[grad] MoE-LoRA params with grad: {g_lora_moe}/{n_moe}")
print(f"[grad] decoder-attn LoRA params with grad: {g_lora_attn}/{n_attn}")
print(f"[grad] trainable BASE params w/ nonzero grad (should be 0): {base_has_grad}")

# --- save/load roundtrip ---
nsv = save_lora_state(model, "/tmp/moe_lora_test.pt")
print(f"[save] {nsv} lora tensors")
nl = load_lora_state(model, "/tmp/moe_lora_test.pt")
print(f"[load] {nl} lora tensors")
print("OK")
