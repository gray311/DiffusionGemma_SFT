"""Verify we can extract decoder router probabilities and compute a switch-style
load-balancing aux loss. Tests both output_router_logits=True and forward hooks."""
import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, "/weka/home/ext-yingzima/DiffusionGemma_SFT")
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
from train_diffusiongemma_sft import MultimodalSFTDataset, MultimodalCollator
from moe_lora import apply_moe_and_decoder_lora

MODEL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
DATA = "/weka/home/ext-yingzima/NVIDIA_Internship/data/test.json"
dev = "cuda:0"

proc = AutoProcessor.from_pretrained(MODEL)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").to(dev)
model.config.use_cache = False
vocab = model.config.text_config.vocab_size
ne = model.config.text_config.num_experts
tk = model.config.text_config.top_k_experts
print(f"num_experts={ne} top_k={tk}", flush=True)

# --- hook decoder routers only ---
captured = []
def hook(mod, inp, out):
    captured.append(out[0])   # router_probabilities (T, E), fp32
hooks = []
for name, mod in model.named_modules():
    if "decoder" in name and type(mod).__name__ == "DiffusionGemmaTextRouter":
        hooks.append(mod.register_forward_hook(hook))
print(f"hooked {len(hooks)} decoder routers", flush=True)

ds = MultimodalSFTDataset(DATA, proc, "/weka/home/xliu316/", "/weka/home/ext-yingzima/", 1024)
coll = MultimodalCollator(canvas_length=model.config.canvas_length, pad_token_id=0)
b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll([ds[0]]).items()}
am = b["attention_mask"]; am = None if bool((am == 1).all()) else am
captured.clear()
with torch.no_grad():
    out = model(input_ids=b["input_ids"], attention_mask=am, decoder_input_ids=b["canvas_input_ids"],
                decoder_position_ids=b["decoder_position_ids"], decoder_attention_mask=b["decoder_attention_mask"],
                pixel_values=b["pixel_values"].to(model.dtype), image_position_ids=b["image_position_ids"],
                mm_token_type_ids=b["mm_token_type_ids"])
# captured holds BOTH encoder and decoder router calls (decoder routers run during
# both the encoder prefill and the canvas denoise because enc/dec SHARE weights).
# The hook fires once per decoder-router module per forward pass it participates in.
print(f"hook captured: {len(captured)} tensors, shapes={[tuple(c.shape) for c in captured[:3]]}", flush=True)
# keep only the canvas-denoise calls: shape[0] == canvas tokens (256), not ctx tokens
canvas_len = b["canvas_input_ids"].shape[1]
dec_probs = [c for c in captured if c.shape[0] == canvas_len]
print(f"decoder(canvas) router tensors: {len(dec_probs)} (canvas_len={canvas_len})", flush=True)

# switch-style load-balancing on the DECODER router probs (canvas tokens only)
def load_balancing_loss(probs_list, num_experts, top_k):
    probs = torch.cat([p.reshape(-1, num_experts).float() for p in probs_list], dim=0)  # (T*, E)
    _, sel = torch.topk(probs, top_k, dim=-1)                  # (T*, K)
    expert_mask = F.one_hot(sel, num_experts).float()          # (T*, K, E)
    tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)     # (E,) frac of tokens with expert e in top-k
    router_prob_per_expert = probs.mean(dim=0)                 # (E,)
    return num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

aux_all = load_balancing_loss(captured, ne, tk)
aux_dec = load_balancing_loss(dec_probs, ne, tk) if dec_probs else aux_all
print(f"aux_loss all-router-calls: {aux_all.item():.4f}", flush=True)
print(f"aux_loss decoder/canvas only: {aux_dec.item():.4f}  (perfectly-balanced min = {tk}.0)", flush=True)
# perfectly balanced router => tokens_per_expert=top_k/E each, prob=1/E => loss = N*Σ (top_k/E)*(1/E)= top_k. So min ~= top_k.
