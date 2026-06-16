"""MoE-LoRA for DiffusionGemma: grouped per-expert low-rank adapters on the
expert weights (which are raw 3D nn.Parameters, so stock peft can't touch them),
plus a manual LoRA on the decoder attention Linears (the denoising path).

Everything is added manually (no peft) so it composes and saves cleanly.

    n = apply_moe_and_decoder_lora(model, r=16, alpha=32, moe=True, decoder_attn=True)
    ...train (only the *_lora_* params have requires_grad=True)...
    save_lora_state(model, path)        # at the end of training
    # eval: rebuild structure then load
    apply_moe_and_decoder_lora(base, ...); load_lora_state(base, path)
"""
import math

import torch
import torch.nn as nn
from transformers.integrations.moe import _grouped_linear


# --------------------------------------------------------------------------- #
# Expert MoE-LoRA: replicate grouped_mm_experts_forward, adding a grouped
# low-rank delta at the gate_up and down projections.
# --------------------------------------------------------------------------- #
def _moe_lora_forward(self, hidden_states, top_k_index, top_k_weights):
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids)
    selected = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
    expert_ids_g = expert_ids_g.clamp(max=self.num_experts - 1)
    selected = selected.masked_fill(sentinel_mask, 0.0)   # not in-place (autograd-safe)

    s = self._moe_lora_scaling
    # --- up (gate_up) projection: base + grouped LoRA ---
    proj = _grouped_linear(selected, self.gate_up_proj, offsets, is_transposed=self.is_transposed)
    gu_mid = _grouped_linear(selected, self.lora_gate_up_A, offsets, is_transposed=False)   # (S, r)
    gu_delta = _grouped_linear(gu_mid, self.lora_gate_up_B, offsets, is_transposed=False)   # (S, 2*inter)
    proj = proj + s * gu_delta

    proj = self._apply_gate(proj) if self.has_gate else self.act_fn(proj)   # (S, inter)

    # --- down projection: base + grouped LoRA ---
    base_d = _grouped_linear(proj, self.down_proj, offsets, is_transposed=self.is_transposed)
    dn_mid = _grouped_linear(proj, self.lora_down_A, offsets, is_transposed=False)          # (S, r)
    dn_delta = _grouped_linear(dn_mid, self.lora_down_B, offsets, is_transposed=False)      # (S, hidden)
    out = base_d + s * dn_delta

    weighted = out * sample_weights_g.unsqueeze(-1)
    weighted = weighted.masked_fill(sentinel_mask, 0.0)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted = weighted[inv_perm]
    final = weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final.to(hidden_states.dtype)


def _add_moe_lora(experts, r, alpha):
    ne, h, im = experts.num_experts, experts.hidden_dim, experts.intermediate_dim
    dev, dt = experts.gate_up_proj.device, experts.gate_up_proj.dtype
    gate_up_out = experts.gate_up_proj.shape[1]   # 2*intermediate
    down_out = experts.down_proj.shape[1]         # hidden
    # gate_up: in=h, out=gate_up_out ; down: in=im, out=down_out
    experts.lora_gate_up_A = nn.Parameter(torch.zeros(ne, r, h, device=dev, dtype=dt))
    experts.lora_gate_up_B = nn.Parameter(torch.zeros(ne, gate_up_out, r, device=dev, dtype=dt))
    experts.lora_down_A = nn.Parameter(torch.zeros(ne, r, im, device=dev, dtype=dt))
    experts.lora_down_B = nn.Parameter(torch.zeros(ne, down_out, r, device=dev, dtype=dt))
    for A in (experts.lora_gate_up_A, experts.lora_down_A):
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))   # B stays 0 -> initial delta = 0
    experts._moe_lora_scaling = alpha / r
    experts.gate_up_proj.requires_grad_(False)
    experts.down_proj.requires_grad_(False)
    # bind the LoRA forward (bypasses the grouped_mm dispatch)
    experts.forward = _moe_lora_forward.__get__(experts, type(experts))


# --------------------------------------------------------------------------- #
# Manual LoRA on nn.Linear (decoder attention).
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


def _wrap_decoder_attention(model, r, alpha):
    n = 0
    for name, mod in list(model.named_modules()):
        if "decoder.layers" in name and name.endswith("self_attn"):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(mod, proj, None)
                if isinstance(lin, nn.Linear):
                    setattr(mod, proj, LoRALinear(lin, r, alpha))
                    n += 1
    return n


# --------------------------------------------------------------------------- #
def apply_moe_and_decoder_lora(model, r=16, alpha=32, moe=True, decoder_attn=True):
    # freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)
    n_moe = n_attn = 0
    if moe:
        for _, mod in model.named_modules():
            if type(mod).__name__ == "DiffusionGemmaTextExperts":
                _add_moe_lora(mod, r, alpha)
                n_moe += 1
    if decoder_attn:
        n_attn = _wrap_decoder_attention(model, r, alpha)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MoE-LoRA: experts={n_moe}  decoder_attn_linears={n_attn}  "
          f"trainable={trainable/1e6:.1f}M", flush=True)
    return model


def save_lora_state(model, path):
    sd = {k: v for k, v in model.state_dict().items()
          if (".lora_" in k or "_lora_" in k)}
    torch.save(sd, path)
    return len(sd)


def load_lora_state(model, path, strict=False):
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    loaded = [k for k in sd]
    return len(loaded)
