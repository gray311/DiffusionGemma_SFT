"""SFT for google/diffusiongemma-26B-A4B-it with the HuggingFace Trainer.

DiffusionGemma is a *block-diffusion* model (encoder-decoder): an autoregressive
encoder prefills the prompt context into a KV cache, and a bidirectional decoder
denoises a fixed-length block of tokens (a "canvas", default 256) conditioned on
that cache. The model's `forward` returns ONLY logits over the canvas — the
diffusion loss is computed here in the trainer (confirmed by HF PRs #46568 /
#46572).

Training objective (uniform discrete diffusion — matches the model's sampler,
which initializes a canvas with RANDOM vocab tokens and renoises rejected
positions with random tokens; there is NO [MASK] token):

  1. Take a (prompt, response) pair. Pick one response block of `canvas_length`
     tokens as the canvas x0; everything before it (prompt + earlier response
     tokens) is the encoder context.
  2. Sample a noise level t ~ U(eps, 1). Corrupt the canvas: each canvas token is
     replaced by a uniform-random vocab token with prob t  ->  x_t.
  3. logits = model(input_ids=context, decoder_input_ids=x_t, ...)
  4. Loss = cross-entropy(logits, x0) over the (valid) canvas positions. Because
     the corruption is uniform (the model can't tell which tokens are clean), the
     denoiser is trained to predict x0 at EVERY canvas position. Optional 1/t
     ELBO-style weighting via --weight_by_t.

Notes / scope:
  * Text SFT. Multimodal (image) SFT needs pixel_values routed through the
    encoder — hooks are marked with `MULTIMODAL:` below.
  * 26B params do not fit for full fine-tuning on one 80GB GPU, so LoRA on the
    attention projections is the default. Gradient checkpointing is OFF because
    transformers 5.12.1 sets supports_gradient_checkpointing=False for this model
    (PR #46572 re-enables it in a later release).
  * Self-conditioning (the model's `self_conditioning_logits/_mask`) is supported
    via --self_cond_prob (default 0, i.e. off; doubles the forward cost when on).

Smoke test (a few steps on the toy data, no real training):
    python train_diffusiongemma_sft.py --smoke

Real run example: see run_sft.sh
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    DiffusionGemmaForBlockDiffusion,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
@dataclass
class ScriptArgs:
    model_path: str = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
    train_file: str = "data/example_sft.jsonl"
    # JSONL; each line either {"messages": [{"role","content"}, ...]} OR
    # {"prompt": "...", "response": "..."}.
    # --- multimodal (image) SFT ---
    multimodal: bool = False
    # JSON list of {"image": [paths], "question": "...<image>...", "answer": "..."}.
    image_path_from: str = "/weka/home/xliu316/"      # rewrite image paths ...
    image_path_to: str = "/weka/home/ext-yingzima/"   # ... to here
    max_context_len: int = 1024  # encoder context cap (prompt + response prefix)
    single_block: bool = True    # train on the first canvas block (256 tokens) only
    eps_t: float = 1e-3          # min noise level
    weight_by_t: bool = False    # 1/t ELBO-style weighting of the per-token loss
    self_cond_prob: float = 0.0  # prob of enabling self-conditioning per example
    encoder_ar_loss_weight: float = 0.0  # add lambda * AR next-token loss on the encoder context
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # lora_mode: "peft" -> stock peft on nn.Linear (can't reach MoE experts);
    #            "moe"  -> manual MoE-LoRA on the 128 experts (88% of params,
    #                      raw 3D Parameters peft can't touch) + manual LoRA on
    #                      the decoder attention. Fits one 80GB A100 (~70GB peak).
    lora_mode: str = "peft"
    moe_decoder_attn: bool = True   # (moe mode) also LoRA the decoder attention
    # MoE load-balancing (router auxiliary) loss — keeps the router from
    # collapsing onto a few experts (HF PR #46642). 0 = off (default, matching
    # the PR). Typical: 1e-2. Only has an effect when the router is trainable:
    # full FT, or moe-LoRA with --train_router True (auto-enabled when coef>0).
    router_aux_loss_coef: float = 0.0
    train_router: bool = False      # (moe mode) unfreeze decoder routers
    # IMPORTANT (this model is special): encoder attention is wrapped in
    # Gemma4ClippableLinear (inner ".linear") while DECODER attention is a bare
    # nn.Linear, and encoder/decoder share weights. Targeting "q_proj.linear"
    # ONLY hits the encoder (prefill) and misses the decoder denoising path
    # entirely. Use "all-linear" so peft wraps every nn.Linear on BOTH sides
    # (enc+dec attention + dense MLP). The 128 MoE experts are raw Parameters
    # and can't be LoRA'd (need full FT for those).
    lora_target: str = "all-linear"
    # placement: "" -> load to CPU, Trainer moves to a single GPU (e.g. 80GB H100,
    # where the 52GB model fits). "auto" -> shard across all visible GPUs
    # (model-parallel; needed when no single GPU holds the model, e.g. 4xL40S).
    device_map: str = ""
    # (model-parallel) per-GPU GiB cap at load time to FORCE the model to split
    # across GPUs. 0 = use 92% of GPU0 (which leaves the model on one GPU if it
    # fits). Set e.g. 30 to split the 50GB base ~evenly over 2 GPUs.
    mp_cap_gib: int = 0
    # attn impl: the sdpa mask builder mishandles this model's sliding+vision
    # bidirectional mask during a training forward (5D-expand crash), so default
    # to eager. Set "sdpa"/"flash_attention_2" once that path is fixed upstream.
    attn_implementation: str = "eager"
    # misc
    smoke: bool = False


# --------------------------------------------------------------------------- #
# Dataset: tokenize once into (prompt_ids, response_ids)
# --------------------------------------------------------------------------- #
class ChatSFTDataset(Dataset):
    def __init__(self, path, processor, max_context_len):
        self.examples = []
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "messages" in row:
                    msgs = row["messages"]
                    assert msgs[-1]["role"] == "assistant", "last message must be assistant"
                    prompt_msgs = msgs[:-1]
                    answer = msgs[-1]["content"]
                else:
                    prompt_msgs = [{"role": "user", "content": row["prompt"]}]
                    answer = row["response"]
                # prompt = chat template up to the generation prompt
                prompt_ids = processor.apply_chat_template(
                    prompt_msgs, tokenize=True, add_generation_prompt=True,
                )
                if isinstance(prompt_ids, dict):
                    prompt_ids = prompt_ids["input_ids"]
                if hasattr(prompt_ids, "tolist"):
                    prompt_ids = prompt_ids.tolist()
                    if prompt_ids and isinstance(prompt_ids[0], list):
                        prompt_ids = prompt_ids[0]
                # response tokens (plus EOS) — what the canvas must denoise
                resp_ids = tok(answer, add_special_tokens=False)["input_ids"]
                eos = tok.eos_token_id if tok.eos_token_id is not None else 1
                resp_ids = resp_ids + [eos]
                # cap context so prompt fits
                if len(prompt_ids) > max_context_len:
                    prompt_ids = prompt_ids[-max_context_len:]
                self.examples.append({"prompt_ids": prompt_ids, "response_ids": resp_ids})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


# --------------------------------------------------------------------------- #
# Multimodal dataset: image+text prompt -> processor (input_ids w/ image tokens
# + pixel_values + image_position_ids + mm_token_type_ids); answer -> canvas.
# Processed lazily (1000s of images won't fit if eager).
# --------------------------------------------------------------------------- #
class MultimodalSFTDataset(Dataset):
    def __init__(self, path, processor, path_from, path_to, max_context_len):
        self.data = json.load(open(path))
        self.processor = processor
        self.tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.path_from, self.path_to = path_from, path_to
        self.max_context_len = max_context_len

    def __len__(self):
        return len(self.data)

    def _fix(self, p):
        return p.replace(self.path_from, self.path_to)

    def __getitem__(self, i):
        from PIL import Image
        ex = self.data[i]
        imgs = [Image.open(self._fix(p)).convert("RGB") for p in ex["image"]]
        # images are passed as content items; drop the literal <image> markers
        qtext = ex["question"].replace("<image>", "").strip()
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": qtext})
        proc = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        # input_ids/attention_mask/mm_token_type_ids have a batch dim (1, seq) ->
        # drop it. pixel_values/image_position_ids are (num_images, ...) with NO
        # batch dim (first dim = image count) -> keep all images.
        per_image = {"pixel_values", "image_position_ids"}
        prompt = {k: (v if k in per_image else v[0]) for k, v in proc.items()}
        eos = self.tok.eos_token_id if self.tok.eos_token_id is not None else 1
        resp_ids = self.tok(ex["answer"], add_special_tokens=False)["input_ids"] + [eos]
        return {"prompt": prompt, "response_ids": resp_ids}


@dataclass
class MultimodalCollator:
    """batch_size=1 collator for multimodal SFT: the whole image+text prompt is
    the encoder context, the answer's first block is the canvas."""
    canvas_length: int
    pad_token_id: int

    def __call__(self, batch):
        assert len(batch) == 1, "multimodal SFT uses per_device_train_batch_size=1"
        ex = batch[0]
        p = ex["prompt"]
        L = self.canvas_length
        r = ex["response_ids"][:L]
        cmask = [1] * len(r) + [0] * (L - len(r))
        canvas = r + [self.pad_token_id] * (L - len(r))

        ctx = p["input_ids"]                       # (ctx_len,)
        ctx_len = ctx.shape[0]
        attn = p["attention_mask"]                 # (ctx_len,)
        dec_pos = (int(attn.sum()) + torch.arange(L)).long()
        dec_attn = torch.cat([attn, torch.ones(L, dtype=attn.dtype)])

        out = {
            "input_ids": ctx.unsqueeze(0),
            "attention_mask": attn.unsqueeze(0),
            # (num_images, patches, dim) — NOT batched; the encoder flattens
            # all images of the (batch=1) example together.
            "pixel_values": p["pixel_values"],
            "image_position_ids": p["image_position_ids"],
            "mm_token_type_ids": p["mm_token_type_ids"].unsqueeze(0),
            "canvas_input_ids": torch.tensor(canvas, dtype=torch.long).unsqueeze(0),
            "canvas_loss_mask": torch.tensor(cmask, dtype=torch.bool).unsqueeze(0),
            "decoder_position_ids": dec_pos.unsqueeze(0),
            "decoder_attention_mask": dec_attn.unsqueeze(0),
        }
        return out


# --------------------------------------------------------------------------- #
# Collator: pick a random response block as the canvas; build encoder context,
# right-padded; emit clean canvas x0 + masks + decoder positions.
# --------------------------------------------------------------------------- #
@dataclass
class BlockDiffusionCollator:
    canvas_length: int
    pad_token_id: int
    max_context_len: int
    single_block: bool = True
    generator: torch.Generator | None = None

    def _rand_block_start(self, n_resp):
        # one-block training: the canvas is the response's first 256 tokens
        # (context = prompt). Otherwise pick a random block.
        if self.single_block:
            return 0
        n_blocks = max(1, math.ceil(n_resp / self.canvas_length))
        k = int(torch.randint(0, n_blocks, (1,), generator=self.generator).item())
        return k * self.canvas_length

    def __call__(self, batch):
        L = self.canvas_length
        contexts, canvases, canvas_masks = [], [], []
        for ex in batch:
            p, r = ex["prompt_ids"], ex["response_ids"]
            start = self._rand_block_start(len(r))
            context = p + r[:start]
            context = context[-self.max_context_len:]
            block = r[start:start + L]
            cmask = [1] * len(block) + [0] * (L - len(block))
            block = block + [self.pad_token_id] * (L - len(block))
            contexts.append(context)
            canvases.append(block)
            canvas_masks.append(cmask)

        ctx_len = max(len(c) for c in contexts)
        input_ids, attn = [], []
        for c in contexts:
            pad = ctx_len - len(c)
            input_ids.append(c + [self.pad_token_id] * pad)   # right-pad
            attn.append([1] * len(c) + [0] * pad)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attn = torch.tensor(attn, dtype=torch.long)
        canvas = torch.tensor(canvases, dtype=torch.long)
        canvas_mask = torch.tensor(canvas_masks, dtype=torch.bool)

        # decoder positions: each canvas starts at that example's TRUE context len
        true_len = attn.sum(dim=1)  # (B,)
        dec_pos = (true_len[:, None] + torch.arange(L)[None, :]).to(torch.long)
        # decoder attention over [context_cache | canvas]; canvas always visible
        dec_attn = torch.cat([attn, torch.ones((attn.shape[0], L), dtype=torch.long)], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "canvas_input_ids": canvas,         # clean x0
            "canvas_loss_mask": canvas_mask,    # valid response positions
            "decoder_position_ids": dec_pos,
            "decoder_attention_mask": dec_attn,
        }


# --------------------------------------------------------------------------- #
# Core: corrupt the canvas (uniform noise) and compute the denoising loss.
# Standalone so it can be unit-tested without the full Trainer.
# --------------------------------------------------------------------------- #
def compute_diffusion_loss(model, inputs, *, vocab_size, eps_t=1e-3,
                           weight_by_t=False, self_cond_prob=0.0,
                           encoder_ar_loss_weight=0.0, pad_token_id=0,
                           router_aux_collector=None, router_aux_loss_coef=0.0,
                           return_outputs=False):
    x0 = inputs["canvas_input_ids"]
    cmask = inputs["canvas_loss_mask"]
    B, L = x0.shape
    dev = x0.device

    # forward (uniform) noise: replace each valid canvas token w/ prob t
    t = torch.empty(B, 1, device=dev).uniform_(eps_t, 1.0)
    corrupt = (torch.rand(B, L, device=dev) < t) & cmask
    rand_tok = torch.randint(0, vocab_size, (B, L), device=dev)
    x_t = torch.where(corrupt, rand_tok, x0)

    # The encoder builds its own causal mask via create_masks_for_generate when
    # attention_mask is None; passing a raw 2D mask there hits a shape bug. With
    # no padding (the common per_device_batch=1 case) None is equivalent, so use
    # it. (Multi-example batches with padding need length-packing instead.)
    # No padding (per_device_batch=1) -> pass attention_mask=None so the encoder
    # builds its own causal mask. (Padded multi-example batches need length
    # packing, not naive padding, due to the encoder's mask path.)
    enc_am = inputs["attention_mask"]
    enc_am = None if bool((enc_am == 1).all()) else enc_am
    fwd = dict(
        input_ids=inputs["input_ids"],
        attention_mask=enc_am,
        decoder_input_ids=x_t,
        decoder_position_ids=inputs["decoder_position_ids"],
        decoder_attention_mask=inputs["decoder_attention_mask"],
    )
    # multimodal: route image tensors to the encoder (vision tower + the
    # bidirectional image-token mask is driven by mm_token_type_ids).
    for k in ("pixel_values", "image_position_ids", "mm_token_type_ids"):
        if inputs.get(k) is not None:
            fwd[k] = inputs[k]
    if "pixel_values" in fwd:
        # match the vision tower weights; model may be a DeepSpeedEngine (no .dtype)
        vt_dtype = next(p for p in model.parameters() if p.dtype.is_floating_point).dtype
        fwd["pixel_values"] = fwd["pixel_values"].to(vt_dtype)

    # optional self-conditioning: a 1st no-grad pass feeds the 2nd
    sc_logits, sc_mask = None, None
    if self_cond_prob > 0:
        sc_mask = torch.rand(B, device=dev) < self_cond_prob
        if sc_mask.any():
            with torch.no_grad():
                sc_logits = model(**fwd).logits.detach()

    # clear any router captures from the no-grad self-conditioning pass so only
    # the loss-bearing forward's routing contributes to the aux loss.
    if router_aux_collector is not None:
        router_aux_collector.reset()
    out = model(self_conditioning_logits=sc_logits, self_conditioning_mask=sc_mask, **fwd)
    logits = out.logits.float()  # (B, L, V)

    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), x0.reshape(-1), reduction="none",
    ).view(B, L)
    ce = ce * cmask
    if weight_by_t:
        ce = ce / t
    loss = ce.sum() / cmask.sum().clamp(min=1)

    # optional AR next-token loss on the encoder context
    if encoder_ar_loss_weight > 0 and getattr(out, "encoder_last_hidden_state", None) is not None:
        enc_logits = model.get_output_embeddings()(out.encoder_last_hidden_state).float()
        ctx = inputs["input_ids"]
        ar = F.cross_entropy(
            enc_logits[:, :-1].reshape(-1, enc_logits.shape[-1]),
            ctx[:, 1:].reshape(-1), ignore_index=pad_token_id, reduction="mean",
        )
        loss = loss + encoder_ar_loss_weight * ar

    # MoE load-balancing auxiliary loss (keeps the router from collapsing onto a
    # few experts). Only meaningful when the router is trainable (full FT, or
    # MoE-LoRA with train_router=True). Logged via out for visibility.
    if router_aux_collector is not None and router_aux_loss_coef > 0:
        aux = router_aux_collector.aux_loss()
        if aux is not None:
            loss = loss + router_aux_loss_coef * aux

    return (loss, out) if return_outputs else loss


# --------------------------------------------------------------------------- #
# Trainer wrapper around the loss above.
# --------------------------------------------------------------------------- #
class DiffusionGemmaSFTTrainer(Trainer):
    def __init__(self, *a, vocab_size, eps_t, weight_by_t, self_cond_prob,
                 encoder_ar_loss_weight, pad_token_id, skip_move=False,
                 router_aux_collector=None, router_aux_loss_coef=0.0, **kw):
        self._skip_move = skip_move  # set before super().__init__ (it may move)
        super().__init__(*a, **kw)
        self.vocab_size = vocab_size
        self.eps_t = eps_t
        self.weight_by_t = weight_by_t
        self.self_cond_prob = self_cond_prob
        self.encoder_ar_loss_weight = encoder_ar_loss_weight
        self.pad_token_id = pad_token_id
        self.router_aux_collector = router_aux_collector
        self.router_aux_loss_coef = router_aux_loss_coef

    def _move_model_to_device(self, model, device):
        # When the model is sharded across GPUs (device_map), it's already
        # placed; moving it via .to() crashes on accelerate-dispatched tensors.
        if self._skip_move:
            return
        super()._move_model_to_device(model, device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return compute_diffusion_loss(
            model, inputs, vocab_size=self.vocab_size, eps_t=self.eps_t,
            weight_by_t=self.weight_by_t, self_cond_prob=self.self_cond_prob,
            encoder_ar_loss_weight=self.encoder_ar_loss_weight,
            pad_token_id=self.pad_token_id,
            router_aux_collector=self.router_aux_collector,
            router_aux_loss_coef=self.router_aux_loss_coef,
            return_outputs=return_outputs,
        )


# --------------------------------------------------------------------------- #
def main():
    parser = HfArgumentParser((ScriptArgs, TrainingArguments))
    sa, ta = parser.parse_args_into_dataclasses()
    set_seed(ta.seed)
    # our dataset yields {prompt_ids, response_ids}; the custom collator needs
    # them, so stop the Trainer from stripping "unused" columns.
    ta.remove_unused_columns = False
    ta.label_names = []  # loss is computed in compute_loss, not from "labels"

    if sa.smoke:
        ta.max_steps = 4
        ta.per_device_train_batch_size = max(1, ta.per_device_train_batch_size)
        ta.logging_steps = 1
        ta.report_to = []
        ta.save_strategy = "no"

    print("Loading processor + model ...", flush=True)
    processor = AutoProcessor.from_pretrained(sa.model_path)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # Placement (see ScriptArgs.device_map):
    #  - single GPU: load to CPU, Trainer moves to cuda:0. (device_map="auto" on
    #    a single GPU leaves meta tensors that crash Trainer._move_model_to_device.)
    #  - model-parallel: device_map="auto" shards across visible GPUs; the
    #    Trainer detects multi-device hf_device_map and skips the .to() move.
    # Placement priority:
    #  - DeepSpeed (--deepspeed cfg): the recommended multi-GPU path. With ZeRO-3
    #    the 52GB base is param-sharded across GPUs. `ta` is already parsed, so
    #    the global HfDeepSpeedConfig is live and from_pretrained loads under
    #    zero.Init (no full per-rank copy). Do NOT use device_map here.
    #  - device_map="auto": naive model-parallel fallback (no DeepSpeed).
    #  - "": single GPU (fits on an 80GB H100); Trainer moves it to cuda:0.
    using_deepspeed = bool(getattr(ta, "deepspeed", None))
    load_kw = dict(dtype=torch.bfloat16, attn_implementation=sa.attn_implementation)
    if sa.device_map and not using_deepspeed:
        load_kw["device_map"] = sa.device_map
        n = torch.cuda.device_count()
        # mp_cap_gib > 0: cap per-GPU load budget so the (50GB) model is FORCED to
        # split across GPUs. Without this, device_map="auto" packs the whole model
        # onto GPU0 (it fits in 80GB) -> 1 device -> Trainer falls back to
        # DataParallel (wrong) instead of model-parallel, and one GPU's activation
        # spike on the heavy 3-image examples OOMs. A low cap leaves the rest of
        # each GPU free for activations.
        if sa.mp_cap_gib > 0:
            per = sa.mp_cap_gib
        else:
            per = int(torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.92)
        load_kw["max_memory"] = {i: f"{per}GiB" for i in range(n)}
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(sa.model_path, **load_kw)
    model.config.use_cache = False
    # Text-only SFT: disable the vision bidirectional-attention mask (no image
    # tokens -> semantic no-op, and its block-mask builder crashes on text).
    # Multimodal SFT KEEPS it: image tokens must attend bidirectionally.
    if not sa.multimodal and getattr(model.config.text_config, "use_bidirectional_attention", None) == "vision":
        model.config.text_config.use_bidirectional_attention = None
    canvas_length = model.config.canvas_length
    vocab_size = model.config.text_config.vocab_size
    print(f"canvas_length={canvas_length}  vocab_size={vocab_size}  pad_id={pad_id}", flush=True)

    if sa.use_lora and sa.lora_mode == "moe":
        # Manual MoE-LoRA: adapts the 128 experts (the bulk of the params, where
        # the reasoning lives) which stock peft cannot reach, plus the decoder
        # attention. No peft. Saved separately via save_lora_state at the end.
        from moe_lora import apply_moe_and_decoder_lora
        # train the router whenever the load-balancing loss is on, else it's a no-op
        train_router = sa.train_router or sa.router_aux_loss_coef > 0
        apply_moe_and_decoder_lora(model, r=sa.lora_r, alpha=sa.lora_alpha,
                                   moe=True, decoder_attn=sa.moe_decoder_attn,
                                   train_router=train_router)
        # model-parallel (device_map="auto" shards the 50GB base across GPUs):
        # mark it so the Trainer uses model-parallel, not DataParallel (which
        # would replicate the model and double the per-step batch).
        base_dm = getattr(model, "hf_device_map", None)
        if base_dm and len(set(base_dm.values())) > 1:
            model.is_parallelizable = True
            model.model_parallel = True
    elif sa.use_lora:
        from peft import LoraConfig, get_peft_model
        # "all-linear" -> peft special string; a regex (contains regex chars) ->
        # pass as a string (peft re.fullmatch's it against module names);
        # otherwise a comma list of module-name suffixes.
        if sa.lora_target == "all-linear" or any(c in sa.lora_target for c in r".*()\|["):
            tgt = sa.lora_target
        else:
            tgt = [s for s in sa.lora_target.split(",") if s]
        lcfg = LoraConfig(
            r=sa.lora_r, lora_alpha=sa.lora_alpha, lora_dropout=sa.lora_dropout,
            target_modules=tgt, bias="none", task_type="FEATURE_EXTRACTION",
        )
        base_dm = getattr(model, "hf_device_map", None)
        model = get_peft_model(model, lcfg)
        # report encoder/decoder coverage so a mis-mount (e.g. enc-only) is obvious
        _b = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
        print(f"LoRA attached: {len(_b)}  enc={sum('encoder' in n for n in _b)} "
              f"dec={sum('decoder' in n for n in _b)}", flush=True)
        model.print_trainable_parameters()
        # peft hides hf_device_map; re-expose it so Trainer treats a sharded
        # (multi-GPU) model as model-parallel and skips the .to(device) move
        # (which crashes on accelerate-dispatched tensors).
        if base_dm and len(set(base_dm.values())) > 1:
            model.hf_device_map = base_dm
            model.is_parallelizable = True
            model.model_parallel = True

    # Resolve a data path relative to this file if needed.
    train_file = sa.train_file
    if not os.path.isabs(train_file):
        train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), train_file)
    if sa.multimodal:
        ds = MultimodalSFTDataset(train_file, processor, sa.image_path_from,
                                  sa.image_path_to, sa.max_context_len)
        collator = MultimodalCollator(canvas_length=canvas_length, pad_token_id=pad_id)
    else:
        ds = ChatSFTDataset(train_file, processor, sa.max_context_len)
        collator = BlockDiffusionCollator(
            canvas_length=canvas_length, pad_token_id=pad_id,
            max_context_len=sa.max_context_len, single_block=sa.single_block,
        )
    print(f"dataset: {len(ds)} examples ({'multimodal' if sa.multimodal else 'text'})", flush=True)

    # MoE load-balancing aux loss: hook the decoder routers (works for both moe-LoRA
    # and full FT; needs the router trainable, handled above for moe mode).
    router_aux_collector = None
    if sa.router_aux_loss_coef > 0:
        from moe_lora import RouterAuxCollector
        router_aux_collector = RouterAuxCollector(model)
        print(f"router aux loss ON: coef={sa.router_aux_loss_coef} "
              f"hooks={len(router_aux_collector.handles)}", flush=True)

    trainer = DiffusionGemmaSFTTrainer(
        model=model, args=ta, train_dataset=ds, data_collator=collator,
        vocab_size=vocab_size, eps_t=sa.eps_t, weight_by_t=sa.weight_by_t,
        self_cond_prob=sa.self_cond_prob, encoder_ar_loss_weight=sa.encoder_ar_loss_weight,
        pad_token_id=pad_id,
        skip_move=bool(sa.device_map and not using_deepspeed),
        router_aux_collector=router_aux_collector,
        router_aux_loss_coef=sa.router_aux_loss_coef,
    )
    trainer.train()
    if not sa.smoke:
        if sa.use_lora and sa.lora_mode == "moe":
            # save only the LoRA tensors (the 25B base is unchanged on disk)
            from moe_lora import save_lora_state
            os.makedirs(ta.output_dir, exist_ok=True)
            path = os.path.join(ta.output_dir, "moe_lora.pt")
            n = save_lora_state(trainer.model, path)
            print(f"saved {n} MoE-LoRA tensors to {path}", flush=True)
        else:
            trainer.save_model(ta.output_dir)
            print(f"saved to {ta.output_dir}", flush=True)


if __name__ == "__main__":
    main()
