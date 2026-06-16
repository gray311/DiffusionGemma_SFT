"""SFT for google/diffusiongemma-26B-A4B-it — aligned with the official TRL
recipe (trl examples/scripts/sft_diffusion_gemma.py, PR #6003), with two
deliberate departures we keep:

  1. LoRA mounting: we adapt the MoE EXPERTS (+ decoder attention) via moe_lora.py
     instead of the official "attention + dense-MLP, experts frozen" recipe,
     because we specifically want to fine-tune the experts.
  2. MoE load-balancing (router aux) loss: kept on (--router_aux_loss_coef). The
     official code will add this upstream later.

Everything else mirrors the official block-diffusion objective:
  * One full sequence per example via the TRAINING chat template
    (chat_templates/diffusion_gemma_training.jinja); the supervised span is the
    final assistant turn (content + closing `<turn|>`), `labels=-100` elsewhere.
  * compute_loss derives the response span from `labels`, selects ONE response
    block at random (so answers longer than canvas_length are covered over
    training), encodes the full clean sequence, and the decoder mask cuts the KV
    cache off at the prompt + clean blocks BEFORE the selected one.
  * Clean canvas = the block, EOS-filled past the response end; flat CE over the
    whole canvas (no 1/t weighting). Plus an autoregressive co-loss on the
    encoder (with final_logit_softcapping), and self-conditioning at p=0.5.

Smoke test: python train_diffusiongemma_sft.py --smoke
Real run: see run_moe_lora.sh
"""
from __future__ import annotations

import json
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

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_TEMPLATE_PATH = os.path.join(HERE, "chat_templates", "diffusion_gemma_training.jinja")


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
@dataclass
class ScriptArgs:
    model_path: str = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it"
    train_file: str = "data/example_sft.jsonl"
    # data_format: "qa" -> {"image":[...], "question":"...<image>...", "answer":"..."}
    #              "conversations" -> {"image":[...], "conversations":[{from,value}...]}
    data_format: str = "qa"
    multimodal: bool = False
    # image path resolution: if image_path_from is non-empty and present in a path,
    # replace it with image_path_to; otherwise a RELATIVE path is joined onto
    # image_path_to (treated as a base dir). Absolute paths pass through.
    image_path_from: str = "/weka/home/xliu316/"
    image_path_to: str = "/weka/home/ext-yingzima/"
    max_examples: int = 0             # 0 = all; else use the first N examples
    max_images: int = 0               # 0 = all; else cap images/example (memory)
    max_length: int = 1024            # full-sequence cap (official default)
    eps_t: float = 1e-3               # min corruption ratio
    self_cond_prob: float = 0.5       # self-conditioning probability (official: 0.5)
    ar_loss: bool = True              # encoder autoregressive co-loss (official: on)
    # LoRA (our MoE-expert mounting, kept)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0         # official lora_dropout = 0.0
    lora_mode: str = "moe"            # "moe" -> moe_lora.py ; "peft" -> stock peft
    # which nn.Linears to LoRA in moe mode: "official" -> enc/dec attention +
    # dense MLP (so the AR co-loss can train the encoder, like the official
    # recipe) ; "decoder_attn" -> decoder attention only ; "none" -> experts only.
    lora_linears: str = "official"
    lora_target: str = "all-linear"   # only used when lora_mode == "peft"
    # MoE load-balancing aux loss (kept). 0 = off. >0 auto-unfreezes the router.
    router_aux_loss_coef: float = 0.0
    train_router: bool = False
    # placement
    device_map: str = ""
    mp_cap_gib: int = 0
    attn_implementation: str = "sdpa"
    smoke: bool = False


# --------------------------------------------------------------------------- #
# Dataset: one full sequence per example via the TRAINING chat template.
# Returns input_ids / attention_mask / labels (+ multimodal tensors). The block
# selection happens later in compute_loss (mirrors the official recipe).
# --------------------------------------------------------------------------- #
def _resolve_image(p, path_from, path_to):
    if path_from and path_from in p:
        return p.replace(path_from, path_to)
    if not os.path.isabs(p):
        return os.path.join(path_to, p)   # path_to as a base dir for relative paths
    return p


def _to_messages(ex, data_format, path_from, path_to, max_images=0):
    """Return (messages, image_paths). Last message is the assistant response."""
    fix = lambda p: _resolve_image(p, path_from, path_to)
    cap = lambda lst: lst[:max_images] if max_images > 0 else lst
    if data_format == "qa":
        imgs = cap([fix(p) for p in ex.get("image", []) or []])
        qtext = ex["question"].replace("<image>", "").strip()
        answer = ex["answer"]
    elif data_format == "conversations":
        imgs = [fix(p) for p in ex.get("image", []) or []]
        conv = ex["conversations"]
        # use the prompt up to (and including) the final assistant/gpt turn
        last_gpt = max(i for i, m in enumerate(conv) if m["from"] in ("gpt", "assistant"))
        user_text = "\n".join(m["value"] for m in conv[:last_gpt] if m["from"] in ("human", "user"))
        user_text = user_text.replace("<image>", "").strip()
        answer = conv[last_gpt]["value"]
        qtext = user_text
    else:
        raise ValueError(f"unknown data_format {data_format}")

    if imgs:
        content = [{"type": "image", "image": p} for p in imgs] + [{"type": "text", "text": qtext}]
    else:
        content = qtext
    messages = [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]
    return messages, imgs


class SFTDataset(Dataset):
    def __init__(self, path, processor, sa: ScriptArgs):
        self.data = json.load(open(path)) if path.endswith(".json") else \
            [json.loads(l) for l in open(path) if l.strip()]
        if sa.max_examples > 0:
            self.data = self.data[: sa.max_examples]
        self.processor = processor
        self.tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.sa = sa
        self.train_template = open(TRAIN_TEMPLATE_PATH).read()
        self.eot_id = self.tok.convert_tokens_to_ids("<end_of_turn>")

    def __len__(self):
        return len(self.data)

    def _n_supervised(self, messages_textonly):
        """Token count of the supervised suffix (final assistant content + <turn|>),
        from a text-only render (images count as 1 literal token in the user turn,
        so the assistant suffix length is unaffected by image expansion)."""
        out = self.tok.apply_chat_template(
            messages_textonly, chat_template=self.train_template, tokenize=True,
            return_assistant_tokens_mask=True, return_dict=True,
        )
        return int(sum(out["assistant_masks"]))

    def __getitem__(self, i):
        sa = self.sa
        ex = self.data[i]
        messages, img_paths = _to_messages(ex, sa.data_format, sa.image_path_from, sa.image_path_to)

        # text-only copy (replace image dicts with a placeholder string) for the
        # supervised-suffix count
        msgs_txt = []
        for m in messages:
            c = m["content"]
            if isinstance(c, list):
                c = "".join("<image>" if it.get("type") == "image" else it.get("text", "") for it in c)
            msgs_txt.append({"role": m["role"], "content": c})
        n_sup = self._n_supervised(msgs_txt)

        if img_paths:
            from PIL import Image
            imgs = [Image.open(p).convert("RGB") for p in img_paths]
            content = [{"type": "image", "image": im} for im in imgs] + \
                      [{"type": "text", "text": messages[0]["content"][-1]["text"]}]
            full_msgs = [{"role": "user", "content": content},
                         {"role": "assistant", "content": messages[1]["content"]}]
            proc = self.processor.apply_chat_template(
                full_msgs, chat_template=self.train_template, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            per_image = {"pixel_values", "image_position_ids"}
            item = {k: (v if k in per_image else v[0]) for k, v in proc.items()}
            input_ids = item["input_ids"]
        else:
            ids = self.tok.apply_chat_template(
                messages, chat_template=self.train_template, tokenize=True, return_tensors="pt",
            )[0]
            item = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            input_ids = ids

        # labels: supervise only the final assistant span (the suffix of length n_sup)
        seq_len = input_ids.shape[0]
        labels = torch.full((seq_len,), -100, dtype=torch.long)
        if n_sup > 0:
            labels[-n_sup:] = input_ids[-n_sup:]
        item["labels"] = labels

        # right-truncate to max_length if needed (keeps prompt+images+early answer)
        if seq_len > sa.max_length:
            for k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
                if k in item and torch.is_tensor(item[k]) and item[k].shape[0] == seq_len:
                    item[k] = item[k][: sa.max_length]
        return item


@dataclass
class DiffusionCollator:
    pad_token_id: int

    def __call__(self, batch):
        maxlen = max(b["input_ids"].shape[0] for b in batch)
        input_ids, attn, labels = [], [], []
        mm = {"pixel_values": [], "image_position_ids": [], "mm_token_type_ids": []}
        has_mm = "pixel_values" in batch[0]
        for b in batch:
            n = b["input_ids"].shape[0]
            pad = maxlen - n
            input_ids.append(F.pad(b["input_ids"], (0, pad), value=self.pad_token_id))
            attn.append(F.pad(b["attention_mask"], (0, pad), value=0))
            labels.append(F.pad(b["labels"], (0, pad), value=-100))
            if has_mm:
                mm["pixel_values"].append(b["pixel_values"])
                mm["image_position_ids"].append(b["image_position_ids"])
                mm["mm_token_type_ids"].append(F.pad(b["mm_token_type_ids"], (0, pad), value=0))
        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }
        if has_mm:
            # per-example images differ in count; batch=1 is the supported path.
            out["pixel_values"] = torch.cat(mm["pixel_values"], dim=0)
            out["image_position_ids"] = torch.cat(mm["image_position_ids"], dim=0)
            out["mm_token_type_ids"] = torch.stack(mm["mm_token_type_ids"])
        return out


# --------------------------------------------------------------------------- #
# Block-diffusion loss (official) + MoE router aux loss (ours).
# --------------------------------------------------------------------------- #
def compute_diffusion_loss(model, inputs, *, vocab_size, canvas_length, eps_t,
                           self_cond_prob, ar_loss, final_logit_softcapping,
                           eos_token_id, router_aux_collector=None,
                           router_aux_loss_coef=0.0, return_outputs=False):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    block_size = canvas_length

    # response span from labels (the final supervised assistant turn)
    supervised = labels != -100
    positions = torch.arange(seq_len, device=device)
    span_starts = supervised & ~F.pad(supervised, (1, 0))[:, :-1]
    span_end = torch.where(supervised, positions, torch.full_like(positions, -1)).amax(dim=1)
    prefix_len = torch.where(span_starts, positions, torch.full_like(positions, -1)).amax(dim=1).clamp(min=0)
    response_len = (span_end - prefix_len + 1) * (span_end >= 0)

    # select one response block; encoder reads the full clean sequence, the decoder
    # may only see the prompt + clean response blocks BEFORE the selected one.
    num_blocks = (response_len - 1).clamp(min=0) // block_size + 1
    block_idx = (torch.rand(batch_size, device=device) * num_blocks).long()
    encoder_len = prefix_len + block_idx * block_size

    # clean canvas: the block, EOS-filled past the end of the response (supervised)
    offsets = torch.arange(block_size, device=device)
    abs_idx = (encoder_len[:, None] + offsets).clamp(max=seq_len - 1)
    in_response = offsets < (response_len - block_idx * block_size)[:, None]
    canvas_target = torch.where(in_response, input_ids.gather(1, abs_idx), torch.tensor(eos_token_id, device=device))

    # uniform random-token corruption (no mask token), per-example t ~ U(eps, 1)
    t = eps_t + (1 - eps_t) * torch.rand(batch_size, 1, device=device)
    corrupt = torch.rand(batch_size, block_size, device=device) < t
    random_tokens = torch.randint(vocab_size, (batch_size, block_size), device=device)
    canvas_ids = torch.where(corrupt, random_tokens, canvas_target)

    cache_mask = (torch.arange(seq_len, device=device) < encoder_len[:, None]).long()
    canvas_mask = torch.ones(batch_size, block_size, dtype=torch.long, device=device)
    mk = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": canvas_ids,
        "decoder_attention_mask": torch.cat([cache_mask, canvas_mask], dim=1),
        "decoder_position_ids": encoder_len[:, None] + offsets,
    }
    for k in ("pixel_values", "image_position_ids", "mm_token_type_ids"):
        if inputs.get(k) is not None:
            mk[k] = inputs[k]
    if "pixel_values" in mk:
        vt_dtype = next(p for p in model.parameters() if p.dtype.is_floating_point).dtype
        mk["pixel_values"] = mk["pixel_values"].to(vt_dtype)

    # two-pass self-conditioning, gated per example
    if self_cond_prob > 0:
        with torch.no_grad():
            mk["self_conditioning_logits"] = model(**mk).logits
        mk["self_conditioning_mask"] = torch.rand(batch_size, device=device) < self_cond_prob

    # only the loss-bearing forward's routing should feed the aux loss
    if router_aux_collector is not None:
        router_aux_collector.reset()
    outputs = model(**mk)

    # flat CE over the whole canvas (corrupted and clean alike); no 1/t weighting
    diffusion_loss = F.cross_entropy(outputs.logits.flatten(0, 1).float(), canvas_target.flatten())
    loss = diffusion_loss

    # autoregressive co-loss on the encoder (text positions only). Gather the
    # valid positions BEFORE the lm_head so we never materialize a (seq x vocab)
    # logits tensor — over a long 4-image sequence that float32 tensor alone is
    # ~2GB and OOMs. Numerically identical to project-then-mask.
    if ar_loss and getattr(outputs, "encoder_last_hidden_state", None) is not None:
        head = model.get_output_embeddings()
        ar_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        if inputs.get("mm_token_type_ids") is not None:
            ar_mask = ar_mask & (inputs["mm_token_type_ids"][:, 1:] == 0)  # skip image tokens
        if ar_mask.any():
            hidden = outputs.encoder_last_hidden_state[:, :-1][ar_mask]    # (N, H), N << seq
            targets = input_ids[:, 1:][ar_mask]                           # (N,)
            enc_logits = head(hidden.to(head.weight.dtype)).float()        # (N, vocab)
            if final_logit_softcapping:
                enc_logits = torch.tanh(enc_logits / final_logit_softcapping) * final_logit_softcapping
            ar = F.cross_entropy(enc_logits, targets)
            loss = loss + ar

    # MoE load-balancing aux loss (ours; only meaningful with a trainable router)
    if router_aux_collector is not None and router_aux_loss_coef > 0:
        aux = router_aux_collector.aux_loss()
        if aux is not None:
            loss = loss + router_aux_loss_coef * aux

    return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
class DiffusionGemmaSFTTrainer(Trainer):
    def __init__(self, *a, vocab_size, canvas_length, eps_t, self_cond_prob, ar_loss,
                 final_logit_softcapping, eos_token_id, skip_move=False,
                 router_aux_collector=None, router_aux_loss_coef=0.0, **kw):
        self._skip_move = skip_move
        super().__init__(*a, **kw)
        self.vocab_size = vocab_size
        self.canvas_length = canvas_length
        self.eps_t = eps_t
        self.self_cond_prob = self_cond_prob
        self.ar_loss = ar_loss
        self.final_logit_softcapping = final_logit_softcapping
        self.eos_token_id = eos_token_id
        self.router_aux_collector = router_aux_collector
        self.router_aux_loss_coef = router_aux_loss_coef

    def _move_model_to_device(self, model, device):
        if self._skip_move:
            return
        super()._move_model_to_device(model, device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return compute_diffusion_loss(
            model, inputs, vocab_size=self.vocab_size, canvas_length=self.canvas_length,
            eps_t=self.eps_t, self_cond_prob=self.self_cond_prob, ar_loss=self.ar_loss,
            final_logit_softcapping=self.final_logit_softcapping, eos_token_id=self.eos_token_id,
            router_aux_collector=self.router_aux_collector,
            router_aux_loss_coef=self.router_aux_loss_coef, return_outputs=return_outputs,
        )


# --------------------------------------------------------------------------- #
def main():
    parser = HfArgumentParser((ScriptArgs, TrainingArguments))
    sa, ta = parser.parse_args_into_dataclasses()
    set_seed(ta.seed)
    ta.remove_unused_columns = False
    ta.label_names = []

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

    using_deepspeed = bool(getattr(ta, "deepspeed", None))
    load_kw = dict(dtype=torch.bfloat16, attn_implementation=sa.attn_implementation)
    if sa.device_map and not using_deepspeed:
        load_kw["device_map"] = sa.device_map
        n = torch.cuda.device_count()
        per = sa.mp_cap_gib if sa.mp_cap_gib > 0 else int(torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.92)
        load_kw["max_memory"] = {i: f"{per}GiB" for i in range(n)}
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(sa.model_path, **load_kw)
    model.config.use_cache = False
    canvas_length = model.config.canvas_length
    vocab_size = model.config.text_config.vocab_size
    softcap = model.config.text_config.final_logit_softcapping
    print(f"canvas_length={canvas_length}  vocab_size={vocab_size}  pad_id={pad_id}  softcap={softcap}", flush=True)

    # ---- LoRA (our MoE-expert mounting; the deliberate departure we keep) ----
    if sa.use_lora and sa.lora_mode == "moe":
        from moe_lora import apply_moe_and_decoder_lora
        train_router = sa.train_router or sa.router_aux_loss_coef > 0
        apply_moe_and_decoder_lora(model, r=sa.lora_r, alpha=sa.lora_alpha,
                                   moe=True, linears=sa.lora_linears,
                                   train_router=train_router)
        base_dm = getattr(model, "hf_device_map", None)
        if base_dm and len(set(base_dm.values())) > 1:
            model.is_parallelizable = True
            model.model_parallel = True
    elif sa.use_lora:
        from peft import LoraConfig, get_peft_model
        tgt = sa.lora_target if (sa.lora_target == "all-linear" or
                                 any(c in sa.lora_target for c in r".*()\|[")) else \
            [s for s in sa.lora_target.split(",") if s]
        lcfg = LoraConfig(r=sa.lora_r, lora_alpha=sa.lora_alpha, lora_dropout=sa.lora_dropout,
                          target_modules=tgt, bias="none", task_type="FEATURE_EXTRACTION")
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    # ---- data ----
    train_file = sa.train_file
    if not os.path.isabs(train_file):
        train_file = os.path.join(HERE, train_file)
    ds = SFTDataset(train_file, processor, sa)
    collator = DiffusionCollator(pad_token_id=pad_id)
    print(f"dataset: {len(ds)} examples  format={sa.data_format}  multimodal={sa.multimodal}", flush=True)

    # ---- MoE load-balancing aux loss (ours; kept) ----
    router_aux_collector = None
    if sa.router_aux_loss_coef > 0:
        from moe_lora import RouterAuxCollector
        router_aux_collector = RouterAuxCollector(model)
        print(f"router aux loss ON: coef={sa.router_aux_loss_coef} hooks={len(router_aux_collector.handles)}", flush=True)

    trainer = DiffusionGemmaSFTTrainer(
        model=model, args=ta, train_dataset=ds, data_collator=collator,
        vocab_size=vocab_size, canvas_length=canvas_length, eps_t=sa.eps_t,
        self_cond_prob=sa.self_cond_prob, ar_loss=sa.ar_loss,
        final_logit_softcapping=softcap, eos_token_id=tok.eos_token_id,
        skip_move=bool(sa.device_map and not using_deepspeed),
        router_aux_collector=router_aux_collector,
        router_aux_loss_coef=sa.router_aux_loss_coef,
    )
    trainer.train()
    if not sa.smoke:
        if sa.use_lora and sa.lora_mode == "moe":
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
