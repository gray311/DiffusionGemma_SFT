#!/usr/bin/env bash
# LoRA SFT of DiffusionGemma-26B-A4B-it on a single H100.
# Activate the env that has transformers>=5.12 + the model:
#   conda activate dgemma
set -e
cd "$(dirname "$0")"

export HF_HOME=/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python train_diffusiongemma_sft.py \
  --model_path /weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/diffusiongemma-26B-A4B-it \
  --train_file data/example_sft.jsonl \
  --output_dir /weka/home/ext-yingzima/scratchaszalay1_ssci/yy/dgemma_sft_out \
  --use_lora True --lora_r 16 --lora_alpha 32 \
  --max_context_len 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --bf16 True \
  --logging_steps 1 \
  --save_strategy epoch \
  --report_to none
