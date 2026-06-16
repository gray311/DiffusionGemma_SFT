#!/usr/bin/env bash
# Multi-GPU LoRA SFT of DiffusionGemma-26B-A4B-it with DeepSpeed ZeRO-3.
# ZeRO-3 param-shards the 52GB base across the GPUs, so it fits on e.g. 4xL40S
# (46GB) or scales across multiple H100s.
#   conda activate dgemma
set -e
cd "$(dirname "$0")"

export HF_HOME=/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=${NUM_GPUS:-4}

deepspeed --num_gpus=${NUM_GPUS} train_diffusiongemma_sft.py \
  --deepspeed ds_config_zero3.json \
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
