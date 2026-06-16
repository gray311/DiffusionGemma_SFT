#!/bin/bash
# MoE-LoRA SFT on the 1000-example multimodal driving set, single A100 (~70GB).
set -e
cd /weka/home/ext-yingzima/DiffusionGemma_SFT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dgemma

# DMAP="auto" -> model-parallel across CUDA_VISIBLE_DEVICES (needs the heavy
# 3-image/long-context examples' activations to exceed one 80GB GPU). DMAP=""
# -> single GPU. When model-parallel, expose both GPUs.
DMAP=${DMAP:-}
if [ "$DMAP" = "auto" ]; then
  export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
else
  export CUDA_VISIBLE_DEVICES=${GPU:-0}
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/dgemma_moe_lora}

python train_diffusiongemma_sft.py \
  --train_file /weka/home/ext-yingzima/NVIDIA_Internship/data/test.json \
  --multimodal True \
  --use_lora True --lora_mode moe --lora_r 16 --lora_alpha 32 --moe_decoder_attn True \
  --router_aux_loss_coef ${AUXCOEF:-0} \
  --device_map "$DMAP" --mp_cap_gib ${MPCAP:-0} --attn_implementation ${ATTN:-sdpa} \
  --output_dir "$OUT" \
  --num_train_epochs ${EPOCHS:-5} \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps ${ACCUM:-4} \
  --learning_rate ${LR:-2e-4} \
  --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --bf16 True --gradient_checkpointing False \
  --optim ${OPTIM:-paged_adamw_8bit} \
  --logging_steps 10 --save_strategy no --report_to none \
  --dataloader_num_workers 2 \
  "$@"
