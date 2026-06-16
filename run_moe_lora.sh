#!/bin/bash
# MoE-LoRA SFT, aligned with the official TRL recipe (hyperparameters below),
# keeping our two departures: MoE-expert LoRA mounting + router aux loss.
set -e
cd /weka/home/ext-yingzima/DiffusionGemma_SFT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dgemma

DMAP=${DMAP:-}
if [ "$DMAP" = "auto" ]; then
  export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
else
  export CUDA_VISIBLE_DEVICES=${GPU:-0}
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/dgemma_moe_lora}

python train_diffusiongemma_sft.py \
  --train_file ${TRAIN_FILE:-/weka/home/ext-yingzima/NVIDIA_Internship/data/test.json} \
  --data_format ${DATA_FORMAT:-qa} --multimodal True \
  --image_path_to ${IMGBASE:-/weka/home/ext-yingzima/} \
  --max_length ${MAXLEN:-1024} \
  --use_lora True --lora_mode moe --lora_r ${LORAR:-16} --lora_alpha ${LORALPHA:-32} \
  --lora_linears ${LINEARS:-official} \
  --max_examples ${MAXEX:-0} --max_images ${MAXIMG:-0} \
  --router_aux_loss_coef ${AUXCOEF:-0} \
  --self_cond_prob ${SELFCOND:-0.5} --ar_loss ${ARLOSS:-True} --ar_loss_weight ${ARW:-1.0} \
  --device_map "$DMAP" --mp_cap_gib ${MPCAP:-0} --attn_implementation ${ATTN:-sdpa} \
  --output_dir "$OUT" \
  --num_train_epochs ${EPOCHS:-3} \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps ${ACCUM:-8} \
  --learning_rate ${LR:-1.5e-4} \
  --adam_beta1 0.95 --adam_beta2 0.99 --weight_decay 1e-4 \
  --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1.5e-5}' \
  --warmup_steps 25 \
  --bf16 True --gradient_checkpointing False \
  --optim ${OPTIM:-paged_adamw_8bit} \
  --logging_steps 10 --save_strategy no --report_to none \
  --dataloader_num_workers 2 \
  "$@"
