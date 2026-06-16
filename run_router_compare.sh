#!/bin/bash
# Wait for the aux-loss training to finish, then compare decoder-router expert
# distributions: no-aux ckpt vs with-aux ckpt, on 100 test examples, and plot.
cd /weka/home/ext-yingzima/DiffusionGemma_SFT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dgemma
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

NOAUX=/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/dgemma_moe_lora
WITHAUX=/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/dgemma_moe_lora_aux

# 1. wait until aux training has saved its checkpoint
echo "waiting for aux training to finish ..."
until grep -qa "saved .* MoE-LoRA tensors" logs/train_aux.log 2>/dev/null; do
  if ! pgrep -f "train_diffusiongemma_sft.py" >/dev/null && ! grep -qa "saved .* MoE-LoRA tensors" logs/train_aux.log 2>/dev/null; then
    echo "training process gone without saving — check logs/train_aux.log"; exit 1
  fi
  sleep 60
done
echo "aux training done; final losses:"
tr '\r' '\n' < logs/train_aux.log | grep -aoE "'loss': '[0-9.]+'.*'epoch': '[0-9.]+'" | tail -3

# 2. analyze both checkpoints on GPU 1 (now free)
echo "=== analyzing no-aux ckpt ==="
CUDA_VISIBLE_DEVICES=1 python router_analysis.py analyze $NOAUX   /tmp/router_noaux.json 100 0
echo "=== analyzing with-aux ckpt ==="
CUDA_VISIBLE_DEVICES=1 python router_analysis.py analyze $WITHAUX /tmp/router_withaux.json 100 0

# 3. plot the two distributions
python router_analysis.py plot /tmp/router_noaux.json:no-aux /tmp/router_withaux.json:with-aux \
  /weka/home/ext-yingzima/DiffusionGemma_SFT/router_dist.png
echo "ALL DONE -> router_dist.png"
