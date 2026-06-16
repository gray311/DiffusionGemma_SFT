#!/bin/bash
# Detach the training into its own session so it survives the Claude Bash-tool
# shell terminating (the tool kills its process group on completion/timeout).
cd /weka/home/ext-yingzima/DiffusionGemma_SFT
export GPU=${GPU:-0}
export OUT=${OUT:-/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/dgemma_moe_lora}
export EPOCHS=${EPOCHS:-5}
export ACCUM=${ACCUM:-4}
export LR=${LR:-2e-4}
export DMAP=${DMAP:-}
export GPUS=${GPUS:-0,1}
export MPCAP=${MPCAP:-0}
export OPTIM=${OPTIM:-paged_adamw_8bit}
export AUXCOEF=${AUXCOEF:-0}
export PYTHONUNBUFFERED=1
LOG=${LOG:-logs/train_moe_lora.log}

setsid bash run_moe_lora.sh > "$LOG" 2>&1 < /dev/null &
CHILD=$!
disown
# give setsid time to fork into a fresh session before this shell (and its pgid) dies
sleep 2
echo "detached training: child pid $CHILD, log $LOG"
exit 0
