#!/bin/bash

#SBATCH --job-name=Decay-CViT-CLIP
#SBATCH --partition=all
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=1-00:00:00

# === SET THIS to the dated run directory containing the checkpoints ===
export LC_TIME=C
CKPT_DIR="/data/ship/vessel_re_id/checkpoints/pretrain_chi_vit_clip/Feb_2026/Feb26_09-34-22"

DECAY_PCT=0.1
TARGETS=(40 80 100 120 160)
RUN_NAME=$(date +"%b_%Y/%b%d_%H-%M-%S")

for T in "${TARGETS[@]}"; do
    CKPT_EPOCH=$(python3 -c "import math; print(int(math.floor((1.0 - ${DECAY_PCT}) * ${T})))")
    CKPT_PATH="${CKPT_DIR}/transformer_checkpoint_${CKPT_EPOCH}.pth"

    if [ ! -f "$CKPT_PATH" ]; then
        echo "Checkpoint not found: $CKPT_PATH — skipping T=${T}"
        continue
    fi

    echo "=== Decay phase for T=${T}: resuming from epoch ${CKPT_EPOCH} ==="

    python -m torch.distributed.launch --nproc_per_node=8 --master_port=6203 \
        train_pair.py \
        --config_file configs/pretrain_chi_vit_clip.yml \
        --log_dir_name "$RUN_NAME" \
        OUTPUT_DIR "/data/ship/vessel_re_id/checkpoints/pretrain_chi_vit_clip_decay_${T}" \
        SOLVER.RESUME_FROM "$CKPT_PATH" \
        MODEL.DIST_TRAIN True \
        SOLVER.USE_MULTI_PRETRAIN True \
        SOLVER.IMS_PER_BATCH 1024 \
        SOLVER.GRAD_ACCUM_STEPS 1 \
        SOLVER.BASE_LR 5e-4 \
        SOLVER.MAX_EPOCHS "$T" \
        SOLVER.CHECKPOINT_PERIOD -1 \
        SOLVER.SCHEDULER_TYPE wsd \
        SOLVER.WSD_DECAY_PCT "$DECAY_PCT" \
        MODEL.METRIC_LOSS_TYPE clip \
        MODEL.USE_AMP True \
        SOLVER.DETERMINISTIC False \
        DATALOADER.NUM_WORKERS 8

    echo "=== Done T=${T} ==="
done
