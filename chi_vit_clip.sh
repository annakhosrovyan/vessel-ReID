#!/bin/bash

#SBATCH --job-name=CViT-CLIP
#SBATCH --partition=all
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=5-00:00:00


python -m torch.distributed.launch --nproc_per_node=8 --master_port=6203 \
    train_pair.py \
    --config_file configs/pretrain_chi_vit_clip.yml \
    OUTPUT_DIR /data/ship/vessel_re_id/checkpoints/pretrain_chi_vit_clip \
    MODEL.DIST_TRAIN True \
    SOLVER.USE_MULTI_PRETRAIN True \
    SOLVER.IMS_PER_BATCH 1024 \
    SOLVER.GRAD_ACCUM_STEPS 1 \
    SOLVER.BASE_LR 5e-4 \
    SOLVER.MAX_EPOCHS 200 \
    SOLVER.CHECKPOINT_PERIOD -1 \
    SOLVER.SCHEDULER_TYPE wsd \
    SOLVER.WSD_DECAY_PCT 0.1 \
    SOLVER.WSD_DECAY_TARGETS "(20, 40, 80, 100, 120, 160)" \
    MODEL.METRIC_LOSS_TYPE clip \
    MODEL.USE_AMP True \
    SOLVER.DETERMINISTIC False \
    DATALOADER.NUM_WORKERS 8
