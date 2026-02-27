#!/bin/bash

#SBATCH --job-name=CViT-iBOT
#SBATCH --partition=all
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=5-00:00:00


python -m torch.distributed.launch --nproc_per_node=8 --master_port=6204 \
    train_pair.py \
    --config_file configs/pretrain_chi_vit_ibot.yml \
    OUTPUT_DIR /data/ship/vessel_re_id/checkpoints/pretrain_chi_vit_ibot \
    MODEL.DIST_TRAIN True \
    SOLVER.RESUME_FROM "/data/ship/vessel_re_id/checkpoints/pretrain_chi_vit_ibot/Feb_2026/Feb26_13-30-02/transformer_checkpoint_144.pth" \
    SOLVER.USE_MULTI_PRETRAIN True \
    SOLVER.IMS_PER_BATCH 512 \
    SOLVER.GRAD_ACCUM_STEPS 2 \
    SOLVER.BASE_LR 5e-4 \
    SOLVER.MAX_EPOCHS 200 \
    SOLVER.CHECKPOINT_PERIOD -1 \
    SOLVER.SCHEDULER_TYPE wsd \
    SOLVER.WSD_DECAY_PCT 0.1 \
    SOLVER.WSD_DECAY_TARGETS "(20, 40, 80, 100, 120, 160)" \
    MODEL.METRIC_LOSS_TYPE ibot \
    MODEL.USE_AMP True \
    SOLVER.DETERMINISTIC False \
    DATALOADER.NUM_WORKERS 8 \
    IBOT.MODALITY_PURE_SAMPLING True

