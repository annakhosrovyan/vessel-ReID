import torch
import wandb
import logging
import numpy as np
from utils.metrics import R1_mAP_eval


def compute_pair_distance_stats(distmat: np.ndarray, 
                                q_pids: np.ndarray, 
                                g_pids: np.ndarray, 
                                q_camids: np.ndarray, 
                                g_camids: np.ndarray
                                ) -> dict:

    rgb, sar = 0, 1
    same_pid = q_pids[:, None] == g_pids[None, :]
    diff_pid = q_pids[:, None] != g_pids[None, :]

    q_is_rgb = q_camids[:, None] == rgb
    q_is_sar = q_camids[:, None] == sar
    g_is_rgb = g_camids[None, :] == rgb
    g_is_sar = g_camids[None, :] == sar

    masks = {
        "positive/rgb_rgb": same_pid & q_is_rgb & g_is_rgb,
        "positive/sar_sar": same_pid & q_is_sar & g_is_sar,
        "positive/sar_rgb": same_pid & q_is_sar & g_is_rgb,
        "positive/rgb_sar": same_pid & q_is_rgb & g_is_sar,
        "positive/all": same_pid,
        "negative/rgb_rgb": diff_pid & q_is_rgb & g_is_rgb,
        "negative/sar_sar": diff_pid & q_is_sar & g_is_sar,
        "negative/sar_rgb": diff_pid & q_is_sar & g_is_rgb,
        "negative/rgb_sar": diff_pid & q_is_rgb & g_is_sar,
        "negative/all": diff_pid,
    }

    stats = {}
    for k, m in masks.items():
        vals = distmat[m]
        stats[f"{k}_count"] = int(vals.size)
        if vals.size == 0:
            stats[f"{k}_mean"] = float("nan")
            stats[f"{k}_max"] = float("nan")
            stats[f"{k}_min"] = float("nan")
            stats[f"{k}_std"] = float("nan")
            stats[f"{k}_median"] = float("nan")
        else:
            stats[f"{k}_mean"] = float(np.mean(vals))
            stats[f"{k}_max"] = float(np.max(vals))
            stats[f"{k}_min"] = float(np.min(vals))
            stats[f"{k}_std"] = float(np.std(vals))
            stats[f"{k}_median"] = float(np.median(vals))
    return stats


class ValidationMetricsTracker:
    def __init__(self, cfg, local_rank):
        self.logger = logging.getLogger("Tracking Validation Metrics")
        self.cfg = cfg
        self.local_rank = local_rank

    def run(self, model, epoch, val_loader=None, num_query_val=0):
        if val_loader is None or self.local_rank != 0:
            return 0.0

        self.logger.info(f"Epoch {epoch}    tracking validation metrics...")
        evaluator = R1_mAP_eval(num_query_val, max_rank=50, feat_norm=self.cfg.TEST.FEAT_NORM)
        evaluator.reset()
        model.eval()

        for _, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
            with torch.no_grad():
                img = img.to("cuda")
                camids = camids.to("cuda")
                img_wh = img_wh.to("cuda")

                feat = model(img, cam_label=camids, img_wh=img_wh)
                evaluator.update((feat, vid, camid))

        cmc, mAP, distmat, pids, camids, _, _ = evaluator.compute()
        self.log_metrics(epoch, mAP, cmc)

        if self.cfg.SOLVER.TRACK_VALIDATION_METRICS:
            self.log_distance_stats(distmat, pids, camids, evaluator.num_query)
            
        return mAP

    def log_metrics(self, epoch, mAP, cmc):
        self.logger.info(f"Validation Results - Epoch: {epoch}")
        self.logger.info(f"mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            self.logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

        wandb.log({
            "val/mAP": mAP,
            "val/rank1": cmc[0],
            "val/rank5": cmc[4],
            "val/rank10": cmc[9],
        })

    def log_distance_stats(self, distmat, pids, camids, q_count):
        q_pids = np.asarray(pids[:q_count])
        g_pids = np.asarray(pids[q_count:])
        q_camids = np.asarray(camids[:q_count])
        g_camids = np.asarray(camids[q_count:])

        pair_stats = compute_pair_distance_stats(distmat, q_pids, g_pids, q_camids, g_camids)
        self.logger.info(f"Pair distance stats: {pair_stats}")
        wandb.log({f"val/pair_distance_stats/{k}": v for k, v in pair_stats.items()})
