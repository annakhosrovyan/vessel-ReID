import torch
import wandb
import logging
import numpy as np
from utils.metrics import R1_mAP_eval


def compute_stats(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
        }
    return {
        "mean": float(np.mean(vals)),
        "max": float(np.max(vals)),
        "min": float(np.min(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
    }


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
        val_stats = compute_stats(vals)
        for stat_name, stat_val in val_stats.items():
            stats[f"{k}_{stat_name}"] = stat_val
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
            self.log_distance_stats(distmat, pids, camids, evaluator.num_query, collection_name='hoss')
            
        return mAP

    def run_pair(self, model, epoch, val_loader=None, collection_name='pretrain'):
        if val_loader is None or self.local_rank != 0:
            return 0.0
        
        self.logger.info(f"Epoch {epoch}    tracking validation metrics for pairs...")

        model.eval()
        feats1_list, feats2_list = [], []
        pids_list, camids1_list, camids2_list = [], [], []
        for _, batch in enumerate(val_loader):
            with torch.no_grad():
                
                img1, pids1, camid1, _, img_wh1 = batch[0]
                img2, _, camid2, _, img_wh2 = batch[1]

                img1 = img1.to("cuda")
                camid1 = camid1.to("cuda")
                img_wh1 = img_wh1.to("cuda")

                img2 = img2.to("cuda")
                camid2 = camid2.to("cuda")
                img_wh2 = img_wh2.to("cuda")

                pids_list.extend(pids1)
                camids1_list.append(camid1)
                camids2_list.append(camid2)

                feat1 = model(img1, cam_label=camid1, img_wh=img_wh1)
                feat2 = model(img2, cam_label=camid2, img_wh=img_wh2)
                
                feats1_list.append(feat1)
                feats2_list.append(feat2)

        q_feats = torch.cat(feats1_list, dim=0)
        g_feats = torch.cat(feats2_list, dim=0)
        distmat = torch.cdist(q_feats, g_feats).cpu().numpy()
        
        q_pids = np.array(pids_list)
        g_pids = np.array(pids_list)
        q_camids = torch.cat(camids1_list, dim=0).cpu().numpy()
        g_camids = torch.cat(camids2_list, dim=0).cpu().numpy()
        
        pids = np.concatenate([q_pids, g_pids])
        camids = np.concatenate([q_camids, g_camids])
        q_count = len(q_pids)

        self.log_distance_stats(distmat, pids, camids, q_count, collection_name=collection_name)

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

    def log_distance_stats(self, distmat, pids, camids, q_count, collection_name='val'):
        q_pids = np.asarray(pids[:q_count])
        g_pids = np.asarray(pids[q_count:])
        q_camids = np.asarray(camids[:q_count])
        g_camids = np.asarray(camids[q_count:])

        stats = compute_pair_distance_stats(distmat, q_pids, g_pids, q_camids, g_camids)
        wandb.log({
            f"{collection_name}/{k}": v for k, v in stats.items()
        })
