import torch
import wandb
import logging
import numpy as np
from loss.contrastive_loss import clip_loss
from utils.metrics import R1_mAP_eval, euclidean_distance


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

        cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
        self.log_metrics(epoch, mAP, cmc)

        if self.cfg.SOLVER.TRACK_VALIDATION_METRICS:
            self.log_distance_stats(distmat, pids, camids, evaluator.num_query, collection_name='hoss')
            self.log_posneg_margins(distmat, pids, camids, evaluator.num_query, collection_name='hoss', epoch=epoch)
            self.log_posneg_margins_by_modalities(distmat, pids, camids, evaluator.num_query, collection_name='hoss', epoch=epoch)
        
        if hasattr(model, "module"):
            logit_scale = model.module.logit_scale.exp().item()
        else:
            logit_scale = model.logit_scale.exp().item()

        q_pids = np.asarray(pids[:evaluator.num_query])
        g_pids = np.asarray(pids[evaluator.num_query:])
        q_camids = np.asarray(camids[:evaluator.num_query])
        g_camids = np.asarray(camids[evaluator.num_query:])

        rgb, sar = 0, 1
        pid_to_rgb = {}
        pid_to_sar = {}

        for i, (pid_v, cid_v) in enumerate(zip(q_pids, q_camids)):
            if cid_v == rgb and pid_v not in pid_to_rgb:
                pid_to_rgb[pid_v] = ("q", i)
            if cid_v == sar and pid_v not in pid_to_sar:
                pid_to_sar[pid_v] = ("q", i)
        for j, (pid_v, cid_v) in enumerate(zip(g_pids, g_camids)):
            if cid_v == rgb and pid_v not in pid_to_rgb:
                pid_to_rgb[pid_v] = ("g", j)
            if cid_v == sar and pid_v not in pid_to_sar:
                pid_to_sar[pid_v] = ("g", j)

        common = [pid for pid in pid_to_rgb if pid in pid_to_sar]
        rgb_feats = []
        sar_feats = []
        for pid_v in common:
            src_r, idx_r = pid_to_rgb[pid_v]
            src_s, idx_s = pid_to_sar[pid_v]
            rgb_feat = qf[idx_r] if src_r == "q" else gf[idx_r]
            sar_feat = qf[idx_s] if src_s == "q" else gf[idx_s]
            rgb_feats.append(rgb_feat)
            sar_feats.append(sar_feat)
        rgb_mat = torch.stack(rgb_feats, dim=0)
        sar_mat = torch.stack(sar_feats, dim=0)
        sim_union = torch.matmul(sar_mat, rgb_mat.t()) * float(logit_scale)
        loss_sar_rgb = clip_loss(sim_union)
        wandb.log({
            "val/epoch": epoch,
            "epoch": epoch,
            "hoss/clip_loss": float(loss_sar_rgb.item()),
            "hoss/clip_pairs": len(common),
        })

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
        if str(self.cfg.TEST.FEAT_NORM).lower() in ("yes","true","1","y","t","on"):
            q_feats = torch.nn.functional.normalize(q_feats, dim=1, p=2)
            g_feats = torch.nn.functional.normalize(g_feats, dim=1, p=2)
        distmat = euclidean_distance(q_feats, g_feats)
        
        q_pids = np.array(pids_list)
        g_pids = np.array(pids_list)
        q_camids = torch.cat(camids1_list, dim=0).cpu().numpy()
        g_camids = torch.cat(camids2_list, dim=0).cpu().numpy()
        
        pids = np.concatenate([q_pids, g_pids])
        camids = np.concatenate([q_camids, g_camids])
        q_count = len(q_pids)

        self.log_distance_stats(distmat, pids, camids, q_count, collection_name=collection_name)
        self.log_posneg_margins(distmat, pids, camids, q_count, collection_name=collection_name, epoch=epoch)
        
        if hasattr(model, "module"):
            logit_scale = model.module.logit_scale.exp().item()
        else:
            logit_scale = model.logit_scale.exp().item()
        sim = torch.matmul(q_feats, g_feats.t()) * float(logit_scale)
        clip_loss_val = clip_loss(sim)
        wandb.log({
            "epoch": epoch,
            "val/epoch": epoch,
            f"{collection_name}/clip_loss": clip_loss_val.item(),
        })

    def log_metrics(self, epoch, mAP, cmc):
        self.logger.info(f"Validation Results - Epoch: {epoch}")
        self.logger.info(f"mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            self.logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

        wandb.log({
            "epoch": epoch,
            "val/epoch": epoch,
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

    def log_posneg_margins(self, distmat, pids, camids, q_count, collection_name='val', epoch=0):
        q_pids = np.asarray(pids[:q_count])
        g_pids = np.asarray(pids[q_count:])
        q_camids = np.asarray(camids[:q_count])
        g_camids = np.asarray(camids[q_count:])
        margins = []
        rows = []
        for i in range(q_count):
            pos_mask = (g_pids == q_pids[i])
            if not np.any(pos_mask):
                continue
            pos_mean = float(np.mean(distmat[i, pos_mask]))
            neg_mask = ~pos_mask
            if not np.any(neg_mask):
                continue
            neg_min = float(np.min(distmat[i, :][neg_mask]))
            margin = float(neg_min - pos_mean)
            rows.append([int(q_pids[i]), int(q_camids[i]), pos_mask.sum(), pos_mean, neg_min, margin])
            margins.append(margin)
        if len(margins) > 0:
            data = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows]
            table = wandb.Table(data=data, columns=["pid", "q_camid", "pos_count", "pos_mean", "neg_min", "margin"])
            wandb.log({
                f"{collection_name}/margin_mean": float(np.mean(margins)),
                f"{collection_name}/margin_min": float(np.min(margins)),
                f"{collection_name}/margin_max": float(np.max(margins)),
                f"{collection_name}/margin_std": float(np.std(margins)),
                f"{collection_name}/margin_hist": wandb.HogwildHistogram(margins) if hasattr(wandb, "HogwildHistogram") else wandb.Histogram(np.array(margins)),
                f"{collection_name}/margins": table,
                "val/epoch": epoch,
            })

    def _compute_margins_filtered(self, distmat, q_pids, g_pids, q_camids, g_camids, q_mode=None, g_mode=None):
        q_indices = np.arange(len(q_pids)) if q_mode is None else np.where(q_camids == q_mode)[0]
        margins = []
        for i in q_indices:
            g_mask = np.ones_like(g_pids, dtype=bool) if g_mode is None else (g_camids == g_mode)
            pos_mask = (g_pids == q_pids[i]) & g_mask
            if not np.any(pos_mask):
                continue
            pos_mean = float(np.mean(distmat[i, pos_mask]))
            neg_mask = (g_pids != q_pids[i]) & g_mask
            if not np.any(neg_mask):
                continue
            neg_min = float(np.min(distmat[i, :][neg_mask]))
            margins.append(float(neg_min - pos_mean))
        return margins

    def log_posneg_margins_by_modalities(self, distmat, pids, camids, q_count, collection_name='val', epoch=0):
        q_pids = np.asarray(pids[:q_count])
        g_pids = np.asarray(pids[q_count:])
        q_camids = np.asarray(camids[:q_count])
        g_camids = np.asarray(camids[q_count:])
        labels = {0: "rgb", 1: "sar"}
        for q_mode in (0, 1):
            for g_mode in (0, 1):
                margins = self._compute_margins_filtered(distmat, q_pids, g_pids, q_camids, g_camids, q_mode=q_mode, g_mode=g_mode)
                tag = f"{labels[q_mode]}_{labels[g_mode]}"
                if len(margins) == 0:
                    continue
                wandb.log({
                    f"{collection_name}/margin_mean_{tag}": float(np.mean(margins)),
                    f"{collection_name}/margin_min_{tag}": float(np.min(margins)),
                    f"{collection_name}/margin_max_{tag}": float(np.max(margins)),
                    f"{collection_name}/margin_std_{tag}": float(np.std(margins)),
                    f"{collection_name}/margin_hist_{tag}": wandb.Histogram(np.array(margins)),
                    "val/epoch": epoch,
                })
