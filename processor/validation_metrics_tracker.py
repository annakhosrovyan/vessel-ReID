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

def compute_threshold_accuracy(entries):
    if len(entries) == 0:
        return 0.0, float("nan")
    distances = np.array([entry[2] for entry in entries], dtype=np.float64)
    thetas = np.unique(distances)
    best_acc = -1.0
    best_theta = float(thetas[0])
    total = len(entries)
    for theta in thetas:
        correct = 0
        for pid, pred_pid, dist, has_pair in entries:
            if dist > theta:
                if not has_pair:
                    correct += 1
            else:
                if has_pair and pred_pid == pid:
                    correct += 1
        acc = correct / total
        if acc > best_acc or (acc == best_acc and theta < best_theta):
            best_acc = acc
            best_theta = float(theta)
    return float(best_acc), float(best_theta)


class ValidationMetricsTracker:
    def __init__(self, cfg, local_rank):
        self.logger = logging.getLogger("Tracking Validation Metrics")
        self.cfg = cfg
        self.local_rank = local_rank


    def run(self, model, epoch, val_loaders=None):
        if val_loaders is None or self.local_rank != 0:
            return 0.0, float("nan"), 0.0, 0.0, np.zeros(0, dtype=np.float32)

        self.logger.info(f"Epoch {epoch}    tracking validation metrics...")
        model.eval()

        mode_results = {}
        all_best_acc = 0.0
        all_best_theta = float("nan")

        for mode, (val_loader, num_query_val) in val_loaders.items():
            evaluator = R1_mAP_eval(num_query_val, max_rank=50, feat_norm=self.cfg.TEST.FEAT_NORM)
            evaluator.reset()

            for _, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to("cuda")
                    camids = camids.to("cuda")
                    img_wh = img_wh.to("cuda")

                    feat = model(img, cam_label=camids, img_wh=img_wh)
                    evaluator.update((feat, vid, camid))

            cmc, mAP, mINP, distmat, pids, camids, qf, gf = evaluator.compute()
            mode_results[mode] = (mAP, mINP, cmc)

            logger = logging.getLogger("train")
            logger.info(f"HOSS {mode} - mAP: {mAP:.1%}, mINP: {mINP:.1%}, Rank-1: {cmc[0]:.1%}")

            wandb.log({
                "epoch": epoch,
                f"hoss_{mode}/mAP": float(mAP),
                f"hoss_{mode}/mINP": float(mINP),
                f"hoss_{mode}/rank1": float(cmc[0]),
                f"hoss_{mode}/rank5": float(cmc[4]),
                f"hoss_{mode}/rank10": float(cmc[9]),
            })

            collection_name = f"hoss_{mode}"
            self.log_distance_stats(distmat, pids, camids, evaluator.num_query, collection_name=collection_name)
            self.log_posneg_margins(distmat, pids, camids, evaluator.num_query, collection_name=collection_name, epoch=epoch)
            self.log_posneg_margins_by_modalities(distmat, pids, camids, evaluator.num_query, collection_name=collection_name, epoch=epoch)

            q_pids = np.asarray(pids[:evaluator.num_query])
            g_pids = np.asarray(pids[evaluator.num_query:])
            g_pid_set = set(int(pid) for pid in g_pids)
            min_indices = np.argmin(distmat, axis=1)
            threshold_entries = []
            for i in range(q_pids.shape[0]):
                pred_pid = int(g_pids[min_indices[i]])
                dist = float(distmat[i, min_indices[i]])
                pid = int(q_pids[i])
                has_pair = pid in g_pid_set
                threshold_entries.append((pid, pred_pid, dist, has_pair))

            best_acc, best_theta = compute_threshold_accuracy(threshold_entries)
            wandb.log({
                "epoch": epoch,
                f"hoss_{mode}/threshold_accuracy": float(best_acc),
                f"hoss_{mode}/threshold_theta": float(best_theta),
            })

            if mode == "all":
                all_best_acc = best_acc
                all_best_theta = best_theta
                self.log_metrics(epoch, float(mAP), float(mINP), cmc)

        all_mAP, all_mINP, all_cmc = mode_results["all"]
        logger = logging.getLogger("train")
        logger.info(f"HOSS threshold accuracy: {all_best_acc:.1%} with theta {all_best_theta:.6f}")
        wandb.log({
            "epoch": epoch,
            "val/threshold_accuracy": float(all_best_acc),
            "val/threshold_theta": float(all_best_theta),
        })

        return all_best_acc, all_best_theta, float(all_mAP), float(all_mINP), all_cmc

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

    def log_metrics(self, epoch, mAP, mINP, cmc):
        self.logger.info(f"Validation Results - Epoch: {epoch}")
        self.logger.info(f"mAP: {mAP:.1%}")
        self.logger.info(f"mINP: {mINP:.1%}")
        for r in [1, 5, 10]:
            self.logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

        wandb.log({
            "epoch": epoch,
            "val/epoch": epoch,
            "val/mAP": mAP,
            "val/mINP": mINP,
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
