import os
import json
import time
import logging
import numpy as np
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as T

from datasets.hoss import HOSS
from datasets.optisar_pair_val import OptiSarPairVal
from utils.meter import AverageMeter
from utils.wandb_utils import configure_wandb_metrics
from utils.metrics import R1_mAP_eval, euclidean_distance
from datasets.bases import ImageDataset
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from processor.validation_metrics_tracker import ValidationMetricsTracker
from datasets.make_dataloader import val_pair_collate_fn, val_collate_fn, PadToSquareAndResize
from loss.triplet_loss import TripletLoss



def _setup_validation_dataloader(cfg):
    val_loaders_hoss = {}
    val_loader_optisar_pair = None
    val_triplet_loader = None
    # val_transforms = T.Compose([
    #     T.Resize(cfg.INPUT.SIZE_TEST),
    #     T.ToTensor(),
    # ])
    if (getattr(cfg.MODEL, "TRANSFORMER_TYPE", "") == "chivit_base"
        and getattr(cfg.DATASETS, "NAMES", "") == "HOSS"
    ):
        val_transforms = T.Compose([
            PadToSquareAndResize(size=cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
        ])
    else:
        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
        ])

    normalize_rgb = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN_RGB, 
                                std=cfg.INPUT.PIXEL_STD_RGB)
    normalize_sar = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN_SAR, 
                                std=cfg.INPUT.PIXEL_STD_SAR)


    eval_modes = ['rgb_sar', 'sar_rgb', 'rgb_mixed', 'sar_mixed', 'all']
    for mode in eval_modes:
        val_ds = HOSS(root=cfg.DATASETS.ROOT_DIR, eval_mode=mode, verbose=False)
        val_set_hoss = ImageDataset(val_ds.query_val + val_ds.gallery_val, val_transforms, normalize_rgb=normalize_rgb, normalize_sar=normalize_sar)
        val_loader_hoss = DataLoader(
            val_set_hoss, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=val_collate_fn
        )
        num_query_hoss = len(val_ds.query_val)
        val_loaders_hoss[mode] = (val_loader_hoss, num_query_hoss)
        if mode == "all":
            from datasets.sampler import RandomIdentitySampler
            triplet_sampler = RandomIdentitySampler(
                val_ds.query_val + val_ds.gallery_val,
                cfg.SOLVER.IMS_PER_BATCH,
                cfg.DATALOADER.NUM_INSTANCE,
            )
            val_triplet_loader = DataLoader(
                val_set_hoss,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=triplet_sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                collate_fn=val_collate_fn,
            )

    if cfg.SOLVER.TRACK_VALIDATION_METRICS_OPTISAR:
        if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
            raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
        dataset_optisar = OptiSarPairVal(root=cfg.SOLVER.PRETRAIN_TRACK_VALIDATION_DIR)
        val_set_optisar_pretrain = ImageDataset(dataset_optisar.train_pair, val_transforms, pair=True, normalize_rgb=normalize_rgb, normalize_sar=normalize_sar)
        val_loader_optisar_pair = DataLoader(
            val_set_optisar_pretrain, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2), shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, 
            collate_fn=val_pair_collate_fn
        )

    return val_loaders_hoss, val_loader_optisar_pair, val_triplet_loader


def do_train_pair(cfg, 
            model, 
            train_loader_pair, 
            optimizer, 
            scheduler, 
            loss_func,
            local_rank,
            start_epoch=0,
            scaler_state_dict=None,
            ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None

    if local_rank == 0:
        wandb.init(
            project=cfg.WANDB.PROJECT,
            name=cfg.WANDB.NAME,
            config=cfg,
            tags=["pretraining", "clip-loss", cfg.MODEL.TRANSFORMER_TYPE]
        )
        configure_wandb_metrics()

    val_loaders_hoss, val_loader_optisar_pair, _ = _setup_validation_dataloader(cfg)
    validation_metrics_tracker = ValidationMetricsTracker(cfg, local_rank)

    if device:
        model.to(torch.device("cuda", local_rank))
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    scaler = GradScaler('cuda')
    if scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)

    best_mAP = 0.0
    best_val_acc = -1.0
    best_val_theta = float("nan")

    # train pair
    if cfg.MODEL.PAIR:
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model.module.train_with_pair()
        else:
            model.train_with_pair()
        for epoch in range(start_epoch + 1, epochs + 1):
            start_time = time.time()
            loss_meter.reset()
            scheduler.step(epoch)
            if hasattr(train_loader_pair, "sampler") and hasattr(train_loader_pair.sampler, "set_epoch"):
                train_loader_pair.sampler.set_epoch(epoch)
            if hasattr(train_loader_pair, "set_epoch"):
                train_loader_pair.set_epoch(epoch)
            model.train()
            batch_iter = train_loader_pair
            if not cfg.MODEL.DIST_TRAIN or local_rank == 0:
                batch_iter = tqdm(batch_iter, total=len(train_loader_pair), unit="batch", leave=False)
            for n_iter, (img, vid, target_cam) in enumerate(batch_iter):
                optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                with autocast('cuda', enabled=cfg.MODEL.USE_AMP):
                    logits_per_sar = model(img, target, cam_label=target_cam)
                    loss = loss_func(logits_per_sar)

                scaler.scale(loss).backward()

                # Gradient clipping to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Compute gradient norms before clipping
                total_norm = 0.0
                param_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                # Get logit_scale value for monitoring
                if hasattr(model, 'module'):
                    logit_scale = model.module.logit_scale.exp().item()
                else:
                    logit_scale = model.logit_scale.exp().item()

                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), img.shape[0])

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0 and (not cfg.MODEL.DIST_TRAIN or local_rank == 0):
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}, GradNorm: {:.3f}, LogitScale: {:.3f}".format(
                            epoch, (n_iter + 1), len(train_loader_pair), loss_meter.avg, 
                            scheduler._get_lr(epoch)[0], total_norm, logit_scale
                        )
                    )
                    if local_rank == 0:
                        wandb.log({
                            "epoch": epoch,
                            "train/clip_loss": loss_meter.avg,
                            "train/lr": optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                            "train/grad_norm": total_norm,
                            "train/logit_scale": logit_scale,
                            "train/scaler_scale": scaler.get_scale(),
                        })

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info(
                    "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                        epoch, time_per_batch, train_loader_pair.batch_size / time_per_batch
                    )
                )

            if epoch % checkpoint_period == 0:
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'loss': loss_meter.avg,
                        }
                        torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_checkpoint_{}.pth".format(epoch)))
                        torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_checkpoint_latest.pth"))
                else:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': loss_meter.avg,
                    }
                    torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_checkpoint_{}.pth".format(epoch)))
                    torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_checkpoint_latest.pth"))
            
            if local_rank == 0:
                wandb.log({
                    "epoch": epoch,
                    "train/clip_loss_epoch": loss_meter.avg,
                })
        
            if epoch % eval_period == 0:
                best_acc, best_theta, mAP_all, mINP_all, cmc_all = validation_metrics_tracker.run(model=model, epoch=epoch, val_loaders=val_loaders_hoss)
                if local_rank == 0 and best_acc > best_val_acc:
                    best_val_acc = best_acc
                    best_val_theta = best_theta
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(cfg.OUTPUT_DIR, 'best_model_threshold_acc.pth'))
                    threshold_payload = {
                        "epoch": epoch,
                        "threshold": float(best_val_theta),
                        "accuracy": float(best_val_acc),
                    }
                    with open(os.path.join(cfg.OUTPUT_DIR, "best_threshold.json"), "w") as f:
                        json.dump(threshold_payload, f)
                if local_rank == 0 and mAP_all > best_mAP:
                    best_mAP = mAP_all
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(cfg.OUTPUT_DIR, 'best_model_mAP.pth'))
                    logger.info("New best model saved with mAP: {:.1%} at epoch {}".format(best_mAP, epoch))
                if val_loader_optisar_pair is not None:
                    validation_metrics_tracker.run_pair(model, epoch, val_loader_optisar_pair, collection_name='optisar')



def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             start_epoch=0,
             scaler_state_dict=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    val_loaders_hoss, val_loader_optisar_pair, val_triplet_loader = _setup_validation_dataloader(cfg)
    validation_metrics_tracker = ValidationMetricsTracker(cfg, local_rank)

    if device:
        model.to(torch.device("cuda", local_rank))
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = GradScaler()
    if scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)

    best_mAP = 0.0
    best_val_acc = -1.0
    best_val_theta = float("nan")
    best_val_loss = float("inf")
    best_val_loss_theta = float("nan")

    val_triplet = None
    if "triplet" in cfg.MODEL.METRIC_LOSS_TYPE and val_triplet_loader is not None:
        if cfg.MODEL.NO_MARGIN:
            val_triplet = TripletLoss()
        else:
            val_triplet = TripletLoss(cfg.SOLVER.MARGIN)
    if wandb.run is None:   # prevent wandb from initializing multiple times during pretraining
        wandb.init(
            project=cfg.WANDB.PROJECT,
            name=cfg.WANDB.NAME,
            config=cfg
        )
        configure_wandb_metrics()

    # train
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model.module.train_with_single()
    else:
        model.train_with_single()
    for epoch in range(start_epoch + 1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, img_wh) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            img_wh = img_wh.to(device)

            with autocast('cuda', enabled=cfg.MODEL.USE_AMP):
                score, feat = model(img, target, cam_label=target_cam, img_wh=img_wh)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_last_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'optimizer_center_state_dict': optimizer_center.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': loss_meter.avg,
                    }
                    torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_checkpoint_{}.pth".format(epoch)))
                    torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_checkpoint_latest.pth"))
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_center_state_dict': optimizer_center.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss_meter.avg,
                }
                torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_checkpoint_{}.pth".format(epoch)))
                torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_checkpoint_latest.pth"))

        val_loss_avg = None
        if epoch % eval_period == 0:
            if (cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0) or (not cfg.MODEL.DIST_TRAIN and local_rank == 0):
                if val_triplet is not None:
                    val_loss_meter = AverageMeter()
                    model.eval()
                    with torch.no_grad():
                        for img, vid, camid, camids_batch, target_view, imgpath, img_wh in val_triplet_loader:
                            img = img.to(device)
                            target = torch.as_tensor(vid, dtype=torch.int64, device=device)
                            camids_batch = camids_batch.to(device)
                            img_wh = img_wh.to(device)
                            feat = model(img, cam_label=camids_batch, img_wh=img_wh)
                            tri_loss, _, _ = val_triplet(feat, target)
                            val_loss_meter.update(tri_loss.item(), img.shape[0])
                    val_loss_avg = val_loss_meter.avg

                best_acc, best_theta, mAP_all, mINP_all, cmc_all = validation_metrics_tracker.run(model=model, epoch=epoch, val_loaders=val_loaders_hoss)
                last_threshold_payload = {
                    "epoch": epoch,
                    "threshold": float(best_theta),
                    "accuracy": float(best_acc),
                    "mAP": float(mAP_all),
                }
                with open(os.path.join(cfg.OUTPUT_DIR, "last_threshold.json"), "w") as f:
                    json.dump(last_threshold_payload, f)
                if best_acc > best_val_acc:
                    best_val_acc = best_acc
                    best_val_theta = best_theta
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(cfg.OUTPUT_DIR, 'best_model_threshold_acc.pth'))
                    threshold_payload = {
                        "epoch": epoch,
                        "threshold": float(best_val_theta),
                        "accuracy": float(best_val_acc),
                    }
                    with open(os.path.join(cfg.OUTPUT_DIR, "best_threshold.json"), "w") as f:
                        json.dump(threshold_payload, f)
                    logger.info("New best model saved with threshold accuracy: {:.1%} at epoch {}".format(best_val_acc, epoch))
                if mAP_all > best_mAP:
                    best_mAP = mAP_all
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(cfg.OUTPUT_DIR, 'best_model_mAP.pth'))
                    threshold_payload = {
                        "epoch": epoch,
                        "threshold": float(best_theta),
                        "mAP": float(best_mAP),
                    }
                    with open(os.path.join(cfg.OUTPUT_DIR, "best_threshold_mAP.json"), "w") as f:
                        json.dump(threshold_payload, f)
                    logger.info("New best model saved with mAP: {:.1%} at epoch {}".format(best_mAP, epoch))
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    best_val_loss_theta = best_theta
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(cfg.OUTPUT_DIR, 'best_model_val_loss.pth'))
                    threshold_payload = {
                        "epoch": epoch,
                        "threshold": float(best_val_loss_theta),
                        "val_loss": float(best_val_loss),
                    }
                    with open(os.path.join(cfg.OUTPUT_DIR, "best_threshold_val_loss.json"), "w") as f:
                        json.dump(threshold_payload, f)
                    logger.info("New best model saved with val loss: {:.6f} at epoch {}".format(best_val_loss, epoch))
                torch.cuda.empty_cache()
            if (val_loader_optisar_pair is not None and 
                ((cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0) or 
                 (not cfg.MODEL.DIST_TRAIN and local_rank == 0))):
                validation_metrics_tracker.run_pair(model, epoch, val_loader_optisar_pair, collection_name='optisar')

        if local_rank == 0:
            log_payload = {
                "epoch": epoch,
                "train_loss": loss_meter.avg,
                "train_acc": acc_meter.avg,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if val_loss_avg is not None:
                log_payload["val_loss"] = val_loss_avg
            wandb.log(log_payload)

    if local_rank == 0:
        wandb.finish()

    logger.info("Training completed. Best mAP: {:.1%}".format(best_mAP))


def do_inference(cfg,
                 model,
                 test_loader,
                 num_query,
                 cross_id_modality=False):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cross_id_modality=cross_id_modality)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(test_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, mINP, _, _, _, _, _ = evaluator.compute()
    logger.info("Test Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    logger.info("mINP: {:.1%}".format(mINP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    eval_modes = ["rgb_sar", "sar_rgb", "rgb_mixed", "sar_mixed", "all"]

    def resolve_threshold_path():
        if cfg.TEST.WEIGHT:
            weight_path = cfg.TEST.WEIGHT
            weight_dir = os.path.dirname(weight_path)
            weight_name = os.path.basename(weight_path)

            candidates = []
            if "best_model_val_loss" in weight_name:
                candidates.append(os.path.join(weight_dir, "best_threshold_val_loss.json"))
            if "best_model_threshold_acc" in weight_name:
                candidates.append(os.path.join(weight_dir, "best_threshold.json"))
            if "best_model_mAP" in weight_name:
                candidates.append(os.path.join(weight_dir, "best_threshold_mAP.json"))
            if "checkpoint" in weight_name:
                candidates.append(os.path.join(weight_dir, "last_threshold.json"))

            candidates.extend([
                os.path.join(weight_dir, "best_threshold.json"),
                os.path.join(weight_dir, "best_threshold_val_loss.json"),
                os.path.join(weight_dir, "best_threshold_mAP.json"),
                os.path.join(weight_dir, "last_threshold.json"),
            ])

            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate

        for name in ["best_threshold.json", "best_threshold_val_loss.json", "best_threshold_mAP.json", "last_threshold.json"]:
            candidate = os.path.join(cfg.OUTPUT_DIR, name)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError("no threshold json found in checkpoint directory or OUTPUT_DIR")

    def load_threshold(threshold_path):
        with open(threshold_path, "r") as f:
            payload = json.load(f)
        if "threshold" not in payload:
            raise KeyError("best_threshold.json missing 'threshold'")
        return float(payload["threshold"])

    def accuracy_for_theta(entries, theta):
        if len(entries) == 0:
            return 0.0
        correct = 0
        for pid, pred_pid, dist, has_pair in entries:
            if dist > theta:
                if not has_pair:
                    correct += 1
            else:
                if has_pair and pred_pid == pid:
                    correct += 1
        return correct / len(entries)

    threshold_path = resolve_threshold_path()
    theta = load_threshold(threshold_path)
    logger.info("Using threshold {:.6f} from {}".format(theta, threshold_path))

    # val_transforms = T.Compose([
    #     T.Resize(cfg.INPUT.SIZE_TEST),
    #     T.ToTensor(),
    # ])
    if (getattr(cfg.MODEL, "TRANSFORMER_TYPE", "") == "chivit_base"
        and getattr(cfg.DATASETS, "NAMES", "") == "HOSS"
    ):
        val_transforms = T.Compose([
            PadToSquareAndResize(size=cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
        ])
    else:
        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
        ])
    normalize_rgb = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN_RGB, std=cfg.INPUT.PIXEL_STD_RGB)
    normalize_sar = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN_SAR, std=cfg.INPUT.PIXEL_STD_SAR)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    all_entries = []
    for mode in eval_modes:
        ds = HOSS(root=cfg.DATASETS.ROOT_DIR, eval_mode=mode, verbose=False)
        test_set = ImageDataset(ds.query + ds.gallery, val_transforms, normalize_rgb=normalize_rgb, normalize_sar=normalize_sar)
        loader = DataLoader(
            test_set,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
        )
        feats = []
        pids = []
        camids = []
        for _, (img, pid, camid, camids_batch, target_view, imgpath, img_wh) in enumerate(loader):
            with torch.no_grad():
                img = img.to(device)
                camids_batch = camids_batch.to(device)
                img_wh = img_wh.to(device)
                feat = model(img, cam_label=camids_batch, img_wh=img_wh)
                feats.append(feat.cpu())
                pids.extend(pid)
                camids.extend(camid)
        feats = torch.cat(feats, dim=0)
        if str(cfg.TEST.FEAT_NORM).lower() in ("yes", "true", "1", "y", "t", "on"):
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        q_count = len(ds.query)
        qf = feats[:q_count]
        gf = feats[q_count:]
        distmat = euclidean_distance(qf, gf)
        q_pids = np.asarray(pids[:q_count])
        g_pids = np.asarray(pids[q_count:])
        g_pid_set = set(int(pid) for pid in g_pids)
        min_indices = np.argmin(distmat, axis=1)
        entries = []
        for i in range(q_pids.shape[0]):
            pred_pid = int(g_pids[min_indices[i]])
            dist = float(distmat[i, min_indices[i]])
            pid = int(q_pids[i])
            has_pair = pid in g_pid_set
            entries.append((pid, pred_pid, dist, has_pair))
        acc = accuracy_for_theta(entries, theta)
        logger.info("Test threshold accuracy {}: {:.1%}".format(mode, acc))
        all_entries.extend(entries)

    overall_acc = accuracy_for_theta(all_entries, theta)
    logger.info("Test threshold accuracy all modes: {:.1%}".format(overall_acc))
    return cmc[0], cmc[4]
