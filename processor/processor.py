import os
import sys
import json
import math
import time
import logging
from contextlib import nullcontext
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
from utils.wandb_utils import init_wandb_run
from utils.metrics import R1_mAP_eval, euclidean_distance
from datasets.bases import ImageDataset
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from processor.validation_metrics_tracker import ValidationMetricsTracker
from datasets.make_dataloader import val_pair_collate_fn, val_collate_fn
from processor.ibot_utils import (
    build_ibot_multichannel_batch,
    cancel_last_layer_gradients,
    compute_teacher_momentum,
    generate_ibot_masks,
    generate_ibot_views,
    sample_channel_subsets,
)



def _save_checkpoint(cfg, save_dir, epoch, checkpoint, save_numbered=True, save_latest=False):
    """Save checkpoint to disk. Handles distributed rank check internally."""
    is_main = not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0
    if not is_main:
        return
    if save_numbered:
        torch.save(checkpoint, os.path.join(save_dir, cfg.MODEL.NAME + "_checkpoint_{}.pth".format(epoch)))
    if save_latest:
        torch.save(checkpoint, os.path.join(save_dir, cfg.MODEL.NAME + "_checkpoint_latest.pth"))


def _setup_validation_dataloader(cfg):
    val_loaders_hoss = {}
    val_loader_optisar_pair = None
    
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

    if cfg.SOLVER.TRACK_VALIDATION_METRICS_OPTISAR:
        if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
            raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
        dataset_optisar = OptiSarPairVal(root=cfg.SOLVER.PRETRAIN_TRACK_VALIDATION_DIR)
        val_set_optisar_pretrain = ImageDataset(dataset_optisar.train_pair, val_transforms, pair=True, normalize_rgb=normalize_rgb, normalize_sar=normalize_sar)
        val_loader_optisar_pair = DataLoader(
            val_set_optisar_pretrain, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2), shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, 
            collate_fn=val_pair_collate_fn
        )

    return val_loaders_hoss, val_loader_optisar_pair


def _resolve_grad_accum_steps(cfg):
    grad_accum_steps = int(getattr(cfg.SOLVER, "GRAD_ACCUM_STEPS", 1))
    if grad_accum_steps < 1:
        raise ValueError("SOLVER.GRAD_ACCUM_STEPS must be >= 1")
    return grad_accum_steps


def _distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def do_train_pair(cfg,
            model,
            train_loader_pair,
            optimizer,
            scheduler,
            loss_func,
            local_rank,
            start_epoch=0,
            scaler_state_dict=None,
            log_dir=None,
            ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    log_grad_norm = bool(getattr(cfg.SOLVER, "LOG_GRAD_NORM", True))
    grad_accum_steps = _resolve_grad_accum_steps(cfg)

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None
    if grad_accum_steps > 1:
        logger.info("Using gradient accumulation: {} steps".format(grad_accum_steps))

    decay_target_save_epochs = set()
    if cfg.SOLVER.WSD_DECAY_TARGETS:
        for t in cfg.SOLVER.WSD_DECAY_TARGETS:
            save_at = int(math.floor((1.0 - cfg.SOLVER.WSD_DECAY_PCT) * t))
            decay_target_save_epochs.add(save_at)
        logger.info("WSD decay-target checkpoints at epochs: {}".format(sorted(decay_target_save_epochs)))

    save_dir = log_dir or cfg.OUTPUT_DIR
    writer = None
    if local_rank == 0:
        init_wandb_run(
            cfg,
            logger=logger,
            tags=["pretraining", "clip-loss", cfg.MODEL.TRANSFORMER_TYPE],
        )
        writer = SummaryWriter(log_dir=save_dir)

    val_loaders_hoss, val_loader_optisar_pair = _setup_validation_dataloader(cfg)
    validation_metrics_tracker = ValidationMetricsTracker(cfg, local_rank)

    if device:
        model.to(torch.device("cuda", local_rank))
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            if local_rank == 0:
                logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    global_step = 0
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
        training_start_time = time.time()
        for epoch in range(start_epoch + 1, epochs + 1):
            start_time = time.time()
            loss_meter.reset()
            scheduler.step(epoch)
            if hasattr(train_loader_pair, "sampler") and hasattr(train_loader_pair.sampler, "set_epoch"):
                train_loader_pair.sampler.set_epoch(epoch)
            if hasattr(train_loader_pair, "set_epoch"):
                train_loader_pair.set_epoch(epoch)
            model.train()
            num_batches = len(train_loader_pair)
            tail_window_size = num_batches % grad_accum_steps
            tail_start_iter = num_batches - tail_window_size if tail_window_size > 0 else 0
            use_tqdm = (not cfg.MODEL.DIST_TRAIN or local_rank == 0) and sys.stdout.isatty()
            batch_iter = tqdm(train_loader_pair, total=num_batches, unit="batch", leave=False) if use_tqdm else train_loader_pair
            optimizer.zero_grad()
            model_ref = model.module if hasattr(model, 'module') else model
            for n_iter, (img, vid, target_cam) in enumerate(batch_iter):
                iter_idx = n_iter + 1
                is_update_step = (iter_idx % grad_accum_steps == 0) or (iter_idx == num_batches)
                loss_divisor = tail_window_size if (tail_window_size > 0 and iter_idx > tail_start_iter) else grad_accum_steps
                should_log = ((n_iter + 1) % log_period == 0) and (not cfg.MODEL.DIST_TRAIN or local_rank == 0)
                img = img.to(device, non_blocking=True)
                target = vid.to(device, non_blocking=True)
                target_cam = target_cam.to(device, non_blocking=True)
                sync_context = model.no_sync() if hasattr(model, "no_sync") and not is_update_step else nullcontext()
                with sync_context:
                    with autocast('cuda', enabled=cfg.MODEL.USE_AMP):
                        logits_per_sar = model(img, target, cam_label=target_cam)
                        loss = loss_func(logits_per_sar)
                    scaler.scale(loss / float(loss_divisor)).backward()

                total_norm = None
                if is_update_step:
                    # Gradient clipping to prevent explosion
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if should_log and log_grad_norm:
                        norm_sq = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                norm_sq += param_norm.item() ** 2
                        total_norm = norm_sq ** 0.5

                logit_scale = None
                if should_log:
                    logit_scale = model_ref.logit_scale.exp().item()

                if is_update_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                loss_meter.update(loss.item(), img.shape[0])
                global_step += 1

                if should_log:
                    elapsed = time.time() - start_time
                    batches_left = num_batches - (n_iter + 1)
                    eta_epoch = (elapsed / (n_iter + 1)) * batches_left if (n_iter + 1) > 0 else 0
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_epoch))
                    grad_norm_str = "{:.3f}".format(total_norm) if total_norm is not None else "n/a"
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}, GradNorm: {}, LogitScale: {:.3f}, ETA: {}".format(
                            epoch, (n_iter + 1), num_batches, loss_meter.avg,
                            scheduler._get_lr(epoch)[0], grad_norm_str, logit_scale, eta_str
                        )
                    )
                    if local_rank == 0:
                        wandb_payload = {
                            "epoch": epoch,
                            "train/clip_loss": loss_meter.avg,
                            "train/lr": optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                            "train/logit_scale": logit_scale,
                            "train/scaler_scale": scaler.get_scale(),
                        }
                        if total_norm is not None:
                            wandb_payload["train/grad_norm"] = total_norm
                        wandb.log(wandb_payload)
                        if writer:
                            for k, v in wandb_payload.items():
                                if k != "epoch":
                                    writer.add_scalar(k, v, global_step)

            end_time = time.time()
            epoch_time = end_time - start_time
            time_per_batch = epoch_time / (n_iter + 1)
            if not cfg.MODEL.DIST_TRAIN or local_rank == 0:
                epochs_done = epoch - start_epoch
                epochs_left = epochs - epoch
                avg_epoch_time = (end_time - training_start_time) / epochs_done
                eta_training = avg_epoch_time * epochs_left
                eta_training_str = time.strftime("%H:%M:%S", time.gmtime(eta_training))
                logger.info(
                    "Epoch {} done. Time: {:.0f}s, Time/batch: {:.3f}s, Speed: {:.1f} samples/s, ETA: {} ({} epochs left)".format(
                        epoch, epoch_time, time_per_batch,
                        cfg.SOLVER.IMS_PER_BATCH / time_per_batch,
                        eta_training_str, epochs_left
                    )
                )

            should_save_periodic = checkpoint_period > 0 and epoch % checkpoint_period == 0
            should_save_decay = epoch in decay_target_save_epochs
            save_latest_every = cfg.SOLVER.SAVE_LATEST_EVERY_EPOCH
            if should_save_periodic or should_save_decay or save_latest_every:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss_meter.avg,
                }
                _save_checkpoint(cfg, save_dir, epoch, checkpoint,
                                 save_numbered=should_save_periodic or should_save_decay,
                                 save_latest=save_latest_every)

            if local_rank == 0:
                wandb.log({
                    "epoch": epoch,
                    "train/clip_loss_epoch": loss_meter.avg,
                })
                if writer:
                    writer.add_scalar("train/clip_loss_epoch", loss_meter.avg, epoch)

            if epoch % eval_period == 0:
                eval_model = model.module if hasattr(model, "module") else model
                _distributed_barrier()
                best_acc, best_theta, mAP_all, mINP_all, cmc_all = validation_metrics_tracker.run(model=eval_model, epoch=epoch, val_loaders=val_loaders_hoss)
                if local_rank == 0 and best_acc > best_val_acc:
                    best_val_acc = best_acc
                    best_val_theta = best_theta
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(save_dir, 'best_model_threshold_acc.pth'))
                    threshold_payload = {
                        "epoch": epoch,
                        "threshold": float(best_val_theta),
                        "accuracy": float(best_val_acc),
                    }
                    with open(os.path.join(save_dir, "best_threshold.json"), "w") as f:
                        json.dump(threshold_payload, f)
                if local_rank == 0 and mAP_all > best_mAP:
                    best_mAP = mAP_all
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(save_dir, 'best_model_mAP.pth'))
                    logger.info("New best model saved with mAP: {:.1%} at epoch {}".format(best_mAP, epoch))
                if val_loader_optisar_pair is not None:
                    validation_metrics_tracker.run_pair(eval_model, epoch, val_loader_optisar_pair, collection_name='optisar')
                _distributed_barrier()

    # Save final checkpoint with epoch number
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    _save_checkpoint(cfg, save_dir, epochs, checkpoint)

    if writer:
        writer.close()


def do_train_pair_ibot(cfg,
            model,
            train_loader_pair,
            optimizer,
            scheduler,
            loss_func,
            local_rank,
            start_epoch=0,
            scaler_state_dict=None,
            log_dir=None,
            ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    log_grad_norm = bool(getattr(cfg.SOLVER, "LOG_GRAD_NORM", True))
    grad_accum_steps = _resolve_grad_accum_steps(cfg)

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("train")
    logger.info("start iBOT pretraining")
    if grad_accum_steps > 1:
        logger.info("Using gradient accumulation: {} steps".format(grad_accum_steps))

    decay_target_save_epochs = set()
    if cfg.SOLVER.WSD_DECAY_TARGETS:
        for t in cfg.SOLVER.WSD_DECAY_TARGETS:
            save_at = int(math.floor((1.0 - cfg.SOLVER.WSD_DECAY_PCT) * t))
            decay_target_save_epochs.add(save_at)
        logger.info("WSD decay-target checkpoints at epochs: {}".format(sorted(decay_target_save_epochs)))

    save_dir = log_dir or cfg.OUTPUT_DIR
    writer = None
    if local_rank == 0:
        init_wandb_run(
            cfg,
            logger=logger,
            tags=["pretraining", "ibot-loss", cfg.MODEL.TRANSFORMER_TYPE],
        )
        writer = SummaryWriter(log_dir=save_dir)

    val_loaders_hoss, val_loader_optisar_pair = _setup_validation_dataloader(cfg)
    validation_metrics_tracker = ValidationMetricsTracker(cfg, local_rank)

    if device:
        model.to(torch.device("cuda", local_rank))
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            if local_rank == 0:
                logger.info("Using {} GPUs for iBOT training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    if isinstance(loss_func, nn.Module):
        loss_func.to(torch.device("cuda", local_rank))

    global_step = 0
    loss_meter = AverageMeter()
    cls_meter = AverageMeter()
    patch_meter = AverageMeter()
    scaler = GradScaler('cuda')
    if scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)

    best_mAP = 0.0
    best_val_acc = -1.0
    best_val_theta = float("nan")

    training_start_time = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        cls_meter.reset()
        patch_meter.reset()

        scheduler.step(epoch)
        if hasattr(train_loader_pair, "sampler") and hasattr(train_loader_pair.sampler, "set_epoch"):
            train_loader_pair.sampler.set_epoch(epoch)
        if hasattr(train_loader_pair, "set_epoch"):
            train_loader_pair.set_epoch(epoch)

        model.train()
        num_batches = len(train_loader_pair)
        tail_window_size = num_batches % grad_accum_steps
        tail_start_iter = num_batches - tail_window_size if tail_window_size > 0 else 0
        num_optimizer_updates = max(1, (num_batches + grad_accum_steps - 1) // grad_accum_steps)
        optimizer_updates_done = 0
        use_tqdm = (not cfg.MODEL.DIST_TRAIN or local_rank == 0) and sys.stdout.isatty()
        batch_iter = tqdm(train_loader_pair, total=num_batches, unit="batch", leave=False) if use_tqdm else train_loader_pair
        optimizer.zero_grad()
        model_ref = model.module if hasattr(model, 'module') else model

        for n_iter, (img, vid, target_cam) in enumerate(batch_iter):
            iter_idx = n_iter + 1
            is_update_step = (iter_idx % grad_accum_steps == 0) or (iter_idx == num_batches)
            loss_divisor = tail_window_size if (tail_window_size > 0 and iter_idx > tail_start_iter) else grad_accum_steps
            should_log = ((n_iter + 1) % log_period == 0) and (not cfg.MODEL.DIST_TRAIN or local_rank == 0)

            img = img.to(device, non_blocking=True)
            target_cam = target_cam.to(device, non_blocking=True)

            pair_batch, channel_pool = build_ibot_multichannel_batch(
                img,
                rgb_channels=cfg.MODEL.RGB_CHANNELS,
                sar_channels=cfg.MODEL.SAR_CHANNELS,
            )
            global_channel_idxs, local_channel_idxs = sample_channel_subsets(
                channel_pool=channel_pool,
                sampling_subset=bool(cfg.IBOT.SAMPLING_SUBSET),
                sync_channel_count=bool(cfg.IBOT.SYNC_CHANNEL_COUNT),
                modality_pure_sampling=bool(getattr(cfg.IBOT, "MODALITY_PURE_SAMPLING", False)),
                rgb_channels=cfg.MODEL.RGB_CHANNELS,
                sar_channels=cfg.MODEL.SAR_CHANNELS,
                device=pair_batch.device,
            )
            global_views, local_views = generate_ibot_views(pair_batch, cfg)
            channel_to_pos = {ch: idx for idx, ch in enumerate(channel_pool)}
            global_positions = [channel_to_pos[ch] for ch in global_channel_idxs]
            local_positions = [channel_to_pos[ch] for ch in local_channel_idxs]
            global_views = [view[:, global_positions, :, :] for view in global_views]
            local_views = [view[:, local_positions, :, :] for view in local_views]
            global_masks = generate_ibot_masks(global_views, cfg, epoch=epoch)

            sync_context = model.no_sync() if hasattr(model, "no_sync") and not is_update_step else nullcontext()
            with sync_context:
                with autocast('cuda', enabled=cfg.MODEL.USE_AMP):
                    student_output, teacher_output, student_local_cls = model(
                        global_views,
                        mode='ibot',
                        local_x=local_views,
                        global_masks=global_masks,
                        global_channel_idxs=global_channel_idxs,
                        local_channel_idxs=local_channel_idxs,
                    )
                    losses = loss_func(
                        student_output=student_output,
                        teacher_output=teacher_output,
                        student_local_cls=student_local_cls,
                        student_masks=global_masks,
                        epoch=epoch,
                        num_channels=len(global_channel_idxs),
                    )
                    loss = losses['loss']
                scaler.scale(loss / float(loss_divisor)).backward()

            total_norm = None
            if is_update_step:
                scaler.unscale_(optimizer)
                if float(cfg.IBOT.CLIP_GRAD) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.IBOT.CLIP_GRAD))

                if epoch <= int(cfg.IBOT.FREEZE_LAST_LAYER_EPOCHS):
                    if hasattr(model_ref, 'cancel_gradients_last_layer'):
                        model_ref.cancel_gradients_last_layer()
                    else:
                        cancel_last_layer_gradients(model_ref)

                if should_log and log_grad_norm:
                    norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            norm_sq += param_norm.item() ** 2
                    total_norm = norm_sq ** 0.5

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                with torch.no_grad():
                    momentum = compute_teacher_momentum(
                        base_momentum=cfg.IBOT.MOMENTUM_TEACHER,
                        epoch=epoch,
                        iter_idx=optimizer_updates_done,
                        num_iters=num_optimizer_updates,
                        total_epochs=epochs,
                    )
                    if hasattr(model_ref, 'momentum_update_teacher'):
                        model_ref.momentum_update_teacher(momentum)
                optimizer_updates_done += 1

            bs = pair_batch.shape[0]
            loss_meter.update(loss.item(), bs)
            cls_meter.update(losses['cls'].item(), bs)
            patch_meter.update(losses['patch'].item(), bs)
            global_step += 1

            if should_log:
                elapsed = time.time() - start_time
                batches_left = num_batches - (n_iter + 1)
                eta_epoch = (elapsed / (n_iter + 1)) * batches_left if (n_iter + 1) > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_epoch))
                grad_norm_str = "{:.3f}".format(total_norm) if total_norm is not None else "n/a"
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, CLS: {:.3f}, Patch: {:.3f}, Base Lr: {:.2e}, GradNorm: {}, ETA: {}".format(
                        epoch, (n_iter + 1), num_batches, loss_meter.avg, cls_meter.avg, patch_meter.avg,
                        scheduler._get_lr(epoch)[0], grad_norm_str, eta_str
                    )
                )
                if local_rank == 0:
                    payload = {
                        "epoch": epoch,
                        "train/ibot_loss": loss_meter.avg,
                        "train/ibot_cls_loss": cls_meter.avg,
                        "train/ibot_patch_loss": patch_meter.avg,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/epoch": epoch,
                        "train/num_global_channels": len(global_channel_idxs),
                        "train/num_local_channels": len(local_channel_idxs),
                        "train/scaler_scale": scaler.get_scale(),
                    }
                    if total_norm is not None:
                        payload["train/grad_norm"] = total_norm
                    wandb.log(payload)
                    if writer:
                        for k, v in payload.items():
                            if k != "epoch":
                                writer.add_scalar(k, v, global_step)

        end_time = time.time()
        epoch_time = end_time - start_time
        time_per_batch = epoch_time / max(1, num_batches)
        if not cfg.MODEL.DIST_TRAIN or local_rank == 0:
            epochs_done = epoch - start_epoch
            epochs_left = epochs - epoch
            avg_epoch_time = (end_time - training_start_time) / max(1, epochs_done)
            eta_training = avg_epoch_time * epochs_left
            eta_training_str = time.strftime("%H:%M:%S", time.gmtime(eta_training))
            logger.info(
                "Epoch {} done. Time: {:.0f}s, Time/batch: {:.3f}s, Speed: {:.1f} samples/s, ETA: {} ({} epochs left)".format(
                    epoch, epoch_time, time_per_batch,
                    cfg.SOLVER.IMS_PER_BATCH / max(time_per_batch, 1e-6),
                    eta_training_str, epochs_left
                )
            )

        should_save_periodic = checkpoint_period > 0 and epoch % checkpoint_period == 0
        should_save_decay = epoch in decay_target_save_epochs
        save_latest_every = cfg.SOLVER.SAVE_LATEST_EVERY_EPOCH
        if should_save_periodic or should_save_decay or save_latest_every:
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': loss_meter.avg,
            }
            if isinstance(loss_func, nn.Module):
                checkpoint['ibot_loss_state_dict'] = loss_func.state_dict()
            _save_checkpoint(cfg, save_dir, epoch, checkpoint,
                             save_numbered=should_save_periodic or should_save_decay,
                             save_latest=save_latest_every)

        if local_rank == 0:
            wandb.log({
                "epoch": epoch,
                "train/ibot_loss_epoch": loss_meter.avg,
                "train/ibot_cls_loss_epoch": cls_meter.avg,
                "train/ibot_patch_loss_epoch": patch_meter.avg,
            })
            if writer:
                writer.add_scalar("train/ibot_loss_epoch", loss_meter.avg, epoch)
                writer.add_scalar("train/ibot_cls_loss_epoch", cls_meter.avg, epoch)
                writer.add_scalar("train/ibot_patch_loss_epoch", patch_meter.avg, epoch)

        if epoch % eval_period == 0:
            eval_model = model.module if hasattr(model, "module") else model
            _distributed_barrier()
            best_acc, best_theta, mAP_all, mINP_all, cmc_all = validation_metrics_tracker.run(model=eval_model, epoch=epoch, val_loaders=val_loaders_hoss)
            if local_rank == 0 and best_acc > best_val_acc:
                best_val_acc = best_acc
                best_val_theta = best_theta
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'best_model_threshold_acc.pth'))
                threshold_payload = {
                    "epoch": epoch,
                    "threshold": float(best_val_theta),
                    "accuracy": float(best_val_acc),
                }
                with open(os.path.join(save_dir, "best_threshold.json"), "w") as f:
                    json.dump(threshold_payload, f)
            if local_rank == 0 and mAP_all > best_mAP:
                best_mAP = mAP_all
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, os.path.join(save_dir, 'best_model_mAP.pth'))
                logger.info("New best model saved with mAP: {:.1%} at epoch {}".format(best_mAP, epoch))
            if val_loader_optisar_pair is not None:
                validation_metrics_tracker.run_pair(eval_model, epoch, val_loader_optisar_pair, collection_name='optisar')
            _distributed_barrier()

    # Save final checkpoint with epoch number
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    if isinstance(loss_func, nn.Module):
        checkpoint['ibot_loss_state_dict'] = loss_func.state_dict()
    _save_checkpoint(cfg, save_dir, epochs, checkpoint)

    if writer:
        writer.close()


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
             scaler_state_dict=None,
             log_dir=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    decay_target_save_epochs = set()
    if cfg.SOLVER.WSD_DECAY_TARGETS:
        for t in cfg.SOLVER.WSD_DECAY_TARGETS:
            save_at = int(math.floor((1.0 - cfg.SOLVER.WSD_DECAY_PCT) * t))
            decay_target_save_epochs.add(save_at)
        logger.info("WSD decay-target checkpoints at epochs: {}".format(sorted(decay_target_save_epochs)))

    val_loaders_hoss, val_loader_optisar_pair = _setup_validation_dataloader(cfg)
    validation_metrics_tracker = ValidationMetricsTracker(cfg, local_rank)

    if device:
        model.to(torch.device("cuda", local_rank))
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            if local_rank == 0:
                logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = GradScaler()
    if scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)

    save_dir = log_dir or cfg.OUTPUT_DIR
    writer = None
    best_mAP = 0.0
    best_val_acc = -1.0
    best_val_theta = float("nan")
    if local_rank == 0:
        init_wandb_run(cfg, logger=logger)
        writer = SummaryWriter(log_dir=save_dir)

    global_step = 0
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
            global_step += 1

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_last_lr()[0]))
                if writer:
                    writer.add_scalar("train/loss", loss_meter.avg, global_step)
                    writer.add_scalar("train/acc", acc_meter.avg, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        end_time = time.time()
        epoch_time = end_time - start_time
        time_per_batch = epoch_time / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN or local_rank == 0:
            logger.info("Epoch {} done. Time: {:.0f}s, Time/batch: {:.3f}s, Speed: {:.1f} samples/s"
                    .format(epoch, epoch_time, time_per_batch, train_loader.batch_size / time_per_batch))

        should_save_periodic = checkpoint_period > 0 and epoch % checkpoint_period == 0
        should_save_decay = epoch in decay_target_save_epochs
        save_latest_every = cfg.SOLVER.SAVE_LATEST_EVERY_EPOCH
        if should_save_periodic or should_save_decay or save_latest_every:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_center_state_dict': optimizer_center.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': loss_meter.avg,
            }
            _save_checkpoint(cfg, save_dir, epoch, checkpoint,
                             save_numbered=should_save_periodic or should_save_decay,
                             save_latest=save_latest_every)

        if epoch % eval_period == 0:
            eval_model = model.module if hasattr(model, "module") else model
            _distributed_barrier()
            if (cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0) or (not cfg.MODEL.DIST_TRAIN and local_rank == 0):
                best_acc, best_theta, mAP_all, mINP_all, cmc_all = validation_metrics_tracker.run(model=eval_model, epoch=epoch, val_loaders=val_loaders_hoss)
                if best_acc > best_val_acc:
                    best_val_acc = best_acc
                    best_val_theta = best_theta
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(save_dir, 'best_model_threshold_acc.pth'))
                    threshold_payload = {
                        "epoch": epoch,
                        "threshold": float(best_val_theta),
                        "accuracy": float(best_val_acc),
                    }
                    with open(os.path.join(save_dir, "best_threshold.json"), "w") as f:
                        json.dump(threshold_payload, f)
                    logger.info("New best model saved with threshold accuracy: {:.1%} at epoch {}".format(best_val_acc, epoch))
                if mAP_all > best_mAP:
                    best_mAP = mAP_all
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, os.path.join(save_dir, 'best_model_mAP.pth'))
                    logger.info("New best model saved with mAP: {:.1%} at epoch {}".format(best_mAP, epoch))
                torch.cuda.empty_cache()
            if (val_loader_optisar_pair is not None and
                ((cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0) or
                 (not cfg.MODEL.DIST_TRAIN and local_rank == 0))):
                validation_metrics_tracker.run_pair(eval_model, epoch, val_loader_optisar_pair, collection_name='optisar')
            _distributed_barrier()

        if local_rank == 0:
            wandb.log({
                "epoch": epoch,
                "train_loss": loss_meter.avg,
                "train_acc": acc_meter.avg,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            if writer:
                writer.add_scalar("train/loss_epoch", loss_meter.avg, epoch)
                writer.add_scalar("train/acc_epoch", acc_meter.avg, epoch)
                writer.add_scalar("train/lr_epoch", scheduler.get_last_lr()[0], epoch)

    # Save final checkpoint with epoch number
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_center_state_dict': optimizer_center.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    _save_checkpoint(cfg, save_dir, epochs, checkpoint)

    if local_rank == 0:
        if writer:
            writer.close()
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
            logger.info('Using {} GPUs for inference'.format(torch.cuda.device_count()))
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
            weight_dir = os.path.dirname(cfg.TEST.WEIGHT)
            candidate = os.path.join(weight_dir, "best_threshold.json")
            if os.path.exists(candidate):
                return candidate
        candidate = os.path.join(cfg.OUTPUT_DIR, "best_threshold.json")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError("best_threshold.json not found in checkpoint directory or OUTPUT_DIR")

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
