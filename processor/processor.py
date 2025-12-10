import os
import time
import torch
import wandb
import logging
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as T

from datasets.hoss import HOSS
from datasets.optisar_pair_val import OptiSarPairVal
from utils.meter import AverageMeter
from utils.wandb_utils import configure_wandb_metrics
from utils.metrics import R1_mAP_eval
from datasets.bases import ImageDataset
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from processor.validation_metrics_tracker import ValidationMetricsTracker
from datasets.make_dataloader import val_pair_collate_fn, val_collate_fn



def _setup_validation_dataloader(cfg):
    val_loader_hoss, num_query_hoss, val_loader_optisar_pair = None, 0, None
    
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
        
    if cfg.SOLVER.TRACK_VALIDATION_METRICS:
        val_ds = HOSS()
        val_set_hoss = ImageDataset(val_ds.query_val + val_ds.gallery_val, val_transforms)
        val_loader_hoss = DataLoader(
            val_set_hoss, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=val_collate_fn
        )
        num_query_hoss = len(val_ds.query_val)

    if cfg.SOLVER.TRACK_VALIDATION_METRICS_OPTISAR:
        if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
            raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
        dataset_optisar = OptiSarPairVal(root=cfg.SOLVER.PRETRAIN_TRACK_VALIDATION_DIR)
        val_set_optisar_pretrain = ImageDataset(dataset_optisar.train_pair, val_transforms, pair=True)
        val_loader_optisar_pair = DataLoader(
            val_set_optisar_pretrain, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2), shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, 
            collate_fn=val_pair_collate_fn
        )

    return val_loader_hoss, num_query_hoss, val_loader_optisar_pair


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

    logger = logging.getLogger("transreid.train")
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

    val_loader_hoss, num_query_hoss, val_loader_optisar_pair = _setup_validation_dataloader(cfg)
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
            model.train()
            for n_iter, (img, vid, target_cam) in enumerate(train_loader_pair):
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
                if (n_iter + 1) % log_period == 0:
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
                if val_loader_hoss is not None:
                    validation_metrics_tracker.run(model=model, epoch=epoch, val_loader=val_loader_hoss, num_query_val=num_query_hoss)
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

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    _, _, val_loader_optisar_pair = _setup_validation_dataloader(cfg)
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

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            img_wh = img_wh.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view, img_wh=img_wh)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, distmat, pids, camids, _, _ = evaluator.compute()

                    if cfg.SOLVER.TRACK_VALIDATION_METRICS:
                        validation_metrics_tracker.log_distance_stats(distmat, pids, camids, evaluator.num_query, collection_name='val')
                        validation_metrics_tracker.log_posneg_margins(distmat, pids, camids, evaluator.num_query, collection_name='val', epoch=epoch)
                        validation_metrics_tracker.log_posneg_margins_by_modalities(distmat, pids, camids, evaluator.num_query, collection_name='val', epoch=epoch)

                    if val_loader_optisar_pair is not None:
                        validation_metrics_tracker.run_pair(model, epoch, val_loader_optisar_pair, collection_name='optisar')

                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    
                    if mAP > best_mAP:
                        best_mAP = mAP
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                        logger.info("New best model saved with mAP: {:.1%} at epoch {}".format(best_mAP, epoch))
                    
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        img_wh = img_wh.to(device)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, distmat, pids, camids, _, _ = evaluator.compute()

                if cfg.SOLVER.TRACK_VALIDATION_METRICS:
                    validation_metrics_tracker.log_distance_stats(distmat, pids, camids, evaluator.num_query, collection_name='hoss')
                    validation_metrics_tracker.log_posneg_margins(distmat, pids, camids, evaluator.num_query, collection_name='hoss', epoch=epoch)
                    validation_metrics_tracker.log_posneg_margins_by_modalities(distmat, pids, camids, evaluator.num_query, collection_name='hoss', epoch=epoch)
                    
                    if val_loader_optisar_pair is not None:
                        validation_metrics_tracker.run_pair(model, epoch, val_loader_optisar_pair, collection_name='optisar')

                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                
                if mAP > best_mAP:
                    best_mAP = mAP
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                    logger.info("New best model saved with mAP: {:.1%} at epoch {}".format(best_mAP, epoch))
                
                torch.cuda.empty_cache()

            if local_rank == 0:
                wandb.log({
                    "epoch": epoch,
                    "val_mAP": mAP,
                    "val_rank1": cmc[0],
                    "val_rank5": cmc[4],
                    "val_rank10": cmc[9]
                })

        if local_rank == 0:
            wandb.log({
                "epoch": epoch,
                "train_loss": loss_meter.avg,
                "train_acc": acc_meter.avg,
                "learning_rate": scheduler.get_last_lr()[0]
            })

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

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Test Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
