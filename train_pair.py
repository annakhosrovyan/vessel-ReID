import os
import warnings
import torch
import random
import argparse
import numpy as np

from config import cfg
from loss import make_loss
from model import make_model
from solver import make_optimizer
from processor import do_train_pair, do_train_pair_ibot
from utils.logdir import setup_log_dir
from datasets import make_dataloader_pair
from datasets.multi_clip_pair import make_multi_dataset_clip_loader
from solver.scheduler_factory import create_scheduler
from utils.checkpoint_utils import resume_from_checkpoint


def set_seed(seed, deterministic=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--log_dir_name", default="", type=str, help="override log subdirectory name (share across runs)")
    args = parser.parse_args()
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not cfg.MODEL.DIST_TRAIN:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    set_seed(cfg.SOLVER.SEED, deterministic=cfg.SOLVER.DETERMINISTIC)

    log_dir, logger = setup_log_dir(cfg, args, log_dir_name=args.log_dir_name)
    use_multi_dataset = getattr(cfg.SOLVER, "USE_MULTI_PRETRAIN", False)
    if use_multi_dataset:
        train_loader_pair, num_classes, camera_num = make_multi_dataset_clip_loader(cfg)
    else:
        train_loader_pair, num_classes, camera_num = make_dataloader_pair(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)

    loss_func, center_criterion = make_loss(cfg, num_classes)
    
    optimizer, _ = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    start_epoch, scaler_state_dict = resume_from_checkpoint(cfg, cfg.SOLVER.RESUME_FROM, model, optimizer, scheduler, args.local_rank)
    if cfg.MODEL.METRIC_LOSS_TYPE == 'ibot' and cfg.SOLVER.RESUME_FROM and isinstance(loss_func, torch.nn.Module):
        ckpt = torch.load(cfg.SOLVER.RESUME_FROM, map_location='cpu')
        if isinstance(ckpt, dict) and "ibot_loss_state_dict" in ckpt:
            ibot_state = ckpt["ibot_loss_state_dict"]
            # Handle schedule length mismatch when resuming with different MAX_EPOCHS
            schedule_keys = ["teacher_temp_schedule", "teacher_temp2_schedule"]
            saved_schedules = {}
            for k in schedule_keys:
                if k in ibot_state:
                    saved_schedules[k] = ibot_state.pop(k)
            loss_func.load_state_dict(ibot_state, strict=False)
            for k, saved in saved_schedules.items():
                current = getattr(loss_func, k)
                n = current.numel()
                if saved.numel() >= n:
                    current.copy_(saved[:n])
                else:
                    current[:saved.numel()].copy_(saved)
            if args.local_rank == 0:
                logger.info("Loaded iBOT loss state from {}".format(cfg.SOLVER.RESUME_FROM))

    warnings.filterwarnings("once", message=".*upsample_bicubic2d_backward.*")

    train_fn = do_train_pair_ibot if cfg.MODEL.METRIC_LOSS_TYPE == 'ibot' else do_train_pair
    train_fn(
        cfg,
        model,
        train_loader_pair,
        optimizer,
        scheduler,
        loss_func,
        args.local_rank,
        start_epoch=start_epoch,
        scaler_state_dict=scaler_state_dict,
        log_dir=log_dir,
    )
