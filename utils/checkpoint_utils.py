import torch
from typing import Optional, Tuple, Union
from torch.optim import Optimizer


def move_optimizer_state_to_device(optimizer: Optimizer, device: Union[torch.device, str]) -> None:
    dev = device if isinstance(device, torch.device) else torch.device(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(dev, non_blocking=True)


def resume_from_checkpoint(cfg,
                           resume_path: str,
                           model,
                           optimizer,
                           scheduler,
                           local_rank: int = 0,
                           optimizer_center=None) -> Tuple[int, Optional[dict]]:
    start_epoch = 0
    scaler_state_dict = None
    if not resume_path:
        return start_epoch, scaler_state_dict
    ckpt = torch.load(resume_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            device = torch.device(f"cuda:{local_rank}") if cfg.MODEL.DIST_TRAIN else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            move_optimizer_state_to_device(optimizer, device)
        if optimizer_center is not None and "optimizer_center_state_dict" in ckpt:
            optimizer_center.load_state_dict(ckpt["optimizer_center_state_dict"])
            device = torch.device(f"cuda:{local_rank}") if cfg.MODEL.DIST_TRAIN else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            move_optimizer_state_to_device(optimizer_center, device)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
        if "scaler_state_dict" in ckpt:
            scaler_state_dict = ckpt["scaler_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return start_epoch, scaler_state_dict

