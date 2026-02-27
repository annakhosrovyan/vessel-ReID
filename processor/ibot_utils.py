import math
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


_GAUSSIAN_BLUR = T.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0))


def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _broadcast_int(value: int, device: torch.device) -> int:
    if not _is_dist_ready():
        return int(value)
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


def _sample_pred_ratio(pred_ratio, pred_ratio_var, epoch: int, pred_start_epoch: int) -> float:
    if int(epoch) < int(pred_start_epoch):
        return 0.0

    if isinstance(pred_ratio, (list, tuple)):
        if isinstance(pred_ratio_var, (list, tuple)):
            vars_list = list(pred_ratio_var)
        else:
            vars_list = [pred_ratio_var] * len(pred_ratio)
        sampled = []
        for prm, prv in zip(pred_ratio, vars_list):
            prm = float(prm)
            prv = float(prv)
            if prm < prv:
                raise ValueError(f"pred_ratio ({prm}) must be >= pred_ratio_var ({prv})")
            sampled.append(random.uniform(prm - prv, prm + prv) if prv > 0 else prm)
        return float(random.choice(sampled))

    prm = float(pred_ratio)
    prv = float(pred_ratio_var)
    if prm < prv:
        raise ValueError(f"pred_ratio ({prm}) must be >= pred_ratio_var ({prv})")
    return float(random.uniform(prm - prv, prm + prv) if prv > 0 else prm)


def _apply_flip_rot(x: torch.Tensor) -> torch.Tensor:
    if random.random() >= 0.5:
        return x
    op = random.choice(("hflip", "vflip", "rot90"))
    if op == "hflip":
        return torch.flip(x, dims=[2])
    if op == "vflip":
        return torch.flip(x, dims=[1])
    k = random.choice((1, 3))
    return torch.rot90(x, k, dims=(1, 2))


def _make_one_crop(
    img: torch.Tensor,
    out_size: int,
    scale: Sequence[float],
    blur_p: float,
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> torch.Tensor:
    i, j, h, w = T.RandomResizedCrop.get_params(img, scale=tuple(scale), ratio=ratio)
    crop = TF.resized_crop(
        img,
        top=i,
        left=j,
        height=h,
        width=w,
        size=[int(out_size), int(out_size)],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    crop = _apply_flip_rot(crop)
    if random.random() < float(blur_p):
        crop = _GAUSSIAN_BLUR(crop)
    return crop


def _make_view_batch(batch: torch.Tensor, out_size: int, scale: Sequence[float], blur_p: float) -> torch.Tensor:
    crops = [_make_one_crop(batch[idx], out_size=out_size, scale=scale, blur_p=blur_p) for idx in range(batch.shape[0])]
    return torch.stack(crops, dim=0)


def build_ibot_multichannel_batch(
    img: torch.Tensor,
    rgb_channels: Sequence[int],
    sar_channels: Sequence[int],
) -> Tuple[torch.Tensor, List[int]]:
    if img.shape[0] % 2 != 0:
        raise ValueError(f"Expected even batch for pair pretraining, got {img.shape[0]}")
    half = img.shape[0] // 2
    rgb = img[:half]
    sar = img[half:]

    rgb_needed = len(rgb_channels)
    sar_needed = len(sar_channels)
    if rgb.shape[1] < rgb_needed:
        raise ValueError(f"RGB tensor has {rgb.shape[1]} channels but {rgb_needed} are required")
    if sar.shape[1] < sar_needed:
        raise ValueError(f"SAR tensor has {sar.shape[1]} channels but {sar_needed} are required")

    rgb = rgb[:, :rgb_needed, :, :]
    sar = sar[:, :sar_needed, :, :]
    merged = torch.cat([rgb, sar], dim=1)
    channel_pool = list(rgb_channels) + list(sar_channels)
    return merged, channel_pool


def sample_channel_subsets(
    channel_pool: Sequence[int],
    sampling_subset: bool,
    sync_channel_count: bool,
    device: torch.device,
    modality_pure_sampling: bool = False,
    rgb_channels: Sequence[int] = (),
    sar_channels: Sequence[int] = (),
) -> Tuple[List[int], List[int]]:
    if len(channel_pool) == 0:
        raise ValueError("Channel pool cannot be empty")

    channel_pool = list(channel_pool)

    if not modality_pure_sampling:
        num_ch = len(channel_pool)
        global_num = random.randint(1, num_ch)
        if sync_channel_count and _is_dist_ready():
            if dist.get_rank() != 0:
                global_num = 0
            global_num = _broadcast_int(global_num, device=device)
        global_num = max(1, min(global_num, num_ch))
        global_channels = random.sample(channel_pool, k=global_num)

        local_max = global_num if sampling_subset else num_ch
        local_num = random.randint(1, local_max)
        if sync_channel_count and _is_dist_ready():
            if dist.get_rank() != 0:
                local_num = 0
            local_num = _broadcast_int(local_num, device=device)
        local_num = max(1, min(local_num, local_max))
        if sampling_subset:
            local_channels = random.sample(global_channels, k=local_num)
        else:
            local_channels = random.sample(channel_pool, k=local_num)
        return global_channels, local_channels

    channel_pool_set = set(channel_pool)
    rgb_pool = [ch for ch in rgb_channels if ch in channel_pool_set]
    sar_pool = [ch for ch in sar_channels if ch in channel_pool_set]
    if len(rgb_pool) == 0 and len(sar_pool) == 0:
        raise ValueError("No RGB/SAR channels found in channel_pool for modality-pure sampling")

    available_modalities = []
    if len(rgb_pool) > 0:
        available_modalities.append(0)  # RGB
    if len(sar_pool) > 0:
        available_modalities.append(1)  # SAR

    global_modality = random.choice(available_modalities)
    local_modality = random.choice(available_modalities)
    if sync_channel_count and _is_dist_ready():
        if dist.get_rank() != 0:
            global_modality = 0
            local_modality = 0
        global_modality = _broadcast_int(global_modality, device=device)
        local_modality = _broadcast_int(local_modality, device=device)

    def _resolve_modality_pool(modality: int) -> List[int]:
        if modality == 0 and len(rgb_pool) > 0:
            return rgb_pool
        if modality == 1 and len(sar_pool) > 0:
            return sar_pool
        if len(rgb_pool) > 0:
            return rgb_pool
        return sar_pool

    global_pool = _resolve_modality_pool(global_modality)
    local_pool = _resolve_modality_pool(local_modality)

    global_num = random.randint(1, len(global_pool))
    if sync_channel_count and _is_dist_ready():
        if dist.get_rank() != 0:
            global_num = 0
        global_num = _broadcast_int(global_num, device=device)
    global_num = max(1, min(global_num, len(global_pool)))
    global_channels = random.sample(global_pool, k=global_num)

    # Keep local/global modality independent. If modalities differ, subset mode
    # is overridden so local channels are drawn from its own modality pool.
    same_modality_pool = global_pool is local_pool
    if sampling_subset and same_modality_pool:
        local_max = global_num
        local_source = global_channels
    else:
        local_max = len(local_pool)
        local_source = local_pool

    local_num = random.randint(1, local_max)
    if sync_channel_count and _is_dist_ready():
        if dist.get_rank() != 0:
            local_num = 0
        local_num = _broadcast_int(local_num, device=device)
    local_num = max(1, min(local_num, local_max))
    local_channels = random.sample(local_source, k=local_num)
    return global_channels, local_channels


def generate_ibot_views(batch: torch.Tensor, cfg) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    global_views = []
    local_views = []

    for gi in range(int(cfg.IBOT.GLOBAL_CROPS_NUMBER)):
        blur_p = 1.0 if gi == 0 else 0.1
        global_views.append(
            _make_view_batch(
                batch,
                out_size=int(cfg.IBOT.GLOBAL_SIZE),
                scale=cfg.IBOT.GLOBAL_CROPS_SCALE,
                blur_p=0,
            )
        )
    for _ in range(int(cfg.IBOT.LOCAL_CROPS_NUMBER)):
        local_views.append(
            _make_view_batch(
                batch,
                out_size=int(cfg.IBOT.LOCAL_SIZE),
                scale=cfg.IBOT.LOCAL_CROPS_SCALE,
                blur_p=0,
            )
        )
    return global_views, local_views


def generate_ibot_masks(global_views: List[torch.Tensor], cfg, epoch: int) -> List[torch.Tensor]:
    if str(cfg.IBOT.PRED_SHAPE).lower() != "rand":
        raise ValueError(f"Unsupported IBOT.PRED_SHAPE={cfg.IBOT.PRED_SHAPE}. Only 'rand' is implemented.")

    patch = int(cfg.MODEL.STRIDE_SIZE[0])
    masks = []
    for view in global_views:
        b, _, h, w = view.shape
        h_tokens = h // patch
        w_tokens = w // patch
        view_masks = []
        for _ in range(b):
            ratio = _sample_pred_ratio(
                pred_ratio=cfg.IBOT.PRED_RATIO,
                pred_ratio_var=cfg.IBOT.PRED_RATIO_VAR,
                epoch=epoch,
                pred_start_epoch=cfg.IBOT.PRED_START_EPOCH,
            )
            n_tokens = h_tokens * w_tokens
            n_masked = int(max(0, min(n_tokens, int(ratio * n_tokens))))
            mask = np.hstack(
                [
                    np.zeros(n_tokens - n_masked, dtype=np.bool_),
                    np.ones(n_masked, dtype=np.bool_),
                ]
            )
            np.random.shuffle(mask)
            view_masks.append(mask.reshape(h_tokens, w_tokens))
        mask_t = torch.from_numpy(np.stack(view_masks, axis=0)).to(device=view.device, dtype=torch.bool)
        masks.append(mask_t)
    return masks


def compute_teacher_momentum(base_momentum: float, epoch: int, iter_idx: int, num_iters: int, total_epochs: int) -> float:
    if num_iters <= 0 or total_epochs <= 0:
        return float(base_momentum)
    total_steps = max(1, int(num_iters * total_epochs))
    cur_step = int((epoch - 1) * num_iters + iter_idx)
    cur_step = max(0, min(cur_step, total_steps - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * cur_step / (total_steps - 1 if total_steps > 1 else 1)))
    return float(1.0 - (1.0 - float(base_momentum)) * cosine)


def cancel_last_layer_gradients(model) -> None:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "student_head.last_layer" in name:
            p.grad = None
