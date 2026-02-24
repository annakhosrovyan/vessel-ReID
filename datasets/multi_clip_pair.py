from pathlib import Path
import math

import torch
import torch.distributed as dist
import torchvision.transforms as T

from .msaw import create_msaw_dataloader
from .openearthmap_sar import create_openearthmap_sar_dataloader
from .qxs_saropt import create_qxs_saropt_dataloader
from .sar2opt import create_sar2opt_dataloader
from .sarptical import create_sarptical_dataloader


DATASET_ROOTS = {
    "MSAW": "/nfs/h100/raid/rs/vessel_detection/RGB_SAR_datasets/pretraining/MSAW/train",
    "OpenEarthMap-SAR": "/nfs/h100/raid/rs/vessel_detection/RGB_SAR_datasets/pretraining/OpenEarthMap-SAR/train",
    "QXS-SAROPT": "/nfs/h100/raid/rs/vessel_detection/RGB_SAR_datasets/pretraining/QXS-SAROPT/QXSLAB_SAROPT",
    "SAR2Opt": "/nfs/h100/raid/rs/vessel_detection/RGB_SAR_datasets/pretraining/SAR2Opt",
    "SARptical": "/nfs/h100/raid/rs/vessel_detection/RGB_SAR_datasets/pretraining/SARptical",
}


def _build_loaders(cfg):
    for name, root in DATASET_ROOTS.items():
        if not Path(root).exists():
            raise ValueError("Root for dataset {} does not exist: {}".format(name, root))

    batch_size_pair = int(cfg.SOLVER.IMS_PER_BATCH / 2)
    if batch_size_pair <= 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH must be positive and even.")

    num_workers = cfg.DATALOADER.NUM_WORKERS

    loaders = []
    names = ["MSAW", "OpenEarthMap-SAR", "QXS-SAROPT", "SAR2Opt", "SARptical"]
    loaders.append(create_msaw_dataloader(DATASET_ROOTS["MSAW"], batch_size_pair, num_workers, shuffle=True))
    loaders.append(create_openearthmap_sar_dataloader(DATASET_ROOTS["OpenEarthMap-SAR"], batch_size_pair, num_workers, shuffle=True))
    loaders.append(create_qxs_saropt_dataloader(DATASET_ROOTS["QXS-SAROPT"], batch_size_pair, num_workers, shuffle=True))
    loaders.append(create_sar2opt_dataloader(DATASET_ROOTS["SAR2Opt"], batch_size_pair, num_workers, shuffle=True))
    loaders.append(create_sarptical_dataloader(DATASET_ROOTS["SARptical"], batch_size_pair, num_workers, shuffle=True))

    total_pairs = 0
    for name, loader in zip(names, loaders):
        n_pairs = len(loader.dataset)
        total_pairs += n_pairs
        print("=> {} CLIP train pairs: {}".format(name, n_pairs))
    print("=> Total CLIP train pairs (all datasets): {}".format(total_pairs))

    return loaders


class MultiDatasetClipLoader:
    def __init__(self, loaders, global_batch_size, cfg):
        if global_batch_size % 2 != 0:
            raise ValueError("global_batch_size must be even.")
        self.loaders = list(loaders)
        self.datasets = [loader.dataset for loader in self.loaders]
        self.batch_size = int(global_batch_size)
        self.half_batch = self.batch_size // 2
        self.total_pairs = sum(len(dataset) for dataset in self.datasets)
        if self.total_pairs == 0:
            raise ValueError("No training pairs found in multi-dataset loader.")
        self.seed = int(getattr(cfg.SOLVER, "SEED", 0))
        self.epoch = 0
        self.train_transforms = T.Compose(
            [
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            ]
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _apply_transforms(self, batch):
        out = []
        for i in range(batch.shape[0]):
            out.append(self.train_transforms(batch[i]))
        return torch.stack(out, dim=0)

    def __iter__(self):
        indices = []
        for dataset_idx, dataset in enumerate(self.datasets):
            for sample_idx in range(len(dataset)):
                indices.append((dataset_idx, sample_idx))
        g = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(indices), generator=g, device="cpu").tolist()
        shuffled = [indices[i] for i in order]
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            half_batch = (self.batch_size // world_size) // 2
            if half_batch <= 0:
                half_batch = 1
            my_indices = shuffled[rank::world_size]
        else:
            half_batch = self.half_batch
            my_indices = shuffled
        for start in range(0, len(my_indices), half_batch):
            batch_pairs = my_indices[start:start + half_batch]
            rgb_list = []
            sar_list = []
            for dataset_idx, sample_idx in batch_pairs:
                rgb_tensor, sar_tensor = self.datasets[dataset_idx][sample_idx]
                rgb_tensor = self.train_transforms(rgb_tensor)
                sar_tensor = self.train_transforms(sar_tensor)
                rgb_list.append(rgb_tensor)
                sar_list.append(sar_tensor)
            rgb_batch = torch.stack(rgb_list, dim=0)
            sar_batch = torch.stack(sar_list, dim=0)
            if rgb_batch.shape[0] != sar_batch.shape[0]:
                raise ValueError("RGB and SAR batch size mismatch: {} vs {}".format(rgb_batch.shape[0], sar_batch.shape[0]))
            b = rgb_batch.shape[0]
            imgs = torch.cat([rgb_batch, sar_batch], dim=0)
            vids = torch.arange(b, device=imgs.device, dtype=torch.long)
            vids = vids.repeat(2)
            cam_rgb = torch.zeros(b, device=imgs.device, dtype=torch.long)
            cam_sar = torch.ones(b, device=imgs.device, dtype=torch.long)
            cams = torch.cat([cam_rgb, cam_sar], dim=0)
            yield imgs, vids, cams

    def __len__(self):
        if self.total_pairs == 0:
            return 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            half_batch = (self.batch_size // world_size) // 2
            if half_batch <= 0:
                return 0
            num_my = (self.total_pairs + world_size - 1 - rank) // world_size
            return int(math.ceil(num_my / half_batch))
        if self.half_batch <= 0:
            return 0
        return int(math.ceil(self.total_pairs / self.half_batch))


def make_multi_dataset_clip_loader(cfg):
    loaders = _build_loaders(cfg)
    global_batch_size = int(cfg.SOLVER.IMS_PER_BATCH)
    loader = MultiDatasetClipLoader(loaders, global_batch_size, cfg)
    num_classes = 0
    camera_num = 2
    return loader, num_classes, camera_num

