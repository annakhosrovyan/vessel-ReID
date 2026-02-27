from pathlib import Path
import os

import torch
import torch.distributed as dist
import torchvision.transforms as T

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .msaw import MsawPairDataset
from .openearthmap_sar import OpenEarthMapSarPairDataset
from .qxs_saropt import QxsSaroptPairDataset
from .sar2opt import Sar2OptPairDataset
from .sarptical import SarpticalPairDataset


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # Keep imports local to avoid pulling in heavy modules at import time.
    import random
    import numpy as np

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _is_rank0():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def rank0_print(*args, **kwargs):
    if _is_rank0():
        print(*args, **kwargs)


def _build_datasets(root_dir):
    dataset_builders = [
        ("MSAW", MsawPairDataset, "RGB_SAR_datasets/pretraining/MSAW/train"),
        ("OpenEarthMap-SAR", OpenEarthMapSarPairDataset, "RGB_SAR_datasets/pretraining/OpenEarthMap-SAR/train"),
        ("QXS-SAROPT", QxsSaroptPairDataset, "RGB_SAR_datasets/pretraining/QXS-SAROPT/QXSLAB_SAROPT"),
        ("SAR2Opt", Sar2OptPairDataset, "RGB_SAR_datasets/pretraining/SAR2Opt"),
        ("SARptical", SarpticalPairDataset, "RGB_SAR_datasets/pretraining/SARptical"),
    ]

    datasets = []
    total_pairs = 0
    for name, dataset_cls, rel_path in dataset_builders:
        root = os.path.join(root_dir, rel_path)
        if not Path(root).exists():
            raise ValueError("Root for dataset {} does not exist: {}".format(name, root))
        dataset = dataset_cls(root=root)
        datasets.append((name, dataset))
        n_pairs = len(dataset)
        total_pairs += n_pairs
        rank0_print("=> {} CLIP train pairs: {}".format(name, n_pairs))

    if total_pairs == 0:
        raise ValueError("No training pairs found in multi-dataset loader.")
    rank0_print("=> Total CLIP train pairs (all datasets): {}".format(total_pairs))
    return datasets


class MultiSourcePairDataset(Dataset):
    def __init__(self, datasets, cfg):
        self.datasets = [dataset for _, dataset in datasets]
        self.index_map = []
        for dataset_idx, dataset in enumerate(self.datasets):
            for sample_idx in range(len(dataset)):
                self.index_map.append((dataset_idx, sample_idx))
        if not self.index_map:
            raise ValueError("No training pairs found in multi-dataset loader.")
        self.train_transforms = T.Compose(
            [
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            ]
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.index_map[index]
        rgb_tensor, sar_tensor = self.datasets[dataset_idx][sample_idx]
        rgb_tensor = self.train_transforms(rgb_tensor)
        sar_tensor = self.train_transforms(sar_tensor)
        return rgb_tensor, sar_tensor


def multi_clip_pair_collate_fn(batch):
    if len(batch) == 0:
        raise ValueError("Empty batch received in multi_clip_pair_collate_fn.")
    rgb_batch = torch.stack([item[0] for item in batch], dim=0)
    sar_batch = torch.stack([item[1] for item in batch], dim=0)
    if rgb_batch.shape[0] != sar_batch.shape[0]:
        raise ValueError(
            "RGB and SAR batch size mismatch: {} vs {}".format(rgb_batch.shape[0], sar_batch.shape[0])
        )
    b = rgb_batch.shape[0]
    imgs = torch.cat([rgb_batch, sar_batch], dim=0)
    vids = torch.arange(b, dtype=torch.long).repeat(2)
    cams = torch.cat(
        [
            torch.zeros(b, dtype=torch.long),
            torch.ones(b, dtype=torch.long),
        ],
        dim=0,
    )
    return imgs, vids, cams


def _build_loader_kwargs(num_workers):
    loader_kwargs = {
        "pin_memory": True,
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return loader_kwargs


def make_multi_dataset_clip_loader(cfg):
    global_batch_size = int(cfg.SOLVER.IMS_PER_BATCH)
    if global_batch_size % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH must be even for pair training.")

    datasets = _build_datasets(cfg.DATASETS.ROOT_DIR)
    train_set = MultiSourcePairDataset(datasets, cfg)
    num_workers = int(cfg.DATALOADER.NUM_WORKERS)
    seed = int(getattr(cfg.SOLVER, "SEED", 0))
    loader_kwargs = _build_loader_kwargs(num_workers)

    generator = torch.Generator()

    if cfg.MODEL.DIST_TRAIN and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if global_batch_size % world_size != 0:
            raise ValueError(
                "cfg.SOLVER.IMS_PER_BATCH ({}) must be divisible by world size ({})".format(
                    global_batch_size, world_size
                )
            )
        local_batch_size = global_batch_size // world_size
        if local_batch_size % 2 != 0:
            raise ValueError(
                "Per-rank batch size ({}) must be even so RGB/SAR pairs stay aligned. "
                "Adjust SOLVER.IMS_PER_BATCH or GPU count.".format(local_batch_size)
            )
        pair_batch_size = local_batch_size // 2
        if pair_batch_size <= 0:
            raise ValueError("Per-rank pair batch size must be > 0.")

        sampler = DistributedSampler(
            train_set,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
        generator.manual_seed(seed + rank)
        train_loader_pair = DataLoader(
            train_set,
            batch_size=pair_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=multi_clip_pair_collate_fn,
            worker_init_fn=_seed_worker,
            generator=generator,
            **loader_kwargs
        )
    else:
        pair_batch_size = global_batch_size // 2
        if pair_batch_size <= 0:
            raise ValueError("Pair batch size must be > 0.")
        generator.manual_seed(seed)
        train_loader_pair = DataLoader(
            train_set,
            batch_size=pair_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=multi_clip_pair_collate_fn,
            worker_init_fn=_seed_worker,
            generator=generator,
            **loader_kwargs
        )

    num_classes = 0
    camera_num = 2
    return train_loader_pair, num_classes, camera_num
