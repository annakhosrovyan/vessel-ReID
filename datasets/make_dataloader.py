import torch
import random
import numpy as np
import torch.distributed as dist
import torchvision.transforms as T

from .hoss import HOSS
from .pretrain import Pretrain
from .bases import ImageDataset
from torch.utils.data import DataLoader
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from timm.data.random_erasing import RandomErasing
from torch.utils.data.distributed import DistributedSampler



__factory = {
    'HOSS': HOSS,
    'Pretrain': Pretrain,
}

def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_collate_fn(batch):
    imgs, pids, camids, viewids , _, img_size = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    img_size = torch.tensor(img_size, dtype=torch.float32)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_size


def train_pair_collate_fn(batch):
    rgb_batch = [i[0] for i in batch]
    sar_batch = [i[1] for i in batch]
    batch = rgb_batch + sar_batch
    imgs, pids, camids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, img_size = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    img_size = torch.tensor(img_size, dtype=torch.float32)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths, img_size


def val_pair_collate_fn(batch):
    batch_img1 = []
    batch_img2 = []
    for item in batch:
        batch_img1.append(item[0])
        batch_img2.append(item[1])
    
    imgs1, pids1, camids1, _, img_size1 = zip(*batch_img1)
    imgs2, _, camids2, _, img_size2 = zip(*batch_img2)

    pids = torch.tensor(pids1, dtype=torch.int64)
    camids1 = torch.tensor(camids1, dtype=torch.int64)
    camids2 = torch.tensor(camids2, dtype=torch.int64)
    img_size1 = torch.tensor(img_size1, dtype=torch.float32)
    img_size2 = torch.tensor(img_size2, dtype=torch.float32)

    return (
        (torch.stack(imgs1, dim=0), pids, camids1, _, img_size1),
        (torch.stack(imgs2, dim=0), pids, camids2, _, img_size2),
    )


def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, eval_mode=cfg.DATASETS.EVAL_MODE)

    train_set = ImageDataset(dataset.train, train_transforms)
    val_set = ImageDataset(dataset.query_val + dataset.gallery_val, val_transforms)
    train_set_pair = ImageDataset(dataset.train_pair, train_transforms, pair=True)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError('cfg.SOLVER.IMS_PER_BATCH should be even number')
    train_loader_pair = DataLoader(
        train_set_pair, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2), shuffle=True, num_workers=num_workers,
        collate_fn=train_pair_collate_fn
    )
    return train_loader, val_loader, len(dataset.query_val), train_loader_pair, test_loader, len(dataset.query), num_classes, cam_num


def make_dataloader_pair(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set_pair = ImageDataset(dataset.train_pair, train_transforms, pair=True)
    num_classes = dataset.num_train_pair_pids
    cam_num = dataset.num_train_pair_cams

    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
    g = torch.Generator()
    g.manual_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        sampler = DistributedSampler(train_set_pair, shuffle=True, seed=int(cfg.SOLVER.SEED), drop_last=False)
        train_loader_pair = DataLoader(
            train_set_pair,
            batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2),
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=train_pair_collate_fn,
            worker_init_fn=_seed_worker,
            generator=g,
            pin_memory=True,
        )
    else:
        train_loader_pair = DataLoader(
            train_set_pair,
            batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2),
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_pair_collate_fn,
            worker_init_fn=_seed_worker,
            generator=g,
            pin_memory=True,
        )
    return train_loader_pair, num_classes, cam_num