import torch
import rasterio
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset


MSAW_STATS = {
    "rgb": {
        "mean": (61.94295608236733, 65.51181709202064, 59.61640723990328),
        "std": (59.43658133337928, 58.38572155496566, 56.148644066179514),
    },
    "sar": {
        "mean": (22.283795357861944, 17.287254059850007),
        "std": (17.42711587515321, 14.325639331364542),
    },
}


class MsawPairDataset(Dataset):
    def __init__(self, root):
        self.root_path = Path(root)
        mean_rgb = MSAW_STATS["rgb"]["mean"]
        std_rgb = MSAW_STATS["rgb"]["std"]
        self.rgb_mean = (mean_rgb[0] / 255.0, mean_rgb[1] / 255.0, mean_rgb[2] / 255.0)
        self.rgb_std = (std_rgb[0] / 255.0, std_rgb[1] / 255.0, std_rgb[2] / 255.0)
        self.pairs = self._build_pairs()

        self.rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ]
        )

    def _build_pairs(self):
        pairs = []
        for aoi_dir in sorted(self.root_path.iterdir()):
            if not aoi_dir.is_dir():
                continue
            rgb_dir = aoi_dir / "PS-RGB"
            sar_dir = aoi_dir / "SAR-Intensity"
            if not rgb_dir.is_dir() or not sar_dir.is_dir():
                continue
            sar_index = {p.name: p for p in sar_dir.glob("*.tif")}
            for rgb_path in sorted(rgb_dir.glob("*.tif")):
                sar_path = sar_index.get(rgb_path.name)
                if sar_path is None:
                    continue
                pairs.append((rgb_path, sar_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, path):
        with rasterio.open(path) as src:
            arr = src.read()
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"Unexpected RGB array shape {arr.shape} for path {path}.")
        arr = np.transpose(arr, (1, 2, 0))
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        return self.rgb_transform(img)

    def _load_sar(self, path):
        with rasterio.open(path) as src:
            arr = src.read()
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] < 2:
            raise ValueError(f"Unexpected SAR array shape {arr.shape} for path {path}.")
        vh_img = arr[0]
        vv_img = arr[1]
        vh_mean, vv_mean = MSAW_STATS["sar"]["mean"]
        vh_std, vv_std = MSAW_STATS["sar"]["std"]
        vh_img = (vh_img - vh_mean) / vh_std
        vv_img = (vv_img - vv_mean) / vv_std
        h, w = vh_img.shape
        sar_array = np.zeros((3, h, w), dtype=np.float32)
        sar_array[0] = vh_img
        sar_array[1] = vv_img
        sar_tensor = torch.from_numpy(sar_array)
        return sar_tensor

    def __getitem__(self, index):
        rgb_path, sar_path = self.pairs[index]
        rgb_tensor = self._load_rgb(rgb_path)
        sar_tensor = self._load_sar(sar_path)
        return rgb_tensor, sar_tensor


def create_msaw_dataloader(
    root,
    batch_size,
    num_workers,
    shuffle,
    ):
    dataset = MsawPairDataset(root=root)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader

