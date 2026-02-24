import torch
import rasterio
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset


OPENEARTHMAP_STATS = {
    "rgb": {
        "mean": (125.43796351587088, 129.16948470083307, 115.961803235332),
        "std": (55.21942652778412, 46.23354608543217, 45.04121566551131),
    },
    "sar": {
        "mean": 49.489188712159894,
        "std": 34.91482279047886,
    },
}


class OpenEarthMapSarPairDataset(Dataset):
    def __init__(self, root):
        self.root_path = Path(root)
        mean_rgb = OPENEARTHMAP_STATS["rgb"]["mean"]
        std_rgb = OPENEARTHMAP_STATS["rgb"]["std"]
        self.rgb_mean = (mean_rgb[0] / 255.0, mean_rgb[1] / 255.0, mean_rgb[2] / 255.0)
        self.rgb_std = (std_rgb[0] / 255.0, std_rgb[1] / 255.0, std_rgb[2] / 255.0)
        self.sar_mean = OPENEARTHMAP_STATS["sar"]["mean"]
        self.sar_std = OPENEARTHMAP_STATS["sar"]["std"]
        self.pairs = self._build_pairs()
        self.rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ]
        )

    def _build_pairs(self):
        rgb_dir = self.root_path / "rgb_images"
        sar_dir = self.root_path / "sar_images"
        pairs = []
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
            arr = src.read(1)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected SAR array shape {arr.shape} for path {path}.")
        sar_img = (arr - self.sar_mean) / self.sar_std
        sar_array = np.stack([sar_img, sar_img, sar_img], axis=0)
        sar_tensor = torch.from_numpy(sar_array)
        return sar_tensor

    def __getitem__(self, index):
        rgb_path, sar_path = self.pairs[index]
        rgb_tensor = self._load_rgb(rgb_path)
        sar_tensor = self._load_sar(sar_path)
        return rgb_tensor, sar_tensor


def create_openearthmap_sar_dataloader(
    root,
    batch_size,
    num_workers,
    shuffle,
    ):
    dataset = OpenEarthMapSarPairDataset(root=root)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader

