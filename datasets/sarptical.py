import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


SARPTICAL_STATS = {
    "rgb": {
        "mean": (111.1114562982645, 111.9334402360704, 107.41995320363722),
        "std": (49.98057666530153, 45.49713528002184, 41.27858080176325),
    },
    "sar": {
        "mean": -10.157133397760928,
        "std": 8.093608981327103,
    },
}


class SarpticalPairDataset(Dataset):
    def __init__(self, root):
        self.root_path = Path(root)
        mean_rgb = SARPTICAL_STATS["rgb"]["mean"]
        std_rgb = SARPTICAL_STATS["rgb"]["std"]
        self.rgb_mean = (mean_rgb[0] / 255.0, mean_rgb[1] / 255.0, mean_rgb[2] / 255.0)
        self.rgb_std = (std_rgb[0] / 255.0, std_rgb[1] / 255.0, std_rgb[2] / 255.0)
        self.sar_mean = SARPTICAL_STATS["sar"]["mean"]
        self.sar_std = SARPTICAL_STATS["sar"]["std"]
        self.pairs = self._build_pairs()
        self.rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ]
        )

    def _build_pairs(self):
        rgb_dir = self.root_path / "rgb"
        sar_dir = self.root_path / "sar"
        pairs = []
        sar_index = {p.stem: p for p in sar_dir.glob("*.mat")}
        for rgb_path in sorted(rgb_dir.glob("*.png")):
            stem = rgb_path.stem
            sar_path = sar_index.get(stem)
            if sar_path is None:
                continue
            pairs.append((rgb_path, sar_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        return self.rgb_transform(img)

    def _load_sar(self, path):
        mat = sio.loadmat(path)
        arr = None
        for v in mat.values():
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                arr = v
                break
        if arr is None:
            raise ValueError(f"No array data found in {path}.")
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
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


def create_sarptical_dataloader(
    root,
    batch_size,
    num_workers,
    shuffle,
    ):
    dataset = SarpticalPairDataset(root=root)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader

