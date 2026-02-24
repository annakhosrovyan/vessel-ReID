import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


QXS_SAROPT_STATS = {
    "rgb": {
        "mean": (111.45507924041748, 105.52763311920167, 105.3858320968628),
        "std": (51.41432623470519, 43.80156869568528, 42.21833938783467),
    },
    "sar": {
        "mean": 69.03371307067871,
        "std": 60.15848431521505,
    },
}


class QxsSaroptPairDataset(Dataset):
    def __init__(self, root):
        self.root_path = Path(root)
        mean_rgb = QXS_SAROPT_STATS["rgb"]["mean"]
        std_rgb = QXS_SAROPT_STATS["rgb"]["std"]
        self.rgb_mean = (mean_rgb[0] / 255.0, mean_rgb[1] / 255.0, mean_rgb[2] / 255.0)
        self.rgb_std = (std_rgb[0] / 255.0, std_rgb[1] / 255.0, std_rgb[2] / 255.0)
        self.sar_mean = QXS_SAROPT_STATS["sar"]["mean"]
        self.sar_std = QXS_SAROPT_STATS["sar"]["std"]
        self.pairs = self._build_pairs()
        self.rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ]
        )

    def _build_pairs(self):
        rgb_dir = self.root_path / "opt_256_oc_0.2"
        sar_dir = self.root_path / "sar_256_oc_0.2"
        pairs = []
        sar_index = {p.name: p for p in sar_dir.glob("*.png")}
        for rgb_path in sorted(rgb_dir.glob("*.png")):
            sar_path = sar_index.get(rgb_path.name)
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
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
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


def create_qxs_saropt_dataloader(
    root,
    batch_size,
    num_workers,
    shuffle,
    ):
    dataset = QxsSaroptPairDataset(root=root)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader

