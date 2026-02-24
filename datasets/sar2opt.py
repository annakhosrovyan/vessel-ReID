import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


SAR2OPT_STATS = {
    "rgb": {
        "mean": (93.65373660978202, 106.8238501669424, 100.4524427697701),
        "std": (46.332444616495174, 43.8242600215277, 46.93036048445625),
    },
    "sar": {
        "mean": 105.84093485749412,
        "std": 56.14827957746546,
    },
}


class Sar2OptPairDataset(Dataset):
    def __init__(self, root):
        self.root_path = Path(root)
        mean_rgb = SAR2OPT_STATS["rgb"]["mean"]
        std_rgb = SAR2OPT_STATS["rgb"]["std"]
        self.rgb_mean = (mean_rgb[0] / 255.0, mean_rgb[1] / 255.0, mean_rgb[2] / 255.0)
        self.rgb_std = (std_rgb[0] / 255.0, std_rgb[1] / 255.0, std_rgb[2] / 255.0)
        self.sar_mean = SAR2OPT_STATS["sar"]["mean"]
        self.sar_std = SAR2OPT_STATS["sar"]["std"]
        self.pairs = self._build_pairs()
        self.rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ]
        )

    def _build_pairs(self):
        sar_dir = self.root_path / "A"
        rgb_dir = self.root_path / "B"
        pairs = []
        sar_index = {p.name: p for p in sar_dir.glob("*.jpg")}
        for rgb_path in sorted(rgb_dir.glob("*.jpg")):
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
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
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


def create_sar2opt_dataloader(
    root,
    batch_size,
    num_workers,
    shuffle,
    ):
    dataset = Sar2OptPairDataset(root=root)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader

