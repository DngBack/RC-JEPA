"""
ImageNet-C / CIFAR-100-C corrupted dataset loading.

Expected directory layout
-------------------------
root/{corruption_type}/{severity}/{class_folder}/{images}
"""

import os

from torch.utils.data import Dataset
from torchvision import datasets, transforms

CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


class CorruptedDataset(Dataset):

    def __init__(
        self,
        root: str,
        corruption: str = "gaussian_noise",
        severity: int = 1,
        transform=None,
    ):
        if severity < 1 or severity > 5:
            raise ValueError("severity must be 1-5")
        if corruption not in CORRUPTION_TYPES:
            raise ValueError(f"Unknown corruption: {corruption}")

        self.corruption = corruption
        self.severity = severity

        corrupt_dir = os.path.join(root, corruption, str(severity))
        if not os.path.isdir(corrupt_dir):
            corrupt_dir = os.path.join(root, corruption)

        self.dataset = datasets.ImageFolder(corrupt_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def build_corruption_datasets(
    root: str,
    severities: tuple[int, ...] = (1, 2, 3, 4, 5),
    img_size: int = 224,
) -> dict[tuple[str, int], CorruptedDataset]:
    """Build a dict mapping (corruption, severity) -> dataset."""
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    out: dict[tuple[str, int], CorruptedDataset] = {}
    for corruption in CORRUPTION_TYPES:
        for sev in severities:
            try:
                out[(corruption, sev)] = CorruptedDataset(
                    root, corruption, sev, transform,
                )
            except Exception as exc:
                print(f"  skip {corruption}/{sev}: {exc}")
    return out
