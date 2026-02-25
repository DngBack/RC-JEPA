"""
Dataset builders for RC-JEPA.

Supports ImageNet-100, full ImageNet, and CIFAR-100 (for fast prototyping).
"""

import os

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_transform(img_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class ImageNet100Dataset(Dataset):
    """ImageNet-100 subset (ImageFolder layout). Optionally filter to 100 classes."""

    def __init__(self, root: str, transform=None, class_file: str | None = None):
        self.dataset = datasets.ImageFolder(root, transform=transform)

        if class_file is not None and os.path.exists(class_file):
            with open(class_file) as f:
                keep = {line.strip() for line in f if line.strip()}

            class_to_new = {c: i for i, c in enumerate(sorted(keep))}
            self.dataset.samples = [
                (p, class_to_new[self.dataset.classes[l]])
                for p, l in self.dataset.samples
                if self.dataset.classes[l] in keep
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def build_dataset(
    dataset_name: str,
    data_dir: str,
    img_size: int = 224,
    is_train: bool = True,
    class_file: str | None = None,
) -> Dataset:
    transform = get_transform(img_size, is_train)

    if dataset_name == "cifar100":
        return datasets.CIFAR100(
            root=data_dir, train=is_train, download=True, transform=transform,
        )
    if dataset_name == "imagenet100":
        return ImageNet100Dataset(data_dir, transform=transform, class_file=class_file)
    # default: plain ImageFolder
    return datasets.ImageFolder(data_dir, transform=transform)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
