from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR100, ImageFolder


class CommonDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.train_dataset, self.val_dataset, self.test_dataset = [None] * 3
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers=4)


class MNISTDataModule(CommonDataModule):
    def __init__(self, data_dir: str = 'data/mnist', val_split: float = 0.2, batch_size: int = 128):
        super().__init__(batch_size=batch_size)

        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.val_split = val_split

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        dataset = MNIST(self.data_dir, train=True, transform=self.transform, download=False)
        self.test_dataset = MNIST(self.data_dir, train=False, transform=self.transform, download=False)

        val_size = int(len(dataset) * self.val_split)
        self.train_dataset, self.val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])


class CIFAR100DataModule(CommonDataModule):
    def __init__(self, data_dir: str = 'data/cifar100', val_split: float = 0.2, batch_size: int = 128):
        super().__init__(batch_size=batch_size)

        self.dims = (3, 32, 32)
        self.data_dir = data_dir
        self.val_split = val_split

        self.train_dataset, self.val_dataset, self.test_dataset = [None] * 3
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

    def prepare_data(self):
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        dataset = CIFAR100(self.data_dir, train=True, transform=self.transform, download=False)
        self.test_dataset = CIFAR100(self.data_dir, train=False, transform=self.transform, download=False)

        val_size = int(len(dataset) * self.val_split)
        self.train_dataset, self.val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])


class TinyImagenetTestDataset(Dataset):
    def __init__(self, data_path: str, transform: transforms.Compose):
        self.image_paths = list(Path(data_path).glob('*'))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(str(self.image_paths[idx])).convert('RGB')
        return self.transform(image)


class TinyImageNetDataModule(CommonDataModule):
    def __init__(self, data_dir: str = 'data/tiny_imagenet', val_split: float = 0.2, batch_size: int = 128):
        super().__init__(batch_size=batch_size)

        self.data_dir = data_dir
        self.val_split = val_split

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.test_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = ImageFolder(str(Path(self.data_dir, 'train')), transform=self.train_transforms)
        self.test_dataset = TinyImagenetTestDataset(str(Path(self.data_dir, 'test', 'images')), self.test_transforms)

        val_size = int(len(dataset) * self.val_split)
        self.train_dataset, self.val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
