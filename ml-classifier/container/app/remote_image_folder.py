import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models

import fsspec
from pelicanfs.core import PelicanFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.cached import WholeFileCacheFileSystem

from PIL import Image

class RemoteImageFolder(VisionDataset):

    def __init__(self, root, fs = LocalFileSystem, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.fs = fs
        if os.path.isdir(root):
            self._init_local(root)
        else:
            self._init_remote(root)

    def _init_local(self, root):
        self.root = root
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.imgs = self._make_dataset_local()

    def _init_remote(self, root, transform=None):
        self.root = root
        self.classes = sorted([item['name'] for item in self.fs.ls(root)])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.imgs = self._make_dataset_remote()

    def _make_dataset_local(self):
        images = []
        for class_idx, cls_name in enumerate(self.classes):
            class_path = os.path.join(self.root, cls_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith('.jpg') or img_name.lower().endswith('.jpeg') or img_name.lower().endswith('.png'):
                    img_path = os.path.join(class_path, img_name)
                    images.append((img_path, class_idx))
        return images

    def _make_dataset_remote(self):
        images = []
        for class_idx, cls_name in enumerate(self.classes):
            class_path = os.path.join(self.root, cls_name)
            files = self.fs.ls(class_path)
            for item in files:
                img_path = item['name']
                if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg') or img_path.lower().endswith('.png'):
                    images.append((img_path, class_idx))
        return images

    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        if isinstance(self.fs, PelicanFileSystem) or isinstance(self.fs, WholeFileCacheFileSystem):
            with self.fs.open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        else:
            img = read_image(img_path)
            img = transforms.ToPILImage()(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
