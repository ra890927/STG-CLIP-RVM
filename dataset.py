import os
import re
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path
from typing import Union, Optional, Callable


class ComponentMaskDataset(Dataset):
    """
    讀取 cache.h5，回傳 (canvas, masks, component_ids, num_components)。

    canvas  : FloatTensor (1, H, W)  [0, 1]
    masks   : FloatTensor (N, H, W)  {0, 1}
    component_ids  : LongTensor (N,)
    num_components : int

    搭配 pad_collate_fn 使用，N 不同的 sample 會被 zero-pad 對齊。
    """

    def __init__(
        self,
        cache_path: str,
        image_size: int = 32,
        alphabet: Optional[str] = None,
        augment: Optional[Callable] = None
    ):
        self.cache_path = f'{cache_path}/cache.h5'
        self.image_size = image_size
        self.augment = augment

        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 只在 __init__ 時開一次來讀 key list，實際讀資料在 __getitem__ 開
        # （多進程時每個 worker 各自開自己的 file handle）
        with h5py.File(self.cache_path, 'r') as f:
            self.keys = sorted(f.keys())

            if alphabet is not None:
                self.keys = list(filter(
                    lambda k: k in alphabet if re.match(r'^[\u4e00-\u9fff]+$', k) else True,
                    self.keys
                ))

        self._file: Optional[h5py.File] = None  # lazy open per worker

    def _get_file(self) -> h5py.File:
        """每個 DataLoader worker 懶開自己的 file handle。"""
        if self._file is None:
            self._file = h5py.File(self.cache_path, 'r', swmr=True)
        return self._file

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        f = self._get_file()
        grp = f[self.keys[idx]]

        # canvas (H, W) uint8 → (1, H, W) float [0, 1]
        canvas = Image.fromarray(grp['canvas'][:])

        masks_np = grp['masks'][:].astype(np.float32)  # (N, H, W)
        mask_pils = [Image.fromarray((m * 255).astype(np.uint8)) for m in masks_np]

        if self.augment is not None:
            canvas, mask_pils = self.augment(canvas, mask_pils)

        # PIL masks → tensor
        masks = torch.stack([
            torch.from_numpy(np.array(m).astype(np.float32) / 255.0)
            for m in mask_pils
        ])  # (N, H, W)

        if self.transforms is not None:
            canvas = self.transforms(canvas)

        if masks.size(1) != self.image_size or masks.size(2) != self.image_size:
            masks = F.interpolate(
                masks.unsqueeze(1), size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze(1)

        return {
            'canvas': canvas,           # (1, H, W)
            'masks': masks,             # (N, H, W)
            'component': grp.attrs['component'],
            'num_components': int(grp.attrs['num_components']),
        }

class HWDBDataset(Dataset):
    def __init__(self, fp: Union[str, Path], image_size) -> None:
        super().__init__()

        self.records = self.get_all_data_from_csv(fp)

        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index) -> tuple:
        image, label = self.records[index]
        img = Image.open(image).convert('L')  # 灰階

        if self.transforms:
            img = self.transforms(img)

        return img, label

    def get_all_data_from_csv(self, fp: Union[str, Path]):
        # labels.csv: image paths are relative to the split directory
        root = str(fp)
        df = pd.read_csv(os.path.join(root, 'labels.csv'), dtype=str, keep_default_na=False)
        return [(os.path.join(root, image), label) for image, label in zip(df['image'], df['label'])]
