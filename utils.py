import torch
import socket
import logging
import torch.distributed as dist

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from shutil import rmtree
from typing import List, Union, Callable, Optional

from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import HWDBDataset, ComponentMaskDataset


def get_dataloader(
    data_pathes: List[str],
    batch_size: int,
    shuffle: bool = True,
    image_size: int = 32,
    num_workers: int = 2,
    ddp: bool = True,
    seed: int = 0,
):
    dataset = ConcatDataset(
        [HWDBDataset(fp, image_size) for fp in data_pathes]
    )

    sampler = DistributedSampler(
        dataset, shuffle=shuffle, seed=seed
    ) if ddp else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
    )
    return dataloader, sampler


def pad_collate_fn(batch: List[dict], n_max: int = 30) -> dict:
    """
    處理每張圖部件數量不同的情況。
    masks 和 component_ids 用零 padding 對齊到 batch 內最大的 N。

    輸出：
        canvas          FloatTensor (B, 1, H, W)
        masks           FloatTensor (B, N_max, H, W)
        component_ids   LongTensor  (B, N_max)   padding = -1
        num_components  LongTensor  (B,)
        valid_mask      BoolTensor  (B, N_max)   True = 真實部件
    """
    canvases = torch.stack([b['canvas'] for b in batch])        # (B, 1, H, W)
    comp_text = tuple(b['component'] for b in batch)            # (B,)

    H, W = batch[0]['masks'].shape[-2:]
    masks_pad = torch.zeros(len(batch), n_max, H, W, dtype=torch.float32)

    for i, b in enumerate(batch):
        n = b['num_components']
        if n > 0:
            masks_pad[i, :n] = b['masks']

    return canvases, comp_text, masks_pad


def get_radical_mask_dataloader(
    data_pathes: List[str],
    batch_size: int,
    alphabet: Optional[str] = None,
    augment: Optional[Callable] = None,
    shuffle: bool = True,
    image_size: int = 32,
    num_workers: int = 2,
    ddp: bool = True,
):

    dataset = ConcatDataset([
        ComponentMaskDataset(fp, image_size, alphabet=alphabet, augment=augment)
        for fp in data_pathes
    ])

    sampler = DistributedSampler(dataset, shuffle=shuffle) if ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
    )
    return dataloader, sampler


def remove_and_create_dir(path: Union[str, Path]) -> None:
    runs = Path(f'runs/{str(path)}')
    history = Path(f'history/{str(path)}')

    history.parent.mkdir(exist_ok=True, parents=True)
    if history.is_dir():
        res = input(f'{history} is already exist. Do you want to remove it? (Y/n) ')
        if res != 'n':
            logging.warning(f'Iterative remove {history}')
            rmtree(history)
    history.mkdir()

    runs.parent.mkdir(exist_ok=True, parents=True)
    if runs.is_dir():
        if res != 'n':
            logging.warning(f'Iterative remove {runs}')
            rmtree(runs)
    runs.mkdir()


def get_alphabet(alphabet: Union[str, Path]) -> list:
    with open(alphabet, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_free_port(begin=8081, end=8090) -> int:
    for port in range(begin, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise Exception('No free port available.')


def get_printed_image(alphabet: str, image_size: int = 32, chans=1) -> torch.Tensor:
    """
    Get the printed image path from the environment variable.

    Returns:
        The path to the printed image.
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * chans, std=[0.5] * chans)
    ])

    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        alphabet = tqdm(alphabet, desc='Loading printed images')

    res = []
    for c in alphabet:
        img = Image.open(f'./dataset/standard/main/{c}.png')
        img = img.convert('L') if chans == 1 else img.convert('RGB')
        res.append(transform(img))
    return torch.stack(res, dim=0)
