import time
import torch
import random
import logging

import torch.distributed as dist

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tqdm import tqdm
from argparse import Namespace
from typing import Any, Callable, Optional
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class LinearWarmupLRScheduler(torch.optim.lr_scheduler.SequentialLR):
    def __init__(
        self,
        optimizer,
        total_epochs: int,
        iters_per_epoch: int,
        warmup_epochs: int = 3,
        start_factor: float = 1e-2,
        min_anneal_lr: float = 1e-6,
    ):
        warmup_steps = warmup_epochs * iters_per_epoch
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, total_iters=warmup_steps
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, eta_min=min_anneal_lr,
            T_max=(total_epochs - warmup_epochs) * iters_per_epoch,
        )

        super().__init__(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )


class PrintedCharGeometricAugment:
    """
    Joint geometric augmentation for printed character image + component binary masks.

    Args:
        img   : PIL Image (H, W, 3)
        masks : list of PIL Images, each binary mask for one component

    Returns:
        aug_img   : PIL Image
        aug_masks : list of PIL Images
    """

    def __init__(
        self,
        degrees: tuple[float, float] = (-10, 10),
        translate: tuple[float, float] = (0.4, 0.4),
        scale_range: tuple[float, float] = (0.7, 1.3),
        p: float = 0.8,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.p = p

        self._transform = T.RandomAffine(
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale_range,
        )

    def __call__(self, img, masks: list):
        if random.random() > self.p:
            return img, masks

        params = self._transform.get_params(
            self._transform.degrees,
            self._transform.translate,
            self._transform.scale,
            None,  # no need for shear
            img.size,  # (W, H)
        )

        aug_img = TF.affine(img, *params, fill=255, interpolation=TF.InterpolationMode.BILINEAR)
        aug_masks = [
            TF.affine(m, *params, fill=0, interpolation=TF.InterpolationMode.NEAREST)
            for m in masks
        ]

        return aug_img, aug_masks


def print_pbar(pbar: Any, des: str = '', **kloss: float) -> None:
    if not isinstance(pbar, tqdm):
        return
    pbar.set_description(des, refresh=False)
    losses = {name: round(loss, 4) for name, loss in kloss.items()}
    if len(losses):
        pbar.set_postfix(losses, refresh=True)
    pbar.refresh()


def sync_dist_processes(func: Callable) -> Callable:
    def warpper(*args, **kwargs):
        torch.cuda.empty_cache()
        ret = func(*args, **kwargs)
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier(device_ids=[dist.get_rank()])
        return ret
    return warpper


def gather_from_all_processes(data: Any) -> list[Any]:
    gathered_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_list, data)

    gathered_result_records = []
    for i in range(len(data)):
        for j in range(dist.get_world_size()):
            gathered_result_records.append(gathered_list[j][i])

    return gathered_result_records


class BaseEngine:
    """
    Shared training scaffolding: tensorboard writer, epoch loop, optimizer,
    records.txt and best-checkpoint bookkeeping.  Subclasses build the model
    and data, and implement ``train`` and ``eval``.
    """

    def __init__(self, args: Namespace, device: torch.device, enable_writer: bool = True) -> None:
        self.args = args
        self.device = device
        self.best = {}

        self.writer = None
        if enable_writer and dist.get_rank() == 0:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(
                log_dir=f'runs/{self.args.expr}/run_{run_id}'
            )

    def __del__(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def __call__(self) -> None:
        for epoch in range(self.args.epochs):
            self.train(epoch)
            self.eval(epoch)

    def train(self, epoch: int) -> None:
        raise NotImplementedError

    def eval(self, epoch: int, save_weight: bool = True):
        raise NotImplementedError

    def build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr,
            weight_decay=self.args.weight_decay, betas=tuple(self.args.betas), eps=self.args.eps
        )

    @sync_dist_processes
    def train_one_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        step: Callable[[Any], Any],
        tag: str = '',
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        """Run one epoch; ``step`` maps a batch to a model output with ``.loss``."""
        self.model.train()
        dataloader.sampler.set_epoch(epoch)

        pbar = tqdm(dataloader) if dist.get_rank() == 0 else dataloader
        for batch_id, batch in enumerate(pbar):
            output = step(batch)

            self.optim.zero_grad()
            output.loss.backward()
            self.optim.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if self.writer is not None:
                # x-axis: samples seen, in units of eval_bs
                effective_batch_size = dist.get_world_size() * dataloader.batch_size
                ts = (
                    epoch * len(dataloader.dataset) +
                    batch_id * effective_batch_size
                ) // self.args.eval_bs

                for name, val in output.items():
                    self.writer.add_scalar(f'{tag}{name}', val.item(), ts)

            print_pbar(
                pbar, f'Epoch: {epoch}',
                **dict({k: v.item() for k, v in output.items()}),
            )

    def write_record(self, line: str) -> None:
        if dist.get_rank() == 0:
            with open(f'history/{self.args.expr}/records.txt', 'a+', encoding='utf-8') as f:
                f.write(line + '\n')

    def save_if_best(self, name: str, value: float, higher_is_better: bool = True) -> None:
        if dist.get_rank() != 0:
            return

        best = self.best.get(name)
        if best is None or (value > best if higher_is_better else value < best):
            logging.info(f'Saving {name} best weight')
            self.best[name] = value
            torch.save(
                self.model.module.state_dict(),
                f'history/{self.args.expr}/{name}-best.pth'
            )
