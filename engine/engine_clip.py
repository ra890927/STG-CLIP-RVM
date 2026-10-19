import torch
import torch.distributed as dist

from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from argparse import Namespace

from models import build_clip
from utils import get_alphabet, get_dataloader, get_printed_image

from .base import (
    BaseEngine,
    print_pbar,
    sync_dist_processes,
    gather_from_all_processes,
    LinearWarmupLRScheduler,
)


class EngineCLIP(BaseEngine):
    def __init__(
        self,
        args: Namespace,
        device: torch.device = torch.device('cuda'),
        criterion: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        super().__init__(args, device, enable_writer=not args.test)
        self.criterion = criterion

        model = build_clip(
            args.resume,
            img_size=args.image_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
        )
        self.alphabet = get_alphabet(args.alphabet)

        model = model.to(self.device)
        self.model = DDP(model, device_ids=[device])

        self.trn_loader, _ = get_dataloader(
            args.train, args.bs,
            image_size=args.image_size, shuffle=True
        )
        self.sen_loader, _ = get_dataloader(
            args.seen, args.eval_bs,
            image_size=args.image_size, shuffle=False
        )
        self.usn_loader, _ = get_dataloader(
            args.unseen, args.eval_bs,
            image_size=args.image_size, shuffle=False
        )

        self.optim = self.build_optimizer()

        self.lr_scheduler = LinearWarmupLRScheduler(
            optimizer=self.optim,
            total_epochs=args.epochs,
            iters_per_epoch=len(self.trn_loader),
            warmup_epochs=args.warmup_epochs,
            min_anneal_lr=args.min_anneal_lr,
        )

        ref_image = get_printed_image(self.alphabet, self.args.image_size)
        self.ref_image = ref_image.to(self.device)

    def train(self, epoch: int) -> None:
        mask_ratio = self.args.mask if epoch >= self.args.warmup_epochs else 0

        def step(batch):
            data, label = batch
            image = data.to(self.device)
            stdnd_char = self.ref_image[[self.alphabet.index(c) for c in label]]
            return self.model(image, label, stdnd_char, mask_ratio)

        self.train_one_epoch(epoch, self.trn_loader, step, lr_scheduler=self.lr_scheduler)

    @sync_dist_processes
    def eval(self, epoch: int, save_weight: bool = True) -> tuple[float, float]:
        sen_acc = self.eval_recognition(
            'seen', epoch, self.sen_loader,
            topk=self.args.topk, dump_records=False
        )

        usn_acc = self.eval_recognition(
            'unseen', epoch, self.usn_loader, topk=self.args.topk
        )

        self.write_record(f'Epoch: {epoch:3d} | seen: {sen_acc:.8f}, unseen: {usn_acc:.8f}')

        if save_weight:
            self.save_if_best('sen', sen_acc)
            self.save_if_best('usn', usn_acc)

        return sen_acc, usn_acc

    @torch.no_grad()
    def eval_recognition(
        self,
        title: str,
        epoch: int,
        dataloader: DataLoader,
        topk: int = 4,
        dump_records: bool = True,
    ):
        self.model.eval()

        result_records = []
        text_feats = self._get_text_features()

        pbar = tqdm(dataloader) if dist.get_rank() == 0 else dataloader
        for batch_id, (data, label) in enumerate(pbar):
            image = data.to(self.device)

            logits_per_image = self.model.module.compute_sim(image, text_feats)

            probs, indices = torch.topk(logits_per_image.softmax(dim=-1), k=topk)

            # calculate test loss w/o backpropagation
            gt = torch.tensor(
                [self.alphabet.index(c) for c in label], device=self.device
            )
            loss = self.criterion(logits_per_image, gt)

            if self.writer is not None:
                ts = epoch * len(dataloader) + batch_id
                self.writer.add_scalar(f'{title}-loss', loss, ts)

            print_pbar(pbar, f'Eval {title.capitalize()}', loss=float(loss.item()))

            for i, (prob, idx) in enumerate(zip(probs, indices)):
                pred_str = ''.join([self.alphabet[j] for j in idx.tolist()])
                prob_str = ' '.join([f'{p:.2f}' for p in prob.tolist()])

                result_records.append(
                    (label[i], str(label[i] == pred_str[0]), pred_str, prob_str)
                )

        result_records = gather_from_all_processes(result_records)
        result_records = result_records[: len(dataloader.dataset)]

        acc = [item for item in result_records if item[1] == 'True']
        acc = len(acc) / len(dataloader.dataset) * 100

        if dist.get_rank() == 0 and dump_records:
            with open(f'history/{self.args.expr}/result_{epoch:03d}.txt', 'w+', encoding='utf-8') as f:
                f.write('id | label | correct | pred | prob\n')
                for i, (label, correct, pred, prob) in enumerate(result_records):
                    f.write(f'{i:9d} | {label} | {correct:5} | {pred} | {prob}\n')

        return acc

    @torch.no_grad()
    def _get_text_features(self, model=None) -> torch.Tensor:
        model = self.model.module if model is None else model

        model.eval()
        text_features = []
        for i in range((len(self.alphabet) - 1) // 100 + 1):
            s, e = 100 * i, min(100 * (i + 1), len(self.alphabet))
            sub_text_features = model.forward_text(self.alphabet[s:e])
            text_features.append(sub_text_features)
        return torch.cat(text_features, dim=0)
