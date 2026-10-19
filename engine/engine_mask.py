import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from argparse import Namespace

from models import build_rvm
from utils import get_alphabet, get_dataloader, get_radical_mask_dataloader

from .base import (
    BaseEngine,
    print_pbar,
    sync_dist_processes,
    gather_from_all_processes,
    PrintedCharGeometricAugment,
)


class EngineMaskPretrain(BaseEngine):
    def __init__(
        self,
        args: Namespace,
        device: torch.device = torch.device('cuda'),
    ) -> None:
        super().__init__(args, device, enable_writer=not args.test)

        self.alphabet = get_alphabet(args.alphabet)

        model = build_rvm(args.resume, image_size=args.image_size)
        self.model = DDP(model.to(self.device), device_ids=[device], find_unused_parameters=True)

        # pretrain batches use eval_bs on purpose: printed radical masks need more memory
        self.char_mask_loader, _ = get_radical_mask_dataloader(
            args.char_mask, args.eval_bs, alphabet=self.alphabet,
            augment=PrintedCharGeometricAugment(),
            image_size=args.image_size, shuffle=True,
        )
        self.char_mask_evl_loader, _ = get_radical_mask_dataloader(
            args.char_mask, args.eval_bs, alphabet=self.alphabet, augment=None,
            image_size=args.image_size, shuffle=False,
        )

        self.optim = self.build_optimizer()

    def train(self, epoch: int) -> None:
        self.train_pretrain(epoch)

    def train_pretrain(self, epoch: int) -> None:
        def step(batch):
            char_data, char_label, char_radical_mask = batch
            return self.model(
                char_data.to(self.device), char_label, char_radical_mask.to(self.device),
                mode='pretrain'
            )

        self.train_one_epoch(epoch, self.char_mask_loader, step, tag='pretrain/')

    @sync_dist_processes
    def eval(self, epoch: int, save_weight: bool = True) -> float:
        loss_char_msk = self.eval_character_radical_mask(epoch)

        records = self.collect_radical_scores(self.char_mask_evl_loader, 'Eval Char Radical Logits')
        if dist.get_rank() == 0:
            with open(f'history/{self.args.expr}/char_radical_logits_{epoch:03}.txt', 'w+', encoding='utf-8') as f:
                f.write('char | mean_mask_logit | radical_tokens | radical_logits\n')
                for label, presence, _, _, tokens, logits in records:
                    f.write(f'{label} | {presence:.4f} | {tokens} | {logits}\n')

        self.write_record(f'Epoch: {epoch:3d} | char_mask: {loss_char_msk:.8f}')

        if save_weight:
            self.save_if_best('usn', loss_char_msk, higher_is_better=False)

        return loss_char_msk

    @torch.no_grad()
    def eval_character_radical_mask(self, epoch: int) -> float:
        self.model.eval()

        loss_total = 0.0
        loader = self.char_mask_evl_loader
        pbar = tqdm(loader) if dist.get_rank() == 0 else loader

        for batch_id, (char_data, char_label, char_radical_mask) in enumerate(pbar):
            output = self.model.module.forward_pretrain_radical_mask(
                char_data.to(self.device), char_label, char_radical_mask.to(self.device)
            )
            loss_total += output.loss.detach().item()

            if self.writer is not None:
                ts = epoch * len(loader) + batch_id
                for name, val in output.items():
                    self.writer.add_scalar(f'eval-char-mask-{name}', val.item(), ts)

            print_pbar(pbar, 'Eval Char Mask', loss=float(output.loss.detach().item()))

        loss_tensor = torch.tensor(loss_total, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        return loss_tensor.item() / len(loader.dataset)

    @torch.no_grad()
    def collect_radical_scores(self, dataloader: DataLoader, desc: str) -> list[tuple]:
        """
        Score every image against its own label, gathered from all ranks in
        dataset order: (label, presence, coverage, score, radical_tokens, radical_logits).
        """
        self.model.eval()
        rvm = self.model.module

        records = []
        pbar = tqdm(dataloader) if dist.get_rank() == 0 else dataloader
        for batch in pbar:
            image, label = batch[0].to(self.device), batch[1]
            scores = rvm.score(image, label)

            input_ids = scores.input_ids.cpu()
            token_mask = scores.token_mask.cpu()
            token_scores = scores.token_scores.cpu()
            for i in range(len(label)):
                radical_text = rvm.tokenizer.convert_ids_to_tokens(input_ids[i][token_mask[i]])
                radical_logits = token_scores[i][token_mask[i]].tolist()

                records.append((
                    label[i],
                    round(scores.presence[i].item(), 4),
                    round(scores.coverage[i].item(), 4),
                    round(scores.score[i].item(), 4),
                    ' '.join(radical_text),
                    ' '.join(str(round(x, 4)) for x in radical_logits)
                ))

            print_pbar(pbar, desc)

        records = gather_from_all_processes(records)
        return records[: len(dataloader.dataset)]


class EngineMaskFinetune(EngineMaskPretrain):
    def __init__(
        self,
        args: Namespace,
        device: torch.device = torch.device('cuda'),
    ) -> None:
        super().__init__(args, device)

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

    def train(self, epoch: int) -> None:
        self.train_pretrain(epoch)
        self.train_finetune(epoch)

    def train_finetune(self, epoch: int) -> None:
        def step(batch):
            data, label = batch
            return self.model(data.to(self.device), label, mode='finetune')

        self.train_one_epoch(epoch, self.trn_loader, step, tag='finetune/')

    @sync_dist_processes
    def eval(self, epoch: int, save_weight: bool = True) -> tuple[float, float]:
        sen_logits = self.eval_handwritten_character_radical_logits(
            'seen', epoch, self.sen_loader, dump_records=False
        )

        usn_logits = self.eval_handwritten_character_radical_logits(
            'unseen', epoch, self.usn_loader, dump_records=True
        )

        self.write_record(f'Epoch: {epoch:3d} | seen: {sen_logits:.8f} | unseen: {usn_logits:.8f}')

        if save_weight:
            self.save_if_best('usn', usn_logits)

        return sen_logits, usn_logits

    @torch.no_grad()
    def eval_handwritten_character_radical_logits(
        self,
        title: str,
        epoch: int,
        dataloader: DataLoader,
        dump_records: bool = True
    ) -> float:
        records = self.collect_radical_scores(
            dataloader, f'Eval {title.capitalize()} Char Radical Logits'
        )

        mean_score = sum([item[3] for item in records]) / len(records) * 100

        if dist.get_rank() == 0 and dump_records:
            with open(f'history/{self.args.expr}/char_radical_logits_{epoch:03d}.txt', 'w+', encoding='utf-8') as f:
                f.write('char | mean_mask_logit | mask_ratio | radical_tokens | radical_logits\n')
                for i, (label, _, coverage, score, tokens, logits) in enumerate(records):
                    f.write(f'{i:9d} | {label} | {score:.4f} | {coverage:.4f} | {tokens} | {logits}\n')

        return mean_score
