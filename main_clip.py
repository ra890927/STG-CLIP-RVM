import os
import logging
import traceback

import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pathlib import Path
from argparse import ArgumentParser, Namespace

from engine import EngineCLIP
from utils import remove_and_create_dir, get_free_port


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--expr', type=str, required=True)

    # model arguments
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=2048)
    parser.add_argument('--mask', type=float, default=0.5)

    # hyperparameters
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--gpus', type=str, default='0')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.98])
    parser.add_argument('--eps', type=float, default=1e-6)

    # learning rate scheduler
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--min_anneal_lr', type=float, default=1e-6)

    # evaluation related
    parser.add_argument('--resume', type=str)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--test', action='store_true', default=False)

    # dataset
    parser.add_argument('--train', action='append', type=str)
    parser.add_argument('--seen', action='append', type=str)
    parser.add_argument('--unseen', action='append', type=str)

    parser.add_argument('--alphabet', type=str, default='./data/char_3755.txt')

    args = parser.parse_args()

    # default processing
    if args.train == None or len(args.train) == 0:
        args.train = ['./dataset/HWDB0/seen', './dataset/HWDB1/seen']
    if args.seen == None or len(args.seen) == 0:
        args.seen = ['./dataset/ICDAR/seen']
    if args.unseen == None or len(args.unseen) == 0:
        args.unseen = ['./dataset/ICDAR/unseen']

    return args


def save_args(args: Namespace) -> None:
    with open(Path('./history') / args.expr / 'args.json', 'w+', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)


def main(
    rank: torch.device,
    world_size: int,
    port: int,
    args: Namespace
) -> None:
    logging.info(f'Initializing process group rank {rank}...')

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{port}',
        world_size=world_size,
        device_id=device,
        rank=rank
    )

    engine = EngineCLIP(args=args, device=rank)

    try:
        if args.test:
            sen, usn = engine.eval(0, save_weight=False)
            if rank == 0:
                print(f'Seen: {sen}, Unseen: {usn}')
        else:
            engine()
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt')
    except:
        traceback.print_exc()
    finally:
        logging.warning(f'Process {rank} is shutting down...')

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_argument()

    print(json.dumps(vars(args), indent=2))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    remove_and_create_dir(args.expr)

    save_args(args)

    logging.info('Initializing...')

    port = get_free_port()
    world_size = torch.cuda.device_count()

    logging.info(f'Backend: tcp://localhost:{port}')

    try:
        mp.spawn(
            main,
            args=(world_size, port, args),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        if mp.get_start_method() == 'spawn':
            mp.current_process()._clean()
