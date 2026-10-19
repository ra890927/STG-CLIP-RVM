import os
import torch

from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils import get_alphabet, get_dataloader
from models import build_clip, build_rvm
from engine.base import print_pbar


class Evaluator:
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.clip = build_clip(args.clip).cuda()
        self.rvm = build_rvm(args.rvm).cuda()

        self.alphabet = get_alphabet(args.alphabet)

        self.clip_dataloader, _ = get_dataloader(
            args.dataset, args.bs, image_size=32, shuffle=False, ddp=False
        )

        self.rvm_dataloader, _ = get_dataloader(
            args.dataset, args.bs, image_size=64, shuffle=False, ddp=False
        )

    def __call__(self):
        clip_records = self.eval_global_recognition()
        rvm_records = self.eval_local_recognition(clip_records)

        num_gated = sum([int(prob[0] - prob[1] <= self.args.delta) for _, _, prob in clip_records])
        print(f'Gated Samples (delta={self.args.delta}): {num_gated / len(clip_records) * 100:.2f}%')

        acc = self.compute_accuracy_by_threshold(
            clip_records, rvm_records, self.args.delta
        )
        print(f'Pipeline Accuracy: {acc:.2f}%')

    def compute_accuracy_by_threshold(self, clip_records, rvm_records, threshold):
        acc = 0
        for clip_res, rvm_res in zip(clip_records, rvm_records):
            label, clip_pred, clip_prob = clip_res
            if clip_prob[0] - clip_prob[1] <= threshold:
                acc += int(label == rvm_res[0])
            else:
                acc += int(label == clip_pred[0])

        return acc / len(clip_records) * 100

    @torch.no_grad()
    def eval_global_recognition(self):
        self.clip.eval()

        result_records = []
        text_feats = self._get_text_features()

        for batch_id, (data, label) in enumerate(pbar := tqdm(self.clip_dataloader)):
            image = data.cuda()

            logits_per_image = self.clip.compute_sim(image, text_feats)

            probs, indices = torch.topk(logits_per_image.softmax(dim=-1), k=self.args.topk)

            print_pbar(pbar, 'CLIP Stage Eval')

            for i, (prob, idx) in enumerate(zip(probs, indices)):
                pred_str = ''.join([self.alphabet[j] for j in idx.tolist()])
                result_records.append((label[i], pred_str, prob.tolist()))

        acc = sum([int(item[0] == item[1][0]) for item in result_records])
        print(f'CLIP Stage Accuracy: {acc / len(result_records) * 100:.2f}%')

        return result_records

    @torch.no_grad()
    def eval_local_recognition(self, clip_records):
        self.rvm.eval()

        result_records = []
        for batch_id, (data, label) in enumerate(pbar := tqdm(self.rvm_dataloader)):
            image = data.cuda()

            chunk = clip_records[batch_id * self.args.bs: (batch_id + 1) * self.args.bs]
            topk_text = ''.join([item[1] for i, item in enumerate(chunk)])
            topk_text = [c for c in topk_text]

            scores = self.rvm.score(image, topk_text).score.view(-1, self.args.topk)
            indices = torch.argsort(scores, dim=-1, descending=True)

            char_indices = torch.tensor(
                [self.alphabet.index(c) for c in topk_text], device=torch.device('cuda')
            ).view(-1, self.args.topk)
            char_indices = torch.gather(char_indices, dim=1, index=indices)

            for idx in char_indices:
                pred_str = ''.join([self.alphabet[i] for i in idx.tolist()])
                result_records.append(pred_str)

            print_pbar(pbar, 'RVM Stage Eval')

        return result_records

    @torch.no_grad()
    def _get_text_features(self) -> torch.Tensor:
        self.clip.eval()
        text_features = []
        for i in range((len(self.alphabet) - 1) // 100 + 1):
            s, e = 100 * i, min(100 * (i + 1), len(self.alphabet))
            sub_text_features = self.clip.forward_text(self.alphabet[s:e])
            text_features.append(sub_text_features)
        return torch.cat(text_features, dim=0)


def parse_argument() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--clip', type=str, required=True)
    parser.add_argument('--rvm', type=str, required=True)

    parser.add_argument('--dataset', action='append', type=str)
    parser.add_argument('--alphabet', type=str, default='./data/char_3755.txt')

    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--topk', default=2, type=int)
    parser.add_argument('--delta', default=0.08, type=float,
                        help='margin gating: run RVM when top-1 minus top-2 CLIP prob <= delta')
    parser.add_argument('--gpus', type=str, default='0')

    args = parser.parse_args()

    if args.dataset == None or len(args.dataset) == 0:
        args.dataset = ['./dataset/ICDAR/unseen']

    return args


def main():
    args = parse_argument()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    evaluator = Evaluator(args)
    evaluator()


if __name__ == '__main__':
    main()
