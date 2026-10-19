import json
import torch
import torch.nn.functional as F

from transformers import BertTokenizer


def build_tokenizer(config_path: str = 'configs/tokenizer.json') -> 'ChineseCharacterTokenizer':
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(config['unit_alphabet'], "r", encoding="utf-8") as f:
        never_split_set = [line.strip() for line in f.readlines() if len(line) > 3]

    tokenizer = ChineseCharacterTokenizer(
        radical_decompose=config['radical_decompose'], geometric_mask=config['geometric_mask'],
        vocab_file=config['unit_alphabet'], never_split=never_split_set, do_lower_case=False
    )

    return tokenizer


class ChineseCharacterTokenizer(BertTokenizer):
    LAYOUT = ['⿰', '⿱', '⿴', '⿵', '⿷', '⿶', '⿸', '⿹', '⿺', '⿻', '⿲', '⿳']

    def __init__(
        self,
        radical_decompose: str,
        geometric_mask: str,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        with open(radical_decompose, 'r', encoding='utf-8') as f:
            self.radical_decompose = {}
            for line in f.readlines():
                c, decomp = line.strip().split(':')
                self.radical_decompose[c] = decomp

        data = torch.load(geometric_mask, weights_only=True)
        self.alphabet = data['alphabet']
        self.geometric_mask = torch.cat(
            [data['geometric_mask'], torch.zeros(1, *data['geometric_mask'].size()[1:])], dim=0
        )

    def get_decompose_string(self, text):
        return ([
            t if t not in self.radical_decompose else self.radical_decompose[t]
            for t in text
        ])

    def parse_decompose_string(self, decomp):
        pos = [0]
        results = [None] * len(decomp)

        def parse(depth=0, sibling_idx=0):
            cur_pos = pos[0]
            pos[0] += 1
            t = decomp[cur_pos]

            if t in self.LAYOUT:
                num_child = 3 if t in ['⿲', '⿳'] else 2
                for i in range(num_child):
                    parse(depth + 1, i)

            results[cur_pos] = {
                'depth': depth,
                'sibling_idx': sibling_idx,
                'is_layout': t in self.LAYOUT,
            }

        parse()

        return results

    def get_decompose_additional_features(self, decomp, max_len: int = 30):
        ids_depth, ids_sibling = [], []
        for decomp_item in decomp:
            parsed_tree = self.parse_decompose_string(decomp_item.split())

            depth_ = torch.tensor([item['depth'] for item in parsed_tree])
            sibling_ = torch.tensor([item['sibling_idx'] for item in parsed_tree])

            pad_size = max_len - depth_.size(0) - 1
            depth_add_special_token = F.pad(depth_, (1, pad_size), value=-1)
            sibling_add_special_token = F.pad(sibling_, (1, pad_size), value=-1)

            ids_depth.append(depth_add_special_token)
            ids_sibling.append(sibling_add_special_token)

        return torch.stack(ids_depth, dim=0).long(), torch.stack(ids_sibling, dim=0).long()

    def __call__(self, text, **kwargs):
        radical_decomp = self.get_decompose_string(text)

        res = super().__call__(radical_decomp, **kwargs)

        if all([len(c) == 1 for c in text]):
            text_indices = torch.tensor(
                [self.alphabet.find(c) for c in text], dtype=torch.long
            )
            # character without geometric mask
            text_indices[text_indices == -1] = len(self.alphabet)
            ids_mask = self.geometric_mask[text_indices]

            max_len = kwargs.get('max_length', 30)
            ids_depth, ids_sibling = self.get_decompose_additional_features(
                radical_decomp, max_len
            )

            res.update({
                'token_ids_mask': ids_mask,
                'token_ids_depth': ids_depth,
                'token_ids_sibling': ids_sibling
            })

        return res

    def get_layout_mask(self, input_ids):
        layout_ids = self.convert_tokens_to_ids(self.LAYOUT)
        layout_ids = torch.tensor(layout_ids).to(input_ids.device)
        res = torch.zeros_like(input_ids)
        return res.masked_fill(torch.isin(input_ids, layout_ids), 1).bool()
