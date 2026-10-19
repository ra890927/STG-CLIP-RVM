import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss
from transformers.modeling_outputs import ModelOutput

from random import sample
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from .tokenizer import build_tokenizer
from .encoders import ResNet, BertConfig, BertAttention, BertIntermediate, BertOutput


class Block(nn.Module):
    def __init__(self, config: BertConfig, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention

        self.attention = BertAttention(
            config, is_cross_attention=is_cross_attention
        )

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # ---------- Attention ----------
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        # ---------- FFN ----------
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, attention_outputs[1] if output_attentions else None


class Decoder(nn.Module):
    def __init__(self, config: BertConfig, image_size: int = 64, num_layers: int = 6):
        super().__init__()

        self.config = config
        self.image_size = image_size

        self.norm = nn.LayerNorm(config.hidden_size)

        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.layers = nn.ModuleList([
            Block(config, is_cross_attention=True)
            for _ in range(num_layers)
        ])

        self.match_head = nn.Linear(config.hidden_size, 1)
        self.mask_head = nn.Linear(
            config.hidden_size, image_size * image_size
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, query_tokens, encoder_hidden_states, output_attentions=False):
        hidden_states = self.norm(self.query_proj(query_tokens))

        attn_weights = []
        for layer in self.layers:
            hidden_states, attn_weight = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attn_weights.append(attn_weight)

        predict_match = self.match_head(hidden_states).squeeze(-1)
        predict_mask = self.mask_head(hidden_states).view(
            *hidden_states.size()[:-1], self.image_size, self.image_size
        )

        return predict_match, predict_mask, attn_weights


@dataclass
class RVMConfig:
    max_txt_len: int = 30
    num_decoder_layers: int = 6
    finetune_neg_ratio: int = 2
    pretrain_neg_ratio: int = 3
    rtm_weight: float = 1.0
    focal_weight: float = 5.0
    dice_weight: float = 1.0
    focal_alpha: float = 0.85
    focal_gamma: float = 2.0
    mask_threshold: float = 0.3
    foreground_threshold: float = 0.4

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> 'RVMConfig':
        with open(path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))


class RVM(nn.Module):
    """Radical Verification Module."""

    def __init__(
        self,
        image_size: int = 64,
        rvm_config_path: str = 'configs/rvm.json',
        bert_config_path: str = 'configs/bert.json',
        tokenizer_config_path: str = 'configs/tokenizer.json',
    ):
        super().__init__()

        self.image_size = image_size
        self.config = RVMConfig.from_json_file(rvm_config_path)
        self.max_txt_len = self.config.max_txt_len

        self.encoder = ResNet()

        bert_config = BertConfig.from_json_file(bert_config_path)
        self.decoder = Decoder(bert_config, image_size, self.config.num_decoder_layers)

        self.tokenizer = build_tokenizer(tokenizer_config_path)

        self.word_embeddings = nn.Embedding(
            bert_config.vocab_size, bert_config.hidden_size, padding_idx=bert_config.pad_token_id
        )

        # ResNet downsamples by 16, one token per feature map position
        num_image_tokens = (image_size // 16) ** 2
        self.pos_embeddings = nn.Parameter(torch.zeros(1, num_image_tokens, bert_config.hidden_size))

        self.unit_alphabet = [
            t for t in self.tokenizer.get_vocab()
            if t not in self.tokenizer.all_special_tokens and t not in self.tokenizer.LAYOUT
        ]

    def forward(self, *args, mode: str, **kwargs):
        # training must go through forward() so that DDP synchronizes gradients
        if mode == 'pretrain':
            return self.forward_pretrain_radical_mask(*args, **kwargs)
        if mode == 'finetune':
            return self.forward_finetune_radical_match(*args, **kwargs)
        raise ValueError(f'unknown mode {mode!r}')

    def forward_finetune_radical_match(self, image, text, reduction='mean'):
        image_embeds = self.encode_image(image)

        # positve pair
        loss_pos = self.get_radical_match_loss(image_embeds, text, 1.0, reduction)

        # negative pair
        neg_tokens = []
        for ds in self.tokenizer.get_decompose_string(text):
            pos_token_set = set(ds.split()) - set(self.tokenizer.LAYOUT)
            neg_tokens.append(
                self.sample_negative_tokens(pos_token_set, self.config.finetune_neg_ratio)
            )

        loss_neg = self.get_radical_match_loss(image_embeds, neg_tokens, 0.0, reduction)

        return RVMOutput(
            loss=loss_pos + loss_neg,
            loss_rtm=loss_pos + loss_neg,
        )

    def forward_pretrain_radical_mask(self, image, text, radical_mask):
        text_add_neg = []
        for t in text:
            neg_text = self.sample_negative_tokens(set(t.split()), self.config.pretrain_neg_ratio)
            text_add_neg.append(t + ' ' + neg_text)

        text_tokens = self.tokenize(text_add_neg, image.device)

        if radical_mask.size(1) < self.max_txt_len:
            padding = torch.zeros(
                radical_mask.size(0), self.max_txt_len -
                radical_mask.size(1), *radical_mask.size()[2:],
                device=radical_mask.device
            )
            radical_mask = torch.cat([radical_mask, padding], dim=1)

        # right shift by 1, for cls token
        cls_pad = torch.zeros(
            radical_mask.size(0), 1, *radical_mask.size()[2:], device=radical_mask.device
        )
        radical_mask = torch.cat([cls_pad, radical_mask[:, :-1]], dim=1)

        if radical_mask.size(-1) != self.image_size:
            radical_mask = F.interpolate(
                radical_mask, size=(self.image_size, self.image_size), mode='nearest'
            )

        text_tokens.update({'token_ids_mask': radical_mask})

        loss_rtm, loss_focal, loss_dice = self.get_radical_mask_loss(image, text_tokens)

        return RVMOutput(
            loss=(
                self.config.rtm_weight * loss_rtm +
                self.config.focal_weight * loss_focal +
                self.config.dice_weight * loss_dice
            ),
            loss_rtm=loss_rtm,
            loss_dice=loss_dice,
            loss_focal=loss_focal,
        )

    def tokenize(self, text, device):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

    def build_queries(self, text_tokens):
        """Embed radical tokens; special and layout tokens are replaced by padding."""
        layout_mask = ~self.tokenizer.get_layout_mask(text_tokens.input_ids)

        special_tokens = torch.tensor(self.tokenizer.all_special_ids).to(text_tokens.input_ids.device)
        special_token_mask = text_tokens.attention_mask.masked_fill(
            torch.isin(text_tokens.input_ids, special_tokens), 0
        ).bool()

        mask = special_token_mask & layout_mask

        query_tokens = text_tokens.input_ids.masked_fill(~mask, self.tokenizer.pad_token_id)
        return self.word_embeddings(query_tokens), mask

    def encode_image(self, image, topk_size=1):
        image_embeds = self.encoder(image) + self.pos_embeddings
        return image_embeds.repeat_interleave(topk_size, dim=0)

    def sample_negative_tokens(self, pos_token_set, ratio):
        unused_alphabet = set(self.unit_alphabet) - pos_token_set
        return ' '.join(sample(unused_alphabet, ratio * len(pos_token_set)))

    def get_radical_match_loss(self, image_embeds, text, target, reduction='mean'):
        text_tokens = self.tokenize(text, image_embeds.device)
        query_tokens, mask = self.build_queries(text_tokens)

        predict_match, _, _ = self.decoder(query_tokens, image_embeds)

        targets = torch.full_like(predict_match[mask], target)
        return F.binary_cross_entropy_with_logits(
            predict_match[mask], targets, reduction=reduction
        )

    def get_radical_mask_loss(self, image, text_tokens, reduction='mean', eps=1e-6):
        query_tokens, mask = self.build_queries(text_tokens)
        image_embeds = self.encode_image(image)

        predict_match, predict_mask, _ = self.decoder(query_tokens, image_embeds)

        target_mask = self.merge_same_token_mask(
            text_tokens.input_ids, text_tokens.token_ids_mask
        )

        target_match = (target_mask.sum(dim=(-2, -1)) > 0).float()
        loss_rtm = F.binary_cross_entropy_with_logits(
            predict_match[mask], target_match[mask], reduction=reduction
        )

        mask = mask & target_match.bool()  # only compute positive samples

        loss_focal = sigmoid_focal_loss(
            predict_mask[mask], target_mask[mask],
            alpha=self.config.focal_alpha, gamma=self.config.focal_gamma, reduction=reduction
        )

        probs = torch.sigmoid(predict_mask)

        numerator = 2 * (probs * target_mask).sum(dim=(-2, -1))
        denominator = probs.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1))

        loss_dice = 1 - (numerator + eps) / (denominator + eps)
        loss_dice = loss_dice[mask].mean()

        return loss_rtm, loss_focal, loss_dice

    def merge_same_token_mask(self, input_ids, token_ids_mask):
        B, L, H, W = token_ids_mask.shape

        # scatter max: 把相同 index 的 mask 合併
        _, inverse = torch.unique(input_ids, return_inverse=True)  # inverse: (B, L)
        num_unique = inverse.max().item() + 1

        unique_ids_expanded = inverse.view(B, L, 1, 1).expand(B, L, H, W)  # (B, L, H, W)

        # 建立 (B, num_unique, H, W) 的 buffer 做 scatter max
        buffer = torch.zeros(B, num_unique, H, W, device=token_ids_mask.device)
        buffer.scatter_reduce_(
            dim=1,
            index=unique_ids_expanded,
            src=token_ids_mask.float(),
            reduce='amax',
            include_self=True
        )  # (B, num_unique, H, W)

        # 用 unique_ids 把合併後的 mask gather 回 (B, L, H, W)
        return buffer.gather(dim=1, index=unique_ids_expanded)

    def score(self, image, candidates, output_attentions=False) -> 'RadicalScores':
        """
        Verify each image against its candidate characters.

        ``candidates`` holds ``K`` characters per image, flattened image-major,
        so ``len(candidates) == K * image.size(0)``.  Every returned tensor has
        ``K * B`` rows in that same order.
        """
        topk_size = len(candidates) // image.size(0)

        text_tokens = self.tokenize(candidates, image.device)
        query_tokens, mask = self.build_queries(text_tokens)
        image_embeds = self.encode_image(image, topk_size)

        predict_match, predict_mask, attn_weights = self.decoder(
            query_tokens, image_embeds, output_attentions=output_attentions
        )

        # spatial coverage: predicted radical union / foreground pixels
        predict_mask = predict_mask.sigmoid() * mask.unsqueeze(-1).unsqueeze(-1).float()
        predict_mask = predict_mask.max(dim=1).values  # (B, H, W)
        predict_mask = (predict_mask > self.config.mask_threshold).float()

        foreground_mask = (image < self.config.foreground_threshold).squeeze(1)
        foreground_mask = foreground_mask.repeat_interleave(topk_size, dim=0)
        coverage = predict_mask.sum(dim=(-2, -1)) / foreground_mask.sum(dim=(-2, -1)).clamp(min=1)

        # radical presence: mean match probability over radical tokens
        token_scores = predict_match.sigmoid() * mask.float()
        presence = token_scores.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return RadicalScores(
            input_ids=text_tokens.input_ids,
            token_mask=mask,
            token_scores=token_scores,
            presence=presence,
            coverage=coverage,
            score=presence * coverage,
            attn_weights=attn_weights,
        )


@dataclass
class RadicalScores:
    input_ids: torch.Tensor     # (N, L)
    token_mask: torch.Tensor    # (N, L) True for radical tokens
    token_scores: torch.Tensor  # (N, L) match probability, 0 outside token_mask
    presence: torch.Tensor      # (N,)  s_pres
    coverage: torch.Tensor      # (N,)  s_cov
    score: torch.Tensor         # (N,)  s_local = s_pres * s_cov
    attn_weights: list


@dataclass
class RVMOutput(ModelOutput):
    """
    Output for RVM model.
    """
    loss: Optional[torch.FloatTensor] = None

    loss_rtm: Optional[torch.FloatTensor] = None

    loss_dice: Optional[torch.FloatTensor] = None

    loss_focal: Optional[torch.FloatTensor] = None


def build_rvm(fp: Union[str, Path] = None, **kwargs) -> RVM:
    model = RVM(**kwargs)
    if fp is not None:
        model.load_state_dict(
            torch.load(fp, map_location='cpu', weights_only=True),
        )
    return model
