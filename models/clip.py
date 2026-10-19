import torch

from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import ModelOutput

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, Tuple

from .dist import concat_all_gather_unique
from .tokenizer import build_tokenizer
from .encoders import VisionTransformer, BertConfig, BertModel


class CLIP(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        embed_dim: int = 2048,
        bert_config_path: str = 'configs/bert.json',
        tokenizer_config_path: str = 'configs/tokenizer.json',
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.label_smoothing = label_smoothing

        self.visual_encoder = self.init_visual_encoder(
            img_size=img_size, patch_size=patch_size,
        )

        encoder_config = BertConfig.from_json_file(bert_config_path)
        self.text_encoder = BertModel(encoder_config)

        self.tokenizer = build_tokenizer(tokenizer_config_path)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        self.vision_proj = nn.Linear(encoder_config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(encoder_config.hidden_size, embed_dim)

        self.temp = nn.Parameter(torch.log(1 / 0.07 * torch.ones([])))

        self.max_txt_len = self.text_encoder.config.max_position_embeddings

        self.initialize_parameters(encoder_config.hidden_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_parameters(self, hidden_size):
        nn.init.zeros_(self.text_proj.bias)
        nn.init.zeros_(self.vision_proj.bias)
        nn.init.trunc_normal_(self.text_proj.weight, std=hidden_size ** -0.5)
        nn.init.trunc_normal_(self.vision_proj.weight, std=hidden_size ** -0.5)

    def forward(self, image, text, standard, mask_ratio=0):
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        text_outputs = self.text_encoder(
            text_tokens.input_ids,
            token_ids_mask=text_tokens.token_ids_mask,
            token_ids_depth=text_tokens.token_ids_depth,
            token_ids_sibling=text_tokens.token_ids_sibling,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            is_decoder=True,
        )   # blip2 qformer

        batch_idx = torch.arange(image.size(0)).to(image.device)
        last_token_idx = text_tokens.attention_mask.sum(dim=-1) - 1

        text_feats = F.normalize(
            self.text_proj(text_outputs.last_hidden_state[batch_idx, last_token_idx]), dim=-1
        )

        image_embeds = self.visual_encoder(image, mask_ratio=mask_ratio)
        image_feats = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        loss_itc = self.get_contrastive_loss(
            image_feats, text_feats, text_tokens.input_ids
        )

        loss_img = self.get_style_disentangle_loss(image_feats, standard)

        return ClipOutput(
            loss=loss_itc + loss_img,
            loss_itc=loss_itc,
            loss_img=loss_img,
        )

    def get_contrastive_loss(self, image_feats, text_feats, gather_label, reduction='mean'):
        image_feats_all, _ = concat_all_gather_unique(image_feats, gather_label)
        text_feats_all, unique_labels = concat_all_gather_unique(
            text_feats, gather_label
        )

        sim_i2t = image_feats @ text_feats_all.t() * self.temp.float().exp()
        sim_t2i = text_feats @ image_feats_all.t() * self.temp.float().exp()

        targets = torch.all(
            gather_label.unsqueeze(1) == unique_labels.unsqueeze(0), dim=-1
        ).long().argmax(dim=1)

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=self.label_smoothing, reduction=reduction) +
            F.cross_entropy(sim_t2i, targets, label_smoothing=self.label_smoothing, reduction=reduction)
        ) / 2

        return loss_itc

    def get_style_disentangle_loss(self, image_feats, standard, reduction='mean'):
        with torch.no_grad():
            standard_feats = self.forward_image(standard)

        targets = torch.ones(image_feats.size(0)).to(image_feats.device)

        return F.cosine_embedding_loss(
            image_feats, standard_feats, targets, reduction=reduction
        )

    def init_visual_encoder(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        use_grad_checkpoint: bool = False,
        ckpt_layer: int = 0,
        drop_path_rate: float = 0,
    ):
        return VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            use_grad_checkpointing=use_grad_checkpoint,
            ckpt_layer=ckpt_layer,
            drop_path_rate=drop_path_rate,
        )

    def compute_sim(self, image: torch.Tensor = None, text: Union[torch.Tensor, Tuple[str]] = None):
        image_feats = self.forward_image(image)
        text_feats = text if isinstance(text, torch.Tensor) else self.forward_text(text)
        return image_feats @ text_feats.t() * self.temp.exp()

    def forward_image(self, image, mask_ratio=0):
        image_embeds = self.visual_encoder(image, mask_ratio=mask_ratio)
        return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

    def forward_text(self, text):
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_outputs = self.text_encoder(
            text_tokens.input_ids,
            token_ids_mask=text_tokens.token_ids_mask,
            token_ids_depth=text_tokens.token_ids_depth,
            token_ids_sibling=text_tokens.token_ids_sibling,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            is_decoder=True,
        )   # blip2 qformer

        batch_idx = torch.arange(len(text)).to(self.device)
        last_token_idx = text_tokens.attention_mask.sum(dim=-1) - 1

        return F.normalize(
            self.text_proj(text_outputs.last_hidden_state[batch_idx, last_token_idx]), dim=-1
        )


@dataclass
class ClipOutput(ModelOutput):
    """
    Output for CLIP model.
    """
    loss: Optional[torch.FloatTensor] = None

    loss_itc: Optional[torch.FloatTensor] = None

    loss_img: Optional[torch.FloatTensor] = None


def build_clip(fp: Union[str, Path] = None, **kwargs) -> CLIP:
    model = CLIP(**kwargs)
    if fp is not None:
        model.load_state_dict(
            torch.load(fp, map_location='cpu', weights_only=True),
        )
    return model
