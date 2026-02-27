import math
import random
import torch
import torch.nn as nn

import os
from functools import partial

from .chi_vit_utils import Block
from .chi_vit_utils import trunc_normal_


def rank0_print(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)


class PatchEmbedPerChannel(nn.Module):
    """Image to patch embedding with channel-aware offsets."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        enable_sample=False,
        add_ch_embed=True,
        shared_proj=True,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        self.shared_proj = shared_proj
        self.add_ch_embed = add_ch_embed

        if shared_proj:
            self.proj = nn.Conv3d(
                1,
                embed_dim,
                kernel_size=(1, patch_size, patch_size),
                stride=(1, patch_size, patch_size),
            )
        else:
            self.proj = nn.Conv2d(
                in_channels=in_chans,
                out_channels=embed_dim * in_chans,
                kernel_size=patch_size,
                stride=patch_size,
                groups=in_chans,
            )

        if add_ch_embed:
            self.channel_embed = nn.parameter.Parameter(torch.zeros(1, embed_dim, in_chans, 1, 1))
            trunc_normal_(self.channel_embed, std=0.02)
        else:
            self.channel_embed = None

        rank0_print("enable_sample:", enable_sample)
        rank0_print("shared_proj:", shared_proj)

    def forward(self, x, channel_idxs):
        b, cin, h, w = x.shape
        if self.training and self.enable_sample:
            cin_new = random.randint(1, cin)
            channels = random.sample(range(cin), k=cin_new)
            cin = cin_new
            x = x[:, channels, :, :]
            channel_idxs = channels

        if isinstance(channel_idxs, torch.Tensor):
            channel_idxs = channel_idxs.flatten().tolist()
        if cin != len(channel_idxs):
            x = x[:, channel_idxs, :, :]
            cin = x.shape[1]

        if self.shared_proj:
            x = self.proj(x.unsqueeze(1))  # B, D, C, H', W'
        else:
            x_padded = torch.zeros(b, self.in_chans, h, w, device=x.device, dtype=x.dtype)
            for i, ch in enumerate(channel_idxs):
                x_padded[:, ch, :, :] = x[:, i, :, :]
            x_proj = self.proj(x_padded)
            h_out, w_out = x_proj.shape[2], x_proj.shape[3]
            x_proj = x_proj.view(b, self.embed_dim, self.in_chans, h_out, w_out)
            x = x_proj[:, :, channel_idxs, :, :]

        if self.add_ch_embed:
            x = x + self.channel_embed[:, :, channel_idxs, :, :]
        return x


class AttChannelEmbed(nn.Module):
    def __init__(self, embed_dim, in_chans, num_heads, mlp_ratio, qkv_bias, qk_scale, norm_layer):
        super().__init__()
        self.channel_embeds = nn.Parameter(torch.zeros(1, in_chans, 1, embed_dim))
        trunc_normal_(self.channel_embeds, std=0.02)
        self.att_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
        )

    def forward(self, x, out_size, channel_idxs):
        b, _, cout = x.shape
        cls_token = x[:, :1]
        x = x[:, 1:].reshape(b, -1, out_size[0] * out_size[1], cout)
        for i, ch in enumerate(channel_idxs):
            x_ch = x[:, i, :, :]
            ch_embed = self.channel_embeds[:, ch, :, :].expand(b, -1, -1)
            x_ch = torch.cat((ch_embed, x_ch), dim=1)
            x_ch = self.att_block(x_ch)
            x[:, i, :, :] = x_ch[:, 1:]
        x = x.reshape(b, -1, cout)
        return torch.cat((cls_token, x), dim=1)


class ChiViTIBOT(nn.Module):
    """ChiViT backbone with optional masked image modeling for iBOT pretraining."""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=12,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        enable_sample=False,
        add_ch_embed=True,
        shared_proj=True,
        masked_im_modeling=True,
        return_all_tokens=False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.in_chans = in_chans
        self.add_ch_embed = add_ch_embed
        self.return_all_tokens = return_all_tokens
        self.masked_im_modeling = bool(masked_im_modeling)

        rank0_print(f"add_ch_embed value: {add_ch_embed}")
        self.patch_embed = PatchEmbedPerChannel(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            enable_sample=enable_sample,
            add_ch_embed=add_ch_embed,
            shared_proj=shared_proj,
        )
        if not self.add_ch_embed:
            self.att_channel_embed = AttChannelEmbed(
                embed_dim=embed_dim,
                in_chans=in_chans,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
            )

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_extra_tokens = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches // self.in_chans + self.num_extra_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.masked_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h, c):
        num_extra_tokens = self.num_extra_tokens if hasattr(self, "num_extra_tokens") else 1
        npatch = x.shape[1] - num_extra_tokens
        n_pos = self.pos_embed.shape[1] - num_extra_tokens
        if npatch == n_pos and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(n_pos)), int(math.sqrt(n_pos)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(n_pos), h0 / math.sqrt(n_pos)),
            mode="bicubic",
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)
        patch_pos_embed = patch_pos_embed.expand(1, c, -1, dim).reshape(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def mask_model(self, x, mask):
        if mask is None:
            return x
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.dim() != 3:
            raise ValueError(f"Expected mask with shape [B,H,W], got {tuple(mask.shape)}")

        b, _, cin, h, w = x.shape
        if mask.shape != (b, h, w):
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match patch grid {(b, h, w)}")
        x_perm = x.permute(0, 3, 4, 2, 1)  # B,H,W,C,D
        x_perm[mask] = self.masked_embed.to(dtype=x.dtype)
        return x_perm.permute(0, 4, 3, 1, 2).contiguous()

    def prepare_tokens(self, x, channel_idxs, mask=None):
        b, _, w, h = x.shape
        x = self.patch_embed(x, channel_idxs)  # B, D, C, H', W'
        out_size = (x.shape[-2], x.shape[-1])
        cin_new = x.shape[2]

        if self.masked_im_modeling and mask is not None:
            x = self.mask_model(x, mask)

        x = x.flatten(2).transpose(1, 2)  # B, C*H*W, D
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h, cin_new)

        if not self.add_ch_embed:
            x = self.att_channel_embed(x, out_size, channel_idxs)
        return self.pos_drop(x)

    def forward(self, x, channel_idxs=None, mask=None, return_all_tokens=None):
        x = self.prepare_tokens(x, channel_idxs, mask=mask)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        use_all_tokens = self.return_all_tokens if return_all_tokens is None else bool(return_all_tokens)
        if use_all_tokens:
            return x
        return x[:, 0, :]

    def load_param(self, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        state_dict = checkpoint["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        model_dict = self.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            elif k in model_dict:
                rank0_print(f"Shape mismatch {k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
            else:
                rank0_print(f"Key not in model: {k}")
        self.load_state_dict(filtered_dict, strict=False)
        rank0_print(f"Loading pretrained ChiViT iBOT from {model_path}: loaded {len(filtered_dict)}/{len(model_dict)} keys")


def chivit_base_ibot(patch_size=16, **kwargs):
    model = ChiViTIBOT(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        in_chans=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
