import timm
import torch
import torch.nn as nn
from pathlib import Path

class DinoV3(nn.Module):
    def __init__(self, 
                img_size=[224, 224], 
                model_name: str = "vit_base_patch16_dinov3.lvd1689m", 
                global_pool: str = "avg",
                **kwargs
                ):
        super(DinoV3, self).__init__()
        self.model_name = model_name
        self.dinov3 = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.feat_dim = self.dinov3.num_features
        self.img_size = img_size

    def forward(self, x):
        return self.dinov3(x)

    def load_param(self, model_path: str):
        if Path(model_path).exists():
            print(f'Loading pretrained weights from checkpoint: {model_path}')
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('base.dinov3.'):
                    new_key = key.replace('base.dinov3.', '')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            print(f'Loaded {len(new_state_dict)} parameters from checkpoint')
            self.dinov3.load_state_dict(new_state_dict, strict=False)
        else:
            print(f'Loading pretrained model from timm: {model_path}')
            self.dinov3 = timm.create_model(model_path, pretrained=True, num_classes=0, img_size=self.img_size)
            self.feat_dim = self.dinov3.num_features



class DinoV3DualEmbed(DinoV3):
    def __init__(self, 
                img_size=[224, 224], 
                model_name: str = "vit_base_patch16_dinov3.lvd1689m", 
                global_pool: str = "avg",
                **kwargs
                ):
        super().__init__(img_size, model_name, global_pool, **kwargs)
        
        embed_dim = self.dinov3.embed_dim
        patch_size = self.dinov3.patch_embed.patch_size
        self.proj = self.dinov3.patch_embed.proj
        self.patch_size = patch_size
        in_ch = self.proj.in_channels

        self.patch_embed_rgb = nn.Conv2d(
            in_channels=in_ch, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            bias=(self.proj.bias is not None)
        )

        self.patch_embed_sar = nn.Conv2d(
            in_channels=in_ch, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            bias=(self.proj.bias is not None)
        )
        
        with torch.no_grad():
            self.patch_embed_rgb.weight.copy_(self.proj.weight)

            w_mean = self.proj.weight.mean(dim=1, keepdim=True)
            self.patch_embed_sar.weight.copy_(w_mean.repeat(1, in_ch, 1, 1) / in_ch)

            if self.patch_embed_rgb.bias is not None:
                self.patch_embed_rgb.bias.copy_(self.proj.bias)
            if self.patch_embed_sar.bias is not None:
                self.patch_embed_sar.bias.copy_(self.proj.bias)

    def forward(self, x, cam_label=None):
        cam_label = cam_label.to(x.device)
        B = x.shape[0]
        D = self.dinov3.embed_dim
        ps_h, ps_w = self.patch_size
        H, W = x.shape[-2] // ps_h, x.shape[-1] // ps_w

        tokens = x.new_zeros((B, H, W, D))
        rgb_idx = torch.nonzero(cam_label == 0, as_tuple=True)[0]
        sar_idx = torch.nonzero(cam_label == 1, as_tuple=True)[0]
        if rgb_idx.numel() > 0:
            rgb_feat = self.patch_embed_rgb(x[rgb_idx])
            rgb_feat = rgb_feat.permute(0, 2, 3, 1)
            rgb_feat = self.dinov3.patch_embed.norm(rgb_feat)
            tokens[rgb_idx] = rgb_feat.to(tokens.dtype)
        if sar_idx.numel() > 0:
            sar_in = x[sar_idx]
            sar_feat = self.patch_embed_sar(sar_in)
            sar_feat = sar_feat.permute(0, 2, 3, 1)
            sar_feat = self.dinov3.patch_embed.norm(sar_feat)
            tokens[sar_idx] = sar_feat.to(tokens.dtype)

        x = self.dinov3._pos_embed(tokens)[0]
        x = self.dinov3.pos_drop(x)
        for blk in self.dinov3.blocks:
            x = blk(x)
        x = self.dinov3.norm(x)
        x = x[:, self.dinov3.num_prefix_tokens:, :].mean(dim=1)
        x = self.dinov3.fc_norm(x)
        return x