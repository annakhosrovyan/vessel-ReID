import timm
import torch
import torch.nn as nn
from pathlib import Path

class DinoV3(nn.Module):
    def __init__(self, img_size=[224, 224], model_name: str = "vit_base_patch16_dinov3.lvd1689m", **kwargs):
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