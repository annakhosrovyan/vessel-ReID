import timm
import torch
import torch.nn as nn

class DinoV3(nn.Module):
    def __init__(self, img_size=[224, 224], model_name: str = "vit_base_patch16_dinov3.lvd1689m", **kwargs):
        super(DinoV3, self).__init__()
        self.model_name = model_name
        self.dinov3 = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.feat_dim = self.dinov3.num_features
        self.img_size = img_size

    def forward(self, x):
        return self.dinov3(x)

    def load_param(self, model_path):
            print(f'Loading pretrained weights from timm: {model_path} with img_size={self.img_size}')
            self.dinov3 = timm.create_model(model_path, pretrained=True, num_classes=0, img_size=self.img_size)
            self.feat_dim = self.dinov3.num_features