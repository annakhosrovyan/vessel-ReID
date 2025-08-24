import torch
import torch.nn as nn
from transformers import AutoModel

class DinoV3(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super(DinoV3, self).__init__()
        print(f"Loading pretrained DINOv3 model: {model_name}")
        self.dinov3 = AutoModel.from_pretrained(model_name)
        self.feat_dim = 768

    def forward(self, x):
        outputs = self.dinov3(x)
        cls_features = outputs.last_hidden_state[:, 0]
        return cls_features

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        self.load_state_dict(param_dict)
        print(f'Loaded fine-tuned weights from {model_path}')