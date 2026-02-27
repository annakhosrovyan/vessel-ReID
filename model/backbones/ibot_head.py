import torch
import torch.nn as nn

from .chi_vit_utils import trunc_normal_


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        norm=None,
        act="gelu",
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        norm_last_layer=True,
    ):
        super().__init__()
        norm_layer = self._build_norm(norm, hidden_dim)
        act_layer = self._build_act(act)

        nlayers = max(int(nlayers), 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim if bottleneck_dim > 0 else out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(act_layer)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm_layer is not None:
                    layers.append(self._build_norm(norm, hidden_dim))
                layers.append(self._build_act(act))
            layers.append(nn.Linear(hidden_dim, bottleneck_dim if bottleneck_dim > 0 else out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _build_norm(self, norm, dim):
        if norm is None:
            return None
        if norm == "bn":
            return nn.BatchNorm1d(dim)
        if norm == "ln":
            return nn.LayerNorm(dim)
        if norm == "syncbn":
            return nn.SyncBatchNorm(dim)
        raise ValueError(f"Unsupported head norm type: {norm}")

    def _build_act(self, act):
        if act == "gelu":
            return nn.GELU()
        if act == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported head activation: {act}")

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        return x


class iBOTHead(DINOHead):
    def __init__(
        self,
        in_dim,
        out_dim,
        patch_out_dim=8192,
        norm=None,
        act="gelu",
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        norm_last_layer=True,
        shared_head=False,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            norm=norm,
            act=act,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            norm_last_layer=norm_last_layer,
        )

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

    def forward(self, x):
        # CLS-only path (e.g., local crops or eval helpers)
        if x.dim() == 2:
            return super().forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            cls_logits = self.last_layer(x[:, 0])
            patch_logits = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            cls_logits = self.mlp[-1](x[:, 0])
            patch_logits = self.mlp2(x[:, 1:])
        return cls_logits, patch_logits
