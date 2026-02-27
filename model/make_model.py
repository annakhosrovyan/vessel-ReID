import os
import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
from .backbones.vit_transoss import vit_base_patch16_224_TransOSS
from .backbones.dinov3 import DinoV3, DinoV3DualEmbed
from .backbones.chi_vit import chivit_base
from .backbones.chi_vit_ibot import chivit_base_ibot
from .backbones.ibot_head import iBOTHead
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


def _is_rank0():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def rank0_print(*args, **kwargs):
    if _is_rank0():
        print(*args, **kwargs)


def _parse_optional_norm(norm_value):
    if norm_value is None:
        return None
    if isinstance(norm_value, str):
        stripped = norm_value.strip()
        if stripped in ("", "none", "None"):
            return None
        return stripped
    return norm_value


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            rank0_print('using resnet50 as a backbone')
        else:
            rank0_print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            rank0_print('Loading pretrained model......from {}'.format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        param_obj = torch.load(trained_path, map_location='cpu')
        if isinstance(param_obj, dict):
            extracted = None
            for k in ('state_dict', 'model_state_dict', 'model', 'model_state', 'net', 'weights', 'params'):
                v = param_obj.get(k) if isinstance(param_obj, dict) else None
                if isinstance(v, dict):
                    extracted = v
                    break
            if extracted is None:
                extracted = {k: v for k, v in param_obj.items() if torch.is_tensor(v)}
            param_dict = extracted
        else:
            param_dict = param_obj
        model_state = self.state_dict()
        matched = 0
        for k, v in param_dict.items():
            key = k.replace('module.', '')
            if key in model_state and isinstance(v, torch.Tensor) and model_state[key].shape == v.shape:
                model_state[key].copy_(v)
                matched += 1
        total = len(model_state)
        rank0_print(f'Loaded {matched}/{total} tensors from checkpoint')
        if matched == 0:
            raise ValueError(f'No parameters matched when loading {trained_path}. Check checkpoint format and key names.')
        rank0_print('Loading pretrained model from {}'.format(trained_path))


    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        rank0_print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, cfg, factory, logit_scale_init_value=2.6592):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.model_type = cfg.MODEL.TRANSFORMER_TYPE

        rank0_print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.MIE:
            camera_num = camera_num
        else:
            camera_num = 0
        model_class = factory.get(cfg.MODEL.TRANSFORMER_TYPE)
        if model_class is None:
            raise ValueError('Unsupported model type: {}'.format(cfg.MODEL.TRANSFORMER_TYPE))
        
        model_kwargs = {
            'img_size': cfg.INPUT.SIZE_TRAIN,
            'stride_size': cfg.MODEL.STRIDE_SIZE,
            'patch_size': cfg.MODEL.STRIDE_SIZE[0],
            'drop_path_rate': cfg.MODEL.DROP_PATH,
            'drop_rate': cfg.MODEL.DROP_OUT,
            'attn_drop_rate': cfg.MODEL.ATT_DROP_RATE,
            'camera': camera_num,
            'mie_coe': cfg.MODEL.MIE_COE,
            'sse': cfg.MODEL.SSE,
        }

        self.base = model_class(**model_kwargs)
        rank0_print("pretrain_choice: ", pretrain_choice)
        if pretrain_choice == 'imagenet':
            if cfg.MODEL.PRETRAIN_PATH:
                self.base.load_param(model_path)
                rank0_print('Loading pretrained model......from {}'.format(model_path))
            else:
                rank0_print('WARNING: PRETRAIN_PATH is empty, training from scratch!')
        
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            rank0_print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            rank0_print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            rank0_print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            rank0_print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.train_pair = False
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))
        self.rgb_channel_idxs = list(getattr(cfg.MODEL, 'RGB_CHANNELS', [0, 1, 2]))
        self.sar_channel_idxs = list(getattr(cfg.MODEL, 'SAR_CHANNELS', [10, 11]))


    def train_with_pair(self,):
        self.train_pair = True


    def train_with_single(self,):
        self.train_pair = False


    def forward(self, x, label=None, cam_label=None, img_wh=None):
        if self.model_type == 'dinov3':
            global_feat = self.base(x)
        elif self.model_type == 'dinov3_dual_embed':
            global_feat = self.base(x, cam_label=cam_label)
        elif self.model_type == 'chivit_base':
            B = x.shape[0]
            rgb_idx = torch.nonzero(cam_label == 0, as_tuple=True)[0]
            sar_idx = torch.nonzero(cam_label == 1, as_tuple=True)[0]
            n_sar_ch = len(self.sar_channel_idxs)
            rgb_out = self.base(x[rgb_idx], channel_idxs=self.rgb_channel_idxs) if rgb_idx.numel() > 0 else None
            sar_out = self.base(x[sar_idx][:, :n_sar_ch, :, :], channel_idxs=self.sar_channel_idxs) if sar_idx.numel() > 0 else None
            
            feat_shape = rgb_out.shape[1:] if rgb_out is not None else sar_out.shape[1:]
            global_feat = torch.empty((B,) + feat_shape, dtype=x.dtype, device=x.device)
            
            if rgb_idx.numel() > 0:
                global_feat[rgb_idx] = rgb_out
            if sar_idx.numel() > 0:
                global_feat[sar_idx] = sar_out
        else:
            global_feat = self.base(x, cam_label=cam_label, img_wh=img_wh)

        if self.training:
            if self.train_pair:
                b_s = global_feat.size(0)
                # normalized features
                opt_embeds = global_feat[0:b_s // 2]
                sar_embeds = global_feat[b_s // 2:]
                opt_embeds = opt_embeds / opt_embeds.norm(p=2, dim=-1, keepdim=True)
                sar_embeds = sar_embeds / sar_embeds.norm(p=2, dim=-1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.logit_scale.exp()
                logits_per_sar = torch.matmul(sar_embeds, opt_embeds.t()) * logit_scale
                return logits_per_sar

            else:
                feat = self.bottleneck(global_feat)
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat, label)
                else:
                    cls_score = self.classifier(feat)

                return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                feat = self.bottleneck(global_feat)
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat


    def load_param(self, trained_path):
        param_obj = torch.load(trained_path, map_location='cpu')
        if isinstance(param_obj, dict):
            extracted = None
            for k in ('state_dict', 'model_state_dict', 'model', 'model_state', 'net', 'weights', 'params'):
                v = param_obj.get(k) if isinstance(param_obj, dict) else None
                if isinstance(v, dict):
                    extracted = v
                    break
            if extracted is None:
                extracted = {k: v for k, v in param_obj.items() if torch.is_tensor(v)}
            param_dict = extracted
        else:
            param_dict = param_obj
        model_state = self.state_dict()
        matched = 0
        for k, v in param_dict.items():
            key = k.replace('module.', '')
            if key in model_state and isinstance(v, torch.Tensor) and model_state[key].shape == v.shape:
                model_state[key].copy_(v)
                matched += 1
        total = len(model_state)
        rank0_print(f'Loaded {matched}/{total} tensors from checkpoint')
        if matched == 0:
            raise ValueError(f'No parameters matched when loading {trained_path}. Check checkpoint format and key names.')
        rank0_print('Loading pretrained model from {}'.format(trained_path))


    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        rank0_print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_ibot(nn.Module):
    def __init__(self, cfg):
        super(build_transformer_ibot, self).__init__()
        self.cfg = cfg
        self.in_planes = 768
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.rgb_channel_idxs = list(getattr(cfg.MODEL, 'RGB_CHANNELS', [0, 1, 2]))
        self.sar_channel_idxs = list(getattr(cfg.MODEL, 'SAR_CHANNELS', [10, 11]))
        self.channel_pool = self.rgb_channel_idxs + self.sar_channel_idxs

        backbone_kwargs = {
            'img_size': cfg.INPUT.SIZE_TRAIN,
            'patch_size': cfg.MODEL.STRIDE_SIZE[0],
            'drop_path_rate': cfg.MODEL.DROP_PATH,
            'drop_rate': cfg.MODEL.DROP_OUT,
            'attn_drop_rate': cfg.MODEL.ATT_DROP_RATE,
            'add_ch_embed': True,
            'shared_proj': True,
            'masked_im_modeling': True,
            'return_all_tokens': False,
        }
        self.student_backbone = chivit_base_ibot(**backbone_kwargs)
        self.teacher_backbone = chivit_base_ibot(**backbone_kwargs)

        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            if cfg.MODEL.PRETRAIN_PATH:
                self.student_backbone.load_param(cfg.MODEL.PRETRAIN_PATH)
                self.teacher_backbone.load_param(cfg.MODEL.PRETRAIN_PATH)
                rank0_print('Loading pretrained model......from {}'.format(cfg.MODEL.PRETRAIN_PATH))
            else:
                rank0_print('WARNING: PRETRAIN_PATH is empty, training from scratch!')

        head_norm = _parse_optional_norm(getattr(cfg.IBOT, "NORM_IN_HEAD", None))
        self.student_head = iBOTHead(
            in_dim=self.in_planes,
            out_dim=cfg.IBOT.OUT_DIM,
            patch_out_dim=cfg.IBOT.PATCH_OUT_DIM,
            norm=head_norm,
            act=cfg.IBOT.ACT_IN_HEAD,
            nlayers=cfg.IBOT.HEAD_NLAYERS,
            hidden_dim=cfg.IBOT.HEAD_HIDDEN_DIM,
            bottleneck_dim=cfg.IBOT.HEAD_BOTTLENECK_DIM,
            norm_last_layer=cfg.IBOT.NORM_LAST_LAYER,
            shared_head=cfg.IBOT.SHARED_HEAD,
        )
        self.teacher_head = iBOTHead(
            in_dim=self.in_planes,
            out_dim=cfg.IBOT.OUT_DIM,
            patch_out_dim=cfg.IBOT.PATCH_OUT_DIM,
            norm=head_norm,
            act=cfg.IBOT.ACT_IN_HEAD,
            nlayers=cfg.IBOT.HEAD_NLAYERS,
            hidden_dim=cfg.IBOT.HEAD_HIDDEN_DIM,
            bottleneck_dim=cfg.IBOT.HEAD_BOTTLENECK_DIM,
            norm_last_layer=cfg.IBOT.NORM_LAST_LAYER,
            shared_head=cfg.IBOT.SHARED_HEAD_TEACHER,
        )
        self._init_teacher_from_student()
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    @torch.no_grad()
    def _init_teacher_from_student(self):
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict(), strict=False)
        self.teacher_head.load_state_dict(self.student_head.state_dict(), strict=False)

    def _forward_backbone_grouped(self, backbone, views, channel_idxs, masks=None, return_all_tokens=True):
        if not isinstance(views, list):
            views = [views]
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        sizes = [inp.shape[-1] for inp in views]
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor(sizes), return_counts=True)[1], 0)
        outputs = []
        start_idx = 0
        for end_idx in idx_crops:
            end = int(end_idx.item())
            inp = torch.cat(views[start_idx:end], dim=0)
            inp_mask = None
            if masks is not None:
                inp_mask = torch.cat(masks[start_idx:end], dim=0)
            out = backbone(inp, channel_idxs=channel_idxs, mask=inp_mask, return_all_tokens=return_all_tokens)
            outputs.append(out)
            start_idx = end
        return torch.cat(outputs, dim=0)

    def _forward_eval_features(self, x, cam_label):
        if cam_label is None:
            if x.shape[1] == len(self.channel_pool):
                return self.student_backbone(x, channel_idxs=self.channel_pool, return_all_tokens=False)
            return self.student_backbone(x, channel_idxs=self.rgb_channel_idxs[: x.shape[1]], return_all_tokens=False)

        b = x.shape[0]
        rgb_idx = torch.nonzero(cam_label == 0, as_tuple=True)[0]
        sar_idx = torch.nonzero(cam_label == 1, as_tuple=True)[0]
        global_feat = torch.empty((b, self.in_planes), dtype=x.dtype, device=x.device)
        if rgb_idx.numel() > 0:
            rgb = x[rgb_idx][:, : len(self.rgb_channel_idxs), :, :]
            rgb_out = self.student_backbone(rgb, channel_idxs=self.rgb_channel_idxs, return_all_tokens=False)
            global_feat[rgb_idx] = rgb_out
        if sar_idx.numel() > 0:
            sar = x[sar_idx][:, : len(self.sar_channel_idxs), :, :]
            sar_out = self.student_backbone(sar, channel_idxs=self.sar_channel_idxs, return_all_tokens=False)
            global_feat[sar_idx] = sar_out
        return global_feat

    @torch.no_grad()
    def momentum_update_teacher(self, momentum):
        student_backbone_params = dict(self.student_backbone.named_parameters())
        for name, teacher_p in self.teacher_backbone.named_parameters():
            student_p = student_backbone_params.get(name)
            if student_p is None:
                continue
            teacher_p.data.mul_(momentum).add_((1.0 - momentum) * student_p.detach().data)

        student_head_params = dict(self.student_head.named_parameters())
        for name, teacher_p in self.teacher_head.named_parameters():
            student_p = student_head_params.get(name)
            if student_p is None:
                continue
            teacher_p.data.mul_(momentum).add_((1.0 - momentum) * student_p.detach().data)

    def cancel_gradients_last_layer(self):
        for name, p in self.student_head.named_parameters():
            if p.grad is not None and "last_layer" in name:
                p.grad = None

    def forward(
        self,
        x,
        label=None,
        cam_label=None,
        img_wh=None,
        mode='eval',
        local_x=None,
        global_masks=None,
        global_channel_idxs=None,
        local_channel_idxs=None,
    ):
        if mode == 'ibot':
            if global_channel_idxs is None or local_channel_idxs is None:
                raise ValueError("global_channel_idxs and local_channel_idxs are required for iBOT mode")
            if not isinstance(x, list):
                x = [x]
            if local_x is None:
                local_x = []

            student_tokens = self._forward_backbone_grouped(
                self.student_backbone,
                x,
                channel_idxs=global_channel_idxs,
                masks=global_masks,
                return_all_tokens=True,
            )
            student_output = self.student_head(student_tokens)

            with torch.no_grad():
                teacher_tokens = self._forward_backbone_grouped(
                    self.teacher_backbone,
                    x,
                    channel_idxs=global_channel_idxs,
                    masks=None,
                    return_all_tokens=True,
                )
                teacher_output = self.teacher_head(teacher_tokens)

            student_local_cls = None
            if len(local_x) > 0:
                local_tokens = self._forward_backbone_grouped(
                    self.student_backbone,
                    local_x,
                    channel_idxs=local_channel_idxs,
                    masks=None,
                    return_all_tokens=True,
                )
                local_out = self.student_head(local_tokens)
                student_local_cls = local_out[0] if isinstance(local_out, tuple) else local_out

            return student_output, teacher_output, student_local_cls

        global_feat = self._forward_eval_features(x, cam_label=cam_label)
        if self.training:
            return global_feat
        if self.neck_feat == 'after':
            return self.bottleneck(global_feat)
        return global_feat


__factory_T_type = {
    'vit_base_patch16_224_TransOSS': vit_base_patch16_224_TransOSS,
    'dinov3': DinoV3,
    'dinov3_dual_embed': DinoV3DualEmbed,
    'chivit_base': chivit_base,
    'chivit_base_ibot': chivit_base_ibot,
}


def make_model(cfg, num_class, camera_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.METRIC_LOSS_TYPE == 'ibot':
            model = build_transformer_ibot(cfg)
        else:
            model = build_transformer(num_class, camera_num, cfg, __factory_T_type)
        rank0_print(f'===========building transformer: {cfg.MODEL.TRANSFORMER_TYPE}===========')
    else:
        model = Backbone(num_class, cfg)
        rank0_print('===========building ResNet===========')
    return model
