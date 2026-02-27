import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def _build_temp_schedule(warmup_temp, target_temp, warmup_epochs, max_epochs, delay_epochs=0):
    max_epochs = int(max(1, max_epochs))
    warmup_epochs = int(max(0, warmup_epochs))
    delay_epochs = int(max(0, delay_epochs))

    schedule = np.ones(max_epochs, dtype=np.float32) * float(target_temp)
    if delay_epochs > 0:
        schedule[: min(delay_epochs, max_epochs)] = float(warmup_temp)
    if warmup_epochs > 0:
        start = min(delay_epochs, max_epochs)
        end = min(start + warmup_epochs, max_epochs)
        warm = np.linspace(float(warmup_temp), float(target_temp), num=max(1, end - start), dtype=np.float32)
        schedule[start:end] = warm[: end - start]
    return schedule


class IBOTLoss(nn.Module):
    """Epoch-based iBOT loss for CLS and masked patch tokens."""

    def __init__(
        self,
        out_dim,
        patch_out_dim,
        ngcrops,
        nlcrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_patch_temp,
        teacher_patch_temp,
        warmup_teacher_temp_epochs,
        max_epochs,
        lambda1=1.0,
        lambda2=1.0,
        center_momentum=0.9,
        center_momentum2=0.9,
        mim_start_epoch=0,
        student_temp=0.1,
    ):
        super().__init__()
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.center_momentum2 = float(center_momentum2)
        self.ngcrops = int(ngcrops)
        self.nlcrops = int(nlcrops)
        self.ncrops = self.ngcrops + self.nlcrops
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))

        teacher_temp_schedule = _build_temp_schedule(
            warmup_temp=warmup_teacher_temp,
            target_temp=teacher_temp,
            warmup_epochs=warmup_teacher_temp_epochs,
            max_epochs=max_epochs,
            delay_epochs=0,
        )
        teacher_patch_temp_schedule = _build_temp_schedule(
            warmup_temp=warmup_teacher_patch_temp,
            target_temp=teacher_patch_temp,
            warmup_epochs=warmup_teacher_temp_epochs,
            max_epochs=max_epochs,
            delay_epochs=mim_start_epoch,
        )
        self.register_buffer("teacher_temp_schedule", torch.from_numpy(teacher_temp_schedule))
        self.register_buffer("teacher_temp2_schedule", torch.from_numpy(teacher_patch_temp_schedule))

    def _current_temps(self, epoch, device, dtype):
        idx = int(max(1, epoch)) - 1
        idx = min(idx, int(self.teacher_temp_schedule.numel()) - 1)
        t_cls = self.teacher_temp_schedule[idx].to(device=device, dtype=dtype)
        t_patch = self.teacher_temp2_schedule[idx].to(device=device, dtype=dtype)
        return t_cls, t_patch

    def forward(self, student_output, teacher_output, student_local_cls, student_masks, epoch, num_channels):
        """
        Args:
            student_output: tuple(student_cls_logits, student_patch_logits), globals only.
            teacher_output: tuple(teacher_cls_logits, teacher_patch_logits), globals only.
            student_local_cls: local-crop student cls logits or None.
            student_masks: list of [B,H,W] masks for global crops.
            epoch: 1-based epoch index.
            num_channels: number of selected global channels for this step.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        if int(num_channels) <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}")

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls], dim=0)

        student_cls = student_cls / self.student_temp
        student_patch = student_patch / self.student_temp

        student_cls_chunks = student_cls.chunk(self.ncrops)
        student_patch_chunks = student_patch.chunk(self.ngcrops)

        t_cls, t_patch = self._current_temps(epoch, teacher_cls.device, teacher_cls.dtype)
        teacher_cls_probs = F.softmax((teacher_cls - self.center) / t_cls, dim=-1).detach().chunk(self.ngcrops)
        teacher_patch_probs = F.softmax((teacher_patch - self.center2) / t_patch, dim=-1).detach().chunk(self.ngcrops)

        total_loss1 = torch.tensor(0.0, device=teacher_cls.device, dtype=teacher_cls.dtype)
        total_loss2 = torch.tensor(0.0, device=teacher_cls.device, dtype=teacher_cls.dtype)
        n_loss_terms1 = 0
        n_loss_terms2 = 0

        for q in range(self.ngcrops):
            for v in range(self.ncrops):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_probs[q] * F.log_softmax(student_patch_chunks[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_masks[v].flatten(1).to(loss2.device)
                    if mask.shape[1] != loss2.shape[1]:
                        repeat_factor = max(1, int(math.ceil(float(loss2.shape[1]) / float(mask.shape[1]))))
                        mask = mask.repeat(1, repeat_factor)[:, : loss2.shape[1]]

                    mask = mask.float()
                    denom = mask.sum(dim=-1).clamp(min=1.0)
                    loss2 = torch.sum(loss2 * mask, dim=-1) / denom
                    total_loss2 = total_loss2 + loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(
                        -teacher_cls_probs[q] * F.log_softmax(student_cls_chunks[v], dim=-1),
                        dim=-1,
                    )
                    total_loss1 = total_loss1 + loss1.mean()
                    n_loss_terms1 += 1

        if n_loss_terms1 > 0:
            total_loss1 = (total_loss1 / n_loss_terms1) * self.lambda1
        if n_loss_terms2 > 0:
            total_loss2 = (total_loss2 / n_loss_terms2) * self.lambda2

        total = total_loss1 + total_loss2
        self.update_center(teacher_cls.detach(), teacher_patch.detach())
        return {"cls": total_loss1, "patch": total_loss2, "loss": total}

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)

        world_size = 1
        if _is_dist_ready():
            dist.all_reduce(cls_center)
            dist.all_reduce(patch_center)
            world_size = dist.get_world_size()

        cls_center = cls_center / (teacher_cls.shape[0] * world_size)
        patch_center = patch_center / (teacher_patch.shape[0] * world_size)
        patch_center = patch_center.unsqueeze(1)

        self.center = self.center * self.center_momentum + cls_center * (1.0 - self.center_momentum)
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1.0 - self.center_momentum2)
