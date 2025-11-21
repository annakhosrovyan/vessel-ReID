import math
import torch

from .scheduler import Scheduler


class WSDLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min: float = 0.0,
        warmup_t: int = 0,
        warmup_lr_init: float = 0.0,
        warmup_prefix: bool = False,
        t_in_epochs: bool = True,
        final_decay_pct: float = 0.1,
        noise_range_t=None,
        noise_pct: float = 0.67,
        noise_std: float = 1.0,
        noise_seed: int = 42,
        initialize: bool = True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )
        assert t_initial > 0
        assert 0.0 <= final_decay_pct <= 1.0
        assert lr_min >= 0.0
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = int(warmup_t)
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.final_decay_pct = final_decay_pct
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t: int):
        if t < self.warmup_t:
            return [self.warmup_lr_init + t * s for s in self.warmup_steps]
        if self.warmup_prefix:
            t = t - self.warmup_t
        total = self.t_initial
        decay_start = int(math.floor((1.0 - self.final_decay_pct) * total))
        decay_len = max(1, total - decay_start)
        if t < decay_start:
            return [v for v in self.base_values]
        progress = min(max(t - decay_start, 0), decay_len)
        lrs = []
        for v in self.base_values:
            lr = v - (v - self.lr_min) * (progress / decay_len)
            lrs.append(lr)
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        return None

