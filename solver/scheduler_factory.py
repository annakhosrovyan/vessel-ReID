""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .lr_scheduler import WarmupMultiStepLR
from .wsd_lr import WSDLRScheduler


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    scheduler_type = cfg.SOLVER.SCHEDULER_TYPE
    warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    
    if scheduler_type == 'step':
        milestones = [int(num_epochs * 0.4), int(num_epochs * 0.7)]
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=warmup_t,
            warmup_method=cfg.SOLVER.WARMUP_METHOD
        )
    elif scheduler_type == 'wsd':
        lr_min = 0.002 * cfg.SOLVER.BASE_LR
        warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
        noise_range = None
        lr_scheduler = WSDLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            t_in_epochs=True,
            final_decay_pct=cfg.SOLVER.WSD_DECAY_PCT,
            noise_range_t=noise_range,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
        )
    else:
        lr_min = 0.002 * cfg.SOLVER.BASE_LR
        warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
        noise_range = None

        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_epochs,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
                noise_range_t=noise_range,
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
            )

    return lr_scheduler
