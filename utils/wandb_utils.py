import os
from typing import Optional, Sequence

import wandb


def configure_wandb_metrics() -> None:
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("hoss/*", step_metric="epoch")
    wandb.define_metric("optisar/*", step_metric="epoch")
    wandb.define_metric("val_mAP", step_metric="epoch")
    wandb.define_metric("val_rank1", step_metric="epoch")
    wandb.define_metric("val_rank5", step_metric="epoch")
    wandb.define_metric("val_rank10", step_metric="epoch")


def init_wandb_run(cfg, logger=None, tags: Optional[Sequence[str]] = None) -> bool:
    """Initialize W&B and gracefully fall back when no API key is available.

    Returns:
        bool: True when initialized in online/offline mode, False when fallback
        to disabled mode is used.
    """
    if wandb.run is not None:
        return getattr(wandb.run.settings, "mode", "online") != "disabled"

    mode = str(
        os.environ.get("WANDB_MODE", getattr(cfg.WANDB, "MODE", "online"))
    ).lower()
    allow_no_key_fallback = bool(getattr(cfg.WANDB, "ALLOW_NO_KEY_FALLBACK", True))

    init_kwargs = {
        "project": cfg.WANDB.PROJECT,
        "name": cfg.WANDB.NAME,
        "config": cfg,
        "mode": mode,
    }
    if tags:
        init_kwargs["tags"] = list(tags)

    try:
        wandb.init(**init_kwargs)
        configure_wandb_metrics()
        return mode != "disabled"
    except Exception as exc:
        should_fallback = (
            mode == "online"
            and allow_no_key_fallback
            and "No API key configured" in str(exc)
        )
        if not should_fallback:
            raise

        if logger is not None:
            logger.warning(
                "W&B init failed (no API key). Continuing with WANDB mode='disabled'."
            )
        fallback_kwargs = {
            "project": cfg.WANDB.PROJECT,
            "name": cfg.WANDB.NAME,
            "config": cfg,
            "mode": "disabled",
        }
        if tags:
            fallback_kwargs["tags"] = list(tags)
        wandb.init(**fallback_kwargs)
        configure_wandb_metrics()
        return False
