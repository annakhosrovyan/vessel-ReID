import wandb

def configure_wandb_metrics() -> None:
    wandb.define_metric("train/epoch")
    wandb.define_metric("val/epoch")
    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")
    wandb.define_metric("hoss/*", step_metric="val/epoch")
    wandb.define_metric("optisar/*", step_metric="val/epoch")
    wandb.define_metric("val_mAP", step_metric="epoch")
    wandb.define_metric("val_rank1", step_metric="epoch")
    wandb.define_metric("val_rank5", step_metric="epoch")
    wandb.define_metric("val_rank10", step_metric="epoch")

