import torch.nn as nn
import torch


def resolve_clip_style(metric_loss_type: str, clip_style: str) -> str:
    metric = str(metric_loss_type).lower()
    style = str(clip_style).lower()
    if metric == "mixed_clip":
        return "mixed"
    if metric == "cross_clip":
        return "cross"
    if metric != "clip":
        raise ValueError(f"resolve_clip_style only supports clip metric types, got {metric_loss_type}")
    if style in ("cross", "mixed"):
        return style
    raise ValueError(f"Invalid MODEL.CLIP_STYLE '{clip_style}'. Expected 'cross' or 'mixed'.")


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def mixed_clip_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    cam_labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")
    if labels.ndim != 1 or cam_labels.ndim != 1:
        raise ValueError("labels and cam_labels must be 1D tensors")
    if embeddings.size(0) != labels.size(0) or embeddings.size(0) != cam_labels.size(0):
        raise ValueError("embeddings, labels, and cam_labels must have the same batch dimension")

    embeds = nn.functional.normalize(embeddings, dim=-1, p=2)
    logits = torch.matmul(embeds, embeds.t()) * logit_scale
    batch_size = logits.size(0)
    eye_mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(eye_mask, -1e9)

    same_id = labels[:, None].eq(labels[None, :])
    diff_cam = cam_labels[:, None].ne(cam_labels[None, :])
    pos_mask = same_id & diff_cam
    pos_count = pos_mask.sum(dim=1)
    if not torch.all(pos_count == 1):
        raise ValueError("mixed_clip_loss expects exactly one cross-modal positive per anchor")
    targets = pos_mask.to(torch.int64).argmax(dim=1)
    return nn.functional.cross_entropy(logits, targets)