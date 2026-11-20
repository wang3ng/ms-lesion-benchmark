"""Loading MedSAM (SAM ViT-B weights from bowang-lab/MedSAM) and its parts."""

from __future__ import annotations

from pathlib import Path

import torch
from segment_anything import sam_model_registry

from msbench.config import load_config, paths


def checkpoint_path(cfg: dict | None = None) -> Path:
    p = paths(cfg or load_config())["medsam_checkpoint"]
    if not p.exists():
        raise FileNotFoundError(f"MedSAM checkpoint missing: {p}")
    return p


def load_medsam(cfg: dict | None = None, device: str = "cuda"):
    """Full MedSAM model (image encoder + prompt encoder + mask decoder), eval mode."""
    model = sam_model_registry["vit_b"](checkpoint=None)
    state = torch.load(checkpoint_path(cfg), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def decode(model, embeddings: torch.Tensor, boxes_1024: torch.Tensor, return_iou: bool = False):
    """Low-res mask logits (B, 1, 256, 256) for B (embedding, box) pairs, one box per embedding.

    With return_iou=True also returns SAM's predicted-IoU score of each mask, shape (B,).

    Same computation as SAM's MaskDecoder.predict_masks with multimask_output=False,
    except that embeddings are NOT repeated per prompt (MedSAM's change to SAM,
    which allows a batch of different slices with one box each).
    """
    pe, dec = model.prompt_encoder, model.mask_decoder
    sparse, dense = pe(points=None, boxes=boxes_1024[:, None, :], masks=None)
    output_tokens = torch.cat([dec.iou_token.weight, dec.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse), dim=1)
    src = embeddings + dense
    pos_src = torch.repeat_interleave(pe.get_dense_pe(), tokens.shape[0], dim=0)
    b, c, h, w = src.shape
    hs, src = dec.transformer(src, pos_src, tokens)
    mask_tokens_out = hs[:, 1 : (1 + dec.num_mask_tokens), :]
    src = src.transpose(1, 2).view(b, c, h, w)
    up = dec.output_upscaling(src)
    hyper_in = torch.stack([dec.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                            for i in range(dec.num_mask_tokens)], dim=1)
    b, c, h, w = up.shape
    masks = (hyper_in @ up.view(b, c, h * w)).view(b, -1, h, w)
    if return_iou:
        iou = dec.iou_prediction_head(hs[:, 0, :])
        return masks[:, 0:1], iou[:, 0]
    return masks[:, 0:1]
