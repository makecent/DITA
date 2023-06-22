import math

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on videos.
    """

    def __init__(self, num_feats=128, temperature=10000, normalize=False, offset=0, scale=None):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        # bs, 1, w -> bs, w
        mask = mask.squeeze(dim=1)
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)  # N x T
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_feats * 2, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_feats * 2))

        pos_x = x_embed[:, :, None] / dim_t  # N x T x C
        # n,c,t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_x.permute(0, 2, 1)  # N x C x T
        return pos.unsqueeze(dim=2)


def build_position_encoding(args):
    feat_dim = args.hidden_dim
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(feat_dim, normalize=True)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
