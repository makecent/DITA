import math
import warnings
from typing import Optional

import mmengine
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.models.layers import DeformableDetrTransformerEncoder, DeformableDetrTransformerDecoder, \
    DeformableDetrTransformerDecoderLayer, DeformableDetrTransformerEncoderLayer, DinoTransformerDecoder
from mmengine.model import ModuleList
from mmengine.model import constant_init, xavier_init
from mmdet.models.layers import MLP
from my_modules.layers.pseudo_layers import Pseudo2DLinear


def zero_y_reference_points(forward_method):
    def wrapper(self, *args, **kwargs):
        if 'reference_points' in kwargs:
            reference_points = kwargs['reference_points'].clone()
            if reference_points.shape[-1] == 2:
                reference_points[..., 1] = 0.5
            elif reference_points.shape[-1] == 4:
                reference_points[..., 1] = 0.5
                reference_points[..., 3] = 0.
            kwargs['reference_points'] = reference_points
        return forward_method(self, *args, **kwargs)

    return wrapper


class CustomMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    """
    Customized DeformableAttention:
    a. sampling_offsets linear layer is changed to output only x offsets and y offsets are fixed to be zeros.
    b. Init the sampling_offsets bias with in 1D style
    c. decorate the forward() function to fix the reference point on y-axis to be zeros.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[mmengine.ConfigDict] = None,
                 value_proj_ratio: float = 1.0):
        super(MultiScaleDeformableAttention, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        # Change the sampling_offsets layer to only output x offsets. The y offsets are fixed to be zeros.
        self.sampling_offsets = Pseudo2DLinear(
            self.embed_dims, self.num_heads * self.num_levels * self.num_points)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        # Re-init the sampling_offsets bias in 1D style
        thetas = torch.arange(self.num_heads,
                              device=device) * (4 * math.pi / self.num_heads)
        grid_init = thetas.cos()[:, None]
        grid_init = grid_init.view(self.num_heads, 1, 1, 1).repeat(
            1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @zero_y_reference_points
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# Below part is to replace the original Deformable Attention with our customized Attention (to support 1D),
# You do NOT have to read them at all.


class CustomDeformableDetrTransformerEncoder(DeformableDetrTransformerEncoder):

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            CustomDeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims


class CustomDeformableDetrTransformerDecoder(DeformableDetrTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            CustomDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

class CustomDinoTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            CustomDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)


class CustomDeformableDetrTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = CustomMultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)


class CustomDeformableDetrTransformerEncoderLayer(DeformableDetrTransformerEncoderLayer):

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = CustomMultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
