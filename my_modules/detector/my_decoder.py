import torch
from mmdet.models.layers import DetrTransformerDecoder
from mmdet.models.layers.transformer.utils import MLP, coordinate_to_encoding, inverse_sigmoid
from torch import Tensor, nn

from my_modules.layers import CustomDeformableDetrTransformerDecoderLayer


class MyTransformerDecoder(DetrTransformerDecoder):
    """Transformer encoder of DINO."""
    def __init__(self, with_dn=False, *args, **kwargs):
        self.with_dn = with_dn
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = nn.ModuleList([
            CustomDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        if self.with_dn:
            self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                      self.embed_dims, 2)
        # self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, query_pos: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tensor:
        intermediate = []
        # intermediate_reference_points = [reference_points]  # DINO add the initial reference_points
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
            if self.with_dn:
                query_sine_embed = coordinate_to_encoding(  # DINO compute query_pos based on each layer's referece_points
                    reference_points_input[:, :, 0, :])
                query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points, eps=1e-5)  # DINO use eps=1e-3
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                # intermediate.append(self.norm(query))   # DINO add apply LayerNorm on each intermediate output
                intermediate.append(query)  # DINO add apply LayerNorm on each intermediate output
                # intermediate_reference_points.append(new_reference_points)    # DINO add the reference un-detached
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), intermediate_reference_points

        return query, reference_points
