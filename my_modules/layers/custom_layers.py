from mmdet.models.layers import DeformableDetrTransformerEncoder, DeformableDetrTransformerDecoder, \
    DeformableDetrTransformerDecoderLayer, DeformableDetrTransformerEncoderLayer
from mmengine.model import ModuleList


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


class CustomDeformableDetrTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    @zero_y_reference_points
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class CustomDeformableDetrTransformerEncoderLayer(DeformableDetrTransformerEncoderLayer):

    @zero_y_reference_points
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
