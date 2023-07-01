from mmdet.models.detectors import DeformableDETR
from mmdet.registry import MODELS

from my_modules.layers.pseudo_layers import Pseudo2DLinear
from my_modules.loss.positional_encoding import PositionEmbeddingSine
from ..layers import CustomDeformableDetrTransformerDecoder, CustomDeformableDetrTransformerEncoder


@MODELS.register_module()
class CustomDeformableDETR(DeformableDETR):
    """
    The Customized DeformableDETR that input memory (encoder output) into the head.
    This customized DeformableDETR output memory into the head for the RoI purpose,
    without any functional changes from the original DeformableDETR.
    """

    def _init_layers(self) -> None:
        pos_cfg = self.positional_encoding
        enc_cfg = self.encoder
        dec_cfg = self.decoder
        super()._init_layers()
        self.encoder = CustomDeformableDetrTransformerEncoder(**enc_cfg)
        self.decoder = CustomDeformableDetrTransformerDecoder(**dec_cfg)
        self.positional_encoding = PositionEmbeddingSine(
            **pos_cfg)
        if not self.as_two_stage:
            self.reference_points_fc = Pseudo2DLinear(self.embed_dims, 1)
