# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
from mmdet.models.detectors import DeformableDETR
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from torch import Tensor
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

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        # louis: input encoder memory into2 the head for RoIAlign and actionness regression.
        memory = encoder_outputs_dict['memory']
        level_start_index = decoder_inputs_dict['level_start_index'].cpu()
        # memory [N, W, C] -> [N, C, W] -> [N, C, 1, W] -> split on last dimension (W)
        mlvl_memory = torch.tensor_split(memory.transpose(1, 2).unsqueeze(2), level_start_index[1:], dim=-1)
        head_inputs_dict['memory'] = mlvl_memory
        return head_inputs_dict
