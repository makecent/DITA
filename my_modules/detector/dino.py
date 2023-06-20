from typing import Dict, Tuple

import torch
from mmdet.models.detectors import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from torch import Tensor


@MODELS.register_module()
class CustomDINO(DINO):
    """
    The Customized DINO that input memory (encoder output) into the head.
    This customized DINO output memory into the head for the RoI purpose,
    without any functional changes from the original DINO.
    """

    def forward_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        # louis: input encoder memory into the head for RoIAlign and actionness regression.
        memory = encoder_outputs_dict['memory']
        level_start_index = decoder_inputs_dict['level_start_index'].cpu()
        # memory [N, W, C] -> [N, C, W] -> [N, C, 1, W] -> split on last dimension (W)
        mlvl_memory = torch.tensor_split(memory.transpose(1, 2).unsqueeze(2), level_start_index[1:], dim=-1)
        head_inputs_dict['memory'] = mlvl_memory
        return head_inputs_dict
