# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
from torch.nn import functional as F
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
    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        input_img_h, input_img_w = batch_input_shape
        masks = mlvl_feats[0].new_zeros((batch_size, input_img_h, input_img_w))  # Following TadTR, we don't mask padding
        # for img_id in range(batch_size):
        #     img_h, img_w = img_shape_list[img_id]
        #     masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            mask = mask.flatten(1)
            spatial_shape = (h, w)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        mask_flatten = torch.cat(mask_flatten, 1)

        spatial_shapes = torch.as_tensor(  # (num_level, 2)
            spatial_shapes,
            dtype=torch.long,
            device=feat_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(  # (bs, num_level, 2)
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict
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
