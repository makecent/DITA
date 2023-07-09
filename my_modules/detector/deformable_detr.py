from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.detectors import DeformableDETR
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from torch import Tensor, nn

from my_modules.layers.pseudo_layers import Pseudo2DLinear
from my_modules.loss.positional_encoding import CustomSinePositionalEncoding
from ..layers import CustomDeformableDetrTransformerDecoder, CustomDeformableDetrTransformerEncoder
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS

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
        self.positional_encoding = CustomSinePositionalEncoding(
            **pos_cfg)
        if not self.as_two_stage:
            self.reference_points_fc = Pseudo2DLinear(self.embed_dims, 2, delta=False)
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims * 2)
        self.pos_trans_fc = nn.Identity()
        self.pos_trans_norm = nn.Identity()

        # self.pos_trans_fc = nn.Linear(self.embed_dims,
        #                               self.embed_dims)
        # self.pos_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            # nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(
                self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). It will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
                It will only be used when `as_two_stage` is `True`.

        Returns:
            tuple[dict, dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and `reference_points`. The reference_points of
              decoder input here are 4D boxes when `as_two_stage` is `True`,
              otherwise 2D points, although it has `points` in its name.
              The reference_points in encoder is always 2D points.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `enc_outputs_class` and
              `enc_outputs_coord`. They are both `None` when 'as_two_stage'
              is `False`. The dict is empty when `self.training` is `False`.
        """
        batch_size, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                                          self.decoder.num_layers](output_memory) + output_proposals
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            # We only use the first channel in enc_outputs_class as foreground,
            # the other (num_classes - 1) channels are actually not used.
            # Its targets are set to be 0s, which indicates the first
            # class (foreground) because we use [0, num_classes - 1] to
            # indicate class labels, background class is indicated by
            # num_classes (similar convention in RPN).
            # See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/deformable_detr_head.py#L241 # noqa
            # This follows the official implementation of Deformable DETR.
            topk_proposals = torch.topk(
                enc_outputs_class.max(-1)[0], self.num_queries, dim=1)[1]
            cls_out_features = self.bbox_head.cls_branches[self.decoder.num_layers].out_features
            topk_scores = torch.gather(
                enc_outputs_class, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            # pos_trans_out = self.pos_trans_fc(
            #     self.get_proposal_pos_embed(topk_coords_unact[..., ::2], num_pos_feats=256))
            # pos_trans_out = self.pos_trans_norm(pos_trans_out)
            # query_pos, query = torch.split(pos_trans_out, c, dim=2)

            # query = self.query_embedding.weight[:, None, :]
            # query = query.repeat(1, batch_size, 1).transpose(0, 1)

            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)

        else:
            enc_outputs_class, enc_outputs_coord = None, None
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=topk_scores,
            enc_outputs_coord=topk_coords) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    # def pre_transformer(
    #         self,
    #         mlvl_feats: Tuple[Tensor],
    #         batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
    #     """Process image features before feeding them to the transformer.
    #
    #     The forward procedure of the transformer is defined as:
    #     'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
    #     More details can be found at `TransformerDetector.forward_transformer`
    #     in `mmdet/detector/base_detr.py`.
    #
    #     Args:
    #         mlvl_feats (tuple[Tensor]): Multi-level features that may have
    #             different resolutions, output from neck. Each feature has
    #             shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
    #         batch_data_samples (list[:obj:`DetDataSample`], optional): The
    #             batch data samples. It usually includes information such
    #             as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
    #             Defaults to None.
    #
    #     Returns:
    #         tuple[dict]: The first dict contains the inputs of encoder and the
    #         second dict contains the inputs of decoder.
    #
    #         - encoder_inputs_dict (dict): The keyword args dictionary of
    #           `self.forward_encoder()`, which includes 'feat', 'feat_mask',
    #           and 'feat_pos'.
    #         - decoder_inputs_dict (dict): The keyword args dictionary of
    #           `self.forward_decoder()`, which includes 'memory_mask'.
    #     """
    #     batch_size = mlvl_feats[0].size(0)
    #
    #     # construct binary masks for the transformer.
    #     assert batch_data_samples is not None
    #     batch_input_shape = batch_data_samples[0].batch_input_shape
    #     img_shape_list = [sample.img_shape for sample in batch_data_samples]
    #     input_img_h, input_img_w = batch_input_shape
    #     masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
    #     for img_id in range(batch_size):
    #         img_h, img_w = img_shape_list[img_id]
    #         masks[img_id, :img_h, :img_w] = 0
    #     # NOTE following the official DETR repo, non-zero values representing
    #     # ignored positions, while zero values means valid positions.
    #
    #     mlvl_masks = []
    #     mlvl_pos_embeds = []
    #     for feat in mlvl_feats:
    #         mlvl_masks.append(
    #             F.interpolate(masks[None],
    #                           size=feat.shape[-2:]).to(torch.bool).squeeze(0))
    #         mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))
    #
    #     feat_flatten = []
    #     lvl_pos_embed_flatten = []
    #     mask_flatten = []
    #     spatial_shapes = []
    #     for lvl, (feat, mask, pos_embed) in enumerate(
    #             zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
    #         batch_size, c, h, w = feat.shape
    #         # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
    #         feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
    #         pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
    #         lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
    #         # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
    #         mask = mask.flatten(1)
    #         spatial_shape = (h, w)
    #
    #         feat_flatten.append(feat)
    #         lvl_pos_embed_flatten.append(lvl_pos_embed)
    #         mask_flatten.append(mask)
    #         spatial_shapes.append(spatial_shape)
    #
    #     # (bs, num_feat_points, dim)
    #     feat_flatten = torch.cat(feat_flatten, 1)
    #     lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    #     # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
    #     mask_flatten = torch.cat(mask_flatten, 1)
    #
    #     spatial_shapes = torch.as_tensor(  # (num_level, 2)
    #         spatial_shapes,
    #         dtype=torch.long,
    #         device=feat_flatten.device)
    #     level_start_index = torch.cat((
    #         spatial_shapes.new_zeros((1,)),  # (num_level)
    #         spatial_shapes.prod(1).cumsum(0)[:-1]))
    #     valid_ratios = torch.stack(  # (bs, num_level, 2)
    #         [self.get_valid_ratio(m) for m in mlvl_masks], 1)
    #
    #     encoder_inputs_dict = dict(
    #         feat=feat_flatten,
    #         feat_mask=mask_flatten,
    #         feat_pos=lvl_pos_embed_flatten,
    #         spatial_shapes=spatial_shapes,
    #         level_start_index=level_start_index,
    #         valid_ratios=valid_ratios)
    #     decoder_inputs_dict = dict(
    #         memory_mask=mask_flatten,
    #         spatial_shapes=spatial_shapes,
    #         level_start_index=level_start_index,
    #         valid_ratios=valid_ratios)
    #     return encoder_inputs_dict, decoder_inputs_dict
