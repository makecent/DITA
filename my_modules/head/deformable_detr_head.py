import copy
from typing import Dict, List

import torch
import torch.nn as nn
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList
from torch import Tensor

from my_modules.layers.pseudo_layers import Pseudo4DRegLinear


@MODELS.register_module()
class CustomDeformableDETRHead(DeformableDETRHead):
    """
    Customized Deformable DETR Head to support Temporal Action Detection.
    1. We modify the regression branches to output 2 (x1, x2) rather than 4 (x1, y1, x2, y2).
    1. The original head doesn't have a correct loss_and_predict() function, we add it.
    """

    def _init_layers(self) -> None:
        """Change the regression output dimension from 4 to 2"""
        super()._init_layers()
        for reg_branch in self.reg_branches:
            reg_branch[-1] = Pseudo4DRegLinear(self.embed_dims)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.constant_(self.reg_branches[0][-1].bias.data[1:], -2.0)  # [2:] -> [1:]
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[1:], 0.0)  # [2:] -> [1:]

    def loss_by_feat(
            self,
            all_layers_cls_scores: Tensor,
            all_layers_bbox_preds: Tensor,
            enc_cls_scores: Tensor,
            enc_bbox_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passes in,
                otherwise, it would be `None`.
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = super(DeformableDETRHead, self).loss_by_feat(all_layers_cls_scores,
                                         all_layers_bbox_preds,
                                         batch_gt_instances, batch_img_metas,
                                         batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # proposal_gt_instances = copy.deepcopy(batch_gt_instances)
            # for i in range(len(proposal_gt_instances)):
            #     proposal_gt_instances[i].labels = torch.zeros_like(
            #         proposal_gt_instances[i].labels)
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        return loss_dict
