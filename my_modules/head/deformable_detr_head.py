from mmdet.models.dense_heads import DeformableDETRHead

from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import (InstanceList)
from mmengine.structures import InstanceData
from torch import Tensor

from my_modules.layers.pseudo_layers import Pseudo4DLinear


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
            reg_branch[-1] = Pseudo4DLinear(self.embed_dims)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.constant_(self.reg_branches[0][-1].bias.data[1:], -2.0)  # [2:] -> [1:]
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[1:], 0.0)  # [2:] -> [1:]

    def loss_and_predict(
            self,
            hidden_states: Tensor, references: List[Tensor],
            enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
            batch_data_samples: SampleList,
            rescale: bool = False) -> Tuple[dict, InstanceList]:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return losses, predictions

    # TODO: below is for bugging purpose, remove it later

    # def _predict_by_feat_single(self,
    #                             cls_score: Tensor,
    #                             bbox_pred: Tensor,
    #                             img_meta: dict,
    #                             rescale: bool = True) -> InstanceData:
    #     """Transform outputs from the last decoder layer into bbox predictions
    #     for each image.
    #
    #     Args:
    #         cls_score (Tensor): Box score logits from the last decoder layer
    #             for each image. Shape [num_queries, cls_out_channels].
    #         bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
    #             for each image, with coordinate format (cx, cy, w, h) and
    #             shape [num_queries, 4].
    #         img_meta (dict): Image meta info.
    #         rescale (bool): If True, return boxes in original image
    #             space. Default True.
    #
    #     Returns:
    #         :obj:`InstanceData`: Detection results of each image
    #         after the post process.
    #         Each item usually contains following keys.
    #
    #             - scores (Tensor): Classification scores, has a shape
    #               (num_instance, )
    #             - labels (Tensor): Labels of bboxes, has a shape
    #               (num_instances, ).
    #             - bboxes (Tensor): Has a shape (num_instances, 4),
    #               the last dimension 4 arrange as (x1, y1, x2, y2).
    #     """
    #     assert len(cls_score) == len(bbox_pred)  # num_queries
    #     max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
    #     img_shape = img_meta['img_shape']
    #     # exclude background
    #     if self.loss_cls.use_sigmoid:
    #         cls_score = cls_score.sigmoid()
    #         scores, indexes = cls_score.view(-1).topk(max_per_img)
    #         det_labels = indexes % self.num_classes
    #         bbox_index = indexes // self.num_classes
    #         bbox_pred = bbox_pred[bbox_index]
    #     else:
    #         scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
    #         scores, bbox_index = scores.topk(max_per_img)
    #         bbox_pred = bbox_pred[bbox_index]
    #         det_labels = det_labels[bbox_index]
    #
    #     det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
    #     det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
    #     det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
    #     # det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
    #     # det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
    #     if rescale:
    #         assert img_meta.get('scale_factor') is not None
    #         det_bboxes /= det_bboxes.new_tensor(
    #             img_meta['scale_factor']).repeat((1, 2))
    #
    #     results = InstanceData()
    #     results.bboxes = det_bboxes
    #     results.scores = scores
    #     results.labels = det_labels
    #     return results
