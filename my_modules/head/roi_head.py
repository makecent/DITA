# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List

import torch
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi, bbox_overlaps
from mmdet.utils import ConfigType, InstanceList, OptMultiConfig
from mmengine.model import BaseModule
from torch import nn


@MODELS.register_module()
class MyRoIHead(BaseModule):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor: ConfigType,
                 actionness_loss: ConfigType,
                 bbox_head: ConfigType,
                 init_cfg: OptMultiConfig = None,
                 expand_roi_factor=1.5,
                 active=True,  # experimental arguments, set to False can deactivate the RoI
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.active = active
        if active:
            self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
            self.roi_size = bbox_roi_extractor['roi_layer']['output_size'][1]
            self.dim = bbox_roi_extractor['out_channels']
            self.expand_roi_factor = expand_roi_factor
            self.actionness_fc = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.roi_size * self.dim, self.dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim, self.dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim, 1),
                nn.Sigmoid())
            self.actionness_loss = MODELS.build(actionness_loss)
        else:
            warnings.warn("Please note that the RoIHead is now deactivated, no RoI will be applied")

        bbox_head.update(kwargs)
        self.bbox_head = MODELS.build(bbox_head)

    @property
    def num_classes(self):
        return self.bbox_head.num_classes

    @property
    def reg_branches(self):
        return self.bbox_head.reg_branches

    @property
    def cls_branches(self):
        return self.bbox_head.cls_branches

    def forward(self, *args, **kwargs) -> tuple:
        return self.bbox_head(*args, **kwargs)

    def loss(self, batch_data_samples: List[DetDataSample], **head_inputs_dict) -> dict:
        memory = head_inputs_dict.pop('memory')

        if self.active:
            bbox_head_loss, bbox_pred = self.bbox_head.loss_and_predict(batch_data_samples=batch_data_samples,
                                                                        rescale=False,
                                                                        **head_inputs_dict)
            actionness_pred = self.actionness_forward(memory, bbox_pred, batch_data_samples)
            actionness_target = self.get_actionness_target(bbox_pred, batch_data_samples)
            actionness_loss = self.actionness_loss(actionness_pred.reshape(-1),
                                                   torch.cat(actionness_target, 0))
            bbox_head_loss.update(dict(actionness_loss=actionness_loss))
        else:
            bbox_head_loss = self.bbox_head.loss(batch_data_samples=batch_data_samples,
                                                 rescale=False,
                                                 **head_inputs_dict)

        return bbox_head_loss

    def predict(self, batch_data_samples: List[DetDataSample], rescale, **head_inputs_dict) -> InstanceList:
        memory = head_inputs_dict.pop('memory')
        bbox_pred = self.bbox_head.predict(batch_data_samples=batch_data_samples,
                                           rescale=False,
                                           **head_inputs_dict)
        if self.active:
            actionness_pred = self.actionness_forward(memory, bbox_pred, batch_data_samples).reshape(len(bbox_pred), -1)
        else:
            actionness_pred = [pred.scores for pred in bbox_pred]
        return self.post_process(bbox_pred, actionness_pred, batch_data_samples, rescale)

    def actionness_forward(self, memory, bbox_pred, batch_data_samples):
        # Expand the range of (x1, x2)
        ex_bbox_pred = [res.bboxes.clone().detach() for res in bbox_pred]
        for bboxes, data_sample in zip(ex_bbox_pred, batch_data_samples):
            max_len = data_sample.metainfo['img_shape'][1]
            length = bboxes[:, 2] - bboxes[:, 0]
            center = (bboxes[:, 2] + bboxes[:, 0]) / 2
            bboxes[:, 0] = (center - length * self.expand_roi_factor / 2).clamp(min=0, max=max_len)
            bboxes[:, 2] = (center + length * self.expand_roi_factor / 2).clamp(min=0, max=max_len)

        # actionness regression prediction
        rois = bbox2roi(ex_bbox_pred).detach()
        bbox_feats = self.bbox_roi_extractor(memory[:self.bbox_roi_extractor.num_inputs], rois)
        actionness_pred = self.actionness_fc(bbox_feats)
        return actionness_pred

    @staticmethod
    def get_actionness_target(bbox_pred, batch_data_samples):
        batch_bboxes = [res.bboxes for res in bbox_pred]
        batch_gt_bboxes = [data_sample.gt_instances.bboxes for data_sample in batch_data_samples]

        # Fix the y1 y2
        for res in bbox_pred:
            res.bboxes[:, 1] = 0.1
            res.bboxes[:, 3] = 0.9

        actionness_target = []
        for bboxes, gt_bboxes in zip(batch_bboxes, batch_gt_bboxes):
            iou_mat = bbox_overlaps(bboxes, gt_bboxes, mode='iou', is_aligned=False)
            gt_iou = iou_mat.max(dim=1)[0]
            actionness_target.append(gt_iou.detach())

        return actionness_target

    @staticmethod
    def post_process(bbox_pred, actionness_pred, batch_data_samples, rescale):
        for pred, data_sample, actionness in zip(bbox_pred, batch_data_samples, actionness_pred):
            img_meta = data_sample.metainfo
            if rescale:
                assert img_meta.get('scale_factor') is not None
                pred.bboxes /= pred.bboxes.new_tensor(
                    img_meta['scale_factor']).repeat((1, 2))
                # using actionness regression results as confidence scores of bboxes instead of classification score
                pred.scores = torch.sqrt(pred.scores * actionness)
        return bbox_pred
