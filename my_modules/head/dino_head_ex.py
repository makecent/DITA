import torch
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.structures import InstanceData
from torch import Tensor

from my_modules.head.dino_head import CustomDINOHead


@MODELS.register_module()
class MyMultiLevelHead(CustomDINOHead):
    # Modify the head so that the queries are split into different groups and each group of queries are
    # trained with only bbox targets in a specific range. Different group handle different ranges of targets.
    # This can help each query focus on its own target range to improve the performance.
    """
    Args:
        range_list (tuple[float]): The list of ranges of targets.
        range_prob_list (tuple[float]): The list of probabilities of each range.
    """

    def __init__(self,
                 range_list=(0.04, 0.07, 0.11, 0.17),
                 range_prob_list=(0.2, 0.2, 0.2, 0.2, 0.2),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_ranges = len(range_list) + 1
        self.range_prob_list = range_prob_list
        if range_prob_list is None:
            self.range_prob_list = (1.0 / self.num_ranges,) * self.num_ranges
        else:
            assert len(range_prob_list) == self.num_ranges
            assert sum(range_prob_list) == 1.0
        self.range_list = (0, *range_list, 1.0)
        # train_256 segments len distribution = (0.04, 0.07, 0.11, 0.17)
        # test segments len distribution = (0.05, 0.11, 0.18, 0.25)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """
        We modify the get_targets_single function to split the queries into different groups and each group of
        queries are trained with only bbox targets in a specific range.
        Below are original comments:
        Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # split predictions to levels
        total_num_queries = cls_score.size(0)
        num_queries_per_group = [0] + [int(prob * total_num_queries) for prob in self.range_prob_list]
        cut_points = torch.cumsum(torch.tensor(num_queries_per_group), dim=0)
        cls_score_mlvl, bbox_pred_mlvl = [], []
        for i in range(self.num_ranges):
            cls_score_mlvl.append(cls_score[cut_points[i]: cut_points[i + 1]])
            bbox_pred_mlvl.append(bbox_pred[cut_points[i]: cut_points[i + 1]])

        # split_target_to_levels
        _, img_w = img_meta['img_shape']
        range_list = [prob * img_w for prob in self.range_list]
        gt_bboxes = gt_instances.bboxes
        bboxes_len = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_instance_mlvl = []
        for j in range(self.num_ranges):
            gt_instance_mlvl.append(gt_instances[torch.logical_and(range_list[j] < bboxes_len, bboxes_len <= range_list[j + 1])])

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single_level,
                                                     cls_score_mlvl, bbox_pred_mlvl,
                                                     gt_instance_mlvl, [img_meta] * self.num_ranges)
        for k, (pos_inds, neg_inds) in enumerate(zip(pos_inds_list, neg_inds_list)):
            pos_inds_list[k] = pos_inds + cut_points[k]
            neg_inds_list[k] = neg_inds + cut_points[k]

        return tuple(map(torch.cat, (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                         pos_inds_list, neg_inds_list)))

    def _get_targets_single_level(self, cls_score: Tensor, bbox_pred: Tensor,
                                  gt_instances: InstanceData,
                                  img_meta: dict) -> tuple:
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)
