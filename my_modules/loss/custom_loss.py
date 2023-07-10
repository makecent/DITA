import torch
import torch.nn.functional as F
from mmdet.models.losses import L1Loss, GIoULoss, IoULoss, FocalLoss, weight_reduce_loss
from mmdet.models.task_modules import IoUCost, BBoxL1Cost
from mmdet.registry import MODELS
from mmdet.registry import TASK_UTILS
from mmengine.structures import InstanceData
from torch import Tensor


def zero_out_loss_coordinates_decorator(forward_method):
    def wrapper(self, pred: Tensor, target: Tensor, *args, **kwargs):
        pred = pred.clone()
        pred[:, 1] = pred[:, 1] * 0 + target[:, 1]
        pred[:, 3] = pred[:, 3] * 0 + target[:, 3]
        return forward_method(self, pred, target, *args, **kwargs)

    return wrapper


def zero_out_pred_coordinates_decorator(forward_method):
    def wrapper(self, pred_instances: InstanceData, gt_instances: InstanceData, *args, **kwargs):
        pred_instances.bboxes[:, 1] = gt_instances.bboxes[0, 1]
        pred_instances.bboxes[:, 3] = gt_instances.bboxes[0, 3]
        return forward_method(self, pred_instances, gt_instances, *args, **kwargs)

    return wrapper

def py_sigmoid_focal_loss(pred,
                          target,
                          iou,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (iou - pred_sigmoid) * target*iou + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@MODELS.register_module()
class PositionFocalLoss(FocalLoss):
    def forward(self,
                pred,
                target,
                iou,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if pred.dim() == target.dim():
                # this means that target is already in One-Hot form.
                calculate_loss_func = py_sigmoid_focal_loss
            # elif torch.cuda.is_available() and pred.is_cuda:
            #     calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                iou,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls


@MODELS.register_module(force=True)
class CustomL1Loss(L1Loss):
    """Custom L1 loss so that y1, y2 don't contribute to the loss by multiplying them with zeros."""

    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        return super().forward(pred, target, *args, **kwargs)


@MODELS.register_module(force=True)
class CustomIoULoss(IoULoss):
    """Custom IoU loss so that y1, y2 don't contribute to the loss by multiplying them with zeros."""

    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        return super().forward(pred, target, *args, **kwargs)


@MODELS.register_module(force=True)
class CustomGIoULoss(GIoULoss):
    """Custom GIoU loss so that y1, y2 don't contribute to the loss by multiplying them with zeros
    """

    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        return super().forward(pred, target, *args, **kwargs)


@TASK_UTILS.register_module(force=True)
class CustomIoUCost(IoUCost):
    @zero_out_pred_coordinates_decorator
    def __call__(self, pred_instances: InstanceData, gt_instances: InstanceData, *args, **kwargs):
        return super().__call__(pred_instances, gt_instances, *args, **kwargs)


@TASK_UTILS.register_module(force=True)
class CustomBBoxL1Cost(BBoxL1Cost):
    @zero_out_pred_coordinates_decorator
    def __call__(self, pred_instances: InstanceData, gt_instances: InstanceData, *args, **kwargs):
        return super().__call__(pred_instances, gt_instances, *args, **kwargs)
