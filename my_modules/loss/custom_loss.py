from mmdet.models.losses import L1Loss, IoULoss, GIoULoss
from mmdet.models.task_modules import IoUCost, BBoxL1Cost
from mmdet.registry import MODELS, TASK_UTILS
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
