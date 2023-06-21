from typing import List, Tuple

from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from torch import nn, Tensor

from .pseudo_4d_fc import Pseudo4DLinear


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
