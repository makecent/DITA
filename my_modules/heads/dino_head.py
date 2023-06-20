from typing import Dict, List, Tuple

from mmdet.models.dense_heads import DINOHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from torch import Tensor


@MODELS.register_module()
class CustomDINOHead(DINOHead):
    """
    Computes losses and predictions for the DINO model.
    This customized DINOHead computes loss and predictions
    in a single forward pass for improved computational efficiency,
    without any functional changes from the original DINOHead.
    """

    def loss_and_predict(
            self,
            hidden_states: Tensor, references: List[Tensor],
            enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
            batch_data_samples: SampleList, dn_meta: Dict[str, int],
            rescale: bool = False) -> Tuple[dict, InstanceList]:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return losses, predictions
