import torch.nn
from mmdet.registry import MODELS


@MODELS.register_module()
class PseudoBackbone(torch.nn.Module):
    def forward(self, x):
        return [x]  # mimic the multi-scale feature
