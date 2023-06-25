import torch
from torch import nn


class Pseudo4DLinear(nn.Linear):
    """
    This is a pseudo linear layer whose out_dimension is 4 but the inner linear computation
    actually is computed based on out_dimension=2.
    Note that our computation is based on that the default y1, y2 are 0.1 and 0.9, respectively.
    """

    def __init__(self, in_dim, format='cxcywh'):
        super().__init__(in_dim, 2)
        assert format in ['cxcywh', 'xyxy']
        self.format = format

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        if self.format == 'cxcywh':
            cy = torch.full_like(x[..., :1], 0.5)
            h = torch.full_like(x[..., :1], 0.8)
            x = torch.cat((x[..., :1], cy, x[..., 1:], h), dim=-1)
        else:
            y1 = torch.full_like(x[..., :1], 0.1)
            y2 = torch.full_like(x[..., :1], 0.9)
            x = torch.cat((x[..., :1], y1, x[..., 1:], y2), dim=-1)
        return x


class Pseudo2DLinear(nn.Linear):
    """
    This is a pseudo linear layer whose out_dimension is 4 but the inner linear computation
    actually is computed based on out_dimension=2.
    Note that our computation is based on that the default y1, y2 are 0.1 and 0.9, respectively.
    """

    def forward(self, x: torch.Tensor):
        x = super().forward(x)

        # Interleave the output values with zeros (the offsets on y-axis)
        reshaped_output = x.view(-1, 1)
        zeros_tensor = torch.zeros_like(reshaped_output)
        interleaved_output = torch.cat((reshaped_output, zeros_tensor), dim=1)
        x = interleaved_output.view(*x.shape[:-1], x.shape[-1]*2)
        return x
