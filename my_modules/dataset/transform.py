import random
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


def segment_overlaps(segments1,
                     segments2,
                     mode='iou',
                     is_aligned=False,
                     eps=1e-6,
                     detect_overlap_edge=False):
    """Calculate overlap between two set of segments.
    If ``is_aligned`` is ``False``, then calculate the ious between each
    segment of segments1 and segments2, otherwise the ious between each aligned
     pair of segments1 and segments2.
    Args:
        segments1 (Tensor): shape (m, 2) in <t1, t2> format or empty.
        segments2 (Tensor): shape (n, 2) in <t1, t2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    Example:
        >>> segments1 = torch.FloatTensor([
        >>>     [0, 10],
        >>>     [10, 20],
        >>>     [32, 38],
        >>> ])
        >>> segments2 = torch.FloatTensor([
        >>>     [0, 20],
        >>>     [0, 19],
        >>>     [10, 20],
        >>> ])
        >>> segment_overlaps(segments1, segments2)
        tensor([[0.5000, 0.5263, 0.0000],
                [0.0000, 0.4500, 1.0000],
                [0.0000, 0.0000, 0.0000]])
    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 9],
        >>> ])
        >>> assert tuple(segment_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(segment_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(segment_overlaps(empty, empty).shape) == (0, 0)
    """

    is_numpy = False
    if isinstance(segments1, np.ndarray):
        segments1 = torch.from_numpy(segments1)
        is_numpy = True
    if isinstance(segments2, np.ndarray):
        segments2 = torch.from_numpy(segments2)
        is_numpy = True

    segments1, segments2 = segments1.float(), segments2.float()

    assert mode in ['iou', 'iof']
    # Either the segments are empty or the length of segments's last dimenstion
    # is 2
    assert (segments1.size(-1) == 2 or segments1.size(0) == 0)
    assert (segments2.size(-1) == 2 or segments2.size(0) == 0)

    rows = segments1.size(0)
    cols = segments2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return segments1.new(rows, 1) if is_aligned else segments2.new(
            rows, cols)

    if is_aligned:
        start = torch.max(segments1[:, 0], segments2[:, 0])  # [rows]
        end = torch.min(segments1[:, 1], segments2[:, 1])  # [rows]

        overlap = end - start
        if detect_overlap_edge:
            overlap[overlap == 0] += eps
        overlap = overlap.clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        start = torch.max(segments1[:, None, 0], segments2[:,
                                                 0])  # [rows, cols]
        end = torch.min(segments1[:, None, 1], segments2[:, 1])  # [rows, cols]

        overlap = end - start
        if detect_overlap_edge:
            overlap[overlap == 0] += eps
        overlap = overlap.clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    if is_numpy:
        ious = ious.numpy()

    return ious


@TRANSFORMS.register_module()
class SlidingWindow(BaseTransform):

    def __init__(self,
                 window_size: int,  # the feature length input to the model
                 iof_thr=0.75):
        self.window_size = window_size
        # Only windows with IoF (Intersection over Foreground) > iof_thr for at least one action are valid.
        self.iof_thr = iof_thr

    def get_valid_mask(self, segments, patch, iof_thr):
        gt_iofs = segment_overlaps(segments, patch, mode='iof')[:, 0]
        patch_iofs = segment_overlaps(patch, segments, mode='iof')[0, :]
        iofs = np.maximum(gt_iofs, patch_iofs)
        mask = iofs > iof_thr

        return mask

    def transform(self,
                  results: Dict, ) -> Optional[Union[Dict, Tuple[List, List]]]:
        feat, feat_len = results['feat'], results['feat_len']

        # Convert the format of segment annotations from second-unit to feature-unit.
        # feat_stride tells that we extract one feature for every 'feat_stride' frames
        segments = results['segments'] * results['fps'] / results['feat_stride']

        # Conduct sliding window
        if feat_len > self.window_size:
            # If feat_len greater than the window_size, we randomly crop a window_size of feature.
            while True:
                start_idx = random.randint(0, feat_len - self.window_size)
                end_idx = start_idx + self.window_size - 1

                # If no segments in the cropped window, then re-crop. Ignored segments (Ambiguous) do not count.
                valid_mask = self.get_valid_mask(segments, np.array([[start_idx, end_idx]], dtype=np.float32), iof_thr=self.iof_thr)
                valid_mask = valid_mask & ~results.get('gt_ignore_flags', np.full(valid_mask.shape, True))
                if not valid_mask.any():
                    continue

                # Convert the segment annotations to be relative to the cropped window.
                segments = segments[valid_mask].clip(min=start_idx, max=end_idx) - start_idx
                results['segments'] = segments
                results['feat'] = feat[start_idx: end_idx + 1]
                results['feat_len'] = self.window_size
        else:
            # Padding the feature to window size if the feat is too short
            results['segments'] = segments
            results['feat'] = np.pad(feat, ((0, feat_len - self.window_size), (0, 0)), constant_values=0)

        return results


@TRANSFORMS.register_module()
class ReFormat(BaseTransform):

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        results['img_shape'] = (1, results['feat_len'])
        results['img'] = results['feat'][None]
        # [x1 x2] to [x1 y1 x2 y2]
        gt_bboxes = np.insert(results['segments'], 2, 0.9, axis=-1)
        gt_bboxes = np.insert(gt_bboxes, 1, 0.1, axis=-1)
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = results['labels']
