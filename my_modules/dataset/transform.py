import random
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from torch.nn import functional as F


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
    # Either the segments are empty or the length of segments' last dimenstion is 2
    assert (segments1.size(-1) == 2 or segments1.size(0) == 0)
    assert (segments2.size(-1) == 2 or segments2.size(0) == 0)

    rows = segments1.size(0)
    cols = segments2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        ious = segments1.new(rows, 1) if is_aligned else segments2.new(rows, cols)
        return ious.numpy() if is_numpy else ious

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


# Separate out the just loading function as a new transform

@TRANSFORMS.register_module()
class FeatDecode(BaseTransform):

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        offset_feat, feat_len, feat_path = results['offset_feat'], results['feat_len'], results['feat_path']
        feat = np.load(feat_path)[offset_feat: offset_feat + feat_len]
        results['feat'] = feat
        results['feat_len'] = len(feat)
        return results


@TRANSFORMS.register_module()
class RandCropFeat(BaseTransform):

    def __init__(self, min_crop_ratio=0.9):
        self.min_crop_ratio = min_crop_ratio

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        feat, feat_len = results['feat'], results['feat_len']
        crop_len = int(random.uniform(self.min_crop_ratio, 1) * feat_len)
        feat = feat[:crop_len]
        results['feat'] = feat
        results['feat_len'] = len(feat)
        return results


@TRANSFORMS.register_module()
class PadFeat(BaseTransform):

    def __init__(self, pad_len=256):
        self.pad_len = pad_len

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        feat, feat_len = results['feat'], results['feat_len']
        if feat_len < self.pad_len:
            feat = np.pad(feat, ((0, self.pad_len - feat_len), (0, 0)), constant_values=0)
            results['feat'] = feat
        return results


@TRANSFORMS.register_module()
class SlidingWindow(BaseTransform):

    def __init__(self,
                 window_size: int,  # the feature length input to the model
                 iof_thr=0.75,
                 attempts=1000,
                 crop_ratio=None):
        self.window_size = window_size
        # Only windows with IoF (Intersection over Foreground) > iof_thr for at least one action are valid.
        self.iof_thr = iof_thr
        self.attempts = attempts
        self.crop_ratio = crop_ratio

    @staticmethod
    def get_valid_mask(segments, patch, iof_thr, ignore_flags=None):
        gt_iofs = segment_overlaps(segments, patch, mode='iof')[:, 0]
        # patch_iofs = segment_overlaps(patch, segments, mode='iof')[0, :]
        # iofs = np.maximum(gt_iofs, patch_iofs)
        # mask = iofs >= iof_thr
        mask = gt_iofs >= iof_thr
        if ignore_flags is not None:
            mask = mask & ~ignore_flags
        return mask

    def transform(self,
                  results: Dict, ) -> Optional[Union[Dict, Tuple[List, List]]]:
        feat, feat_len = results['feat'], results['feat_len']
        # Convert the format of segment annotations from second-unit to feature-unit.
        # feat_stride tells that we extract one feature for every 'feat_stride' frames
        segments_feat = results['segments'] * results['fps'] / results['feat_stride']

        # Conduct sliding window
        if feat_len > self.window_size:
            crop_size = self.window_size
        elif self.crop_ratio is not None:
            crop_size = random.randint(
                max(round(self.crop_ratio[0] * feat_len), 1),
                min(round(self.crop_ratio[1] * feat_len), feat_len))
        else:
            crop_size = feat_len

        for i in range(self.attempts):
            start_idx = random.randint(0, feat_len - crop_size)
            end_idx = start_idx + crop_size

            # If no segments in the cropped window, then re-crop. Ignored segments (Ambiguous) do not count.
            valid_mask = self.get_valid_mask(segments_feat,
                                             np.array([[start_idx, end_idx]], dtype=np.float32),
                                             iof_thr=self.iof_thr,
                                             ignore_flags=results.get('gt_ignore_flags',
                                                                      np.full(segments_feat.shape[0], False)))
            if not valid_mask.any():
                continue

            # Convert the segment annotations to be relative to the cropped window.
            segments_feat = segments_feat[valid_mask].clip(min=start_idx, max=end_idx) - start_idx
            results['segments'] = segments_feat
            results['labels'] = results['labels'][valid_mask]
            if 'gt_ignore_flags' in results:
                results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_mask]
            results['feat'] = feat[start_idx: end_idx]
            results['feat_len'] = crop_size
            break
        else:
            raise RuntimeError(
                f"Could not found a valid crop after {self.attempts} attempts, "
                f"you may need modify the window size or number of attempts")

        return results


@TRANSFORMS.register_module()
class LoadFeat(BaseTransform):

    def transform(self, results: Dict, ) -> Optional[Union[Dict, Tuple[List, List]]]:
        results['feat'] = np.load(results['feat_path'])

        return results


@TRANSFORMS.register_module()
class RescaleFeat(BaseTransform):

    def __init__(self, window_size: int, training=True):  # the feature length input to the model:
        self.window_size = window_size
        self.training = training

    def transform(self, results: Dict, ) -> Optional[Union[Dict, Tuple[List, List]]]:
        # Perform linear interpolation to fix the length of each feat
        feat, feat_len, segments = results['feat'], results['feat_len'], results['segments']
        feat = F.interpolate(torch.from_numpy(np.ascontiguousarray(feat.transpose())).unsqueeze(0),
                             size=self.window_size,
                             mode='linear', align_corners=False).squeeze(0).numpy().transpose()
        if self.training:
            # segments_f = segments * self.window_size / data_info['duration']  # solution 1
            segments = segments * self.window_size / feat_len  # solution 2
        results['feat'] = feat
        results['feat_len'] = self.window_size
        results['segments'] = segments
        # scale_factor is be used in the inference stage to convert
        # the predicted segments from feature level to video level (in seconds)
        results['scale_factor'] = results['scale_factor'] * self.window_size / feat_len
        return results


@TRANSFORMS.register_module()
class ReFormat(BaseTransform):

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        # [x1 x2] to [x1 y1 x2 y2]
        gt_bboxes = np.insert(results['segments'], 2, 0.9, axis=-1)
        gt_bboxes = np.insert(gt_bboxes, 1, 0.1, axis=-1)
        results['gt_bboxes'] = gt_bboxes
        if 'overlap' in results and results['overlap'].size > 0:
            overlap = np.insert(results['overlap'], 2, 0.9, axis=-1)
            overlap = np.insert(overlap, 1, 0.1, axis=-1)
            results['overlap'] = overlap

        results.update({'gt_bboxes_labels': results.pop('labels')})
        results.update({"img_id": results.pop("video_name")})
        results.update({'img': results.pop('feat')[None]})
        results.update({'ori_shape': (1, results.pop('feat_len'))})
        # results.update({'img_shape': results['ori_shape']})
        results.update({'img_shape': (1, results['img'].shape[1])})

        results['img_path'] = ''
        if 'scale_factor' not in results:
            results['scale_factor'] = [1.0, 1.0]
        else:
            results['scale_factor'] = [results['scale_factor'], 1.0]  # [w_scale, h_scale]
        results['flip'] = False
        results['flip_direction'] = None
        return results
