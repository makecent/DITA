import warnings
from copy import deepcopy
from pathlib import Path

import mmengine
import numpy as np
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS

from my_modules.dataset.transform import SlidingWindow


@DATASETS.register_module()
class Thumos14FeatDataset(BaseDetDataset):
    """Thumos14 dataset for temporal action detection."""

    metainfo = dict(classes=('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                             'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
                             'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                             'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'))

    def __init__(self,
                 feat_stride,  # feature are extracted every n frames
                 skip_short=False,  # skip too short annotations
                 ambiguous=False,  # whether track ambiguous annotations. Set False if you are not going to use them.
                 fix_slice=True,
                 # whether slice the feature to windows with fixed stride or leave it to pipeline which perform random slice.
                 iof_thr=0.75,  # The Intersection over Foreground (IoF) threshold used to filter sliding windows
                 window_size=None,  # only applicable to testing phase, should be equal to the training window size.
                 window_stride=None,  # only applicable to testing phase, the fixed window stride in testing.
                 **kwargs):
        self.feat_stride = feat_stride
        self.skip_short = skip_short
        self.ambiguous = ambiguous
        self.fix_slice = fix_slice
        self.iof_thr = iof_thr
        self.window_size = window_size
        self.window_stride = window_stride
        if fix_slice:
            assert isinstance(window_size, int)
            assert isinstance(window_stride, int)
        super(Thumos14FeatDataset, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        ann_file = mmengine.load(self.ann_file)
        for video_name, video_info in ann_file.items():
            # Parsing ground truth
            segments, labels, ignore_flags = self.parse_labels(video_name, video_info)

            # Loading features
            feat_path = Path(self.data_prefix['feat']).joinpath(video_name)
            if mmengine.exists(str(feat_path) + '.npy'):
                feat = np.load(str(feat_path) + '.npy')
            else:
                warnings.warn(f"Cannot find feature file {str(feat_path)}, skipped")
                continue

            data_info = dict(video_name=video_name,
                             duration=float(video_info['duration']),
                             fps=float(video_info['FPS']),
                             feat_stride=self.feat_stride,
                             segments=segments,
                             labels=labels,
                             gt_ignore_flags=ignore_flags)

            if not self.fix_slice:
                data_info.update(dict(feat=feat, feat_len=len(feat)))
                if not self.ambiguous:
                    data_info.pop('gt_ignore_flags')
                data_list.append(data_info)
            else:
                # Perform fixed-stride sliding window
                feat_windows = [feat[i: i + self.window_size] for i in range(0, len(feat), self.window_stride)]
                for i, feat_window in enumerate(feat_windows):
                    start_idx = float(i * self.window_stride)
                    feat_win_len = len(feat_window)
                    end_idx = start_idx + feat_win_len

                    # Padding windows that are shorter than the target window size.
                    if feat_win_len < self.window_size:
                        feat_window = np.pad(feat_window,
                                             ((0, self.window_size - feat_win_len), (0, 0)),
                                             constant_values=0)
                    data_info.update(dict(offset=start_idx,
                                          feat_len=feat_win_len,  # before padding for computing the valid feature mask
                                          feat=feat_window))

                    # During the training, windows has no segment annotated are skipped
                    if not self.test_mode:
                        valid_mask = SlidingWindow.get_valid_mask(segments,
                                                                  np.array([[start_idx, end_idx]], dtype=np.float32),
                                                                  iof_thr=self.iof_thr,
                                                                  ignore_flags=ignore_flags)
                        if not valid_mask.any():
                            continue
                        _segments = segments[valid_mask].clip(min=start_idx, max=end_idx) - start_idx
                        _labels = labels[valid_mask]
                        _ignore_flags = ignore_flags[valid_mask]
                        data_info.update(dict(segments=_segments,
                                              labels=_labels,
                                              gt_ignore_flags=_ignore_flags))
                    if not self.ambiguous:
                        data_info.pop('gt_ignore_flags')
                    data_list.append(deepcopy(data_info))
        return data_list

    def parse_labels(self, video_name, video_info):
        # Segments information
        segments = []
        labels = []
        ignore_flags = []
        video_duration = video_info['duration']
        for segment, label in zip(video_info['segments'], video_info['labels']):

            # Skip annotations that are out of range.
            if not (0 <= segment[0] < segment[1] <= video_duration):
                print(f"invalid segment annotation in {video_name}: {segment}, duration={video_duration}, skipped")
                continue

            # Skip annotations that are too short. The threshold could be the stride of feature extraction.
            # For example, if the features were extracted every 8 frames,
            # then the threshold should be greater than 8/30 = 0.27s
            if segment[1] - segment[0] < 0.3 and self.skip_short:
                print(f"too short segment annotation in {video_name}: {segment}, skipped")

            # Skip ambiguous annotations or label them as ignored ground truth
            if label == 'Ambiguous':
                if self.ambiguous:
                    segments.append(segment)
                    labels.append(-1)
                    ignore_flags.append(1)
            else:
                segments.append(segment)
                labels.append(self.metainfo['classes'].index(label))

        return np.array(segments, np.float32), np.array(labels, np.int), np.array(ignore_flags, dtype=np.int)
