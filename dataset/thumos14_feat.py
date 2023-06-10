import warnings
from pathlib import Path

import mmengine
import numpy as np
from mmaction.datasets import BaseActionDataset
from mmaction.registry import DATASETS


@DATASETS.register_module()
class Thumos14FeatDataset(BaseActionDataset):
    """Thumos14 dataset for temporal action detection."""

    metainfo = dict(classes=('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                             'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
                             'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
                             'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                             'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
                             'VolleyballSpiking'))

    def __init__(self,
                 feat_stride=8,  # feature are extracted every frames
                 **kwargs):
        self.feat_stride = feat_stride
        super(Thumos14FeatDataset, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        data = mmengine.load(self.ann_file)
        for video_name, video_info in data['database'].items():
            # Loading features
            feat_path = Path(self.data_prefix['feats']).joinpath(video_name)
            feat = mmengine.load(str(feat_path))
            feat_len = len(feat)

            data_info = dict(video_name=video_name,
                             feat=feat,
                             feat_stride=self.feat_stride,
                             feat_len=feat_len,
                             duration=float(video_info['duration']))

            # Segments information
            segments = []
            labels = []
            ignore_flags = []
            video_duration = video_info['duration']
            for ann in video_info['annotations']:
                label = ann['label']
                segment = ann['segment']
                assert 0 <= segment[0] <= video_duration, f"invalid segment annotation {segment}"
                assert segment[0] < segment[1] <= video_duration, f"invalid segment annotation {segment}"

                if label in self.metainfo['classes']:
                    ignore_flags.append(0)
                    labels.append(self.metainfo['classes'].index(label))
                else:
                    ignore_flags.append(1)
                    labels.append(-1)
                segments.append(segment)

            if not segments or np.all(ignore_flags):
                warnings.warn(f'No valid segments found in video {video_name}. Excluded')
                continue

            data_info.update(dict(
                segments=np.array(segments, dtype=np.float32),
                labels=np.array(labels, dtype=np.int64),
                ignore_flags=np.array(ignore_flags, dtype=np.float32)))

            data_list.append(data_info)

            # standard_ann_file = dict()
            # standard_ann_file['metainfo'] = dict(classes=self.CLASSES)
            # standard_ann_file['data_list'] = data_list
            # mmengine.dump(standard_ann_file, 'train.json')
        return data_list


from typing import Dict, Optional, Tuple, List, Union

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import random


@TRANSFORMS.register_module()
class SlidingWindow(BaseTransform):

    def __init__(self, window_size: int):
        self.window_size = window_size

    def transform(self,
                  results: Dict, ) -> Optional[Union[Dict, Tuple[List, List]]]:
        feat, feat_len = results['feat'], results['feat_len']
        if feat_len > self.window_size:
            start_index = random.randint(0, feat_len - self.window_size)
            feat_window = feat[start_index: start_index + self.window_size]
            results['feat'] = feat_window
        return results
