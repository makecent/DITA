import warnings
from pathlib import Path

import mmengine
import numpy as np
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class Thumos14FeatDataset(BaseDetDataset):
    """Thumos14 dataset for temporal action detection."""

    metainfo = dict(classes=('Ambiguous', 'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                             'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
                             'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                             'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'))

    def __init__(self,
                 feat_stride,  # feature are extracted every n frames
                 skip_short=False,  # skip too short annotations
                 ambiguous=False,  # whether track ambiguous annotations. Set False if you are not going to use them.
                 **kwargs):
        self.feat_stride = feat_stride
        self.skip_short = skip_short
        self.ambiguous = ambiguous
        super(Thumos14FeatDataset, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        ann_file = mmengine.load(self.ann_file)
        for video_name, video_info in ann_file.items():
            # Loading features
            feat_path = Path(self.data_prefix['feat']).joinpath(video_name)
            if mmengine.exists(str(feat_path) + '.npy'):
                feat = np.load(str(feat_path) + '.npy')
            else:
                warnings.warn(f"Cannot find feature file {str(feat_path)}, skipped")
                continue
            feat_len = len(feat)

            data_info = dict(video_name=video_name,
                             duration=float(video_info['duration']),
                             fps=float(video_info['FPS']),
                             feat_stride=self.feat_stride,
                             feat_len=feat_len,
                             feat=feat)

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
                label = self.metainfo['classes'].index(label) - 1  # minus 1 so that Ambiguous is labeled as -1.
                if self.ambiguous:
                    segments.append(segment)
                    labels.append(label)
                    ignore_flags.append(1 if label == -1 else 0)
                elif label != -1:
                    segments.append(segment)
                    labels.append(label)
                else:
                    continue

            data_info.update(dict(segments=np.array(segments, dtype=np.float32),
                                  labels=np.array(labels, dtype=np.int64)))
            if self.ambiguous:
                data_info.update(dict(gt_ignore_flags=np.array(ignore_flags, dtype=np.float32)))
            data_list.append(data_info)

        return data_list
