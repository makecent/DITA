import os.path as osp
import warnings
from copy import deepcopy

import mmengine
import numpy as np
import torch
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS
from mmengine import print_log, MMLogger
from torch.nn import functional as F


@DATASETS.register_module()
class ANet13FeatDataset(BaseDetDataset):
    """ActivityNet-1.3 dataset for temporal action detection.
    This version of dataset class is based on the annotation file and features downloaded from ActionFormer repo.
    We are generating our own features and annotation file, and the new dataset class is under development.
    Please stay tuned.
    """

    metainfo = dict(
        classes=('Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle', 'BMX', 'Baking cookies',
                 'Ballet', 'Bathing dog', 'Baton twirling', 'Beach soccer', 'Beer pong', 'Belly dance',
                 'Blow-drying hair', 'Blowing leaves', 'Braiding hair', 'Breakdancing', 'Brushing hair',
                 'Brushing teeth', 'Building sandcastles', 'Bullfighting', 'Bungee jumping', 'Calf roping',
                 'Camel ride', 'Canoeing', 'Capoeira', 'Carving jack-o-lanterns', 'Changing car wheel', 'Cheerleading',
                 'Chopping wood', 'Clean and jerk', 'Cleaning shoes', 'Cleaning sink', 'Cleaning windows',
                 'Clipping cat claws', 'Cricket', 'Croquet', 'Cumbia', 'Curling', 'Cutting the grass',
                 'Decorating the Christmas tree', 'Disc dog', 'Discus throw', 'Dodgeball', 'Doing a powerbomb',
                 'Doing crunches', 'Doing fencing', 'Doing karate', 'Doing kickboxing', 'Doing motocross',
                 'Doing nails', 'Doing step aerobics', 'Drinking beer', 'Drinking coffee', 'Drum corps',
                 'Elliptical trainer', 'Fixing bicycle', 'Fixing the roof', 'Fun sliding down', 'Futsal',
                 'Gargling mouthwash', 'Getting a haircut', 'Getting a piercing', 'Getting a tattoo', 'Grooming dog',
                 'Grooming horse', 'Hammer throw', 'Hand car wash', 'Hand washing clothes', 'Hanging wallpaper',
                 'Having an ice cream', 'High jump', 'Hitting a pinata', 'Hopscotch', 'Horseback riding', 'Hula hoop',
                 'Hurling', 'Ice fishing', 'Installing carpet', 'Ironing clothes', 'Javelin throw', 'Kayaking',
                 'Kite flying', 'Kneeling', 'Knitting', 'Laying tile', 'Layup drill in basketball', 'Long jump',
                 'Longboarding', 'Making a cake', 'Making a lemonade', 'Making a sandwich', 'Making an omelette',
                 'Mixing drinks', 'Mooping floor', 'Mowing the lawn', 'Paintball', 'Painting', 'Painting fence',
                 'Painting furniture', 'Peeling potatoes', 'Ping-pong', 'Plastering', 'Plataform diving',
                 'Playing accordion', 'Playing badminton', 'Playing bagpipes', 'Playing beach volleyball',
                 'Playing blackjack', 'Playing congas', 'Playing drums', 'Playing field hockey', 'Playing flauta',
                 'Playing guitarra', 'Playing harmonica', 'Playing ice hockey', 'Playing kickball', 'Playing lacrosse',
                 'Playing piano', 'Playing polo', 'Playing pool', 'Playing racquetball', 'Playing rubik cube',
                 'Playing saxophone', 'Playing squash', 'Playing ten pins', 'Playing violin', 'Playing water polo',
                 'Pole vault', 'Polishing forniture', 'Polishing shoes', 'Powerbocking', 'Preparing pasta',
                 'Preparing salad', 'Putting in contact lenses', 'Putting on makeup', 'Putting on shoes', 'Rafting',
                 'Raking leaves', 'Removing curlers', 'Removing ice from car', 'Riding bumper cars', 'River tubing',
                 'Rock climbing', 'Rock-paper-scissors', 'Rollerblading', 'Roof shingle removal', 'Rope skipping',
                 'Running a marathon', 'Sailing', 'Scuba diving', 'Sharpening knives', 'Shaving', 'Shaving legs',
                 'Shot put', 'Shoveling snow', 'Shuffleboard', 'Skateboarding', 'Skiing', 'Slacklining',
                 'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Snow tubing', 'Snowboarding', 'Spinning',
                 'Spread mulch', 'Springboard diving', 'Starting a campfire', 'Sumo', 'Surfing', 'Swimming',
                 'Swinging at the playground', 'Table soccer', 'Tai chi', 'Tango', 'Tennis serve with ball bouncing',
                 'Throwing darts', 'Trimming branches or hedges', 'Triple jump', 'Tug of war', 'Tumbling',
                 'Using parallel bars', 'Using the balance beam', 'Using the monkey bar', 'Using the pommel horse',
                 'Using the rowing machine', 'Using uneven bars', 'Vacuuming floor', 'Volleyball', 'Wakeboarding',
                 'Walking the dog', 'Washing dishes', 'Washing face', 'Washing hands', 'Waterskiing', 'Waxing skis',
                 'Welding', 'Windsurfing', 'Wrapping presents', 'Zumba'),
        wrong_videos=())

    def __init__(self,
                 feat_stride,  # feature are extracted every n frames
                 skip_short=False,  # skip too short annotations
                 skip_wrong=False,  # skip videos that are wrong annotated
                 split='training',
                 **kwargs):
        self.feat_stride = feat_stride
        self.skip_short = skip_short
        self.skip_wrong = skip_wrong
        self.split = split
        super(ANet13FeatDataset, self).__init__(**kwargs)

    def load_data_list(self):
        # feat_offset = 0.5 * 16 / self.feat_stride
        data_list = []
        ann_file = mmengine.load(self.ann_file)
        for video_name, video_info in ann_file['database'].items():
            if self.skip_wrong and video_name in self.metainfo['wrong_videos']:
                continue
            if self.split != video_info['subset']:
                continue
            # Parsing ground truth
            segments, labels, ignore_flags = self.parse_labels(video_name, video_info)
            if len(segments) == 0:
                continue

            # Loading features
            feat_path = osp.join(self.data_prefix['feat'], 'v_' + video_name) + '.npy'
            if mmengine.exists(feat_path):
                feat = np.load(feat_path)
            else:
                warnings.warn(f"Cannot find feature file {feat_path}, skipped")
                continue
            fps = 25.0
            if not self.test_mode:
                segments = segments * fps / self.feat_stride
            data_info = dict(video_name=video_name,
                             duration=float(video_info['duration']),
                             fps=fps,
                             feat_path=feat_path,
                             feat_stride=self.feat_stride,
                             scale_factor=fps / self.feat_stride,
                             feat_len=len(feat),
                             labels=labels,
                             segments=segments,
                             gt_ignore_flags=ignore_flags)
            data_list.append(data_info)
        print_log(f"number of feature windows:\t {len(data_list)}", logger=MMLogger.get_current_instance())
        return data_list

    def parse_labels(self, video_name, video_info):
        # Segments information
        segments = []
        labels = []
        ignore_flags = []
        video_duration = video_info['duration']
        for ann in video_info['annotations']:
            segment, label = ann['segment'], ann['label']
            # Skip annotations that are out of range.
            if not (0 <= segment[0] < segment[1] <= video_duration) and self.skip_wrong:
                print_log(f"invalid segment annotation in {video_name}: {segment}, duration={video_duration}, skipped",
                          logger=MMLogger.get_current_instance())
                continue

            # Skip annotations that are too short. The threshold could be the stride of feature extraction.
            # For example, if the features were extracted every 8 frames,
            # then the threshold should be greater than 8/30 = 0.27s
            if isinstance(self.skip_short, (int, float)):
                if segment[1] - segment[0] < self.skip_short:
                    print_log(f"too short segment annotation in {video_name}: {segment}, skipped",
                              logger=MMLogger.get_current_instance())

            labels.append(self.metainfo['classes'].index(label))
            ignore_flags.append(False)
            segments.append(segment)

        return np.array(segments, np.float32), np.array(labels, np.int64), np.array(ignore_flags, dtype=bool)
