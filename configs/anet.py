# dataset settings
dataset_type = 'ANet13FeatDataset'
data_root = 'my_data/anet/'

train_pipeline = [
    dict(type='LoadFeat'),
    dict(type='RescaleFeat', window_size=256, training=True),
    dict(type='ReFormat'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride'))
]
test_pipeline = [
    dict(type='LoadFeat'),
    dict(type='RescaleFeat', window_size=256, training=False),
    dict(type='ReFormat'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/actionformer/anet1.3_i3d_filtered.json',
        feat_stride=16,
        split='training',
        skip_short=0.6,   # skip action annotations with duration less than 0.6 seconds
        data_prefix=dict(feat='features/ant_feat_ActionFormer-I3D_1024'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/actionformer/anet1.3_i3d_filtered.json',
        feat_stride=16,
        split='validation',
        skip_short=False,
        data_prefix=dict(feat='features/ant_feat_ActionFormer-I3D_1024'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='ANetMetric',
    metric='mAP',
    iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    nms_cfg=dict(type='nms', iou_thr=0.6))
test_evaluator = val_evaluator
