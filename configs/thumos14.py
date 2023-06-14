# dataset settings
dataset_type = 'Thumos14FeatDataset'
data_root = 'my_data/thumos14/'

train_pipeline = [
    dict(type='SlidingWindow', window_size=128, iof_thr=0.75),
    dict(type='ReFormat'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='SlidingWindow', window_size=128, iof_thr=0.75, attempts=1000),
    dict(type='ReFormat'),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/louis/thumos14_val.json',
        feat_stride=8,
        skip_short=True,
        data_prefix=dict(feat='features/thumos_feat_TadTR_64input_8stride_2048'),
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
        ann_file='annotations/louis/thumos14_test.json',
        feat_stride=8,
        data_prefix=dict(feat='features/thumos_feat_TadTR_64input_8stride_2048'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='TH14Metric',
    iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
    eval_mode='area',
    metric='mAP')
test_evaluator = val_evaluator
