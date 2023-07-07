_base_ = [
    './tadtr.py'
]

# 1. Use cosine annealing lr to replace the original step lr in TadTR
optim_wrapper = dict(optimizer=dict(lr=5e-4))
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=16, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=4,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=12,
        begin=4,
        end=16,
        eta_min_ratio=0.01,
        convert_to_iter_based=True)
]

# 2. Use the self-supervised features (VideoMAE2)
train_pipeline = [
    dict(type='SlidingWindow', window_size=256, just_loading=True),
    dict(type='ReFormat'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'offset'))]
test_pipeline = [
    dict(type='SlidingWindow', window_size=256, just_loading=True),
    dict(type='ReFormat'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'offset', 'overlap'))]
train_dataloader = dict(
    dataset=dict(feat_stride=4,
                 fix_slice=True,
                 on_the_fly=True,
                 window_size=256,
                 iof_thr=0.75,
                 window_stride=64,  # overlap=0.75
                 pipeline=train_pipeline,
                 data_prefix=dict(feat='features/thumos_feat_VideoMAE2-RGB_I3D-Flow_2432')))
val_dataloader = dict(
    dataset=dict(feat_stride=4,
                 fix_slice=True,
                 on_the_fly=True,
                 window_size=256,
                 window_stride=192,  # overlap=0.25
                 pipeline=test_pipeline,
                 data_prefix=dict(feat='features/thumos_feat_VideoMAE2-RGB_I3D-Flow_2432')))

# 3. Use multi-level features via temporal 1d convolution layers
# model setting
model = dict(
    num_feature_levels=4,
    as_two_stage=False,
    backbone=dict(type='PseudoBackbone', multi_scale=False),  # No backbone since we use pre-extracted features.
    neck=[
        dict(
            type='DownSampler1D',
            num_levels=4,
            in_channels=2432,
            out_channels=2432,
            out_indices=(0, 1, 2, 3),
            mask=False),
        dict(
            type='ChannelMapper',
            in_channels=[2432, 2432, 2432, 2432],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4)],
    positional_encoding=dict(offset=-0.5),
    encoder=dict(num_layers=4, layer_cfg=dict(self_attn_cfg=dict(num_levels=4))),
    decoder=dict(num_layers=4, layer_cfg=dict(cross_attn_cfg=dict(num_levels=4))),
    bbox_head=dict(loss_iou=dict(_delete_=True, type='CustomGIoULoss', loss_weight=2.0)),  # from iou to giou
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),  # from 6.0 to 2.0
                dict(type='CustomBBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='CustomIoUCost', iou_mode='giou', weight=2.0)])),  # iou to giou
    test_cfg=dict(max_per_img=200)
)
# val_evaluator = dict(
#     type='TH14Metric',
#     metric='mAP',
#     iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
#     nms_cfg=dict(type='nms', iou_thr=0.2))
