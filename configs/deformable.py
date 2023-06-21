_base_ = [
    './thumos14.py'
]
custom_imports = dict(imports=['my_modules'], allow_failed_imports=False)

# TadTR (based on DeFormableDETR) setting: (DINO, TadTR)
enc_layers = 4  # 6, 4
dec_layers = 4  # 6, 4
dim_feedforward = 1024  # 2048, 1024
dropout = 0.1  # 0.0, 0.1
temperature = 10000  # 20, 10000

act_loss_coef = 4  # NA, 4
cls_loss_coef = 2  # 1.0, 2.0
seg_loss_coef = 5  # 5.0, 5.0
iou_loss_coef = 2  # 2.0, 2.0

max_per_img = 100  # 300, 100
lr = 0.0002  # 1e-4, 2e-4

# model setting
model = dict(
    type='DeformableDETR',
    num_queries=40,  # num_matching_queries, should be smaller than the window size
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=1,
    data_preprocessor=dict(type='DetDataPreprocessor'),
    backbone=dict(type='PseudoBackbone'),  # No backbone since we use pre-extracted features.
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1),
    encoder=dict(
        num_layers=enc_layers,  # 6 for DeformableDETR
        layer_cfg=dict(
            self_attn_cfg=dict(
                num_levels=1,
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=dim_feedforward,
                ffn_drop=dropout))),
    decoder=dict(
        num_layers=dec_layers,  # 6 for DeformableDETR
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=dropout,
                batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=dim_feedforward,
                ffn_drop=dropout)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5, temperature=temperature),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=20,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_coef),
        loss_bbox=dict(type='CustomL1Loss', loss_weight=seg_loss_coef),  # customized to ignore y1, y2
        loss_iou=dict(type='CustomGIoULoss', loss_weight=iou_loss_coef)),  # customized to ignore y1, y2
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='CustomBBoxL1Cost', weight=5.0, box_format='xywh'),  # customized to ignore y1, y2
                dict(type='CustomIoUCost', iou_mode='giou', weight=2.0)  # customized to ignore y1, y2
            ])),
    test_cfg=dict(max_per_img=max_per_img))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                    'sampling_offsets': dict(lr_mult=0.1),
                                    'reference_points': dict(lr_mult=0.1)
                                    }))

# learning policy
# TadTR uses 30 epochs, but since we use random sliding windows rather than fixed overlapping windows,
# we should increase the number of epochs to maximize utilization of the video content.
max_epochs = 16  # 16 for TadTR
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)  # 1 for TadTR

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[14],  # 14 for TadTR
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
