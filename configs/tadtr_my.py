_base_ = [
    'default_runtime.py', './thumos14.py'
]
custom_imports = dict(imports=['my_modules'], allow_failed_imports=False)
# Compared with TadTR:
# 1. Use multi-level features via temporal 1d convolution layers
# 2. Reduce the number of epoch from 16 to 12, lr from 2e-4 to 1e-4
# 3. Use the self-supervised features (VideoMAE2), window-size=256
# 4. Use GIoU loss and cost
# 5. Use memory fusion
enc_layers = 4
dec_layers = 4
num_levels = 4
feature_dim = 2432
ffn_dim = 2048
embed_dim = 256
dropout = 0.1

cls_loss_coef = 2
seg_loss_coef = 5
iou_loss_coef = 2

max_per_img = 200
lr = 0.0001

# model setting
model = dict(
    type='MyDeformableDETR',
    num_queries=200,
    with_box_refine=True,
    as_two_stage=False,
    num_feature_levels=num_levels,
    data_preprocessor=dict(type='DetDataPreprocessor'),
    backbone=dict(type='PseudoBackbone', multi_scale=False),  # No backbone since we use pre-extracted features.
    neck=[
        dict(
            type='DownSampler1D',
            num_levels=num_levels,
            in_channels=feature_dim,
            out_channels=feature_dim,
            out_indices=range(0, num_levels + 1),
            mask=False),
        dict(
            type='ChannelMapper',
            in_channels=[feature_dim] * num_levels,
            kernel_size=1,
            out_channels=embed_dim,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=num_levels)],
    encoder=dict(
        num_layers=enc_layers,
        layer_cfg=dict(
            self_attn_cfg=dict(
                num_levels=num_levels,   # Using multi-level features
                embed_dims=embed_dim,
                num_points=4,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=embed_dim,
                feedforward_channels=ffn_dim,
                ffn_drop=dropout)),
        memory_fuse=True),  # Using memory fusion
    decoder=dict(
        num_layers=dec_layers,  # 6 for DeformableDETR
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=embed_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=embed_dim,
                num_points=4,
                num_levels=num_levels,   # Using multi-level features
                dropout=dropout,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=embed_dim,
                feedforward_channels=ffn_dim,
                ffn_drop=dropout)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=embed_dim//2, normalize=True, offset=-0.5, temperature=10000),
    bbox_head=dict(
        type='CustomDINOHead',
        num_classes=20,
        embed_dims=embed_dim,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_coef),
        loss_bbox=dict(type='CustomL1Loss', loss_weight=seg_loss_coef),
        loss_iou=dict(type='CustomGIoULoss', loss_weight=iou_loss_coef)),
    dn_cfg=dict(label_noise_scale=0.5, box_noise_scale=1.0,
                group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
                # group_cfg=dict(dynamic=False, num_groups=5)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),  # from 6.0 to 2.0
                dict(type='CustomBBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='CustomIoUCost', iou_mode='giou', weight=2.0)])),  # from iou to giou
    test_cfg=dict(max_per_img=200)  # from 100 to 200 since window size is scaled
)

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
                                    'reference_points': dict(lr_mult=0.1)}))
# learning policy
max_epochs = 6  # 16 for TadTR
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[5],
        gamma=0.1)]
# max_epochs = 16
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.001,
#         by_epoch=True,
#         begin=0,
#         end=4,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         by_epoch=True,
#         T_max=12,
#         begin=4,
#         end=16,
#         eta_min_ratio=0.01,
#         convert_to_iter_based=True)
# ]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# dataset settings
window_size = 256
training_stride = 32
testing_stride = 64
train_pipeline = [
    dict(type='FeatDecode'),
    # dict(type='RandCropFeat', min_crop_ratio=0.6),
    dict(type='PadFeat', pad_len=window_size),
    dict(type='ReFormat'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'offset_sec'))]
test_pipeline = [
    dict(type='FeatDecode'),
    dict(type='PadFeat', pad_len=window_size),
    dict(type='ReFormat'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'offset_sec', 'overlap'))]
train_dataloader = dict(
    dataset=dict(feat_stride=4,
                 fix_slice=True,
                 on_the_fly=True,
                 window_size=window_size,
                 iof_thr=0.75,
                 window_stride=training_stride,
                 pipeline=train_pipeline,
                 data_prefix=dict(feat='features/thumos_feat_VideoMAE2-RGB_I3D-Flow_2432')))
val_dataloader = dict(
    dataset=dict(feat_stride=4,
                 fix_slice=True,
                 on_the_fly=True,
                 window_size=window_size,
                 window_stride=testing_stride,
                 pipeline=test_pipeline,
                 data_prefix=dict(feat='features/thumos_feat_VideoMAE2-RGB_I3D-Flow_2432')))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='TADmAPMetric',
    metric='mAP',
    iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
    nms_in_overlap=False,   # True for TadTR
    nms_cfg=dict(type='nms', iou_thr=0.6))  # 0.4 for TadTR
test_evaluator = val_evaluator
