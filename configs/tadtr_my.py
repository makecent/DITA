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

# 2. Use stride-4 features (used by the ActionFormer)
train_dataloader = dict(
    dataset=dict(feat_stride=4,
                 window_size=256,
                 window_stride=64,  # overlap=0.75
                 data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features')))
val_dataloader = dict(
    dataset=dict(feat_stride=4,
                 window_size=256,
                 window_stride=192,  # overlap=0.25
                 data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features')))

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
            in_channels=2048,
            out_channels=2048,
            out_indices=(0, 1, 2, 3),
            mask=False),
        dict(
            type='ChannelMapper',
            in_channels=[2048, 2048, 2048, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4)],
    positional_encoding=dict(offset=-0.5),
    encoder=dict(num_layers=4, layer_cfg=dict(self_attn_cfg=dict(num_levels=4))),
    decoder=dict(num_layers=4, layer_cfg=dict(cross_attn_cfg=dict(num_levels=4)))
)
