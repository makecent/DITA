_base_ = [
    './tadtr_my.py'
]
train_dataloader = dict(
    dataset=dict(feat_stride=4,
                 window_size=256,
                 window_stride=64,  # overlap=0.75
                 data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features')))
val_dataloader = dict(
    dataset=dict(feat_stride=4,
                 window_size=256,
                 window_stride=64,  # overlap=0.75
                 data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features')))
