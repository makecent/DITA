_base_ = [
    './tadtr.py'
]
train_dataloader = dict(
    dataset=dict(feat_stride=4,
                 data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features')))
val_dataloader = dict(
    dataset=dict(feat_stride=4,
                 data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features')))
