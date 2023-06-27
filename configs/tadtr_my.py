_base_ = [
    './tadtr.py'
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
# Use cosine annealing lr to replace the original step lr in TadTR
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
        T_max=16,
        begin=4,
        end=20,
        eta_min_ratio=0.01,
        convert_to_iter_based=True)
]