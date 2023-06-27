_base_ = [
    '.tadtr.py'
]

# Use cosine annealing lr to replace the original step lr in TadTR
param_scheduler = [
    dict(
        type='LinearLR',
        by_epoch=True,
        start_factor=0.001,
        end=4,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=16,
        eta_min_ratio=0.01,
        convert_to_iter_based=True)
]