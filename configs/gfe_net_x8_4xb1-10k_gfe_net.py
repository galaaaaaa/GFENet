import copy
import os
_base_ = [
    "_base_/default_runtime.py",
    "_base_/datasets/groklst_dataset_x8-64-512.py",
    "_base_/schedules/schedule_20k.py",
]

scale = 8

# model settings
model = dict(
    type="BaseEditModel",
    generator=dict(
        type="GFENet",
        in_channels=10,
        num_feats=32,
        kernel_size=3,
        scale=scale,
        norm_flag=2, 
        hr_block='DEB',
        norm_dict=dict(mean=282.51, std=14.59, min=243.53, max=320.85),
        deploy=False,
    ),
    pixel_loss=dict(
        type="CombinedLoss",
        loss_weight=1.0,
        w_data=1.0,
        w_grad=0.1,
        w_freq=0.5,
        reduction="mean",
    ),
    train_cfg=dict(),
    test_cfg=dict(
        metrics=[
            dict(type="RMSE", scaling=1.0, prefix="lst"),
            dict(type="LSTMAE", scaling=1.0, prefix="lst"),  # abs(pred-gt)
            dict(type="BIAS", scaling=1.0, prefix="lst"),  # pred-gt
            dict(type="CC", scaling=1.0, prefix="lst"),
            dict(type="RSD", scaling=1.0, prefix="lst"),
            dict(type="LSTPSNR", scaling=1.0, prefix="lst"),
            dict(type="LSTSSIM", scaling=1.0, prefix="lst"),
            dict(type="SAM", scaling=1.0, prefix="lst"),
            dict(type="ERGAS", scaling=1.0, prefix="lst"),
            dict(type="SCC", scaling=1.0, prefix="lst"),
            dict(type="Q", scaling=1.0, prefix="lst"),
            dict(type="DLambda", scaling=1.0, prefix="lst"),
            dict(type="DS", scaling=1.0, prefix="lst"),
            dict(type="QNR", scaling=1.0, prefix="lst"),
        ]
    ),
    data_preprocessor=dict(
        type="LSTDataPreprocessor",
        mean=None,
        std=None,
    ),
)

# Override test_evaluator from base config
test_evaluator = [
    dict(
        type="Evaluator",
        metrics=[
            dict(type="RMSE", scaling=1.0, prefix="lst"),
            dict(type="LSTMAE", scaling=1.0, prefix="lst"),  # abs(pred-gt)
            dict(type="BIAS", scaling=1.0, prefix="lst"),  # pred-gt
            dict(type="CC", scaling=1.0, prefix="lst"),
            dict(type="RSD", scaling=1.0, prefix="lst"),
            dict(type="LSTPSNR", scaling=1.0, prefix="lst"),
            dict(type="LSTSSIM", scaling=1.0, prefix="lst"),
            dict(type="SAM", scaling=1.0, prefix="lst"),
            dict(type="ERGAS", scaling=1.0, prefix="lst"),
            dict(type="SCC", scaling=1.0, prefix="lst"),
            dict(type="Q", scaling=1.0, prefix="lst"),
            dict(type="DLambda", scaling=1.0, prefix="lst"),
            dict(type="DS", scaling=1.0, prefix="lst"),
            dict(type="QNR", scaling=1.0, prefix="lst"),
        ]
    )
]
