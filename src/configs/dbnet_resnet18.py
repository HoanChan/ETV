# dbnet_pubtabnet_resnet18_fpnc.py

_base_ = [
    'mmocr::_base_/default_runtime.py'
]

# Dataset
dataset_type = 'PubTabNetDataset'
train_img_prefix = 'F:/data/vitabset/train'
train_ann_file = 'F:/data/vitabset/train.bz2'
val_img_prefix = 'F:/data/vitabset/val'
val_ann_file = 'F:/data/vitabset/val.bz2'
test_img_prefix = 'F:/data/vitabset/test'
test_ann_file = 'F:/data/vitabset/test.bz2'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Pad', size=(640, 640), pad_val=0),
    dict(type='PackTextDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=0),
    dict(type='PackTextDetInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        data_prefix={'img_path': train_img_prefix},
        pipeline=train_pipeline,
        test_mode=False,
        # Các tham số khác nếu cần
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        data_prefix={'img_path': val_img_prefix},
        pipeline=test_pipeline,
        test_mode=True,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator

# Model
model = dict(
    type='DBNet',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # Sử dụng checkpoint pretrain của MMOCR DBNet-ResNet18-FPNC
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015/dbnet_resnet18_fpnc_1200e_icdar2015_20221101_191201-5a1d8b0b.pth'
        )
    ),
    neck=dict(
        type='FPNC',
        in_channels=[64, 128, 256, 512],
        lateral_channels=256,
        out_channels=256,
        num_outs=4),
    det_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=1.0, beta=10.0, bbce_loss=True, reduction='mean')),
    postprocessor=dict(
        type='DBPostprocessor',
        text_repr_type='poly',
        mask_thr=0.3,
        min_text_score=0.3,
        min_text_width=5,
        unclip_ratio=1.5,
        arcLength_ratio=0.01,
        max_candidates=1000)
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3),
    clip_grad=None
)

# Learning policy
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-5, power=0.9, by_epoch=True, begin=0, end=1200)
]

# Train, val, test config
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1200, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10)
)

# Custom imports
custom_imports = dict(
    imports=[
        'datasets.table_dataset',
        # Thêm các module custom nếu cần
    ],
    allow_failed_imports=False
)