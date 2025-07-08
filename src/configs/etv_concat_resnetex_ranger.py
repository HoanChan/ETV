# Configuration for TableMaster ConcatLayer with TableResNetExtract and Ranger optimizer
# Compatible with mmOCR 1.0.1 and your local src modules

_base_ = [
    './_etv_base.py',
]

# Model settings
model = dict(
    type='TABLEMASTER',
    backbone=dict(
        type='TableResNetExtra',
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint=None),
        # You can set checkpoint to a pretrained model if available
    ),
    encoder=None,  # TableMaster does not use a separate encoder
    decoder=dict(
        type='TableMasterConcatDecoder',
        in_channels=512,
        num_classes=1200,  # Update according to your dictionary size
        max_seq_len=600,
        start_idx=0,
        padding_idx=1,
        mask=True,
        # Add other decoder params if needed
    ),
    dictionary=dict(
        type='TableMasterDictionary',
        dict_file='src/data/structure_vocab.txt',
        with_padding=True,
        with_unknown=True,
        same_start_end=True,
        with_start=True,
        with_end=True,
    ),
    loss=dict(
        type='MasterTFLoss',
        ignore_index=1,
        reduction='mean',
    ),
    postprocessor=dict(
        type='TableMasterPostprocessor',
        max_seq_len=600,
    ),
    label_convertor=None,
)

# Dataset settings
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='TableDataset',
        data_root='data/train/',
        ann_file='train_ann.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TableResize', size=(608, 608)),
            dict(type='PackInputs'),
        ],
        dictionary='src/data/structure_vocab.txt',
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type='TableDataset',
        data_root='data/val/',
        ann_file='val_ann.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TableResize', size=(608, 608)),
            dict(type='PackInputs'),
        ],
        dictionary='src/data/structure_vocab.txt',
    ),
)
test_dataloader = val_dataloader

# Evaluation metric
val_evaluator = dict(
    type='TEDSMetric',
    structure_only=False,
    ignore_case=True,
    ignore_symbol=True,
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Ranger',
        lr=0.001,
        weight_decay=0.0,
    ),
    clip_grad=None,
)

# Learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50),
)

# Runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Custom imports for local src modules
custom_imports = dict(
    imports=[
        'src.models.backbones.resnet_extra',
        'src.models.decoders.table_master_concat_decoder',
        'src.models.dictionaries.table_master_dictionary',
        'src.models.losses.master_tf_loss',
        'src.models.metric.teds_metric',
        'src.models.postprocessors.table_master_postprocessor',
        'src.datasets.table_dataset',
        'src.datasets.transforms.table_resize',
        'src.datasets.transforms.pack_inputs',
        'src.optimizer.ranger',
    ],
    allow_failed_imports=False,
)
