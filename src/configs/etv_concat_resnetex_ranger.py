# Configuration for TableMaster ConcatLayer with TableResNetExtract and Ranger optimizer
# Compatible with mmOCR 1.0.1 and local src modules

_base_ = [ # https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#inherit-configuration-files-across-repository
    'mmocr::textrecog/_base_/default_runtime.py',               # https://github.com/open-mmlab/mmocr/blob/v1.0.1/configs/textrecog/_base_/default_runtime.py
    'mmocr::textrecog/_base_/schedules/schedule_adam_base.py',  # https://github.com/open-mmlab/mmocr/blob/v1.0.1/configs/textrecog/_base_/schedules/schedule_adam_base.py                         
    '_etv_base.py' # file:///./_etv_base.py
]

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py
    optimizer=dict(
        type='Ranger', # file:///./../optimizer/ranger.py
        lr=0.001,
        weight_decay=0.0,
    ),
)

# Learning policy
param_scheduler = [
    dict(
        type='LinearLR', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L121
        start_factor=0.1, 
        by_epoch=True, 
        begin=0, 
        end=5
    ),
    dict(
        type='CosineAnnealingLR', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L48
        T_max=95, 
        by_epoch=True, 
        begin=5, 
        end=100
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py
        interval=1, 
        max_keep_ckpts=5
    ),
    logger=dict(
        type='LoggerHook', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py
        interval=50
    ),
)

# Runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1) # https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L21
val_cfg = dict(type='ValLoop') # https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L330
test_cfg = dict(type='TestLoop') # https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L417
# region Evaluator
val_evaluator = dict(
    type='MultiDatasetsEvaluator', # https://github.com/open-mmlab/mmocr/blob/main/mmocr/evaluation/evaluator/multi_datasets_evaluator.py
    metrics=[
        dict(
            type='TEDSMetric', # file:///./../models/metrics/teds_metric.py
            structure_only=True,
            ignore_nodes=None,
            collect_device='cpu',
            prefix=None
        )
    ],
    dataset_prefixes=None)
test_evaluator = val_evaluator
# endregion
# Custom imports for local src modules
custom_imports = dict(
    imports=[
        'models.backbones.resnet_extra',
        'models.decoders.table_master_concat_decoder',
        'models.dictionaries.table_master_dictionary',
        'models.losses.master_tf_loss',
        'models.metrics.teds_metric',
        'models.postprocessors.table_master_postprocessor',
        'datasets.table_dataset',
        'datasets.transforms.table_resize',
        'datasets.transforms.table_pad',
        'datasets.transforms.pack_inputs',
        'optimizer.ranger',
    ],
    allow_failed_imports=False,
)
