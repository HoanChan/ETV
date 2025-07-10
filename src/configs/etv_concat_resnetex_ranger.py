# Configuration for TableMaster ConcatLayer with TableResNetExtract and Ranger optimizer
# Compatible with mmOCR 1.0.1 and local src modules

_base_ = [ # https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#inherit-configuration-files-across-repository
    'mmocr::textrecog/_base_/default_runtime.py',               # https://github.com/open-mmlab/mmocr/blob/v1.0.1/configs/textrecog/_base_/default_runtime.py
    'mmocr::textrecog/_base_/schedules/schedule_adam_base.py',  # https://github.com/open-mmlab/mmocr/blob/v1.0.1/configs/textrecog/_base_/schedules/schedule_adam_base.py                         
    '_etv_base.py' # file:///./_etv_base.py
]

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/amp_optimizer_wrapper.py#L24
    optimizer=dict(
        type='Ranger', # file:///./../optimizer/ranger.py
        lr=0.001,
        weight_decay=0.0,
    ),
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
param_scheduler = [
    # Linear warm-up
    dict(
        type='LinearLR', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L121
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=50),
    # Step decay
    dict(
        type='MultiStepLR', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L150
        by_epoch=True,
        milestones=[12, 15],
        gamma=0.1)
]
# Runtime settings
train_cfg = dict(max_epochs=17) # https://github.com/open-mmlab/mmocr/blob/v1.0.1/configs/textrecog/_base_/schedules/schedule_adam_base.py
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

load_from = None  # No pre-trained model
resume = False  # Do not resume training from a checkpoint

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
