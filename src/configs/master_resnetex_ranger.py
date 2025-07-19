# Configuration for Master text recognition with ResNetExtra and Ranger optimizer
# Compatible with mmOCR 1.0.1 and local src modules

_base_ = [ # https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#inherit-configuration-files-across-repository
    'mmocr::textrecog/_base_/default_runtime.py',               # https://github.com/open-mmlab/mmocr/blob/v1.0.1/configs/textrecog/_base_/default_runtime.py
    'mmocr::textrecog/_base_/schedules/schedule_adam_base.py',  # https://github.com/open-mmlab/mmocr/blob/v1.0.1/configs/textrecog/_base_/schedules/schedule_adam_base.py                         
    '_master_base.py' # file:///./_master_base.py
]

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper', # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/amp_optimizer_wrapper.py#L24
    optimizer=dict(
        type='Ranger', # file:///./../optimizer/ranger.py
        lr=0.001,
        weight_decay=0.0,
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
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

# Evaluator
val_evaluator = dict(
    type='MultiDatasetsEvaluator', # https://github.com/open-mmlab/mmocr/blob/main/mmocr/evaluation/evaluator/multi_datasets_evaluator.py
    metrics=[
        dict(
            type='WordMetric', # mmocr built-in word accuracy metric
            mode=['exact', 'ignore_case', 'ignore_case_symbol'],
            collect_device='cpu',
            prefix='Acc'
        )
    ],
    dataset_prefixes=None)
test_evaluator = val_evaluator

# Checkpoint settings
load_from = None  # No pre-trained model
resume = False  # Do not resume training from a checkpoint
