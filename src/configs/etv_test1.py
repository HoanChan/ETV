_base_ = [
    'mmocr::textrecog/_base_/default_runtime.py',               # mmocr:: tương đương với link tới folder /configs/ của mmocr
    'mmocr::textrecog/_base_/schedules/schedule_adam_base.py',  # Xem thêm tại https://github.com/open-mmlab/mmocr/tree/v1.0.1/configs
    'datasets/vitabset_data.py',                                # https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#inherit-configuration-files-across-repository
    '_etv_base.py'
]

optim_wrapper = dict(optimizer=dict(lr=4e-4))
train_cfg = dict(max_epochs=12)

# learning policy
param_scheduler = [
    dict(type='LinearLR', end=100, by_epoch=False),
    dict(type='MultiStepLR', milestones=[11], end=12),
]

train_dataset = _base_.vitabset_rec_test
val_dataset = _base_.vitabset_rec_test
test_dataset = _base_.vitabset_rec_test

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_evaluator = dict(dataset_prefixes=['Toy'])
test_evaluator = dict(dataset_prefixes=['Toy'])
