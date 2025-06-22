_base_ = [
    'mmocr::textrecog/_base_/default_runtime.py',
    'mmocr::textrecog/_base_/schedules/schedule_adam_base.py',
    'datasets/vitabset_data.py',
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
    dataset=test_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(dataset_prefixes=['Toy'])
test_evaluator = val_evaluator
