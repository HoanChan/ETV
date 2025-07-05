from config import *

test_dataset = {
    'ann_file': VITABSET_TEST_JSON,
    'data_prefix': {'img_path': VITABSET_TEST_IMAGE_ROOT},
    'task_type': 'both',
    'split_filter': None,  # Load all splits available in the file
    'max_structure_len': 500,
    'max_cell_len': 500,
    'ignore_empty_cells': True
}
TRAIN_STATE = False  # Set to False for test mode
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    #dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[ 'filename', 'ori_shape', 'img_shape', 'scale_factor', 'img_norm_cfg', 'ori_filename', 'pad_shape' ]),
]

test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)