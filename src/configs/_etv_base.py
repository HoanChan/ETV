from config import *
# region Dataset
data_pipeline = [
    dict(type='LoadImageFromFile'), # 
    dict(
        type='TableResize', # https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/transforms/loading.py#L17
        keep_ratio=True,
        long_size=480
    ),
    dict(
        type='TablePad', # file:///./../datasets/transforms/table_pad.py
        size=(480, 480),
        pad_val=0,
        # return_mask=True,
        # mask_ratio=(8, 8)
    ),
    dict(
        type='PackInputs', # file:///./../datasets/transforms/pack_inputs.py
        keys=['img'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'scale_factor', 'img_norm_cfg', 'ori_filename', 'pad_shape')
    )
]

test_dataset = dict(
    type='PubTabNetDataset', # file:///./../datasets/table_dataset.py
    ann_file=VITABSET_TEST_JSON,
    data_prefix={'img_path': VITABSET_TEST_IMAGE_ROOT},
    task_type='both',
    split_filter=None,  # Load all splits available in the file
    max_structure_len=500,
    # max_cell_len=500,
    ignore_empty_cells=True,
    max_data=-1,  # -1 để load toàn bộ, >0 để giới hạn số lượng sample
    random_sample=False, 
    pipeline=data_pipeline
)

test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset
)
# endregion
# region Model
dictionary = dict(
    type='TableMasterDictionary', # file:///./../models/dictionaries/table_master_dictionary.py
    dict_file='{{ fileDirname }}/../data/structure_vocab.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='MASTER',
    backbone=dict(
        type='ResNet',
        in_channels=3,
        stem_channels=[64, 128],
        block_cfgs=dict(
            type='BasicBlock',
            plugins=dict(
                cfg=dict(
                    type='GCAModule',
                    ratio=0.0625,
                    n_head=1,
                    pooling_type='att',
                    is_att_scale=False,
                    fusion_type='channel_add'),
                position='after_conv2')),
        arch_layers=[1, 2, 5, 3],
        arch_channels=[256, 256, 512, 512],
        strides=[1, 1, 1, 1],
        plugins=[
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                stages=(True, True, False, False),
                position='before_stage'),
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=(2, 1), stride=(2, 1)),
                stages=(False, False, True, False),
                position='before_stage'),
            dict(
                cfg=dict(
                    type='ConvModule',
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')),
                stages=(True, True, True, True),
                position='after_stage')
        ],
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Constant', val=1, layer='BatchNorm2d'),
        ]),
    encoder=None,
    decoder=dict(
        type='MasterDecoder',
        d_model=512,
        n_head=8,
        attn_drop=0.,
        ffn_drop=0.,
        d_inner=2048,
        n_layers=3,
        feat_pe_drop=0.2,
        feat_size=6 * 40,
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', reduction='mean', ignore_first_char=True),
        max_seq_len=30,
        dictionary=dictionary),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]))
# endregion

# region Evaluator
val_evaluator = dict(
    type='TEDSMetric',
    structure_only=False,
    n_jobs=1,
    ignore_nodes=None,
    collect_device='cpu',
    prefix=None
)
# endregion