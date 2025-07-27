try:
    sysmod = __import__('sys')
    osmod = __import__('os')
    if 'google.colab' in sysmod.modules:
        ENV = 'colab'
    elif osmod.name == 'nt':
        ENV = 'windows'
    else:
        ENV = 'linux'
except:
    ENV = 'unknown'

if ENV == 'windows':
    prefix = "F:/data/vitabset"
    vocab_path = "d:/BIG Projects/Python/ETV/src/data/structure_vocab.txt"
# elif ENV == 'linux':
#     prefix = "/mnt/data/vitabset"
#     vocab_path = "/home/user/ETV/src/data/structure_vocab.txt"
else:  # colab
    prefix = "/content/vitabset"
    vocab_path = "/content/ETV/src/data/structure_vocab.txt"

VITABSET_TRAIN_IMAGE_ROOT = f"{prefix}/train"
VITABSET_TRAIN_JSON = f"{prefix}/train.bz2"
VITABSET_VAL_IMAGE_ROOT = f"{prefix}/val"
VITABSET_VAL_JSON = f"{prefix}/val.bz2"
VITABSET_TEST_IMAGE_ROOT = f"{prefix}/test"
VITABSET_TEST_JSON = f"{prefix}/test.bz2"
STRUCTURE_VOCAB_FILE = vocab_path
# region Dictionary
start_end_same = False
alphabet_len = len(open(STRUCTURE_VOCAB_FILE, 'r').readlines())
if start_end_same:
    PAD = alphabet_len + 2
else:
    PAD = alphabet_len + 3

dictionary = dict(
    type='TableMasterDictionary', # file:///./../models/dictionaries/table_master_dictionary.py
    dict_file=STRUCTURE_VOCAB_FILE,
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)
# endregion
# region Dataset
data_pipeline = [
    dict(type='LoadImageFromFile'), # https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/transforms/loading.py#L17
    dict(
        type='LoadTokens', # file:///./../datasets/transforms/load_tokens.py
        with_structure=True,
        with_cell=False,
        max_structure_token_len=600,
        max_cell_token_len=600
    ),
    dict(
        type="KeyMapper", # https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/wrappers.py#L103
        mapping=dict(
            img='img',
            img_shape='img_shape',
            gt_bboxes='bboxes',
        ),
        auto_remap=True, # restore original keys when done
        transforms=[
            dict(
                type='Resize', # https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/transforms/ocr_transforms.py#L501
                scale=(480, 480),
                keep_ratio=True,
            ),
        ]
    ),
    dict(
        type='Pad', # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/transforms/transforms.py#L705
        size=(480, 480),
    ),
    dict(type='BboxEncode'), # file:///./../datasets/transforms/bbox_encode.py
    dict(
        type='PadData', # file:///./../datasets/transforms/pad_data.py
        dictionary=dictionary,
        max_seq_len=600,
        max_bbox_len=600,
        pad_with='auto'
    ),
    dict(
        type='PackInputs', # file:///./../datasets/transforms/pack_inputs.py
        keys=['img'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        meta_keys=('ori_shape', 'img_shape', 'scale_factor', 'pad_shape', 'valid_ratio')
    )
]

train_dataset = dict(
    type='PubTabNetDataset', # file:///./../datasets/table_dataset.py
    ann_file=VITABSET_TRAIN_JSON,
    data_prefix={'img_path': VITABSET_TRAIN_IMAGE_ROOT},
    split_filter=None,  # Load all splits available in the file
    max_structure_len=600,
    # max_cell_len=600,
    ignore_empty_cells=True,
    max_data=-1,  # -1 để load toàn bộ, >0 để giới hạn số lượng sample
    random_sample=False, 
    pipeline=data_pipeline,
    test_mode=False,
)
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=train_dataset
)

test_dataset = train_dataset.copy()
test_dataset.update(
    ann_file=VITABSET_TEST_JSON,
    data_prefix={'img_path': VITABSET_TEST_IMAGE_ROOT},
    test_mode=True,
)

test_dataloader = train_dataloader.copy()
test_dataloader.update(dataset=test_dataset)

val_dataset = train_dataset.copy()
val_dataset.update(
    ann_file=VITABSET_VAL_JSON,
    data_prefix={'img_path': VITABSET_VAL_IMAGE_ROOT},
    test_mode=False,
)

val_dataloader = train_dataloader.copy()
val_dataloader.update(dataset=val_dataset)

# endregion
# region Model
model = dict(
    type='TableMaster', # file:///./../models/recognizer/table_master.py
    backbone = dict(
        type='ResNet',
        in_channels=3,  # input_dim=3
        stem_channels=[64, 128],  # từ conv1 (64) và conv2 (128)
        block_cfgs=dict(
            type='BasicBlock'
        ),
        arch_layers=[1, 2, 5, 3],  # layers=[1,2,5,3] từ TableResNetExtra
        arch_channels=[256, 256, 512, 512],  # channels cho từng layer
        strides=[1, 1, 1, 1],  # tất cả stride=1 trong _make_layer
        plugins=[
            # MaxPool2d sau conv2 (maxpool1)
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                stages=(True, False, False, False),  # chỉ ở stage đầu tiên
                position='before_stage'
            ),
            # MaxPool2d sau conv3 (maxpool2) 
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                stages=(False, True, False, False),  # chỉ ở stage thứ hai
                position='before_stage'
            ),
            # MaxPool2d với kernel (2,1) sau conv4 (maxpool3)
            dict(
                cfg=dict(type='Maxpool2d', kernel_size=(2, 1), stride=(2, 1)),
                stages=(False, False, True, False),  # chỉ ở stage thứ ba
                position='before_stage'
            ),
            # ConvModule sau mỗi layer (conv3, conv4, conv5, conv6)
            dict(
                cfg=dict(
                    type='ConvModule',
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                stages=(True, True, True, True),  # áp dụng cho tất cả stages
                position='after_stage'
            ),
            # GCAModule được áp dụng dựa trên layers=[False, True, True, True]
            dict(
                cfg=dict(
                    type='GCAModule',
                    ratio=0.0625,      # ratio=0.0625 từ gcb_config
                    n_head=1,          # headers=1 từ gcb_config  
                    pooling_type='att', # mặc định pooling_type='att'
                    is_att_scale=False, # att_scale=False từ gcb_config
                    fusion_type='channel_add'  # fusion_type="channel_add" từ gcb_config
                ),
                stages=(False, True, True, True),  # layers=[False, True, True, True] từ gcb_config
                position='after_stage'
            )
        ],
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d'),  # Kaiming initialization cho Conv2d
            dict(type='Constant', val=1, layer='BatchNorm2d'),  # BatchNorm weight = 1
        ]
    ),
    encoder=dict(
        type='PositionalEncoding', # file:///./../models/encoders/positional_encoding.py
        d_model=512,
        dropout=0.2,
        max_len=5000
    ),
    decoder=dict(
        type='TableMasterConcatDecoder', # file:///./../models/decoders/table_master_concat_decoder.py
        n_layers=3,
        n_head=8,
        d_model=512,
        max_seq_len=600,
        dictionary=dictionary,
        decoder=dict(
            self_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.0
            ),
            src_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.0
            ),
            feed_forward=dict(
                d_model=512,
                d_ff=2024,
                dropout=0.0
            ),
            size=512,
            dropout=0.0
        ),
        postprocessor=dict(
            type='TableMasterPostprocessor', # file:///./../models/postprocessors/table_master_postprocessor.py
            dictionary=dictionary,
            max_seq_len=600,
            start_end_same=False
        ),
        tokens_loss=dict(
            type='MasterTFLoss', # file:///./../models/losses/master_tf_loss.py
            ignore_index=PAD,
            reduction='mean',
            flatten=True
        ),
        bboxes_loss=dict(
            type='TableL1Loss', # file:///./../models/losses/table_l1_loss.py
            reduction='sum',
            lambda_horizon=1.0,
            lambda_vertical=1.0,
            eps=1e-9,
        ),
    ),
    data_preprocessor=dict(
        type='mmocr.TextRecogDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]))
# endregion

# region Imports
custom_imports = dict(
    imports=[
        'datasets.table_dataset',
        'datasets.transforms.load_tokens',
        'datasets.transforms.pack_inputs',
        'datasets.transforms.bbox_encode',
        'datasets.transforms.pad_data',
        'models.backbones.resnet_extra',
        'models.backbones.table_resnet_extra',
        'models.decoders.table_master_concat_decoder',
        'models.dictionaries.table_master_dictionary',
        'models.encoders.positional_encoding',
        'models.losses.master_tf_loss',
        'models.losses.table_l1_loss',
        'models.metrics.teds_metric',
        'models.postprocessors.table_master_postprocessor',
        'models.recognizer.table_master',
        'optimizer.ranger',
        'structures.table_master_data_sample',
        'visualization.table_master_visualizer',
    ],
    allow_failed_imports=False,
)
# endregion