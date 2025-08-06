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
    backbone=dict(
        type='ResNet',                              # Thay thế 'TableResNetExtra'
        in_channels=3,                              # = input_dim=3 trong TableResNetExtra.__init__()
        stem_channels=[64, 128],                    # = conv1(3→64) + conv2(64→128) trong TableResNetExtra
        block_cfgs=dict(
            type='BasicBlock',                      # = BasicBlock class trong TableResNetExtra
            plugins=dict(                           # = ContextBlock trong TableResNetExtra.BasicBlock
                cfg=dict(
                    type='GCAModule',               # = ContextBlock (Global Context Attention) 
                    ratio=0.0625,                   # = gcb_config['ratio'] = 0.0625
                    n_head=1,                       # = gcb_config['headers'] = 1
                    pooling_type='att',             # Attention pooling type
                    is_att_scale=False,             # = gcb_config['att_scale'] = False  
                    fusion_type='channel_add'),     # = gcb_config['fusion_type'] = 'channel_add'
                position='after_conv2')),           # Áp dụng sau conv2 trong BasicBlock
        arch_layers=[1, 2, 5, 3],                   # = layers=[1,2,5,3] - số blocks trong mỗi stage
        arch_channels=[256, 256, 512, 512],         # Channels: layer1=256, layer2=256, layer3=512, layer4=512
        strides=[1, 1, 1, 1],                       # Tất cả stride=1 giống TableResNetExtra
        plugins=[
            dict(# MAXPOOL PLUGIN 1 - Thay thế maxpool1 & maxpool2 trong TableResNetExtra
                cfg=dict(type='Maxpool2d', kernel_size=2, stride=(2, 2)),
                stages=(True, True, False, False),  # stage1: maxpool1, stage2: maxpool2
                position='before_stage'),           # Trước mỗi stage
            dict(# MAXPOOL PLUGIN 2 - Thay thế maxpool3 trong TableResNetExtra
                cfg=dict(type='Maxpool2d', kernel_size=3, stride=3),
                stages=(False, False, True, False), # Chỉ stage3: maxpool3 (kernel=3 khác với maxpool1,2)
                position='before_stage'),           # Trước stage3
            dict(# CONV PLUGIN - Thay thế conv3, conv4, conv5, conv6 trong TableResNetExtra
                cfg=dict(
                    type='ConvModule',              # ConvModule = Conv2d + BN + ReLU
                    kernel_size=3,                  # = kernel_size=3 trong conv3,4,5,6
                    stride=1,                       # = stride=1 trong conv3,4,5,6
                    padding=1,                      # = padding=1 trong conv3,4,5,6
                    norm_cfg=dict(type='BN'),       # = BatchNorm2d trong conv3,4,5,6
                    act_cfg=dict(type='ReLU')),     # = ReLU trong conv3,4,5,6
                stages=(True, True, True, True),    # Tất cả 4 stages: conv3, conv4, conv5, conv6
                position='after_stage')             # Sau mỗi stage
        ],
        init_cfg=[# WEIGHT INITIALIZATION - Tương ứng init_weights trong TableResNetExtra
            dict(type='Kaiming', layer='Conv2d'),           # = nn.init.kaiming_normal_ trong TableResNetExtra.init_weights
            dict(type='Constant', val=1, layer='BatchNorm2d'), # = nn.init.constant_(weight=1) trong TableResNetExtra.init_weights
        ]
    ),
    
    # ===== DETAILED BACKBONE MAPPING: TableResNetExtra → mmOCR ResNet =====
    # 
    # TableResNetExtra.__init__():                    mmOCR ResNet Config:
    # ├── input_dim=3                              → in_channels=3
    # ├── conv1: 3→64, 3x3, s=1, p=1              → stem_channels[0]: 3→64
    # ├── bn1 + relu1                              → auto-generated in stem
    # ├── conv2: 64→128, 3x3, s=1, p=1            → stem_channels[1]: 64→128  
    # ├── bn2 + relu2                              → auto-generated in stem
    # ├── maxpool1: 2x2, s=2                      → plugins[0] stages=(True,...)
    # ├── layer1: BasicBlock×1, 128→256            → arch_layers[0]=1, arch_channels[0]=256
    # ├── conv3: 256→256, 3x3, s=1, p=1           → plugins[2] stages=(True,...)
    # ├── bn3 + relu3                              → auto-generated in ConvModule
    # ├── maxpool2: 2x2, s=2                      → plugins[0] stages=(...,True,...)
    # ├── layer2: BasicBlock×2, 256→256            → arch_layers[1]=2, arch_channels[1]=256
    # ├── conv4: 256→256, 3x3, s=1, p=1           → plugins[2] stages=(...,True,...)
    # ├── bn4 + relu4                              → auto-generated in ConvModule
    # ├── maxpool3: 2x2, s=2                      → plugins[1] stages=(...,True,...) kernel=3
    # ├── layer3: BasicBlock×5, 256→512            → arch_layers[2]=5, arch_channels[2]=512
    # ├── conv5: 512→512, 3x3, s=1, p=1           → plugins[2] stages=(...,True,...)
    # ├── bn5 + relu5                              → auto-generated in ConvModule
    # ├── layer4: BasicBlock×3, 512→512            → arch_layers[3]=3, arch_channels[3]=512
    # ├── conv6: 512→512, 3x3, s=1, p=1           → plugins[2] stages=(...,True)
    # └── bn6 + relu6                              → auto-generated in ConvModule
    #
    # TableResNetExtra.BasicBlock:                 mmOCR ResNet Config:
    # ├── conv1, bn1, conv2, bn2                   → auto-generated in BasicBlock
    # └── ContextBlock (GCA attention)             → block_cfgs.plugins.cfg (GCAModule)
    #
    # TableResNetExtra.forward():                  mmOCR ResNet Behavior:
    # ├── Feature extraction flow                  → auto-handled by ResNet
    # ├── f.append(x) after conv3                  → multi-scale feature output
    # ├── f.append(x) after conv4                  → multi-scale feature output  
    # └── f.append(x) after conv6                  → multi-scale feature output
    # ================================================================
    encoder=dict(
        type='PositionalEncoding', # file:///./../models/encoders/positional_encoding.py
        d_model=512,
        dropout=0.2,
        max_len=5000
    ),
    decoder=dict(
        type='TableMasterDecoder', # file:///./../models/decoders/table_master_decoder.py
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
        'models.decoders.table_master_decoder',
        'models.dictionaries.table_master_dictionary',
        'models.encoders.positional_encoding',
        'models.losses.master_tf_loss',
        'models.losses.table_l1_loss',
        'models.metrics.teds_metric',
        'models.postprocessors.table_master_postprocessor',
        'models.recognizer.table_master',
        'structures.table_master_data_sample',
        'visualization.table_master_visualizer',
    ],
    allow_failed_imports=False,
)
# endregion