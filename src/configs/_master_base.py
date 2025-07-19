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
    vocab_path = "d:/BIG Projects/Python/ETV/src/data/textline_recognition_alphabet.txt"
# elif ENV == 'linux':
#     prefix = "/mnt/data/vitabset"
#     vocab_path = "/home/user/ETV/src/data/textline_recognition_alphabet.txt"
else:  # colab
    prefix = "/content/vitabset"
    vocab_path = "/content/ETV/src/data/textline_recognition_alphabet.txt"

VITABSET_TRAIN_IMAGE_ROOT = f"{prefix}/train"
VITABSET_TRAIN_JSON = f"{prefix}/train.bz2"
VITABSET_VAL_IMAGE_ROOT = f"{prefix}/val"
VITABSET_VAL_JSON = f"{prefix}/val.bz2"
VITABSET_TEST_IMAGE_ROOT = f"{prefix}/test"
VITABSET_TEST_JSON = f"{prefix}/test.bz2"
TEXT_VOCAB_FILE = vocab_path

# region Dictionary
start_end_same = False
alphabet_len = len(open(TEXT_VOCAB_FILE, 'r').readlines())
if start_end_same:
    PAD = alphabet_len + 2
else:
    PAD = alphabet_len + 3

dictionary = dict(
    type='Dictionary', # mmocr built-in dictionary for text recognition
    dict_file=TEXT_VOCAB_FILE,
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
        type='LoadOCRAnnotations', # mmocr built-in for text recognition
        with_text=True,
    ),
    dict(
        type="KeyMapper", # https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/wrappers.py#L103
        mapping=dict(
            img='img',
            img_shape='img_shape',
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
    dict(
        type='PackTextRecogInputs', # mmocr built-in for text recognition
        keys=['img'],
        meta_keys=('ori_shape', 'img_shape', 'scale_factor', 'pad_shape', 'valid_ratio')
    )
]

train_dataset = dict(
    type='OCRDataset', # mmocr built-in text recognition dataset
    ann_file=VITABSET_TRAIN_JSON,
    data_prefix={'img_path': VITABSET_TRAIN_IMAGE_ROOT},
    max_data=-1,  # -1 để load toàn bộ, >0 để giới hạn số lượng sample
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
    type='MasterTextRecognizer', # file:///./../models/recognizer/master_recognizer.py
    backbone=dict(
        type='ResNetExtra', # file:///./../models/backbones/resnet_extra.py
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        ),
        layers=[1,2,5,3]
    ),
    encoder=dict(
        type='PositionalEncoding', # file:///./../models/encoders/positional_encoding.py
        d_model=512,
        dropout=0.2,
        max_len=5000
    ),
    decoder=dict(
        type='MasterDecoder', # file:///./../models/decoders/master_decoder.py
        n_layers=6,
        n_head=8,
        d_model=512,
        max_seq_len=100,
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
            type='AttentionPostprocessor', # mmocr built-in postprocessor
            dictionary=dictionary,
            max_seq_len=100,
        ),
        text_loss=dict(
            type='CELoss', # mmocr built-in cross-entropy loss
            ignore_index=PAD,
            reduction='mean',
        ),
    ),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]))
# endregion

# region Imports
custom_imports = dict(
    imports=[
        'models.backbones.resnet_extra',
        'models.decoders.master_decoder',
        'models.encoders.positional_encoding',
        'models.recognizer.master_recognizer',
        'optimizer.ranger',
        'structures.table_master_data_sample',
    ],
    allow_failed_imports=False,
)
# endregion
