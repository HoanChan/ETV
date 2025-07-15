# region Imports
custom_imports = dict(
    imports=[
        'models.recognizer.TableMaster',
        'models.backbones.TableResNetExtra',
        'models.encoders.PositionalEncoding',
        'models.decoders.TableMasterDecoder',
        'models.losses.MasterTFLoss',
        'models.losses.TableL1Loss',
        'models.optimizer.Ranger',
        'datasets.pipelines.TableResize',
        'datasets.pipelines.TablePad',
        'datasets.pipelines.TableBboxEncode',
        'datasets.utils.TableLoader',
        'datasets.utils.TableParser',
    ],
    allow_failed_imports=False
)
# endregion
# region Convertor
alphabet_file = 'src/data/structure_vocab.txt' # file://./../data/structure_vocab.txt
alphabet_len = len(open(alphabet_file, 'r').readlines())
max_seq_len = 500

start_end_same = False
label_convertor = dict(
            type='TableMasterConvertor', # file://./../models/convertors/table_master.py
            dict_file=alphabet_file,
            max_seq_len=max_seq_len,
            start_end_same=start_end_same,
            with_unknown=True)

if start_end_same:
    PAD = alphabet_len + 2
else:
    PAD = alphabet_len + 3
# endregion
# region Model
model = dict(
    type='TableMaster', # file://./../models/recognizer/table_master.py
    backbone=dict(
        type='TableResNetExtra', # file://./../models/backbones/table_resnet_extra.py
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        ),
        layers=[1,2,5,3]),
    encoder=dict(
        type='PositionalEncoding', # file://./../models/encoders/positional_encoding.py
        d_model=512,
        dropout=0.2,
        max_len=5000),
    decoder=dict(
        type='TableMasterDecoder', # file://./../models/decoders/table_master_decoder.py
        N=3,
        decoder=dict(
            self_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            src_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            feed_forward=dict(
                d_model=512,
                d_ff=2024,
                dropout=0.),
            size=512,
            dropout=0.),
        d_model=512),
    loss=dict(
        type='MasterTFLoss', # file://./../models/losses/master_tf_loss.py
        ignore_index=PAD, 
        reduction='mean'
    ),
    bbox_loss=dict(
        type='TableL1Loss', # file://./../models/losses/table_l1_loss.py
        reduction='sum'
    ),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)
# endregion
# region Pipeline
TRAIN_STATE = True
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'), # https://github.com/open-mmlab/mmdetection/blob/v2.28.2/mmdet/datasets/pipelines/loading.py
    dict(
        type='TableResize', # file://./../datasets/pipelines/table_resize.py
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad', # file://./../datasets/pipelines/table_pad.py
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(type='TableBboxEncode'), # file://./../datasets/pipelines/table_bbox_endcode.py
    dict(type='ToTensorOCR'), # https://github.com/open-mmlab/mmocr/blob/v0.2.0/mmocr/datasets/pipelines/ocr_transforms.py#L124
    dict(type='NormalizeOCR', **img_norm_cfg), # https://github.com/open-mmlab/mmocr/blob/v0.2.0/mmocr/datasets/pipelines/ocr_transforms.py#L137
    dict(
        type='Collect', # https://github.com/open-mmlab/mmdetection/blob/v2.28.2/mmdet/datasets/pipelines/formatting.py#L290
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'text', 'scale_factor', 'bbox', 'bbox_masks', 'pad_shape']),
]
# Valid pipeline: copy train_pipeline và sửa meta_keys
valid_pipeline = train_pipeline.copy()
valid_pipeline[-1]['meta_keys'] = ['filename', 'ori_shape', 'img_shape', 'scale_factor', 'img_norm_cfg', 'ori_filename', 'bbox', 'bbox_masks', 'pad_shape']

# Test pipeline: copy train_pipeline, bỏ TableBboxEncode và sửa meta_keys
test_pipeline = train_pipeline.copy()
test_pipeline.pop(3)  # Bỏ TableBboxEncode (index 3)
test_pipeline[-1]['meta_keys'] = ['filename', 'ori_shape', 'img_shape', 'scale_factor', 'img_norm_cfg', 'ori_filename', 'pad_shape']
# endregion
# region Dataset
VITABSET_TRAIN_IMAGE_ROOT = "F:/data/vitabset/train"
VITABSET_TRAIN_JSON = "F:/data/vitabset/train.bz2"
VITABSET_VAL_IMAGE_ROOT = "F:/data/vitabset/val"
VITABSET_VAL_JSON = "F:/data/vitabset/val.bz2"
VITABSET_TEST_IMAGE_ROOT = "F:/data/vitabset/test"
VITABSET_TEST_JSON = "F:/data/vitabset/test.bz2"
STRUCTURE_VOCAB_FILE = "d:/BIG Projects/Python/ETV/src/data/structure_vocab.txt"
train = dict(
    type='OCRDataset', # https://github.com/open-mmlab/mmocr/blob/v0.6.3/mmocr/datasets/ocr_dataset.py
    img_prefix=VITABSET_TRAIN_IMAGE_ROOT,
    ann_file=VITABSET_TRAIN_JSON,
    loader=dict(
        type='TableLoader', # file://./../datasets/utils/table_loader.py
        max_data=100,
        random_sample=False,
        split_filter=None,
        parser=dict(
            type='TableParser', # file://./../datasets/utils/table_parser.py
            max_structure_len=500,
        )
    ),
    pipeline=train_pipeline,
    test_mode=False
)

valid = train.copy()
valid.update(
    img_prefix=VITABSET_VAL_IMAGE_ROOT,
    ann_file=VITABSET_VAL_JSON,
    pipeline=valid_pipeline,
    dataset_info='table_master_dataset',
    test_mode=True
)


test = valid.copy()
test.update(
    img_prefix=VITABSET_TEST_IMAGE_ROOT,
    ann_file=VITABSET_TEST_JSON,
    pipeline=test_pipeline,
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train]),
    val=dict(type='ConcatDataset', datasets=[valid]),
    test=dict(type='ConcatDataset', datasets=[test])
)
# endregion
# region Others
# optimizer
optimizer = dict(type='Ranger', lr=1e-3) # file://./../models/optimizer/ranger2020.py
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    step=[12, 15])
total_epochs = 17

# evaluation
evaluation = dict(interval=1, metric='acc')

# fp16
fp16 = dict(loss_scale='dynamic')

# checkpoint setting
checkpoint_config = dict(interval=1)

# log_config
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')

    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# if raise find unused_parameters, use this.
# find_unused_parameters = True
# endregion