from config import *
# region Dataset
data_pipeline = [
    dict(type='LoadImageFromFile'), # https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/transforms/loading.py#L17
    dict(
        type='TableResize', # file:///./../datasets/transforms/table_resize.py
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

train_dataset = dict(
    type='PubTabNetDataset', # file:///./../datasets/table_dataset.py
    ann_file=VITABSET_TRAIN_JSON,
    data_prefix={'img_path': VITABSET_TRAIN_IMAGE_ROOT},
    task_type='structure',  # 'structure' or 'content' or 'both'
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
test_dataloader.update(data_set=test_dataset)

val_dataset = train_dataset.copy()
val_dataset.update(
    ann_file=VITABSET_VAL_JSON,
    data_prefix={'img_path': VITABSET_VAL_IMAGE_ROOT},
    test_mode=False,
)

val_dataloader = train_dataloader.copy()
val_dataloader.update(data_set=val_dataset)

# endregion
# region Model
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

model = dict(
    type='TABLEMASTER', # file:///./../models/recognizer/table_master.py
    backbone=dict(
        type='TableResNetExtra', # file:///./../models/backbones/table_resnet_extra.py
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
        type='TableMasterConcatDecoder', # file:///./../models/decoders/table_master_concat_decoder.py
        n_layers=3,
        n_heads=8,
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
            dropout=0.
        ),
        d_model=512,
        postprocessor=dict(type='TableMasterPostprocessor'), # file:///./../models/postprocessors/table_master_postprocessor.py),
        module_loss=dict(
            type='TableLoss', # file:///./../models/losses/table_loss.py
            loss_token=dict(
                type='MASTERTFLoss', # file:///./../models/losses/master_tf_loss.py
                ignore_index=PAD, 
                reduction='mean'
            ),
            loss_bbox=dict(
                type='TableL1Loss', # file:///./../models/losses/table_l1_loss.py
                reduction='sum'
            ),
        ),
        max_seq_len=600,
        dictionary=dictionary
    ),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]))
# endregion

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