# dbnet_pubtabnet_resnet18_fpnc.py

_base_ = [
    'mmocr::textdet/dbnet/_base_dbnet_resnet18_fpnc.py',
    'mmocr::textdet/_base_/pretrain_runtime.py',
    'mmocr::textdet/_base_/schedules/schedule_sgd_100k.py',
]

# Dataset
dataset_type = 'PubTabNetDataset'
train_img_prefix = 'F:/data/vitabset/train'
train_ann_file = 'F:/data/vitabset/train.bz2'
val_img_prefix = 'F:/data/vitabset/val'
val_ann_file = 'F:/data/vitabset/val.bz2'
test_img_prefix = 'F:/data/vitabset/test'
test_ann_file = 'F:/data/vitabset/test.bz2'

def create_polygon(bbox):
    """Create a polygon from a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, y1, x2, y2, x1, y2]

data_pipeline = [
    dict(
        type='Filter', # file://./../datasets/transforms/filter.py
        conditions={'type': 'content'}
    ), 
    dict(
        type='Update', # file://./../datasets/transforms/update.py
        mapping={
            'bbox_label': 1, # add bbox_label
            'polygon': lambda ins: create_polygon(ins.get('bbox',[0,0,0,0])),
            'ignore': lambda ins: ins.get('bbox',[0,0,0,0]) == [0, 0, 0, 0] # ignore if bbox is empty
        }
    ),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape')
    )
]

train_dataset = dict(
    type=dataset_type,
    ann_file=train_ann_file,
    data_prefix={'img_path': train_img_prefix},
    pipeline=data_pipeline,
    test_mode=False,
    split_filter=None,  # Load all splits available in the file
    max_structure_len=600,
    # max_cell_len=600,
    ignore_empty_cells=True,
    max_data=2000,  # -1 để load toàn bộ, >0 để giới hạn số lượng sample
    random_sample=False, 
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset
)

val_dataset = dict(
    type=dataset_type,
    ann_file=val_ann_file,
    data_prefix={'img_path': val_img_prefix},
    pipeline=data_pipeline,
    test_mode=True,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)

test_dataloader = val_dataloader

val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator

# Custom imports
custom_imports = dict(
    imports=[
        'datasets.table_dataset',
        'datasets.transforms.filter',  # file://./../datasets/transforms/filter.py
        'datasets.transforms.update',  # file://./../datasets/transforms/update.py
    ],
    allow_failed_imports=False
)