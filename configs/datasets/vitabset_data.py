vitabset_data_root = r'F:\data\vitabset\val_lmdb'
# custom_imports = dict(imports=['lmdb_dataset'], allow_failed_imports=False)
vitabset_rec_test = dict(
    type = 'CustomLMDBDataset',                  # Dataset class for handling LMDB format
    lmdb_path = vitabset_data_root, # Path to the LMDB folder containing the dataset
    img_color_type = 'color',                    # Color type of the images, can be 'color', 'grayscale', etc.
    metainfo = {
                    "dataset_type": "test_dataset", # Type of the dataset
                    "task_name": "test_task"        # Name of the task, e.g., "text_recognition"
                },
    data_root = vitabset_data_root,  # Root directory of the dataset
    data_prefix = dict(img_path=''), # Prefix for image paths, empty means images are stored in the root directory
    filter_cfg = None,               # No filtering configuration
    indices = None,                  # Use all data in the dataset or set to a specific range to limit the dataset for testing
    serialize_data = True,           # Whether to serialize data for efficient loading
    # pipeline = [],                   # Processing pipeline for data augmentation and transformation
    test_mode = True,               # Indicates that this is not in test mode
    lazy_init = False,               # Whether to load annotations lazily
    max_refetch = 1000,              # Maximum number of attempts to fetch a valid image if the initial fetch fails
)