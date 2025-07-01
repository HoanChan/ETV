from typing import Callable, List, Optional, Sequence, Tuple, Union
from mmocr.datasets import RecogLMDBDataset
from mmocr.registry import DATASETS

@DATASETS.register_module()
class CustomLMDBDataset(RecogLMDBDataset):
    def __init__(
        self,
        lmdb_path: str = '',
        img_color_type: str = 'color',
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = '',
        data_prefix: dict = dict(img_path=''),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
    ) -> None:

        super().__init__(
            ann_file=lmdb_path,
            img_color_type=img_color_type,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)
        
    def parse_data_info(self, raw_anno_info: Tuple[Optional[str], str]) -> Union[dict, List[dict]]:
        """
        Parse raw annotation to target format. The annotation format in OpenMMLab shown as follows.
        .. code-block:: none
            {
                "metainfo":                         # Metadata about the dataset
                {
                    "dataset_type": "test_dataset", # Type of the dataset
                    "task_name": "test_task"        # Name of the task, e.g., "text_recognition"
                },
                "data_list":                        # List of data items (images and annotations)
                [
                {
                    "img_path": "test_img.jpg",     # Path to the image file
                    "height": 604,
                    "width": 640,
                    "instances":                    # List of instances (objects) in the image
                    [
                    {
                        "bbox": [0, 0, 10, 20],                 # Bounding box coordinates [x1, y1, x2, y2] of the object
                        "bbox_label": 1,                        # Label for the object (class id)
                        "mask": [[0,0],[0,10],[10,20],[20,0]],  # Polygon mask of the object for segmentation
                        "extra_anns": [1,2,3]                   # Additional information about the object, e.g., attributes or metadata
                    },
                    ...
                    ]
                },
                ...
                ]
            }
        Args:
            raw_anno_info (str): One raw data information loaded from ``ann_file``.

        Returns:
            (dict): Parsed annotation. (an item in ``data_list``)
        """
        data_info = {}
        img_key, text = raw_anno_info
        data_info['img_key'] = img_key # This is the key to retrieve the image from the LMDB database by prepare_data in RecogLMDBDataset class
        data_info['instances'] = self.text_to_instances(text)  # Convert text to instances format
        return data_info
    
    def text_to_instances(self, text: str) -> List[dict]:
        """
        Convert text to instances format.
        
        Args:
            text (str): The text to be converted.
        
        Returns:
            List[dict]: A list of instances parsed from the text.
        """
        objs = [dict(text=text)] #json.loads(text) if text else []
        # if not isinstance(objs, list):
        #     raise ValueError(f'Expected a list of objects, but got {type(objs)}: {objs.__repr__()}')
        return objs