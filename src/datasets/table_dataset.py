import os
import bz2
import json
from typing import Dict, List, Optional
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_local_path

try:
    import ujson
    json_loads = ujson.loads
    json_load = ujson.load
except ImportError:
    json_loads = json.loads
    json_load = json.load

try:
    from mmocr.registry import DATASETS
except ImportError:
    # Fallback for older versions
    from mmengine.registry import Registry
    DATASETS = Registry('dataset')


@DATASETS.register_module()
class PubTabNetDataset(BaseDataset):
    """PubTabNet Dataset for table structure and content recognition.
    
    This dataset follows mmOCR 1.x data format standards.
    Each sample returns a dict with 'instances' key containing task-specific data.

    The annotation format for PubTabNet:
    {
        'filename': str,
        'split': str,  # 'train', 'val', or 'test'
        'imgid': int,
        'html': {
            'structure': {'tokens': [str]},
            'cells': [
                {
                    'tokens': [str],
                    'bbox': [x0, y0, x1, y1]  # only non-empty cells have this
                }
            ]
        }
    }

    Supported file formats:
        - .json: Standard JSON format
        - .jsonl: Line-by-line JSON format
        - .json.bz2: JSON compressed with bz2
        - .jsonl.bz2: JSONL compressed with bz2

    Args:
        task_type (str): Task type, can be 'structure', 'content', or 'both'.
            - 'structure': Only table structure recognition
            - 'content': Only cell content recognition
            - 'both': Both structure and content recognition
            Defaults to 'both'.
        split_filter (str, optional): Filter data by split ('train', 'val', 'test').
            If None, all splits will be included. Defaults to None.
        max_structure_len (int): Maximum sequence length for structure tokens.
            Defaults to 500.
        max_cell_len (int): Maximum sequence length for cell content.
            Defaults to 150.
        ignore_empty_cells (bool): Whether to ignore empty cells in training.
            Defaults to True.
        **kwargs: Other arguments passed to BaseDataset.
    """

    METAINFO = dict(
        dataset_name='PubTabNet',
        task_name='table_recognition',
        paper_info=dict(
            title='PubTabNet: A Large-scale Dataset for Table Recognition',
            authors='Zheng et al.',
            url='https://github.com/ibm-aur-nlp/PubTabNet'
        )
    )

    def __init__(self,
                 task_type: str = 'both',
                 split_filter: Optional[str] = None,
                 max_structure_len: int = 500,
                 max_cell_len: int = 150,
                 ignore_empty_cells: bool = True,
                 **kwargs):
        
        assert task_type in ['structure', 'content', 'both'], f"task_type must be 'structure', 'content', or 'both', got {task_type}"
        
        if split_filter is not None:
            assert split_filter in ['train', 'val', 'test'], f"split_filter must be 'train', 'val', or 'test', got {split_filter}"

        self.task_type = task_type
        self.split_filter = split_filter
        self.max_structure_len = max_structure_len
        self.max_cell_len = max_cell_len
        self.ignore_empty_cells = ignore_empty_cells
        
        super().__init__(**kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load data list from annotation file."""
        with get_local_path(self.ann_file) as local_path:
            # Check if file is compressed with bz2
            if local_path.endswith('.bz2'):
                with bz2.open(local_path, 'rt', encoding='utf-8') as f:
                    # All bz2 files are treated as JSONL format
                    raw_data_list = [json_loads(line.strip()) for line in f if line.strip()]
            else:
                with open(local_path, 'r', encoding='utf-8') as f:
                    if local_path.endswith('.jsonl'):
                        # Line-by-line JSON format
                        raw_data_list = [json_loads(line.strip()) for line in f if line.strip()]
                    else:
                        # Standard JSON format
                        raw_data_list = json_load(f)

        data_list = []
        for raw_data_info in raw_data_list:
            # Filter by split if specified
            if self.split_filter is not None:
                if raw_data_info.get('split') != self.split_filter:
                    continue
            
            data_info = self.parse_data_info(raw_data_info)
            if data_info is not None:
                data_list.append(data_info)

        return data_list

    def parse_data_info(self, raw_data_info: Dict) -> Optional[Dict]:
        """Parse raw data info to mmOCR format with 'instances' key.
        https://mmocr.readthedocs.io/en/latest/basic_concepts/datasets.html
        https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py
        {
            "metainfo":
            {
              "dataset_type": "test_dataset",
              "task_name": "test_task"
            },
            "data_list":
            [
              {
                "img_path": "test_img.jpg",
                "height": 604,
                "width": 640,
                "instances":
                [
                  {
                    "bbox": [0, 0, 10, 20],
                    "bbox_label": 1,
                    "mask": [[0,0],[0,10],[10,20],[20,0]],
                    "extra_anns": [1,2,3]
                  },
                  {
                    "bbox": [10, 10, 110, 120],
                    "bbox_label": 2,
                    "mask": [[10,10],[10,110],[110,120],[120,10]],
                    "extra_anns": [4,5,6]
                  }
                ]
              },
            ]
        }
        
        """
        try:
            data_info = {}
            
            # Basic image information (required by mmOCR)
            data_info['img_path'] = os.path.join(
                self.data_prefix.get('img_path', ''), 
                raw_data_info['filename']
            )
            
            # Add sample_idx for compatibility
            data_info['sample_idx'] = raw_data_info.get('imgid', 0)
            
            # Parse HTML structure and content
            html_data = raw_data_info.get('html', {})
            
            # Create instances list (mmOCR 1.x format)
            instances = []
            
            if self.task_type in ['structure', 'both']:
                # Structure recognition instance
                structure_data = html_data.get('structure', {})
                structure_tokens = structure_data.get('tokens', [])
                structure_text = ''.join(structure_tokens[:self.max_structure_len])
                
                structure_instance = {
                    'text': structure_text
                }
                if self.task_type == 'both':
                    structure_instance['task_type'] = 'structure'
                instances.append(structure_instance)
            
            if self.task_type in ['content', 'both']:
                # Cell content recognition instances
                cells_data = html_data.get('cells', [])
                
                for idx, cell in enumerate(cells_data):
                    cell_tokens = cell.get('tokens', [])
                    cell_bbox = cell.get('bbox', [])
                    
                    # Skip empty cells if configured
                    if not cell_tokens and self.ignore_empty_cells:
                        continue
                    
                    cell_text = ''.join(cell_tokens[:self.max_cell_len]) if cell_tokens else ''
                    
                    cell_instance = {
                        'text': cell_text,
                        'cell_id': idx
                    }
                    if self.task_type == 'both':
                        cell_instance['task_type'] = 'content'
                    
                    if len(cell_bbox) != 4: continue # Ensure bbox is valid

                    cell_instance['bbox'] = cell_bbox

                    instances.append(cell_instance)
            
            # Store instances in the required format
            data_info['instances'] = instances
            
            # Add metadata
            data_info['img_info'] = {
                'height': None,  # Will be set by pipeline
                'width': None,   # Will be set by pipeline
                'split': raw_data_info.get('split', 'train')
            }
            
            return data_info
            
        except Exception as e:
            print(f"Error parsing data info for {raw_data_info.get('filename', 'unknown')}: {e}")
            return None

    def __repr__(self) -> str:
        """Print the basic information of the dataset."""
        repr_str = (f'{self.__class__.__name__}('
                   f'task_type={self.task_type}, '
                   f'split_filter={self.split_filter}, '
                   f'num_samples={len(self)}, '
                   f'ann_file={self.ann_file})')
        return repr_str