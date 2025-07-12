import os
import bz2
import random
from typing import Dict, List, Optional
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_local_path

import ujson

from mmocr.registry import DATASETS

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

    Ouput format:
    {
        'img_path': str,  # path to the image
        'sample_idx': int,  # sample index
        'instances': [
            {
                'tokens': [str],
                'type': 'structure',      # 'structure' or 'content'
                'cell_id': int,           # only for content
                'bbox': [x0, y0, x1, y1]  # only for content
            }
        ],
        'img_info': {
            'height': int,  # image height
            'width': int,   # image width
            'split': str    # 'train', 'val', or 'test'
        }
    }

    Supported file formats:
        - .jsonl: Line-by-line JSON format
        - .bz2: JSONL compressed with bz2

    Args:
        split_filter (str, optional): Filter data by split ('train', 'val', 'test').
            If None, all splits will be included. Defaults to None.
        max_structure_len (int): Maximum sequence length for structure tokens.
            Defaults to 500.
        max_cell_len (int): Maximum sequence length for cell content.
            Defaults to 150.
        ignore_empty_cells (bool): Whether to ignore empty cells in training.
            Defaults to True.
        max_data (int): Maximum number of data samples to load. If -1, load all data.
            Useful for debugging and testing. Defaults to -1.
        random_sample (bool): Whether to randomly sample max_data samples instead of 
            taking the first max_data samples. Only effective when max_data > 0.
            If True, all data will be loaded into memory for random sampling.
            If False, only reads up to max_data samples (memory efficient).
            Defaults to False.
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
                 split_filter: Optional[str] = None,
                 max_structure_len: int = 500,
                 max_cell_len: int = 150,
                 ignore_empty_cells: bool = True,
                 max_data: int = -1,
                 random_sample: bool = False,
                 **kwargs):
        
        if split_filter is not None:
            assert split_filter in ['train', 'val', 'test'], f"split_filter must be 'train', 'val', or 'test', got {split_filter}"

        self.split_filter = split_filter
        self.max_structure_len = max_structure_len
        self.max_cell_len = max_cell_len
        self.ignore_empty_cells = ignore_empty_cells
        self.max_data = max_data
        self.random_sample = random_sample
        
        super().__init__(**kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load data list from annotation file."""
        with get_local_path(self.ann_file) as local_path:
            if not (local_path.endswith('.jsonl') or local_path.endswith('.bz2')):
                raise ValueError(f"Unsupported file format. Only .jsonl and .bz2 are supported, got {local_path}")
            
            file_opener = bz2.open if local_path.endswith('.bz2') else open
            data_list = []
            
            with file_opener(local_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        raw_data_info = ujson.loads(line.strip())
                        if self.split_filter and raw_data_info.get('split') != self.split_filter:
                            continue
                        data_info = self.parse_data_info(raw_data_info)
                        if data_info:
                            data_list.append(data_info)
                            # Early exit for sequential reading when max_data is set and random_sample is False
                            if self.max_data > 0 and not self.random_sample and len(data_list) >= self.max_data:
                                break
                    except:
                        continue
                
                # Apply sampling/limiting
                if self.max_data >= 0:
                    if self.random_sample and len(data_list) > self.max_data:
                        data_list = random.sample(data_list, self.max_data)
                    else:
                        data_list = data_list[:self.max_data]
            
            return data_list

    def parse_data_info(self, raw_data_info: Dict) -> Optional[Dict]:
        """Parse raw data info to mmOCR format with 'instances' key."""
        try:
            html_data = raw_data_info.get('html', {})
            instances = []
            
            # Structure recognition
            structure_tokens = html_data.get('structure', {}).get('tokens', [])
            instances.append({
                'tokens': structure_tokens[:self.max_structure_len],
                'type': 'structure'
            })
            
            # Cell content recognition
            for idx, cell in enumerate(html_data.get('cells', [])):
                cell_tokens = cell.get('tokens', [])
                cell_bbox = cell.get('bbox', [])
                
                if not cell_tokens and self.ignore_empty_cells:
                    continue
                if len(cell_bbox) != 4:
                    continue
                
                instances.append({
                    'tokens': cell_tokens[:self.max_cell_len],
                    'cell_id': idx,
                    'type': 'content',
                    'bbox': cell_bbox
                })
        
            return {
                'img_path': os.path.join(self.data_prefix.get('img_path', ''), raw_data_info['filename']),
                'sample_idx': raw_data_info.get('imgid', 0),
                'instances': instances,
                'img_info': {
                    'height': None,
                    'width': None,
                    'split': raw_data_info.get('split', 'train')
                }
            }
        except Exception as e:
            print(f"Error parsing data info for {raw_data_info.get('filename', 'unknown')}: {e}")
            return None

    def __repr__(self) -> str:
        """Print the basic information of the dataset."""
        return (f'{self.__class__.__name__}('
                f'split_filter={self.split_filter}, max_data={self.max_data}, '
                f'random_sample={self.random_sample}, num_samples={len(self)})')