from typing import Dict, List
from mmocr.datasets.builder import LOADERS, build_parser
from .loader import Loader
import os
import bz2
import random
from typing import Dict, List, Optional
import json

@LOADERS.register_module()
class TableLoader(Loader):
    """PubTabNet Dataset loader for table structure and content recognition.

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
        - .jsonl: Line-by-line JSON format
        - .bz2: JSONL compressed with bz2

    Args:
        ann_file (str): Path to the annotation file.
        parser (dict): Construct parser to parse original annotation infos.
        repeat (int): Repeated times of annotations
        max_seq_len (int): Maximum sequence length for labels.
        max_data (int, optional): Maximum number of data items to load. If -1, all data is loaded.
        random_sample (bool, optional): If True, randomly sample from the dataset up to max_data. Defaults to False.
        split_filter (str, optional): Filter data by split ('train', 'val', 'test'). If None, all splits will be included. Defaults to None.
    """
    def __init__(self,
                 ann_file: str,
                 parser,
                 max_data: int = -1,
                 random_sample: bool = False,
                 split_filter: Optional[str] = None):
        
        super().__init__(ann_file, parser)

        if split_filter is not None:
            assert split_filter in ['train', 'val', 'test'], f"split_filter must be 'train', 'val', or 'test', got {split_filter}"

        self.split_filter = split_filter
        self.max_data = max_data
        self.random_sample = random_sample

    def _load(self, ann_file: str) -> List[Dict]:
        """Load data list from annotation file."""
        if not (ann_file.endswith('.jsonl') or ann_file.endswith('.bz2')):
            raise ValueError(f"Unsupported file format. Only .jsonl and .bz2 are supported, got {ann_file}")
        
        file_opener = bz2.open if ann_file.endswith('.bz2') else open
        data_list = []
        
        with file_opener(ann_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw_data_info = json.loads(line.strip())
                    if self.split_filter and raw_data_info.get('split') != self.split_filter:
                        continue
                    if raw_data_info:
                        data_list.append(raw_data_info)
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