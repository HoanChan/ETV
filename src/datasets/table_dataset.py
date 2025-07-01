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
    
    This dataset supports both table structure recognition and cell content recognition
    tasks for table understanding.

    The annotation format for PubTabNet:
    {
        'filename': str,
        'split': str,  # 'train', 'val', or 'test'
        'imgid': int,
        'html': {
            'structure': {'tokens': [str]},
            'cell': [
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
        max_seq_len (int): Maximum sequence length for structure tokens.
            Defaults to 500.
        max_cell_len (int): Maximum sequence length for cell content.
            Defaults to 150.
        ignore_empty_cells (bool): Whether to ignore empty cells in training.
            Defaults to True.
        normalize_bbox (bool): Whether to normalize bbox coordinates to [0, 1].
            Defaults to True.
        **kwargs: Other arguments passed to BaseDataset.
    """

    METAINFO = dict(
        dataset_name='PubTabNet',
        paper_info=dict(
            title='PubTabNet: A Large-scale Dataset for Table Recognition',
            authors='Zheng et al.',
            url='https://github.com/ibm-aur-nlp/PubTabNet'
        )
    )

    def __init__(self,
                 task_type: str = 'both',
                 split_filter: Optional[str] = None,
                 max_seq_len: int = 500,
                 max_cell_len: int = 150,
                 ignore_empty_cells: bool = True,
                 normalize_bbox: bool = True,
                 **kwargs):
        
        assert task_type in ['structure', 'content', 'both'], \
            f"task_type must be 'structure', 'content', or 'both', got {task_type}"
        
        if split_filter is not None:
            assert split_filter in ['train', 'val', 'test'], \
                f"split_filter must be 'train', 'val', or 'test', got {split_filter}"
        
        self.task_type = task_type
        self.split_filter = split_filter
        self.max_seq_len = max_seq_len
        self.max_cell_len = max_cell_len
        self.ignore_empty_cells = ignore_empty_cells
        self.normalize_bbox = normalize_bbox
        
        super().__init__(**kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load data list from annotation file."""
        with get_local_path(self.ann_file) as local_path:
            # Check if file is compressed with bz2
            if local_path.endswith('.bz2'):
                with bz2.open(local_path, 'rt', encoding='utf-8') as f:
                    if local_path.endswith('.jsonl.bz2'):
                        # Line-by-line JSON format compressed with bz2
                        raw_data_list = [json_loads(line.strip()) for line in f if line.strip()]
                    else:
                        # Standard JSON format compressed with bz2
                        raw_data_list = json_load(f)
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
        """Parse raw data info to the format needed by the model."""
        try:
            data_info = {}
            
            # Basic image information
            data_info['img_path'] = os.path.join(self.data_prefix.get('img_path', ''), 
                                               raw_data_info['filename'])
            
            # Image size will be determined later in the pipeline
            data_info['height'] = None
            data_info['width'] = None
            data_info['img_id'] = raw_data_info.get('imgid', 0)
            data_info['split'] = raw_data_info.get('split', 'train')
            
            # Parse HTML structure and content
            html_data = raw_data_info.get('html', {})
            
            # Structure tokens
            structure_data = html_data.get('structure', {})
            structure_tokens = structure_data.get('tokens', [])
            
            if self.task_type in ['structure', 'both']:
                data_info['structure_text'] = ' '.join(structure_tokens[:self.max_seq_len])
                data_info['structure_tokens'] = structure_tokens[:self.max_seq_len]
            
            # Cell data - PubTabNet uses 'cell' key
            cells_data = html_data.get('cell', [])
            
            if self.task_type in ['content', 'both']:
                cell_contents = []
                cell_bboxes = []
                cell_masks = []  # Mask for valid cells (non-empty)
                
                for cell in cells_data:
                    cell_tokens = cell.get('tokens', [])
                    cell_bbox = cell.get('bbox', [])
                    
                    # Cell content
                    if cell_tokens:
                        cell_content = ' '.join(cell_tokens[:self.max_cell_len])
                        cell_contents.append(cell_content)
                        cell_masks.append(1)
                    else:
                        if not self.ignore_empty_cells:
                            cell_contents.append('')
                            cell_masks.append(0)
                        else:
                            continue
                    
                    # Cell bbox
                    if len(cell_bbox) == 4:
                        bbox = cell_bbox.copy()
                        # Note: bbox normalization will be handled in the pipeline
                        # when actual image size is available
                        cell_bboxes.append(bbox)
                    else:
                        cell_bboxes.append([0.0, 0.0, 0.0, 0.0])
                
                data_info['cell_content'] = cell_contents
                data_info['cell_bbox'] = cell_bboxes
                data_info['cell_masks'] = cell_masks
                data_info['num_cells'] = len(cell_contents)
            
            return data_info
            
        except Exception as e:
            print(f"Error parsing data info: {e}")
            print(f"Raw data: {raw_data_info}")
            return None

    def get_data_info(self, idx: int) -> Dict:
        """Get data info by index."""
        data_info = super().get_data_info(idx)
        
        # Add task-specific information
        data_info['task_type'] = self.task_type
        
        return data_info

    def __repr__(self) -> str:
        """Print the basic information of the dataset."""
        repr_str = (f'{self.__class__.__name__}('
                   f'task_type={self.task_type}, '
                   f'split_filter={self.split_filter}, '
                   f'num_samples={len(self)}, '
                   f'ann_file={self.ann_file})')
        return repr_str
