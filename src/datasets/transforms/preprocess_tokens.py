# Copyright (c) Lê Hoàn Chân. All rights reserved.
from mmcv.transforms import BaseTransform
from mmocr.registry import TRANSFORMS
from datasets.transforms.transforms_utils import remove_thead_Bb, process_token

@TRANSFORMS.register_module()
class PreprocessTokens(BaseTransform):
    """
    Preprocess and normalize table structure tokens before loading.

    This transform applies normalization steps to the raw structure tokens in the annotation:
        - Remove bold tags ('<b>', '</b>') from table head cells using remove_thead_Bb.
        - Merge common tokens and insert empty bbox tokens using process_token.
    
    Input annotation format:
        {
            'img': ...  # original table image
            'instances': [
                {
                    'tokens': [...],     # list of tokens for the instance (table or cell)
                    'task_type': ...     # 'structure' or 'content',
                    'cell_id': ...,      # only for content
                    'bbox': [...],       # only for content
                    'img': ...           # only for content if already cropped
                },
                ...
            ],
        }
    
    After processing, the structure instance's 'tokens' field will be normalized and ready for further transforms (e.g., LoadTokens).

    Args:
        None
    """
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, results: dict) -> dict:
        """
        Preprocess structure tokens in the annotation.
        - Remove bold tags in table head.
        - Merge and encode tokens for structure instance.
        """
        if 'instances' not in results:
            return results
        # Find structure instance(s)
        structures = [inst for inst in results['instances'] if inst.get('task_type') == 'structure']
        cells = [cell for cell in results['instances'] if cell.get('task_type') == 'content']
        for instance in structures:
            tokens = instance.get('tokens', [])
            tokens = remove_thead_Bb(tokens)
            tokens = process_token(tokens, cells)
            instance['tokens'] = tokens
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
