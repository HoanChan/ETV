from typing import Callable, Dict, List, Optional, Any, Union
from mmcv.transforms.base import BaseTransform
from mmocr.registry import TRANSFORMS

@TRANSFORMS.register_module()
class Filter(BaseTransform):
    """
    Filter instances in a list by multiple key conditions.

    Args:
        conditions (dict): Dict of key to condition or value.
            If value is a function, it should take the value and return True/False.
            If value is not a function, instance[key] == value is checked.
        input_key (str): The key in results containing the list of instances. Default: 'instances'.
        output_key (Optional[str]): The key to store filtered results. If None, overwrite input_key.
        mode (str): 'and' (all conditions must be True) or 'or' (any condition True). Default: 'and'.
    
    Example:
        dict(
            type='Filter',
            conditions={
                'type': 'content',
                'bbox': lambda x: len(x) == 4
            },
            input_key='instances',
            output_key='filtered_instances',
            mode='and'
        )
    This will filter instances where 'type' is 'content' and 'bbox' is a list of length 4.
    """
    def __init__(self,
                 conditions: Dict[str, Union[Callable[[Any], bool], Any]],
                 input_key: str = 'instances',
                 output_key: Optional[str] = None,
                 mode: str = 'and'):
        super().__init__()
        self.conditions = conditions
        self.input_key = input_key
        self.output_key = output_key or input_key
        assert mode in ('and', 'or'), "mode must be 'and' or 'or'"
        self.mode = mode

    def _check_instance(self, inst: Dict) -> bool:
        results = []
        for key, cond in self.conditions.items():
            value = inst.get(key)
            if callable(cond):
                results.append(cond(value))
            else:
                results.append(value == cond)
        if not results:
            return True
        if self.mode == 'and':
            return all(results)
        else:
            return any(results)

    def transform(self, results: Dict) -> Dict:
        instances = results.get(self.input_key, [])
        filtered = [inst for inst in instances if self._check_instance(inst)]
        results[self.output_key] = filtered
        return results

    def __repr__(self):
        return (f"Filter(conditions={self.conditions}, input_key='{self.input_key}', "
                f"output_key='{self.output_key}', mode='{self.mode}')")
