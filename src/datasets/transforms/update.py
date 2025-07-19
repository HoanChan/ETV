from typing import Dict, Any, Optional, Callable, Union
from mmcv.transforms.base import BaseTransform
from mmocr.registry import TRANSFORMS

@TRANSFORMS.register_module()
class Update(BaseTransform):
    """
    Update instances in a list by adding or updating fields from a mapping dictionary.

    Args:
        mapping (dict): Dict[str, Union[Callable[[Any], Any], Any]].
            If the value is callable, the current value of the key will be passed to the function and the returned value will be used for updating.
            If the value is not callable, it will be assigned directly.
        input_key (str): The key in results containing the list of instances (each is a dict). Default: 'instances'.
        output_key (Optional[str]): The key to store updated results. If None, overwrites input_key.
    Example:
        dict(
            type='Update',
            mapping={
                'source': 'generated',
                'score': lambda i: i.get('score', 0) + 1
            },
            input_key='instances',
            output_key='updated_instances'
        )
    In the example above, each instance will be updated with 'source' set to 'generated', and 'score' will be increased by 1 based on its current value.
    """
    def __init__(self,
                 mapping: Dict[str, Union[Callable[[Any], Any], Any]],
                 input_key: str = 'instances',
                 output_key: Optional[str] = None):
        super().__init__()
        self.mapping = mapping
        self.input_key = input_key
        self.output_key = output_key or input_key

    def transform(self, results: Dict) -> Dict:
        instances = results.get(self.input_key, [])
        updated = []
        for inst in instances:
            inst = inst.copy()
            for key, value in self.mapping.items():
                if callable(value):
                    inst[key] = value(inst)
                else:
                    inst[key] = value
            updated.append(inst)
        results[self.output_key] = updated
        return results

    def __repr__(self):
        return (f"Update(mapping={self.mapping}, input_key='{self.input_key}', "
                f"output_key='{self.output_key}')")