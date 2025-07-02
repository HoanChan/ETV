from typing import List
from mmocr.registry import TASK_UTILS
from .base_dictionary import BaseDictionary

@TASK_UTILS.register_module()
class TableMasterCellDictionary(BaseDictionary):
    """Dictionary for cell content recognition in TableMaster.
    
    This dictionary manages character-level tokens for text content inside table cells.
    
    Args:
        dict_file (str): Path to cell content alphabet file
        with_start (bool): Whether to add start token. Defaults to False.
        with_end (bool): Whether to add end token. Defaults to False.
        same_start_end (bool): Whether start and end tokens are the same. Defaults to False.
        with_padding (bool): Whether to add padding token. Defaults to False.
        with_unknown (bool): Whether to add unknown token. Defaults to False.
        start_token (str): Start token string. Defaults to '<BOS>'.
        end_token (str): End token string. Defaults to '<EOS>'.
        start_end_token (str): Combined start/end token when same_start_end=True. Defaults to '<BOS/EOS>'.
        padding_token (str): Padding token string. Defaults to '<PAD>'.
        unknown_token (str): Unknown token string. Defaults to '<UKN>'.
    """
    
    def __init__(self,
                 dict_file: str,
                 with_start: bool = False,
                 with_end: bool = False, 
                 same_start_end: bool = False,
                 with_padding: bool = False,
                 with_unknown: bool = False,
                 start_token: str = '<BOS>',
                 end_token: str = '<EOS>',
                 start_end_token: str = '<BOS/EOS>',
                 padding_token: str = '<PAD>',
                 unknown_token: str = '<UKN>',
                 **kwargs) -> None:
        
        super().__init__(
            dict_file=dict_file,
            with_start=with_start,
            with_end=with_end,
            same_start_end=same_start_end,
            with_padding=with_padding,
            with_unknown=with_unknown,
            start_token=start_token,
            end_token=end_token,
            start_end_token=start_end_token,
            padding_token=padding_token,
            unknown_token=unknown_token,
            **kwargs
        )
    
    def _join_tokens(self, tokens: List[str]) -> str:
        """Join tokens without separator for cell content.
        
        Args:
            tokens (List[str]): List of tokens to join
            
        Returns:
            str: Joined string without commas (more readable for cell content)
        """
        return ''.join(tokens)

