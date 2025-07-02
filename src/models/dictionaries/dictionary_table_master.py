from typing import List, Optional, Union
import mmengine

from mmocr.registry import MODELS, TASK_UTILS
from mmocr.models.common.dictionary import Dictionary


@TASK_UTILS.register_module()
class TableMasterDictionary(Dictionary):
    """Dictionary for table structure recognition in TableMaster.
    
    This dictionary manages table structure tokens like <td>, <tr>, <tbody>, etc.
    and special tokens like <SOS>, <EOS>, <PAD>, <UKN>.
    
    Args:
        dict_file (str): Path to structure alphabet file
        with_start (bool): Whether to add start token. Defaults to True.
        with_end (bool): Whether to add end token. Defaults to True.  
        same_start_end (bool): Whether start and end tokens are the same. Defaults to False.
        with_padding (bool): Whether to add padding token. Defaults to True.
        with_unknown (bool): Whether to add unknown token. Defaults to True.
        start_token (str): Start token string. Defaults to '<SOS>'.
        end_token (str): End token string. Defaults to '<EOS>'.
        start_end_token (str): Combined start/end token when same_start_end=True.
        padding_token (str): Padding token string. Defaults to '<PAD>'.
        unknown_token (str): Unknown token string. Defaults to '<UKN>'.
    """
    
    def __init__(self,
                 dict_file: str,
                 with_start: bool = True,
                 with_end: bool = True,
                 same_start_end: bool = False,
                 with_padding: bool = True,
                 with_unknown: bool = True,
                 start_token: str = '<SOS>',
                 end_token: str = '<EOS>',
                 start_end_token: Optional[str] = None,
                 padding_token: str = '<PAD>',
                 unknown_token: str = '<UKN>',
                 **kwargs) -> None:
        
        # Load structure alphabet from file
        dict_list = []
        if dict_file:
            for line in mmengine.list_from_file(dict_file):
                line = line.strip('\n')  # Preserve space tokens
                if line != '':
                    dict_list.append(line)
        
        super().__init__(
            dict_list=dict_list,
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
    
    def idx2str(self, indexes: List[int]) -> str:
        """Convert character indices to structure string.
        
        Args:
            indexes (List[int]): Character indices
            
        Returns:
            str: Structure string joined with commas
        """
        string = [self._dict[i] for i in indexes]
        # Join structure tokens with commas for table parsing
        return ','.join(string)


@TASK_UTILS.register_module()
class TableMasterCellDictionary(Dictionary):
    """Dictionary for cell content recognition in TableMaster.
    
    This dictionary manages character-level tokens for text content inside table cells.
    
    Args:
        dict_file (str): Path to cell content alphabet file
        with_start (bool): Whether to add start token. Defaults to True.
        with_end (bool): Whether to add end token. Defaults to True.
        same_start_end (bool): Whether start and end tokens are the same. Defaults to False.
        with_padding (bool): Whether to add padding token. Defaults to True.
        with_unknown (bool): Whether to add unknown token. Defaults to True.
        start_token (str): Start token string. Defaults to '<SOS>'.
        end_token (str): End token string. Defaults to '<EOS>'.
        start_end_token (str): Combined start/end token when same_start_end=True.
        padding_token (str): Padding token string. Defaults to '<PAD>'.
        unknown_token (str): Unknown token string. Defaults to '<UKN>'.
    """
    
    def __init__(self,
                 dict_file: str,
                 with_start: bool = True,
                 with_end: bool = True, 
                 same_start_end: bool = False,
                 with_padding: bool = True,
                 with_unknown: bool = True,
                 start_token: str = '<SOS>',
                 end_token: str = '<EOS>',
                 start_end_token: Optional[str] = None,
                 padding_token: str = '<PAD>',
                 unknown_token: str = '<UKN>',
                 **kwargs) -> None:
        
        # Load cell content alphabet from file  
        dict_list = []
        if dict_file:
            for line in mmengine.list_from_file(dict_file):
                line = line.strip('\n')  # Preserve space characters
                if line != '':
                    dict_list.append(line)
        
        super().__init__(
            dict_list=dict_list,
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
    
    def idx2str(self, indexes: List[int]) -> str:
        """Convert character indices to cell content string.
        
        Args:
            indexes (List[int]): Character indices
            
        Returns:
            str: Cell content string
        """
        string = [self._dict[i] for i in indexes]
        # Join characters directly for cell content
        return ''.join(string)


@TASK_UTILS.register_module()
class MasterDictionary(Dictionary):
    """Basic dictionary for Master recognition tasks.
    
    Simplified version for non-table recognition tasks.
    
    Args:
        dict_file (Optional[str]): Path to alphabet file
        dict_list (Optional[List[str]]): Character list
        dict_type (str): Dictionary type like 'DICT90'. Defaults to 'DICT90'.
        with_start (bool): Whether to add start token. Defaults to True.
        with_end (bool): Whether to add end token. Defaults to True.
        same_start_end (bool): Whether start and end tokens are the same. Defaults to True.
        with_padding (bool): Whether to add padding token. Defaults to True.
        with_unknown (bool): Whether to add unknown token. Defaults to True.
        start_token (str): Start token string. Defaults to '<SOS>'.
        end_token (str): End token string. Defaults to '<EOS>'.
        start_end_token (str): Combined start/end token when same_start_end=True.
        padding_token (str): Padding token string. Defaults to '<PAD>'.
        unknown_token (str): Unknown token string. Defaults to '<UKN>'.
    """
    
    def __init__(self,
                 dict_file: Optional[str] = None,
                 dict_list: Optional[List[str]] = None,
                 dict_type: str = 'DICT90',
                 with_start: bool = True,
                 with_end: bool = True,
                 same_start_end: bool = True,
                 with_padding: bool = True,
                 with_unknown: bool = True,
                 start_token: str = '<SOS>',
                 end_token: str = '<EOS>',
                 start_end_token: Optional[str] = None,
                 padding_token: str = '<PAD>',
                 unknown_token: str = '<UKN>',
                 **kwargs) -> None:
        
        # Handle different dictionary sources
        if dict_file:
            dict_list = []
            for line in mmengine.list_from_file(dict_file):
                line = line.strip('\n')
                if line != '':
                    dict_list.append(line)
        elif dict_list is None:
            # Use predefined dictionary types
            if dict_type == 'DICT36':
                dict_list = list('0123456789abcdefghijklmnopqrstuvwxyz')
            elif dict_type == 'DICT90':
                chars = list('0123456789abcdefghijklmnopqrstuvwxyz')
                chars.extend(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                chars.extend(list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '))
                dict_list = chars
            else:
                raise ValueError(f"Unknown dict_type: {dict_type}")
        
        super().__init__(
            dict_list=dict_list,
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
