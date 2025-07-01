# Copyright (c) OpenMMLab. All rights reserved.
# Dictionary classes for mmOCR 1.x TableMaster implementation

from typing import List, Optional, Union
import mmengine

from mmocr.registry import MODELS
from mmocr.models.common.dictionary import Dictionary


@MODELS.register_module()
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
    
    def idx2str(self, indexes: List[List[int]]) -> List[str]:
        """Convert character indices to structure strings.
        
        Args:
            indexes (List[List[int]]): Character indices
            
        Returns:
            List[str]: Structure strings joined with commas
        """
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            # Join structure tokens with commas for table parsing
            string = ','.join(string)
            strings.append(string)
        return strings


@MODELS.register_module()
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
    
    def idx2str(self, indexes: List[List[int]]) -> List[str]:
        """Convert character indices to cell content strings.
        
        Args:
            indexes (List[List[int]]): Character indices
            
        Returns:
            List[str]: Cell content strings
        """
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            # Join characters directly for cell content
            string = ''.join(string)
            strings.append(string)
        return strings


@MODELS.register_module()
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
            dict_list = self._get_predefined_dict(dict_type)
        
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
    
    def _get_predefined_dict(self, dict_type: str) -> List[str]:
        """Get predefined dictionary by type.
        
        Args:
            dict_type (str): Dictionary type
            
        Returns:
            List[str]: Character list
        """
        # Basic character sets - extend as needed
        if dict_type == 'DICT36':
            return list('0123456789abcdefghijklmnopqrstuvwxyz')
        elif dict_type == 'DICT90':
            chars = list('0123456789abcdefghijklmnopqrstuvwxyz')
            chars.extend(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            chars.extend(list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '))
            return chars
        else:
            raise ValueError(f"Unknown dict_type: {dict_type}")
