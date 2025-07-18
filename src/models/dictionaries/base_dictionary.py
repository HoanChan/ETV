# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import List, Sequence

from mmocr.registry import TASK_UTILS
from mmocr.utils import list_from_file
from mmocr.models.common.dictionary import Dictionary


@TASK_UTILS.register_module()
class BaseDictionary(Dictionary):
    """The class generates a dictionary for recognition. It supports both
    single characters and multi-character tokens. It pre-defines four
    special tokens: ``start_token``, ``end_token``, ``pad_token``, and
    ``unknown_token``, which will be sequentially placed at the end of the
    dictionary when their corresponding flags are True.

    Args:
        dict_file (str): The path of dictionary file where each line
            contains a token (character or multi-character string).
        with_start (bool): The flag to control whether to include the start
            token. Defaults to False.
        with_end (bool): The flag to control whether to include the end token.
            Defaults to False.
        same_start_end (bool): The flag to control whether the start token and
            end token are the same. It only works when both ``with_start`` and
            ``with_end`` are True. Defaults to False.
        with_padding (bool):The padding token may represent more than a
            padding. It can also represent tokens like the blank token in CTC
            or the background token in SegOCR. Defaults to False.
        with_unknown (bool): The flag to control whether to include the
            unknown token. Defaults to False.
        start_token (str): The start token as a string. Defaults to '<BOS>'.
        end_token (str): The end token as a string. Defaults to '<EOS>'.
        start_end_token (str): The start/end token as a string. if start and
            end is the same. Defaults to '<BOS/EOS>'.
        padding_token (str): The padding token as a string.
            Defaults to '<PAD>'.
        unknown_token (str, optional): The unknown token as a string. If it's
            set to None and ``with_unknown`` is True, the unknown token will be
            skipped when converting string to index. Defaults to '<UKN>'.
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
                 unknown_token: str = '<UKN>') -> None:
        self.with_start = with_start
        self.with_end = with_end
        self.same_start_end = same_start_end
        self.with_padding = with_padding
        self.with_unknown = with_unknown
        self.start_end_token = start_end_token
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.unknown_token = unknown_token

        assert isinstance(dict_file, str)
        self._dict = []
        for line_num, line in enumerate(list_from_file(dict_file)):
            line = line.strip('\r\n')
            if line != '':
                self._dict.append(line)

        self._char2idx = {char: idx for idx, char in enumerate(self._dict)}

        self._update_dict()
        assert len(set(self._dict)) == len(self._dict), \
            'Invalid dictionary: Has duplicated characters.'

    @property
    def num_classes(self) -> int:
        """int: Number of output classes. Special tokens are counted.
        """
        return len(self._dict)

    @property
    def dict(self) -> list:
        """list: Returns a list of tokens to recognize, where special
        tokens are counted."""
        return self._dict

    def char2idx(self, char: str, strict: bool = True) -> int:
        """Convert a token to an index via ``Dictionary.dict``.

        Args:
            char (str): The token to convert to index.
            strict (bool): The flag to control whether to raise an exception
                when the token is not in the dictionary. Defaults to True.

        Return:
            int: The index of the token.
        """
        char_idx = self._char2idx.get(char, None)
        if char_idx is None:
            if self.with_unknown:
                return self.unknown_idx
            elif not strict:
                return None
            else:
                raise Exception(f'Token: {char} not in dict,'
                                ' please check gt_label and use'
                                ' custom dict file,'
                                ' or set "with_unknown=True"')
        return char_idx

    def str2idx(self, string: str) -> List:
        """Convert a string to a list of indexes via ``Dictionary.dict``.
        Note: This method processes character by character, so it's best suited
        for single-character tokens.

        Args:
            string (str): The string to convert to indexes with comma-separated tokens.
        Return:
            list: The list of indexes of the string.
        """
        idx = list()
        for s in string.split(','):
            char_idx = self.char2idx(s)
            if char_idx is None:
                if self.with_unknown:
                    continue
                raise Exception(f'Token: {s} not in dict,'
                                ' please check gt_label and use'
                                ' custom dict file,'
                                ' or set "with_unknown=True"')
            idx.append(char_idx)
        return idx

    def idx2str(self, index: Sequence[int]) -> str:
        """Convert a list of index to string.

        Args:
            index (list[int]): The list of indexes to convert to string.

        Return:
            str: The converted string joined with commas.
        """
        assert isinstance(index, (list, tuple))
        if not index:
            return ''
            
        tokens = []
        for i in index:
            assert i < len(self._dict), f'Index: {i} out of range! Index ' \
                                        f'must be less than {len(self._dict)}'
            tokens.append(self._dict[i])
        
        return self._join_tokens(tokens)
    
    def _join_tokens(self, tokens: List[str]) -> str:
        """Join tokens with appropriate separator.
        
        Args:
            tokens (List[str]): List of tokens to join
            
        Returns:
            str: Joined string with commas
        """
        return ','.join(tokens)

    def _update_dict(self):
        """Update the dict with tokens according to parameters."""
        # BOS/EOS
        self.start_idx = None
        self.end_idx = None
        if self.with_start and self.with_end and self.same_start_end:
            self._dict.append(self.start_end_token)
            self.start_idx = len(self._dict) - 1
            self.end_idx = self.start_idx
        else:
            if self.with_start:
                self._dict.append(self.start_token)
                self.start_idx = len(self._dict) - 1
            if self.with_end:
                self._dict.append(self.end_token)
                self.end_idx = len(self._dict) - 1

        # padding
        self.padding_idx = None
        if self.with_padding:
            self._dict.append(self.padding_token)
            self.padding_idx = len(self._dict) - 1

        # unknown
        self.unknown_idx = None
        if self.with_unknown and self.unknown_token is not None:
            self._dict.append(self.unknown_token)
            self.unknown_idx = len(self._dict) - 1

        # update char2idx
        self._char2idx = {}
        for idx, char in enumerate(self._dict):
            self._char2idx[char] = idx
