import pytest
import tempfile
import os
from models.dictionaries.base_dictionary import BaseDictionary


@pytest.fixture
def create_temp_dict_file():
    """Fixture to create temporary dictionary file."""
    def _create_file(content):
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    return _create_file


def test_basic_dictionary_creation(create_temp_dict_file):
    """Test basic dictionary creation with simple characters."""
    dict_content = "a\nb\nc\nd\ne"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = BaseDictionary(dict_file=dict_file)
        assert dictionary.num_classes == 5
        assert dictionary.dict == ['a', 'b', 'c', 'd', 'e']
        assert dictionary.char2idx('a') == 0
        assert dictionary.char2idx('e') == 4
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("with_start,with_end,with_padding,with_unknown,expected_tokens,expected_count", [
    (True, True, True, True, ['<BOS>', '<EOS>', '<PAD>', '<UKN>'], 7),
    (True, False, False, False, ['<BOS>'], 4),
    (False, True, False, False, ['<EOS>'], 4),
    (False, False, True, False, ['<PAD>'], 4),
    (False, False, False, True, ['<UKN>'], 4),
    (True, True, False, False, ['<BOS>', '<EOS>'], 5),
])
def test_dictionary_with_special_tokens(create_temp_dict_file, with_start, with_end, with_padding, with_unknown, expected_tokens, expected_count):
    """Test dictionary with various special token combinations."""
    dict_content = "a\nb\nc"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = BaseDictionary(
            dict_file=dict_file,
            with_start=with_start,
            with_end=with_end,
            with_padding=with_padding,
            with_unknown=with_unknown
        )
        assert dictionary.num_classes == expected_count
        for token in expected_tokens:
            assert token in dictionary.dict
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("input_string,expected_indices", [
    ("a,b,c", [0, 1, 2]),
    ("a", [0]),
    ("e,d,c", [4, 3, 2]),
    ("a,c,e", [0, 2, 4]),
])
def test_str_to_idx_conversion(create_temp_dict_file, input_string, expected_indices):
    """Test string to index conversion with various inputs."""
    dict_content = "a\nb\nc\nd\ne"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = BaseDictionary(dict_file=dict_file)
        indices = dictionary.str2idx(input_string)
        assert indices == expected_indices
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("input_indices,expected_string", [
    ([0, 1, 2], "a,b,c"),
    ([0], "a"),
    ([4, 3, 2], "e,d,c"),
    ([0, 2, 4], "a,c,e"),
    ([], ""),
])
def test_idx_to_str_conversion(create_temp_dict_file, input_indices, expected_string):
    """Test index to string conversion with various inputs."""
    dict_content = "a\nb\nc\nd\ne"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = BaseDictionary(dict_file=dict_file)
        result = dictionary.idx2str(input_indices)
        assert result == expected_string
    finally:
        os.unlink(dict_file)


def test_unknown_character_handling_without_unknown_token(create_temp_dict_file):
    """Test handling of unknown characters without unknown token."""
    dict_content = "a\nb\nc"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = BaseDictionary(dict_file=dict_file, with_unknown=False)
        with pytest.raises(Exception):
            dictionary.char2idx('x')
    finally:
        os.unlink(dict_file)


def test_unknown_character_handling_with_unknown_token(create_temp_dict_file):
    """Test handling of unknown characters with unknown token."""
    dict_content = "a\nb\nc"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = BaseDictionary(dict_file=dict_file, with_unknown=True)
        unknown_idx = dictionary.char2idx('x')
        assert unknown_idx == dictionary.unknown_idx
    finally:
        os.unlink(dict_file)
