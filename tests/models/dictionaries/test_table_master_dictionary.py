import pytest
import tempfile
import os
from models.dictionaries.table_master_dictionary import TableMasterDictionary


def create_temp_dict_file(content):
    """Helper function to create temporary dictionary file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


@pytest.mark.parametrize("dict_content,expected_tokens,expected_count", [
    (
        "<td></td>\n<tr></tr>\n<tbody></tbody>\n<table></table>",
        ['<td></td>', '<tr></tr>', '<tbody></tbody>', '<table></table>'],
        4
    ),
    (
        "<td></td>\n<tr></tr>",
        ['<td></td>', '<tr></tr>'],
        2
    ),
    (
        "<td></td>",
        ['<td></td>'],
        1
    )
])
def test_table_master_dictionary_creation(dict_content, expected_tokens, expected_count):
    """Test TableMasterDictionary creation with different table structure tokens."""
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterDictionary(dict_file=dict_file)
        assert dictionary.num_classes == expected_count
        assert dictionary.dict == expected_tokens
        assert dictionary.char2idx(expected_tokens[0]) == 0
        if expected_count > 1:
            assert dictionary.char2idx(expected_tokens[-1]) == expected_count - 1
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("special_tokens,expected_additional_count", [
    ({}, 0),  # No special tokens
    ({"with_start": True}, 1),  # Only start token
    ({"with_end": True}, 1),  # Only end token
    ({"with_padding": True}, 1),  # Only padding token
    ({"with_unknown": True}, 1),  # Only unknown token
    ({"with_start": True, "with_end": True}, 2),  # Start and end tokens
    ({"with_start": True, "with_end": True, "with_padding": True, "with_unknown": True}, 4),  # All special tokens
])
def test_table_master_with_special_tokens(special_tokens, expected_additional_count):
    """Test TableMasterDictionary with different combinations of special tokens."""
    dict_content = "<td></td>\n<tr></tr>\n<tbody></tbody>"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterDictionary(dict_file=dict_file, **special_tokens)
        # Base structure tokens (3) + special tokens
        assert dictionary.num_classes == 3 + expected_additional_count
        
        if special_tokens.get("with_start"):
            assert '<BOS>' in dictionary.dict
        if special_tokens.get("with_end"):
            assert '<EOS>' in dictionary.dict
        if special_tokens.get("with_padding"):
            assert '<PAD>' in dictionary.dict
        if special_tokens.get("with_unknown"):
            assert '<UKN>' in dictionary.dict
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("indices,expected_result", [
    ([0], "<td></td>"),
    ([0, 1], "<td></td>,<tr></tr>"),
    ([0, 1, 2], "<td></td>,<tr></tr>,<tbody></tbody>"),
    ([1, 2], "<tr></tr>,<tbody></tbody>"),
])
def test_join_tokens_with_commas(indices, expected_result):
    """Test that TableMasterDictionary joins tokens with commas."""
    dict_content = "<td></td>\n<tr></tr>\n<tbody></tbody>"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterDictionary(dict_file=dict_file)
        result = dictionary.idx2str(indices)
        assert result == expected_result
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("input_str,expected_indices,indices_to_convert,expected_str", [
    ("<td></td>,<tr></tr>", [0, 1], [0, 1], "<td></td>,<tr></tr>"),
    ("<td></td>", [0], [0], "<td></td>"),
    ("<tr></tr>,<tbody></tbody>", [1, 2], [1, 2], "<tr></tr>,<tbody></tbody>"),
    ("<td></td>,<tbody></tbody>", [0, 2], [0, 2], "<td></td>,<tbody></tbody>"),
])
def test_structure_token_conversion(input_str, expected_indices, indices_to_convert, expected_str):
    """Test conversion of structure tokens to indices and back."""
    dict_content = "<td></td>\n<tr></tr>\n<tbody></tbody>\n<table></table>"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterDictionary(dict_file=dict_file)
        
        # Test str2idx
        indices = dictionary.str2idx(input_str)
        assert indices == expected_indices
        
        # Test idx2str
        result = dictionary.idx2str(indices_to_convert)
        assert result == expected_str
    finally:
        os.unlink(dict_file)
