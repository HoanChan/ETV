import pytest
import tempfile
import os
from models.dictionaries.table_master_cell_dictionary import TableMasterCellDictionary


def create_temp_dict_file(content):
    """Helper function to create temporary dictionary file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def test_cell_dictionary_creation():
    """Test TableMasterCellDictionary creation with cell content characters."""
    dict_content = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n \n.\n,\n-"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterCellDictionary(dict_file=dict_file)
        assert dictionary.num_classes == 40  # 26 letters + 10 digits + 4 punctuation
        assert 'a' in dictionary.dict
        assert '9' in dictionary.dict
        assert ' ' in dictionary.dict
        assert '.' in dictionary.dict
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("with_start,with_end,with_padding,with_unknown,expected_classes,expected_tokens", [
    (True, True, True, True, 10, ['<BOS>', '<EOS>', '<PAD>', '<UKN>']),
    (True, False, False, False, 7, ['<BOS>']),
    (False, True, False, False, 7, ['<EOS>']),
    (False, False, True, False, 7, ['<PAD>']),
    (False, False, False, True, 7, ['<UKN>']),
    (False, False, False, False, 6, []),
])
def test_cell_dictionary_with_special_tokens(with_start, with_end, with_padding, with_unknown, expected_classes, expected_tokens):
    """Test TableMasterCellDictionary with different special token combinations."""
    dict_content = "a\nb\nc\n1\n2\n3"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterCellDictionary(
            dict_file=dict_file,
            with_start=with_start,
            with_end=with_end,
            with_padding=with_padding,
            with_unknown=with_unknown
        )
        # Base chars + special tokens
        assert dictionary.num_classes == expected_classes
        
        for token in expected_tokens:
            assert token in dictionary.dict
    finally:
        os.unlink(dict_file)


def test_join_tokens_without_separator():
    """Test that TableMasterCellDictionary joins tokens without separator."""
    dict_content = "H\ne\nl\no\n \nW\nr\nd\na\nb\nc"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterCellDictionary(dict_file=dict_file)
        # Test joining characters for cell content (should be readable text)  
        # H=0, e=1, l=2, o=3, space=4, W=5, r=6, d=7, a=8, b=9, c=10
        result = dictionary.idx2str([0, 1, 2, 2, 3, 5, 3, 6, 2, 7])  # "HelloWorld" (no space)
        assert result == "HelloWorld"  # No separators for readability
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("input_str,expected_indices,expected_output", [
    ("a,b,c, ,1,2,3", [0, 1, 2, 5, 6, 7, 8], "abc 123"),
    ("a,b,c", [0, 1, 2], "abc"),
    ("1,2,3", [6, 7, 8], "123"),
    (" ", [5], " "),
])
def test_cell_content_conversion(input_str, expected_indices, expected_output):
    """Test conversion of cell content characters with different inputs."""
    dict_content = "a\nb\nc\nd\ne\n \n1\n2\n3"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterCellDictionary(dict_file=dict_file)
        
        # Test str2idx with comma-separated characters
        indices = dictionary.str2idx(input_str)
        assert indices == expected_indices
        
        # Test idx2str - should join without commas for readability
        result = dictionary.idx2str(expected_indices)
        assert result == expected_output
    finally:
        os.unlink(dict_file)


@pytest.mark.parametrize("attribute", [
    'char2idx', 'str2idx', 'idx2str', 'num_classes', 'dict'
])
def test_cell_dictionary_inheritance(attribute):
    """Test that TableMasterCellDictionary properly inherits from BaseDictionary."""
    dict_content = "a\nb\nc"
    dict_file = create_temp_dict_file(dict_content)
    
    try:
        dictionary = TableMasterCellDictionary(dict_file=dict_file)
        
        # Test inherited methods work correctly
        assert hasattr(dictionary, attribute)
        
        # Test basic functionality for specific cases
        if attribute == 'char2idx':
            assert dictionary.char2idx('a') == 0
        elif attribute == 'num_classes':
            assert dictionary.num_classes == 3
    finally:
        os.unlink(dict_file)
