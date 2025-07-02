import pytest
import tempfile
import os
from models.dictionaries.table_master_cell_dictionary import TableMasterCellDictionary


class TestTableMasterCellDictionary:
    """Test cases for TableMasterCellDictionary class."""

    def create_temp_dict_file(self, content):
        """Helper method to create temporary dictionary file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(content)
        temp_file.close()
        return temp_file.name

    def test_cell_dictionary_creation(self):
        """Test TableMasterCellDictionary creation with cell content characters."""
        dict_content = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n \n.\n,\n-"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterCellDictionary(dict_file=dict_file)
            assert dictionary.num_classes == 40  # 26 letters + 10 digits + 4 punctuation
            assert 'a' in dictionary.dict
            assert '9' in dictionary.dict
            assert ' ' in dictionary.dict
            assert '.' in dictionary.dict
        finally:
            os.unlink(dict_file)

    def test_cell_dictionary_with_special_tokens(self):
        """Test TableMasterCellDictionary with special tokens."""
        dict_content = "a\nb\nc\n1\n2\n3"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterCellDictionary(
                dict_file=dict_file,
                with_start=True,
                with_end=True,
                with_padding=True,
                with_unknown=True
            )
            # Base chars + special tokens = 10
            assert dictionary.num_classes == 10
            assert '<BOS>' in dictionary.dict
            assert '<EOS>' in dictionary.dict
            assert '<PAD>' in dictionary.dict
            assert '<UKN>' in dictionary.dict
        finally:
            os.unlink(dict_file)

    def test_join_tokens_without_separator(self):
        """Test that TableMasterCellDictionary joins tokens without separator."""
        dict_content = "H\ne\nl\no\n \nW\nr\nd\na\nb\nc"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterCellDictionary(dict_file=dict_file)
            # Test joining characters for cell content (should be readable text)  
            # H=0, e=1, l=2, o=3, space=4, W=5, r=6, d=7, a=8, b=9, c=10
            result = dictionary.idx2str([0, 1, 2, 2, 3, 5, 3, 6, 2, 7])  # "HelloWorld" (no space)
            assert result == "HelloWorld"  # No separators for readability
        finally:
            os.unlink(dict_file)

    def test_cell_content_conversion(self):
        """Test conversion of cell content characters."""
        dict_content = "a\nb\nc\nd\ne\n \n1\n2\n3"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterCellDictionary(dict_file=dict_file)
            
            # Test str2idx with comma-separated characters
            indices = dictionary.str2idx("a,b,c, ,1,2,3")
            assert indices == [0, 1, 2, 5, 6, 7, 8]
            
            # Test idx2str - should join without commas for readability
            result = dictionary.idx2str([0, 1, 2, 5, 6, 7, 8])
            assert result == "abc 123"
        finally:
            os.unlink(dict_file)

    def test_cell_dictionary_inheritance(self):
        """Test that TableMasterCellDictionary properly inherits from BaseDictionary."""
        dict_content = "a\nb\nc"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterCellDictionary(dict_file=dict_file)
            
            # Test inherited methods work correctly
            assert hasattr(dictionary, 'char2idx')
            assert hasattr(dictionary, 'str2idx')
            assert hasattr(dictionary, 'idx2str')
            assert hasattr(dictionary, 'num_classes')
            assert hasattr(dictionary, 'dict')
            
            # Test basic functionality
            assert dictionary.char2idx('a') == 0
            assert dictionary.num_classes == 3
        finally:
            os.unlink(dict_file)

    def teardown_method(self):
        """Clean up any temporary files."""
        pass
