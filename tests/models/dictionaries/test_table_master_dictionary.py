import pytest
import tempfile
import os
from models.dictionaries.table_master_dictionary import TableMasterDictionary


class TestTableMasterDictionary:
    """Test cases for TableMasterDictionary class."""

    def create_temp_dict_file(self, content):
        """Helper method to create temporary dictionary file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(content)
        temp_file.close()
        return temp_file.name

    def test_table_master_dictionary_creation(self):
        """Test TableMasterDictionary creation with table structure tokens."""
        dict_content = "<td></td>\n<tr></tr>\n<tbody></tbody>\n<table></table>"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterDictionary(dict_file=dict_file)
            assert dictionary.num_classes == 4
            assert dictionary.dict == ['<td></td>', '<tr></tr>', '<tbody></tbody>', '<table></table>']
            assert dictionary.char2idx('<td></td>') == 0
            assert dictionary.char2idx('<table></table>') == 3
        finally:
            os.unlink(dict_file)

    def test_table_master_with_special_tokens(self):
        """Test TableMasterDictionary with special tokens."""
        dict_content = "<td></td>\n<tr></tr>\n<tbody></tbody>"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterDictionary(
                dict_file=dict_file,
                with_start=True,
                with_end=True,
                with_padding=True,
                with_unknown=True
            )
            # Base structure tokens + special tokens = 7
            assert dictionary.num_classes == 7
            assert '<BOS>' in dictionary.dict
            assert '<EOS>' in dictionary.dict
            assert '<PAD>' in dictionary.dict
            assert '<UKN>' in dictionary.dict
        finally:
            os.unlink(dict_file)

    def test_join_tokens_with_commas(self):
        """Test that TableMasterDictionary joins tokens with commas."""
        dict_content = "<td></td>\n<tr></tr>\n<tbody></tbody>"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterDictionary(dict_file=dict_file)
            result = dictionary.idx2str([0, 1, 2])
            assert result == "<td></td>,<tr></tr>,<tbody></tbody>"
        finally:
            os.unlink(dict_file)

    def test_structure_token_conversion(self):
        """Test conversion of structure tokens to indices and back."""
        dict_content = "<td></td>\n<tr></tr>\n<tbody></tbody>\n<table></table>"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = TableMasterDictionary(dict_file=dict_file)
            
            # Test str2idx
            indices = dictionary.str2idx("<td></td>,<tr></tr>")
            assert indices == [0, 1]
            
            # Test idx2str
            result = dictionary.idx2str([0, 1])
            assert result == "<td></td>,<tr></tr>"
        finally:
            os.unlink(dict_file)

    def teardown_method(self):
        """Clean up any temporary files."""
        pass
