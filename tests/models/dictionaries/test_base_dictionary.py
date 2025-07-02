import pytest
import tempfile
import os
from models.dictionaries.base_dictionary import BaseDictionary


class TestBaseDictionary:
    """Test cases for BaseDictionary class."""

    def create_temp_dict_file(self, content):
        """Helper method to create temporary dictionary file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(content)
        temp_file.close()
        return temp_file.name

    def test_basic_dictionary_creation(self):
        """Test basic dictionary creation with simple characters."""
        dict_content = "a\nb\nc\nd\ne"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = BaseDictionary(dict_file=dict_file)
            assert dictionary.num_classes == 5
            assert dictionary.dict == ['a', 'b', 'c', 'd', 'e']
            assert dictionary.char2idx('a') == 0
            assert dictionary.char2idx('e') == 4
        finally:
            os.unlink(dict_file)

    def test_dictionary_with_special_tokens(self):
        """Test dictionary with special tokens."""
        dict_content = "a\nb\nc"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = BaseDictionary(
                dict_file=dict_file,
                with_start=True,
                with_end=True,
                with_padding=True,
                with_unknown=True
            )
            # Base chars + start + end + padding + unknown = 7
            assert dictionary.num_classes == 7
            assert '<BOS>' in dictionary.dict
            assert '<EOS>' in dictionary.dict
            assert '<PAD>' in dictionary.dict
            assert '<UKN>' in dictionary.dict
        finally:
            os.unlink(dict_file)

    def test_str_to_idx_conversion(self):
        """Test string to index conversion."""
        dict_content = "a\nb\nc\nd\ne"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = BaseDictionary(dict_file=dict_file)
            indices = dictionary.str2idx("a,b,c")
            assert indices == [0, 1, 2]
        finally:
            os.unlink(dict_file)

    def test_idx_to_str_conversion(self):
        """Test index to string conversion."""
        dict_content = "a\nb\nc\nd\ne"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            dictionary = BaseDictionary(dict_file=dict_file)
            result = dictionary.idx2str([0, 1, 2])
            assert result == "a,b,c"
        finally:
            os.unlink(dict_file)

    def test_unknown_character_handling(self):
        """Test handling of unknown characters."""
        dict_content = "a\nb\nc"
        dict_file = self.create_temp_dict_file(dict_content)
        
        try:
            # Without unknown token - should raise exception
            dictionary = BaseDictionary(dict_file=dict_file, with_unknown=False)
            with pytest.raises(Exception):
                dictionary.char2idx('x')
            
            # With unknown token - should return unknown index
            dictionary_with_unknown = BaseDictionary(dict_file=dict_file, with_unknown=True)
            unknown_idx = dictionary_with_unknown.char2idx('x')
            assert unknown_idx == dictionary_with_unknown.unknown_idx
        finally:
            os.unlink(dict_file)

    def teardown_method(self):
        """Clean up any temporary files."""
        pass
