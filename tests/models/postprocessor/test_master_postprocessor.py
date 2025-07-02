import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from mmocr.structures import TextRecogDataSample
from models.postprocessors.master_postprocessor import MasterPostprocessor


class TestMasterPostprocessor:
    """Test cases for MasterPostprocessor class."""

    @pytest.fixture
    def mock_dictionary(self):
        """Create a mock dictionary for testing."""
        dictionary = Mock()
        dictionary.num_classes = 10
        dictionary.start_idx = 0
        dictionary.end_idx = 1
        dictionary.padding_idx = 2
        dictionary.unknown_idx = 3
        dictionary.dict = ['<BOS>', '<EOS>', '<PAD>', '<UKN>', 'a', 'b', 'c', 'd', 'e', 'f']
        return dictionary

    @pytest.fixture
    def dictionary_config(self):
        """Create dictionary config for testing."""
        return {
            'type': 'BaseDictionary',
            'dict_file': 'dummy_path.txt',
            'with_start': True,
            'with_end': True,
            'with_padding': True,
            'with_unknown': True
        }

    @pytest.fixture
    def postprocessor(self, dictionary_config, mock_dictionary):
        """Create MasterPostprocessor instance for testing."""
        with patch('models.postprocessors.master_postprocessor.BaseTextRecogPostprocessor.__init__'):
            processor = MasterPostprocessor(
                dictionary=dictionary_config,
                max_seq_len=40,
                start_end_same=True
            )
            # Mock the dictionary and ignore_indexes
            processor.dictionary = mock_dictionary
            processor.ignore_indexes = [0, 2, 3]  # start, padding, unknown
            processor.max_seq_len = 40
            return processor

    def test_init(self, dictionary_config):
        """Test MasterPostprocessor initialization."""
        with patch('models.postprocessors.master_postprocessor.BaseTextRecogPostprocessor.__init__') as mock_init:
            processor = MasterPostprocessor(
                dictionary=dictionary_config,
                max_seq_len=50,
                start_end_same=False
            )
            
            # Check that parent __init__ was called with correct arguments
            mock_init.assert_called_once_with(
                dictionary=dictionary_config,
                max_seq_len=50
            )
            assert processor.start_end_same == False

    def test_init_default_values(self, dictionary_config):
        """Test MasterPostprocessor initialization with default values."""
        with patch('models.postprocessors.master_postprocessor.BaseTextRecogPostprocessor.__init__'):
            processor = MasterPostprocessor(dictionary=dictionary_config)
            assert processor.start_end_same == True

    def test_get_single_prediction_2d_tensor(self, postprocessor):
        """Test get_single_prediction with 2D tensor input."""
        # Create a 2D tensor (T, C) where T=5, C=10
        # Use large values to ensure high probability after softmax
        probs = torch.full((5, 10), -10.0)  # Low baseline
        probs[0, 4] = 10.0  # 'a'
        probs[1, 5] = 10.0  # 'b'
        probs[2, 6] = 10.0  # 'c'
        probs[3, 1] = 10.0  # '<EOS>' - should stop here
        probs[4, 7] = 10.0  # 'd' - should be ignored after EOS
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        
        assert char_indexes == [4, 5, 6]  # 'a', 'b', 'c'
        assert len(char_scores) == 3
        # Scores should be high due to large logit values
        assert all(score > 0.99 for score in char_scores)

    def test_get_single_prediction_3d_tensor(self, postprocessor):
        """Test get_single_prediction with 3D tensor input (should be squeezed)."""
        # Create a 3D tensor (1, T, C) where T=3, C=10
        probs = torch.full((1, 3, 10), -10.0)  # Low baseline
        probs[0, 0, 4] = 10.0  # 'a'
        probs[0, 1, 5] = 10.0  # 'b'
        probs[0, 2, 1] = 10.0  # '<EOS>'
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        
        assert char_indexes == [4, 5]  # 'a', 'b'
        assert len(char_scores) == 2
        assert all(score > 0.99 for score in char_scores)

    def test_get_single_prediction_with_ignore_indexes(self, postprocessor):
        """Test get_single_prediction ignores specified indexes."""
        # Create tensor with ignore indexes (start, padding, unknown)
        probs = torch.full((6, 10), -10.0)  # Low baseline
        probs[0, 0] = 10.0  # '<BOS>' - should be ignored
        probs[1, 4] = 10.0  # 'a'
        probs[2, 2] = 10.0  # '<PAD>' - should be ignored
        probs[3, 5] = 10.0  # 'b'
        probs[4, 3] = 10.0  # '<UKN>' - should be ignored
        probs[5, 6] = 10.0  # 'c'
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        
        assert char_indexes == [4, 5, 6]  # 'a', 'b', 'c'
        assert len(char_scores) == 3
        assert all(score > 0.99 for score in char_scores)

    def test_get_single_prediction_empty_result(self, postprocessor):
        """Test get_single_prediction with only ignore indexes."""
        # Create tensor with only ignore indexes
        probs = torch.full((3, 10), -10.0)  # Low baseline
        probs[0, 0] = 10.0  # '<BOS>'
        probs[1, 2] = 10.0  # '<PAD>'
        probs[2, 3] = 10.0  # '<UKN>'
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        
        assert char_indexes == []
        assert char_scores == []

    def test_get_single_prediction_early_end(self, postprocessor):
        """Test get_single_prediction stops at end token."""
        # Create tensor with end token in the middle
        probs = torch.full((5, 10), -10.0)  # Low baseline
        probs[0, 4] = 10.0  # 'a'
        probs[1, 5] = 10.0  # 'b'
        probs[2, 1] = 10.0  # '<EOS>' - should stop here
        probs[3, 6] = 10.0  # 'c' - should not be included
        probs[4, 7] = 10.0  # 'd' - should not be included
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        
        assert char_indexes == [4, 5]  # 'a', 'b'
        assert len(char_scores) == 2

    def test_get_single_prediction_with_data_sample(self, postprocessor):
        """Test get_single_prediction with data_sample parameter."""
        probs = torch.full((2, 10), -10.0)  # Low baseline
        probs[0, 4] = 10.0  # 'a'
        probs[1, 5] = 10.0  # 'b'
        
        data_sample = TextRecogDataSample()
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs, data_sample)
        
        assert char_indexes == [4, 5]
        assert len(char_scores) == 2

    def test_tensor2idx_2d_input(self, postprocessor):
        """Test _tensor2idx with 2D tensor."""
        outputs = torch.full((3, 10), -10.0)  # Low baseline
        outputs[0, 4] = 2.0  # 'a' - highest probability
        outputs[0, 5] = 1.0  # 'b' - lower probability
        outputs[1, 5] = 3.0  # 'b'
        outputs[2, 1] = 4.0  # '<EOS>'
        
        str_index, str_score = postprocessor._tensor2idx(outputs)
        
        assert str_index == [4, 5]  # 'a', 'b' (stops at EOS)
        assert len(str_score) == 2
        # Scores should be softmax probabilities, so they should be between 0 and 1
        assert all(0 <= score <= 1 for score in str_score)

    def test_tensor2idx_3d_input_squeeze(self, postprocessor):
        """Test _tensor2idx with 3D tensor (should squeeze first dimension)."""
        outputs = torch.full((1, 2, 10), -10.0)  # Low baseline
        outputs[0, 0, 4] = 2.0  # 'a'
        outputs[0, 1, 5] = 3.0  # 'b'
        
        str_index, str_score = postprocessor._tensor2idx(outputs)
        
        assert str_index == [4, 5]
        assert len(str_score) == 2

    def test_tensor2idx_softmax_calculation(self, postprocessor):
        """Test that _tensor2idx correctly applies softmax."""
        outputs = torch.full((2, 10), -10.0)  # Low baseline
        # Set high values for specific indices
        outputs[0, 4] = 10.0  # Very high probability for 'a'
        outputs[0, 5] = 1.0   # Lower probability for 'b'
        outputs[1, 5] = 10.0  # Very high probability for 'b'
        
        str_index, str_score = postprocessor._tensor2idx(outputs)
        
        assert str_index == [4, 5]
        # First score should be very close to 1.0 due to high logit
        assert str_score[0] > 0.99
        # Second score should also be very close to 1.0
        assert str_score[1] > 0.99

    def test_tensor2idx_with_numpy_conversion(self, postprocessor):
        """Test that _tensor2idx correctly converts tensors to numpy and lists."""
        outputs = torch.full((2, 10), -10.0)  # Low baseline
        outputs[0, 4] = 1.0
        outputs[1, 5] = 1.0
        
        str_index, str_score = postprocessor._tensor2idx(outputs)
        
        # Check that results are Python lists, not tensors or numpy arrays
        assert isinstance(str_index, list)
        assert isinstance(str_score, list)
        assert all(isinstance(idx, int) for idx in str_index)
        assert all(isinstance(score, float) for score in str_score)


    def test_master_empty_tensor(self, postprocessor):
        """Test MasterPostprocessor with empty tensor."""
        # Empty 2D tensor
        probs = torch.empty(0, 10)
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        assert char_indexes == []
        assert char_scores == []

    def test_master_single_timestep(self, postprocessor):
        """Test MasterPostprocessor with single timestep."""
        probs = torch.full((1, 10), -10.0)
        probs[0, 4] = 10.0  # 'a'
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        assert char_indexes == [4]
        assert len(char_scores) == 1

    def test_master_only_ignored_tokens(self, postprocessor):
        """Test MasterPostprocessor with only ignored tokens."""
        probs = torch.full((3, 10), -10.0)
        probs[0, 0] = 10.0  # start
        probs[1, 2] = 10.0  # padding
        probs[2, 3] = 10.0  # unknown
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        assert char_indexes == []
        assert char_scores == []

    def test_master_immediate_end_token(self, postprocessor):
        """Test MasterPostprocessor with immediate end token."""
        probs = torch.full((2, 10), -10.0)
        probs[0, 1] = 10.0  # end token at start
        probs[1, 4] = 10.0  # 'a' - should be ignored
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        assert char_indexes == []
        assert char_scores == []

    def test_master_wrong_dimension_tensor(self, postprocessor):
        """Test MasterPostprocessor with wrong tensor dimensions."""
        # 1D tensor - should now handle gracefully after fix
        probs_1d = torch.full((10,), 0.1)  
        # After fix, this should work by treating as single timestep
        char_indexes, char_scores = postprocessor.get_single_prediction(probs_1d)
        # Should return valid result (though might be empty due to ignore indexes)
        assert isinstance(char_indexes, list)
        assert isinstance(char_scores, list)
        
        # 4D tensor - should handle by raising clear error
        probs_4d = torch.full((1, 1, 3, 10), -10.0)
        with pytest.raises(ValueError, match="Expected 2D or 3D tensor"):
            postprocessor.get_single_prediction(probs_4d)

    def test_master_nan_inf_values(self, postprocessor):
        """Test MasterPostprocessor with NaN and Inf values."""
        probs = torch.full((3, 10), 0.0)
        probs[0, 4] = float('inf')  # Inf value
        probs[1, 5] = float('nan')  # NaN value
        probs[2, 6] = 1.0  # Normal value
        
        # Should handle NaN/Inf gracefully or raise appropriate error
        try:
            char_indexes, char_scores = postprocessor.get_single_prediction(probs)
            # Check if scores contain NaN/Inf
            assert all(not np.isnan(score) for score in char_scores)
            assert all(not np.isinf(score) for score in char_scores)
        except (ValueError, RuntimeError):
            # Also acceptable to raise error for invalid inputs
            pass