import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from mmocr.structures import TextRecogDataSample
from models.postprocessors.master_postprocessor import MasterPostprocessor


@pytest.fixture
def mock_dictionary():
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
def dictionary_config():
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
def postprocessor(dictionary_config, mock_dictionary):
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

@pytest.mark.parametrize("max_seq_len,start_end_same,expected_start_end,expected_max_seq_len", [
    (50, False, False, 50),
    (None, None, True, 40),  # Default case uses default max_seq_len=40
])
def test_init(dictionary_config, max_seq_len, start_end_same, expected_start_end, expected_max_seq_len):
    """Test MasterPostprocessor initialization."""
    with patch('models.postprocessors.master_postprocessor.BaseTextRecogPostprocessor.__init__') as mock_init:
        kwargs = {'dictionary': dictionary_config}
        if max_seq_len is not None:
            kwargs['max_seq_len'] = max_seq_len
        if start_end_same is not None:
            kwargs['start_end_same'] = start_end_same
            
        processor = MasterPostprocessor(**kwargs)
        
        # Check that parent __init__ was called with correct arguments
        expected_call_kwargs = {
            'dictionary': dictionary_config,
            'max_seq_len': expected_max_seq_len
        }
        
        mock_init.assert_called_once_with(**expected_call_kwargs)
        assert processor.start_end_same == expected_start_end

@pytest.mark.parametrize("tensor_shape,expected_indexes,expected_scores_len", [
    # 2D tensor case
    ((5, 10), [4, 5, 6], 3),
    # 3D tensor case (squeezed)
    ((1, 3, 10), [4, 5], 2),
])
def test_get_single_prediction_tensor_dimensions(postprocessor, tensor_shape, expected_indexes, expected_scores_len):
    """Test get_single_prediction with different tensor dimensions."""
    if len(tensor_shape) == 2:
        # 2D tensor
        probs = torch.full(tensor_shape, -10.0)
        probs[0, 4] = 10.0  # 'a'
        probs[1, 5] = 10.0  # 'b'
        probs[2, 6] = 10.0  # 'c'
        probs[3, 1] = 10.0  # '<EOS>' - should stop here
        if tensor_shape[0] > 4:
            probs[4, 7] = 10.0  # 'd' - should be ignored after EOS
    else:
        # 3D tensor
        probs = torch.full(tensor_shape, -10.0)
        probs[0, 0, 4] = 10.0  # 'a'
        probs[0, 1, 5] = 10.0  # 'b'
        probs[0, 2, 1] = 10.0  # '<EOS>'
        
    char_indexes, char_scores = postprocessor.get_single_prediction(probs)
    
    assert char_indexes == expected_indexes
    assert len(char_scores) == expected_scores_len
    assert all(score > 0.99 for score in char_scores)


def test_get_single_prediction_with_ignore_indexes(postprocessor):
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


@pytest.mark.parametrize("prob_setup,expected_result", [
    # Empty result with only ignore indexes
    ([(0, 0), (1, 2), (2, 3)], ([], [])),  # BOS, PAD, UKN
    # Early end token
    ([(0, 4), (1, 5), (2, 1), (3, 6), (4, 7)], ([4, 5], 2)),  # a, b, EOS, c, d
    # Immediate end token
    ([(0, 1), (1, 4)], ([], [])),  # EOS at start, a
])
def test_get_single_prediction_special_cases(postprocessor, prob_setup, expected_result):
    """Test get_single_prediction with various special cases."""
    probs = torch.full((max(len(prob_setup), 5), 10), -10.0)  # Low baseline
    for pos, idx in prob_setup:
        if pos < probs.shape[0]:
            probs[pos, idx] = 10.0
    
    char_indexes, char_scores = postprocessor.get_single_prediction(probs)
    expected_indexes, expected_len = expected_result
    
    assert char_indexes == expected_indexes
    if isinstance(expected_len, int):
        assert len(char_scores) == expected_len
    else:
        assert char_scores == expected_len


def test_get_single_prediction_with_data_sample(postprocessor):
    """Test get_single_prediction with data_sample parameter."""
    probs = torch.full((2, 10), -10.0)  # Low baseline
    probs[0, 4] = 10.0  # 'a'
    probs[1, 5] = 10.0  # 'b'
    
    data_sample = TextRecogDataSample()
    
    char_indexes, char_scores = postprocessor.get_single_prediction(probs, data_sample)
    
    assert char_indexes == [4, 5]
    assert len(char_scores) == 2


@pytest.mark.parametrize("tensor_shape,expected_score_range", [
    # 2D tensor
    ((3, 10), True),  # Should have score range check
    # 3D tensor
    ((1, 2, 10), False),  # Skip score range check for 3D
])
def test_tensor2idx_input_dimensions(postprocessor, tensor_shape, expected_score_range):
    """Test _tensor2idx with different input dimensions."""
    outputs = torch.full(tensor_shape, -10.0)  # Low baseline
    
    if len(tensor_shape) == 2:
        outputs[0, 4] = 2.0  # 'a'
        outputs[0, 5] = 1.0  # 'b' 
        outputs[1, 5] = 3.0  # 'b'
        outputs[2, 1] = 4.0  # '<EOS>'
    else:  # 3D tensor
        outputs[0, 0, 4] = 2.0  # 'a'
        outputs[0, 1, 5] = 3.0  # 'b'
    
    str_index, str_score = postprocessor._tensor2idx(outputs)
    
    assert str_index == [4, 5]
    assert len(str_score) == 2
    if expected_score_range:
        assert all(0 <= score <= 1 for score in str_score)


def test_tensor2idx_softmax_calculation(postprocessor):
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


def test_tensor2idx_with_numpy_conversion(postprocessor):
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


@pytest.mark.parametrize("tensor_setup,expected_indexes,expected_scores_len", [
    # Empty tensor
    ("empty", [], []),
    # Single timestep
    ("single", [4], 1),
    # Only ignored tokens
    ("ignored", [], []),
])
def test_master_edge_cases(postprocessor, tensor_setup, expected_indexes, expected_scores_len):
    """Test MasterPostprocessor with various edge cases."""
    if tensor_setup == "empty":
        probs = torch.empty(0, 10)
    elif tensor_setup == "single":
        probs = torch.full((1, 10), -10.0)
        probs[0, 4] = 10.0  # 'a'
    elif tensor_setup == "ignored":
        probs = torch.full((3, 10), -10.0)
        probs[0, 0] = 10.0  # start
        probs[1, 2] = 10.0  # padding
        probs[2, 3] = 10.0  # unknown
    
    char_indexes, char_scores = postprocessor.get_single_prediction(probs)
    assert char_indexes == expected_indexes
    if isinstance(expected_scores_len, int):
        assert len(char_scores) == expected_scores_len
    else:
        assert char_scores == expected_scores_len


def test_master_wrong_dimension_tensor(postprocessor):
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


def test_master_nan_inf_values(postprocessor):
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