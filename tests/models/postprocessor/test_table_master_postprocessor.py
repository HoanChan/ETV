import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from mmocr.structures import TextRecogDataSample
from models.postprocessors.table_master_postprocessor import TableMasterPostprocessor


@pytest.fixture
def mock_dictionary():
    """Create a mock structure dictionary for testing."""
    dictionary = Mock()
    dictionary.num_classes = 15
    dictionary.start_idx = 0
    dictionary.end_idx = 1
    dictionary.padding_idx = 2
    dictionary.unknown_idx = 3
    dictionary.dict = [
        '<BOS>', '<EOS>', '<PAD>', '<UKN>', 
        '<table>', '</table>', '<tr>', '</tr>', 
        '<td>', '</td>', '<td></td>', 'text1', 'text2', 'text3', 'text4'
    ]
    
    def idx2str(indexes_list):
        """Mock idx2str method."""
        # Handle both single list and list of lists
        if isinstance(indexes_list, list) and len(indexes_list) > 0:
            if isinstance(indexes_list[0], int):
                # Single list case - convert to list of lists
                indexes_list = [indexes_list]
        
        results = []
        for indexes in indexes_list:
            string_parts = []
            for idx in indexes:
                if 0 <= idx < len(dictionary.dict):
                    string_parts.append(dictionary.dict[idx])
            results.append(','.join(string_parts))
        return results
    
    dictionary.idx2str = idx2str
    return dictionary


@pytest.fixture
def mock_cell_dictionary():
    """Create a mock cell dictionary for testing."""
    cell_dictionary = Mock()
    cell_dictionary.num_classes = 10
    cell_dictionary.start_idx = 0
    cell_dictionary.end_idx = 1
    cell_dictionary.padding_idx = 2
    cell_dictionary.unknown_idx = 3
    cell_dictionary.dict = ['<BOS>', '<EOS>', '<PAD>', '<UKN>', 'a', 'b', 'c', 'd', 'e', 'f']
    
    def idx2str(indexes_list):
        """Mock idx2str method for cell dictionary."""
        # Handle both single list and list of lists
        if isinstance(indexes_list, list) and len(indexes_list) > 0:
            if isinstance(indexes_list[0], int):
                # Single list case - convert to list of lists
                indexes_list = [indexes_list]
                
        results = []
        for indexes in indexes_list:
            string_parts = []
            for idx in indexes:
                if 0 <= idx < len(cell_dictionary.dict):
                    string_parts.append(cell_dictionary.dict[idx])
            results.append(''.join(string_parts))
        return results
    
    cell_dictionary.idx2str = idx2str
    return cell_dictionary


@pytest.fixture
def dictionary_config():
    """Create structure dictionary config for testing."""
    return {
        'type': 'BaseDictionary',
        'dict_file': 'structure_dict.txt',
        'with_start': True,
        'with_end': True,
        'with_padding': True,
        'with_unknown': True
    }


@pytest.fixture
def cell_dictionary_config():
    """Create cell dictionary config for testing."""
    return {
        'type': 'BaseDictionary',
        'dict_file': 'cell_dict.txt',
        'with_start': True,
        'with_end': True,
        'with_padding': True,
        'with_unknown': True
    }


@pytest.fixture
def postprocessor(dictionary_config, cell_dictionary_config, mock_dictionary, mock_cell_dictionary):
    """Create TableMasterPostprocessor instance for testing."""
    with patch('models.postprocessors.table_master_postprocessor.BaseTextRecogPostprocessor.__init__'), \
         patch('models.postprocessors.table_master_postprocessor.MODELS.build', return_value=mock_cell_dictionary):
        
        processor = TableMasterPostprocessor(
            dictionary=dictionary_config,
            cell_dictionary=cell_dictionary_config,
            max_seq_len=500,
            max_seq_len_cell=100,
            start_end_same=False
        )
        
        # Mock the dictionaries and ignore_indexes
        processor.dictionary = mock_dictionary
        processor.cell_dictionary = mock_cell_dictionary
        processor.ignore_indexes = [0, 2, 3]  # start, padding, unknown
        
        return processor


@pytest.fixture
def table_mock_dictionary():
    """Create mock table dictionary."""
    dictionary = Mock()
    dictionary.num_classes = 20
    dictionary.start_idx = 0
    dictionary.end_idx = 1
    dictionary.padding_idx = 2
    dictionary.unknown_idx = 3
    dictionary.dict = [
        '<BOS>', '<EOS>', '<PAD>', '<UKN>', 
        '<table>', '</table>', '<tr>', '</tr>', 
        '<td>', '</td>', '<td></td>', '<th>', '</th>', '<th></th>',
        'text1', 'text2', 'text3', 'text4', 'text5', 'text6'
    ]
    
    def idx2str(indexes_list):
        if isinstance(indexes_list, list) and len(indexes_list) > 0:
            if isinstance(indexes_list[0], int):
                indexes_list = [indexes_list]
        results = []
        for indexes in indexes_list:
            string_parts = []
            for idx in indexes:
                if 0 <= idx < len(dictionary.dict):
                    string_parts.append(dictionary.dict[idx])
            results.append(','.join(string_parts))
        return results
    
    dictionary.idx2str = idx2str
    return dictionary


@pytest.fixture
def table_postprocessor(table_mock_dictionary, mock_dictionary):
    """Create TableMasterPostprocessor for testing."""
    with patch('models.postprocessors.table_master_postprocessor.BaseTextRecogPostprocessor.__init__'), \
         patch('models.postprocessors.table_master_postprocessor.MODELS.build', return_value=mock_dictionary):
        
        processor = TableMasterPostprocessor(
            dictionary={'type': 'BaseDictionary'},
            cell_dictionary={'type': 'BaseDictionary'},
            max_seq_len=500,
            max_seq_len_cell=100,
            start_end_same=False
        )
        
        processor.dictionary = table_mock_dictionary
        processor.cell_dictionary = mock_dictionary
        processor.ignore_indexes = [0, 2, 3]
        return processor


# Initialization tests
@pytest.mark.parametrize("max_seq_len,max_seq_len_cell,expected_seq_len,expected_cell_len", [
    (600, 120, 600, 120),  # Custom values
    (None, None, 500, 100),  # Default values (when None passed to fixture)
])
def test_init_valid_config(dictionary_config, cell_dictionary_config, max_seq_len, max_seq_len_cell, expected_seq_len, expected_cell_len):
    """Test TableMasterPostprocessor initialization with valid config."""
    with patch('models.postprocessors.table_master_postprocessor.BaseTextRecogPostprocessor.__init__') as mock_init, \
         patch('models.postprocessors.table_master_postprocessor.MODELS.build') as mock_build:
        
        # Use default values if None
        init_args = {
            'dictionary': dictionary_config,
            'cell_dictionary': cell_dictionary_config,
            'start_end_same': False
        }
        if max_seq_len is not None:
            init_args['max_seq_len'] = max_seq_len
        if max_seq_len_cell is not None:
            init_args['max_seq_len_cell'] = max_seq_len_cell
            
        processor = TableMasterPostprocessor(**init_args)
        
        # Check that parent __init__ was called with correct arguments
        expected_init_args = {
            'dictionary': dictionary_config,
            'max_seq_len': expected_seq_len
        }
        mock_init.assert_called_once_with(**expected_init_args)
        
        # Check that cell dictionary was built
        mock_build.assert_called_once_with(cell_dictionary_config)
        
        assert processor.max_seq_len_cell == expected_cell_len
        assert processor.start_end_same == False


def test_init_invalid_start_end_same(dictionary_config, cell_dictionary_config):
    """Test TableMasterPostprocessor initialization with invalid start_end_same."""
    with patch('models.postprocessors.table_master_postprocessor.BaseTextRecogPostprocessor.__init__'), \
         patch('models.postprocessors.table_master_postprocessor.MODELS.build'):
        
        with pytest.raises(AssertionError, match="TableMaster requires start_end_same=False"):
            TableMasterPostprocessor(
                dictionary=dictionary_config,
                cell_dictionary=cell_dictionary_config,
                start_end_same=True
            )


def test_init_default_values(dictionary_config, cell_dictionary_config):
    """Test TableMasterPostprocessor initialization with default values."""
    with patch('models.postprocessors.table_master_postprocessor.BaseTextRecogPostprocessor.__init__'), \
         patch('models.postprocessors.table_master_postprocessor.MODELS.build'):
        
        processor = TableMasterPostprocessor(
            dictionary=dictionary_config,
            cell_dictionary=cell_dictionary_config
        )
        
        assert processor.max_seq_len_cell == 100
        assert processor.start_end_same == False


@pytest.mark.parametrize("prob_config,expected_indexes,expected_scores_count", [
    (
        # Normal case with multiple tokens
        [(0, 4, 10.0), (1, 8, 10.0), (2, 1, 10.0)],  # positions and values
        [4, 8],  # '<table>', '<td>'
        2
    ),
    (
        # Single token case
        [(0, 4, 10.0), (1, 1, 10.0)],  # '<table>', '<EOS>'
        [4],  # Only '<table>'
        1
    ),
    (
        # Empty result (immediate EOS)
        [(0, 1, 10.0)],  # immediate '<EOS>'
        [],
        0
    ),
])
def test_get_single_prediction(postprocessor, prob_config, expected_indexes, expected_scores_count):
    """Test get_single_prediction method with various configurations."""
    # Create a 2D tensor with appropriate size
    max_pos = max([pos for pos, _, _ in prob_config]) + 1 if prob_config else 1
    probs = torch.full((max_pos, 15), -10.0)  # Low baseline
    
    # Set specified probabilities
    for pos, idx, value in prob_config:
        probs[pos, idx] = value
    
    char_indexes, char_scores = postprocessor.get_single_prediction(probs)
    
    assert char_indexes == expected_indexes
    assert len(char_scores) == expected_scores_count
    if expected_scores_count > 0:
        assert all(score > 0.99 for score in char_scores)


@pytest.mark.parametrize("batch_config,expected_results", [
    (
        # Two samples with different tokens
        {
            'samples': [
                [(0, 4, 10.0), (1, 8, 10.0), (2, 1, 10.0)],  # First sample
                [(0, 6, 10.0), (1, 9, 10.0), (2, 1, 10.0)]   # Second sample
            ]
        },
        {
            'indexes': [[4, 8], [6, 9]],  # Expected indexes for both samples
            'num_samples': 2
        }
    ),
    (
        # Single sample
        {
            'samples': [
                [(0, 4, 10.0), (1, 1, 10.0)]  # Single sample with immediate EOS
            ]
        },
        {
            'indexes': [[4]],
            'num_samples': 1
        }
    ),
    (
        # Three samples with varying lengths
        {
            'samples': [
                [(0, 4, 10.0), (1, 8, 10.0), (2, 1, 10.0)],
                [(0, 6, 10.0), (1, 1, 10.0)],  # Shorter sequence
                [(0, 5, 10.0), (1, 7, 10.0), (2, 9, 10.0), (3, 1, 10.0)]  # Longer sequence
            ]
        },
        {
            'indexes': [[4, 8], [6], [5, 7, 9]],
            'num_samples': 3
        }
    ),
])
def test_tensor2idx_batch_processing(postprocessor, batch_config, expected_results):
    """Test _tensor2idx with batch processing."""
    samples = batch_config['samples']
    batch_size = len(samples)
    max_seq_len = max(len(sample) for sample in samples)
    
    # Create a 3D tensor (N, T, C)
    outputs = torch.full((batch_size, max_seq_len, 15), -10.0)  # Low baseline
    
    # Fill tensor with specified values
    for sample_idx, sample in enumerate(samples):
        for pos, token_idx, value in sample:
            outputs[sample_idx, pos, token_idx] = value
    
    indexes, scores = postprocessor._tensor2idx(outputs)
    
    assert len(indexes) == expected_results['num_samples']
    assert len(scores) == expected_results['num_samples']
    
    for i, expected_idx in enumerate(expected_results['indexes']):
        assert indexes[i] == expected_idx
        assert len(scores[i]) == len(expected_idx)


@pytest.mark.parametrize("tensor_config,expected_indexes,expected_scores_count", [
    (
        # Normal case with ignored tokens
        [(0, 0, 10.0), (1, 4, 10.0), (2, 2, 10.0), (3, 8, 10.0), (4, 3, 10.0)],
        [4, 8],  # Only '<table>', '<td>' (ignoring BOS, PAD, UKN)
        2
    ),
    (
        # All ignored tokens
        [(0, 0, 10.0), (1, 2, 10.0), (2, 3, 10.0)],
        [],  # All should be ignored
        0
    ),
    (
        # Mix of valid and ignored tokens at different positions
        [(0, 2, 10.0), (1, 4, 10.0), (2, 0, 10.0), (3, 8, 10.0), (4, 2, 10.0)],
        [4, 8],  # Skip PAD and BOS tokens
        2
    ),
])
def test_tensor2idx_with_ignore_indexes(postprocessor, tensor_config, expected_indexes, expected_scores_count):
    """Test _tensor2idx ignores specified indexes."""
    max_pos = max([pos for pos, _, _ in tensor_config]) + 1 if tensor_config else 1
    outputs = torch.full((1, max_pos, 15), -10.0)  # Low baseline
    
    # Set specified values
    for pos, idx, value in tensor_config:
        outputs[0, pos, idx] = value
    
    indexes, scores = postprocessor._tensor2idx(outputs)
    
    assert indexes[0] == expected_indexes
    assert len(scores[0]) == expected_scores_count


@pytest.mark.parametrize("cell_config,expected_results", [
    (
        # Two cells with different content
        {
            'cells': [
                [(0, 4, 10.0), (1, 5, 10.0), (2, 1, 10.0)],  # First cell: 'a', 'b', '<EOS>'
                [(0, 6, 10.0), (1, 7, 10.0), (2, 1, 10.0)]   # Second cell: 'c', 'd', '<EOS>'
            ]
        },
        {
            'indexes': [[4, 5], [6, 7]],
            'num_cells': 2
        }
    ),
    (
        # Single cell with minimal content
        {
            'cells': [
                [(0, 4, 10.0), (1, 1, 10.0)]  # 'a', '<EOS>'
            ]
        },
        {
            'indexes': [[4]],
            'num_cells': 1
        }
    ),
    (
        # Empty cell (immediate EOS)
        {
            'cells': [
                [(0, 1, 10.0)]  # immediate '<EOS>'
            ]
        },
        {
            'indexes': [[]],
            'num_cells': 1
        }
    ),
    (
        # Multiple cells with varying lengths
        {
            'cells': [
                [(0, 4, 10.0), (1, 5, 10.0), (2, 6, 10.0), (3, 1, 10.0)],  # 'a', 'b', 'c', '<EOS>'
                [(0, 7, 10.0), (1, 1, 10.0)],  # 'd', '<EOS>'
                [(0, 8, 10.0), (1, 9, 10.0), (2, 1, 10.0)]   # 'e', 'f', '<EOS>'
            ]
        },
        {
            'indexes': [[4, 5, 6], [7], [8, 9]],
            'num_cells': 3
        }
    ),
])
def test_tensor2idx_cell_processing(postprocessor, cell_config, expected_results):
    """Test _tensor2idx_cell for cell content processing."""
    cells = cell_config['cells']
    batch_size = len(cells)
    max_seq_len = max(len(cell) for cell in cells)
    
    # Create cell content tensor
    outputs = torch.full((batch_size, max_seq_len, 10), -10.0)  # Low baseline
    
    # Fill tensor with specified values
    for cell_idx, cell in enumerate(cells):
        for pos, token_idx, value in cell:
            outputs[cell_idx, pos, token_idx] = value
    
    indexes, scores = postprocessor._tensor2idx_cell(outputs)
    
    assert len(indexes) == expected_results['num_cells']
    assert len(scores) == expected_results['num_cells']
    
    for i, expected_idx in enumerate(expected_results['indexes']):
        assert indexes[i] == expected_idx
        assert len(scores[i]) == len(expected_idx)


@pytest.mark.parametrize("strings,expected_masks", [
    (
        ['<table>,<td></td>,</table>', '<tr>,<td,</tr>'],
        [[0, 1, 0], [0, 1, 0]]
    ),
    (
        ['<td></td>', '<td'],
        [[1], [1]]
    ),
    (
        ['<td>', '</td>', '<table>'],
        [[0], [0], [0]]
    ),
])
def test_get_pred_bbox_mask(postprocessor, strings, expected_masks):
    """Test _get_pred_bbox_mask generation."""
    masks = postprocessor._get_pred_bbox_mask(strings)
    
    assert len(masks) == len(expected_masks)
    for actual, expected in zip(masks, expected_masks):
        assert actual == expected


@pytest.mark.parametrize("bbox_data,bbox_mask,expected_indices", [
    (
        np.array([
            [0.1, 0.1, 0.5, 0.5],  # Valid bbox
            [-0.1, 0.2, 0.6, 0.7], # Invalid (negative coordinate)
            [0.2, 0.3, 1.2, 0.8],  # Invalid (> 1.0)
            [0.3, 0.4, 0.7, 0.9]   # Valid bbox
        ]),
        np.array([1, 1, 0, 1]),
        [0, 3]  # Indices of valid bboxes that should be kept
    ),
    (
        np.array([
            [0.0, 0.0, 1.0, 1.0],   # Boundary values - should be valid
            [0.5, 0.5, 0.5, 0.5],   # Zero area bbox
            [-1e-10, 0.0, 1.0, 1.0], # Very slightly negative
            [0.0, 0.0, 1.0000001, 1.0] # Very slightly > 1.0
        ]),
        np.array([1, 1, 1, 1]),
        [0]  # Only first bbox should be valid
    ),
])
def test_filter_invalid_bbox(postprocessor, bbox_data, bbox_mask, expected_indices):
    """Test _filter_invalid_bbox method."""
    filtered_bbox = postprocessor._filter_invalid_bbox(bbox_data, bbox_mask)
    
    assert filtered_bbox.shape == bbox_data.shape
    
    # Check that expected valid bboxes are kept
    for i in expected_indices:
        np.testing.assert_array_almost_equal(filtered_bbox[i], bbox_data[i])
    
    # Check that invalid bboxes are zeroed
    for i in range(len(bbox_data)):
        if i not in expected_indices:
            np.testing.assert_array_almost_equal(filtered_bbox[i], [0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("bbox_config,mask_config,scale_config,expected_count", [
    (
        # Two samples with different number of valid bboxes
        {
            'outputs': [
                [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]],
                [[0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.8, 0.8]]
            ]
        },
        {
            'masks': [[1, 1], [1, 0]]  # First sample: both valid, second: only first valid
        },
        {
            'scale_factor': [2.0, 2.0],
            'pad_shape': [100, 200],
            'img_shape': [80, 160]
        },
        2  # Two samples
    ),
    (
        # Single sample with multiple bboxes
        {
            'outputs': [
                [[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.9, 0.9]]
            ]
        },
        {
            'masks': [[1, 1]]
        },
        {
            'scale_factor': [1.0, 1.0],
            'pad_shape': [50, 100],
            'img_shape': [50, 100]
        },
        1  # One sample
    ),
    (
        # All invalid bboxes
        {
            'outputs': [
                [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]
            ]
        },
        {
            'masks': [[0, 0]]  # All invalid
        },
        {
            'scale_factor': [1.5, 1.5],
            'pad_shape': [75, 150],
            'img_shape': [60, 120]
        },
        1  # One sample but with no valid bboxes
    ),
])
def test_decode_bboxes(postprocessor, bbox_config, mask_config, scale_config, expected_count):
    """Test _decode_bboxes method with various configurations."""
    # Create normalized bbox outputs
    outputs_bbox = torch.tensor(bbox_config['outputs'])
    pred_bbox_masks = mask_config['masks']
    
    # Create mock data samples
    data_samples = []
    for i in range(expected_count):
        data_sample = TextRecogDataSample()
        data_sample.set_metainfo(scale_config)
        data_samples.append(data_sample)
    
    pred_bboxes = postprocessor._decode_bboxes(outputs_bbox, pred_bbox_masks, data_samples)
    
    assert len(pred_bboxes) == expected_count
    
    # Verify bbox transformations for first sample
    if expected_count > 0 and len(pred_bbox_masks[0]) > 0:
        w_scale = scale_config['pad_shape'][1] / scale_config['scale_factor'][0]
        h_scale = scale_config['pad_shape'][0] / scale_config['scale_factor'][1]
        
        # Check that bboxes are properly denormalized and scaled
        original_bbox = bbox_config['outputs'][0][0]  # First bbox of first sample
        if pred_bbox_masks[0][0] == 1:  # If this bbox is valid
            expected_x1 = original_bbox[0] * w_scale
            expected_y1 = original_bbox[1] * h_scale
            expected_x2 = original_bbox[2] * w_scale
            expected_y2 = original_bbox[3] * h_scale
            
            np.testing.assert_array_almost_equal(
                pred_bboxes[0][0], 
                [expected_x1, expected_y1, expected_x2, expected_y2], 
                decimal=4
            )


@pytest.mark.parametrize("bboxes,strings,expected_shapes", [
    (
        # Normal case: more bboxes than tokens
        [
            np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]]),
            np.array([[0.4, 0.4, 0.8, 0.8], [0.5, 0.5, 0.9, 0.9]])
        ],
        ['<table>,<tr>', '<td></td>'],
        [(2, 4), (1, 4)]
    ),
    (
        # Equal number of bboxes and tokens
        [
            np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]),
            np.array([[0.3, 0.3, 0.7, 0.7]])
        ],
        ['<table>,<tr>', '<td>'],
        [(2, 4), (1, 4)]
    ),
    (
        # Fewer bboxes than tokens (edge case)
        [
            np.array([[0.1, 0.1, 0.5, 0.5]]),  # Only 1 bbox
            np.array([[0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]])
        ],
        ['<table>,<tr>,<td>', '<td>,</td>'],  # 3 and 2 tokens respectively
        [(1, 4), (2, 4)]  # Should be limited by available bboxes
    ),
    (
        # Empty strings case
        [
            np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]),
            np.array([[0.3, 0.3, 0.7, 0.7]])
        ],
        ['', '<td>'],
        [(0, 4), (1, 4)]
    ),
    (
        # Single element cases
        [
            np.array([[0.1, 0.1, 0.5, 0.5]])
        ],
        ['<td></td>'],
        [(1, 4)]
    ),
])
def test_adjust_bboxes_len(postprocessor, bboxes, strings, expected_shapes):
    """Test _adjust_bboxes_len method with various configurations."""
    adjusted_bboxes = postprocessor._adjust_bboxes_len(bboxes, strings)
    
    assert len(adjusted_bboxes) == len(expected_shapes)
    for i, expected_shape in enumerate(expected_shapes):
        assert adjusted_bboxes[i].shape == expected_shape
        
        # Check that the content matches the original data (first N elements)
        num_tokens = expected_shape[0]
        if num_tokens > 0:
            np.testing.assert_array_equal(adjusted_bboxes[i], bboxes[i][:num_tokens])
        else:
            # For empty case, should be empty array with correct shape
            assert adjusted_bboxes[i].shape[0] == 0


@pytest.mark.parametrize("str_scores,expected_scores", [
    (
        # Normal case with varying lengths
        [[0.9, 0.8, 0.7], [0.95, 0.85], [], [1.0]],
        [0.8, 0.9, 0.0, 1.0]
    ),
    (
        # Different lengths and values
        [[0.5], [0.1, 0.2, 0.3], [0.99, 0.98, 0.97, 0.96]],
        [0.5, 0.2, 0.975]
    ),
    (
        # Edge cases: all zeros, all ones, very small numbers
        [[0.0, 0.0], [1.0, 1.0], [1e-10, 1e-9, 1e-8]],
        [0.0, 1.0, 1e-9]
    ),
    (
        # Single score lists
        [[0.7], [0.3], [0.99]],
        [0.7, 0.3, 0.99]
    ),
    (
        # Empty scores only
        [[], [], []],
        [0.0, 0.0, 0.0]
    ),
    (
        # Mixed with negative values (edge case)
        [[-0.1, 0.5, 0.8], [0.0], [0.2, 0.4, 0.6, 0.8]],
        [0.4, 0.0, 0.5]
    ),
    (
        # Large numbers
        [[100.0, 200.0], [50.0, 150.0, 250.0]],
        [150.0, 150.0]
    ),
])
def test_get_avg_scores(postprocessor, str_scores, expected_scores):
    """Test _get_avg_scores method with various score configurations."""
    avg_scores = postprocessor._get_avg_scores(str_scores)
    
    assert len(avg_scores) == len(expected_scores)
    for actual, expected in zip(avg_scores, expected_scores):
        assert abs(actual - expected) < 1e-6


def test_format_table_outputs_integration(postprocessor):
    """Test format_table_outputs integration method."""
    # Create mock outputs
    structure_outputs = torch.full((2, 3, 15), -10.0)
    structure_outputs[0, 0, 4] = 10.0  # '<table>'
    structure_outputs[0, 1, 8] = 10.0  # '<td>'
    structure_outputs[0, 2, 1] = 10.0  # '<EOS>'
    
    structure_outputs[1, 0, 6] = 10.0  # '<tr>'
    structure_outputs[1, 1, 9] = 10.0  # '</td>'
    structure_outputs[1, 2, 1] = 10.0  # '<EOS>'
    
    bbox_outputs = torch.tensor([
        [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.0, 0.0, 0.0, 0.0]],
        [[0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.8, 0.8], [0.0, 0.0, 0.0, 0.0]]
    ])
    
    # Cell outputs for each sample
    cell_outputs = [
        torch.full((1, 3, 10), -10.0),  # Empty cell
        torch.full((2, 3, 10), -10.0)   # Cell with content
    ]
    # Add content to second cell output
    cell_outputs[1][0, 0, 4] = 10.0  # 'a'
    cell_outputs[1][0, 1, 5] = 10.0  # 'b'
    cell_outputs[1][0, 2, 1] = 10.0  # '<EOS>'
    
    # Create mock data samples
    data_samples = []
    for i in range(2):
        data_sample = TextRecogDataSample()
        data_sample.set_metainfo({
            'scale_factor': [1.0, 1.0],
            'pad_shape': [100, 100],
            'img_shape': [100, 100]
        })
        data_samples.append(data_sample)
    
    results = postprocessor.format_table_outputs(
        structure_outputs, bbox_outputs, cell_outputs, data_samples
    )
    
    assert len(results) == 2
    
    # Check first result
    result_0 = results[0]
    assert 'structure_text' in result_0
    assert 'structure_score' in result_0
    assert 'bboxes' in result_0
    assert 'cell_texts' in result_0
    assert 'cell_scores' in result_0
    
    # Check that structure text contains expected tokens
    assert '<table>' in result_0['structure_text']
    assert '<td>' in result_0['structure_text']


@pytest.mark.parametrize("cell_output_config,expected_result", [
    (
        # Completely empty cell (only EOS)
        {'shape': (1, 1, 10), 'tokens': [(0, 0, 1, 10.0)]},  # Only EOS
        {'num_samples': 1, 'expected_lengths': [0]}
    ),
    (
        # Single element cells
        {'shape': (2, 2, 10), 'tokens': [(0, 0, 4, 10.0), (0, 1, 1, 10.0), (1, 0, 5, 10.0), (1, 1, 1, 10.0)]},
        {'num_samples': 2, 'expected_lengths': [1, 1]}
    ),
    (
        # Mixed empty and non-empty cells
        {'shape': (3, 3, 10), 'tokens': [
            (0, 0, 1, 10.0),  # First cell: immediate EOS
            (1, 0, 4, 10.0), (1, 1, 5, 10.0), (1, 2, 1, 10.0),  # Second cell: 'a', 'b', EOS
            (2, 0, 6, 10.0), (2, 1, 1, 10.0)  # Third cell: 'c', EOS
        ]},
        {'num_samples': 3, 'expected_lengths': [0, 2, 1]}
    ),
])
def test_empty_cell_content_handling(postprocessor, cell_output_config, expected_result):
    """Test handling of various cell content configurations including empty cells."""
    shape = cell_output_config['shape']
    cell_output = torch.full(shape, -10.0)
    
    # Set specified tokens
    for sample, pos, token, value in cell_output_config['tokens']:
        cell_output[sample, pos, token] = value
    
    cell_indexes, cell_scores = postprocessor._tensor2idx_cell(cell_output)
    
    assert len(cell_indexes) == expected_result['num_samples']
    assert len(cell_scores) == expected_result['num_samples']
    
    for i, expected_length in enumerate(expected_result['expected_lengths']):
        assert len(cell_indexes[i]) == expected_length
        assert len(cell_scores[i]) == expected_length


@pytest.mark.parametrize("missing_attribute,test_config,expected_behavior", [
    (
        'end_idx',
        {'tokens': [(0, 0, 4, 10.0), (0, 1, 1, 10.0)]},  # 'a', then EOS using main dict
        {'should_fallback': True, 'expected_length': 1}
    ),
    (
        'start_idx', 
        {'tokens': [(0, 0, 0, 10.0), (0, 1, 4, 10.0), (0, 2, 1, 10.0)]},  # start, 'a', EOS
        {'should_fallback': True, 'expected_length': 1}  # Should ignore start and process 'a'
    ),
])
def test_cell_dictionary_fallback(postprocessor, missing_attribute, test_config, expected_behavior):
    """Test fallback to main dictionary for cell processing when attributes are missing."""
    # Save original attribute if it exists
    original_value = None
    if hasattr(postprocessor.cell_dictionary, missing_attribute):
        original_value = getattr(postprocessor.cell_dictionary, missing_attribute)
        delattr(postprocessor.cell_dictionary, missing_attribute)
    
    try:
        outputs = torch.full((1, len(test_config['tokens']), 10), -10.0)
        
        # Set specified tokens
        for sample, pos, token, value in test_config['tokens']:
            outputs[sample, pos, token] = value
        
        indexes, scores = postprocessor._tensor2idx_cell(outputs)
        
        assert len(indexes) == 1
        if expected_behavior['should_fallback']:
            assert len(indexes[0]) == expected_behavior['expected_length']
        
    finally:
        # Restore attribute if it existed
        if original_value is not None:
            setattr(postprocessor.cell_dictionary, missing_attribute, original_value)


# Table-specific tests
def test_table_invalid_bbox_filter_edge_cases(table_postprocessor):
    """Test edge cases in bbox filtering."""
    # Test with bboxes having exactly boundary values
    output_bbox = np.array([
        [0.0, 0.0, 1.0, 1.0],   # Boundary values - should be valid
        [0.5, 0.5, 0.5, 0.5],   # Zero area bbox - might be problematic
        [-1e-10, 0.0, 1.0, 1.0], # Very slightly negative - should be invalid
        [0.0, 0.0, 1.0000001, 1.0] # Very slightly > 1.0 - should be invalid
    ])
    
    pred_bbox_mask = np.array([1, 1, 1, 1])
    
    filtered_bbox = table_postprocessor._filter_invalid_bbox(output_bbox, pred_bbox_mask)
    
    # First bbox should be kept (boundary values are valid)
    np.testing.assert_array_almost_equal(filtered_bbox[0], [0.0, 0.0, 1.0, 1.0])
    
    # Second bbox (zero area) - behavior depends on implementation
    # Third and fourth should be zeroed (invalid coordinates)
    np.testing.assert_array_almost_equal(filtered_bbox[2], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(filtered_bbox[3], [0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("mismatch_config,expected_error", [
    (
        # Fewer cell outputs than structure samples
        {
            'num_structure_samples': 3,
            'num_cell_outputs': 1,
            'num_data_samples': 3
        },
        (IndexError, ValueError)
    ),
    (
        # More cell outputs than needed
        {
            'num_structure_samples': 1,
            'num_cell_outputs': 3,
            'num_data_samples': 1
        },
        None  # Should handle gracefully by using only what's needed
    ),
    (
        # Mismatched data samples count
        {
            'num_structure_samples': 2,
            'num_cell_outputs': 2,
            'num_data_samples': 1
        },
        (IndexError, ValueError)
    ),
])
def test_table_mismatched_cell_outputs_length(table_postprocessor, mismatch_config, expected_error):
    """Test handling of mismatched cell_outputs length."""
    num_structure = mismatch_config['num_structure_samples']
    num_cells = mismatch_config['num_cell_outputs']
    num_data = mismatch_config['num_data_samples']
    
    # Create structure outputs
    structure_outputs = torch.full((num_structure, 3, 20), -10.0)
    for i in range(num_structure):
        structure_outputs[i, 0, 4] = 10.0  # '<table>'
        structure_outputs[i, 1, 8] = 10.0  # '<td>'
        structure_outputs[i, 2, 1] = 10.0  # '<EOS>'
    
    bbox_outputs = torch.zeros((num_structure, 3, 4))
    
    # Create cell outputs (potentially mismatched)
    cell_outputs = []
    for i in range(num_cells):
        cell_outputs.append(torch.full((1, 3, 10), -10.0))
    
    # Create data samples
    data_samples = []
    for i in range(num_data):
        data_sample = TextRecogDataSample()
        data_sample.set_metainfo({
            'scale_factor': [1.0, 1.0],
            'pad_shape': [100, 100],
            'img_shape': [100, 100]
        })
        data_samples.append(data_sample)
    
    if expected_error:
        with pytest.raises(expected_error):
            table_postprocessor.format_table_outputs(
                structure_outputs, bbox_outputs, cell_outputs, data_samples
            )
    else:
        # Should handle gracefully
        results = table_postprocessor.format_table_outputs(
            structure_outputs, bbox_outputs, cell_outputs, data_samples
        )
        assert len(results) == num_structure


@pytest.mark.parametrize("metadata_config,expected_behavior", [
    (
        # Completely missing metainfo
        {'metainfo': None},
        {'should_raise_error': True, 'error_keywords': ['metainfo', 'scale_factor']}
    ),
    (
        # Missing scale_factor
        {'metainfo': {'pad_shape': [100, 100], 'img_shape': [100, 100]}},
        {'should_raise_error': True, 'error_keywords': ['scale_factor']}
    ),
    (
        # Missing pad_shape
        {'metainfo': {'scale_factor': [1.0, 1.0], 'img_shape': [100, 100]}},
        {'should_raise_error': True, 'error_keywords': ['pad_shape']}
    ),
    (
        # Missing img_shape (might be optional)
        {'metainfo': {'scale_factor': [1.0, 1.0], 'pad_shape': [100, 100]}},
        {'should_raise_error': False, 'error_keywords': []}
    ),
    (
        # Empty metainfo dict
        {'metainfo': {}},
        {'should_raise_error': True, 'error_keywords': ['scale_factor', 'pad_shape']}
    ),
])
def test_table_missing_metadata_in_data_sample(table_postprocessor, metadata_config, expected_behavior):
    """Test handling of missing metadata in data samples."""
    structure_outputs = torch.full((1, 2, 20), -10.0)
    structure_outputs[0, 0, 4] = 10.0  # '<table>'
    structure_outputs[0, 1, 1] = 10.0  # '<EOS>'
    
    bbox_outputs = torch.zeros((1, 2, 4))
    cell_outputs = [torch.full((1, 2, 10), -10.0)]
    
    # Create data sample with specified metadata configuration
    data_sample = TextRecogDataSample()
    if metadata_config['metainfo'] is not None:
        data_sample.set_metainfo(metadata_config['metainfo'])
    # If metainfo is None, don't set any metainfo
    
    if expected_behavior['should_raise_error']:
        with pytest.raises((KeyError, AttributeError)) as exc_info:
            table_postprocessor.format_table_outputs(
                structure_outputs, bbox_outputs, cell_outputs, [data_sample]
            )
        
        # Check that error message contains expected keywords
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in expected_behavior['error_keywords'])
    else:
        # Should handle gracefully
        try:
            results = table_postprocessor.format_table_outputs(
                structure_outputs, bbox_outputs, cell_outputs, [data_sample]
            )
            assert len(results) == 1
        except Exception as e:
            # If it still fails, make sure it's not due to the missing metadata we're testing
            error_msg = str(e).lower()
            assert not any(keyword in error_msg for keyword in expected_behavior['error_keywords'])


def test_table_cell_dictionary_attribute_missing(table_postprocessor):
    """Test fallback behavior when cell dictionary is missing attributes."""
    # Remove end_idx from cell dictionary
    if hasattr(table_postprocessor.cell_dictionary, 'end_idx'):
        original_end_idx = table_postprocessor.cell_dictionary.end_idx
        del table_postprocessor.cell_dictionary.end_idx
    
    try:
        outputs = torch.full((1, 3, 10), -10.0)
        outputs[0, 0, 4] = 10.0  # 'a'
        outputs[0, 1, 5] = 10.0  # 'b'
        outputs[0, 2, 1] = 10.0  # Should use main dictionary's end_idx
        
        indexes, scores = table_postprocessor._tensor2idx_cell(outputs)
        
        # Should fallback to main dictionary's end_idx
        assert len(indexes) == 1
        assert len(indexes[0]) == 2  # Should stop at end token using fallback
        
    finally:
        # Restore attribute if it existed
        if 'original_end_idx' in locals():
            table_postprocessor.cell_dictionary.end_idx = original_end_idx


@pytest.mark.parametrize("strings,expected_masks", [
    (
        ['<table>,<unknown_tag>,</table>', '<tr>,<td>,missing_closing_tag', '', '<td></td>'],
        [[0, 0, 0], [0, 0, 0], [], [1]]
    ),
])
def test_table_bbox_mask_with_unexpected_tokens(table_postprocessor, strings, expected_masks):
    """Test bbox mask generation with unexpected tokens."""
    masks = table_postprocessor._get_pred_bbox_mask(strings)
    
    assert len(masks) == len(expected_masks)
    for actual, expected in zip(masks, expected_masks):
        assert actual == expected


@pytest.mark.parametrize("strings,expected_masks", [
    (
        ['<td></td>', '<td', '<td>', '</td>', '<th></th>', '<th>', '<table>,<td></td>,<td,</table>'],
        [[1], [1], [0], [0], [0], [0], [0, 1, 1, 0]]
    ),
])
def test_bbox_mask_exact_token_matching(table_postprocessor, strings, expected_masks):
    """Test that only exact tokens '<td></td>' and '<td' generate bbox masks."""
    masks = table_postprocessor._get_pred_bbox_mask(strings)
    
    assert len(masks) == len(expected_masks)
    for actual, expected in zip(masks, expected_masks):
        assert actual == expected


def test_bbox_mask_special_token_handling(table_postprocessor):
    """Test handling of special tokens (EOS, PAD, SOS) in bbox mask generation."""
    # Mock the special tokens based on dictionary
    table_postprocessor.dictionary.idx2str = lambda x: {
        0: ['<BOS>'],
        1: ['<EOS>'],
        2: ['<PAD>']
    }.get(x[0], ['<UKN>'])
    
    strings = [
        '<BOS>,<td></td>,<EOS>',     # EOS should break the loop
        '<PAD>,<td></td>,<td',       # PAD should be skipped  
        '<BOS>,<table>,<td',         # SOS should be skipped
        '<EOS>',                     # EOS at start should break immediately
    ]
    
    masks = table_postprocessor._get_pred_bbox_mask(strings)
    
    assert len(masks) == 4
    assert masks[0] == [0, 1, 0]     # <BOS>(skip), <td></td>(1), <EOS>(break)
    assert masks[1] == [0, 1, 1]     # <PAD>(skip), <td></td>(1), <td(1)
    assert masks[2] == [0, 0, 1]     # <BOS>(skip), <table>(0), <td(1)
    assert masks[3] == [0]           # <EOS>(break) -> should result in [0] and break


@pytest.mark.parametrize("strings,expected_masks", [
    (
        [' <td></td> , <td ', '<td></td>,  ,<td', '\t<td></td>\t,\n<td\n'],
        [[1, 1], [1, 0, 1], [1, 1]]
    ),
])
def test_bbox_mask_whitespace_handling(table_postprocessor, strings, expected_masks):
    """Test whitespace handling in bbox mask generation."""
    masks = table_postprocessor._get_pred_bbox_mask(strings)
    
    assert len(masks) == len(expected_masks)
    for actual, expected in zip(masks, expected_masks):
        assert actual == expected


# Comprehensive Integration Tests
@pytest.mark.parametrize("integration_scenario", [
    {
        'name': 'simple_table',
        'structure_tokens': [
            (0, 0, 4, 10.0),   # '<table>'
            (0, 1, 8, 10.0),   # '<td>'
            (0, 2, 10, 10.0),  # '<td></td>'
            (0, 3, 1, 10.0),   # '<EOS>'
        ],
        'bbox_data': [
            [0.1, 0.1, 0.5, 0.5],
            [0.2, 0.2, 0.6, 0.6],
            [0.3, 0.3, 0.7, 0.7],
            [0.0, 0.0, 0.0, 0.0]
        ],
        'cell_tokens': [
            (0, 0, 4, 10.0),  # 'a'
            (0, 1, 5, 10.0),  # 'b'
            (0, 2, 1, 10.0),  # '<EOS>'
        ],
        'expected_structure_tokens': ['<table>', '<td>', '<td></td>'],
        'expected_bbox_count': 3,
        'expected_cell_content': 'ab'
    },
    {
        'name': 'complex_table',
        'structure_tokens': [
            (0, 0, 4, 10.0),   # '<table>'
            (0, 1, 6, 10.0),   # '<tr>'
            (0, 2, 8, 10.0),   # '<td>'
            (0, 3, 10, 10.0),  # '<td></td>'
            (0, 4, 7, 10.0),   # '</tr>'
            (0, 5, 5, 10.0),   # '</table>'
            (0, 6, 1, 10.0),   # '<EOS>'
        ],
        'bbox_data': [
            [0.0, 0.0, 1.0, 0.3],  # table
            [0.0, 0.1, 1.0, 0.25], # tr
            [0.1, 0.15, 0.4, 0.25], # td
            [0.5, 0.15, 0.9, 0.25], # td></td>
            [0.0, 0.25, 1.0, 0.3],  # /tr
            [0.0, 0.0, 1.0, 0.3],   # /table
            [0.0, 0.0, 0.0, 0.0]    # EOS
        ],
        'cell_tokens': [
            (0, 0, 6, 10.0),  # 'c'
            (0, 1, 7, 10.0),  # 'd'
            (0, 2, 8, 10.0),  # 'e'
            (0, 3, 1, 10.0),  # '<EOS>'
        ],
        'expected_structure_tokens': ['<table>', '<tr>', '<td>', '<td></td>', '</tr>', '</table>'],
        'expected_bbox_count': 6,
        'expected_cell_content': 'cde'
    },
    {
        'name': 'minimal_table',
        'structure_tokens': [
            (0, 0, 10, 10.0),  # '<td></td>'
            (0, 1, 1, 10.0),   # '<EOS>'
        ],
        'bbox_data': [
            [0.2, 0.2, 0.8, 0.8],
            [0.0, 0.0, 0.0, 0.0]
        ],
        'cell_tokens': [
            (0, 0, 1, 10.0),   # immediate '<EOS>' (empty cell)
        ],
        'expected_structure_tokens': ['<td></td>'],
        'expected_bbox_count': 1,
        'expected_cell_content': ''
    },
])
def test_format_table_outputs_comprehensive(postprocessor, integration_scenario):
    """Comprehensive integration test for format_table_outputs with various table scenarios."""
    scenario = integration_scenario
    
    # Prepare structure outputs
    max_struct_len = max([pos for _, pos, _, _ in scenario['structure_tokens']]) + 1
    structure_outputs = torch.full((1, max_struct_len, 15), -10.0)
    for sample, pos, token, value in scenario['structure_tokens']:
        structure_outputs[sample, pos, token] = value
    
    # Prepare bbox outputs
    bbox_tensor = torch.tensor([scenario['bbox_data']])
    
    # Prepare cell outputs
    max_cell_len = max([pos for _, pos, _, _ in scenario['cell_tokens']]) + 1
    cell_output = torch.full((1, max_cell_len, 10), -10.0)
    for sample, pos, token, value in scenario['cell_tokens']:
        cell_output[sample, pos, token] = value
    cell_outputs = [cell_output]
    
    # Prepare data sample
    data_sample = TextRecogDataSample()
    data_sample.set_metainfo({
        'scale_factor': [1.0, 1.0],
        'pad_shape': [100, 100],
        'img_shape': [100, 100]
    })
    
    # Execute the method
    results = postprocessor.format_table_outputs(
        structure_outputs, bbox_tensor, cell_outputs, [data_sample]
    )
    
    # Verify results
    assert len(results) == 1
    result = results[0]
    
    # Check all required keys are present
    required_keys = ['structure_text', 'structure_score', 'bboxes', 'cell_texts', 'cell_scores']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    # Check structure content
    structure_text = result['structure_text']
    for expected_token in scenario['expected_structure_tokens']:
        assert expected_token in structure_text, f"Missing token {expected_token} in structure"
    
    # Check bbox count
    assert len(result['bboxes']) >= scenario['expected_bbox_count'], \
        f"Expected at least {scenario['expected_bbox_count']} bboxes, got {len(result['bboxes'])}"
    
    # Check cell content (if any)
    if scenario['expected_cell_content']:
        assert len(result['cell_texts']) > 0, "Expected cell texts but got none"
        # Check that expected content appears in cell texts
        all_cell_text = ''.join(result['cell_texts'])
        assert scenario['expected_cell_content'] in all_cell_text or \
               any(scenario['expected_cell_content'] in cell for cell in result['cell_texts'])
    
    # Check score validity
    assert isinstance(result['structure_score'], (int, float))
    assert 0.0 <= result['structure_score'] <= 1.0
    
    for cell_score in result['cell_scores']:
        assert isinstance(cell_score, (int, float))
        assert 0.0 <= cell_score <= 1.0


@pytest.mark.parametrize("error_scenario,expected_exception", [
    (
        {
            'name': 'mismatched_dimensions',
            'structure_shape': (2, 3, 15),  # 2 samples
            'bbox_shape': (1, 3, 4),        # 1 sample (mismatch!)
            'cell_outputs_count': 2,
            'data_samples_count': 2
        },
        (RuntimeError, ValueError, IndexError)
    ),
    (
        {
            'name': 'invalid_tensor_dims',
            'structure_shape': (1, 3),      # Missing class dimension
            'bbox_shape': (1, 3, 4),
            'cell_outputs_count': 1,
            'data_samples_count': 1
        },
        (RuntimeError, IndexError)
    ),
])
def test_format_table_outputs_error_handling(postprocessor, error_scenario, expected_exception):
    """Test error handling in format_table_outputs with various invalid inputs."""
    scenario = error_scenario
    
    # Create potentially invalid inputs
    structure_outputs = torch.randn(scenario['structure_shape'])
    bbox_outputs = torch.randn(scenario['bbox_shape'])
    
    cell_outputs = []
    for i in range(scenario['cell_outputs_count']):
        cell_outputs.append(torch.randn(1, 3, 10))
    
    data_samples = []
    for i in range(scenario['data_samples_count']):
        data_sample = TextRecogDataSample()
        data_sample.set_metainfo({
            'scale_factor': [1.0, 1.0],
            'pad_shape': [100, 100],
            'img_shape': [100, 100]
        })
        data_samples.append(data_sample)
    
    # Should raise expected exception
    with pytest.raises(expected_exception):
        postprocessor.format_table_outputs(
            structure_outputs, bbox_outputs, cell_outputs, data_samples
        )