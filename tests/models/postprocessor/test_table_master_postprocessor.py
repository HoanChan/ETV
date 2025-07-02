import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from mmocr.structures import TextRecogDataSample
from models.postprocessors.table_master_postprocessor import TableMasterPostprocessor


class TestTableMasterPostprocessor:
    """Test cases for TableMasterPostprocessor class."""

    @pytest.fixture
    def mock_dictionary(self):
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
    def mock_cell_dictionary(self):
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
    def dictionary_config(self):
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
    def cell_dictionary_config(self):
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
    def postprocessor(self, dictionary_config, cell_dictionary_config, mock_dictionary, mock_cell_dictionary):
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

    def test_init_valid_config(self, dictionary_config, cell_dictionary_config):
        """Test TableMasterPostprocessor initialization with valid config."""
        with patch('models.postprocessors.table_master_postprocessor.BaseTextRecogPostprocessor.__init__') as mock_init, \
             patch('models.postprocessors.table_master_postprocessor.MODELS.build') as mock_build:
            
            processor = TableMasterPostprocessor(
                dictionary=dictionary_config,
                cell_dictionary=cell_dictionary_config,
                max_seq_len=600,
                max_seq_len_cell=120,
                start_end_same=False
            )
            
            # Check that parent __init__ was called with correct arguments
            mock_init.assert_called_once_with(
                dictionary=dictionary_config,
                max_seq_len=600
            )
            
            # Check that cell dictionary was built
            mock_build.assert_called_once_with(cell_dictionary_config)
            
            assert processor.max_seq_len_cell == 120
            assert processor.start_end_same == False

    def test_init_invalid_start_end_same(self, dictionary_config, cell_dictionary_config):
        """Test TableMasterPostprocessor initialization with invalid start_end_same."""
        with patch('models.postprocessors.table_master_postprocessor.BaseTextRecogPostprocessor.__init__'), \
             patch('models.postprocessors.table_master_postprocessor.MODELS.build'):
            
            with pytest.raises(AssertionError, match="TableMaster requires start_end_same=False"):
                TableMasterPostprocessor(
                    dictionary=dictionary_config,
                    cell_dictionary=cell_dictionary_config,
                    start_end_same=True
                )

    def test_init_default_values(self, dictionary_config, cell_dictionary_config):
        """Test TableMasterPostprocessor initialization with default values."""
        with patch('models.postprocessors.table_master_postprocessor.BaseTextRecogPostprocessor.__init__'), \
             patch('models.postprocessors.table_master_postprocessor.MODELS.build'):
            
            processor = TableMasterPostprocessor(
                dictionary=dictionary_config,
                cell_dictionary=cell_dictionary_config
            )
            
            assert processor.max_seq_len_cell == 100
            assert processor.start_end_same == False

    def test_get_single_prediction(self, postprocessor):
        """Test get_single_prediction method."""
        # Create a 2D tensor (T, C)
        probs = torch.full((3, 15), -10.0)  # Low baseline
        probs[0, 4] = 10.0  # '<table>'
        probs[1, 8] = 10.0  # '<td>'
        probs[2, 1] = 10.0  # '<EOS>'
        
        char_indexes, char_scores = postprocessor.get_single_prediction(probs)
        
        assert char_indexes == [4, 8]  # '<table>', '<td>'
        assert len(char_scores) == 2
        assert all(score > 0.99 for score in char_scores)

    def test_tensor2idx_batch_processing(self, postprocessor):
        """Test _tensor2idx with batch processing."""
        # Create a 3D tensor (N, T, C) where N=2, T=3, C=15
        outputs = torch.full((2, 3, 15), -10.0)  # Low baseline
        
        # First sample
        outputs[0, 0, 4] = 10.0  # '<table>'
        outputs[0, 1, 8] = 10.0  # '<td>'
        outputs[0, 2, 1] = 10.0  # '<EOS>'
        
        # Second sample
        outputs[1, 0, 6] = 10.0  # '<tr>'
        outputs[1, 1, 9] = 10.0  # '</td>'
        outputs[1, 2, 1] = 10.0  # '<EOS>'
        
        indexes, scores = postprocessor._tensor2idx(outputs)
        
        assert len(indexes) == 2
        assert len(scores) == 2
        assert indexes[0] == [4, 8]  # '<table>', '<td>'
        assert indexes[1] == [6, 9]  # '<tr>', '</td>'
        assert all(len(score_list) == len(index_list) for score_list, index_list in zip(scores, indexes))

    def test_tensor2idx_with_ignore_indexes(self, postprocessor):
        """Test _tensor2idx ignores specified indexes."""
        outputs = torch.full((1, 5, 15), -10.0)  # Low baseline
        outputs[0, 0, 0] = 10.0  # '<BOS>' - should be ignored
        outputs[0, 1, 4] = 10.0  # '<table>'
        outputs[0, 2, 2] = 10.0  # '<PAD>' - should be ignored
        outputs[0, 3, 8] = 10.0  # '<td>'
        outputs[0, 4, 3] = 10.0  # '<UKN>' - should be ignored
        
        indexes, scores = postprocessor._tensor2idx(outputs)
        
        assert indexes[0] == [4, 8]  # Only '<table>', '<td>'
        assert len(scores[0]) == 2

    def test_tensor2idx_cell_processing(self, postprocessor):
        """Test _tensor2idx_cell for cell content processing."""
        # Create cell content tensor
        outputs = torch.full((2, 3, 10), -10.0)  # Low baseline
        
        # First cell
        outputs[0, 0, 4] = 10.0  # 'a'
        outputs[0, 1, 5] = 10.0  # 'b'
        outputs[0, 2, 1] = 10.0  # '<EOS>'
        
        # Second cell
        outputs[1, 0, 6] = 10.0  # 'c'
        outputs[1, 1, 7] = 10.0  # 'd'
        outputs[1, 2, 1] = 10.0  # '<EOS>'
        
        indexes, scores = postprocessor._tensor2idx_cell(outputs)
        
        assert len(indexes) == 2
        assert indexes[0] == [4, 5]  # 'a', 'b'
        assert indexes[1] == [6, 7]  # 'c', 'd'
        assert all(len(score_list) == len(index_list) for score_list, index_list in zip(scores, indexes))

    def test_get_pred_bbox_mask(self, postprocessor):
        """Test _get_pred_bbox_mask generation."""
        # Use strings that match the logic in implementation
        strings = [
            '<table>,<td></td>,</table>',  # <td></td> should be 1
            '<tr>,<td,</tr>'               # <td should be 1 (without closing >)
        ]
        
        masks = postprocessor._get_pred_bbox_mask(strings)
        
        assert len(masks) == 2  # Two strings
        # Check that <td></td> and <td tokens get mask value 1
        # Other tokens should get mask value 0
        assert masks[0].tolist() == [0, 1, 0]  # Only <td></td> should be 1
        assert masks[1].tolist() == [0, 1, 0]  # <td should be 1

    def test_filter_invalid_bbox(self, postprocessor):
        """Test _filter_invalid_bbox method."""
        # Create bbox data with some invalid coordinates
        output_bbox = np.array([
            [0.1, 0.1, 0.5, 0.5],  # Valid bbox
            [-0.1, 0.2, 0.6, 0.7], # Invalid (negative coordinate)
            [0.2, 0.3, 1.2, 0.8],  # Invalid (> 1.0)
            [0.3, 0.4, 0.7, 0.9]   # Valid bbox
        ])
        
        pred_bbox_mask = np.array([1, 1, 0, 1])  # Mask indicating which bboxes should be kept
        
        filtered_bbox = postprocessor._filter_invalid_bbox(output_bbox, pred_bbox_mask)
        
        # Check that invalid bboxes are zeroed out
        assert filtered_bbox.shape == output_bbox.shape
        
        # First bbox should be kept (valid coordinates and mask=1)
        np.testing.assert_array_almost_equal(filtered_bbox[0], [0.1, 0.1, 0.5, 0.5])
        
        # Second bbox should be zeroed (invalid coordinates)
        np.testing.assert_array_almost_equal(filtered_bbox[1], [0.0, 0.0, 0.0, 0.0])
        
        # Third bbox should be zeroed (mask=0)
        np.testing.assert_array_almost_equal(filtered_bbox[2], [0.0, 0.0, 0.0, 0.0])
        
        # Fourth bbox should be kept (valid coordinates and mask=1)
        np.testing.assert_array_almost_equal(filtered_bbox[3], [0.3, 0.4, 0.7, 0.9])

    def test_decode_bboxes(self, postprocessor):
        """Test _decode_bboxes method."""
        # Create normalized bbox outputs
        outputs_bbox = torch.tensor([
            [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]],
            [[0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.8, 0.8]]
        ])
        
        pred_bbox_masks = np.array([
            [1, 1],  # Both bboxes valid for first sample
            [1, 0]   # Only first bbox valid for second sample
        ])
        
        # Create mock data samples
        data_samples = []
        for i in range(2):
            data_sample = TextRecogDataSample()
            data_sample.set_metainfo({
                'scale_factor': [2.0, 2.0],
                'pad_shape': [100, 200],  # H, W
                'img_shape': [80, 160]
            })
            data_samples.append(data_sample)
        
        pred_bboxes = postprocessor._decode_bboxes(outputs_bbox, pred_bbox_masks, data_samples)
        
        assert len(pred_bboxes) == 2
        
        # Check first sample - both bboxes should be denormalized and scaled
        expected_bbox_0 = np.array([
            [0.1 * 200 / 2.0, 0.1 * 100 / 2.0, 0.5 * 200 / 2.0, 0.5 * 100 / 2.0],
            [0.2 * 200 / 2.0, 0.2 * 100 / 2.0, 0.6 * 200 / 2.0, 0.6 * 100 / 2.0]
        ])
        np.testing.assert_array_almost_equal(pred_bboxes[0][:2], expected_bbox_0, decimal=4)

    def test_adjust_bboxes_len(self, postprocessor):
        """Test _adjust_bboxes_len method."""
        bboxes = [
            np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]]),
            np.array([[0.4, 0.4, 0.8, 0.8], [0.5, 0.5, 0.9, 0.9]])
        ]
        
        strings = [
            '<table>,<tr>',  # 2 tokens, should keep first 2 bboxes
            '<td></td>'      # 1 token, should keep first 1 bbox
        ]
        
        adjusted_bboxes = postprocessor._adjust_bboxes_len(bboxes, strings)
        
        assert len(adjusted_bboxes) == 2
        assert adjusted_bboxes[0].shape == (2, 4)  # First 2 bboxes kept
        assert adjusted_bboxes[1].shape == (1, 4)  # First 1 bbox kept
        
        np.testing.assert_array_equal(adjusted_bboxes[0], bboxes[0][:2])
        np.testing.assert_array_equal(adjusted_bboxes[1], bboxes[1][:1])

    def test_get_avg_scores(self, postprocessor):
        """Test _get_avg_scores method."""
        str_scores = [
            [0.9, 0.8, 0.7],  # Average = 0.8
            [0.95, 0.85],     # Average = 0.9
            [],               # Empty list, should return 0.0
            [1.0]             # Single value, should return 1.0
        ]
        
        avg_scores = postprocessor._get_avg_scores(str_scores)
        
        assert len(avg_scores) == 4
        assert abs(avg_scores[0] - 0.8) < 1e-6
        assert abs(avg_scores[1] - 0.9) < 1e-6
        assert avg_scores[2] == 0.0
        assert avg_scores[3] == 1.0

    def test_format_table_outputs_integration(self, postprocessor):
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

    def test_empty_cell_content_handling(self, postprocessor):
        """Test handling of empty cell content."""
        # Create a cell output with only 1 element (empty case)
        cell_output = torch.ones((1, 1, 10))
        
        # This should be handled as empty cell
        cell_indexes, cell_scores = postprocessor._tensor2idx_cell(cell_output)
        
        # For empty cells, should return appropriate empty structure
        assert len(cell_indexes) == 1  # One sample
        assert len(cell_scores) == 1

    def test_cell_dictionary_fallback(self, postprocessor):
        """Test fallback to main dictionary for cell processing."""
        # Remove some attributes from cell dictionary to test fallback
        original_end_idx = postprocessor.cell_dictionary.end_idx
        del postprocessor.cell_dictionary.end_idx
        
        outputs = torch.full((1, 2, 10), -10.0)
        outputs[0, 0, 4] = 10.0
        outputs[0, 1, 1] = 10.0  # Should use main dictionary's end_idx
        
        try:
            indexes, scores = postprocessor._tensor2idx_cell(outputs)
            assert len(indexes) == 1
            assert len(indexes[0]) == 1  # Should stop at end token
        finally:
            # Restore the attribute
            postprocessor.cell_dictionary.end_idx = original_end_idx
