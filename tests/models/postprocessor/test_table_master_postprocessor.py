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
        assert masks[0] == [0, 1, 0]  # Only <td></td> should be 1
        assert masks[1] == [0, 1, 0]  # <td should be 1

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
        
        pred_bbox_masks = [
            [1, 1],  # Both bboxes valid for first sample
            [1, 0]   # Only first bbox valid for second sample
        ]
        
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


    @pytest.fixture
    def table_mock_dictionary(self):
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
    def table_postprocessor(self, table_mock_dictionary, mock_dictionary):
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


    def test_table_invalid_bbox_filter_edge_cases(self, table_postprocessor):
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

    def test_table_mismatched_cell_outputs_length(self, table_postprocessor):
        """Test handling of mismatched cell_outputs length."""
        # Create structure for 2 samples but only 1 cell output
        structure_outputs = torch.full((2, 3, 20), -10.0)
        structure_outputs[0, 0, 4] = 10.0  # '<table>'
        structure_outputs[0, 1, 8] = 10.0  # '<td>'
        structure_outputs[0, 2, 1] = 10.0  # '<EOS>'
        
        structure_outputs[1, 0, 6] = 10.0  # '<tr>'
        structure_outputs[1, 1, 9] = 10.0  # '</td>'
        structure_outputs[1, 2, 1] = 10.0  # '<EOS>'
        
        bbox_outputs = torch.zeros((2, 3, 4))
        cell_outputs = [torch.full((1, 3, 10), -10.0)]  # Only 1 output for 2 samples!
        
        data_samples = []
        for i in range(2):
            data_sample = TextRecogDataSample()
            data_sample.set_metainfo({
                'scale_factor': [1.0, 1.0],
                'pad_shape': [100, 100],
                'img_shape': [100, 100]
            })
            data_samples.append(data_sample)
        
        # This should either handle gracefully or raise a clear error
        with pytest.raises((IndexError, ValueError)):
            table_postprocessor.format_table_outputs(
                structure_outputs, bbox_outputs, cell_outputs, data_samples
            )

    def test_table_missing_metadata_in_data_sample(self, table_postprocessor):
        """Test handling of missing metadata in data samples."""
        structure_outputs = torch.full((1, 2, 20), -10.0)
        structure_outputs[0, 0, 4] = 10.0  # '<table>'
        structure_outputs[0, 1, 1] = 10.0  # '<EOS>'
        
        bbox_outputs = torch.zeros((1, 2, 4))
        cell_outputs = [torch.full((1, 2, 10), -10.0)]
        
        # Data sample with missing metadata
        data_sample = TextRecogDataSample()
        # No metainfo set - should use defaults
        
        try:
            results = table_postprocessor.format_table_outputs(
                structure_outputs, bbox_outputs, cell_outputs, [data_sample]
            )
            # Should handle missing metadata gracefully
            assert len(results) == 1
        except (KeyError, AttributeError) as e:
            # If it fails, it should give a clear error message
            assert "metainfo" in str(e) or "scale_factor" in str(e)

    def test_table_cell_dictionary_attribute_missing(self, table_postprocessor):
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

    def test_table_bbox_mask_with_unexpected_tokens(self, table_postprocessor):
        """Test bbox mask generation with unexpected tokens."""
        strings = [
            '<table>,<unknown_tag>,</table>',  # Unknown tag
            '<tr>,<td>,missing_closing_tag',   # <td> with > should NOT get mask 1 (only <td without > does)
            '',                                 # Empty string
            '<td></td>'                        # Single token
        ]
        
        masks = table_postprocessor._get_pred_bbox_mask(strings)
        
        assert len(masks) == 4
        # Unknown tags should get mask 0
        assert masks[0] == [0, 0, 0]
        # <td> with closing > should get mask 0 (only <td without > gets mask 1)
        assert masks[1] == [0, 0, 0]
        # Empty string should result in empty mask
        assert len(masks[2]) == 0
        # Single <td></td> should get mask 1
        assert masks[3] == [1]

    def test_bbox_mask_exact_token_matching(self, table_postprocessor):
        """Test that only exact tokens '<td></td>' and '<td' generate bbox masks."""
        strings = [
            '<td></td>',         # Should be 1 - exact match
            '<td',               # Should be 1 - exact match (no closing >)
            '<td>',              # Should be 0 - has closing >
            '</td>',             # Should be 0 - closing tag  
            '<th></th>',         # Should be 0 - not supported per comment
            '<th>',              # Should be 0 - not supported
            '<table>,<td></td>,<td,</table>',  # Mixed case
        ]
        
        masks = table_postprocessor._get_pred_bbox_mask(strings)
        
        assert len(masks) == 7
        assert masks[0] == [1]           # <td></td> -> 1
        assert masks[1] == [1]           # <td -> 1
        assert masks[2] == [0]           # <td> -> 0
        assert masks[3] == [0]           # </td> -> 0
        assert masks[4] == [0]           # <th></th> -> 0
        assert masks[5] == [0]           # <th> -> 0
        assert masks[6] == [0, 1, 1, 0]  # <table>,<td></td>,<td,</table> -> [0,1,1,0]

    def test_bbox_mask_special_token_handling(self, table_postprocessor):
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

    def test_bbox_mask_whitespace_handling(self, table_postprocessor):
        """Test whitespace handling in bbox mask generation."""
        strings = [
            ' <td></td> , <td ',          # Spaces around tokens
            '<td></td>,  ,<td',           # Empty token after split
            '\t<td></td>\t,\n<td\n',      # Tabs and newlines
        ]
        
        masks = table_postprocessor._get_pred_bbox_mask(strings)
        
        assert len(masks) == 3
        assert masks[0] == [1, 1]        # Spaces should be stripped
        assert masks[1] == [1, 0, 1]     # Empty token should get 0
        assert masks[2] == [1, 1]        # Tabs/newlines should be stripped