import pytest
import tempfile
import os
import json
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from datasets.cell_dataset import CellDataset


class TestCellDataset:
    """Test cases for CellDataset."""
    
    @pytest.fixture
    def sample_table_annotation(self):
        """Sample table annotation data for testing."""
        return {
            'filename': 'test_table.png',
            'split': 'train',
            'imgid': 12345,
            'html': {
                'structure': {
                    'tokens': ['<table>', '<tr>', '<td>', '</td>', '<td>', '</td>', '</tr>', '</table>']
                },
                'cells': [  # Changed from 'cell' to 'cells'
                    {
                        'tokens': ['Cell', 'Content', '1'],
                        'bbox': [10, 20, 100, 50]
                    },
                    {
                        'tokens': ['Cell', 'Content', '2'],
                        'bbox': [110, 20, 200, 50]
                    },
                    {
                        'tokens': [],  # Empty cell
                        'bbox': [210, 20, 300, 50]
                    }
                ]
            }
        }
    
    @pytest.fixture
    def temp_json_file(self, sample_table_annotation):
        """Create a temporary JSON file with sample table data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump([sample_table_annotation], temp_file)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple test image
        image = np.ones((100, 400, 3), dtype=np.uint8) * 255
        # Add some patterns to make it identifiable (BGR format for cv2)
        image[20:50, 10:100] = [0, 0, 255]  # Red cell 1 (BGR)
        image[20:50, 110:200] = [0, 255, 0]  # Green cell 2 (BGR)
        image[20:50, 210:300] = [255, 0, 0]  # Blue cell 3 (BGR)
        return image
    
    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Create a temporary image file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.close()
        
        cv2.imwrite(temp_file.name, sample_image)
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture(params=[True, False])
    def ignore_empty_cells(self, request):
        """Parameter fixture for ignore_empty_cells option."""
        return request.param
    
    @pytest.fixture(params=[50, 100, 150])
    def max_cell_len(self, request):
        """Parameter fixture for max_cell_len option."""
        return request.param
    
    @pytest.fixture(params=['content', 'structure', 'both'])
    def task_type(self, request):
        """Parameter fixture for task_type option."""
        return request.param
    
    @pytest.fixture
    def mock_parent_load(self):
        """Mock the parent class load_data_list method."""
        with patch('datasets.table_dataset.PubTabNetDataset.load_data_list') as mock:
            yield mock
    
    def test_init_default_parameters(self, temp_json_file):
        """Test dataset initialization with default parameters."""
        dataset = CellDataset(
            ann_file=temp_json_file,
            data_root='/tmp/test_data',
            lazy_init=True
        )
        
        assert dataset.ann_file == temp_json_file
        assert dataset.data_root == '/tmp/test_data'
        assert dataset.task_type == 'content'  # Should be set by CellDataset
        assert dataset.ignore_empty_cells == True
        assert dataset.max_cell_len == 150
    
    def test_init_custom_parameters(self, temp_json_file, ignore_empty_cells, max_cell_len):
        """Test dataset initialization with custom parameters."""
        dataset = CellDataset(
            ann_file=temp_json_file,
            data_root='/tmp/test_data',
            ignore_empty_cells=ignore_empty_cells,
            max_cell_len=max_cell_len,
            lazy_init=True
        )
        
        assert dataset.ignore_empty_cells == ignore_empty_cells
        assert dataset.max_cell_len == max_cell_len
        assert dataset.task_type == 'content'
    
    def test_load_data_with_cells(self, mock_parent_load, sample_table_annotation):
        """Test loading data list creates individual cell samples."""
        # Mock parent class method to return processed data format
        mock_parent_load.return_value = [{
            'img_path': 'test_table.png',
            'sample_idx': 12345,
            'instances': [
                {'text': 'Cell Content 1', 'bbox': [10, 20, 100, 50], 'task_type': 'content', 'cell_id': 0},
                {'text': 'Cell Content 2', 'bbox': [110, 20, 200, 50], 'task_type': 'content', 'cell_id': 1},
                {'text': '', 'bbox': [210, 20, 300, 50], 'task_type': 'content', 'cell_id': 2}  # Empty cell
            ],
            'img_info': {'height': None, 'width': None, 'split': 'train'}
        }]
        
        dataset = CellDataset(
            ann_file='dummy.json',
            ignore_empty_cells=True,
            lazy_init=True
        )
        
        # Should have 2 cells (empty cell ignored)
        assert len(dataset) == 2
        
        # Check first cell
        cell1 = dataset[0]
        assert cell1['img_path'] == 'test_table.png'
        assert cell1['bbox'] == [10, 20, 100, 50]
        assert cell1['sample_idx'] == '12345_0'
        assert cell1['original_imgid'] == 12345
        assert len(cell1['instances']) == 1
        assert cell1['instances'][0]['bbox'] == [10, 20, 100, 50]
        
        # Check second cell
        cell2 = dataset[1]
        assert cell2['img_path'] == 'test_table.png'
        assert cell2['bbox'] == [110, 20, 200, 50]
        assert cell2['sample_idx'] == '12345_1'
        assert cell2['original_imgid'] == 12345
    
    def test_load_data_include_empty_cells(self, mock_parent_load, sample_table_annotation):
        """Test loading data list includes empty cells when ignore_empty_cells=False."""
        mock_parent_load.return_value = [{
            'img_path': 'test_table.png',
            'sample_idx': 12345,
            'instances': [
                {'text': 'Cell Content 1', 'bbox': [10, 20, 100, 50], 'task_type': 'content', 'cell_id': 0},
                {'text': 'Cell Content 2', 'bbox': [110, 20, 200, 50], 'task_type': 'content', 'cell_id': 1},
                {'text': '', 'bbox': [210, 20, 300, 50], 'task_type': 'content', 'cell_id': 2}  # Empty cell
            ],
            'img_info': {'height': None, 'width': None, 'split': 'train'}
        }]
        
        dataset = CellDataset(
            ann_file='dummy.json',
            ignore_empty_cells=False,
            lazy_init=True
        )
        
        # Should have 3 cells (including empty cell)
        assert len(dataset) == 3

        # Check empty cell is included
        cell3 = dataset[2]
        assert cell3['bbox'] == [210, 20, 300, 50]
        assert cell3['sample_idx'] == '12345_2'
    
    def test_load_data_skip_invalid_bbox(self, mock_parent_load):
        """Test loading data list skips cells with invalid bboxes."""
        # Mock parent class method to return processed data format
        mock_parent_load.return_value = [{
            'img_path': 'test_table.png',
            'sample_idx': 12345,
            'instances': [
                {'text': 'Valid Cell', 'bbox': [10, 20, 100, 50], 'task_type': 'content', 'cell_id': 0},
                {'text': 'Invalid Cell', 'bbox': [10, 20, 100], 'task_type': 'content', 'cell_id': 1},  # Invalid bbox
                {'text': 'Another Invalid', 'bbox': [], 'task_type': 'content', 'cell_id': 2}  # Empty bbox
            ],
            'img_info': {'height': None, 'width': None, 'split': 'train'}
        }]
        
        dataset = CellDataset(
            ann_file='dummy.json',
            ignore_empty_cells=False,
            lazy_init=True
        )
        
        # Should only have 1 cell (valid bbox)
        assert len(dataset) == 1
        assert dataset[0]['bbox'] == [10, 20, 100, 50]

    def test_crop_cell_image_success(self, temp_image_file):
        """Test successful cell image cropping."""
        dataset = CellDataset(ann_file='dummy.json', lazy_init=True)
        
        bbox = [10, 20, 100, 50]
        cropped_image = dataset.crop_cell_image(temp_image_file, bbox)
        
        assert cropped_image is not None
        assert cropped_image.shape == (30, 90, 3)  # h=50-20, w=100-10
        
        # Check if the cropped region has the expected color (red in BGR)
        # Allow some tolerance for image compression
        mean_color = np.mean(cropped_image, axis=(0, 1))
        assert mean_color[2] > 200  # Red channel should be high in BGR
    
    def test_crop_cell_image_file_not_found(self):
        """Test cell image cropping with non-existent file."""
        dataset = CellDataset(ann_file='dummy.json', lazy_init=True)
        
        bbox = [10, 20, 100, 50]
        cropped_image = dataset.crop_cell_image('nonexistent.png', bbox)
        
        assert cropped_image is None
    
    @patch('cv2.imread')
    @patch('PIL.Image.open')
    def test_crop_cell_image_fallback_to_pil(self, mock_pil_open, mock_cv2_imread, sample_image):
        """Test cell image cropping falls back to PIL when cv2 fails."""
        # Mock cv2.imread to return None
        mock_cv2_imread.return_value = None
        
        # Mock PIL to return a valid image
        pil_image = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
        mock_pil_open.return_value = pil_image
        
        dataset = CellDataset(ann_file='dummy.json', lazy_init=True)
        
        bbox = [10, 20, 100, 50]
        cropped_image = dataset.crop_cell_image('test.png', bbox)
        
        assert cropped_image is not None
        assert cropped_image.shape == (30, 90, 3)
        
        # Verify PIL was called as fallback
        mock_pil_open.assert_called_once_with('test.png')
    
    @patch('cv2.imread')
    @patch('PIL.Image.open')
    def test_crop_cell_image_both_methods_fail(self, mock_pil_open, mock_cv2_imread):
        """Test cell image cropping when both cv2 and PIL fail."""
        mock_cv2_imread.return_value = None
        mock_pil_open.side_effect = Exception("File not found")
        
        dataset = CellDataset(ann_file='dummy.json', lazy_init=True)
        
        bbox = [10, 20, 100, 50]
        cropped_image = dataset.crop_cell_image('test.png', bbox)
        
        assert cropped_image is None
    
    @patch.object(CellDataset, 'crop_cell_image')
    def test_get_data_info_success(self, mock_crop):
        """Test get_data_info with successful image cropping."""
        # Mock successful cropping
        mock_cell_image = np.ones((30, 90, 3), dtype=np.uint8) * 255
        mock_crop.return_value = mock_cell_image
        
        dataset = CellDataset(ann_file='dummy.json', lazy_init=True)
        dataset.data_list = [{
            'img_path': 'test.png',
            'bbox': [10, 20, 100, 50],
            'sample_idx': '12345_0',
            'instances': [{'text': 'test'}]
        }]
        
        data_info = dataset.get_data_info(0)
        
        assert data_info['height'] == 30
        assert data_info['width'] == 90
        assert 'img' in data_info
        assert np.array_equal(data_info['img'], mock_cell_image)
        
        # Verify crop_cell_image was called correctly
        mock_crop.assert_called_once_with('test.png', [10, 20, 100, 50])
    
    @patch.object(CellDataset, 'crop_cell_image')
    def test_get_data_info_crop_failure(self, mock_crop):
        """Test get_data_info when image cropping fails."""
        # Mock failed cropping
        mock_crop.return_value = None
        
        dataset = CellDataset(ann_file='dummy.json', lazy_init=True)
        dataset.data_list = [{
            'img_path': 'test.png',
            'bbox': [10, 20, 100, 50],
            'sample_idx': '12345_0',
            'instances': [{'text': 'test'}]
        }]
        
        data_info = dataset.get_data_info(0)
        
        # Should return original data_info without height, width, img
        assert 'height' not in data_info or data_info['height'] is None
        assert 'width' not in data_info or data_info['width'] is None
        assert 'img' not in data_info
        assert data_info['img_path'] == 'test.png'
        assert data_info['bbox'] == [10, 20, 100, 50]
    
    def test_repr(self, temp_json_file):
        """Test string representation of the dataset."""
        dataset = CellDataset(
            ann_file=temp_json_file,
            lazy_init=True
        )
        # Add at least one dummy data to avoid serialization error
        dataset.data_list = [{
            'img_path': 'test.png',
            'bbox': [0, 0, 10, 10],
            'sample_idx': '0',
            'instances': [{'text': 'test'}]
        }]
        
        repr_str = repr(dataset)
        
        assert 'CellDataset' in repr_str
        assert 'task_type=content' in repr_str
        assert 'num_samples=' in repr_str  # Don't check exact count
        assert f'ann_file={temp_json_file}' in repr_str
    
    @pytest.mark.parametrize("ignore_empty,expected_count", [
        (True, 2),   # Skip empty cells
        (False, 3),  # Include empty cells
    ])
    def test_parametrized_empty_cell_handling(self, sample_table_annotation, ignore_empty, expected_count):
        """Parametrized test for empty cell handling."""
        with patch('datasets.table_dataset.PubTabNetDataset.load_data_list') as mock_load:
            mock_load.return_value = [{
                'img_path': 'test_table.png',
                'sample_idx': 12345,
                'instances': [
                    {'text': 'Cell Content 1', 'bbox': [10, 20, 100, 50], 'task_type': 'content', 'cell_id': 0},
                    {'text': 'Cell Content 2', 'bbox': [110, 20, 200, 50], 'task_type': 'content', 'cell_id': 1},
                    {'text': '', 'bbox': [210, 20, 300, 50], 'task_type': 'content', 'cell_id': 2}  # Empty cell
                ],
                'img_info': {'height': None, 'width': None, 'split': 'train'}
            }]
            
            dataset = CellDataset(
                ann_file='dummy.json',
                ignore_empty_cells=ignore_empty,
                lazy_init=True
            )
            
            data_list = dataset.load_data_list()
            assert len(data_list) == expected_count
    
    @pytest.mark.parametrize("max_len", [50, 100, 150, 200])
    def test_parametrized_max_cell_len(self, temp_json_file, max_len):
        """Parametrized test for max_cell_len parameter."""
        dataset = CellDataset(
            ann_file=temp_json_file,
            max_cell_len=max_len,
            lazy_init=True
        )
        
        assert dataset.max_cell_len == max_len
    
    @pytest.mark.parametrize("bbox,expected_shape", [
        ([0, 0, 50, 30], (30, 50, 3)),
        ([10, 5, 60, 35], (30, 50, 3)),
        ([0, 0, 100, 50], (50, 100, 3)),
    ])
    def test_parametrized_crop_shapes(self, temp_image_file, bbox, expected_shape):
        """Parametrized test for different crop shapes."""
        dataset = CellDataset(ann_file='dummy.json', lazy_init=True)
        
        cropped_image = dataset.crop_cell_image(temp_image_file, bbox)
        
        assert cropped_image is not None
        assert cropped_image.shape == expected_shape
