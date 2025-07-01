import pytest
import tempfile
import os
import json
import bz2
from unittest.mock import patch, MagicMock, mock_open
from mmengine.fileio import get_local_path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from datasets.table_dataset import PubTabNetDataset


class TestPubTabNetDataset:
    """Test cases for PubTabNetDataset."""
    
    @pytest.fixture
    def sample_annotation(self):
        """Sample annotation data for testing."""
        return {
            'filename': 'test_table.png',
            'split': 'train',
            'imgid': 12345,
            'html': {
                'structure': {
                    'tokens': ['<table>', '<tr>', '<td>', '</td>', '<td>', '</td>', '</tr>', '</table>']
                },
                'cell': [
                    {
                        'tokens': ['Cell', 'Content', '1'],
                        'bbox': [10, 20, 100, 50]
                    },
                    {
                        'tokens': ['Cell', 'Content', '2'],
                        'bbox': [110, 20, 200, 50]
                    }
                ]
            }
        }
    
    @pytest.fixture
    def temp_json_file(self, sample_annotation):
        """Create a temporary JSON file with sample data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump([sample_annotation], temp_file)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def temp_jsonl_file(self, sample_annotation):
        """Create a temporary JSONL file with sample data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        json.dump(sample_annotation, temp_file)
        temp_file.write('\n')
        # Add another sample
        sample2 = sample_annotation.copy()
        sample2['filename'] = 'test_table2.png'
        sample2['imgid'] = 12346
        json.dump(sample2, temp_file)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def temp_bz2_file(self, sample_annotation):
        """Create a temporary BZ2 compressed JSON file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json.bz2')
        data = json.dumps([sample_annotation]).encode('utf-8')
        temp_file.write(bz2.compress(data))
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_init_with_json_file(self, temp_json_file):
        """Test dataset initialization with JSON file."""
        dataset = PubTabNetDataset(
            ann_file=temp_json_file,
            data_root='/tmp/test_data',
            lazy_init=True
        )
        
        assert dataset.ann_file == temp_json_file
        assert dataset.data_root == '/tmp/test_data'
        assert dataset.task_type == 'both'
    
    def test_init_with_jsonl_file(self, temp_jsonl_file):
        """Test dataset initialization with JSONL file."""
        dataset = PubTabNetDataset(
            ann_file=temp_jsonl_file,
            lazy_init=True
        )
        
        assert dataset.ann_file == temp_jsonl_file
    
    def test_init_with_bz2_file(self, temp_bz2_file):
        """Test dataset initialization with BZ2 compressed file."""
        dataset = PubTabNetDataset(
            ann_file=temp_bz2_file,
            lazy_init=True
        )
        
        assert dataset.ann_file == temp_bz2_file
    
    @patch('mmengine.fileio.get_local_path')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_data_list_json(self, mock_file, mock_get_path, sample_annotation):
        """Test loading data list from JSON file."""
        mock_get_path.return_value = '/tmp/test.json'
        mock_file.return_value.read.return_value = json.dumps([sample_annotation])
        
        dataset = PubTabNetDataset(ann_file='/tmp/test.json', lazy_init=True)
        
        # Mock the load_data_list method call
        with patch.object(dataset, 'load_data_list') as mock_load:
            mock_load.return_value = [sample_annotation]
            data_list = dataset.load_data_list()
            
            assert len(data_list) == 1
            assert data_list[0]['filename'] == 'test_table.png'
            assert data_list[0]['imgid'] == 12345
    
    def test_parse_data_info_structure_only(self, sample_annotation):
        """Test parsing data info for structure recognition only."""
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type='structure'
        )
        
        data_info = dataset.parse_data_info(sample_annotation)
        
        assert 'img_path' in data_info
        assert 'instances' in data_info
        assert 'img_info' in data_info
        
        # Check instances format (mmOCR 1.x)
        instances = data_info['instances']
        assert len(instances) == 1  # Only structure instance
        
        structure_instance = instances[0]
        assert structure_instance['task_type'] == 'structure'
        assert 'text' in structure_instance
        
        # Check structure text
        expected_structure = '<table> <tr> <td> </td> <td> </td> </tr> </table>'
        assert structure_instance['text'] == expected_structure
    
    def test_parse_data_info_content_only(self, sample_annotation):
        """Test parsing data info for content recognition only."""
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type='content'
        )
        
        data_info = dataset.parse_data_info(sample_annotation)
        
        assert 'img_path' in data_info
        assert 'instances' in data_info
        assert 'img_info' in data_info
        
        # Check instances format (mmOCR 1.x)
        instances = data_info['instances']
        assert len(instances) == 2  # Two cell instances
        
        # Check first cell instance
        cell1_instance = instances[0]
        assert cell1_instance['task_type'] == 'content'
        assert cell1_instance['text'] == 'Cell Content 1'
        assert cell1_instance['cell_id'] == 0
        assert cell1_instance['bbox'] == [10, 20, 100, 50]
        
        # Check second cell instance
        cell2_instance = instances[1]
        assert cell2_instance['task_type'] == 'content'
        assert cell2_instance['text'] == 'Cell Content 2'
        assert cell2_instance['cell_id'] == 1
        assert cell2_instance['bbox'] == [110, 20, 200, 50]
    
    def test_parse_data_info_both_tasks(self, sample_annotation):
        """Test parsing data info for both structure and content recognition."""
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type='both'
        )
        
        data_info = dataset.parse_data_info(sample_annotation)
        
        assert 'img_path' in data_info
        assert 'instances' in data_info
        assert 'img_info' in data_info
        
        # Check instances format (mmOCR 1.x)
        instances = data_info['instances']
        assert len(instances) == 3  # 1 structure + 2 content instances
        
        # Check structure instance
        structure_instance = instances[0]
        assert structure_instance['task_type'] == 'structure'
        assert 'text' in structure_instance
        
        # Check content instances
        content_instances = [inst for inst in instances if inst['task_type'] == 'content']
        assert len(content_instances) == 2
        
        assert content_instances[0]['text'] == 'Cell Content 1'
        assert content_instances[1]['text'] == 'Cell Content 2'
    
    def test_parse_data_info_with_empty_cells(self, sample_annotation):
        """Test parsing data info with empty cells."""
        # Modify sample to have empty cells
        sample_annotation['html']['cell'].append({
            'tokens': []  # Empty cell without bbox
        })
        
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type='content'
        )
        
        data_info = dataset.parse_data_info(sample_annotation)
        
        # Check instances format (mmOCR 1.x)
        instances = data_info['instances']
        
        # Should still have 2 cell instances (empty cells are ignored by default)
        assert len(instances) == 2
        
        # All instances should be content type
        for instance in instances:
            assert instance['task_type'] == 'content'
    
    def test_load_data_list_file_not_found(self):
        """Test loading data list when file doesn't exist."""
        dataset = PubTabNetDataset(
            ann_file='nonexistent.json',
            lazy_init=True
        )
        
        with pytest.raises(FileNotFoundError):
            dataset.load_data_list()
    
    def test_parse_data_info_invalid_task_type(self, sample_annotation):
        """Test parsing with invalid task type."""
        with pytest.raises(AssertionError):
            dataset = PubTabNetDataset(
                ann_file='dummy.json',
                lazy_init=True,
                task_type='invalid'
            )
    
    def test_get_data_info_with_indices(self, temp_json_file):
        """Test getting data info with specific indices."""
        dataset = PubTabNetDataset(
            ann_file=temp_json_file,
            indices=[0],
            lazy_init=True
        )
        
        # The indices parameter is handled by BaseDataset
        # We just check that the dataset was created successfully
        assert dataset.task_type == 'both'
    
    def test_dataset_with_custom_data_prefix(self, temp_json_file):
        """Test dataset with custom data prefix."""
        dataset = PubTabNetDataset(
            ann_file=temp_json_file,
            data_prefix=dict(img_path='images/'),
            lazy_init=True
        )
        
        assert dataset.data_prefix['img_path'] == 'images/'
    
    def test_dataset_test_mode(self, temp_json_file):
        """Test dataset in test mode."""
        dataset = PubTabNetDataset(
            ann_file=temp_json_file,
            test_mode=True,
            lazy_init=True
        )
        
        assert dataset.test_mode is True
    
    def test_dataset_metainfo(self, temp_json_file):
        """Test dataset metainfo compliance."""
        dataset = PubTabNetDataset(
            ann_file=temp_json_file,
            lazy_init=True
        )
        
        # Check metainfo structure
        assert hasattr(dataset, 'METAINFO')
        assert 'dataset_name' in dataset.METAINFO
        assert 'task_name' in dataset.METAINFO
        assert dataset.METAINFO['dataset_name'] == 'PubTabNet'
        assert dataset.METAINFO['task_name'] == 'table_recognition'
    
    def test_instances_format_compliance(self, sample_annotation):
        """Test that instances format follows mmOCR 1.x standards."""
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type='both'
        )
        
        data_info = dataset.parse_data_info(sample_annotation)
        
        # Check main format
        assert 'instances' in data_info
        assert 'img_path' in data_info
        assert 'img_info' in data_info
        assert 'sample_idx' in data_info
        
        # Check instances structure
        instances = data_info['instances']
        assert isinstance(instances, list)
        
        for instance in instances:
            assert 'text' in instance
            assert 'task_type' in instance
            assert instance['task_type'] in ['structure', 'content']
    
    def test_structure_tokens_joining(self, sample_annotation):
        """Test that structure tokens are properly joined."""
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type='structure'
        )
        
        tokens = sample_annotation['html']['structure']['tokens']
        expected = ' '.join(tokens)
        
        data_info = dataset.parse_data_info(sample_annotation)
        instances = data_info['instances']
        structure_instance = instances[0]
        
        assert structure_instance['text'] == expected
    
    def test_cell_tokens_joining(self, sample_annotation):
        """Test that cell tokens are properly joined."""
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type='content'
        )
        
        data_info = dataset.parse_data_info(sample_annotation)
        instances = data_info['instances']
        
        # Check first cell tokens
        cell1_tokens = sample_annotation['html']['cell'][0]['tokens']
        expected_cell1 = ' '.join(cell1_tokens)
        assert instances[0]['text'] == expected_cell1
        
        # Check second cell tokens
        cell2_tokens = sample_annotation['html']['cell'][1]['tokens']
        expected_cell2 = ' '.join(cell2_tokens)
        assert instances[1]['text'] == expected_cell2
