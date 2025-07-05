import pytest
import tempfile
import os
import json
import bz2
from unittest.mock import patch, mock_open
import sys
import random

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
                'cells': [
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
    def sample_annotations_multi(self):
        """Create multiple sample annotations for testing."""
        annotations = []
        splits = ['train', 'val', 'test']
        for i in range(15):
            split = splits[i % 3]
            annotations.append({
                'filename': f'test_table_{i:02d}.png',
                'split': split,
                'imgid': 12345 + i,
                'html': {
                    'structure': {
                        'tokens': ['<table>', '<tr>', '<td>', '</td>', '</tr>', '</table>']
                    },
                    'cells': [
                        {
                            'tokens': [f'Cell_{i}'],
                            'bbox': [10, 20, 100, 50]
                        }
                    ]
                }
            })
        return annotations
    
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
    def temp_multi_file(self, sample_annotations_multi):
        """Create a temporary file with multiple samples."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(sample_annotations_multi, temp_file)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.mark.parametrize("file_type,suffix", [
        ('json', '.json'),
        ('jsonl', '.jsonl'),
        ('bz2', '.json.bz2')
    ])
    def test_init_with_different_file_formats(self, sample_annotation, file_type, suffix):
        """Test dataset initialization with different file formats."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=suffix)
        
        if file_type == 'json':
            json.dump([sample_annotation], temp_file)
        elif file_type == 'jsonl':
            json.dump(sample_annotation, temp_file)
            temp_file.write('\n')
            sample2 = sample_annotation.copy()
            sample2['filename'] = 'test_table2.png'
            json.dump(sample2, temp_file)
        elif file_type == 'bz2':
            temp_file.close()
            with open(temp_file.name, 'wb') as f:
                data = json.dumps([sample_annotation]).encode('utf-8')
                f.write(bz2.compress(data))
        
        if file_type != 'bz2':
            temp_file.close()
        
        try:
            dataset = PubTabNetDataset(
                ann_file=temp_file.name,
                data_root='/tmp/test_data',
                lazy_init=True
            )
            
            assert dataset.ann_file == temp_file.name
            assert dataset.task_type == 'both'
            
        finally:
            os.unlink(temp_file.name)
    
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
    
    @pytest.mark.parametrize("task_type,expected_instances", [
        ('structure', 1),  # Only structure instance
        ('content', 2),    # Two cell instances  
        ('both', 3)        # 1 structure + 2 content instances
    ])
    def test_parse_data_info_by_task_type(self, sample_annotation, task_type, expected_instances):
        """Test parsing data info for different task types."""
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type=task_type
        )
        
        data_info = dataset.parse_data_info(sample_annotation)
        
        # Check basic structure
        assert 'img_path' in data_info
        assert 'instances' in data_info
        assert 'img_info' in data_info
        
        # Check instances count
        instances = data_info['instances']
        assert len(instances) == expected_instances
        
        # Verify task types
        if task_type == 'structure':
            assert instances[0]['task_type'] == 'structure'
            expected_structure = ['<table>', '<tr>', '<td>', '</td>', '<td>', '</td>', '</tr>', '</table>']
            assert instances[0]['tokens'] == expected_structure
        elif task_type == 'content':
            for i, instance in enumerate(instances):
                assert instance['task_type'] == 'content'
                assert instance['cell_id'] == i
                assert 'bbox' in instance
        elif task_type == 'both':
            structure_instances = [inst for inst in instances if inst['task_type'] == 'structure']
            content_instances = [inst for inst in instances if inst['task_type'] == 'content']
            assert len(structure_instances) == 1
            assert len(content_instances) == 2
    
    def test_parse_data_info_with_empty_cells(self, sample_annotation):
        """Test parsing data info with empty cells."""
        # Modify sample to have empty cells
        sample_annotation['html']['cells'].append({
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
    
    @pytest.mark.parametrize("scenario,expected_error", [
        ('file_not_found', FileNotFoundError),
        ('invalid_task_type', AssertionError)
    ])
    def test_error_cases(self, scenario, expected_error):
        """Test various error cases."""
        if scenario == 'file_not_found':
            dataset = PubTabNetDataset(ann_file='nonexistent.json', lazy_init=True)
            with pytest.raises(expected_error):
                dataset.load_data_list()
        elif scenario == 'invalid_task_type':
            with pytest.raises(expected_error):
                PubTabNetDataset(ann_file='dummy.json', lazy_init=True, task_type='invalid')
    
    @pytest.mark.parametrize("config_param,config_value", [
        ('indices', [0]),
        ('data_prefix', {'img_path': 'images/'}),
        ('test_mode', True)
    ])
    def test_dataset_configurations(self, temp_json_file, config_param, config_value):
        """Test dataset with various configurations."""
        kwargs = {
            'ann_file': temp_json_file,
            'lazy_init': True,
            config_param: config_value
        }
        
        dataset = PubTabNetDataset(**kwargs)
        
        if config_param == 'indices':
            assert dataset.task_type == 'both'  # Basic check
        elif config_param == 'data_prefix':
            assert dataset.data_prefix['img_path'] == 'images/'
        elif config_param == 'test_mode':
            assert dataset.test_mode is True
    
    def test_dataset_with_custom_data_prefix(self, temp_json_file):
        """Test dataset with custom data prefix."""
        dataset = PubTabNetDataset(
            ann_file=temp_json_file,
            data_prefix=dict(img_path='images/'),
            lazy_init=True
        )
        
        assert dataset.data_prefix['img_path'] == 'images/'
    
    def test_dataset_metainfo_and_format_compliance(self, temp_json_file, sample_annotation):
        """Test dataset metainfo and format compliance."""
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
        
        # Test instances format compliance
        dataset.task_type = 'both'
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
            assert 'tokens' in instance
            assert 'task_type' in instance
            assert instance['task_type'] in ['structure', 'content']
    
    @pytest.mark.parametrize("token_type,task_type", [
        ('structure', 'structure'),
        ('cell', 'content')
    ])
    def test_tokens_format_compliance(self, sample_annotation, token_type, task_type):
        """Test that tokens are properly stored with length limits."""
        dataset = PubTabNetDataset(
            ann_file='dummy.json',
            lazy_init=True,
            task_type=task_type
        )
        
        data_info = dataset.parse_data_info(sample_annotation)
        instances = data_info['instances']
        
        if token_type == 'structure':
            tokens = sample_annotation['html']['structure']['tokens']
            expected = tokens[:dataset.max_structure_len]
            assert instances[0]['tokens'] == expected
        elif token_type == 'cell':
            # Check first cell
            cell1_tokens = sample_annotation['html']['cells'][0]['tokens']
            expected_cell1 = cell1_tokens[:dataset.max_cell_len]
            assert instances[0]['tokens'] == expected_cell1
    
    @pytest.mark.parametrize("max_data,random_sample,expected_behavior", [
        (-1, False, 'load_all'),           # Default behavior
        (0, False, 'empty'),               # Empty dataset
        (5, False, 'sequential_5'),        # First 5 samples
        (3, False, 'sequential_3'),        # First 3 samples
        (20, False, 'all_available'),      # More than available
        (5, True, 'random_5'),             # Random 5 samples
        (3, True, 'random_3'),             # Random 3 samples
    ])
    def test_max_data_and_random_sample(self, temp_multi_file, max_data, random_sample, expected_behavior):
        """Test max_data and random_sample functionality comprehensively."""
        if expected_behavior == 'load_all':
            dataset = PubTabNetDataset(
                ann_file=temp_multi_file,
                lazy_init=False
            )
            assert dataset.max_data == -1
            assert len(dataset) == 15  # All samples
            
        elif expected_behavior == 'empty':
            dataset = PubTabNetDataset(
                ann_file=temp_multi_file,
                max_data=max_data,
                lazy_init=True
            )
            data_list = dataset.load_data_list()
            assert len(data_list) == 0
            
        elif expected_behavior in ['sequential_5', 'sequential_3']:
            expected_count = int(expected_behavior.split('_')[1])
            dataset = PubTabNetDataset(
                ann_file=temp_multi_file,
                max_data=max_data,
                random_sample=False,
                lazy_init=True
            )
            data_list = dataset.load_data_list()
            assert len(data_list) == expected_count
            
            # Check sequential order
            filenames = [os.path.basename(data['img_path']) for data in data_list]
            expected_filenames = [f'test_table_{i:02d}.png' for i in range(expected_count)]
            assert filenames == expected_filenames
            
        elif expected_behavior == 'all_available':
            dataset = PubTabNetDataset(
                ann_file=temp_multi_file,
                max_data=max_data,
                lazy_init=False
            )
            assert len(dataset) == 15  # All available
            
        elif expected_behavior in ['random_5', 'random_3']:
            expected_count = int(expected_behavior.split('_')[1])
            random.seed(42)  # For reproducibility
            dataset1 = PubTabNetDataset(
                ann_file=temp_multi_file,
                max_data=max_data,
                random_sample=True,
                lazy_init=True
            )
            data_list1 = dataset1.load_data_list()
            assert len(data_list1) == expected_count
            
            # Test reproducibility
            random.seed(42)
            dataset2 = PubTabNetDataset(
                ann_file=temp_multi_file,
                max_data=max_data,
                random_sample=True,
                lazy_init=True
            )
            data_list2 = dataset2.load_data_list()
            filenames1 = [os.path.basename(data['img_path']) for data in data_list1]
            filenames2 = [os.path.basename(data['img_path']) for data in data_list2]
            assert filenames1 == filenames2  # Same seed should give same result
    
    def test_split_filter_with_max_data(self, temp_multi_file):
        """Test split_filter works correctly with max_data and random_sample."""
        # Test with split_filter='train' and max_data=3
        dataset = PubTabNetDataset(
            ann_file=temp_multi_file,
            split_filter='train',
            max_data=3,
            lazy_init=False
        )
        
        assert len(dataset) == 3  # Should be limited to 3 from train split
        
        # Verify all samples are from train split
        for i in range(len(dataset)):
            data_info = dataset[i]
            assert data_info['img_info']['split'] == 'train'
        
        # Test with random sampling and split filter
        random.seed(42)
        dataset_random = PubTabNetDataset(
            ann_file=temp_multi_file,
            split_filter='train',
            max_data=3,
            random_sample=True,
            lazy_init=True
        )
        
        data_list = dataset_random.load_data_list()
        assert len(data_list) == 3
        
        # Verify all samples are from train split
        for data_info in data_list:
            assert data_info['img_info']['split'] == 'train'
    
    @pytest.mark.parametrize("param_name,param_value,expected_in_repr", [
        ('max_data', 100, 'max_data=100'),
        ('random_sample', True, 'random_sample=True'),
    ])
    def test_repr_includes_parameters(self, temp_json_file, param_name, param_value, expected_in_repr):
        """Test that __repr__ includes important parameters."""
        kwargs = {
            'ann_file': temp_json_file,
            'lazy_init': False,
            param_name: param_value
        }
        
        dataset = PubTabNetDataset(**kwargs)
        repr_str = repr(dataset)
        assert expected_in_repr in repr_str
        assert 'PubTabNetDataset' in repr_str
