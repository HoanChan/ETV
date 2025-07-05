#!/usr/bin/env python3
"""Test suite for LoadTokens transform."""

import pytest
import numpy as np
from datasets.transforms.load_tokens import LoadTokens


@pytest.fixture
def sample_data():
    """Sample data mimicking PubTabNetDataset output."""
    return {
        'img_path': 'test_image.jpg',
        'sample_idx': 0,
        'instances': [
            {
                'tokens': ['<thead>', '<tr>', '<td>', 'Cell 1', '</td>', '<td>', 'Cell 2', '</td>', '</tr>', '</thead>'],
                'task_type': 'structure'
            },
            {
                'tokens': ['Hello', 'World'],
                'task_type': 'content',
                'cell_id': 0,
                'bbox': [10, 20, 100, 50]
            },
            {
                'tokens': ['Test', 'Cell'],
                'task_type': 'content', 
                'cell_id': 1,
                'bbox': [110, 20, 200, 50]
            }
        ]
    }


@pytest.fixture
def empty_data():
    """Empty data for testing edge cases."""
    return {'img_path': 'test.jpg', 'instances': []}

# Basic initialization tests
@pytest.mark.parametrize("with_structure,with_content,should_fail", [
    (True, True, False), (True, False, False), (False, True, False), (False, False, True)
])
def test_initialization(with_structure, with_content, should_fail):
    """Test initialization parameter validation."""
    if should_fail:
        with pytest.raises(AssertionError):
            LoadTokens(with_structure=with_structure, with_content=with_content)
    else:
        transform = LoadTokens(with_structure=with_structure, with_content=with_content)
        assert transform.with_structure == with_structure
        assert transform.with_content == with_content


def test_load_structure_only(sample_data):
    """Test loading structure tokens only."""
    transform = LoadTokens(with_structure=True, with_content=False)
    result = transform(sample_data.copy())
    
    assert 'gt_structure_tokens' in result
    expected = ['<thead>', '<tr>', '<td>', 'Cell 1', '</td>', '<td>', 'Cell 2', '</td>', '</tr>', '</thead>']
    assert result['gt_structure_tokens'] == expected
    assert 'gt_cell_tokens' not in result
    assert result['gt_tokens'] == expected


def test_load_content_only(sample_data):
    """Test loading content tokens only."""
    transform = LoadTokens(with_structure=False, with_content=True)
    result = transform(sample_data.copy())
    
    assert 'gt_cell_tokens' in result
    assert len(result['gt_cell_tokens']) == 2
    assert result['gt_cell_tokens'][0] == ['Hello', 'World']
    assert result['gt_cell_tokens'][1] == ['Test', 'Cell']
    assert result['gt_cell_ids'] == [0, 1]
    assert 'gt_structure_tokens' not in result


def test_load_both_tokens(sample_data):
    """Test loading both structure and content tokens."""
    transform = LoadTokens(with_structure=True, with_content=True)
    result = transform(sample_data.copy())
    
    expected_structure = ['<thead>', '<tr>', '<td>', 'Cell 1', '</td>', '<td>', 'Cell 2', '</td>', '</tr>', '</thead>']
    assert result['gt_structure_tokens'] == expected_structure
    assert len(result['gt_cell_tokens']) == 2
    assert result['gt_cell_ids'] == [0, 1]
    assert isinstance(result['gt_tokens'], dict)
    assert 'structure' in result['gt_tokens']
    assert 'content' in result['gt_tokens']


def test_bbox_handling(sample_data):
    """Test bounding box loading."""
    transform = LoadTokens(with_structure=False, with_content=True, with_bbox=True)
    result = transform(sample_data.copy())
    
    assert 'gt_cell_bboxes' in result
    assert isinstance(result['gt_cell_bboxes'], np.ndarray)
    assert result['gt_cell_bboxes'].shape == (2, 4)
    expected_bboxes = np.array([[10, 20, 100, 50], [110, 20, 200, 50]], dtype=np.float32)
    np.testing.assert_array_equal(result['gt_cell_bboxes'], expected_bboxes)


@pytest.mark.parametrize("max_len,expected_len", [(None, 10), (5, 5), (15, 10), (0, 0)])
def test_structure_length_limiting(sample_data, max_len, expected_len):
    """Test structure token length limiting."""
    transform = LoadTokens(with_structure=True, with_content=False, max_structure_len=max_len)
    result = transform(sample_data.copy())
    
    assert len(result['gt_structure_tokens']) == expected_len


@pytest.mark.parametrize("max_len,expected_lens", [(None, [2, 2]), (1, [1, 1]), (0, [0, 0])])
def test_cell_length_limiting(sample_data, max_len, expected_lens):
    """Test cell token length limiting."""
    transform = LoadTokens(with_structure=False, with_content=True, max_cell_len=max_len)
    result = transform(sample_data.copy())
    
    for i, expected_len in enumerate(expected_lens):
        assert len(result['gt_cell_tokens'][i]) == expected_len


def test_token_flattening(sample_data):
    """Test token flattening functionality."""
    transform = LoadTokens(with_structure=True, with_content=True, flatten_tokens=True)
    result = transform(sample_data.copy())
    
    assert isinstance(result['gt_tokens'], list)
    expected = (
        ['<thead>', '<tr>', '<td>', 'Cell 1', '</td>', '<td>', 'Cell 2', '</td>', '</tr>', '</thead>'] +
        ['Hello', 'World'] + ['Test', 'Cell']
    )
    assert result['gt_tokens'] == expected


def test_empty_instances(empty_data):
    """Test handling of empty instances list."""
    transform = LoadTokens(with_structure=True, with_content=True, with_bbox=True)
    result = transform(empty_data.copy())
    
    assert result['gt_structure_tokens'] == []
    assert result['gt_cell_tokens'] == []
    assert result['gt_cell_ids'] == []
    assert result['gt_task_types'] == []
    assert 'gt_cell_bboxes' not in result


def test_missing_instances():
    """Test handling when instances key is missing."""
    data = {'img_path': 'test.jpg', 'sample_idx': 0}
    transform = LoadTokens(with_structure=True, with_content=True, with_bbox=True)
    result = transform(data.copy())
    
    assert result['gt_structure_tokens'] == []
    assert result['gt_cell_tokens'] == []
    assert result['gt_cell_ids'] == []
    assert 'gt_cell_bboxes' in result
    assert result['gt_cell_bboxes'].shape == (0, 4)