import pytest
from datasets.transforms.load_tokens import LoadTokens
from datasets.transforms.transforms_utils import get_bbox_nums

def make_instances(structure_tokens, cell_tokens_list, bboxes):
    instances = []
    if structure_tokens:
        instances.append({
            'tokens': structure_tokens,
            'type': 'structure',
        })
    for i, (tokens, bbox) in enumerate(zip(cell_tokens_list, bboxes)):
        instances.append({
            'tokens': tokens,
            'type': 'content',
            'cell_id': i,
            'bbox': bbox,
        })
    return instances

@pytest.fixture
def sample_data():
    """Basic sample data for testing."""
    return {
        'img': 'mock_image',
        'instances': [
            {
                'tokens': ['<table>', '<tr>', '<td>', '</td>', '<td>', '</td>', '</tr>', '</table>'],
                'type': 'structure'
            },
            {
                'tokens': ['Cell1'],
                'type': 'content',
                'cell_id': 0,
                'bbox': [10, 20, 100, 50]
            },
            {
                'tokens': ['Cell2'],
                'type': 'content', 
                'cell_id': 1,
                'bbox': [110, 20, 200, 50]
            }
        ]
    }

# Core functionality tests
@pytest.mark.parametrize("with_structure,with_cell,should_fail", [
    (True, True, False), 
    (True, False, False), 
    (False, True, False), 
    (False, False, True)
])
def test_initialization(with_structure, with_cell, should_fail):
    """Test initialization parameter validation."""
    if should_fail:
        with pytest.raises(AssertionError):
            LoadTokens(with_structure=with_structure, with_cell=with_cell)
    else:
        transform = LoadTokens(with_structure=with_structure, with_cell=with_cell)
        assert transform.with_structure == with_structure
        assert transform.with_cell == with_cell

def test_structure_only(sample_data):
    """Test loading structure tokens only."""
    transform = LoadTokens(with_structure=True, with_cell=False)
    result = transform(sample_data.copy())
    
    assert 'tokens' in result
    assert 'bboxes' in result
    assert 'masks' in result
    assert 'cells' not in result
    assert len(result['bboxes']) == len(result['tokens'])

def test_cell_only(sample_data):
    """Test loading cell tokens only."""
    transform = LoadTokens(with_structure=False, with_cell=True)
    result = transform(sample_data.copy())
    
    assert 'cells' in result
    assert 'tokens' not in result
    assert 'bboxes' not in result
    assert 'masks' not in result

def test_both_structure_and_cell(sample_data):
    """Test loading both structure and cell tokens."""
    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(sample_data.copy())
    
    assert 'tokens' in result
    assert 'bboxes' in result
    assert 'masks' in result
    assert 'cells' in result
    assert len(result['bboxes']) == len(result['tokens'])

@pytest.mark.parametrize("structure_tokens,cell_tokens_list,bboxes", [
    # Basic case with consecutive <td></td> tokens
    (["<td></td>", "<td></td>"], [["a"], ["b"]], [[1,2,3,4], [5,6,7,8]]),
    # Case with non-consecutive tokens
    (["<tr>", "<td></td>", "<td></td>", "</tr>"], [["a"], ["b"]], [[1,2,3,4], [5,6,7,8]]),
])
def test_token_merge_scenarios(structure_tokens, cell_tokens_list, bboxes):
    """Test different token merge scenarios."""
    data = {'img': 'fake_img', 'instances': make_instances(structure_tokens, cell_tokens_list, bboxes)}
    
    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(data.copy())
    
    assert 'tokens' in result
    assert 'bboxes' in result
    assert 'masks' in result
    assert 'cells' in result
    assert len(result['bboxes']) == len(result['tokens'])
    assert len(result['masks']) == len(result['tokens'])

# Edge cases
@pytest.mark.parametrize("max_structure_len,max_cell_len", [
    (None, None),  # No limits
    (5, 3),        # With limits
    (0, 0),        # Zero limits
])
def test_token_length_limits(sample_data, max_structure_len, max_cell_len):
    """Test token length limiting."""
    transform = LoadTokens(
        with_structure=True, 
        with_cell=True,
        max_structure_token_len=max_structure_len,
        max_cell_token_len=max_cell_len
    )
    result = transform(sample_data.copy())
    
    if max_structure_len is not None:
        assert len(result['tokens']) <= max_structure_len
    if max_cell_len is not None:
        for cell in result['cells']:
            assert len(cell['tokens']) <= max_cell_len

def test_empty_instances():
    """Test handling of empty instances."""
    data = {'img': 'mock_image', 'instances': []}
    
    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(data.copy())
    
    assert result['tokens'] == []
    assert result['cells'] == []

def test_missing_instances():
    """Test handling when instances key is missing."""
    data = {'img': 'mock_image'}
    
    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(data.copy())
    
    assert result['tokens'] == []
    assert result['cells'] == []

def test_duplicate_cell_ids():
    """Test that duplicate cell_ids raise an assertion error."""
    data = {
        'img': 'mock_image',
        'instances': [
            {'tokens': ['<table>', '<tr>', '<td>', '</td>', '</tr>', '</table>'], 'type': 'structure'},
            {'tokens': ['Cell1'], 'type': 'content', 'cell_id': 0, 'bbox': [0, 0, 10, 10]},
            {'tokens': ['Cell2'], 'type': 'content', 'cell_id': 0, 'bbox': [10, 0, 20, 10]}  # Duplicate
        ]
    }
    
    transform = LoadTokens(with_structure=True, with_cell=True)
    
    with pytest.raises(AssertionError, match="cell_id must be unique"):
        transform(data)

def test_cell_id_boundary():
    """Test cell mapping with bbox_count boundary."""
    data = {
        'img': 'mock_image',
        'instances': [
            {'tokens': ['<table>', '<tr>', '<td>', '</td>', '</tr>', '</table>'], 'type': 'structure'},
            {'tokens': ['Cell0'], 'type': 'content', 'cell_id': 0, 'bbox': [0, 0, 10, 10]},
            {'tokens': ['Cell5'], 'type': 'content', 'cell_id': 5, 'bbox': [50, 0, 60, 10]}  # Beyond bbox_count
        ]
    }
    
    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(data)
    
    # Only cells within bbox_count should be included
    bbox_count = get_bbox_nums(result['tokens'])
    valid_cells = [cell for cell in result['cells'] if cell['id'] < bbox_count]
    assert len(result['cells']) == len(valid_cells)

def test_repr():
    """Test string representation."""
    transform = LoadTokens(with_structure=True, with_cell=False, max_structure_token_len=100)
    repr_str = repr(transform)
    
    assert 'LoadTokens' in repr_str
    assert 'with_structure=True' in repr_str
    assert 'with_cell=False' in repr_str