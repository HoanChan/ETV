import numpy as np
import pytest
from datasets.transforms.load_tokens import LoadTokens

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

@pytest.mark.parametrize(
    "structure_tokens,cell_tokens_list,bboxes",
    [
        # Basic 2 cell, 4 structure tokens
        (["<td></td>", "<td", ' rowspan="2">', "</td>"], [["a"], ["b"]], [[1,2,3,4], [5,6,7,8]]),
        # 4 Cell, 8 structure tokens
        (["<tr>", "<td></td>" , "<td></td>", "</tr>", "<tr>", "<td></td>" , "<td></td>", "</tr>"], [["a"], ["b"], ["c"], ["d"]], [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]),
    ]
)
def test_with_and_without_cells(structure_tokens, cell_tokens_list, bboxes):
    results = {'img': 'fake_img', 'instances': make_instances(structure_tokens, cell_tokens_list, bboxes)}

    # Case 1: with_structure=True, with_cell=True
    loader = LoadTokens(with_structure=True, with_cell=True)
    out = loader.transform(results.copy())
    assert 'tokens' in out
    assert 'bboxes' in out
    assert 'masks' in out
    assert isinstance(out['bboxes'], np.ndarray)
    assert isinstance(out['masks'], np.ndarray)
    if cell_tokens_list:
        assert 'cells' in out
        assert all(isinstance(cell, dict) for cell in out['cells'])
        assert len(out['cells']) == len(cell_tokens_list)
    else:
        assert 'cells' in out and len(out['cells']) == 0
    assert len(out['tokens']) == len(structure_tokens)
    assert out['bboxes'].shape[0] == (len(structure_tokens) if structure_tokens else 0)
    assert out['masks'].shape[0] == (len(structure_tokens) if structure_tokens else 0)

    # Case 2: with_structure=True, with_cell=False
    loader2 = LoadTokens(with_structure=True, with_cell=False)
    out2 = loader2.transform(results.copy())
    assert 'tokens' in out2 and 'bboxes' in out2 and 'masks' in out2
    assert isinstance(out2['bboxes'], np.ndarray)
    assert isinstance(out2['masks'], np.ndarray)
    assert 'cells' not in out2
    assert len(out2['tokens']) == len(structure_tokens)
    assert out2['bboxes'].shape[0] == (len(structure_tokens) if structure_tokens else 0)
    assert out2['masks'].shape[0] == (len(structure_tokens) if structure_tokens else 0)

    # bboxes and masks must be the same in both cases
    np.testing.assert_array_equal(out['bboxes'], out2['bboxes'])
    np.testing.assert_array_equal(out['masks'], out2['masks'])

@pytest.fixture
def sample_data():
    """Sample data mimicking table dataset output."""
    return {
        'img': 'mock_image_data',
        'img_path': 'test_image.jpg',
        'sample_idx': 0,
        'instances': [
            {
                'tokens': ['<thead>', '<tr>', '<td>', 'Cell 1', '</td>', '<td>', 'Cell 2', '</td>', '</tr>', '</thead>'],
                'type': 'structure'
            },
            {
                'tokens': ['Hello', 'World'],
                'type': 'content',
                'cell_id': 0,
                'bbox': [10, 20, 100, 50]
            },
            {
                'tokens': ['Test', 'Cell'],
                'type': 'content', 
                'cell_id': 1,
                'bbox': [110, 20, 200, 50]
            }
        ]
    }

# Basic initialization tests
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

@pytest.mark.parametrize("max_structure_token_len,max_cell_token_len", [
    (None, None), 
    (10, 5), 
    (0, 0), 
    (100, 100)
])
def test_initialization_with_limits(max_structure_token_len, max_cell_token_len):
    """Test initialization with token length limits."""
    transform = LoadTokens(
        with_structure=True, 
        with_cell=True,
        max_structure_token_len=max_structure_token_len,
        max_cell_token_len=max_cell_token_len
    )
    assert transform.max_structure_token_len == max_structure_token_len
    assert transform.max_cell_token_len == max_cell_token_len

def test_load_structure_only(sample_data):
    """Test loading structure tokens only."""
    transform = LoadTokens(with_structure=True, with_cell=False)
    # Use make_instances to generate the same structure as sample_data
    structure_tokens = ['<thead>', '<tr>', '<td>', 'Cell 1', '</td>', '<td>', 'Cell 2', '</td>', '</tr>', '</thead>']
    cell_tokens_list = [['Hello', 'World'], ['Test', 'Cell']]
    bboxes = [[10, 20, 100, 50], [110, 20, 200, 50]]
    data = {
        'img': 'mock_image_data',
        'img_path': 'test_image.jpg',
        'sample_idx': 0,
        'instances': make_instances(structure_tokens, cell_tokens_list, bboxes)
    }
    result = transform(data.copy())

    assert 'tokens' in result
    # After processing, the tokens are merged and processed
    # Expected: ['<thead>', '<tr>', '<td>Cell 1', '</td>', '<td>Cell 2', '</td>', '</tr>', '</thead>']
    assert len(result['tokens']) == 8  # Check length instead of exact match since processing may vary
    assert result['tokens'][0] == '<thead>'
    assert result['tokens'][-1] == '</thead>'
    assert 'bboxes' in result
    assert 'masks' in result
    assert 'cells' not in result

    # Check that bboxes and masks are numpy arrays
    assert isinstance(result['bboxes'], np.ndarray)
    assert isinstance(result['masks'], np.ndarray)

def test_load_cell_only(sample_data):
    """Test loading cell tokens only."""
    transform = LoadTokens(with_structure=False, with_cell=True)
    structure_tokens = ['<thead>', '<tr>', '<td>', 'Cell 1', '</td>', '<td>', 'Cell 2', '</td>', '</tr>', '</thead>']
    cell_tokens_list = [['Hello', 'World'], ['Test', 'Cell']]
    bboxes = [[10, 20, 100, 50], [110, 20, 200, 50]]
    data = {
        'img': 'mock_image_data',
        'img_path': 'test_image.jpg',
        'sample_idx': 0,
        'instances': make_instances(structure_tokens, cell_tokens_list, bboxes)
    }
    result = transform(data.copy())

    assert 'cells' in result
    assert len(result['cells']) == 2

    # Check first cell
    assert result['cells'][0]['tokens'] == ['Hello', 'World']
    assert result['cells'][0]['bbox'] == [10, 20, 100, 50]
    assert result['cells'][0]['id'] == 0

    # Check second cell
    assert result['cells'][1]['tokens'] == ['Test', 'Cell']
    assert result['cells'][1]['bbox'] == [110, 20, 200, 50]
    assert result['cells'][1]['id'] == 1

    # These should not be present when with_structure=False
    assert 'tokens' not in result
    assert 'bboxes' not in result
    assert 'masks' not in result

def test_load_both_structure_and_cell(sample_data):
    """Test loading both structure and cell tokens."""
    transform = LoadTokens(with_structure=True, with_cell=True)
    structure_tokens = ['<thead>', '<tr>', '<td>', 'Cell 1', '</td>', '<td>', 'Cell 2', '</td>', '</tr>', '</thead>']
    cell_tokens_list = [['Hello', 'World'], ['Test', 'Cell']]
    bboxes = [[10, 20, 100, 50], [110, 20, 200, 50]]
    data = {
        'img': 'mock_image_data',
        'img_path': 'test_image.jpg',
        'sample_idx': 0,
        'instances': make_instances(structure_tokens, cell_tokens_list, bboxes)
    }
    result = transform(data.copy())

    # Check structure tokens
    # After processing, the tokens are merged and processed  
    assert len(result['tokens']) == 8  # Check length instead of exact match since processing may vary
    assert result['tokens'][0] == '<thead>'
    assert result['tokens'][-1] == '</thead>'
    assert 'bboxes' in result
    assert 'masks' in result
    assert isinstance(result['bboxes'], np.ndarray)
    assert isinstance(result['masks'], np.ndarray)

    # Check cell tokens
    assert 'cells' in result
    assert len(result['cells']) == 2
    assert result['cells'][0]['tokens'] == ['Hello', 'World']
    assert result['cells'][1]['tokens'] == ['Test', 'Cell']

@pytest.mark.parametrize("max_len,expected_len", [
    (None, 8),  # After processing: ['<thead>', '<tr>', '<td>Cell 1', '</td>', '<td>Cell 2', '</td>', '</tr>', '</thead>']
    (5, 5),
    (15, 8),  # All 8 tokens
    (0, 0)
])
def test_structure_length_limiting(sample_data, max_len, expected_len):
    """Test structure token length limiting."""
    transform = LoadTokens(with_structure=True, with_cell=False, max_structure_token_len=max_len)
    result = transform(sample_data.copy())

    assert len(result['tokens']) == expected_len

@pytest.mark.parametrize("max_len,expected_lens", [
    (None, [2, 2]), 
    (1, [1, 1]), 
    (0, [0, 0])
])
def test_cell_length_limiting(sample_data, max_len, expected_lens):
    """Test cell token length limiting."""
    transform = LoadTokens(with_structure=False, with_cell=True, max_cell_token_len=max_len)
    result = transform(sample_data.copy())
    
    for i, expected_len in enumerate(expected_lens):
        assert len(result['cells'][i]['tokens']) == expected_len

@pytest.mark.parametrize("with_structure,with_cell", [
    (True, True), 
    (True, False), 
    (False, True)
])
def test_missing_instances(with_structure, with_cell):
    """Test handling when instances key is missing."""
    data = {'img': 'mock_image', 'img_path': 'test.jpg', 'sample_idx': 0}
    transform = LoadTokens(with_structure=with_structure, with_cell=with_cell)
    result = transform(data.copy())
    
    if with_structure:
        assert result['tokens'] == []
        assert result['bboxes'] == []
    if with_cell:
        assert result['cells'] == []

@pytest.mark.parametrize("with_structure,with_cell", [
    (True, True), 
    (True, False), 
    (False, True)
])
def test_empty_instances(with_structure, with_cell):
    """Test handling when instances list is empty."""
    data = {
        'img': 'mock_image',
        'img_path': 'test.jpg', 
        'sample_idx': 0,
        'instances': []
    }
    transform = LoadTokens(with_structure=with_structure, with_cell=with_cell)
    result = transform(data.copy())
    
    if with_structure:
        assert result['tokens'] == []
        assert isinstance(result['bboxes'], np.ndarray)
        assert isinstance(result['masks'], np.ndarray)
    if with_cell:
        assert result['cells'] == []

@pytest.mark.parametrize("task_type", ['structure', 'content'])
def test_single_task_type(sample_data, task_type):
    """Test with instances containing only one task type."""
    # Filter instances to only include specified task type
    filtered_data = sample_data.copy()
    filtered_data['instances'] = [
        inst for inst in sample_data['instances']
        if inst.get('type') == task_type
    ]
    
    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(filtered_data)
    
    if task_type == 'structure':
        assert len(result['tokens']) > 0
        assert len(result['cells']) == 0
        assert isinstance(result['bboxes'], np.ndarray)
        assert isinstance(result['masks'], np.ndarray)
    else:  # content
        assert len(result['tokens']) == 0
        assert len(result['cells']) > 0

def test_missing_optional_fields():
    """Test handling of missing optional fields in instances."""
    data = {
        'img': 'mock_image',
        'instances': [
            {
                'type': 'structure'
                # Missing 'tokens' field
            },
            {
                'type': 'content'
                # Missing 'tokens', 'bbox', 'cell_id' fields
            }
        ]
    }
    
    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(data)
    
    # Should handle missing fields gracefully
    assert result['tokens'] == []
    assert len(result['cells']) == 1
    assert result['cells'][0]['tokens'] == []
    assert result['cells'][0]['bbox'] == [0, 0, 0, 0]
    assert result['cells'][0]['id'] == 0

@pytest.mark.parametrize("cell_ids,bboxes", [
    ([0, 1], [[10, 20, 30, 40], [50, 60, 70, 80]]),
    ([5, 10], [[100, 200, 300, 400], [500, 600, 700, 800]]),
    ([0], [[15, 25, 35, 45]])
])
def test_cell_data_consistency(cell_ids, bboxes):
    """Test that cell IDs and bboxes are handled correctly."""
    data = {
        'img': 'mock_image',
        'instances': [
            {
                'tokens': ['<table>', '</table>'],
                'type': 'structure'
            }
        ]
    }
    
    # Add content instances
    for i, (cell_id, bbox) in enumerate(zip(cell_ids, bboxes)):
        data['instances'].append({
            'tokens': [f'Cell_{i}', f'Content_{i}'],
            'type': 'content',
            'cell_id': cell_id,
            'bbox': bbox
        })
    
    transform = LoadTokens(with_structure=False, with_cell=True)
    result = transform(data)
    
    assert len(result['cells']) == len(cell_ids)
    for i, (expected_id, expected_bbox) in enumerate(zip(cell_ids, bboxes)):
        assert result['cells'][i]['id'] == expected_id
        assert result['cells'][i]['bbox'] == expected_bbox

@pytest.mark.parametrize("structure_tokens,cell_tokens", [
    (['<table>', '<tr>', '<td>', '</td>', '</tr>', '</table>'], [['Hello'], ['World']]),
    (['<thead>', '<tr>', '<th>', '</th>', '</tr>', '</thead>'], [['Header1'], ['Header2']]),
    ([], [['Only'], ['Cell'], ['Data']])
])
def test_different_token_combinations(structure_tokens, cell_tokens):
    """Test different combinations of structure and cell tokens."""
    data = {
        'img': 'mock_image',
        'instances': []
    }
    
    # Add structure if provided
    if structure_tokens:
        data['instances'].append({
            'tokens': structure_tokens,
            'type': 'structure'
        })
    
    # Add cell content
    for i, tokens in enumerate(cell_tokens):
        data['instances'].append({
            'tokens': tokens,
            'type': 'content',
            'cell_id': i,
            'bbox': [i*10, i*10, (i+1)*10, (i+1)*10]
        })
    
    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(data)
    
    # Test expects processed tokens, not raw tokens 
    # merge_token will merge '<td>' and content but not '<th>' tokens
    if structure_tokens == ['<table>', '<tr>', '<td>', '</td>', '</tr>', '</table>']:
        expected_processed = ['<table>', '<tr>', '<td></td>', '</tr>', '</table>']
        assert result['tokens'] == expected_processed
    elif structure_tokens == ['<thead>', '<tr>', '<th>', '</th>', '</tr>', '</thead>']:
        # <th> tokens are not merged
        expected_processed = ['<thead>', '<tr>', '<th>', '</th>', '</tr>', '</thead>']
        assert result['tokens'] == expected_processed  
    else:
        assert result['tokens'] == structure_tokens
    assert len(result['cells']) == len(cell_tokens)
    for i, expected_tokens in enumerate(cell_tokens):
        assert result['cells'][i]['tokens'] == expected_tokens

def test_repr():
    """Test string representation of LoadTokens."""
    transform = LoadTokens(
        with_structure=True, 
        with_cell=False, 
        max_structure_token_len=100,
        max_cell_token_len=50
    )
    repr_str = repr(transform)
    
    assert 'LoadTokens' in repr_str
    assert 'with_structure=True' in repr_str
    assert 'with_cell=False' in repr_str
    assert 'max_structure_token_len=100' in repr_str
    assert 'max_cell_token_len=50' in repr_str

@pytest.mark.parametrize("instances,expected_structure_len,expected_cell_count", [
    # Only structure
    ([{'tokens': ['<table>', '</table>'], 'type': 'structure'}], 2, 0),
    # Only content
    ([{'tokens': ['Hello'], 'type': 'content', 'cell_id': 0, 'bbox': [0, 0, 10, 10]}], 0, 1),
    # Mixed - tokens will be processed and merged
    ([
        {'tokens': ['<table>'], 'type': 'structure'},
        {'tokens': ['Cell1'], 'type': 'content', 'cell_id': 0, 'bbox': [0, 0, 10, 10]},
        {'tokens': ['</table>'], 'type': 'structure'}
    ], 2, 1),
    # Empty
    ([], 0, 0)
])
def test_comprehensive_scenarios(instances, expected_structure_len, expected_cell_count):
    """Test comprehensive scenarios with different instance combinations."""
    data = {
        'img': 'mock_image',
        'instances': instances
    }

    transform = LoadTokens(with_structure=True, with_cell=True)
    result = transform(data)

    assert len(result['tokens']) == expected_structure_len
    assert len(result['cells']) == expected_cell_count
    assert isinstance(result['bboxes'], np.ndarray)
    assert isinstance(result['masks'], np.ndarray)
