import pytest
import numpy as np
from PIL import Image

from datasets.transforms.get_cells import GetCells


# Test fixtures
@pytest.fixture
def sample_img():
    """Create sample image (100x80, RGB)."""
    return np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_instances():
    """Create sample instances with valid cells."""
    return [
        {
            'bbox': [10, 10, 40, 30],  # Valid cell
            'task_type': 'content'
        },
        {
            'bbox': [50, 20, 90, 50],  # Valid cell
            'task_type': 'content'
        },
        {
            'bbox': [5, 5, 8, 8],  # Too small cell (3x3)
            'task_type': 'content'
        },
        {
            'task_type': 'content'  # Missing bbox
        },
        {
            'bbox': [10, 10, 40, 30],  # Structure task (should be filtered)
            'task_type': 'structure'
        }
    ]


@pytest.fixture
def sample_gt_cell_bboxes():
    """Create sample gt_cell_bboxes (from LoadTokens output)."""
    return np.array([
        [10, 10, 40, 30],  # Valid cell
        [50, 20, 90, 50],  # Valid cell
    ], dtype=np.float32)


@pytest.fixture
def sample_results(sample_img, sample_instances):
    """Create sample results dict with raw instances."""
    return {
        'img': sample_img,
        'instances': sample_instances
    }


@pytest.fixture
def sample_results_with_gt_bboxes(sample_img, sample_gt_cell_bboxes):
    """Create sample results dict with processed gt_cell_bboxes (from LoadTokens)."""
    return {
        'img': sample_img,
        'gt_cell_bboxes': sample_gt_cell_bboxes
    }


@pytest.mark.parametrize("params,expected", [
    # Default parameters
    ({}, {'img_key': 'img', 'instances_key': 'instances', 'task_filter': 'content', 'min_cell_size': 5}),
    # Custom parameters
    ({
        'img_key': 'image',
        'instances_key': 'annotations',
        'task_filter': None,
        'min_cell_size': 10
    }, {'img_key': 'image', 'instances_key': 'annotations', 'task_filter': None, 'min_cell_size': 10})
])
def test_init_params(params, expected):
    """Test initialization with various parameters."""
    transform = GetCells(**params)
    for key, value in expected.items():
        assert getattr(transform, key) == value


@pytest.mark.parametrize("task_filter,expected_count,description", [
    ('content', 2, "basic cell extraction with content filter"),
    (None, 3, "no task filtering - includes structure task"),
])
def test_cell_extraction_basic(sample_results, task_filter, expected_count, description):
    """Test basic cell extraction functionality with different task filters."""
    transform = GetCells(task_filter=task_filter)
    results = transform(sample_results.copy())
    
    assert 'cell_imgs' in results
    assert len(results['cell_imgs']) == expected_count
    
    if expected_count >= 2:
        # Check cell shapes for valid extractions
        assert results['cell_imgs'][0].shape == (20, 30, 3)  # h=30-10, w=40-10
        assert results['cell_imgs'][1].shape == (30, 40, 3)  # h=50-20, w=90-50


@pytest.mark.parametrize("img_type,description", [
    ("numpy_rgb", "numpy RGB array"),
    ("pil_image", "PIL Image"),
    ("numpy_gray", "numpy grayscale array"),
])
def test_image_input_types(sample_instances, img_type, description):
    """Test with different image input types."""
    if img_type == "numpy_rgb":
        img = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
        expected_cell_shape_len = 3
    elif img_type == "pil_image":
        img_array = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        expected_cell_shape_len = 3
    else:  # numpy_gray
        img = np.random.randint(0, 255, (80, 100), dtype=np.uint8)
        expected_cell_shape_len = 2
    
    results = {'img': img, 'instances': sample_instances[:2]}  # First 2 valid instances
    
    transform = GetCells()
    results = transform(results)
    
    assert len(results['cell_imgs']) == 2
    assert isinstance(results['cell_imgs'][0], np.ndarray)
    assert len(results['cell_imgs'][0].shape) == expected_cell_shape_len

@pytest.mark.parametrize("test_instances,expected_count", [
    # Test with various invalid/edge case bboxes
    ([
        {
            'bbox': [-5, -5, 20, 25],  # Negative coordinates
            'task_type': 'content'
        },
        {
            'bbox': [80, 70, 150, 120],  # Exceeds image bounds
            'task_type': 'content'
        },
        {
            'bbox': [30, 20, 20, 30],  # Reversed coordinates (x1 < x0)
            'task_type': 'content'
        },
        {
            'bbox': [10, 20, 10, 30],  # Zero width
            'task_type': 'content'
        }
    ], 3),  # Expected count: 3 valid cells after clipping/validation
])
def test_bbox_validation_and_clipping(sample_img, test_instances, expected_count):
    """Test bbox validation and clipping to image boundaries."""
    results = {
        'img': sample_img,
        'instances': test_instances
    }
    
    transform = GetCells()
    results = transform(results)
    
    # Should handle all cases appropriately
    assert len(results['cell_imgs']) == expected_count
    
    # Check that extracted images have valid dimensions
    for cell_img in results['cell_imgs']:
        assert cell_img.shape[0] > 0  # Height > 0
        assert cell_img.shape[1] > 0  # Width > 0


@pytest.mark.parametrize("invalid_instances,description", [
    ([{'bbox': [10, 20], 'task_type': 'content'}], "too few coordinates"),
    ([{'bbox': [10, 20, 30, 40, 50], 'task_type': 'content'}], "too many coordinates"),
    ([{'bbox': ['a', 'b', 'c', 'd'], 'task_type': 'content'}], "non-numeric bbox"),
    ([{'bbox': None, 'task_type': 'content'}], "None bbox"),
])
def test_invalid_bbox_formats(sample_img, invalid_instances, description):
    """Test handling of various invalid bbox formats."""
    results = {'img': sample_img, 'instances': invalid_instances}
    
    transform = GetCells()
    results = transform(results)
    
    # Should extract no cells due to invalid bboxes
    assert len(results['cell_imgs']) == 0


@pytest.mark.parametrize("test_input,expected_count,description", [
    ([], 0, "empty instances list"),
    (None, 0, "missing instances key"),
])
def test_empty_or_missing_instances(sample_img, test_input, expected_count, description):
    """Test with empty or missing instances."""
    if test_input is None:
        results = {'img': sample_img}  # No 'instances' key
    else:
        results = {'img': sample_img, 'instances': test_input}
    
    transform = GetCells()
    results = transform(results)
    
    assert len(results['cell_imgs']) == expected_count
def test_empty_or_missing_instances(sample_img, test_input, expected_count):
    """Test with empty or missing instances."""
    if test_input is None:
        results = {'img': sample_img}  # No 'instances' key
    else:
        results = {'img': sample_img, 'instances': test_input}
    
    transform = GetCells()
    results = transform(results)
    
    assert len(results['cell_imgs']) == expected_count

def test_cell_image_copy(sample_results):
    """Test that cell images are properly copied."""
    transform = GetCells()
    results = transform(sample_results.copy())
    
    # Modify original image
    original_value = sample_results['img'][15, 15, 0]
    sample_results['img'][15, 15, 0] = 255
    
    # Cell image should not be affected (due to .copy())
    cell_img = results['cell_imgs'][0]
    # Cell coordinates are [10:30, 10:40], so [15,15] maps to [5,5] in cell
    assert cell_img[5, 5, 0] == original_value


@pytest.mark.parametrize("params,expected_repr_parts", [
    ({
        'img_key': 'image',
        'instances_key': 'annotations', 
        'task_filter': 'content',
        'min_cell_size': 10
    }, ['GetCell', 'img_key=image', 'instances_key=annotations', 'task_filter=content', 'min_cell_size=10'])
])
def test_repr(params, expected_repr_parts):
    """Test string representation."""
    transform = GetCells(**params)
    repr_str = repr(transform)
    
    for part in expected_repr_parts:
        assert part in repr_str


@pytest.mark.parametrize("invalid_img,expected_error", [
    (np.array([1, 2, 3]), "Invalid image shape"),
])
def test_invalid_inputs(sample_instances, invalid_img, expected_error):
    """Test with invalid inputs."""
    results = {'img': invalid_img, 'instances': sample_instances}
    transform = GetCells()
    
    with pytest.raises(ValueError, match=expected_error):
        transform(results)

def test_priority_gt_bboxes_over_instances(sample_img, sample_instances, sample_gt_cell_bboxes):
    """Test that gt_cell_bboxes takes priority over instances."""
    results = {
        'img': sample_img,
        'gt_cell_bboxes': sample_gt_cell_bboxes,
        'instances': sample_instances  # This should be ignored
    }
    
    transform = GetCells()
    results = transform(results)
    
    # Should extract 2 cells from gt_cell_bboxes, not from instances
    assert len(results['cell_imgs']) == 2
    assert results['cell_imgs'][0].shape == (20, 30, 3)
    assert results['cell_imgs'][1].shape == (30, 40, 3)


@pytest.mark.parametrize("invalid_img,expected_error", [
    (np.array([1, 2, 3]), "Invalid image shape"),
])
def test_invalid_inputs(sample_instances, invalid_img, expected_error):
    """Test with invalid inputs."""
    results = {'img': invalid_img, 'instances': sample_instances}
    transform = GetCells()
    
    with pytest.raises(ValueError, match=expected_error):
        transform(results)


@pytest.mark.parametrize("gt_bboxes_scenario,expected_count,description", [
    ("standard", 2, "standard gt_cell_bboxes extraction"),
    ("empty_fallback", 2, "empty gt_bboxes fallback to instances"),
    ("missing_fallback", 2, "missing gt_bboxes uses instances"),
    ("invalid_coords", 4, "gt_bboxes with invalid coordinates"),
])
def test_gt_cell_bboxes_scenarios(sample_img, sample_instances, sample_gt_cell_bboxes, 
                                  gt_bboxes_scenario, expected_count, description):
    """Test various gt_cell_bboxes scenarios."""
    if gt_bboxes_scenario == "standard":
        results = {'img': sample_img, 'gt_cell_bboxes': sample_gt_cell_bboxes}
    elif gt_bboxes_scenario == "empty_fallback":
        results = {
            'img': sample_img,
            'gt_cell_bboxes': np.array([], dtype=np.float32).reshape(0, 4),
            'instances': sample_instances
        }
    elif gt_bboxes_scenario == "missing_fallback":
        results = {'img': sample_img, 'instances': sample_instances}
    else:  # invalid_coords
        invalid_gt_bboxes = np.array([
            [10, 10, 40, 30],    # Valid
            [-5, -5, 20, 25],    # Negative coordinates
            [80, 70, 150, 120],  # Exceeds image bounds
            [30, 20, 20, 30],    # Reversed coordinates
            [10, 20, 10, 30],    # Zero width
        ], dtype=np.float32)
        results = {'img': sample_img, 'gt_cell_bboxes': invalid_gt_bboxes}
    
    transform = GetCells()
    results = transform(results)
    
    assert len(results['cell_imgs']) == expected_count
    
    # Check that all extracted images have valid dimensions
    for cell_img in results['cell_imgs']:
        assert cell_img.shape[0] > 0  # Height > 0
        assert cell_img.shape[1] > 0  # Width > 0
    
    # For standard case, check exact shapes
    if gt_bboxes_scenario == "standard" and expected_count >= 2:
        assert results['cell_imgs'][0].shape == (20, 30, 3)
        assert results['cell_imgs'][1].shape == (30, 40, 3)

def test_integration_with_load_tokens_output(sample_img):
    """Test integration with realistic LoadTokens output format."""
    # Simulate results after LoadTokens transform
    load_tokens_output = {
        'img': sample_img,
        'gt_cell_tokens': [['cell', '1'], ['cell', '2'], ['cell', '3']],
        'gt_cell_ids': [0, 1, 2],
        'gt_cell_bboxes': np.array([
            [10, 10, 40, 30],
            [50, 20, 90, 50], 
            [5, 60, 35, 75]
        ], dtype=np.float32),
        'gt_task_types': ['content', 'content', 'content'],
        # Original instances still present but should be ignored
        'instances': [
            {'bbox': [0, 0, 10, 10], 'task_type': 'content'},  # Different from gt_cell_bboxes
            {'bbox': [90, 90, 100, 100], 'task_type': 'content'}
        ]
    }
    
    transform = GetCells()
    results = transform(load_tokens_output.copy())
    
    # Should extract 3 cells from gt_cell_bboxes, ignoring instances
    assert len(results['cell_imgs']) == 3
    
    # Verify cell dimensions match gt_cell_bboxes, not instances
    assert results['cell_imgs'][0].shape == (20, 30, 3)  # [10,10,40,30] -> 20x30
    assert results['cell_imgs'][1].shape == (30, 40, 3)  # [50,20,90,50] -> 30x40
    assert results['cell_imgs'][2].shape == (15, 30, 3)  # [5,60,35,75] -> 15x30
    
    # Ensure consistency: len(cell_imgs) == len(gt_cell_tokens) == len(gt_cell_ids)
    assert len(results['cell_imgs']) == len(load_tokens_output['gt_cell_tokens'])
    assert len(results['cell_imgs']) == len(load_tokens_output['gt_cell_ids'])
