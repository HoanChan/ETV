import pytest
import numpy as np
from PIL import Image

from datasets.transforms.get_cells import GetCell


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
            'text': 'Cell 1',
            'task_type': 'content'
        },
        {
            'bbox': [50, 20, 90, 50],  # Valid cell
            'text': 'Cell 2', 
            'task_type': 'content'
        },
        {
            'bbox': [5, 5, 8, 8],  # Too small cell (3x3)
            'text': 'Small',
            'task_type': 'content'
        },
        {
            'text': 'No bbox',  # Missing bbox
            'task_type': 'content'
        },
        {
            'bbox': [10, 10, 40, 30],  # Structure task (should be filtered)
            'text': 'Structure',
            'task_type': 'structure'
        }
    ]


@pytest.fixture
def sample_results(sample_img, sample_instances):
    """Create sample results dict."""
    return {
        'img': sample_img,
        'instances': sample_instances
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
    transform = GetCell(**params)
    for key, value in expected.items():
        assert getattr(transform, key) == value


def test_basic_cell_extraction(sample_results):
    """Test basic cell extraction functionality."""
    transform = GetCell()
    results = transform(sample_results.copy())
    
    # Should extract 2 valid cells (filtering out small cell, missing bbox, and structure task)
    assert 'cell_imgs' in results
    assert 'cell_texts' in results
    assert 'cell_bboxes' in results
    
    assert len(results['cell_imgs']) == 2
    assert len(results['cell_texts']) == 2
    assert len(results['cell_bboxes']) == 2
    
    # Check first cell
    assert results['cell_texts'][0] == 'Cell 1'
    assert results['cell_bboxes'][0] == [10, 10, 40, 30]
    assert results['cell_imgs'][0].shape == (20, 30, 3)  # h=30-10, w=40-10
    
    # Check second cell
    assert results['cell_texts'][1] == 'Cell 2'
    assert results['cell_bboxes'][1] == [50, 20, 90, 50]
    assert results['cell_imgs'][1].shape == (30, 40, 3)  # h=50-20, w=90-50


def test_no_task_filter(sample_results):
    """Test with no task filtering."""
    transform = GetCell(task_filter=None)
    results = transform(sample_results.copy())
    
    # Should extract 3 cells (including structure task, but still filtering small/missing bbox)
    assert len(results['cell_imgs']) == 3
    assert len(results['cell_texts']) == 3
    assert 'Structure' in results['cell_texts']


def test_pil_image_input(sample_instances):
    """Test with PIL Image input."""
    sample_img = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(sample_img)
    results = {
        'img': pil_img,
        'instances': sample_instances
    }
    
    transform = GetCell()
    results = transform(results)
    
    assert len(results['cell_imgs']) == 2
    assert isinstance(results['cell_imgs'][0], np.ndarray)

@pytest.mark.parametrize("test_instances,expected_count,expected_bboxes", [
    # Test with various invalid/edge case bboxes
    ([
        {
            'bbox': [-5, -5, 20, 25],  # Negative coordinates
            'text': 'Negative coords',
            'task_type': 'content'
        },
        {
            'bbox': [80, 70, 150, 120],  # Exceeds image bounds
            'text': 'Out of bounds',
            'task_type': 'content'
        },
        {
            'bbox': [30, 20, 20, 30],  # Reversed coordinates (x1 < x0)
            'text': 'Reversed coords',
            'task_type': 'content'
        },
        {
            'bbox': [10, 20, 10, 30],  # Zero width
            'text': 'Zero width',
            'task_type': 'content'
        }
    ], 3, [(0, 0, 20, 25), (80, 70, 100, 80), (20, 20, 30, 30)]),  # Expected clipped coords
])
def test_bbox_validation_and_clipping(sample_img, test_instances, expected_count, expected_bboxes):
    """Test bbox validation and clipping to image boundaries."""
    results = {
        'img': sample_img,
        'instances': test_instances
    }
    
    transform = GetCell()
    results = transform(results)
    
    # Should handle all cases appropriately
    assert len(results['cell_imgs']) == expected_count
    
    # Check clipped coordinates
    bboxes = results['cell_bboxes']
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        assert 0 <= x0 < x1 <= 100  # Within image width
        assert 0 <= y0 < y1 <= 80   # Within image height


@pytest.mark.parametrize("invalid_instances", [
    # Too few coordinates
    ([{
        'bbox': [10, 20],
        'text': 'Invalid bbox 1',
        'task_type': 'content'
    }]),
    # Too many coordinates  
    ([{
        'bbox': [10, 20, 30, 40, 50],
        'text': 'Invalid bbox 2',
        'task_type': 'content'
    }]),
    # Non-numeric
    ([{
        'bbox': ['a', 'b', 'c', 'd'],
        'text': 'Invalid bbox 3',
        'task_type': 'content'
    }]),
    # None bbox
    ([{
        'bbox': None,
        'text': 'Invalid bbox 4',
        'task_type': 'content'
    }])
])
def test_invalid_bbox_formats(sample_img, invalid_instances):
    """Test handling of invalid bbox formats."""
    results = {
        'img': sample_img,
        'instances': invalid_instances
    }
    
    transform = GetCell()
    results = transform(results)
    
    # Should extract no cells due to invalid bboxes
    assert len(results['cell_imgs']) == 0
    assert len(results['cell_texts']) == 0
    assert len(results['cell_bboxes']) == 0


@pytest.mark.parametrize("min_cell_size,expected_count", [
    (5, 1),   # Default min_cell_size=5, only 5x10 cell passes
    (10, 1),  # min_cell_size=10, 5x10 cell still passes (height=10 >= 10)
    (15, 0)   # min_cell_size=15, no cells meet minimum size
])
def test_min_cell_size_filtering(sample_img, min_cell_size, expected_count):
    """Test minimum cell size filtering."""
    small_instances = [
        {
            'bbox': [10, 10, 14, 14],  # 4x4 cell
            'text': 'Small cell',
            'task_type': 'content'
        },
        {
            'bbox': [20, 20, 25, 30],  # 5x10 cell
            'text': 'Valid cell',
            'task_type': 'content'
        }
    ]
    
    results = {
        'img': sample_img,
        'instances': small_instances
    }
    
    transform = GetCell(min_cell_size=min_cell_size)
    results = transform(results)
    assert len(results['cell_imgs']) == expected_count


@pytest.mark.parametrize("test_input,expected_count", [
    # Empty instances list
    ([], 0),
    # Missing instances key  
    (None, 0)
])
def test_empty_or_missing_instances(sample_img, test_input, expected_count):
    """Test with empty or missing instances."""
    if test_input is None:
        results = {'img': sample_img}  # No 'instances' key
    else:
        results = {'img': sample_img, 'instances': test_input}
    
    transform = GetCell()
    results = transform(results)
    
    assert len(results['cell_imgs']) == expected_count
    assert len(results['cell_texts']) == expected_count
    assert len(results['cell_bboxes']) == expected_count

def test_grayscale_image(sample_instances):
    """Test with grayscale image."""
    gray_img = np.random.randint(0, 255, (80, 100), dtype=np.uint8)
    results = {
        'img': gray_img,
        'instances': sample_instances[:2]  # First 2 valid instances
    }
    
    transform = GetCell()
    results = transform(results)
    
    assert len(results['cell_imgs']) == 2
    # Grayscale cells should have shape (h, w)
    assert len(results['cell_imgs'][0].shape) == 2
    assert len(results['cell_imgs'][1].shape) == 2


def test_cell_image_copy(sample_results):
    """Test that cell images are properly copied."""
    transform = GetCell()
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
    transform = GetCell(**params)
    repr_str = repr(transform)
    
    for part in expected_repr_parts:
        assert part in repr_str


@pytest.mark.parametrize("invalid_img", [
    np.array([1, 2, 3]),  # 1D array
])
def test_invalid_image_shape(sample_instances, invalid_img):
    """Test with invalid image shape."""
    results = {
        'img': invalid_img,
        'instances': sample_instances
    }
    
    transform = GetCell()
    
    with pytest.raises(ValueError, match="Invalid image shape"):
        transform(results)
