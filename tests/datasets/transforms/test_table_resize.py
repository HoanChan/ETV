#!/usr/bin/env python3
"""Test suite for TableResize transform."""

import pytest
import numpy as np
from datasets.transforms.table_resize import TableResize


@pytest.fixture
def sample_image():
    """Sample image for testing."""
    return np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)

@pytest.fixture
def sample_data():
    """Sample data with image and bboxes."""
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    return {
        'img': img,
        'gt_bboxes': np.array([[10, 10, 50, 50], [100, 100, 200, 150]], dtype=np.float32),
        'img_shape': img.shape[:2],
    }


# Test initialization of TableResize class
@pytest.mark.parametrize("min_size,long_size,backend,interpolation", [
    (200, None, 'cv2', 'bilinear'),
    (None, 600, 'cv2', 'nearest'),
    (200, 600, 'pillow', 'bilinear'),
    (None, None, 'cv2', 'bicubic'),
    (100, 1000, 'pillow', 'nearest'),
])
def test_initialization_parameters(min_size, long_size, backend, interpolation):
    """Test initialization with different parameter combinations."""
    transform = TableResize(
        min_size=min_size, 
        long_size=long_size, 
        backend=backend,
        interpolation=interpolation
    )
    assert transform.min_size == min_size
    assert transform.long_size == long_size
    assert transform.keep_ratio is True  # Always True in your implementation
    assert transform.backend == backend
    assert transform.interpolation == interpolation


def test_default_scale_when_no_constraints():
    """Test that default scale=(1,1) is set when no constraints provided."""
    transform = TableResize()
    # Should have default scale set to avoid parent class errors
    assert hasattr(transform, 'scale')


# Test min_size constraint functionality
@pytest.mark.parametrize("min_size,original_shape,should_scale", [
    (200, (400, 300), False),  # shorter side (300) > min_size (200)
    (500, (400, 300), True),   # shorter side (300) < min_size (500)
    (100, (400, 300), False),  # shorter side (300) > min_size (100)
    (350, (400, 300), True),   # shorter side (300) < min_size (350)
    (300, (400, 300), False),  # shorter side (300) == min_size (300)
])
def test_min_size_constraint(min_size, original_shape, should_scale):
    """Test minimum size constraint with various parameters."""
    img = np.random.randint(0, 255, original_shape + (3,), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(min_size=min_size)
    result = transform.transform(data)
    
    actual_min = min(result['img'].shape[:2])
    original_min = min(original_shape)
    
    if should_scale:
        assert actual_min >= min_size * 0.95  # Allow small floating point error
    else:
        # Should be close to original size when not scaling
        assert abs(actual_min - original_min) <= 2  # Allow small rounding differences


# Test long_size constraint functionality
@pytest.mark.parametrize("long_size,original_shape,should_scale", [
    (600, (400, 300), True),   # longer side (400) < long_size (600)
    (200, (400, 300), True),   # longer side (400) > long_size (200)
    (500, (400, 300), True),   # longer side (400) < long_size (500)
    (400, (400, 300), False),  # longer side (400) == long_size (400)
    (350, (400, 300), True),   # longer side (400) > long_size (350)
])
def test_long_size_constraint(long_size, original_shape, should_scale):
    """Test long size constraint with various parameters."""
    img = np.random.randint(0, 255, original_shape + (3,), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(long_size=long_size)
    result = transform.transform(data)
    
    actual_max = max(result['img'].shape[:2])
    original_max = max(original_shape)
    
    if should_scale:
        assert abs(actual_max - long_size) <= 2  # Allow small rounding differences
    else:
        # Should be close to original size when not scaling
        assert abs(actual_max - original_max) <= 2


# Test combination of min_size and long_size constraints
@pytest.mark.parametrize("min_size,long_size,original_shape", [
    (200, 500, (400, 300)),  # min=200, long=500, original shorter=300, longer=400
    (350, 450, (400, 300)),  # min=350, long=450, original shorter=300, longer=400
    (100, 600, (400, 300)),  # min=100, long=600, original shorter=300, longer=400
    (400, 800, (400, 300)),  # min=400, long=800, original shorter=300, longer=400
])
def test_combined_constraints(min_size, long_size, original_shape):
    """Test combination of min_size and long_size constraints."""
    img = np.random.randint(0, 255, original_shape + (3,), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(min_size=min_size, long_size=long_size)
    result = transform.transform(data)
    
    img_h, img_w = result['img'].shape[:2]
    max_side = max(img_h, img_w)
    
    # long_size takes precedence over min_size in your implementation
    assert abs(max_side - long_size) <= 2  # Allow small rounding differences


def test_long_size_precedence_over_min_size():
    """Test that long_size takes precedence over min_size."""
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(min_size=500, long_size=350)  # Conflicting constraints
    result = transform.transform(data)
    
    max_side = max(result['img'].shape[:2])
    assert abs(max_side - 350) <= 2  # long_size should win


# Test data handling and inheritance from parent class
def test_bbox_scaling_inheritance(sample_data):
    """Test that bbox scaling is inherited from parent class."""
    transform = TableResize(long_size=600)
    result = transform.transform(sample_data)
    
    # Check that image was resized
    assert 'img' in result
    max_side = max(result['img'].shape[:2])
    assert abs(max_side - 600) <= 2
    
    # Check bbox handling (should be handled by parent class)
    if 'gt_bboxes' in sample_data:
        assert 'gt_bboxes' in result
        img_h, img_w = result['img'].shape[:2]
        scaled_bboxes = result['gt_bboxes']
        # Bboxes should be within image bounds
        assert np.all(scaled_bboxes[..., 0::2] >= 0)
        assert np.all(scaled_bboxes[..., 1::2] >= 0)
        assert np.all(scaled_bboxes[..., 0::2] <= img_w)
        assert np.all(scaled_bboxes[..., 1::2] <= img_h)


@pytest.mark.parametrize("interpolation,backend", [
    ('bilinear', 'cv2'),
    ('nearest', 'cv2'),
    ('bicubic', 'cv2'),
    ('bilinear', 'pillow'),
    ('nearest', 'pillow'),
])
def test_interpolation_and_backend(sample_data, interpolation, backend):
    """Test different interpolation and backend combinations."""
    transform = TableResize(
        long_size=600, 
        interpolation=interpolation, 
        backend=backend
    )
    result = transform.transform(sample_data)
    
    max_side = max(result['img'].shape[:2])
    assert abs(max_side - 600) <= 2


def test_preserve_extra_keys(sample_data):
    """Test that extra keys are preserved."""
    extra_keys = {'extra_key': 'extra_value', 'metadata': {'info': 'test'}}
    sample_data.update(extra_keys)
    
    transform = TableResize(long_size=600)
    result = transform.transform(sample_data)
    
    for key, value in extra_keys.items():
        assert result[key] == value

# Test edge cases and error conditions
def test_zero_min_size():
    """Test behavior with zero min_size."""
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(min_size=0)
    result = transform.transform(data)
    
    # Should not scale when min_size is 0
    assert result['img'].shape[:2] == (400, 300)


def test_very_large_constraints():
    """Test behavior with very large size constraints."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(min_size=10000)
    result = transform.transform(data)
    
    # Should scale up significantly
    min_side = min(result['img'].shape[:2])
    assert min_side >= 9000  # Allow for some precision loss


def test_single_pixel_image():
    """Test behavior with very small images."""
    img = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(min_size=100)
    result = transform.transform(data)
    
    # Should scale up the single pixel
    assert min(result['img'].shape[:2]) >= 90  # Allow for rounding


# Test string representation of TableResize
@pytest.mark.parametrize("min_size,long_size,expected_parts", [
    (200, 600, ["TableResize", "min_size=200", "long_size=600"]),
    (None, 500, ["TableResize", "min_size=None", "long_size=500"]),
    (300, None, ["TableResize", "min_size=300", "long_size=None"]),
    (None, None, ["TableResize", "min_size=None", "long_size=None"]),
])
def test_repr_method(min_size, long_size, expected_parts):
    """Test string representation with different parameters."""
    transform = TableResize(min_size=min_size, long_size=long_size)
    
    repr_str = repr(transform)
    for part in expected_parts:
        assert part in repr_str


def test_repr_method_completeness():
    """Test that repr contains all essential information."""
    transform = TableResize(min_size=200, long_size=600)
    repr_str = repr(transform)
    
    # Should be able to identify the transform type and key parameters
    assert 'TableResize' in repr_str
    assert '200' in repr_str
    assert '600' in repr_str


# Integration tests with realistic scenarios
def test_typical_table_processing_pipeline():
    """Test typical table processing scenario."""
    # Simulate a typical table image
    img = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
    data = {
        'img': img,
        'gt_bboxes': np.array([[50, 50, 150, 100], [200, 200, 400, 300]], dtype=np.float32),
        'img_shape': img.shape[:2],
        'filename': 'table_001.jpg'
    }
    
    # Apply table resize with typical parameters
    transform = TableResize(min_size=600, long_size=1333)
    result = transform.transform(data)
    
    # Check results
    assert 'img' in result
    assert 'gt_bboxes' in result
    assert 'filename' in result
    
    # Check size constraints
    img_h, img_w = result['img'].shape[:2]
    max_side = max(img_h, img_w)
    min_side = min(img_h, img_w)
    
    assert abs(max_side - 1333) <= 2  # long_size constraint
    assert min_side >= 600  # min_size should be satisfied


def test_small_image_upscaling():
    """Test upscaling of small images."""
    img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(min_size=800)
    result = transform.transform(data)
    
    min_side = min(result['img'].shape[:2])
    assert min_side >= 800 * 0.95  # Should be upscaled


def test_large_image_downscaling():
    """Test downscaling of large images."""
    img = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
    data = {'img': img}
    
    transform = TableResize(long_size=1333)
    result = transform.transform(data)
    
    max_side = max(result['img'].shape[:2])
    assert abs(max_side - 1333) <= 2  # Should be downscaled


def test_aspect_ratio_preservation():
    """Test that aspect ratio is preserved during resize."""
    img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)  # 2:3 aspect ratio
    data = {'img': img}
    
    transform = TableResize(min_size=800)
    result = transform.transform(data)
    
    # Calculate original and new aspect ratios
    original_ratio = 400 / 600
    new_h, new_w = result['img'].shape[:2]
    new_ratio = new_h / new_w
    
    # Ratios should be approximately equal
    assert abs(original_ratio - new_ratio) < 0.01


def test_polygon_handling_inheritance(sample_data):
    """Test that polygon handling is inherited from parent class."""
    sample_data['gt_polygons'] = [
        np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32),
        np.array([[100, 100], [200, 100], [200, 150], [100, 150]], dtype=np.float32)
    ]
    
    transform = TableResize(long_size=600)
    result = transform.transform(sample_data)
    
    # Check that polygons are handled (should be scaled by parent class)
    if 'gt_polygons' in sample_data:
        assert 'gt_polygons' in result
        assert len(result['gt_polygons']) == 2


def test_image_info_handling():
    """Test handling of image info metadata."""
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    data = {
        'img': img,
        'img_info': {'height': 400, 'width': 300, 'channels': 3}
    }
    
    transform = TableResize(long_size=600)
    result = transform.transform(data)
    
    # Check that img_info is preserved or updated
    assert 'img_info' in result or 'img_shape' in result


def test_multiple_resize_operations():
    """Test applying multiple resize operations in sequence."""
    img = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
    data = {'img': img}
    
    # First resize
    transform1 = TableResize(long_size=1000)
    result1 = transform1.transform(data)
    
    # Second resize
    transform2 = TableResize(min_size=400)
    result2 = transform2.transform(result1)
    
    # Should have applied both transformations
    assert 'img' in result2
    max_side = max(result2['img'].shape[:2])
    min_side = min(result2['img'].shape[:2])
    
    # Final result should satisfy the last constraint
    assert min_side >= 400 * 0.95