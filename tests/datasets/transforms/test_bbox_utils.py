"""Test suite for bbox utility functions."""

import pytest
import numpy as np
from datasets.transforms.bbox_utils import (
    xyxy2xywh, normalize_bbox, xywh2xyxy, 
    build_empty_bbox_mask, get_bbox_nums, align_bbox_mask, build_bbox_mask
)


# =============================================================================
# Tests for xyxy2xywh function
# =============================================================================

@pytest.mark.parametrize("xyxy_input,expected_xywh", [
    # Single bbox
    ([[10, 20, 50, 60]], [[30, 40, 40, 40]]),  # center_x, center_y, width, height
    # Multiple bboxes
    ([[0, 0, 100, 100], [50, 50, 150, 150]], 
     [[50, 50, 100, 100], [100, 100, 100, 100]]),
    # Edge case: zero-size bbox
    ([[10, 10, 10, 10]], [[10, 10, 0, 0]]),
    # Negative coordinates
    ([[-10, -20, 30, 40]], [[10, 10, 40, 60]]),
])
def test_xyxy2xywh_conversion(xyxy_input, expected_xywh):
    """Test xyxy to xywh conversion."""
    xyxy = np.array(xyxy_input, dtype=np.float32)
    expected = np.array(expected_xywh, dtype=np.float32)
    
    result = xyxy2xywh(xyxy)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [
    (1, 4),      # Single bbox
    (5, 4),      # Multiple bboxes
    (10, 4),     # Many bboxes
    (0, 4),      # Empty array
])
def test_xyxy2xywh_shapes(shape):
    """Test xyxy2xywh works with different array shapes."""
    if shape[0] == 0:
        # Empty array case
        xyxy = np.array([], dtype=np.float32).reshape(0, 4)
    else:
        xyxy = np.random.randint(0, 100, shape).astype(np.float32)
        # Ensure valid xyxy format (x1 <= x2, y1 <= y2)
        xyxy[:, 2] = np.maximum(xyxy[:, 2], xyxy[:, 0] + 1)
        xyxy[:, 3] = np.maximum(xyxy[:, 3], xyxy[:, 1] + 1)
    
    result = xyxy2xywh(xyxy)
    assert result.shape == shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_xyxy2xywh_dtypes(dtype):
    """Test xyxy2xywh works with different data types."""
    xyxy = np.array([[10, 20, 50, 60]], dtype=dtype)
    result = xyxy2xywh(xyxy)
    expected = np.array([[30, 40, 40, 40]], dtype=dtype)
    np.testing.assert_array_equal(result, expected)


def test_xyxy2xywh_batch_dimensions():
    """Test xyxy2xywh works with additional dimensions (batch processing)."""
    batch_xyxy = np.array([
        [[10, 20, 50, 60], [70, 80, 110, 120]],
        [[0, 0, 40, 40], [50, 50, 90, 90]]
    ], dtype=np.float32)  # Shape: (2, 2, 4)
    
    result = xyxy2xywh(batch_xyxy)
    assert result.shape == (2, 2, 4)


# =============================================================================
# Tests for xywh2xyxy function
# =============================================================================

@pytest.mark.parametrize("xywh_input,expected_xyxy", [
    # Single bbox
    ([[30, 40, 40, 40]], [[10, 20, 50, 60]]),  # x1, y1, x2, y2
    # Multiple bboxes
    ([[50, 50, 100, 100], [100, 100, 100, 100]], 
     [[0, 0, 100, 100], [50, 50, 150, 150]]),
    # Edge case: zero-size bbox
    ([[10, 10, 0, 0]], [[10, 10, 10, 10]]),
    # Center at origin
    ([[0, 0, 20, 30]], [[-10, -15, 10, 15]]),
])
def test_xywh2xyxy_conversion(xywh_input, expected_xyxy):
    """Test xywh to xyxy conversion."""
    xywh = np.array(xywh_input, dtype=np.float32)
    expected = np.array(expected_xyxy, dtype=np.float32)
    
    result = xywh2xyxy(xywh)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("bbox_format", ["xyxy", "xywh"])
def test_bidirectional_conversion(bbox_format):
    """Test that conversion is bidirectional (reversible)."""
    if bbox_format == "xyxy":
        original = np.array([[10, 20, 50, 60], [100, 100, 200, 150]], dtype=np.float32)
        converted = xyxy2xywh(original)
        back_converted = xywh2xyxy(converted)
    else:  # xywh
        original = np.array([[30, 40, 40, 40], [150, 125, 100, 50]], dtype=np.float32)
        converted = xywh2xyxy(original)
        back_converted = xyxy2xywh(converted)
    
    np.testing.assert_array_almost_equal(original, back_converted, decimal=6)


# =============================================================================
# Tests for normalize_bbox function
# =============================================================================

@pytest.mark.parametrize("img_shape,bbox_input,expected_normalized", [
    # Test case 1: Square image, square bbox
    ((100, 100, 3), [[25, 25, 50, 50]], [[0.25, 0.25, 0.5, 0.5]]),
    # Test case 2: Rectangular image, rectangular bbox
    ((200, 100, 3), [[20, 40, 30, 60]], [[0.2, 0.2, 0.3, 0.3]]),
    # Test case 3: Multiple bboxes
    ((100, 100, 3), [[10, 20, 20, 40], [50, 60, 30, 20]], 
     [[0.1, 0.2, 0.2, 0.4], [0.5, 0.6, 0.3, 0.2]]),
    # Test case 4: Full image bbox
    ((200, 300, 3), [[0, 0, 300, 200]], [[0.0, 0.0, 1.0, 1.0]]),
])
def test_normalize_bbox_basic(img_shape, bbox_input, expected_normalized):
    """Test bbox normalization with different image shapes."""
    bbox = np.array(bbox_input, dtype=np.float32)
    expected = np.array(expected_normalized, dtype=np.float32)
    
    result = normalize_bbox(bbox, img_shape)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)


def test_normalize_bbox_does_not_modify_original():
    """Test that normalize_bbox doesn't modify the original array."""
    original_bbox = np.array([[10, 20, 30, 40]], dtype=np.float32)
    bbox_copy = original_bbox.copy()
    img_shape = (100, 100, 3)
    
    normalized = normalize_bbox(bbox_copy, img_shape)
    
    # Original should be unchanged
    np.testing.assert_array_equal(original_bbox, bbox_copy)
    # But normalized should be different
    assert not np.array_equal(original_bbox, normalized)


@pytest.mark.parametrize("img_shape,bbox_input,expected", [
    # Very small image
    ((1, 1, 3), [[0, 0, 1, 1]], [[0.0, 0.0, 1.0, 1.0]]),
    # Very large image  
    ((10000, 10000, 3), [[1000, 2000, 3000, 4000]], [[0.1, 0.2, 0.3, 0.4]]),
])
def test_normalize_bbox_edge_cases(img_shape, bbox_input, expected):
    """Test normalize_bbox with edge cases."""
    bbox = np.array(bbox_input, dtype=np.float32)
    expected = np.array(expected, dtype=np.float32)
    result = normalize_bbox(bbox, img_shape)
    np.testing.assert_array_equal(result, expected)


def test_normalize_bbox_batch_dimensions():
    """Test normalize_bbox with batch dimensions."""
    batch_bbox = np.array([
        [[10, 20, 30, 40], [50, 60, 70, 80]],
        [[5, 10, 15, 20], [25, 30, 35, 40]]
    ], dtype=np.float32)  # Shape: (2, 2, 4)
    
    img_shape = (100, 100, 3)
    result = normalize_bbox(batch_bbox, img_shape)
    
    assert result.shape == (2, 2, 4)
    
    # Check first batch item, first bbox
    expected_first = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    np.testing.assert_array_almost_equal(result[0, 0], expected_first, decimal=6)


# =============================================================================
# Tests for build_empty_bbox_mask function
# =============================================================================

@pytest.mark.parametrize("bboxes_input,expected_mask", [
    # All non-empty bboxes
    ([[10, 20, 30, 40], [50, 60, 70, 80]], [1, 1]),
    # Mixed empty and non-empty
    ([[10, 20, 30, 40], [0, 0, 0, 0], [50, 60, 70, 80]], [1, 0, 1]),
    # All empty bboxes
    ([[0, 0, 0, 0], [0, 0, 0, 0]], [0, 0]),
    # Single non-empty bbox
    ([[15, 25, 35, 45]], [1]),
    # Single empty bbox
    ([[0, 0, 0, 0]], [0]),
    # Empty list
    ([], []),
])
def test_build_empty_bbox_mask(bboxes_input, expected_mask):
    """Test build_empty_bbox_mask function."""
    result = build_empty_bbox_mask(bboxes_input)
    assert result == expected_mask


# =============================================================================
# Tests for get_bbox_nums function
# =============================================================================

@pytest.mark.parametrize("tokens,expected_count", [
    # Basic td tokens - Note: '<td' pattern matches both '<td></td>' and '<td'
    (['<td></td>', '<td', '</td>'], 2),
    # Mix of bbox and non-bbox tokens
    (['<table>', '<tr>', '<td></td>', '<td', '</tr>', '</table>'], 2),
    # Empty bbox tokens
    (['<eb></eb>', '<eb1></eb1>', '<eb2></eb2>'], 3),
    # Mixed eb tokens
    (['<eb3></eb3>', '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>'], 4),
    # Higher number eb tokens
    (['<eb7></eb7>', '<eb8></eb8>', '<eb9></eb9>', '<eb10></eb10>'], 4),
    # No bbox tokens
    (['<table>', '<tr>', '</tr>', '</table>'], 0),
    # All types mixed
    (['<td></td>', '<eb></eb>', '<eb1></eb1>', '<td', '<eb2></eb2>'], 5),
    # Empty list
    ([], 0),
])
def test_get_bbox_nums(tokens, expected_count):
    """Test get_bbox_nums function."""
    result = get_bbox_nums(tokens)
    assert result == expected_count


# =============================================================================
# Tests for align_bbox_mask function
# =============================================================================

@pytest.mark.parametrize("bboxes,empty_mask,tokens,expected_bbox,expected_mask", [
    # Simple case with td tokens - matches '<td></td>' and '<td'
    (
        [[10, 20, 30, 40], [50, 60, 70, 80]], 
        [1, 1],
        ['<table>', '<td></td>', '<td', '</table>'],
        [[0., 0., 0., 0.], [10, 20, 30, 40], [50, 60, 70, 80], [0., 0., 0., 0.]],
        [1, 1, 1, 1]
    ),
    # Case with empty bbox
    (
        [[10, 20, 30, 40], [0, 0, 0, 0]], 
        [1, 0],
        ['<tr>', '<td></td>', '<td', '</tr>'],
        [[0., 0., 0., 0.], [10, 20, 30, 40], [0, 0, 0, 0], [0., 0., 0., 0.]],
        [1, 1, 0, 1]
    ),
    # Case with eb tokens
    (
        [[15, 25, 35, 45]], 
        [1],
        ['<table>', '<eb></eb>', '</table>'],
        [[0., 0., 0., 0.], [15, 25, 35, 45], [0., 0., 0., 0.]],
        [1, 1, 1]
    ),
])
def test_align_bbox_mask(bboxes, empty_mask, tokens, expected_bbox, expected_mask):
    """Test align_bbox_mask function."""
    result_bbox, result_mask = align_bbox_mask(bboxes, empty_mask, tokens)
    assert result_bbox == expected_bbox
    assert result_mask == expected_mask


# =============================================================================
# Tests for build_bbox_mask function
# =============================================================================

@pytest.mark.parametrize("tokens,expected_mask", [
    # Basic td tokens - build_bbox_mask only matches '<td></td>', '<td', '<eb></eb>'
    (['<table>', '<td></td>', '<td', '</table>'], [0, 1, 1, 0]),
    # With eb tokens
    (['<tr>', '<td></td>', '<eb></eb>', '</tr>'], [0, 1, 1, 0]),
    # No bbox tokens
    (['<table>', '<tr>', '</tr>', '</table>'], [0, 0, 0, 0]),
    # All bbox tokens - Note: '<td>' (with closing >) is NOT in the pattern
    (['<td></td>', '<td', '<eb></eb>'], [1, 1, 1]),
    # Empty list
    ([], []),
    # Mixed with other tokens - '<td>' (with >) is NOT matched
    (['<thead>', '<td></td>', '<tbody>', '<td', '<eb></eb>', '</tbody>'], [0, 1, 0, 1, 1, 0]),
])
def test_build_bbox_mask(tokens, expected_mask):
    """Test build_bbox_mask function."""
    result = build_bbox_mask(tokens)
    expected = np.array(expected_mask)
    np.testing.assert_array_equal(result, expected)
