"""Test suite for bbox utility functions."""

import pytest
import numpy as np
from datasets.transforms.bbox_utils import xyxy2xywh, normalize_bbox, xywh2xyxy


class TestBboxUtils:
    """Test cases for bbox utility functions."""

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
    def test_xyxy2xywh(self, xyxy_input, expected_xywh):
        """Test xyxy to xywh conversion."""
        xyxy = np.array(xyxy_input, dtype=np.float32)
        expected = np.array(expected_xywh, dtype=np.float32)
        
        result = xyxy2xywh(xyxy)
        np.testing.assert_array_equal(result, expected)

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
    def test_xywh2xyxy(self, xywh_input, expected_xyxy):
        """Test xywh to xyxy conversion."""
        xywh = np.array(xywh_input, dtype=np.float32)
        expected = np.array(expected_xyxy, dtype=np.float32)
        
        result = xywh2xyxy(xywh)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("bbox_format", ["xyxy", "xywh"])
    def test_bidirectional_conversion(self, bbox_format):
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
    def test_normalize_bbox(self, img_shape, bbox_input, expected_normalized):
        """Test bbox normalization with different image shapes."""
        bbox = np.array(bbox_input, dtype=np.float32)
        expected = np.array(expected_normalized, dtype=np.float32)
        
        result = normalize_bbox(bbox, img_shape)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_normalize_bbox_does_not_modify_original(self):
        """Test that normalize_bbox doesn't modify the original array."""
        original_bbox = np.array([[10, 20, 30, 40]], dtype=np.float32)
        bbox_copy = original_bbox.copy()
        img_shape = (100, 100, 3)
        
        normalized = normalize_bbox(bbox_copy, img_shape)
        
        # Original should be unchanged
        np.testing.assert_array_equal(original_bbox, bbox_copy)
        # But normalized should be different
        assert not np.array_equal(original_bbox, normalized)

    @pytest.mark.parametrize("shape", [
        (1, 4),      # Single bbox
        (5, 4),      # Multiple bboxes
        (10, 4),     # Many bboxes
        (0, 4),      # Empty array
    ])
    def test_function_shapes(self, shape):
        """Test functions work with different array shapes."""
        if shape[0] == 0:
            # Empty array case
            xyxy = np.array([], dtype=np.float32).reshape(0, 4)
        else:
            xyxy = np.random.randint(0, 100, shape).astype(np.float32)
            # Ensure valid xyxy format (x1 <= x2, y1 <= y2)
            xyxy[:, 2] = np.maximum(xyxy[:, 2], xyxy[:, 0] + 1)
            xyxy[:, 3] = np.maximum(xyxy[:, 3], xyxy[:, 1] + 1)
        
        # Test conversions
        xywh = xyxy2xywh(xyxy)
        assert xywh.shape == shape
        
        back_to_xyxy = xywh2xyxy(xywh)
        assert back_to_xyxy.shape == shape
        
        if shape[0] > 0:
            np.testing.assert_array_almost_equal(xyxy, back_to_xyxy, decimal=6)

    def test_normalize_bbox_edge_cases(self):
        """Test normalize_bbox with edge cases."""
        # Test with very small image
        small_img_shape = (1, 1, 3)
        bbox = np.array([[0, 0, 1, 1]], dtype=np.float32)
        result = normalize_bbox(bbox, small_img_shape)
        expected = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        
        # Test with very large image
        large_img_shape = (10000, 10000, 3)
        bbox = np.array([[1000, 2000, 3000, 4000]], dtype=np.float32)
        result = normalize_bbox(bbox, large_img_shape)
        expected = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    def test_functions_with_different_dtypes(self, dtype):
        """Test functions work with different data types."""
        xyxy = np.array([[10, 20, 50, 60]], dtype=dtype)
        
        # Test xyxy2xywh
        xywh = xyxy2xywh(xyxy)
        expected_xywh = np.array([[30, 40, 40, 40]], dtype=dtype)
        np.testing.assert_array_equal(xywh, expected_xywh)
        
        # Test xywh2xyxy
        back_to_xyxy = xywh2xyxy(xywh)
        np.testing.assert_array_equal(xyxy, back_to_xyxy)

    def test_functions_preserve_additional_dimensions(self):
        """Test functions work with additional dimensions (batch processing)."""
        # Test with batch dimension
        batch_xyxy = np.array([
            [[10, 20, 50, 60], [70, 80, 110, 120]],
            [[0, 0, 40, 40], [50, 50, 90, 90]]
        ], dtype=np.float32)  # Shape: (2, 2, 4)
        
        batch_xywh = xyxy2xywh(batch_xyxy)
        assert batch_xywh.shape == (2, 2, 4)
        
        # Test round-trip
        back_to_xyxy = xywh2xyxy(batch_xywh)
        np.testing.assert_array_almost_equal(batch_xyxy, back_to_xyxy, decimal=6)

    def test_normalize_bbox_with_batch_dimension(self):
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
