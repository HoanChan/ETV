#!/usr/bin/env python3
"""Test suite for BboxEncode transform."""

import pytest
import numpy as np
from datasets.transforms.bbox_encode import BboxEncode


@pytest.fixture
def sample_data():
    """Sample data with bboxes in xyxy format."""
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    return {
        'img': img,
        'img_shape': (400, 300, 3),
        'bboxes': np.array([[30, 40, 90, 80], [150, 100, 250, 200]], dtype=np.float32),
        'filename': 'test_image.jpg'
    }


@pytest.fixture
def edge_case_data():
    """Sample data with edge case bboxes."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return {
        'img': img,
        'img_shape': (100, 100, 3),
        'bboxes': np.array([[0, 0, 50, 50], [50, 50, 100, 100]], dtype=np.float32),
        'filename': 'edge_case.jpg'
    }


@pytest.fixture
def invalid_bbox_data():
    """Sample data with invalid bboxes (outside image bounds)."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return {
        'img': img,
        'img_shape': (100, 100, 3),
        'bboxes': np.array([[50, 50, 200, 200]], dtype=np.float32),  # Extends beyond image
        'filename': 'invalid.jpg'
    }


class TestBboxEncode:
    """Test cases for BboxEncode transform."""

    def test_initialization(self):
        """Test basic initialization."""
        transform = BboxEncode()
        assert isinstance(transform, BboxEncode)

    def test_basic_transform(self, sample_data):
        """Test basic bbox encoding transformation."""
        transform = BboxEncode()
        result = transform.transform(sample_data.copy())
        
        # Check that bboxes are available in result
        assert 'bboxes' in result
        
        # Check bbox format conversion (xyxy -> xywh)
        bbox = result['bboxes']
        assert bbox.shape == (2, 4)
        
        # Original: [[30, 40, 90, 80], [150, 100, 250, 200]]
        # Expected xywh: [[60, 60, 60, 40], [200, 150, 100, 100]] (center_x, center_y, width, height)
        # Normalized by img shape (400, 300): height=400, width=300
        expected_normalized = np.array([
            [60/300, 60/400, 60/300, 40/400],    # [0.2, 0.15, 0.2, 0.1]
            [200/300, 150/400, 100/300, 100/400] # [0.667, 0.375, 0.333, 0.25]
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(bbox, expected_normalized, decimal=3)

    @pytest.mark.parametrize("img_shape,bbox_xyxy,expected_xywh_norm", [
        # Test case 1: Simple square image
        ((100, 100, 3), [[20, 20, 60, 60]], [[0.4, 0.4, 0.4, 0.4]]),
        # Test case 2: Rectangular image - FIXED: y center = (20+80)/2=50, normalized = 50/200=0.25
        ((200, 100, 3), [[10, 20, 50, 80]], [[0.3, 0.25, 0.4, 0.3]]),
        # Test case 3: Multiple bboxes
        ((100, 100, 3), [[0, 0, 50, 50], [50, 50, 100, 100]],
         [[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]])
    ])
    def test_bbox_conversion_parametrized(self, img_shape, bbox_xyxy, expected_xywh_norm):
        """Test bbox conversion with different scenarios."""
        img = np.random.randint(0, 255, img_shape, dtype=np.uint8)
        data = {
            'img': img,
            'img_shape': img_shape,
            'bboxes': np.array(bbox_xyxy, dtype=np.float32),
            'filename': 'test.jpg'
        }
        
        transform = BboxEncode()
        result = transform.transform(data)
        
        expected = np.array(expected_xywh_norm, dtype=np.float32)
        np.testing.assert_array_almost_equal(result['bboxes'], expected, decimal=3)

    def test_bbox_validation_valid(self, sample_data):
        """Test bbox validation with valid bboxes."""
        transform = BboxEncode()
        result = transform.transform(sample_data.copy())
        
        # All bboxes should be valid (between 0 and 1)
        bbox = result['bboxes']
        assert np.all(bbox >= 0)
        assert np.all(bbox <= 1)

    def test_bbox_validation_invalid(self, invalid_bbox_data, capsys):
        """Test bbox validation with invalid bboxes."""
        transform = BboxEncode()
        
        # Should raise assertion error for invalid bboxes
        with pytest.raises(AssertionError) as excinfo:
            transform.transform(invalid_bbox_data.copy())
        assert "Box invalid" in str(excinfo.value)

    def test_edge_case_bboxes(self, edge_case_data):
        """Test edge case bboxes (at image boundaries)."""
        transform = BboxEncode()
        result = transform.transform(edge_case_data.copy())
        
        bbox = result['bboxes']
        # Should handle edge cases properly
        assert np.all(bbox >= 0)
        assert np.all(bbox <= 1)

    def test_empty_bbox_handling(self):
        """Test handling of empty bbox arrays."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        data = {
            'img': img,
            'img_shape': (100, 100, 3),
            'bboxes': np.array([], dtype=np.float32).reshape(0, 4),
            'filename': 'empty.jpg'
        }
        
        transform = BboxEncode()
        result = transform.transform(data)
        
        assert result['bboxes'].shape == (0, 4)

    def test_single_bbox(self):
        """Test with single bbox."""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        data = {
            'img': img,
            'img_shape': (200, 200, 3),
            'bboxes': np.array([[50, 50, 150, 150]], dtype=np.float32),
            'filename': 'single.jpg'
        }
        
        transform = BboxEncode()
        result = transform.transform(data)
        
        assert result['bboxes'].shape == (1, 4)
        # Center should be at (100, 100) -> normalized (0.5, 0.5)
        # Size should be (100, 100) -> normalized (0.5, 0.5)
        expected = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result['bboxes'], expected, decimal=3)

    def test_key_adjustment(self, sample_data):
        """Test that keys are properly adjusted in results dict."""
        
        transform = BboxEncode()
        result = transform.transform(sample_data.copy())
        
        # Top level should have bboxes
        assert 'bboxes' in result

    def test_preserve_other_keys(self, sample_data):
        """Test that other keys are preserved."""
        sample_data['extra_key'] = 'extra_value'
        
        transform = BboxEncode()
        result = transform.transform(sample_data.copy())
        
        assert result['extra_key'] == 'extra_value'
        assert result['filename'] == sample_data['filename']

    @pytest.mark.parametrize("bbox_values,should_be_valid", [
        # Valid cases
        ([[0.1, 0.1, 0.2, 0.2]], True),
        ([[0.0, 0.0, 1.0, 1.0]], True),
        ([[0.5, 0.5, 0.3, 0.3]], True),
        # Invalid cases (would be created by bbox extending beyond image)
        ([[-0.1, 0.1, 0.2, 0.2]], False),
        ([[0.1, 0.1, 1.1, 0.2]], False),
        ([[0.1, 0.1, 0.2, 1.1]], False),
    ])
    def test_bbox_validation_edge_cases(self, bbox_values, should_be_valid):
        """Test bbox validation with edge cases."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create bbox that will result in the test values after normalization
        # This is a bit artificial but tests the validation logic
        transform = BboxEncode()
        
        # Test the validation method directly
        bbox_array = np.array(bbox_values, dtype=np.float32)
        is_valid = transform._check_bbox_valid(bbox_array)
        assert is_valid == should_be_valid

    def test_repr_method(self):
        """Test string representation."""
        transform = BboxEncode()
        repr_str = repr(transform)
        assert 'BboxEncode' in repr_str

    def test_different_data_types(self):
        """Test with different bbox data types."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test with int32 bboxes
        data = {
            'img': img,
            'img_shape': (100, 100, 3),
            'bboxes': np.array([[10, 10, 50, 50]], dtype=np.int32),
            'filename': 'int_bbox.jpg'
        }
        
        transform = BboxEncode()
        result = transform.transform(data)
        
        # Should work and produce float32 normalized output
        assert result['bboxes'].dtype == np.float32
        assert result['bboxes'].shape == (1, 4)
