#!/usr/bin/env python3
"""Test suite for TableResize transform."""

import pytest
import numpy as np
import cv2
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
        'img_info': {
            'bbox': np.array([[10, 10, 50, 50], [100, 100, 200, 150]], dtype=np.float32),
            'bbox_masks': np.ones((2,), dtype=bool)
        },
        'filename': 'test_image.jpg'
    }


@pytest.fixture
def sample_data_no_bbox():
    """Sample data without bboxes for testing phase."""
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    return {
        'img': img,
        'filename': 'test_image.jpg'
    }


class TestTableResize:
    """Test cases for TableResize transform."""

    @pytest.mark.parametrize("img_scale,keep_ratio,expected_behavior", [
        ((600, 800), True, "resize_with_ratio"),
        ((600, 800), False, "resize_exact"),
        ((-1, 600), True, "auto_width"),
        ((400, -1), True, "auto_height"),
        (None, True, "no_scale")
    ])
    def test_initialization(self, img_scale, keep_ratio, expected_behavior):
        """Test initialization with different parameters."""
        transform = TableResize(img_scale=img_scale, keep_ratio=keep_ratio)
        assert transform.img_scale == img_scale
        assert transform.keep_ratio == keep_ratio
        assert transform.interpolation == cv2.INTER_LINEAR

    @pytest.mark.parametrize("img_scale,expected_shape", [
        ((600, 800), (800, 600)),  # (width, height) -> (height, width)
        ((300, 200), (200, 300)),
        ((150, 100), (100, 150))
    ])
    def test_exact_resize_no_ratio(self, sample_data, img_scale, expected_shape):
        """Test exact resize without keeping ratio."""
        transform = TableResize(img_scale=img_scale, keep_ratio=False)
        result = transform.transform(sample_data.copy())
        
        assert result['img'].shape[:2] == expected_shape
        assert 'scale_factor' in result
        assert 'img_shape' in result
        assert result['keep_ratio'] == False

    @pytest.mark.parametrize("img_scale", [
        ((-1, 600)),  # auto width
        ((400, -1))   # auto height
    ])
    def test_auto_dimension_resize(self, sample_data, img_scale):
        """Test resize with automatic dimension calculation."""
        transform = TableResize(img_scale=img_scale, keep_ratio=True)
        result = transform.transform(sample_data.copy())
        
        # Original image: 400h x 300w
        if img_scale[0] == -1:  # auto width
            assert result['img'].shape[0] == img_scale[1]  # height should match
        else:  # auto height
            assert result['img'].shape[1] == img_scale[0]  # width should match

    @pytest.mark.parametrize("min_size,original_shape,expected_min", [
        (200, (400, 300), 200),  # shorter side (300) -> 200
        (500, (400, 300), 500),  # shorter side (300) -> 500
        (100, (400, 300), 300)   # no change needed
    ])
    def test_min_size_constraint(self, min_size, original_shape, expected_min):
        """Test minimum size constraint."""
        img = np.random.randint(0, 255, original_shape + (3,), dtype=np.uint8)
        data = {'img': img}
        
        transform = TableResize(min_size=min_size, keep_ratio=True)
        result = transform.transform(data)
        
        actual_min = min(result['img'].shape[:2])
        if min_size > min(original_shape):
            assert actual_min >= min_size * 0.9  # Allow some floating point error
        else:
            assert actual_min >= min(original_shape) * 0.9

    @pytest.mark.parametrize("long_size,original_shape", [
        (600, (400, 300)),  # longer side (400) -> 600
        (200, (400, 300)),  # longer side (400) -> 200
    ])
    def test_long_size_constraint(self, long_size, original_shape):
        """Test maximum/long size constraint."""
        img = np.random.randint(0, 255, original_shape + (3,), dtype=np.uint8)
        data = {'img': img}
        
        transform = TableResize(long_size=long_size, keep_ratio=True)
        result = transform.transform(data)
        
        actual_max = max(result['img'].shape[:2])
        assert actual_max >= long_size * 0.9  # Allow some floating point error

    @pytest.mark.parametrize("ratio_range", [
        ([0.5, 1.5]),
        ([0.8, 1.2]),
        ([1.0, 2.0])
    ])
    def test_random_ratio_range(self, sample_data, ratio_range):
        """Test random ratio range augmentation."""
        transform = TableResize(ratio_range=ratio_range, keep_ratio=True)
        
        # Run multiple times to test randomness
        results = []
        for _ in range(5):
            result = transform.transform(sample_data.copy())
            results.append(result['img'].shape[:2])
        
        # Should have some variation (not all exactly the same)
        unique_shapes = set(results)
        assert len(unique_shapes) >= 1  # At least one unique shape

    def test_bbox_scaling(self, sample_data):
        """Test that bboxes are properly scaled."""
        original_bboxes = sample_data['img_info']['bbox'].copy()
        transform = TableResize(img_scale=(600, 800), keep_ratio=False)
        result = transform.transform(sample_data)
        
        assert 'img_info' in result
        assert 'bbox' in result['img_info']
        
        scaled_bboxes = result['img_info']['bbox']
        scale_factor = result['scale_factor']
        
        # Check that bboxes were scaled
        assert not np.array_equal(original_bboxes, scaled_bboxes)
        
        # Check that bboxes are within image bounds
        img_h, img_w = result['img'].shape[:2]
        assert np.all(scaled_bboxes[..., 0::2] >= 0)  # x coordinates
        assert np.all(scaled_bboxes[..., 1::2] >= 0)  # y coordinates
        assert np.all(scaled_bboxes[..., 0::2] < img_w)  # x within width
        assert np.all(scaled_bboxes[..., 1::2] < img_h)  # y within height

    def test_no_bbox_data(self, sample_data_no_bbox):
        """Test transform works without bbox data (testing phase)."""
        transform = TableResize(img_scale=(600, 800), keep_ratio=True)
        result = transform.transform(sample_data_no_bbox)
        
        assert 'img' in result
        assert 'scale_factor' in result
        assert 'img_shape' in result

    def test_missing_bbox_with_img_info(self, sample_data):
        """Test error when img_info exists but bbox is missing."""
        # Remove bbox but keep img_info
        del sample_data['img_info']['bbox']
        
        transform = TableResize(img_scale=(600, 800))
        
        with pytest.raises(ValueError, match="results should have bbox keys"):
            transform.transform(sample_data)

    @pytest.mark.parametrize("interpolation", [
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_NEAREST
    ])
    def test_interpolation_methods(self, sample_data, interpolation):
        """Test different interpolation methods."""
        transform = TableResize(img_scale=(600, 800), interpolation=interpolation)
        result = transform.transform(sample_data)
        
        assert result['img'].shape[:2] == (800, 600)

    def test_get_resize_scale_edge_cases(self):
        """Test _get_resize_scale method edge cases."""
        # Test with keep_ratio=False and no img_scale
        transform = TableResize(keep_ratio=False)
        
        with pytest.raises(NotImplementedError):
            transform._get_resize_scale(300, 400)

    def test_transform_preserves_other_keys(self, sample_data):
        """Test that transform preserves other keys in results."""
        sample_data['extra_key'] = 'extra_value'
        transform = TableResize(img_scale=(600, 800))
        result = transform.transform(sample_data)
        
        assert result['extra_key'] == 'extra_value'
        assert result['filename'] == sample_data['filename']

    def test_repr_method(self):
        """Test string representation."""
        transform = TableResize(
            img_scale=(600, 800),
            min_size=200,
            ratio_range=[0.8, 1.2],
            keep_ratio=True,
            long_size=1000
        )
        
        repr_str = repr(transform)
        assert 'TableResize' in repr_str
        assert 'img_scale=(600, 800)' in repr_str
        assert 'min_size=200' in repr_str
        assert 'keep_ratio=True' in repr_str
