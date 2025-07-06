"""Test suite for TablePad transform."""

import pytest
import numpy as np
from datasets.transforms.table_pad import TablePad


@pytest.fixture
def sample_data():
    """Sample data with image."""
    img = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    return {
        'img': img,
        'filename': 'test_image.jpg'
    }


@pytest.fixture
def large_image_data():
    """Sample data with image larger than target size."""
    img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    return {
        'img': img,
        'filename': 'large_image.jpg'
    }


class TestTablePad:
    """Test cases for TablePad transform."""

    @pytest.mark.parametrize("size,size_divisor,should_fail", [
        ((500, 400), None, False),     # Valid: size only
        (None, 32, False),             # Valid: size_divisor only  
        ((500, 400), 32, True),        # Invalid: both specified
        (None, None, True),            # Invalid: neither specified
    ])
    def test_initialization_validation(self, size, size_divisor, should_fail):
        """Test initialization parameter validation."""
        if should_fail:
            with pytest.raises(AssertionError):
                TablePad(size=size, size_divisor=size_divisor)
        else:
            transform = TablePad(size=size, size_divisor=size_divisor)
            if size is not None:
                assert transform.size == size[::-1]  # Should be reversed
            else:
                assert transform.size is None
            assert transform.size_divisor == size_divisor

    @pytest.mark.parametrize("target_size,expected_shape", [
        ((500, 400), (400, 500)),      # (width, height) -> (height, width)
        ((800, 600), (600, 800)),
        ((1000, 1000), (1000, 1000))
    ])
    def test_fixed_size_padding(self, sample_data, target_size, expected_shape):
        """Test padding to fixed size."""
        transform = TablePad(size=target_size, pad_val=0)
        result = transform.transform(sample_data.copy())
        
        assert result['img'].shape[:2] == expected_shape
        assert 'pad_shape' in result
        assert result['pad_shape'][:2] == expected_shape
        assert result['pad_fixed_size'] == target_size[::-1]

    @pytest.mark.parametrize("pad_val", [0, 128, 255])
    def test_padding_values(self, sample_data, pad_val):
        """Test different padding values."""
        transform = TablePad(size=(500, 400), pad_val=pad_val)
        result = transform.transform(sample_data.copy())
        
        # Check that padding areas have the correct value
        padded_img = result['img']
        original_h, original_w = sample_data['img'].shape[:2]
        
        # Check bottom padding
        if padded_img.shape[0] > original_h:
            bottom_padding = padded_img[original_h:, :original_w, :]
            assert np.all(bottom_padding == pad_val)
        
        # Check right padding  
        if padded_img.shape[1] > original_w:
            right_padding = padded_img[:original_h, original_w:, :]
            assert np.all(right_padding == pad_val)

    @pytest.mark.parametrize("return_mask,mask_ratio", [
        (True, 2),                     # Return mask with stride 2
        (True, 4),                     # Return mask with stride 4
        (True, (2, 3)),               # Return mask with different strides
        (False, 2),                   # No mask
    ])
    def test_mask_generation(self, sample_data, return_mask, mask_ratio):
        """Test mask generation with different parameters."""
        transform = TablePad(
            size=(500, 400), 
            pad_val=0, 
            return_mask=return_mask,
            mask_ratio=mask_ratio
        )
        result = transform.transform(sample_data.copy())
        
        if return_mask:
            assert 'mask' in result
            assert result['mask'] is not None
            assert result['mask'].ndim == 3  # Should have channel dimension
            assert result['mask'].shape[0] == 1  # Single channel
            
            # Check mask dimensions based on ratio
            if isinstance(mask_ratio, int):
                expected_h = 400 // mask_ratio
                expected_w = 500 // mask_ratio
            else:
                expected_h = 400 // mask_ratio[0]
                expected_w = 500 // mask_ratio[1]
            
            # Allow for +1 due to ceiling division behavior in numpy slicing
            assert result['mask'].shape[1] == expected_h
            assert abs(result['mask'].shape[2] - expected_w) <= 1
        else:
            assert result['mask'] is None

    @pytest.mark.parametrize("keep_ratio", [True, False])
    def test_large_image_handling(self, large_image_data, keep_ratio):
        """Test handling of images larger than target size."""
        transform = TablePad(size=(500, 400), keep_ratio=keep_ratio)
        result = transform.transform(large_image_data.copy())
        
        # Image should be resized to fit within target size
        assert result['img'].shape[:2] == (400, 500)
        
        if keep_ratio:
            # With keep_ratio, image should maintain aspect ratio
            # One dimension should be smaller than target
            original_ratio = large_image_data['img'].shape[0] / large_image_data['img'].shape[1]
            target_ratio = 400 / 500
            
            if original_ratio > target_ratio:
                # Height-constrained
                assert result['img'].shape[0] == 400
            else:
                # Width-constrained  
                assert result['img'].shape[1] == 500

    def test_mask_ratio_validation(self, sample_data):
        """Test mask ratio validation."""
        # Test invalid mask ratio type
        with pytest.raises(NotImplementedError):
            transform = TablePad(size=(500, 400), return_mask=True, mask_ratio="invalid")
            transform.transform(sample_data.copy())

    def test_size_divisor_not_implemented(self, sample_data):
        """Test that size_divisor raises NotImplementedError."""
        transform = TablePad(size_divisor=32)
        
        with pytest.raises(NotImplementedError):
            transform.transform(sample_data.copy())

    @pytest.mark.parametrize("img_shape,target_size", [
        ((100, 80, 3), (200, 150)),    # Smaller image
        ((400, 300, 1), (500, 400)),   # Grayscale image
        ((200, 200, 4), (300, 300)),   # RGBA image
    ])
    def test_different_image_formats(self, img_shape, target_size):
        """Test padding with different image formats."""
        img = np.random.randint(0, 255, img_shape, dtype=np.uint8)
        data = {'img': img}
        
        transform = TablePad(size=target_size, pad_val=0)
        result = transform.transform(data)
        
        expected_shape = target_size[::-1] + (img_shape[2],)
        assert result['img'].shape == expected_shape

    def test_invalid_size_type(self, sample_data):
        """Test error with invalid size type."""
        with pytest.raises(TypeError):
            transform = TablePad(size=500)  # Should be tuple, not int

    def test_exact_size_match(self, sample_data):
        """Test when image size exactly matches target size."""
        # Modify sample data to have exact target size
        sample_data['img'] = np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8)
        
        transform = TablePad(size=(500, 400), pad_val=0)
        result = transform.transform(sample_data.copy())
        
        # Should still work and maintain size
        assert result['img'].shape[:2] == (400, 500)

    def test_transform_preserves_other_keys(self, sample_data):
        """Test that transform preserves other keys in results."""
        sample_data['extra_key'] = 'extra_value'
        transform = TablePad(size=(500, 400))
        result = transform.transform(sample_data)
        
        assert result['extra_key'] == 'extra_value'
        assert result['filename'] == sample_data['filename']

    def test_pad_result_keys(self, sample_data):
        """Test that all expected keys are added to results."""
        transform = TablePad(size=(500, 400), return_mask=True)
        result = transform.transform(sample_data)
        
        required_keys = ['img', 'mask', 'pad_shape', 'pad_fixed_size', 'pad_size_divisor']
        for key in required_keys:
            assert key in result

    def test_repr_method(self):
        """Test string representation."""
        transform = TablePad(
            size=(500, 400),
            pad_val=128,
            keep_ratio=True,
            return_mask=True
        )
        
        repr_str = repr(transform)
        assert 'TablePad' in repr_str
        assert 'size=(400, 500)' in repr_str  # Should be reversed
        assert 'pad_val=128' in repr_str
        assert 'keep_ratio=True' in repr_str
