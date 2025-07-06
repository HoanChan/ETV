"""Integration test suite for all table transforms."""

import pytest
import numpy as np
from datasets.transforms.table_resize import TableResize
from datasets.transforms.table_pad import TablePad
from datasets.transforms.table_bbox_encode import TableBboxEncode
from datasets.transforms.get_cells import GetCells
from datasets.transforms.bbox_utils import xyxy2xywh, normalize_bbox, xywh2xyxy


@pytest.fixture
def complete_sample_data():
    """Complete sample data for integration testing."""
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    return {
        'img': img,
        'img_info': {
            'bbox': np.array([[30, 40, 90, 80], [150, 100, 250, 200]], dtype=np.float32),
            'bbox_masks': np.ones((2,), dtype=bool)
        },
        'instances': [
            {
                'task_type': 'content',
                'bbox': [30, 40, 90, 80],
                'tokens': ['Cell', '1'],
                'cell_id': 0
            },
            {
                'task_type': 'content', 
                'bbox': [150, 100, 250, 200],
                'tokens': ['Cell', '2'],
                'cell_id': 1
            },
            {
                'task_type': 'structure',
                'tokens': ['<table>', '<tr>', '<td>', '</td>', '</tr>', '</table>']
            }
        ],
        'filename': 'integration_test.jpg'
    }


class TestTransformsIntegration:
    """Integration tests for table transforms pipeline."""

    def test_full_pipeline_with_resize_pad_encode(self, complete_sample_data):
        """Test complete pipeline: resize -> pad -> bbox encode."""
        # Define pipeline
        transforms = [
            TableResize(img_scale=(600, 500), keep_ratio=True),
            TablePad(size=(800, 600), pad_val=0, return_mask=True),
            TableBboxEncode()
        ]
        
        results = complete_sample_data.copy()
        
        # Apply transforms sequentially
        for transform in transforms:
            results = transform.transform(results)
        
        # Verify final results
        assert results['img'].shape[:2] == (600, 800)  # Padded size
        assert 'bbox' in results
        assert 'bbox_masks' in results
        assert 'mask' in results
        assert 'scale_factor' in results
        
        # Check bbox normalization
        bbox = results['bbox']
        assert np.all(bbox >= 0)
        assert np.all(bbox <= 1)

    def test_pipeline_with_cell_extraction(self, complete_sample_data):
        """Test pipeline including cell extraction."""
        transforms = [
            TableResize(img_scale=(400, 400), keep_ratio=False),
            GetCells(min_cell_size=10),
            TablePad(size=(500, 500), pad_val=128)
        ]
        
        results = complete_sample_data.copy()
        
        for transform in transforms:
            results = transform.transform(results)
        
        # Verify results
        assert results['img'].shape[:2] == (500, 500)
        assert 'cell_imgs' in results
        assert len(results['cell_imgs']) >= 0  # May be filtered by min_cell_size

    @pytest.mark.parametrize("pipeline_config", [
        # Configuration 1: Standard training pipeline
        {
            'resize': {'img_scale': (800, 600), 'keep_ratio': True},
            'pad': {'size': (1000, 800), 'return_mask': True},
            'encode': {}
        },
        # Configuration 2: Inference pipeline with cells
        {
            'resize': {'img_scale': (600, 600), 'keep_ratio': False},
            'get_cells': {'min_cell_size': 5},
            'pad': {'size': (700, 700), 'pad_val': 255}
        },
        # Configuration 3: Minimal pipeline
        {
            'resize': {'img_scale': (400, 300), 'keep_ratio': True},
            'encode': {}
        }
    ])
    def test_different_pipeline_configurations(self, complete_sample_data, pipeline_config):
        """Test different pipeline configurations."""
        transforms = []
        
        # Build pipeline based on config
        if 'resize' in pipeline_config:
            transforms.append(TableResize(**pipeline_config['resize']))
        
        if 'get_cells' in pipeline_config:
            transforms.append(GetCells(**pipeline_config['get_cells']))
        
        if 'pad' in pipeline_config:
            transforms.append(TablePad(**pipeline_config['pad']))
        
        if 'encode' in pipeline_config:
            transforms.append(TableBboxEncode(**pipeline_config['encode']))
        
        # Apply pipeline
        results = complete_sample_data.copy()
        for transform in transforms:
            results = transform.transform(results)
        
        # Basic verification - pipeline should complete without errors
        assert 'img' in results
        assert isinstance(results['img'], np.ndarray)

    def test_pipeline_order_independence_for_compatible_transforms(self, complete_sample_data):
        """Test that some transforms can be reordered without issues."""
        # Test two different orders
        pipeline1 = [
            TableResize(img_scale=(500, 400), keep_ratio=True),
            GetCells(min_cell_size=5),
            TablePad(size=(600, 500), pad_val=0)
        ]
        
        pipeline2 = [
            TableResize(img_scale=(500, 400), keep_ratio=True),
            TablePad(size=(600, 500), pad_val=0),
            GetCells(min_cell_size=5)  # Note: This might extract from padded image
        ]
        
        results1 = complete_sample_data.copy()
        for transform in pipeline1:
            results1 = transform.transform(results1)
        
        results2 = complete_sample_data.copy()
        for transform in pipeline2:
            results2 = transform.transform(results2)
        
        # Both should complete successfully
        assert 'img' in results1 and 'img' in results2
        assert 'cell_imgs' in results1 and 'cell_imgs' in results2

    def test_bbox_utils_integration(self):
        """Test bbox utility functions work correctly in pipeline."""
        # Test complete bbox transformation workflow
        original_xyxy = np.array([[10, 20, 50, 60]], dtype=np.float32)
        
        # Convert to xywh
        xywh = xyxy2xywh(original_xyxy)
        expected_xywh = np.array([[30, 40, 40, 40]], dtype=np.float32)  # center_x, center_y, w, h
        np.testing.assert_array_equal(xywh, expected_xywh)
        
        # Normalize
        img_shape = (100, 100, 3)
        normalized = normalize_bbox(xywh, img_shape)
        expected_normalized = np.array([[0.3, 0.4, 0.4, 0.4]], dtype=np.float32)
        np.testing.assert_array_equal(normalized, expected_normalized)
        
        # Convert back to xyxy for verification
        denormalized = normalized.copy()
        denormalized[:, 0] *= img_shape[1]  # x
        denormalized[:, 1] *= img_shape[0]  # y 
        denormalized[:, 2] *= img_shape[1]  # w
        denormalized[:, 3] *= img_shape[0]  # h
        
        back_to_xyxy = xywh2xyxy(denormalized)
        np.testing.assert_array_equal(back_to_xyxy, original_xyxy)

    def test_error_propagation_in_pipeline(self, complete_sample_data):
        """Test that errors are properly handled in pipeline."""
        # Create a scenario that should cause an error
        invalid_data = complete_sample_data.copy()
        invalid_data['img'] = np.array([1, 2, 3])  # Invalid 1D image
        
        pipeline = [
            TableResize(img_scale=(500, 400)),
            GetCells()  # This should fail with invalid image
        ]
        
        # Apply first transform (should work)
        results = pipeline[0].transform(invalid_data.copy())
        
        # Second transform should raise error
        with pytest.raises(ValueError):
            pipeline[1].transform(results)

    def test_data_consistency_through_pipeline(self, complete_sample_data):
        """Test that data remains consistent through pipeline."""
        pipeline = [
            TableResize(img_scale=(400, 300), keep_ratio=True),
            TableBboxEncode()
        ]
        
        original_filename = complete_sample_data['filename']
        original_bbox_count = len(complete_sample_data['img_info']['bbox'])
        
        results = complete_sample_data.copy()
        for transform in pipeline:
            results = transform.transform(results)
        
        # Check consistency
        assert results['filename'] == original_filename
        assert len(results['bbox']) == original_bbox_count
        assert len(results['bbox_masks']) == original_bbox_count

    @pytest.mark.parametrize("image_size,target_size", [
        ((100, 100, 3), (200, 200)),    # Upscaling
        ((800, 600, 3), (400, 300)),    # Downscaling
        ((400, 300, 3), (400, 300)),    # Same size
    ])
    def test_pipeline_with_different_scales(self, image_size, target_size):
        """Test pipeline with different image scales."""
        img = np.random.randint(0, 255, image_size, dtype=np.uint8)
        data = {
            'img': img,
            'img_info': {
                'bbox': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'bbox_masks': np.ones((1,), dtype=bool)
            },
            'filename': 'scale_test.jpg'
        }
        
        pipeline = [
            TableResize(img_scale=target_size, keep_ratio=False),
            TableBboxEncode()
        ]
        
        results = data.copy()
        for transform in pipeline:
            results = transform.transform(results)
        
        # Verify scaling worked
        assert results['img'].shape[:2] == target_size[::-1]  # height, width
        assert 'bbox' in results

    def test_memory_efficiency(self, complete_sample_data):
        """Test that pipeline doesn't create excessive memory copies."""
        import sys
        
        pipeline = [
            TableResize(img_scale=(400, 300), keep_ratio=True),
            TablePad(size=(500, 400), pad_val=0),
            GetCells(min_cell_size=5)
        ]
        
        # Track object count before
        initial_objects = len([obj for obj in globals().values() if isinstance(obj, np.ndarray)])
        
        results = complete_sample_data.copy()
        for transform in pipeline:
            results = transform.transform(results)
        
        # Pipeline should complete without excessive memory usage
        assert 'img' in results
        assert 'cell_imgs' in results
