# Copyright (c) Lê Hoàn Chân. All rights reserved.
"""
Comprehensive test suite for all loss functions in the ETV project.

This module provides integration tests and parametrized tests for all
loss functions to ensure consistency and correctness across the codebase.
"""

import pytest
import torch
import torch.nn as nn
import inspect

from models.losses.master_tf_loss import MASTERTFLoss
from models.losses.table_l1_loss import TableL1Loss


def _create_loss_instance(loss_class):
    """Create a loss instance with default parameters."""
    if loss_class == MASTERTFLoss:
        return loss_class()
    elif loss_class == TableL1Loss:
        return loss_class()
    else:
        return loss_class()


# Define all loss classes in one place for consistency
ALL_LOSS_CLASSES = [MASTERTFLoss, TableL1Loss]


@pytest.fixture
def all_loss_classes():
    """Get all available loss classes."""
    return ALL_LOSS_CLASSES


@pytest.mark.parametrize("loss_class", ALL_LOSS_CLASSES)
def test_loss_inheritance(loss_class):
    """Test that all loss classes inherit from nn.Module."""
    loss_instance = _create_loss_instance(loss_class)
    assert isinstance(loss_instance, nn.Module)


@pytest.mark.parametrize("loss_class", ALL_LOSS_CLASSES)
def test_loss_has_forward_method(loss_class):
    """Test that all loss classes have forward method."""
    assert hasattr(loss_class, 'forward')
    assert callable(getattr(loss_class, 'forward'))


@pytest.mark.parametrize("loss_class", ALL_LOSS_CLASSES)
def test_loss_forward_signature(loss_class):
    """Test that forward methods have consistent signatures."""
    forward_method = getattr(loss_class, 'forward')
    sig = inspect.signature(forward_method)
    
    # All forward methods should have outputs and targets_dict parameters
    assert 'outputs' in sig.parameters
    assert 'targets_dict' in sig.parameters


@pytest.mark.parametrize("loss_class,config", [
    (MASTERTFLoss, {'flatten': True, 'reduction': 'mean'}),
    (MASTERTFLoss, {'flatten': False, 'reduction': 'sum'}),
    (MASTERTFLoss, {'flatten': True, 'reduction': 'none', 'ignore_index': 0}),
])
def test_master_tf_loss_configurations(loss_class, config):
    """Test MASTERTFLoss with different configurations."""
    batch_size, seq_length, num_classes = 2, 10, 5
    
    outputs = torch.randn(batch_size, seq_length, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
    targets_dict = {'padded_targets': targets}
    
    loss_fn = loss_class(**config)
    loss = loss_fn(outputs, targets_dict)
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize("config", [
    {'lambda_horizon': 1.0, 'lambda_vertical': 1.0},
    {'lambda_horizon': 0.5, 'lambda_vertical': 2.0},
    {'lambda_horizon': 2.0, 'lambda_vertical': 0.5},
])
def test_table_l1_loss_configurations(config):
    """Test TableL1Loss with different weight configurations."""
    batch_size, seq_length = 2, 10
    
    outputs = torch.randn(batch_size, seq_length, 4)
    bbox = torch.randn(batch_size, seq_length + 1, 4)
    bbox_masks = torch.ones(batch_size, seq_length + 1)
    
    targets_dict = {
        'bbox': bbox,
        'bbox_masks': bbox_masks
    }
    
    loss_fn = TableL1Loss(**config)
    losses = loss_fn(outputs, targets_dict)
    
    assert isinstance(losses, dict)
    assert 'horizon_bbox_loss' in losses
    assert 'vertical_bbox_loss' in losses
    assert torch.isfinite(losses['horizon_bbox_loss'])
    assert torch.isfinite(losses['vertical_bbox_loss'])


@pytest.mark.parametrize("loss_class", ALL_LOSS_CLASSES)
@pytest.mark.parametrize(
    "batch_size,seq_length",
    [
        (1, 5),     # Small
        (2, 10),    # Medium
        (4, 20),    # Large
    ]
)
def test_all_losses_scale_with_batch_size(loss_class, batch_size, seq_length):
    """Test that all losses handle different batch sizes correctly."""
    if loss_class == MASTERTFLoss:
        num_classes = 5
        outputs = torch.randn(batch_size, seq_length, num_classes)
        targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
        targets_dict = {'padded_targets': targets}
        
        loss_fn = loss_class(reduction='mean')
        result = loss_fn(outputs, targets_dict)
        assert torch.isfinite(result)
    
    elif loss_class == TableL1Loss:
        outputs = torch.randn(batch_size, seq_length, 4)
        bbox = torch.randn(batch_size, seq_length + 1, 4)
        bbox_masks = torch.ones(batch_size, seq_length + 1)
        targets_dict = {'bbox': bbox, 'bbox_masks': bbox_masks}
        
        loss_fn = loss_class()
        result = loss_fn(outputs, targets_dict)
        assert torch.isfinite(result['horizon_bbox_loss'])
        assert torch.isfinite(result['vertical_bbox_loss'])


@pytest.mark.parametrize("loss_class", ALL_LOSS_CLASSES)
def test_losses_numerical_stability(loss_class):
    """Test numerical stability of all losses."""
    batch_size, seq_length = 2, 5
    
    if loss_class == MASTERTFLoss:
        # Test with extreme logits
        num_classes = 3
        extreme_outputs = torch.tensor([[[1e6, -1e6, 0]], [[0, 1e6, -1e6]]], dtype=torch.float32)
        extreme_outputs = extreme_outputs.expand(batch_size, seq_length, num_classes)
        targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
        targets_dict = {'padded_targets': targets}
        
        loss_fn = loss_class(reduction='mean')
        result = loss_fn(extreme_outputs, targets_dict)
        assert torch.isfinite(result)
    
    elif loss_class == TableL1Loss:
        # Test with extreme bbox values
        extreme_outputs = torch.tensor([[1e6, -1e6, 1e6, -1e6]], dtype=torch.float32)
        extreme_outputs = extreme_outputs.expand(batch_size, seq_length, 4)
        bbox = torch.randn(batch_size, seq_length + 1, 4)
        bbox_masks = torch.ones(batch_size, seq_length + 1)
        targets_dict = {'bbox': bbox, 'bbox_masks': bbox_masks}
        
        loss_fn = loss_class()
        result = loss_fn(extreme_outputs, targets_dict)
        assert torch.isfinite(result['horizon_bbox_loss'])
        assert torch.isfinite(result['vertical_bbox_loss'])


@pytest.mark.parametrize("loss_class", ALL_LOSS_CLASSES)
def test_losses_gradient_computation(loss_class):
    """Test that all losses support gradient computation."""
    batch_size, seq_length = 2, 5
    
    if loss_class == MASTERTFLoss:
        num_classes = 5
        outputs = torch.randn(batch_size, seq_length, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
        targets_dict = {'padded_targets': targets}
        
        loss_fn = loss_class(reduction='mean')
        result = loss_fn(outputs, targets_dict)
        result.backward()
        
        assert outputs.grad is not None
        assert not torch.isnan(outputs.grad).any()
    
    elif loss_class == TableL1Loss:
        outputs = torch.randn(batch_size, seq_length, 4, requires_grad=True)
        bbox = torch.randn(batch_size, seq_length + 1, 4)
        bbox_masks = torch.ones(batch_size, seq_length + 1)
        targets_dict = {'bbox': bbox, 'bbox_masks': bbox_masks}
        
        loss_fn = loss_class()
        result = loss_fn(outputs, targets_dict)
        total_loss = result['horizon_bbox_loss'] + result['vertical_bbox_loss']
        total_loss.backward()
        
        assert outputs.grad is not None
        assert not torch.isnan(outputs.grad).any()


@pytest.mark.parametrize("loss_class", ALL_LOSS_CLASSES)
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_losses_device_compatibility(loss_class, device):
    """Test that losses work on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch_size, seq_length = 2, 5
    
    if loss_class == MASTERTFLoss:
        num_classes = 5
        outputs = torch.randn(batch_size, seq_length, num_classes).to(device)
        targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
        targets_dict = {'padded_targets': targets}
        
        loss_fn = loss_class(reduction='mean')
        result = loss_fn(outputs, targets_dict)
        assert result.device.type == device
    
    elif loss_class == TableL1Loss:
        outputs = torch.randn(batch_size, seq_length, 4).to(device)
        bbox = torch.randn(batch_size, seq_length + 1, 4)
        bbox_masks = torch.ones(batch_size, seq_length + 1)
        targets_dict = {'bbox': bbox, 'bbox_masks': bbox_masks}
        
        loss_fn = loss_class()
        result = loss_fn(outputs, targets_dict)
        assert result['horizon_bbox_loss'].device.type == device
        assert result['vertical_bbox_loss'].device.type == device


@pytest.mark.parametrize("loss_class,reduction", [
    (MASTERTFLoss, 'mean'),
    (MASTERTFLoss, 'sum'),
    (MASTERTFLoss, 'none'),
])
def test_loss_reduction_modes(loss_class, reduction):
    """Test different reduction modes for losses that support it."""
    batch_size, seq_length, num_classes = 2, 5, 3
    
    outputs = torch.randn(batch_size, seq_length, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
    targets_dict = {'padded_targets': targets}
    
    loss_fn = loss_class(reduction=reduction)
    result = loss_fn(outputs, targets_dict)
    
    if reduction == 'none':
        assert result.shape[0] == batch_size or result.numel() > 1
    else:
        assert result.numel() == 1
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("loss_class,dtype", [
    (MASTERTFLoss, torch.float32),
    (MASTERTFLoss, torch.float64),
    (TableL1Loss, torch.float32),
    (TableL1Loss, torch.float64),
])
def test_loss_dtype_compatibility(loss_class, dtype):
    """Test that losses work with different data types."""
    batch_size, seq_length = 2, 5
    
    if loss_class == MASTERTFLoss:
        num_classes = 3
        outputs = torch.randn(batch_size, seq_length, num_classes, dtype=dtype)
        targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
        targets_dict = {'padded_targets': targets}
        
        loss_fn = loss_class(reduction='mean')
        result = loss_fn(outputs, targets_dict)
        assert result.dtype == dtype
    
    elif loss_class == TableL1Loss:
        outputs = torch.randn(batch_size, seq_length, 4, dtype=dtype)
        bbox = torch.randn(batch_size, seq_length + 1, 4, dtype=dtype)
        bbox_masks = torch.ones(batch_size, seq_length + 1, dtype=dtype)
        targets_dict = {'bbox': bbox, 'bbox_masks': bbox_masks}
        
        loss_fn = loss_class()
        result = loss_fn(outputs, targets_dict)
        assert result['horizon_bbox_loss'].dtype == dtype
        assert result['vertical_bbox_loss'].dtype == dtype


@pytest.mark.parametrize("ignore_index", [0, 1, -1, -100])
def test_master_tf_loss_ignore_index(ignore_index):
    """Test MASTERTFLoss with different ignore_index values."""
    batch_size, seq_length, num_classes = 2, 5, 3
    
    outputs = torch.randn(batch_size, seq_length, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
    
    # Set some targets to ignore_index
    targets[:, 0] = ignore_index
    targets_dict = {'padded_targets': targets}
    
    loss_fn = MASTERTFLoss(ignore_index=ignore_index, reduction='mean')
    result = loss_fn(outputs, targets_dict)
    assert torch.isfinite(result)


@pytest.mark.parametrize("lambda_h,lambda_v", [
    (0.0, 1.0),  # Only vertical loss
    (1.0, 0.0),  # Only horizontal loss
    (0.5, 0.5),  # Equal weights
    (2.0, 1.0),  # Emphasize horizontal
    (1.0, 2.0),  # Emphasize vertical
])
def test_table_l1_loss_weight_combinations(lambda_h, lambda_v):
    """Test TableL1Loss with different weight combinations."""
    batch_size, seq_length = 2, 5
    
    outputs = torch.randn(batch_size, seq_length, 4)
    bbox = torch.randn(batch_size, seq_length + 1, 4)
    bbox_masks = torch.ones(batch_size, seq_length + 1)
    targets_dict = {'bbox': bbox, 'bbox_masks': bbox_masks}
    
    loss_fn = TableL1Loss(lambda_horizon=lambda_h, lambda_vertical=lambda_v)
    result = loss_fn(outputs, targets_dict)
    
    # Check that weights are applied correctly
    if lambda_h == 0.0:
        assert result['horizon_bbox_loss'] == 0.0
    if lambda_v == 0.0:
        assert result['vertical_bbox_loss'] == 0.0
    
    assert torch.isfinite(result['horizon_bbox_loss'])
    assert torch.isfinite(result['vertical_bbox_loss'])


@pytest.mark.parametrize("loss_class", ALL_LOSS_CLASSES)
@pytest.mark.parametrize("requires_grad", [True, False])
def test_loss_gradient_requirements(loss_class, requires_grad):
    """Test losses with and without gradient requirements."""
    batch_size, seq_length = 2, 3
    
    if loss_class == MASTERTFLoss:
        num_classes = 3
        outputs = torch.randn(batch_size, seq_length, num_classes, requires_grad=requires_grad)
        targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
        targets_dict = {'padded_targets': targets}
        
        loss_fn = loss_class(reduction='mean')
        result = loss_fn(outputs, targets_dict)
        
        if requires_grad:
            result.backward()
            assert outputs.grad is not None
        else:
            assert outputs.grad is None
    
    elif loss_class == TableL1Loss:
        outputs = torch.randn(batch_size, seq_length, 4, requires_grad=requires_grad)
        bbox = torch.randn(batch_size, seq_length + 1, 4)
        bbox_masks = torch.ones(batch_size, seq_length + 1)
        targets_dict = {'bbox': bbox, 'bbox_masks': bbox_masks}
        
        loss_fn = loss_class()
        result = loss_fn(outputs, targets_dict)
        total_loss = result['horizon_bbox_loss'] + result['vertical_bbox_loss']
        
        if requires_grad:
            total_loss.backward()
            assert outputs.grad is not None
        else:
            assert outputs.grad is None


# ...existing device compatibility test...
