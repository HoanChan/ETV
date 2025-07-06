# Copyright (c) Lê Hoàn Chân. All rights reserved.
import pytest
import torch
import torch.nn as nn

from models.losses.table_l1_loss import TableL1Loss


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    batch_size = 2
    seq_length = 10
    
    # Create sample outputs (predicted bboxes)
    outputs = torch.randn(batch_size, seq_length, 4)  # (B, L, 4) - (x, y, w, h)
    
    # Create sample targets with start token (seq_length + 1)
    bbox = torch.randn(batch_size, seq_length + 1, 4)
    bbox_masks = torch.ones(batch_size, seq_length + 1)
    
    # Set some masks to 0 to simulate padding
    bbox_masks[:, -2:] = 0
    
    targets_dict = {
        'bbox': bbox,
        'bbox_masks': bbox_masks
    }
    
    return outputs, targets_dict


@pytest.fixture(params=[
    (2, 10),   # Standard case
    (1, 5),    # Small batch
    (4, 20),   # Medium batch  
    (8, 50),   # Large batch
])
def various_input_sizes(request):
    """Create sample data with various input sizes."""
    batch_size, seq_length = request.param
    
    outputs = torch.randn(batch_size, seq_length, 4)
    bbox = torch.randn(batch_size, seq_length + 1, 4)
    bbox_masks = torch.ones(batch_size, seq_length + 1)
    
    targets_dict = {
        'bbox': bbox,
        'bbox_masks': bbox_masks
    }
    
    return outputs, targets_dict

@pytest.mark.parametrize(
    "lambda_horizon,lambda_vertical,eps",
    [
        (1.0, 1.0, 1e-9),     # Default values
        (0.5, 1.5, 1e-8),     # Different weights
        (2.0, 0.8, 1e-6),     # Different epsilon
        (0.0, 1.0, 1e-9),     # Zero horizontal weight
        (1.0, 0.0, 1e-9),     # Zero vertical weight
    ]
)
def test_forward_with_different_parameters(sample_data, lambda_horizon, lambda_vertical, eps):
    """Test forward method with different parameter combinations."""
    outputs, targets_dict = sample_data
    
    loss_fn = TableL1Loss(
        lambda_horizon=lambda_horizon,
        lambda_vertical=lambda_vertical,
        eps=eps
    )
    
    losses = loss_fn(outputs, targets_dict)
    
    # Check that losses dictionary is returned
    assert isinstance(losses, dict)
    assert 'horizon_bbox_loss' in losses
    assert 'vertical_bbox_loss' in losses
    
    # Check that losses are tensors
    assert isinstance(losses['horizon_bbox_loss'], torch.Tensor)
    assert isinstance(losses['vertical_bbox_loss'], torch.Tensor)
    
    # Check that losses are scalars
    assert losses['horizon_bbox_loss'].numel() == 1
    assert losses['vertical_bbox_loss'].numel() == 1
    
    # Check that losses are finite
    assert torch.isfinite(losses['horizon_bbox_loss'])
    assert torch.isfinite(losses['vertical_bbox_loss'])


def test_format_inputs_shapes(sample_data):
    """Test _format_inputs method shapes."""
    outputs, targets_dict = sample_data
    batch_size, seq_length, _ = outputs.shape
    
    loss_fn = TableL1Loss()
    masked_outputs, masked_targets, bbox_masks = loss_fn._format_inputs(outputs, targets_dict)
    
    # Check shapes
    assert masked_outputs.shape == (batch_size, seq_length, 4)
    assert masked_targets.shape == (batch_size, seq_length, 4)
    assert bbox_masks.shape == (batch_size, seq_length, 1)


def test_target_sequence_shifting(sample_data):
    """Test that targets are properly shifted (removing start token)."""
    outputs, targets_dict = sample_data
    original_bbox = targets_dict['bbox']
    original_masks = targets_dict['bbox_masks']
    
    loss_fn = TableL1Loss()
    masked_outputs, masked_targets, bbox_masks = loss_fn._format_inputs(outputs, targets_dict)
    
    # Check that we're using bbox[1:] and masks[1:] (removing start token)
    expected_bbox = original_bbox[:, 1:, :]
    expected_masks = original_masks[:, 1:].unsqueeze(-1)
    
    # The targets should be masked, so we check the original values
    assert masked_targets.shape == expected_bbox.shape
    assert bbox_masks.shape == expected_masks.shape


def test_different_input_sizes(various_input_sizes):
    """Test with different input tensor sizes."""
    outputs, targets_dict = various_input_sizes
    
    loss_fn = TableL1Loss()
    losses = loss_fn(outputs, targets_dict)
    
    assert isinstance(losses, dict)
    assert torch.isfinite(losses['horizon_bbox_loss'])
    assert torch.isfinite(losses['vertical_bbox_loss'])


@pytest.mark.parametrize(
    "mask_pattern,expected_finite", 
    [
        ("partial", True),     # Partial masking
        ("single", True),      # Only one valid element
        ("all_zero", True),    # All masked (should handle with eps)
    ]
)
def test_masking_functionality(sample_data, mask_pattern, expected_finite):
    """Test that masking works correctly with different patterns."""
    outputs, targets_dict = sample_data
    
    if mask_pattern == "partial":
        # Default behavior from fixture (some padding at end)
        pass
    elif mask_pattern == "single":
        # Set all masks to 0 except first element
        targets_dict['bbox_masks'].fill_(0)
        targets_dict['bbox_masks'][:, 1] = 1  # Only one valid element after shifting
    elif mask_pattern == "all_zero":
        # Set all masks to 0
        targets_dict['bbox_masks'].fill_(0)
    
    loss_fn = TableL1Loss()
    losses = loss_fn(outputs, targets_dict)
    
    # Losses should be finite based on expected_finite
    assert torch.isfinite(losses['horizon_bbox_loss']) == expected_finite
    assert torch.isfinite(losses['vertical_bbox_loss']) == expected_finite


@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_invalid_reduction_raises_error(reduction):
    """Test that invalid reduction parameter raises error."""
    with pytest.raises(AssertionError):
        TableL1Loss(reduction=reduction)


@pytest.mark.parametrize(
    "missing_key",
    ["bbox", "bbox_masks"]
)
def test_missing_required_keys(missing_key):
    """Test that missing required keys raise appropriate errors."""
    outputs = torch.randn(2, 10, 4)
    
    if missing_key == "bbox":
        targets_dict = {'bbox_masks': torch.ones(2, 11)}
    else:  # missing bbox_masks
        targets_dict = {'bbox': torch.randn(2, 11, 4)}
    
    loss_fn = TableL1Loss()
    
    with pytest.raises(KeyError):
        loss_fn(outputs, targets_dict)


def test_inheritance_from_module():
    """Test that TableL1Loss properly inherits from nn.Module."""
    loss_fn = TableL1Loss()
    assert isinstance(loss_fn, nn.Module)
    assert hasattr(loss_fn, 'reduction')
    assert hasattr(loss_fn, 'lambda_horizon')
    assert hasattr(loss_fn, 'lambda_vertical')
    assert hasattr(loss_fn, 'eps')


def test_coordinate_separation(sample_data):
    """Test that horizontal and vertical coordinates are processed separately."""
    outputs, targets_dict = sample_data
    
    # Create outputs with distinct patterns for horizontal/vertical
    outputs[:, :, [0, 2]] = 1.0  # x, width
    outputs[:, :, [1, 3]] = 2.0  # y, height
    
    targets_dict['bbox'][:, :, [0, 2]] = 0.0  # Different from outputs
    targets_dict['bbox'][:, :, [1, 3]] = 0.0
    
    loss_fn = TableL1Loss()
    losses = loss_fn(outputs, targets_dict)
    
    # Both losses should be positive since outputs != targets
    assert losses['horizon_bbox_loss'] > 0
    assert losses['vertical_bbox_loss'] > 0


def test_gradient_flow(sample_data):
    """Test that gradients can flow through the loss."""
    outputs, targets_dict = sample_data
    outputs.requires_grad_(True)
    
    loss_fn = TableL1Loss()
    losses = loss_fn(outputs, targets_dict)
    
    # Backward pass on combined loss
    total_loss = losses['horizon_bbox_loss'] + losses['vertical_bbox_loss']
    total_loss.backward()
    
    # Check that gradients are computed
    assert outputs.grad is not None
    assert not torch.isnan(outputs.grad).any()


def test_deterministic_behavior(sample_data):
    """Test that the loss function produces deterministic results."""
    outputs, targets_dict = sample_data
    
    loss_fn = TableL1Loss()
    
    # Compute losses twice with same inputs
    losses1 = loss_fn(outputs, targets_dict)
    losses2 = loss_fn(outputs, targets_dict)
    
    # Results should be identical
    assert torch.equal(losses1['horizon_bbox_loss'], losses2['horizon_bbox_loss'])
    assert torch.equal(losses1['vertical_bbox_loss'], losses2['vertical_bbox_loss'])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_different_dtypes(dtype):
    """Test with different tensor dtypes."""
    batch_size, seq_length = 2, 10
    
    outputs = torch.randn(batch_size, seq_length, 4, dtype=dtype)
    bbox = torch.randn(batch_size, seq_length + 1, 4, dtype=dtype)
    bbox_masks = torch.ones(batch_size, seq_length + 1, dtype=dtype)
    
    targets_dict = {
        'bbox': bbox,
        'bbox_masks': bbox_masks
    }
    
    loss_fn = TableL1Loss()
    losses = loss_fn(outputs, targets_dict)
    
    # Losses should have same dtype as inputs
    assert losses['horizon_bbox_loss'].dtype == dtype
    assert losses['vertical_bbox_loss'].dtype == dtype


@pytest.mark.parametrize("use_cuda", [False, True])
def test_device_compatibility(sample_data, use_cuda):
    """Test that loss function works with different devices."""
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    outputs, targets_dict = sample_data
    
    if use_cuda:
        outputs = outputs.cuda()
    
    loss_fn = TableL1Loss()
    losses = loss_fn(outputs, targets_dict)
    
    assert torch.isfinite(losses['horizon_bbox_loss'])
    assert torch.isfinite(losses['vertical_bbox_loss'])


@pytest.mark.parametrize("with_img_metas", [False, True])
def test_with_img_metas(sample_data, with_img_metas):
    """Test that img_metas parameter doesn't affect computation."""
    outputs, targets_dict = sample_data
    
    loss_fn = TableL1Loss()
    
    if with_img_metas:
        img_metas = [{'img_shape': (224, 224, 3)}, {'img_shape': (256, 256, 3)}]
        losses = loss_fn(outputs, targets_dict, img_metas)
    else:
        losses = loss_fn(outputs, targets_dict)
    
    # Check basic requirements
    assert isinstance(losses, dict)
    assert torch.isfinite(losses['horizon_bbox_loss'])
    assert torch.isfinite(losses['vertical_bbox_loss'])
