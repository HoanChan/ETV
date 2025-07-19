# Copyright (c) Lê Hoàn Chân. All rights reserved.
import pytest
import torch
import torch.nn as nn

from models.losses.master_tf_loss import MasterTFLoss


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    batch_size = 2
    seq_length = 10
    num_classes = 5
    
    # Create sample outputs (logits)
    outputs = torch.randn(batch_size, seq_length, num_classes)
    
    # Create sample targets (includes start token, so length is seq_length + 1)
    targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
    targets_dict = {'padded_targets': targets}
    
    return outputs, targets_dict


@pytest.fixture(
    params=[
        (1, 5, 10),      # Small batch
        (2, 10, 5),      # Default size
        (4, 20, 100),    # Medium batch
        (8, 50, 1000),   # Large batch
    ],
    ids=['small', 'default', 'medium', 'large']
)
def variable_sample_data(request):
    """Create sample data with different sizes for testing."""
    batch_size, seq_length, num_classes = request.param
    
    outputs = torch.randn(batch_size, seq_length, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
    targets_dict = {'padded_targets': targets}
    
    return outputs, targets_dict


@pytest.mark.parametrize(
    "ignore_index,reduction,flatten,expected_shape",
    [
        (-1, 'none', True, None),            # Flattened output shape depends on non-ignored elements
        (-1, 'mean', True, torch.Size([])),  # Scalar output
        (-1, 'sum', True, torch.Size([])),   # Scalar output  
        (-1, 'none', False, (2, 10)),        # (batch_size, seq_length) when not flattened
        (0, 'none', True, None),             # With ignore_index = 0
        (4, 'mean', False, torch.Size([])),  # Different ignore_index with mean reduction
    ],
    ids=['none_flat', 'mean_flat', 'sum_flat', 'none_no_flat', 'ignore_0', 'ignore_4_mean']
)
def test_forward_with_different_parameters(sample_data, ignore_index, reduction, flatten, expected_shape):
    """Test forward method with different parameter combinations."""
    outputs, targets_dict = sample_data
    
    loss_fn = MasterTFLoss(
        ignore_index=ignore_index,
        reduction=reduction,
        flatten=flatten
    )
    
    loss = loss_fn(outputs, targets_dict)
    
    # Check that loss is a tensor
    assert isinstance(loss, torch.Tensor)
    
    # Check output shape for specific cases
    if expected_shape is not None:
        assert loss.shape == expected_shape
    
    # Check that loss values are reasonable (not NaN or infinite)
    assert torch.isfinite(loss).all()



def test_different_input_sizes(variable_sample_data):
    """Test with different input tensor sizes using parametrized fixture."""
    outputs, targets_dict = variable_sample_data
    
    loss_fn = MasterTFLoss()
    loss = loss_fn(outputs, targets_dict)
    
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize(
    "ignore_value",
    [0, 1, 4],
    ids=['ignore_0', 'ignore_1', 'ignore_4']
)
def test_ignore_index_functionality(sample_data, ignore_value):
    """Test that ignore_index works correctly."""
    outputs, targets_dict = sample_data
    
    # Set some target values to ignore_value
    targets_dict['padded_targets'][:, 1:3] = ignore_value
    
    loss_fn = MasterTFLoss(ignore_index=ignore_value, reduction='none', flatten=True)
    loss = loss_fn(outputs, targets_dict)
    
    # Loss should be computed (some values might be 0 due to ignore_index)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape[0] == outputs.shape[0] * outputs.shape[1]


def test_inheritance_from_crossentropyloss():
    """Test that MasterTFLoss properly uses CrossEntropyLoss internally."""
    loss_fn = MasterTFLoss()
    assert isinstance(loss_fn, nn.Module)
    assert hasattr(loss_fn, 'ctc_loss')
    assert isinstance(loss_fn.ctc_loss, nn.CrossEntropyLoss)


@pytest.mark.parametrize(
    "wrong_key",
    ['targets', 'labels', 'ground_truth'],
    ids=['targets', 'labels', 'ground_truth']
)
def test_missing_padded_targets_key(wrong_key):
    """Test that missing 'padded_targets' key raises appropriate error."""
    outputs = torch.randn(2, 10, 5)
    targets_dict = {wrong_key: torch.randint(0, 5, (2, 11))}
    
    loss_fn = MasterTFLoss()
    
    with pytest.raises(KeyError):
        loss_fn(outputs, targets_dict)


@pytest.mark.parametrize(
    "reduction",
    ['none', 'mean', 'sum'],
    ids=['none', 'mean', 'sum']
)
def test_reduction_modes(sample_data, reduction):
    """Test different reduction modes."""
    outputs, targets_dict = sample_data
    
    loss_fn = MasterTFLoss(reduction=reduction)
    loss = loss_fn(outputs, targets_dict)
    
    if reduction == 'none':
        # Should return loss for each element
        assert loss.numel() > 1
    else:
        # Should return scalar
        assert loss.numel() == 1


def test_gradient_flow(sample_data):
    """Test that gradients can flow through the loss."""
    outputs, targets_dict = sample_data
    outputs.requires_grad_(True)
    
    loss_fn = MasterTFLoss(reduction='mean')
    loss = loss_fn(outputs, targets_dict)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients are computed
    assert outputs.grad is not None
    assert not torch.isnan(outputs.grad).any()


def test_deterministic_behavior(sample_data):
    """Test that the loss function produces deterministic results."""
    outputs, targets_dict = sample_data
    
    loss_fn = MasterTFLoss(reduction='mean')
    
    # Compute loss twice with same inputs
    loss1 = loss_fn(outputs, targets_dict)
    loss2 = loss_fn(outputs, targets_dict)
    
    # Results should be identical
    assert torch.equal(loss1, loss2)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=['float32', 'float64']
)
def test_different_dtypes(dtype):
    """Test with different tensor dtypes."""
    batch_size, seq_length, num_classes = 2, 10, 5
    
    outputs = torch.randn(batch_size, seq_length, num_classes, dtype=dtype)
    targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
    targets_dict = {'padded_targets': targets}
    
    loss_fn = MasterTFLoss()
    loss = loss_fn(outputs, targets_dict)
    
    # Loss should have same dtype as outputs
    assert loss.dtype == dtype
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize(
    "config",
    [
        {'ignore_index': -1, 'reduction': 'mean', 'flatten': True},
        {'ignore_index': 0, 'reduction': 'none', 'flatten': False},
        {'ignore_index': 2, 'reduction': 'sum', 'flatten': True},
    ],
    ids=['default', 'no_flatten', 'custom_ignore']
)
def test_configuration_combinations(sample_data, config):
    """Test various configuration combinations efficiently."""
    outputs, targets_dict = sample_data
    
    loss_fn = MasterTFLoss(**config)
    loss = loss_fn(outputs, targets_dict)
    
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss).all()
    
    # Verify shape based on reduction
    if config['reduction'] in ['mean', 'sum']:
        assert loss.shape == torch.Size([])
    else:
        assert loss.numel() > 0


@pytest.mark.parametrize(
    "batch_size,seq_length,num_classes,expected_ratio",
    [
        (1, 5, 10, 1.0),      # Small - all elements should be processed
        (4, 20, 100, 1.0),    # Medium - all elements should be processed  
        (8, 50, 1000, 1.0),   # Large - all elements should be processed
    ],
    ids=['small_all', 'medium_all', 'large_all']
)
def test_loss_computation_efficiency(batch_size, seq_length, num_classes, expected_ratio):
    """Test that loss computation handles different sizes efficiently."""
    outputs = torch.randn(batch_size, seq_length, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, seq_length + 1))
    targets_dict = {'padded_targets': targets}
    
    loss_fn = MasterTFLoss(reduction='none', flatten=True)
    loss = loss_fn(outputs, targets_dict)
    
    # Check that we get loss for expected number of elements
    expected_elements = batch_size * seq_length
    assert loss.numel() == expected_elements
    
    # Check that most elements have valid loss (not all zero due to ignore_index)
    non_zero_ratio = (loss != 0).float().mean().item()
    assert non_zero_ratio >= expected_ratio
