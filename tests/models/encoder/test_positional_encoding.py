from unittest.mock import MagicMock
import pytest
import torch
from models.encoders.positional_encoding import PositionalEncoding

# Test initialization parameters
@pytest.mark.parametrize("d_model,dropout,max_len", [
    (512, 0.0, 5000),      # default values
    (256, 0.1, 1000),      # smaller model
    (1024, 0.2, 10000),    # larger model
    (128, 0.0, 100),       # minimal model
])
def test_positional_encoding_init(d_model, dropout, max_len):
    """Test PositionalEncoding initialization with different parameters"""
    pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
    
    # Check positional encoding buffer shape
    assert pe.pe.shape == (1, max_len, d_model)
    
    # Check dropout layer
    assert pe.dropout.p == dropout
    
    # Verify positional encoding values are within reasonable range
    assert torch.all(pe.pe >= -1) and torch.all(pe.pe <= 1)


# Test forward with different input shapes
@pytest.mark.parametrize("input_shape,expected_output_shape", [
    # 2D input: (batch_size, seq_len, d_model)
    ((2, 100, 512), (2, 100, 512)),
    ((1, 50, 256), (1, 50, 256)),
    ((4, 200, 128), (4, 200, 128)),
    
    # 4D input: (batch_size, channels, height, width)
    ((2, 512, 8, 8), (2, 64, 512)),     # 8*8 = 64 sequence length
    ((1, 256, 4, 10), (1, 40, 256)),    # 4*10 = 40 sequence length
    ((3, 128, 5, 6), (3, 30, 128)),     # 5*6 = 30 sequence length
])
def test_forward_shapes(input_shape, expected_output_shape):
    """Test forward pass with different input shapes"""
    if len(input_shape) == 3:
        d_model = input_shape[2]
    else:
        d_model = input_shape[1]
    
    pe = PositionalEncoding(d_model=d_model, dropout=0.0)
    
    # Create input tensor
    feat = torch.randn(input_shape)
    
    # Forward pass
    output = pe(feat)
    
    # Check output shape
    assert output.shape == expected_output_shape


# Test positional encoding values
@pytest.mark.parametrize("d_model,seq_len", [
    (512, 100),
    (256, 50),
    (128, 200),
])
def test_positional_encoding_values(d_model, seq_len):
    """Test that positional encoding values are computed correctly"""
    pe = PositionalEncoding(d_model=d_model, dropout=0.0)
    
    # Create dummy input
    feat = torch.zeros(1, seq_len, d_model)
    
    # Forward pass
    output = pe(feat)
    
    # Check that positional encoding was added
    expected_pe = pe.pe[:, :seq_len, :]
    assert torch.allclose(output, expected_pe, atol=1e-6)


# Test dropout functionality
@pytest.mark.parametrize("dropout_rate", [0.0, 0.1, 0.3, 0.5])
def test_dropout_functionality(dropout_rate):
    """Test dropout behavior during training and evaluation"""
    pe = PositionalEncoding(d_model=256, dropout=dropout_rate)
    feat = torch.randn(2, 50, 256)
    
    # Test training mode
    pe.train()
    output_train1 = pe(feat)
    output_train2 = pe(feat)
    
    if dropout_rate > 0:
        # With dropout, outputs should be different in training mode
        assert not torch.allclose(output_train1, output_train2, rtol=1e-5)
    else:
        # Without dropout, outputs should be identical
        assert torch.allclose(output_train1, output_train2)
    
    # Test evaluation mode
    pe.eval()
    output_eval1 = pe(feat)
    output_eval2 = pe(feat)
    
    # In eval mode, outputs should always be identical
    assert torch.allclose(output_eval1, output_eval2)


# Test edge cases
@pytest.mark.parametrize("batch_size,seq_len,d_model", [
    (1, 1, 128),           # minimal sequence
    (10, 1000, 512),       # long sequence
    (1, 5000, 512),        # max length sequence
])
def test_edge_cases(batch_size, seq_len, d_model):
    """Test edge cases with extreme parameters"""
    pe = PositionalEncoding(d_model=d_model, max_len=max(seq_len, 5000))
    feat = torch.randn(batch_size, seq_len, d_model)
    
    # Should not raise any errors
    output = pe(feat)
    assert output.shape == (batch_size, seq_len, d_model)


# Test with data_samples parameter
def test_with_data_samples():
    """Test forward pass with data_samples parameter"""
    pe = PositionalEncoding(d_model=256, dropout=0.0)
    feat = torch.randn(2, 100, 256)
    
    # Mock data_samples
    mock_data_samples = [MagicMock(), MagicMock()]
    
    # Should work with data_samples
    output = pe(feat, data_samples=mock_data_samples)
    assert output.shape == (2, 100, 256)


# Test mathematical properties
def test_positional_encoding_properties():
    """Test mathematical properties of positional encoding"""
    pe = PositionalEncoding(d_model=128, dropout=0.0)
    
    # Test that even positions use sin, odd positions use cos
    pe_values = pe.pe.squeeze(0)  # Remove batch dimension
    
    # Check first few positions
    position_0 = pe_values[0]
    position_1 = pe_values[1]
    
    # For position 0, sin(0) = 0, cos(0) = 1
    assert torch.allclose(position_0[0::2], torch.zeros_like(position_0[0::2]), atol=1e-6)
    assert torch.allclose(position_0[1::2], torch.ones_like(position_0[1::2]), atol=1e-6)


# Test device compatibility
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compatibility():
    """Test that the model works on different devices"""
    pe = PositionalEncoding(d_model=256, dropout=0.0)
    
    # Test on CPU
    feat_cpu = torch.randn(2, 50, 256)
    output_cpu = pe(feat_cpu)
    assert output_cpu.device == torch.device('cpu')
    
    # Test on GPU
    pe_gpu = pe.cuda()
    feat_gpu = feat_cpu.cuda()
    output_gpu = pe_gpu(feat_gpu)
    assert output_gpu.device.type == 'cuda'


# Test memory efficiency
def test_memory_efficiency():
    """Test that positional encodings are precomputed and reused"""
    pe = PositionalEncoding(d_model=256, dropout=0.0, max_len=1000)
    
    # The pe buffer should be precomputed
    assert hasattr(pe, 'pe')
    assert pe.pe.requires_grad == False  # Should be a buffer, not parameter
    
    # Multiple forward passes should reuse the same buffer
    feat1 = torch.randn(1, 100, 256)
    feat2 = torch.randn(1, 200, 256)
    
    pe_buffer_before = pe.pe.clone()
    _ = pe(feat1)
    _ = pe(feat2)
    pe_buffer_after = pe.pe.clone()
    
    # Buffer should remain unchanged
    assert torch.allclose(pe_buffer_before, pe_buffer_after)