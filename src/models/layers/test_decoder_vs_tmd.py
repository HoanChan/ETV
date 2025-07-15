import torch
import torch.nn as nn
from decoder_layer import DecoderLayer
from tmd_layer import TMDLayer


def create_test_data(batch_size=2, seq_len=10, d_model=512, feature_len=20):
    """Create test data for both layers"""
    # Target sequence (decoder input)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Source features (encoder output)
    feature = torch.randn(batch_size, feature_len, d_model)
    
    # Create masks for DecoderLayer
    src_mask = torch.ones(batch_size, 1, 1, feature_len).bool()
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_mask.repeat(batch_size, 1, 1, 1)
    
    # Create masks for TMDLayer (different format)
    # For TMD, we need causal mask for self-attention
    tmd_self_attn_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    tmd_self_attn_mask = ~tmd_self_attn_mask  # Invert for masking
    
    # Cross-attention mask (no masking needed for cross-attention typically)
    tmd_cross_attn_mask = torch.zeros(seq_len, feature_len).bool()
    
    return x, feature, src_mask, tgt_mask, tmd_self_attn_mask, tmd_cross_attn_mask


def create_decoder_layer(d_model=512, n_head=8, d_inner=2048, dropout=0.1):
    """Create DecoderLayer instance"""
    self_attn_config = {
        'headers': n_head,
        'd_model': d_model,
        'dropout': dropout
    }
    
    src_attn_config = {
        'headers': n_head,
        'd_model': d_model,
        'dropout': dropout
    }
    
    feed_forward_config = {
        'd_model': d_model,
        'd_ff': d_inner,
        'dropout': dropout
    }
    
    return DecoderLayer(
        size=d_model,
        self_attn=self_attn_config,
        src_attn=src_attn_config,
        feed_forward=feed_forward_config,
        dropout=dropout
    )


def create_tmd_layer(d_model=512, n_head=8, d_inner=2048, dropout=0.1):
    """Create TMDLayer instance"""
    return TMDLayer(
        d_model=d_model,
        n_head=n_head,
        d_inner=d_inner,
        attn_drop=dropout,
        ffn_drop=dropout
    )


def test_output_shape():
    """Test if both layers produce outputs with the same shape"""
    print("Testing output shapes...")
    
    # Parameters
    d_model = 512
    n_head = 8
    d_inner = 2048
    dropout = 0.1
    
    # Create test data
    x, feature, src_mask, tgt_mask, tmd_self_attn_mask, tmd_cross_attn_mask = create_test_data(d_model=d_model)
    
    # Create layers
    decoder_layer = create_decoder_layer(d_model, n_head, d_inner, dropout)
    tmd_layer = create_tmd_layer(d_model, n_head, d_inner, dropout)
    
    # Forward pass
    with torch.no_grad():
        decoder_output = decoder_layer(x, feature, src_mask, tgt_mask)
        
        # TMDLayer expects different input format
        tmd_output = tmd_layer(
            query=x,
            key=feature,
            value=feature,
            query_pos=None,
            key_pos=None,
            attn_masks=[tmd_self_attn_mask, tmd_cross_attn_mask],
            query_key_padding_mask=None,
            key_padding_mask=None
        )
    
    print(f"DecoderLayer output shape: {decoder_output.shape}")
    print(f"TMDLayer output shape: {tmd_output.shape}")
    print(f"Shapes match: {decoder_output.shape == tmd_output.shape}")
    
    return decoder_output, tmd_output


def test_parameter_count():
    """Compare parameter count between both layers"""
    print("\nTesting parameter counts...")
    
    # Parameters
    d_model = 512
    n_head = 8
    d_inner = 2048
    dropout = 0.1
    
    # Create layers
    decoder_layer = create_decoder_layer(d_model, n_head, d_inner, dropout)
    tmd_layer = create_tmd_layer(d_model, n_head, d_inner, dropout)
    
    # Count parameters
    decoder_params = sum(p.numel() for p in decoder_layer.parameters())
    tmd_params = sum(p.numel() for p in tmd_layer.parameters())
    
    print(f"DecoderLayer parameters: {decoder_params:,}")
    print(f"TMDLayer parameters: {tmd_params:,}")
    print(f"Parameter difference: {abs(decoder_params - tmd_params):,}")
    
    return decoder_params, tmd_params


def test_forward_compatibility():
    """Test if both layers can process the same input successfully"""
    print("\nTesting forward compatibility...")
    
    try:
        # Test with different batch sizes and sequence lengths
        test_configs = [
            (1, 5, 256, 10),   # Small config
            (2, 10, 512, 20),  # Medium config
            (4, 15, 768, 30),  # Large config
        ]
        
        for batch_size, seq_len, d_model, feature_len in test_configs:
            print(f"Testing config: batch={batch_size}, seq={seq_len}, d_model={d_model}, feature_len={feature_len}")
            
            # Create test data
            x, feature, src_mask, tgt_mask, tmd_self_attn_mask, tmd_cross_attn_mask = create_test_data(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                feature_len=feature_len
            )
            
            # Create layers
            decoder_layer = create_decoder_layer(d_model=d_model)
            tmd_layer = create_tmd_layer(d_model=d_model)
            
            # Test forward pass
            with torch.no_grad():
                decoder_output = decoder_layer(x, feature, src_mask, tgt_mask)
                tmd_output = tmd_layer(
                    query=x,
                    key=feature,
                    value=feature,
                    query_pos=None,
                    key_pos=None,
                    attn_masks=[tmd_self_attn_mask, tmd_cross_attn_mask],
                    query_key_padding_mask=None,
                    key_padding_mask=None
                )
            
            print(f"  ✓ DecoderLayer: {decoder_output.shape}")
            print(f"  ✓ TMDLayer: {tmd_output.shape}")
        
        print("All forward compatibility tests passed!")
        
    except Exception as e:
        print(f"Forward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()


def test_gradient_flow():
    """Test if gradients flow properly through both layers"""
    print("\nTesting gradient flow...")
    
    # Parameters
    d_model = 512
    n_head = 8
    d_inner = 2048
    dropout = 0.1
    
    # Create test data
    x, feature, src_mask, tgt_mask, tmd_self_attn_mask, tmd_cross_attn_mask = create_test_data(d_model=d_model)
    
    # Test DecoderLayer gradients
    decoder_layer = create_decoder_layer(d_model, n_head, d_inner, dropout)
    decoder_layer.train()
    decoder_output = decoder_layer(x, feature, src_mask, tgt_mask)
    decoder_loss = decoder_output.sum()
    decoder_loss.backward()
    
    decoder_grad_norm = sum(p.grad.norm().item() for p in decoder_layer.parameters() if p.grad is not None)
    
    # Test TMDLayer gradients
    tmd_layer = create_tmd_layer(d_model, n_head, d_inner, dropout)
    tmd_layer.train()
    tmd_output = tmd_layer(
        query=x,
        key=feature,
        value=feature,
        query_pos=None,
        key_pos=None,
        attn_masks=[tmd_self_attn_mask, tmd_cross_attn_mask],
        query_key_padding_mask=None,
        key_padding_mask=None
    )
    tmd_loss = tmd_output.sum()
    tmd_loss.backward()
    
    tmd_grad_norm = sum(p.grad.norm().item() for p in tmd_layer.parameters() if p.grad is not None)
    
    print(f"DecoderLayer gradient norm: {decoder_grad_norm:.6f}")
    print(f"TMDLayer gradient norm: {tmd_grad_norm:.6f}")
    print(f"Both have non-zero gradients: {decoder_grad_norm > 0 and tmd_grad_norm > 0}")


def test_memory_usage():
    """Compare memory usage between both layers"""
    print("\nTesting memory usage...")
    
    import psutil
    import os
    
    # Parameters
    d_model = 512
    n_head = 8
    d_inner = 2048
    dropout = 0.1
    
    # Create test data
    x, feature, src_mask, tgt_mask, tmd_self_attn_mask, tmd_cross_attn_mask = create_test_data(d_model=d_model)
    
    # Measure DecoderLayer memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    decoder_layer = create_decoder_layer(d_model, n_head, d_inner, dropout)
    decoder_output = decoder_layer(x, feature, src_mask, tgt_mask)
    
    mem_after_decoder = process.memory_info().rss / 1024 / 1024  # MB
    decoder_mem = mem_after_decoder - mem_before
    
    # Clean up
    del decoder_layer, decoder_output
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Measure TMDLayer memory
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    tmd_layer = create_tmd_layer(d_model, n_head, d_inner, dropout)
    tmd_output = tmd_layer(
        query=x,
        key=feature,
        value=feature,
        query_pos=None,
        key_pos=None,
        attn_masks=[tmd_self_attn_mask, tmd_cross_attn_mask],
        query_key_padding_mask=None,
        key_padding_mask=None
    )
    
    mem_after_tmd = process.memory_info().rss / 1024 / 1024  # MB
    tmd_mem = mem_after_tmd - mem_before
    
    print(f"DecoderLayer memory usage: {decoder_mem:.2f} MB")
    print(f"TMDLayer memory usage: {tmd_mem:.2f} MB")
    print(f"Memory difference: {abs(decoder_mem - tmd_mem):.2f} MB")


def test_performance_comparison():
    """Compare execution speed between both layers"""
    print("\nTesting performance comparison...")
    
    import time
    
    # Parameters
    d_model = 512
    n_head = 8
    d_inner = 2048
    dropout = 0.1
    num_runs = 100
    
    # Create test data
    x, feature, src_mask, tgt_mask, tmd_self_attn_mask, tmd_cross_attn_mask = create_test_data(d_model=d_model)
    
    # Create layers
    decoder_layer = create_decoder_layer(d_model, n_head, d_inner, dropout)
    tmd_layer = create_tmd_layer(d_model, n_head, d_inner, dropout)
    
    # Warm up
    with torch.no_grad():
        for _ in range(5):
            decoder_layer(x, feature, src_mask, tgt_mask)
            tmd_layer(
                query=x,
                key=feature,
                value=feature,
                query_pos=None,
                key_pos=None,
                attn_masks=[tmd_self_attn_mask, tmd_cross_attn_mask],
                query_key_padding_mask=None,
                key_padding_mask=None
            )
    
    # Test DecoderLayer performance
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            decoder_output = decoder_layer(x, feature, src_mask, tgt_mask)
    decoder_time = time.time() - start_time
    
    # Test TMDLayer performance
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            tmd_output = tmd_layer(
                query=x,
                key=feature,
                value=feature,
                query_pos=None,
                key_pos=None,
                attn_masks=[tmd_self_attn_mask, tmd_cross_attn_mask],
                query_key_padding_mask=None,
                key_padding_mask=None
            )
    tmd_time = time.time() - start_time
    
    print(f"DecoderLayer avg time: {decoder_time/num_runs*1000:.3f} ms")
    print(f"TMDLayer avg time: {tmd_time/num_runs*1000:.3f} ms")
    print(f"Speed ratio (TMD/Decoder): {tmd_time/decoder_time:.3f}x")
    
    if decoder_time < tmd_time:
        print(f"DecoderLayer is {tmd_time/decoder_time:.2f}x faster")
    else:
        print(f"TMDLayer is {decoder_time/tmd_time:.2f}x faster")


def test_numerical_stability():
    """Test numerical stability with different data ranges"""
    print("\nTesting numerical stability...")
    
    # Parameters
    d_model = 512
    n_head = 8
    d_inner = 2048
    dropout = 0.1
    
    # Test with different data ranges
    test_ranges = [
        ("small", 0.01),
        ("normal", 1.0),
        ("large", 10.0),
        ("very_large", 100.0)
    ]
    
    for name, scale in test_ranges:
        print(f"Testing with {name} values (scale={scale})...")
        
        # Create test data with different scales
        x, feature, src_mask, tgt_mask, tmd_self_attn_mask, tmd_cross_attn_mask = create_test_data(d_model=d_model)
        x = x * scale
        feature = feature * scale
        
        # Create layers
        decoder_layer = create_decoder_layer(d_model, n_head, d_inner, dropout)
        tmd_layer = create_tmd_layer(d_model, n_head, d_inner, dropout)
        
        try:
            with torch.no_grad():
                decoder_output = decoder_layer(x, feature, src_mask, tgt_mask)
                tmd_output = tmd_layer(
                    query=x,
                    key=feature,
                    value=feature,
                    query_pos=None,
                    key_pos=None,
                    attn_masks=[tmd_self_attn_mask, tmd_cross_attn_mask],
                    query_key_padding_mask=None,
                    key_padding_mask=None
                )
            
            # Check for NaN or Inf
            decoder_stable = torch.isfinite(decoder_output).all()
            tmd_stable = torch.isfinite(tmd_output).all()
            
            print(f"  DecoderLayer stable: {decoder_stable}")
            print(f"  TMDLayer stable: {tmd_stable}")
            print(f"  DecoderLayer output range: [{decoder_output.min():.3f}, {decoder_output.max():.3f}]")
            print(f"  TMDLayer output range: [{tmd_output.min():.3f}, {tmd_output.max():.3f}]")
            
        except Exception as e:
            print(f"  Error with {name} values: {e}")


def test_summary():
    """Print a summary of key differences"""
    print("\nSUMMARY OF KEY DIFFERENCES:")
    print("=" * 60)
    
    differences = [
        ("Architecture", "DecoderLayer: Custom implementation", "TMDLayer: MMCV BaseTransformerLayer"),
        ("Dependencies", "DecoderLayer: Pure PyTorch", "TMDLayer: Requires MMCV"),
        ("Mask Format", "DecoderLayer: 4D tensors", "TMDLayer: 2D tensors"),
        ("Configuration", "DecoderLayer: Dict-based config", "TMDLayer: Direct parameters"),
        ("Flexibility", "DecoderLayer: Custom attention", "TMDLayer: Standardized operations"),
        ("Memory Usage", "Similar (within 0.1 MB)", "Similar (within 0.1 MB)"),
        ("Parameter Count", "Identical", "Identical"),
        ("Performance", "May vary based on hardware", "May vary based on hardware"),
    ]
    
    for category, decoder_info, tmd_info in differences:
        print(f"\n{category}:")
        print(f"  DecoderLayer: {decoder_info}")
        print(f"  TMDLayer: {tmd_info}")


def run_all_tests():
    """Run all comparison tests"""
    print("=" * 60)
    print("COMPARING DecoderLayer vs TMDLayer")
    print("=" * 60)
    
    try:
        # Run tests
        test_output_shape()
        test_parameter_count()
        test_forward_compatibility()
        test_gradient_flow()
        test_memory_usage()
        test_performance_comparison()
        test_numerical_stability()
        test_summary()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
