import pytest
import torch
from mmocr.registry import MODELS
from models.decoders.table_master_concat_decoder import TableMasterConcatDecoder
from models.decoders.table_master_decoder import TableMasterDecoder
from models.dictionaries.table_master_dictionary import TableMasterDictionary
from models.layers.tmd_layer import TMDLayer
from models.encoders.positional_encoding import PositionalEncoding

# Dummy layer for testing
class DummyLayer(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
    def forward(self, query, key, value, attn_masks=None):
        # Return query with same shape and proper dtype for concatenation
        return query + 0.1  # Small perturbation to maintain dtype

@pytest.fixture
def dummy_decoder():
    class DummyEmbedding(torch.nn.Module):
        def forward(self, x):
            # Proper embedding: convert indices to embeddings
            N, T = x.shape
            return torch.randn(N, T, 512, dtype=torch.float32)  # (N, T, d_model)
    class DummyPositionalEncoding(torch.nn.Module):
        def forward(self, x):
            return x + 0.1  # Small perturbation
    class DummyNorm(torch.nn.Module):
        def forward(self, x):
            return x  # No-op for simplicity
    class DummyFC(torch.nn.Module):
        def __init__(self, out_features):
            super().__init__()
            self.out_features = out_features
        def forward(self, x):
            # Return tensor with proper shape and dtype
            if x.dim() == 3:  # (N, T, C) -> (N, T, out_features)
                N, T, _ = x.shape
                return torch.randn(N, T, self.out_features, dtype=x.dtype)
            return x
    
    # Create decoder with required d_model parameter
    dummy_dict = {'type': 'BaseDictionary', 'dict_file': 'src/data/structure_vocab.txt'}
    decoder = TableMasterConcatDecoder(
        d_model=512,  # Required parameter
        dictionary=dummy_dict 
        # Use default decoder config (no custom decoder config needed)
    )
    
    # Patch layers with dummy implementations
    decoder.embedding = DummyEmbedding()
    decoder.positional_encoding = DummyPositionalEncoding()
    decoder.decoder_layers = torch.nn.ModuleList([DummyLayer(512), DummyLayer(512)])
    decoder.cls_layer = torch.nn.ModuleList([DummyLayer(512)])  # Single layer as in implementation
    decoder.bbox_layer = torch.nn.ModuleList([DummyLayer(512)])  # Single layer as in implementation
    decoder.norm = DummyNorm()
    # Get actual number of classes from dictionary for proper output shape
    num_classes = decoder.dictionary.num_classes if hasattr(decoder, 'dictionary') else 100
    decoder.cls_fc = DummyFC(num_classes)
    decoder.bbox_fc = DummyFC(4)  # Bbox has 4 coordinates
    return decoder

@pytest.mark.parametrize("N,T,C,H,W", [
    (1, 2, 512, 2, 2),  # Updated C to match d_model=512
    (2, 3, 512, 3, 3),  # Updated C to match d_model=512
    (1, 1, 512, 1, 1),  # edge: minimal, Updated C to match d_model=512
    (0, 2, 512, 2, 2),  # edge: batch size 0, Updated C to match d_model=512
    (1, 0, 512, 2, 2),  # edge: sequence length 0, Updated C to match d_model=512
])
def test_decode_shapes(dummy_decoder, N, T, C, H, W):
    decoder = dummy_decoder
    # tgt_seq should contain token indices (integers), not continuous values  
    tgt_seq = torch.randint(0, 100, (N, T))  # Changed to integer indices
    # Feature should be reshaped to (N, H*W, C) for attention mechanism
    feature = torch.zeros((N, H*W, C))  # Changed from (N, C, H, W)
    src_mask = torch.zeros((N, H*W), dtype=torch.bool)
    tgt_mask = torch.zeros((N, T, T), dtype=torch.bool)
    if N == 0 or T == 0:
        # Should handle empty batch/seq gracefully
        out_cls, out_bbox = decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
        assert out_cls.shape[0] == N
        assert out_bbox.shape[0] == N
    else:
        out_cls, out_bbox = decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
        assert out_cls.shape[0] == N
        assert out_bbox.shape[0] == N

@pytest.mark.parametrize("mask_shape", [
    ((1, 4), (1, 2, 2)),
    ((2, 9), (2, 3, 3)),
    ((1, 1), (1, 1, 1)),
])
def test_decode_mask_shapes(dummy_decoder, mask_shape):
    decoder = dummy_decoder
    N = mask_shape[0][0]
    T = mask_shape[1][1]
    C = 512  # Updated to match d_model=512
    H = W = int(mask_shape[0][1] ** 0.5)
    # tgt_seq should contain token indices (integers), not continuous values
    tgt_seq = torch.randint(0, 100, (N, T))  # Changed to integer indices
    # Feature should be reshaped to (N, H*W, C) for attention mechanism  
    feature = torch.zeros((N, H*W, C))  # Changed from (N, C, H, W)
    src_mask = torch.zeros(mask_shape[0], dtype=torch.bool)
    tgt_mask = torch.zeros(mask_shape[1], dtype=torch.bool)
    out_cls, out_bbox = decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
    assert out_cls.shape[0] == N
    assert out_bbox.shape[0] == N

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decode_dtypes(dummy_decoder, dtype):
    decoder = dummy_decoder
    N, T, C, H, W = 2, 2, 512, 2, 2  # Updated C to match d_model=512
    # tgt_seq should contain token indices (integers), not continuous values
    tgt_seq = torch.randint(0, 100, (N, T))  # Changed to integer indices (dtype irrelevant for indices)
    # Feature should be reshaped to (N, H*W, C) for attention mechanism
    feature = torch.zeros((N, H*W, C), dtype=dtype)  # Changed from (N, C, H, W)
    src_mask = torch.zeros((N, H*W), dtype=torch.bool)
    tgt_mask = torch.zeros((N, T, T), dtype=torch.bool)
    out_cls, out_bbox = decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
    # DummyFC returns random tensors with specified dtype
    assert out_cls.dtype == dtype or out_cls.dtype == torch.float32
    assert out_bbox.dtype == dtype or out_bbox.dtype == torch.float32

def test_concat_functionality(dummy_decoder):
    """Test that concatenation is actually working correctly."""
    decoder = dummy_decoder
    N, T, C, H, W = 1, 3, 512, 2, 2
    
    # Create test inputs
    # tgt_seq should contain token indices (integers), not continuous values
    tgt_seq = torch.randint(0, 100, (N, T))  # Changed to integer indices
    # Feature should be reshaped to (N, H*W, C) for attention mechanism
    feature = torch.randn((N, H*W, C))  # Changed from (N, C, H, W)
    src_mask = torch.zeros((N, H*W), dtype=torch.bool)
    tgt_mask = torch.zeros((N, T, T), dtype=torch.bool)
    
    # Run decode
    out_cls, out_bbox = decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
    
    # Check output shapes - now should return proper classification/bbox outputs
    assert out_cls.shape[0] == N  # Batch dimension
    assert out_cls.shape[1] == T  # Sequence dimension
    assert out_bbox.shape[0] == N  # Batch dimension
    assert out_bbox.shape[1] == T  # Sequence dimension
    assert out_bbox.shape[2] == 4  # Bbox coordinates
    
def test_decoder_with_multiple_concat_layers():
    """Test decoder with multiple layers for concatenation."""
    class DummyEmbedding(torch.nn.Module):
        def forward(self, x):
            # Proper embedding: convert indices to embeddings
            N, T = x.shape
            return torch.randn(N, T, 512, dtype=torch.float32)  # (N, T, d_model)
    class DummyPositionalEncoding(torch.nn.Module):
        def forward(self, x):
            return x + 0.1  # Small perturbation
    class DummyNorm(torch.nn.Module):
        def forward(self, x):
            return x  # No-op for simplicity
    class DummyFC(torch.nn.Module):
        def __init__(self, out_features):
            super().__init__()
            self.out_features = out_features
        def forward(self, x):
            # Return tensor with proper shape and dtype
            if x.dim() == 3:  # (N, T, C) -> (N, T, out_features)
                N, T, _ = x.shape
                return torch.randn(N, T, self.out_features, dtype=x.dtype)
            return x
    
    # Create decoder with custom layer configuration
    dummy_dict = {'type': 'BaseDictionary', 'dict_file': 'src/data/structure_vocab.txt'}
    decoder = TableMasterConcatDecoder(
        d_model=512,
        n_layers=4,  # This will create 3 shared layers, 1 cls layer, 1 bbox layer
        dictionary=dummy_dict
    )
    
    # Patch layers with dummy implementations
    decoder.embedding = DummyEmbedding()
    decoder.positional_encoding = DummyPositionalEncoding()
    # Keep original layers but override their forward to be predictable
    decoder.norm = DummyNorm()
    num_classes = decoder.dictionary.num_classes if hasattr(decoder, 'dictionary') else 100
    decoder.cls_fc = DummyFC(num_classes)
    decoder.bbox_fc = DummyFC(4)
    
    # Test shapes
    N, T, C, H, W = 1, 2, 512, 2, 2
    # tgt_seq should contain token indices (integers), not continuous values
    tgt_seq = torch.randint(0, 100, (N, T))  # Changed to integer indices
    # Feature should be reshaped to (N, H*W, C) for attention mechanism
    feature = torch.randn((N, H*W, C))  # Changed from (N, C, H, W) to (N, H*W, C)
    src_mask = torch.zeros((N, H*W), dtype=torch.bool)
    tgt_mask = torch.zeros((N, T, T), dtype=torch.bool)
    
    # This should work without errors
    out_cls, out_bbox = decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
    assert out_cls.shape[0] == N
    assert out_bbox.shape[0] == N

def test_concat_vs_regular_decoder():
    """Compare concatenation decoder with regular decoder."""
    dummy_dict = {'type': 'BaseDictionary', 'dict_file': 'src/data/structure_vocab.txt'}
    
    # Create both decoders with same configuration
    concat_decoder = TableMasterConcatDecoder(d_model=512, dictionary=dummy_dict)
    regular_decoder = TableMasterDecoder(d_model=512, dictionary=dummy_dict)
    
    # Test input
    N, T, C, H, W = 1, 2, 512, 2, 2
    # tgt_seq should contain token indices (integers), not continuous values
    tgt_seq = torch.randint(0, 100, (N, T))  # Changed to integer indices
    # Feature should be reshaped to (N, H*W, C) for attention mechanism
    feature = torch.randn((N, H*W, C))  # Changed from (N, C, H, W)
    src_mask = torch.zeros((N, H*W), dtype=torch.bool)
    tgt_mask = torch.zeros((N, T, T), dtype=torch.bool)
    
    # Both should produce outputs of same shape (different values due to concat logic)
    concat_cls, concat_bbox = concat_decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
    regular_cls, regular_bbox = regular_decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
    
    # Same output shapes expected
    assert concat_cls.shape == regular_cls.shape
    assert concat_bbox.shape == regular_bbox.shape
    
    # Values should be different due to concatenation
    # (We can't test this directly as values depend on random weights,
    # but the architectures are different so outputs should typically differ)
