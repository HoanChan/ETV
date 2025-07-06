import pytest
import torch
from mmocr.registry import MODELS
from models.decoders.table_master_concat_decoder import TableMasterConcatDecoder
from models.dictionaries.table_master_dictionary import TableMasterDictionary
from models.layers.tmd_layer import TMDLayer
from models.encoders.positional_encoding import PositionalEncoding

# Dummy layer for testing
class DummyLayer(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
    def forward(self, query, key, value, attn_masks=None):
        # Just return query with extra dim for concat
        return query + 1

@pytest.fixture
def dummy_decoder():
    class DummyEmbedding(torch.nn.Module):
        def forward(self, x):
            return x + 1
    class DummyPositionalEncoding(torch.nn.Module):
        def forward(self, x):
            return x + 2
    class DummyNorm(torch.nn.Module):
        def forward(self, x):
            return x * 2
    class DummyFC(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=-1)
    dummy_dict = {'type': 'BaseDictionary', 'dict_file': 'src/data/structure_vocab.txt'}
    decoder = TableMasterConcatDecoder(dictionary=dummy_dict)
    # Patch layers
    decoder.embedding = DummyEmbedding()
    decoder.positional_encoding = DummyPositionalEncoding()
    decoder.decoder_layers = torch.nn.ModuleList([DummyLayer(8), DummyLayer(8)])
    decoder.cls_layer = torch.nn.ModuleList([DummyLayer(8), DummyLayer(8)])
    decoder.bbox_layer = torch.nn.ModuleList([DummyLayer(8), DummyLayer(8)])
    decoder.norm = DummyNorm()
    decoder.cls_fc = DummyFC()
    decoder.bbox_fc = DummyFC()
    return decoder

@pytest.mark.parametrize("N,T,C,H,W", [
    (1, 2, 4, 2, 2),
    (2, 3, 5, 3, 3),
    (1, 1, 1, 1, 1), # edge: minimal
    (0, 2, 4, 2, 2), # edge: batch size 0
    (1, 0, 4, 2, 2), # edge: sequence length 0
])
def test_decode_shapes(dummy_decoder, N, T, C, H, W):
    decoder = dummy_decoder
    tgt_seq = torch.zeros((N, T, C))
    feature = torch.zeros((N, C, H, W))
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
    C = 4
    H = W = int(mask_shape[0][1] ** 0.5)
    tgt_seq = torch.zeros((N, T, C))
    feature = torch.zeros((N, C, H, W))
    src_mask = torch.zeros(mask_shape[0], dtype=torch.bool)
    tgt_mask = torch.zeros(mask_shape[1], dtype=torch.bool)
    out_cls, out_bbox = decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
    assert out_cls.shape[0] == N
    assert out_bbox.shape[0] == N

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decode_dtypes(dummy_decoder, dtype):
    decoder = dummy_decoder
    N, T, C, H, W = 2, 2, 4, 2, 2
    tgt_seq = torch.zeros((N, T, C), dtype=dtype)
    feature = torch.zeros((N, C, H, W), dtype=dtype)
    src_mask = torch.zeros((N, H*W), dtype=torch.bool)
    tgt_mask = torch.zeros((N, T, T), dtype=torch.bool)
    out_cls, out_bbox = decoder.decode(tgt_seq, feature, src_mask, tgt_mask)
    assert out_cls.dtype == dtype or out_cls.dtype == torch.float32
    assert out_bbox.dtype == dtype or out_bbox.dtype == torch.float32
