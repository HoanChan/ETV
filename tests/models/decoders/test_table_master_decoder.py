import pytest
import torch
from types import SimpleNamespace
from mmocr.utils import register_all_modules
register_all_modules()
from models.dictionaries.table_master_dictionary import TableMasterDictionary
from models.decoders.table_master_decoder import TableMasterDecoder

DICT_FILE = 'src/data/structure_alphabet.txt'

# DummyTextRecogDataSample mô phỏng TextRecogDataSample thật
class DummyTextRecogDataSample:
    def __init__(self, seq, pad_idx=0, max_len=8):
        padded = torch.full((max_len,), pad_idx, dtype=torch.long)
        padded[:len(seq)] = torch.tensor(seq, dtype=torch.long)
        self.gt_text = SimpleNamespace(padded_indexes=padded)

# Sử dụng d_model=512, feat_size=240 (6*40), các shape hợp lý cho MMOCR
@pytest.mark.parametrize("n_layers, n_head, d_model, feat_size, d_inner, attn_drop, ffn_drop, feat_pe_drop, max_seq_len, batch, h, w", [
    (2, 2, 512, 240, 2048, 0.1, 0.1, 0.1, 8, 2, 6, 40), # normal
    (1, 1, 512, 240, 512, 0.0, 0.0, 0.0, 4, 1, 6, 40),  # minimal
    (3, 4, 512, 240, 4096, 0.5, 0.5, 0.2, 12, 3, 6, 40), # larger
])
def test_forward_train_shapes(n_layers, n_head, d_model, feat_size, d_inner, attn_drop, ffn_drop, feat_pe_drop, max_seq_len, batch, h, w):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(
        n_layers=n_layers, n_head=n_head, d_model=d_model, feat_size=feat_size,
        d_inner=d_inner, attn_drop=attn_drop, ffn_drop=ffn_drop, feat_pe_drop=feat_pe_drop,
        dictionary=dictionary, max_seq_len=max_seq_len
    )
    feat = torch.randn(batch, d_model, h, w)
    data_samples = [DummyTextRecogDataSample([1,2,3], pad_idx=dictionary.padding_idx, max_len=max_seq_len) for _ in range(batch)]
    cls_out, bbox_out = model.forward_train(feat=feat, data_samples=data_samples)
    assert cls_out.shape == (batch, max_seq_len, dictionary.num_classes)
    assert bbox_out.shape == (batch, max_seq_len, 4)
    assert not torch.isnan(cls_out).any()
    assert not torch.isnan(bbox_out).any()

@pytest.mark.parametrize("pad_idx, seq", [
    (0, [1,2,3,0,0]),
    (9, [1,2,3,9,9]),
])
def test_make_target_mask(pad_idx, seq):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True, padding_token=str(pad_idx))
    model = TableMasterDecoder(dictionary=dictionary, d_model=512)
    tgt = torch.tensor([seq])
    mask = model.make_target_mask(tgt, device=tgt.device)
    # In ra mask thực tế để cập nhật expect_mask nếu cần
    print("mask_bool=", (mask[0] == 0).cpu().numpy().astype(int))
    assert mask.shape[1] == mask.shape[2] == len(seq)

@pytest.mark.parametrize("n_layers, n_head, d_model, feat_size, d_inner, attn_drop, ffn_drop, feat_pe_drop, max_seq_len, batch, h, w", [
    (2, 2, 512, 240, 2048, 0.1, 0.1, 0.1, 8, 2, 6, 40),
])
def test_forward_test_shapes(n_layers, n_head, d_model, feat_size, d_inner, attn_drop, ffn_drop, feat_pe_drop, max_seq_len, batch, h, w):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(
        n_layers=n_layers, n_head=n_head, d_model=d_model, feat_size=feat_size,
        d_inner=d_inner, attn_drop=attn_drop, ffn_drop=ffn_drop, feat_pe_drop=feat_pe_drop,
        dictionary=dictionary, max_seq_len=max_seq_len
    )
    feat = torch.randn(batch, d_model, h, w)
    cls_out, bbox_out = model.forward_test(feat=feat)
    assert cls_out.shape == (batch, max_seq_len, dictionary.num_classes)
    assert bbox_out.shape == (batch, max_seq_len, 4)
    assert (cls_out >= 0).all() and (cls_out <= 1).all()
    assert not torch.isnan(cls_out).any()
    assert not torch.isnan(bbox_out).any()

@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_device_compatibility(device):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(dictionary=dictionary, d_model=512).to(device)
    feat = torch.randn(1, 512, 6, 40, device=device)
    data_samples = [DummyTextRecogDataSample([1,2,3], pad_idx=dictionary.padding_idx, max_len=8)]
    cls_out, bbox_out = model.forward_train(feat=feat, data_samples=data_samples)
    assert cls_out.device.type == device
    assert bbox_out.device.type == device

@pytest.mark.parametrize("seq_len", [1, 2, 8])
def test_forward_train_various_seq_len(seq_len):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(dictionary=dictionary, max_seq_len=8, d_model=512)
    feat = torch.randn(1, 512, 6, 40)
    data_samples = [DummyTextRecogDataSample([1]*seq_len, pad_idx=dictionary.padding_idx, max_len=8)]
    cls_out, bbox_out = model.forward_train(feat=feat, data_samples=data_samples)
    assert cls_out.shape == (1, 8, dictionary.num_classes)
    assert bbox_out.shape == (1, 8, 4)

@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_decoder_layers_count(n_layers):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(n_layers=n_layers, dictionary=dictionary, d_model=512)
    assert len(model.decoder_layers) == max(0, n_layers-1)
    assert len(model.cls_layer) == 1
    assert len(model.bbox_layer) == 1

@pytest.mark.parametrize("d_model, n_head, d_inner, expect_error", [
    (512, 2, 2048, False),
    (512, 0, 2048, True),
    (0, 2, 2048, True),
    (-8, 2, 2048, True),
])
def test_invalid_layer_params(d_model, n_head, d_inner, expect_error):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    if expect_error:
        with pytest.raises(Exception):
            TableMasterDecoder(d_model=d_model, n_head=n_head, d_inner=d_inner, dictionary=dictionary)
    else:
        TableMasterDecoder(d_model=d_model, n_head=n_head, d_inner=d_inner, dictionary=dictionary)

@pytest.mark.parametrize("value_fn", [
    lambda shape: torch.zeros(shape),
    lambda shape: torch.ones(shape),
    lambda shape: torch.full(shape, 1e6),
    lambda shape: torch.full(shape, -1e6),
])
def test_forward_train_special_values(value_fn):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(dictionary=dictionary, max_seq_len=8, d_model=512)
    feat = value_fn((1, 512, 6, 40))
    data_samples = [DummyTextRecogDataSample([1,2,3], pad_idx=dictionary.padding_idx, max_len=8)]
    cls_out, bbox_out = model.forward_train(feat=feat, data_samples=data_samples)
    assert cls_out.shape == (1, 8, dictionary.num_classes)
    assert bbox_out.shape == (1, 8, 4)
    assert not torch.isnan(cls_out).any()
    assert not torch.isnan(bbox_out).any()

def test_forward_train_backward():
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(dictionary=dictionary, max_seq_len=8, d_model=512)
    feat = torch.randn(1, 512, 6, 40, requires_grad=True)
    data_samples = [DummyTextRecogDataSample([1,2,3], pad_idx=dictionary.padding_idx, max_len=8)]
    cls_out, bbox_out = model.forward_train(feat=feat, data_samples=data_samples)
    loss = cls_out.sum() + bbox_out.sum()
    loss.backward()
    assert feat.grad is not None
    assert not torch.isnan(feat.grad).any()
