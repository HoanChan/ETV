import pytest
import torch
from types import SimpleNamespace
from mmocr.utils import register_all_modules
register_all_modules()
from models.dictionaries.table_master_dictionary import TableMasterDictionary
from models.decoders.table_master_decoder import TableMasterDecoder

DICT_FILE = 'src/data/structure_alphabet.txt'

# DummyTextRecogDataSample mô phỏng TableMasterDataSample thật
class DummyTextRecogDataSample:
    def __init__(self, seq, pad_idx=0, max_len=8):
        padded = torch.full((max_len,), pad_idx, dtype=torch.long)
        padded[:len(seq)] = torch.tensor(seq, dtype=torch.long)
        self.gt_tokens = SimpleNamespace(padded_indexes=padded)

# Sử dụng d_model=512, feat_size=240 (6*40), các shape hợp lý cho MMOCR
@pytest.mark.parametrize("n_layers, n_head, d_model, size, d_ff, dropout, max_seq_len, batch, h, w", [
    (2, 2, 512, 512, 2048, 0.1, 8, 2, 6, 40), # normal
    (1, 1, 512, 512, 512, 0.0, 4, 1, 6, 40),  # minimal
    (3, 4, 512, 512, 4096, 0.5, 12, 3, 6, 40), # larger
])
def test_forward_train_shapes(n_layers, n_head, d_model, size, d_ff, dropout, max_seq_len, batch, h, w):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(
        n_layers=n_layers, n_head=n_head, d_model=d_model,
        decoder={
            'self_attn': {'headers': n_head, 'd_model': d_model, 'dropout': dropout},
            'src_attn': {'headers': n_head, 'd_model': d_model, 'dropout': dropout},
            'feed_forward': {'d_model': d_model, 'd_ff': d_ff, 'dropout': dropout},
            'size': size,
            'dropout': dropout
        },
        dictionary=dictionary, max_seq_len=max_seq_len
    )
    feat = torch.randn(batch, d_model, h, w)
    data_samples = [DummyTextRecogDataSample([1,2,3], pad_idx=dictionary.padding_idx, max_len=max_seq_len) for _ in range(batch)]
    cls_out, bbox_out = model.forward_train(feat=feat, data_samples=data_samples)
    # In training, output length is input_length - 1 due to teacher forcing (using [:,:-1])
    expected_len = max_seq_len - 1  
    assert cls_out.shape == (batch, expected_len, dictionary.num_classes)
    assert bbox_out.shape == (batch, expected_len, 4)
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
    print("mask_bool=", (mask[0, 0] == 0).cpu().numpy().astype(int))
    # Shape should be [N, 1, l_tgt, l_tgt] for multi-head attention
    assert mask.shape == (1, 1, len(seq), len(seq))

@pytest.mark.parametrize("n_layers, n_head, d_model, size, d_ff, dropout, max_seq_len, batch, h, w", [
    (2, 2, 512, 512, 2048, 0.1, 8, 2, 6, 40),
])
def test_forward_test_shapes(n_layers, n_head, d_model, size, d_ff, dropout, max_seq_len, batch, h, w):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(
        n_layers=n_layers, n_head=n_head, d_model=d_model,
        decoder={
            'self_attn': {'headers': n_head, 'd_model': d_model, 'dropout': dropout},
            'src_attn': {'headers': n_head, 'd_model': d_model, 'dropout': dropout},
            'feed_forward': {'d_model': d_model, 'd_ff': d_ff, 'dropout': dropout},
            'size': size,
            'dropout': dropout
        },
        dictionary=dictionary, max_seq_len=max_seq_len
    )
    feat = torch.randn(batch, d_model, h, w)
    cls_out, bbox_out = model.forward_test(feat=feat)
    # In test/inference, output length is max_seq_len + 1 due to greedy decoding 
    expected_len = max_seq_len + 1
    assert cls_out.shape == (batch, expected_len, dictionary.num_classes)
    assert bbox_out.shape == (batch, expected_len, 4)
    assert (cls_out >= 0).all() and (cls_out <= 1).all()
    assert not torch.isnan(cls_out).any()
    assert not torch.isnan(bbox_out).any()

@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_device_compatibility(device):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(dictionary=dictionary, d_model=512)
    model = model.to(device)
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
    # In training, output length is input_length - 1 
    expected_len = 8 - 1  # max_len - 1
    assert cls_out.shape == (1, expected_len, dictionary.num_classes)
    assert bbox_out.shape == (1, expected_len, 4)

@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_decoder_layers_count(n_layers):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    model = TableMasterDecoder(n_layers=n_layers, dictionary=dictionary, d_model=512)
    assert len(model.decoder_layers) == max(0, n_layers-1)
    assert len(model.cls_layer) == 1
    assert len(model.bbox_layer) == 1

@pytest.mark.parametrize("d_model, n_head, d_ff, expect_error", [
    (512, 2, 2048, False),
    (512, 0, 2048, True),   # n_head=0 causes division by zero
    (0, 2, 2048, False),    # d_model=0 is allowed (just warnings)
    (-8, 2, 2048, True),    # negative d_model may cause issues
])
def test_invalid_layer_params(d_model, n_head, d_ff, expect_error):
    dictionary = TableMasterDictionary(dict_file=DICT_FILE, with_padding=True, with_start=True)
    if expect_error:
        with pytest.raises(Exception):
            TableMasterDecoder(
                d_model=d_model, n_head=n_head,
                decoder={
                    'self_attn': {'headers': n_head, 'd_model': d_model, 'dropout': 0.0},
                    'src_attn': {'headers': n_head, 'd_model': d_model, 'dropout': 0.0},
                    'feed_forward': {'d_model': d_model, 'd_ff': d_ff, 'dropout': 0.0},
                    'size': d_model,
                    'dropout': 0.0
                },
                dictionary=dictionary
            )
    else:
        TableMasterDecoder(
            d_model=d_model, n_head=n_head,
            decoder={
                'self_attn': {'headers': n_head, 'd_model': d_model, 'dropout': 0.0},
                'src_attn': {'headers': n_head, 'd_model': d_model, 'dropout': 0.0},
                'feed_forward': {'d_model': d_model, 'd_ff': d_ff, 'dropout': 0.0},
                'size': d_model,
                'dropout': 0.0
            },
            dictionary=dictionary
        )

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
    expected_len = 8 - 1  # max_len - 1 
    assert cls_out.shape == (1, expected_len, dictionary.num_classes)
    assert bbox_out.shape == (1, expected_len, 4)
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
