import pytest
import torch
from models.layers.tmd_layer import TMDLayer

@pytest.mark.parametrize("d_model, n_head, d_inner, attn_drop, ffn_drop, input_shape, expect_error", [
    # Normal case
    (32, 4, 64, 0.1, 0.1, (2, 10, 32), False),
    # Edge: d_model=1
    (1, 1, 2, 0.0, 0.0, (1, 1, 1), False),
    # Edge: attn_drop=1.0 (all dropped)
    (16, 2, 32, 1.0, 0.5, (3, 5, 16), False),
    # Edge: ffn_drop=1.0 (all dropped)
    (16, 2, 32, 0.5, 1.0, (3, 5, 16), False),
    # Edge: n_head does not divide d_model
    (10, 3, 20, 0.1, 0.1, (2, 4, 10), True),
    # Edge: input last dim != d_model
    (8, 2, 16, 0.1, 0.1, (2, 4, 7), True),
    # Edge: negative d_model
    (-8, 2, 16, 0.1, 0.1, (2, 4, 8), True),
    # Edge: zero n_head
    (8, 0, 16, 0.1, 0.1, (2, 4, 8), True),
])
def test_tmd_layer_forward(d_model, n_head, d_inner, attn_drop, ffn_drop, input_shape, expect_error):
    if expect_error:
        with pytest.raises(Exception):
            layer = TMDLayer(d_model, n_head, d_inner, attn_drop, ffn_drop)
            x = torch.randn(*input_shape)
            layer(x, x, x)
    else:
        layer = TMDLayer(d_model, n_head, d_inner, attn_drop, ffn_drop)
        x = torch.randn(*input_shape)
        # The layer expects (query, key, value)
        out = layer(x, x, x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

@pytest.mark.parametrize("value_fn", [
    lambda shape: torch.zeros(shape),
    lambda shape: torch.ones(shape),
    lambda shape: torch.full(shape, 1e6),
    lambda shape: torch.full(shape, -1e6),
])
def test_tmd_layer_forward_special_values(value_fn):
    d_model, n_head, d_inner = 8, 2, 16
    attn_drop, ffn_drop = 0.1, 0.1
    shape = (2, 4, d_model)
    layer = TMDLayer(d_model, n_head, d_inner, attn_drop, ffn_drop)
    x = value_fn(shape)
    out = layer(x, x, x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_tmd_layer_device(device):
    d_model, n_head, d_inner = 8, 2, 16
    attn_drop, ffn_drop = 0.1, 0.1
    shape = (2, 4, d_model)
    layer = TMDLayer(d_model, n_head, d_inner, attn_drop, ffn_drop).to(device)
    x = torch.randn(*shape, device=device)
    out = layer(x, x, x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

@pytest.mark.parametrize("operation_order", [
    None,
    ('norm', 'self_attn', 'norm', 'ffn'),
    ('self_attn', 'ffn'),
])
def test_tmd_layer_operation_order(operation_order):
    d_model, n_head, d_inner = 8, 2, 16
    attn_drop, ffn_drop = 0.1, 0.1
    shape = (2, 4, d_model)
    layer = TMDLayer(d_model, n_head, d_inner, attn_drop, ffn_drop, operation_order=operation_order)
    x = torch.randn(*shape)
    out = layer(x, x, x)
    assert out.shape == x.shape


def test_tmd_layer_backward():
    d_model, n_head, d_inner = 8, 2, 16
    attn_drop, ffn_drop = 0.1, 0.1
    shape = (2, 4, d_model)
    layer = TMDLayer(d_model, n_head, d_inner, attn_drop, ffn_drop)
    x = torch.randn(*shape, requires_grad=True)
    out = layer(x, x, x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
