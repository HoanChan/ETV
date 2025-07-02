import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Import the module directly without patching to avoid registry conflicts
from models.backbones.table_resnet_extra import TableResNetExtra, conv3x3, conv1x1, get_gcb_config


class TestTableResNetExtra:
    """Test cases for TableResNetExtra backbone"""

    def test_conv3x3_function(self):
        """Test conv3x3 helper function"""
        conv = conv3x3(64, 128, stride=2)
        assert isinstance(conv, nn.Conv2d)
        assert conv.in_channels == 64
        assert conv.out_channels == 128
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (2, 2)
        assert conv.padding == (1, 1)
        assert conv.bias is None

    def test_conv1x1_function(self):
        """Test conv1x1 helper function"""
        conv = conv1x1(64, 128, stride=2)
        assert isinstance(conv, nn.Conv2d)
        assert conv.in_channels == 64
        assert conv.out_channels == 128
        assert conv.kernel_size == (1, 1)
        assert conv.stride == (2, 2)
        assert conv.bias is None

    def test_get_gcb_config_function(self):
        """Test get_gcb_config helper function"""
        # Test None config
        assert get_gcb_config(None, 0) is None
        
        # Test config with False layer
        config = {'layers': [False, True, False, True]}
        assert get_gcb_config(config, 0) is None
        assert get_gcb_config(config, 1) == config
        
        # Test config with True layer
        assert get_gcb_config(config, 1) == config

    def test_init_basic(self):
        """Test basic initialization without GCB config"""
        layers = [2, 2, 2, 2]
        model = TableResNetExtra(layers=layers)
        
        # Check basic attributes - inplanes changes after layer construction
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'layer3')
        assert hasattr(model, 'layer4')
        
        # Check MaxPool3 difference from ResNetExtra
        assert model.maxpool3.kernel_size == 2  # PyTorch returns int for symmetric kernels
        assert model.maxpool3.stride == 2

    def test_init_with_custom_input_dim(self):
        """Test initialization with custom input dimension"""
        layers = [1, 1, 1, 1]
        model = TableResNetExtra(layers=layers, input_dim=1)
        
        assert model.conv1.in_channels == 1

    def test_init_with_gcb_config(self):
        """Test initialization with GCB configuration"""
        layers = [1, 1, 1, 1]
        gcb_config = {
            'layers': [True, False, True, False],
            'ratio': 1.0/8,
            'pooling_type': 'att',
            'fusion_type': 'channel_add'
        }
        
        # Mock ContextBlock to avoid import issues
        with patch('models.backbones.table_resnet_extra.ContextBlock') as mock_cb:
            mock_cb.return_value = nn.Identity()
            model = TableResNetExtra(layers=layers, gcb_config=gcb_config)
            
            # ContextBlock should be called for layers 0 and 2
            assert mock_cb.call_count == 2

    def test_init_with_init_cfg(self):
        """Test initialization with init_cfg"""
        layers = [1, 1, 1, 1]
        init_cfg = {'type': 'Xavier', 'layer': 'Conv2d'}
        model = TableResNetExtra(layers=layers, init_cfg=init_cfg)
        
        assert model.init_cfg == init_cfg

    def test_invalid_layers_assertion(self):
        """Test that assertion fails for insufficient layers"""
        with pytest.raises(AssertionError):
            TableResNetExtra(layers=[1, 1, 1])  # Only 3 layers, need >= 4

    def test_forward_shape(self):
        """Test forward pass output shapes"""
        layers = [1, 1, 1, 1]
        model = TableResNetExtra(layers=layers)
        model.eval()
        
        # Test input: batch_size=2, channels=3, height=48, width=160
        x = torch.randn(2, 3, 48, 160)
        
        with torch.no_grad():
            features = model(x)
        
        # Should return 3 feature maps
        assert len(features) == 3
        
        # Check approximate output sizes (based on comments in forward)
        # f[0]: after conv3 + bn3 + relu3 (24, 80)
        assert features[0].shape[2] == 24  # height
        assert features[0].shape[3] == 80   # width
        assert features[0].shape[1] == 256  # channels
        
        # f[1]: after conv4 + bn4 + relu4 (12, 40)  
        assert features[1].shape[2] == 12   # height
        assert features[1].shape[3] == 40   # width
        assert features[1].shape[1] == 256  # channels
        
        # f[2]: after conv6 + bn6 + relu6 (6, 20) - note: different from comment
        assert features[2].shape[2] == 6    # height
        assert features[2].shape[3] == 20   # width (80->40->20 due to (2,2) pooling)
        assert features[2].shape[1] == 512  # channels

    def test_init_weights_with_init_cfg(self):
        """Test init_weights with init_cfg"""
        layers = [1, 1, 1, 1]
        init_cfg = {'type': 'Xavier', 'layer': 'Conv2d'}
        model = TableResNetExtra(layers=layers, init_cfg=init_cfg)
        
        # Mock mmengine initialize function
        with patch('mmengine.model.initialize') as mock_init:
            model.init_weights()
            mock_init.assert_called_once_with(model, init_cfg)

    def test_init_weights_without_init_cfg(self):
        """Test init_weights without init_cfg (default initialization)"""
        layers = [1, 1, 1, 1]
        model = TableResNetExtra(layers=layers)
        
        # Store original weights
        conv1_weight = model.conv1.weight.clone()
        bn1_weight = model.bn1.weight.clone()
        
        model.init_weights()
        
        # Weights should be different after initialization
        assert not torch.equal(conv1_weight, model.conv1.weight)
        # BN weights should be initialized to 1
        assert torch.allclose(model.bn1.weight, torch.ones_like(model.bn1.weight))

    @patch('models.backbones.table_resnet_extra.ContextBlock')
    def test_make_layer_with_gcb_failure(self, mock_cb):
        """Test _make_layer when ContextBlock creation fails"""
        mock_cb.side_effect = Exception("ContextBlock failed")
        
        layers = [1, 1, 1, 1]
        gcb_config = {
            'layers': [True, False, False, False],
            'ratio': 1.0/16
        }
        
        # Should not raise exception, should print warning
        with patch('builtins.print') as mock_print:
            model = TableResNetExtra(layers=layers, gcb_config=gcb_config)
            mock_print.assert_called()
            assert "Warning: Could not create ContextBlock" in str(mock_print.call_args)

    def test_different_layer_configurations(self):
        """Test with different layer configurations"""
        test_configs = [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 4, 6, 3],
            [1, 2, 3, 4]
        ]
        
        for layers in test_configs:
            model = TableResNetExtra(layers=layers)
            assert hasattr(model, 'layer1')
            assert hasattr(model, 'layer2')
            assert hasattr(model, 'layer3')
            assert hasattr(model, 'layer4')

    def test_comparison_with_resnet_extra(self):
        """Test key differences with ResNetExtra"""
        layers = [1, 1, 1, 1]
        model = TableResNetExtra(layers=layers)
        
        # Key difference: maxpool3 should use (2,2) instead of (2,1)
        assert model.maxpool3.kernel_size == 2  # PyTorch returns int for symmetric kernels
        assert model.maxpool3.stride == 2
        
        # This is different from ResNetExtra which uses (2,1)
        # We can't import ResNetExtra here, but we document the difference

    def test_forward_with_different_input_sizes(self):
        """Test forward pass with different input sizes"""
        layers = [1, 1, 1, 1]
        model = TableResNetExtra(layers=layers)
        model.eval()
        
        test_sizes = [
            (1, 3, 32, 128),   # smaller
            (2, 3, 48, 160),   # standard
            (1, 3, 64, 256),   # larger
        ]
        
        for batch_size, channels, height, width in test_sizes:
            x = torch.randn(batch_size, channels, height, width)
            with torch.no_grad():
                features = model(x)
                
            assert len(features) == 3
            assert all(f.shape[0] == batch_size for f in features)
            assert all(f.dim() == 4 for f in features)  # BCHW format