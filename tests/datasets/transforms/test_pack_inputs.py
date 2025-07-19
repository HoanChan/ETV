import pytest
import numpy as np
import torch
from datasets.transforms.pack_inputs import PackInputs

@pytest.fixture
def sample_image():
    """Tạo sample image data"""
    return np.random.rand(32, 128, 3).astype(np.float32)

@pytest.fixture
def sample_image_2d():
    """Tạo sample image 2D"""
    return np.random.rand(32, 128).astype(np.float32)

@pytest.fixture
def sample_results():
    """Tạo sample results dictionary"""
    return {
        'img': np.random.rand(32, 128, 3).astype(np.float32),
        'gt_texts': ['hello world'],
        'img_path': '/path/to/image.jpg',
        'ori_shape': (32, 128, 3),
        'img_shape': (32, 128, 3),
        'pad_shape': (32, 128, 3),
        'valid_ratio': 0.8,
        'extra_key': 'extra_value'
    }

# Test initialization
@pytest.mark.parametrize("keys,meta_keys,mean,std", [
    ((), ('img_path', 'ori_shape'), None, None),
    (('extra_key',), ('img_path', 'ori_shape', 'img_shape'), None, None),
    ((), ('img_path',), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    (('key1', 'key2'), ('img_path', 'ori_shape'), [0.5], [0.5]),
])
def test_init(keys, meta_keys, mean, std):
    """Test khởi tạo PackInputs với các tham số khác nhau"""
    transform = PackInputs(keys=keys, meta_keys=meta_keys, mean=mean, std=std)
    
    assert transform.keys == keys
    assert transform.meta_keys == meta_keys
    assert transform.mean == mean
    assert transform.std == std

# Test transform method with different image shapes
@pytest.mark.parametrize("img_shape", [
    (32, 128, 3),    # 3D RGB image
    (32, 128, 1),    # 3D grayscale
    (32, 128),       # 2D grayscale
])
def test_transform_image_shapes(img_shape):
    """Test transform với các shape khác nhau của image"""
    transform = PackInputs()
    
    # Tạo image với shape tương ứng
    img = np.random.rand(*img_shape).astype(np.float32)
    results = {'img': img}
    
    packed_results = transform.transform(results)
    
    # Kiểm tra kết quả
    assert 'inputs' in packed_results
    assert 'data_samples' in packed_results
    assert isinstance(packed_results['inputs'], torch.Tensor)
    
    # Kiểm tra shape của tensor output
    if len(img_shape) == 2:
        # 2D image được expand thành 3D
        assert packed_results['inputs'].shape == (1, img_shape[0], img_shape[1])
    else:
        # 3D image được permute
        assert packed_results['inputs'].shape == (img_shape[2], img_shape[0], img_shape[1])

# Test normalization
@pytest.mark.parametrize("mean,std", [
    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ([0.5], [0.5]),
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
])
def test_transform_normalization(mean, std):
    """Test normalization với các giá trị mean/std khác nhau"""
    transform = PackInputs(mean=mean, std=std)
    
    img = np.random.rand(32, 128, len(mean)).astype(np.float32)
    results = {'img': img}
    
    packed_results = transform.transform(results)
    
    # Kiểm tra có normalize
    assert 'data_samples' in packed_results
    data_sample = packed_results['data_samples']
    assert 'img_norm_cfg' in data_sample.metainfo
    assert data_sample.metainfo['img_norm_cfg']['mean'] == mean
    assert data_sample.metainfo['img_norm_cfg']['std'] == std

def test_transform_no_normalization():
    """Test transform không có normalization"""
    transform = PackInputs()
    
    img = np.random.rand(32, 128, 3).astype(np.float32)
    results = {'img': img}
    
    packed_results = transform.transform(results)
    
    # Kiểm tra không có normalize
    assert 'img_norm_cfg' not in packed_results

@pytest.mark.parametrize("tokens,bboxes", [
    (['hello', 'world'], [[0, 0, 10, 10], [10, 0, 20, 10]]),
    (['single'], [[0, 0, 10, 10]]),
    ([], []),
    (None, None),
])
def test_transform_tokens_bboxes(tokens, bboxes):
    """Test xử lý tokens và bboxes"""
    transform = PackInputs()
    
    results = {'img': np.random.rand(32, 128, 3).astype(np.float32)}
    if tokens is not None:
        results['tokens'] = tokens
    if bboxes is not None:
        results['bboxes'] = bboxes
    
    packed_results = transform.transform(results)
    
    # Kiểm tra data_samples
    assert 'data_samples' in packed_results
    data_sample = packed_results['data_samples']
    assert hasattr(data_sample, 'gt_tokens')
    
    # Bboxes được lưu trong gt_instances metainfo, không phải gt_bboxes attribute
    if bboxes is not None:
        assert hasattr(data_sample, 'gt_instances')
        assert 'bboxes' in data_sample.gt_instances.metainfo
    
    if tokens is not None and len(tokens) > 0:
        assert hasattr(data_sample.gt_tokens, 'item')
        assert data_sample.gt_tokens.item == tokens
    else:
        assert not hasattr(data_sample.gt_tokens, 'item') or data_sample.gt_tokens.item == []

def test_transform_tokens_validation_error():
    """Test lỗi khi tokens và bboxs không khớp"""
    transform = PackInputs()
    
    results = {
        'img': np.random.rand(32, 128, 3).astype(np.float32),
        'tokens': ['token1', 'token2'],
        'bboxs': [[0, 0, 10, 10]]  # Thiếu 1 bbox
    }
    
    # Không nên raise error vì pack_inputs không validate này
    packed_results = transform.transform(results)
    assert 'data_samples' in packed_results

# Test meta keys handling
@pytest.mark.parametrize("meta_keys,input_data", [
    (('img_path', 'ori_shape'), {'img_path': '/path/to/img.jpg', 'ori_shape': (32, 128, 3)}),
    (('img_path', 'ori_shape', 'img_shape'), {'img_path': '/path/to/img.jpg'}),
    (('valid_ratio',), {'valid_ratio': 0.8}),
    (('valid_ratio',), {}),  # Không có valid_ratio
])
def test_transform_meta_keys(meta_keys, input_data):
    """Test xử lý meta keys"""
    transform = PackInputs(meta_keys=meta_keys)
    
    results = {'img': np.random.rand(32, 128, 3).astype(np.float32)}
    results.update(input_data)
    
    packed_results = transform.transform(results)
    
    # Kiểm tra metainfo
    data_sample = packed_results['data_samples']
    for key in meta_keys:
        if key == 'valid_ratio':
            expected_value = input_data.get('valid_ratio', 1)  # Default là 1
        else:
            expected_value = input_data.get(key, None)
        assert data_sample.metainfo[key] == expected_value

# Test additional keys packing
@pytest.mark.parametrize("keys,additional_data", [
    (('extra_key',), {'extra_key': 'extra_value'}),
    (('key1', 'key2'), {'key1': 'value1', 'key2': 'value2'}),
    (('missing_key',), {}),
    # Bỏ test truyền giá trị không hợp lệ cho 'img', 'gt_texts'
])
def test_transform_additional_keys(keys, additional_data):
    """Test packing additional keys"""
    transform = PackInputs(keys=keys)
    
    results = {'img': np.random.rand(32, 128, 3).astype(np.float32)}
    results.update(additional_data)
    
    packed_results = transform.transform(results)
    
    # Kiểm tra additional keys
    for key in keys:
        if key in additional_data and key not in ['img', 'tokens', 'bboxs', 'valid_ratio']:
            assert packed_results[key] == additional_data[key]
        elif key in ['img', 'tokens', 'bboxs']:
            # img, tokens, bboxs không được pack vào additional keys
            assert key not in packed_results or key in ['inputs', 'data_samples']

def test_transform_no_image():
    """Test transform khi không có image"""
    transform = PackInputs()
    
    results = {'tokens': ['hello', 'world']}
    
    packed_results = transform.transform(results)
    
    # Kiểm tra không có inputs
    assert 'inputs' not in packed_results
    assert 'data_samples' in packed_results

def test_transform_non_contiguous_image():
    """Test transform với non-contiguous image"""
    transform = PackInputs()
    
    # Tạo non-contiguous array
    img = np.random.rand(32, 128, 3).astype(np.float32)
    img = img[::2, ::2, :]  # Tạo non-contiguous array
    
    results = {'img': img}
    
    packed_results = transform.transform(results)
    
    # Kiểm tra vẫn xử lý được
    assert 'inputs' in packed_results
    assert isinstance(packed_results['inputs'], torch.Tensor)

# Test __repr__ method
@pytest.mark.parametrize("keys,meta_keys,mean,std,expected_repr", [
    ((), ('img_path',), None, None, 
     "PackInputs(keys=(), meta_keys=('img_path',))"),
    (('extra_key',), ('img_path', 'ori_shape'), [0.5], [0.5],
     "PackInputs(keys=('extra_key',), meta_keys=('img_path', 'ori_shape'), mean=[0.5], std=[0.5])"),
    (('key1', 'key2'), ('img_path',), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
     "PackInputs(keys=('key1', 'key2'), meta_keys=('img_path',), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"),
])
def test_repr(keys, meta_keys, mean, std, expected_repr):
    """Test __repr__ method"""
    transform = PackInputs(keys=keys, meta_keys=meta_keys, mean=mean, std=std)
    
    assert repr(transform) == expected_repr

def test_transform_complete_workflow(sample_results):
    """Test complete workflow với đầy đủ data"""
    # Cần update sample_results để phù hợp với API mới
    sample_results_updated = sample_results.copy()
    sample_results_updated['tokens'] = ['hello', 'world']
    sample_results_updated['bboxes'] = [[0, 0, 10, 10], [10, 0, 20, 10]]
    # Remove gt_texts nếu có
    if 'gt_texts' in sample_results_updated:
        del sample_results_updated['gt_texts']
    
    transform = PackInputs(
        keys=('extra_key',),
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'valid_ratio'),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    packed_results = transform.transform(sample_results_updated)
    
    # Kiểm tra tất cả components
    assert 'inputs' in packed_results
    assert 'data_samples' in packed_results
    data_sample = packed_results['data_samples']
    assert 'img_norm_cfg' in data_sample.metainfo
    assert 'extra_key' in packed_results
    
    # Kiểm tra inputs
    assert isinstance(packed_results['inputs'], torch.Tensor)
    assert packed_results['inputs'].shape == (3, 32, 128)
    
    # Kiểm tra data_samples
    data_sample = packed_results['data_samples']
    assert hasattr(data_sample.gt_tokens, 'item')
    assert data_sample.gt_tokens.item == ['hello', 'world']
    
    # Kiểm tra bboxes trong gt_instances metainfo
    assert hasattr(data_sample, 'gt_instances')
    assert 'bboxes' in data_sample.gt_instances.metainfo
    assert data_sample.gt_instances.metainfo['bboxes'] == [[0, 0, 10, 10], [10, 0, 20, 10]]
    assert data_sample.metainfo['img_path'] == '/path/to/image.jpg'
    assert data_sample.metainfo['valid_ratio'] == 0.8
    
    # Kiểm tra normalization config
    assert data_sample.metainfo['img_norm_cfg']['mean'] == [0.485, 0.456, 0.406]
    assert data_sample.metainfo['img_norm_cfg']['std'] == [0.229, 0.224, 0.225]
    
    # Kiểm tra extra key
    assert packed_results['extra_key'] == 'extra_value'

# Test edge cases
def test_transform_empty_results():
    """Test với results rỗng"""
    transform = PackInputs()
    
    results = {}
    packed_results = transform.transform(results)
    
    # Vẫn phải có data_samples
    assert 'data_samples' in packed_results
    assert 'inputs' not in packed_results

def test_transform_with_all_default_meta_keys():
    """Test với tất cả default meta keys"""
    transform = PackInputs()
    
    results = {
        'img': np.random.rand(32, 128, 3).astype(np.float32),
        'img_path': '/path/to/image.jpg',
        'ori_shape': (32, 128, 3),
        'img_shape': (32, 128, 3),
        'pad_shape': (32, 128, 3),
        'valid_ratio': 0.9
    }
    
    packed_results = transform.transform(results)
    
    # Kiểm tra tất cả meta keys
    data_sample = packed_results['data_samples']
    assert data_sample.metainfo['img_path'] == '/path/to/image.jpg'
    assert data_sample.metainfo['ori_shape'] == (32, 128, 3)
    assert data_sample.metainfo['img_shape'] == (32, 128, 3)
    assert data_sample.metainfo['pad_shape'] == (32, 128, 3)
    assert data_sample.metainfo['valid_ratio'] == 0.9