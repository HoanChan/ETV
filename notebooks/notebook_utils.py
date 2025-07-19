"""
Notebook utilities for ETV project.
Common functions used across multiple notebooks for dataset loading, visualization and pipeline debugging.
"""

import sys
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmocr.utils import register_all_modules
from mmocr.registry import DATASETS, TRANSFORMS
from mmengine.dataset import Compose


def setup_environment():
    """Setup environment for notebooks by adding src to path and registering MMOCR modules."""
    # Add the src directory to path
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))
    
    # Register all MMOCR modules
    register_all_modules()
    print("✅ Environment setup complete - MMOCR modules registered")


def load_dataset(config_module, max_data=100, dataset_name="dataset"):
    """
    Load dataset from config with optional data limit.
    
    Args:
        config_module: Config module containing train_dataset
        max_data: Maximum number of samples to load (-1 for all)
        dataset_name: Name for logging purposes
    
    Returns:
        Built dataset object
    """
    dataset = config_module.train_dataset.copy()
    dataset['max_data'] = max_data
    dataset['pipeline'] = None
    
    print(f"\n🔄 Loading {dataset_name}...")
    dataset = DATASETS.build(dataset)
    print(f"✅ {dataset_name} loaded successfully: {len(dataset)} samples")
    print(f"✅ {dataset_name} ready for use!")
    
    return dataset


def print_dict(d, indent=2):
    """Prints a dictionary in a readable format with proper indentation."""
    print(' ' * (indent - 2) + '{')
    for key, value in d.items():
        print(' ' * indent + f"{key}: ", end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 2)
        elif isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                print('[')
                for item in value:
                    print_dict(item, indent + 2)
                print(' ' * indent + ']')
            else:
                print(value)
        else:
            print(value)
    print(' ' * (indent - 2) + '}')


def print_random_sample(dataset):
    """Print a random sample from the dataset in readable format."""
    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"=== Random Sample #{sample_idx} ===")
    print_dict(dataset[sample_idx])


def is_equal(a, b):
    """
    Check if two values are equal, handling numpy arrays and torch tensors properly.
    
    Args:
        a, b: Values to compare
    
    Returns:
        bool: True if values are equal
    """
    if isinstance(a, (np.ndarray, torch.Tensor)) and isinstance(b, (np.ndarray, torch.Tensor)):
        if isinstance(a, torch.Tensor):
            return bool(torch.equal(a, b))
        return bool(np.array_equal(a, b))
    
    try:
        result = a == b
        # If result is an array/tensor, reduce to a bool
        if isinstance(result, (np.ndarray, torch.Tensor)):
            return bool(np.all(result))
        return result
    except:
        # If comparison fails, assume not equal
        return False


def visualize_image_with_bbox(img, title="Image"):
    """
    Visualize an image with a red bounding box around the edges.
    
    Args:
        img: Image array to visualize
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    
    # Add red bounding box
    ax = plt.gca()
    rect = patches.Rectangle(
        (0, 0), 
        img.shape[1]-2, 
        img.shape[0]-2, 
        linewidth=1, 
        edgecolor='red', 
        facecolor='none'
    )
    ax.add_patch(rect)
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def debug_pipeline_step(sample, transform, step_num):
    """
    Debug a single pipeline step by showing what changed.
    
    Args:
        sample: Input sample dict
        transform: Transform to apply
        step_num: Step number for logging
    
    Returns:
        dict: Transformed sample
    """
    print(f"\n=== Step {step_num}: {transform.__class__.__name__} ===")
    print(f"Transform: {transform}")
    print(f"Before: {list(sample.keys())}")
    
    old_sample = sample.copy()
    transformed_sample = transform(sample)
    
    # Find changes
    added = {k: v for k, v in transformed_sample.items() if k not in old_sample}
    modified = {k: v for k, v in transformed_sample.items() 
               if k in old_sample and not is_equal(old_sample[k], v)}
    removed = {k: v for k, v in old_sample.items() if k not in transformed_sample}
    
    print(f"After: {list(transformed_sample.keys())}")
    if added: 
        print(f"Added: {list(added.keys())}")
    if modified: 
        print(f"Modified: {list(modified.keys())}")
    if removed: 
        print(f"Removed: {list(removed.keys())}")
    if not added and not modified and not removed: 
        print("No changes made by the transform.")
    
    # Show detailed info
    if 'img' in transformed_sample:
        print(f"Image shape: {transformed_sample['img'].shape}")
        visualize_image_with_bbox(transformed_sample['img'], f"Step {step_num}: {transform.__class__.__name__}")

    if 'mask' in transformed_sample:
        print(f"Mask shape: {transformed_sample['mask'].shape}")

    if 'data_samples' in transformed_sample:
        print(f"Data samples: {transformed_sample['data_samples']}")
    
    return transformed_sample


def debug_full_pipeline(dataset, pipeline_config, sample_idx=None):
    """
    Debug an entire pipeline by running through all transforms step by step.
    
    Args:
        dataset: Dataset to get sample from
        pipeline_config: Pipeline configuration (list of transforms or data_pipeline)
        sample_idx: Specific sample index to use (random if None)
    
    Returns:
        dict: Final transformed sample
    """
    # Get sample
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)
    
    sample = dataset[sample_idx].copy()
    print(f"🔍 Debugging pipeline with sample #{sample_idx}")
    print(f"Initial sample keys: {list(sample.keys())}")
    
    # Create pipeline
    pipeline = Compose(pipeline_config)
    
    # Debug each step
    for i, transform in enumerate(pipeline.transforms, 1):
        sample = debug_pipeline_step(sample, transform, i)
    
    print(f"\n✅ Pipeline debugging complete!")
    print(f"Final sample keys: {list(sample.keys())}")
    
    return sample


def create_pipeline(config_module, pipeline_name='data_pipeline'):
    """
    Create a pipeline from config module.
    
    Args:
        config_module: Config module containing pipeline definition
        pipeline_name: Name of the pipeline attribute in config
    
    Returns:
        Compose: Built pipeline
    """
    pipeline_config = getattr(config_module, pipeline_name)
    return Compose(pipeline_config)


def check_dataset_paths(img_path, json_path):
    """
    Check if dataset paths exist and provide information about them.
    
    Args:
        img_path: Path to image directory
        json_path: Path to annotation file
    
    Returns:
        tuple: (img_exists, json_exists, img_count, file_size_mb)
    """
    img_exists = os.path.exists(img_path)
    json_exists = os.path.exists(json_path)
    img_count = 0
    file_size_mb = 0
    
    if img_exists:
        print(f"✓ Image root exists: {img_path}")
        # Count image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(img_path) if f.lower().endswith(ext)])
        img_count = len(image_files)
        print(f"  Found {img_count} image files")
    else:
        print(f"✗ Image root not found: {img_path}")

    if json_exists:
        print(f"✓ Annotation file exists: {json_path}")
        file_size_mb = os.path.getsize(json_path) / (1024 * 1024)  # MB
        print(f"  File size: {file_size_mb:.2f} MB")
    else:
        print(f"✗ Annotation file not found: {json_path}")
    
    return img_exists, json_exists, img_count, file_size_mb


def load_bz2_samples(json_path, num_samples=5):
    """
    Load samples from a compressed JSON file (bz2).
    
    Args:
        json_path: Path to compressed JSON file
        num_samples: Number of samples to load
    
    Returns:
        list: List of sample dictionaries
    """
    import bz2
    import json
    
    samples = []
    with bz2.open(json_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples from {json_path}")
    return samples


def get_absolute_paths(relative_paths):
    """
    Convert relative paths to absolute paths.
    
    Args:
        relative_paths: Single path or list of relative paths
    
    Returns:
        Single absolute path or list of absolute paths
    """
    import os.path as osp
    
    if isinstance(relative_paths, str):
        return osp.abspath(relative_paths)
    return [osp.abspath(path) for path in relative_paths]


def quick_dataset_pipeline_test(config_module, max_data=100, dataset_name="dataset", pipeline_name='data_pipeline'):
    """
    Quick test function that sets up environment, loads dataset, creates pipeline and runs debug.
    
    Args:
        config_module: Config module
        max_data: Max samples to load
        dataset_name: Name for logging
        pipeline_name: Pipeline attribute name in config
    
    Returns:
        tuple: (dataset, pipeline, sample_result)
    """
    # Setup
    setup_environment()
    
    # Load dataset
    dataset = load_dataset(config_module, max_data, dataset_name)
    
    # Create pipeline
    pipeline_config = getattr(config_module, pipeline_name)
    
    # Debug pipeline
    result = debug_full_pipeline(dataset, pipeline_config)
    
    return dataset, pipeline_config, result
