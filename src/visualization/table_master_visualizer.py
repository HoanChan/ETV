# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch

from mmocr.registry import VISUALIZERS
from mmocr.visualization.base_visualizer import BaseLocalVisualizer
from structures.table_master_data_sample import TableMasterDataSample


@VISUALIZERS.register_module()
class TableMasterVisualizer(BaseLocalVisualizer):
    """Table Master Visualizer for table structure recognition.
    
    Visualizes table structure tokens and their bounding boxes.
    
    Expected data format from TableMasterPostprocessor:
    - data_sample.pred_tokens.item: str or List[str] - Structure tokens
    - data_sample.pred_instances.bboxes: np.ndarray - Bounding boxes in pixel coordinates (N, 4)
    - data_sample.pred_tokens.scores: float - Prediction confidence score
    
    For ground truth:
    - data_sample.gt_tokens.item: List[str] - Ground truth tokens
    - data_sample.gt_instances.bboxes: np.ndarray - Ground truth bounding boxes (N, 4)
    """

    def _draw_instances(self,
                        image: np.ndarray,
                        bboxes: Optional[np.ndarray],
                        tokens: Optional[Sequence[str]],
                        ) -> np.ndarray:
        """Draw bboxes and tokens on image.

        Args:
            image (np.ndarray): The origin image to draw.
            bboxes (np.ndarray, optional): The bboxes to draw. Shape (N, 4).
            tokens (Sequence[str], optional): The tokens to draw.

        Returns:
            np.ndarray: The image with bboxes and tokens drawn.
        """
        if bboxes is None or len(bboxes) == 0:
            return image

        # Filter valid bboxes (non-zero bboxes)
        valid_indices = []
        valid_bboxes = []
        valid_tokens = []
        
        for i, bbox in enumerate(bboxes):
            # Check if bbox is valid (not all zeros)
            if bbox.sum() > 0:
                valid_indices.append(i)
                valid_bboxes.append(bbox)
                if tokens and i < len(tokens):
                    valid_tokens.append(tokens[i])
                else:
                    valid_tokens.append(f'cell_{i}')
        
        if len(valid_bboxes) == 0:
            return image
            
        valid_bboxes = np.array(valid_bboxes)

        # Create text-only image similar to TextSpottingLocalVisualizer
        img_shape = image.shape[:2]
        text_image = np.full((img_shape[0], img_shape[1], 3), 255, dtype=np.uint8)

        # Draw bboxes on original image
        image = self.get_bboxes_image(
            image, valid_bboxes, colors=self.PALETTE, filling=True)

        # Draw tokens on text image
        if valid_tokens:
            text_image = self.get_labels_image(
                text_image,
                labels=valid_tokens,
                bboxes=valid_bboxes,
                font_families=self.font_families,
                font_properties=self.font_properties)

        # Draw bboxes on text image
        text_image = self.get_bboxes_image(
            text_image, valid_bboxes, colors=self.PALETTE)

        # Concatenate images side by side
        return np.concatenate([image, text_image], axis=1)

    def _debug_data_sample(self, data_sample: TableMasterDataSample) -> None:
        """Debug method to print data sample attributes."""
        print(f"Data sample attributes:")
        for attr in dir(data_sample):
            if not attr.startswith('_'):
                try:
                    value = getattr(data_sample, attr)
                    if not callable(value):
                        print(f"  {attr}: {type(value)} - {value}")
                except:
                    print(f"  {attr}: <error accessing>")
        print()

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional[TableMasterDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       out_file: Optional[str] = None,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (TableMasterDataSample, optional): TableMasterDataSample
                which contains gt and prediction. Defaults to None.
            draw_gt (bool): Whether to draw GT. Defaults to True.
            draw_pred (bool): Whether to draw Predicted results. Defaults to True.
            show (bool): Whether to display the drawn image. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        cat_images = []
        
        if data_sample is not None:
            # Draw ground truth
            if draw_gt and hasattr(data_sample, 'gt_tokens') and hasattr(data_sample, 'gt_instances'):
                gt_tokens = getattr(data_sample.gt_tokens, 'item', None)
                gt_bboxes = getattr(data_sample.gt_instances, 'bboxes', None)
                
                if gt_tokens is not None and gt_bboxes is not None:
                    # Convert token data to list if needed
                    if isinstance(gt_tokens, str):
                        gt_tokens = gt_tokens.split(',')
                    elif not isinstance(gt_tokens, list):
                        gt_tokens = list(gt_tokens)
                    
                    gt_img_data = self._draw_instances(image, gt_bboxes, gt_tokens)
                    cat_images.append(gt_img_data)
            
            # Draw predictions
            if draw_pred and hasattr(data_sample, 'pred_tokens'):
                pred_tokens = getattr(data_sample.pred_tokens, 'item', None)
                pred_bboxes = None
                
                if hasattr(data_sample, 'pred_instances'):
                    pred_bboxes = getattr(data_sample.pred_instances, 'bboxes', None)
                
                if pred_tokens is not None and pred_bboxes is not None:
                    # Convert token to token list
                    if isinstance(pred_tokens, str):
                        pred_tokens = [token.strip() for token in pred_tokens.split(',') if token.strip()]
                    elif not isinstance(pred_tokens, list):
                        pred_tokens = list(pred_tokens)
                    
                    pred_img_data = self._draw_instances(image, pred_bboxes, pred_tokens)
                    cat_images.append(pred_img_data)
        
        # Concatenate images vertically
        cat_images = self._cat_image(cat_images, axis=0)
        if cat_images is None:
            cat_images = image
        
        # Display or save
        if show:
            self.show(cat_images, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, cat_images, step)
        
        if out_file is not None:
            mmcv.imwrite(cat_images[..., ::-1], out_file)
        
        self.set_image(cat_images)
        return self.get_image()

    def visualize_table_structure(self,
                                 image: np.ndarray,
                                 tokens: Optional[Sequence[str]] = None,
                                 bboxes: Optional[np.ndarray] = None,
                                 out_file: Optional[str] = None,
                                 show: bool = False,
                                 wait_time: int = 0) -> np.ndarray:
        """Visualize table structure predictions directly.
        
        Args:
            image (np.ndarray): Input image.
            tokens (Sequence[str], optional): Structure tokens.
            bboxes (np.ndarray, optional): Bboxes corresponding to tokens.
            out_file (str, optional): Output file path.
            show (bool): Whether to display the image.
            wait_time (int): Display wait time.
            
        Returns:
            np.ndarray: Visualized image.
        """
        if bboxes is None or len(bboxes) == 0:
            return image
        
        # Draw instances
        result_image = self._draw_instances(image.copy(), bboxes, tokens)
        
        # Display or save
        if show:
            self.show(result_image, win_name='table_structure', wait_time=wait_time)
        
        if out_file is not None:
            mmcv.imwrite(result_image[..., ::-1], out_file)
        
        return result_image


def visual_pred_bboxes(data_samples: List[TableMasterDataSample],
                      results: List,
                      vis_dir: str = 'vis_results',
                      show: bool = False) -> None:
    """Visualize prediction bboxes for multiple samples.
    
    Args:
        data_samples (List[TableMasterDataSample]): List of data samples.
        results (List): List of prediction results (can be dict or TableMasterDataSample).
        vis_dir (str): Directory to save visualizations.
        show (bool): Whether to display images.
    """
    import os
    
    # Create visualizer
    visualizer = TableMasterVisualizer(save_dir=vis_dir)
    
    # Ensure output directory exists
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, (data_sample, result) in enumerate(zip(data_samples, results)):
        # Get image from data sample
        image = None
        if hasattr(data_sample, 'img_path') and data_sample.img_path:
            try:
                image = mmcv.imread(data_sample.img_path)
            except:
                image = None
        
        if image is None and hasattr(data_sample, 'img'):
            image = data_sample.img
        
        if image is None:
            print(f"Warning: Cannot find image for sample {i}")
            continue
        
        # Handle different result formats
        if isinstance(result, TableMasterDataSample):
            # Direct data sample from model prediction
            pred_sample = result
        elif isinstance(result, dict):
            # Dictionary format from model.predict()
            pred_sample = TableMasterDataSample()
            pred_sample.pred_token = result.get('token', '')
            pred_sample.pred_score = result.get('score', 1.0)
            pred_sample.pred_bbox = result.get('bbox', None)
        else:
            print(f"Warning: Unknown result format for sample {i}")
            continue
        
        # Visualize
        out_file = os.path.join(vis_dir, f'sample_{i:04d}.jpg')
        visualizer.add_datasample(
            name=f'sample_{i}',
            image=image,
            data_sample=pred_sample,
            draw_gt=False,
            draw_pred=True,
            show=show,
            out_file=out_file)
    
    print(f"Visualizations saved to {vis_dir}")
