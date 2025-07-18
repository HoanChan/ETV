# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Union

import mmcv
import mmengine
import numpy as np
import torch

from mmocr.apis.inferencers.base_mmocr_inferencer import BaseMMOCRInferencer, InputsType, PredType
from mmocr.registry import DATASETS, MODELS, TRANSFORMS, VISUALIZERS
from mmocr.utils import ConfigType
from mmocr.structures import TextRecogDataSample
from mmengine import Config
from mmengine.runner import load_checkpoint
from mmengine.device import get_device

class TableMasterInferencer(BaseMMOCRInferencer):
    """TableMaster Inferencer for table structure recognition.
    
    This inferencer is specifically designed for table recognition tasks using
    the TableMaster model architecture.
    
    Args:
        config (Union[ConfigType, str]): Path to the config file or the model config.
        checkpoint (Optional[str]): Path to the checkpoint file. Defaults to None.
        device (Optional[str]): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str): The scope of the model. Defaults to 'mmocr'.
    """

    def __init__(self,
                 config: Union[ConfigType, str],
                 checkpoint: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmocr') -> None:
        
        if device is None:
            device = get_device()
        
        self.device = device
        self.scope = scope
        
        # Load config
        if isinstance(config, str):
            self.config = Config.fromfile(config)
        else:
            self.config = config
            
        # Build model
        model = MODELS.build(self.config.model)
        model = model.to(device)
        model.eval()
        
        # Load checkpoint if provided
        if checkpoint is not None:
            checkpoint = load_checkpoint(model, checkpoint, map_location=device)
            
        self.model = model
        
        # Build data pipeline for inference
        self.pipeline = self._build_inference_pipeline()
        
        # Initialize TableMaster visualizer
        ts = str(datetime.timestamp(datetime.now()))
        self.visualizer = VISUALIZERS.build(
            dict(
                type='TableMasterVisualizer',
                name=f'table_inferencer{ts}',
                save_dir='temp_vis_results'
            ))
    
    def _build_inference_pipeline(self):
        """Build a simplified pipeline for inference."""
        pipeline = []
        
        # Resize transform
        resize_transform = TRANSFORMS.build(dict(
            type='Resize',
            scale=(480, 480),
            keep_ratio=True
        ))
        pipeline.append(resize_transform)
        
        # Pad transform
        pad_transform = TRANSFORMS.build(dict(
            type='Pad',
            size=(480, 480)
        ))
        pipeline.append(pad_transform)
        
        # PackInputs transform (simplified version for inference)
        pack_transform = TRANSFORMS.build(dict(
            type='PackInputs',
            keys=['img'],
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            meta_keys=('ori_shape', 'img_shape', 'scale_factor', 'pad_shape', 'valid_ratio')
        ))
        pipeline.append(pack_transform)
        
        return pipeline
    
    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Preprocess inputs to model format."""
        
        # Convert inputs to list
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
            
        processed_inputs = []
        
        for inp in inputs:
            # Load image
            if isinstance(inp, str):
                img = mmcv.imread(inp)
                img_path = inp
            elif isinstance(inp, np.ndarray):
                img = inp
                img_path = None
            else:
                raise ValueError(f"Unsupported input type: {type(inp)}")
                
            # Prepare data dict for inference
            data = {
                'img': img,
                'img_path': img_path,
                'ori_shape': img.shape[:2],
                'img_shape': img.shape[:2],
            }
            
            # Apply pipeline transformations
            for transform in self.pipeline:
                data = transform(data)
                
            processed_inputs.append(data)
            
        return processed_inputs
    
    def forward(self,
                inputs: InputsType,
                batch_size: int = 1,
                **forward_kwargs) -> PredType:
        """Forward the inputs to the model."""
        
        # Preprocess inputs
        data_samples = self.preprocess(inputs, batch_size, **forward_kwargs)
        
        predictions = []
        
        # Process in batches
        for i in range(0, len(data_samples), batch_size):
            batch_data = data_samples[i:i + batch_size]
            
            # Prepare batch
            batch_inputs = []
            batch_data_samples = []
            
            for data in batch_data:
                # Extract image tensor
                if 'inputs' in data:
                    img_tensor = data['inputs']
                else:
                    img_tensor = data['img']
                    
                batch_inputs.append(img_tensor)
                
                # Create data sample
                data_sample = TextRecogDataSample()
                if 'img_path' in data and data['img_path'] is not None:
                    data_sample.set_metainfo({'img_path': data['img_path']})
                batch_data_samples.append(data_sample)
            
            # Stack tensors
            if len(batch_inputs) > 0:
                batch_inputs = torch.stack(batch_inputs).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    batch_predictions = self.model.predict(
                        batch_inputs, 
                        batch_data_samples
                    )
                    
                predictions.extend(batch_predictions)
        
        return {'predictions': predictions}
    
    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  show: bool = False,
                  wait_time: float = 0.,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  save_vis: bool = False,
                  img_out_dir: str = '',
                  **kwargs) -> Union[List[np.ndarray], None]:
        """Visualize predictions."""
        
        if not draw_pred:
            return None
            
        # Convert inputs to images
        imgs = []
        for inp in inputs if isinstance(inputs, (list, tuple)) else [inputs]:
            if isinstance(inp, str):
                img = mmcv.imread(inp)
            elif isinstance(inp, np.ndarray):
                img = inp
            else:
                raise ValueError(f"Unsupported input type: {type(inp)}")
            imgs.append(img)
        
        visualizations = []
        predictions = preds['predictions']
        
        for i, (img, pred) in enumerate(zip(imgs, predictions)):
            # Use TableMasterVisualizer to draw predictions
            out_file = None
            if save_vis and img_out_dir:
                if hasattr(pred, 'img_path') and pred.img_path:
                    img_name = osp.basename(pred.img_path)
                    img_name = osp.splitext(img_name)[0] + '_vis.jpg'
                else:
                    img_name = f'result_{i}_vis.jpg'
                out_file = osp.join(img_out_dir, img_name)
            
            # Use visualizer to create visualization
            vis_img = self.visualizer.add_datasample(
                name=f'result_{i}',
                image=img,
                data_sample=pred,
                draw_gt=False,  # Only draw predictions for inference
                draw_pred=True,
                show=show,
                wait_time=wait_time,
                out_file=out_file
            )
            
            visualizations.append(vis_img)
                
        return visualizations
    
    def postprocess(self,
                    preds: PredType,
                    visualization: Optional[List[np.ndarray]] = None,
                    print_result: bool = False,
                    save_pred: bool = False,
                    pred_out_dir: str = '',
                    **kwargs) -> Dict:
        """Process the predictions and visualization results."""
        
        predictions = preds['predictions']
        results = []
        
        for i, pred in enumerate(predictions):
            result_dict = {}
            
            # Extract structure tokens prediction
            if hasattr(pred, 'pred_tokens') and pred.pred_tokens is not None:
                if hasattr(pred.pred_tokens, 'item'):
                    result_dict['tokens'] = pred.pred_tokens.item
                else:
                    result_dict['tokens'] = pred.pred_tokens
            elif hasattr(pred, 'pred_text') and pred.pred_text is not None:
                result_dict['tokens'] = pred.pred_text.item
                
            # Extract bboxes if available
            if hasattr(pred, 'pred_instances') and pred.pred_instances is not None:
                if hasattr(pred.pred_instances, 'bboxes'):
                    bboxes = pred.pred_instances.bboxes
                    if torch.is_tensor(bboxes):
                        result_dict['bboxes'] = bboxes.cpu().numpy().tolist()
                    elif isinstance(bboxes, np.ndarray):
                        result_dict['bboxes'] = bboxes.tolist()
                    else:
                        result_dict['bboxes'] = bboxes
                        
                if hasattr(pred.pred_instances, 'scores'):
                    scores = pred.pred_instances.scores
                    if torch.is_tensor(scores):
                        result_dict['scores'] = scores.cpu().numpy().tolist()
                    elif isinstance(scores, np.ndarray):
                        result_dict['scores'] = scores.tolist()
                    else:
                        result_dict['scores'] = scores
                        
            # Extract confidence score if available
            if hasattr(pred, 'pred_tokens') and hasattr(pred.pred_tokens, 'scores'):
                scores = pred.pred_tokens.scores
                if torch.is_tensor(scores):
                    result_dict['token_score'] = scores.cpu().numpy().item() if scores.numel() == 1 else scores.cpu().numpy().tolist()
                else:
                    result_dict['token_score'] = scores
                    
            # Add image path if available
            if hasattr(pred, 'img_path'):
                result_dict['img_path'] = pred.img_path
            elif hasattr(pred, 'metainfo') and 'img_path' in pred.metainfo:
                result_dict['img_path'] = pred.metainfo['img_path']
                
            results.append(result_dict)
            
            # Save prediction if requested
            if save_pred and pred_out_dir:
                if 'img_path' in result_dict and result_dict['img_path']:
                    pred_name = osp.splitext(osp.basename(result_dict['img_path']))[0] + '.json'
                else:
                    pred_name = f'result_{i}.json'
                    
                pred_out_file = osp.join(pred_out_dir, pred_name)
                mmengine.dump(result_dict, pred_out_file, ensure_ascii=False)
        
        # Print results if requested
        if print_result:
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                if 'tokens' in result:
                    print(f"  tokens: {result['tokens']}")
                if 'token_score' in result:
                    print(f"  Token Score: {result['token_score']}")
                if 'bboxes' in result:
                    print(f"  Bboxes: {len(result['bboxes'])} detected")
                if 'scores' in result:
                    print(f"  Bbox Scores: {result['scores']}")
                if 'img_path' in result:
                    print(f"  Image: {result['img_path']}")
                print()
        
        return {
            'predictions': results,
            'visualization': visualization
        }
    
    def __call__(self,
                 inputs: InputsType,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 save_vis: bool = False,
                 save_pred: bool = False,
                 out_dir: str = 'results/',
                 **kwargs) -> Dict:
        """Call the inferencer.
        
        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Batch size. Defaults to 1.
            return_vis (bool): Whether to return visualization results.
            save_vis (bool): Whether to save visualization results.
            save_pred (bool): Whether to save prediction results.
            out_dir (str): Output directory. Defaults to 'results/'.
            
        Returns:
            Dict: Inference and visualization results.
        """
        
        if (save_vis or save_pred) and not out_dir:
            raise ValueError('out_dir must be specified when save_vis or save_pred is True!')
            
        # Setup output directories
        if out_dir:
            img_out_dir = osp.join(out_dir, 'vis')
            pred_out_dir = osp.join(out_dir, 'preds')
            if save_vis:
                mmengine.mkdir_or_exist(img_out_dir)
            if save_pred:
                mmengine.mkdir_or_exist(pred_out_dir)
        else:
            img_out_dir, pred_out_dir = '', ''
        
        # Convert inputs to list
        ori_inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        
        # Forward pass
        preds = self.forward(ori_inputs, batch_size=batch_size, **kwargs)
        
        # Visualize
        visualization = None
        if return_vis or save_vis:
            visualization = self.visualize(
                ori_inputs, 
                preds, 
                save_vis=save_vis,
                img_out_dir=img_out_dir,
                **kwargs
            )
        
        # Postprocess
        results = self.postprocess(
            preds,
            visualization=visualization,
            save_pred=save_pred,
            pred_out_dir=pred_out_dir,
            **kwargs
        )
        
        return results


def init_table_master_inferencer(config_path: str, 
                                 checkpoint_path: str, 
                                 device: Optional[str] = None) -> TableMasterInferencer:
    """Initialize TableMaster inferencer.
    
    Args:
        config_path (str): Path to config file.
        checkpoint_path (str): Path to checkpoint file.
        device (Optional[str]): Device to use. Defaults to None (auto detect).
        
    Returns:
        TableMasterInferencer: Initialized inferencer.
    """
    return TableMasterInferencer(
        config=config_path,
        checkpoint=checkpoint_path,
        device=device
    )
