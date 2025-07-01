# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence
from mmengine.evaluator import BaseMetric
import re
import copy

from .TEDS.TEDS import TEDS
from .post_processing import (insert_text_to_token, text_to_list, htmlPostProcess, deal_bb)

try:
    from mmocr.registry import METRICS
except ImportError:
    # Fallback for older versions
    from mmengine.registry import Registry
    METRICS = Registry('metric')


@METRICS.register_module()
class BatchTEDSMetric(BaseMetric):
    """Batch TEDS metric for table structure recognition task.
    
    This version processes all samples at once using batch evaluation,
    which can be more efficient for large datasets.

    Args:
        structure_only (bool): Whether to evaluate only table structure, ignoring cell content.
            Defaults to False.
        n_jobs (int): Number of parallel jobs for evaluation. Defaults to 4.
        ignore_nodes (list[str], optional): List of HTML tag names to ignore during evaluation.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'table'

    def __init__(self,
                 structure_only: bool = False,
                 n_jobs: int = 4,
                 ignore_nodes: Optional[list] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        
        # Initialize TEDS evaluator
        self.teds_evaluator = TEDS(
            structure_only=structure_only,
            n_jobs=n_jobs,
            ignore_nodes=ignore_nodes
        )
        
        # Store samples for batch processing
        self.pred_samples = {}
        self.gt_samples = {}

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data_samples. Store samples for batch evaluation.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            data_samples (Sequence[Dict]): A batch of outputs from the model.
        """
        for idx, data_sample in enumerate(data_samples):
            # Generate unique sample ID
            sample_id = f"sample_{len(self.pred_samples)}"
            
            # Extract predicted HTML with same post-processing as TEDSMetric
            pred_html = ""
            
            # Priority 1: Handle raw tokens + cells format (TableMASTER format)
            if 'pred_text' in data_sample and 'pred_cells' in data_sample:
                pred_text = data_sample['pred_text']
                pred_cells = data_sample['pred_cells']
                
                # Handle different text formats
                if isinstance(pred_text, dict):
                    pred_text = pred_text.get('item', '')
                elif isinstance(pred_text, str):
                    pass  # Already string
                else:
                    pred_text = str(pred_text)
                
                # Handle different cells formats
                if isinstance(pred_cells, dict):
                    pred_cells = pred_cells.get('item', [])
                elif not isinstance(pred_cells, list):
                    pred_cells = []
                
                # Process using mmocr_teds compatible method
                pred_html = self._process_tokens_to_html(pred_text, pred_cells)
                
            # Priority 2: Direct HTML format
            elif 'pred_table' in data_sample:
                pred_html = data_sample['pred_table'].get('html', '')
                if pred_html and not pred_html.startswith('<html>'):
                    pred_html = self._html_post_process(pred_html)
                    
            elif 'pred_text' in data_sample:
                pred_text = data_sample['pred_text']
                if isinstance(pred_text, dict):
                    pred_html = pred_text.get('item', '')
                else:
                    pred_html = str(pred_text)
                if pred_html and not pred_html.startswith('<html>'):
                    pred_html = self._html_post_process(pred_html)
                    
            elif 'pred_instances' in data_sample:
                pred_instances = data_sample['pred_instances']
                if hasattr(pred_instances, 'html'):
                    pred_html = pred_instances.html
                elif hasattr(pred_instances, 'get'):
                    pred_html = pred_instances.get('html', '')
                if pred_html and not pred_html.startswith('<html>'):
                    pred_html = self._html_post_process(pred_html)
            
            # Extract ground truth HTML
            gt_html = ""
            if 'gt_table' in data_sample:
                gt_html = data_sample['gt_table'].get('html', '')
            elif 'gt_text' in data_sample:
                gt_text = data_sample['gt_text']
                if isinstance(gt_text, dict):
                    gt_html = gt_text.get('item', '')
                else:
                    gt_html = str(gt_text)
            elif 'gt_instances' in data_sample:
                gt_instances = data_sample['gt_instances']
                if hasattr(gt_instances, 'html'):
                    gt_html = gt_instances.html
                elif hasattr(gt_instances, 'get'):
                    gt_html = gt_instances.get('html', '')
            
            # Wrap GT HTML if needed
            if gt_html and not gt_html.startswith('<html>'):
                gt_html = self._html_post_process(gt_html)
            
            # Store samples
            self.pred_samples[sample_id] = pred_html
            self.gt_samples[sample_id] = {'html': gt_html}

    def _process_tokens_to_html(self, pred_text: str, pred_cells: list) -> str:
        """Process tokens and cells to HTML using mmocr_teds compatible method.
        
        Args:
            pred_text (str): Comma-separated structure tokens
            pred_cells (list): List of cell content strings
            
        Returns:
            str: Processed HTML string
        """
        # Convert text to token list
        master_token_list = text_to_list(pred_text)
        
        # Insert cell content into structure tokens
        html = insert_text_to_token(master_token_list, pred_cells)
        
        # Apply post-processing for thead and tbody
        html = deal_bb(html, 'thead')
        html = deal_bb(html, 'tbody')
        
        # Wrap in HTML structure
        return self._html_post_process(html)
    
    def _html_post_process(self, html: str) -> str:
        """Wrap HTML content in proper HTML structure.
        
        Args:
            html (str): Table HTML content
            
        Returns:
            str: Complete HTML document
        """
        if html.startswith('<html>'):
            return html
        return htmlPostProcess(html)

    def compute_metrics(self, results: Sequence[Dict]) -> Dict:
        """Compute the metrics using batch evaluation.

        Args:
            results (list[Dict]): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        if not self.pred_samples or not self.gt_samples:
            return {'teds': 0.0}
        
        # Perform batch evaluation
        scores = self.teds_evaluator.batch_evaluate(self.pred_samples, self.gt_samples)
        
        # Calculate statistics
        teds_scores = list(scores.values())
        avg_teds = sum(teds_scores) / len(teds_scores)
        max_teds = max(teds_scores)
        min_teds = min(teds_scores)
        
        eval_res = {}
        eval_res['teds'] = float(f'{avg_teds:.4f}')
        eval_res['teds_max'] = float(f'{max_teds:.4f}')
        eval_res['teds_min'] = float(f'{min_teds:.4f}')
        
        # Clear stored samples for next evaluation
        self.pred_samples.clear()
        self.gt_samples.clear()
        
        return eval_res