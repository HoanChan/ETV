# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence
from mmengine.evaluator import BaseMetric

from .TEDS.TEDS import TEDS
from .post_processing import ( insert_text_to_token, text_to_list, htmlPostProcess, deal_bb )

try:
    from mmocr.registry import METRICS
except ImportError:
    # Fallback for older versions
    from mmengine.registry import Registry
    METRICS = Registry('metric')


@METRICS.register_module()
class TEDSMetric(BaseMetric):
    """TEDS (Tree Edit Distance based Similarity) metric for table structure recognition task.
    
    This metric evaluates the similarity between predicted and ground truth table structures
    using Tree Edit Distance algorithm. It includes comprehensive post-processing to handle
    various prediction formats and edge cases.

    Args:
        structure_only (bool): Whether to evaluate only table structure, ignoring cell content.
            Defaults to False.
        n_jobs (int): Number of parallel jobs for evaluation. Defaults to 1.
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
                 n_jobs: int = 1,
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

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data_samples. The processed results should be
        stored in ``self.results``, which will be used to compute the metrics
        when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            data_samples (Sequence[Dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # Extract predicted HTML with post-processing
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
            
            # Calculate TEDS score for this sample
            if pred_html and gt_html:
                teds_score = self.teds_evaluator.evaluate(pred_html, gt_html)
            else:
                teds_score = 0.0
            
            result = dict(teds_score=teds_score)
            self.results.append(result)

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
        """Compute the metrics from processed results.

        Args:
            results (list[Dict]): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        if not results:
            return {'teds': 0.0}
        
        # Calculate average TEDS score
        teds_scores = [result['teds_score'] for result in results]
        avg_teds = sum(teds_scores) / len(teds_scores)
        
        eval_res = {}
        eval_res['teds'] = float(f'{avg_teds:.4f}')
        
        # Additional statistics
        max_teds = max(teds_scores)
        min_teds = min(teds_scores)
        
        eval_res['teds_max'] = float(f'{max_teds:.4f}')
        eval_res['teds_min'] = float(f'{min_teds:.4f}')
        
        return eval_res