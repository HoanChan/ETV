# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence
from mmengine.evaluator import BaseMetric

from .TEDS.TEDS import TEDS

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
    using Tree Edit Distance algorithm.

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
            # Extract predicted HTML
            pred_html = ""
            if 'pred_table' in data_sample:
                # If prediction contains table structure
                pred_html = data_sample['pred_table'].get('html', '')
            elif 'pred_text' in data_sample:
                # Fallback to text prediction if available
                pred_html = data_sample['pred_text'].get('item', '')
            elif 'pred_instances' in data_sample:
                # Handle cases where prediction is in instances format
                pred_instances = data_sample['pred_instances']
                if hasattr(pred_instances, 'html'):
                    pred_html = pred_instances.html
                elif hasattr(pred_instances, 'get'):
                    pred_html = pred_instances.get('html', '')
            
            # Extract ground truth HTML
            gt_html = ""
            if 'gt_table' in data_sample:
                # If ground truth contains table structure
                gt_html = data_sample['gt_table'].get('html', '')
            elif 'gt_text' in data_sample:
                # Fallback to text ground truth if available
                gt_html = data_sample['gt_text'].get('item', '')
            elif 'gt_instances' in data_sample:
                # Handle cases where ground truth is in instances format
                gt_instances = data_sample['gt_instances']
                if hasattr(gt_instances, 'html'):
                    gt_html = gt_instances.html
                elif hasattr(gt_instances, 'get'):
                    gt_html = gt_instances.get('html', '')
            
            # Calculate TEDS score for this sample
            teds_score = self.teds_evaluator.evaluate(pred_html, gt_html)
            
            result = dict(teds_score=teds_score)
            self.results.append(result)

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
            
            # Extract predicted HTML
            pred_html = ""
            if 'pred_table' in data_sample:
                pred_html = data_sample['pred_table'].get('html', '')
            elif 'pred_text' in data_sample:
                pred_html = data_sample['pred_text'].get('item', '')
            elif 'pred_instances' in data_sample:
                pred_instances = data_sample['pred_instances']
                if hasattr(pred_instances, 'html'):
                    pred_html = pred_instances.html
                elif hasattr(pred_instances, 'get'):
                    pred_html = pred_instances.get('html', '')
            
            # Extract ground truth HTML
            gt_html = ""
            if 'gt_table' in data_sample:
                gt_html = data_sample['gt_table'].get('html', '')
            elif 'gt_text' in data_sample:
                gt_html = data_sample['gt_text'].get('item', '')
            elif 'gt_instances' in data_sample:
                gt_instances = data_sample['gt_instances']
                if hasattr(gt_instances, 'html'):
                    gt_html = gt_instances.html
                elif hasattr(gt_instances, 'get'):
                    gt_html = gt_instances.get('html', '')
            
            # Store samples
            self.pred_samples[sample_id] = pred_html
            self.gt_samples[sample_id] = {'html': gt_html}

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
