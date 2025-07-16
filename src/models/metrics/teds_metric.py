# Copyright (c) Lê Hoàn Chân. All rights reserved.
from .post_processing import ( insert_text_to_token, text_to_list, htmlPostProcess, deal_bb )
from typing import Dict, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmocr.registry import METRICS
from collections import deque
from lxml import etree, html
from apted import APTED, Config
from apted.helpers import Tree
import distance

class TableTree(Tree):
    """Table tree structure for TEDS calculation."""
    
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation."""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    """Custom configuration for APTED algorithm."""
    
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value."""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1."""
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees."""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


@METRICS.register_module()
class TEDSMetric(BaseMetric):
    """TEDS (Tree Edit Distance based Similarity) metric for table recognition task.

    Args:
        structure_only (bool): Whether to only consider table structure 
            without cell content. Defaults to False.
        ignore_nodes (list, optional): List of node types to ignore during 
            evaluation. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    Note:
        TEDS measures the similarity between predicted and ground truth table
        structures using tree edit distance. Score ranges from 0 to 1, where
        1 indicates perfect match.
    """

    def __init__(self,
                 structure_only: bool = False,
                 ignore_nodes: Optional[list] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = 'TEDS') -> None:
        super().__init__(collect_device, prefix)

        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def _extract_html_from_sample(self, sample: Dict, is_prediction: bool = True) -> str:
        """Extract HTML from data sample following MMOCR patterns.
        
        Args:
            sample: Data sample dict
            is_prediction: Whether this is prediction (True) or ground truth (False)
            
        Returns:
            str: Extracted HTML string
        """
        html_content = ""
        prefix = "pred" if is_prediction else "gt"
        
        # Access through tokens (LabelData) and instances (InstanceData) structure
        tokens_key = f'{prefix}_tokens'
        instances_key = f'{prefix}_instances'
        
        pred_tokens = ""
        pred_cells = []
        
        if tokens_key in sample:
            tokens_obj = sample[tokens_key]
            pred_tokens = getattr(tokens_obj, 'item', '') if hasattr(tokens_obj, 'item') else ''
            
        if instances_key in sample:
            instances_obj = sample[instances_key]
            pred_cells = getattr(instances_obj, 'cells', []) if hasattr(instances_obj, 'cells') else []
        
        # Fallback to old structure if needed
        if not pred_tokens:
            pred_tokens = sample.get(f'{prefix}_tokens', '')
        if not pred_cells:
            pred_cells = sample.get(f'{prefix}_cells', [])
        
        # Extract from dict format if needed
        if isinstance(pred_tokens, dict):
            pred_tokens = pred_tokens.get('item', '')
        if isinstance(pred_cells, dict):
            pred_cells = pred_cells.get('item', [])
        
        # Convert to string and ensure list
        pred_tokens = str(pred_tokens) if pred_tokens else ''
        pred_cells = pred_cells if isinstance(pred_cells, list) else []
        
        if pred_tokens and pred_cells:
            html_content = self._process_tokens_to_html(pred_tokens, pred_cells)
        
        # Post-process HTML
        if html_content and not html_content.startswith('<html>'):
            html_content = self._html_post_process(html_content)
            
        return html_content
    
    def tokenize(self, node):
        """Tokenizes table cells."""
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted."""
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        
        if parent is not None:
            parent.children.append(new_node)
        
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        
        if parent is None:
            return new_node

    def evaluate_single(self, pred_html, gt_html):
        """Computes TEDS score between prediction and ground truth.
        
        Args:
            pred_html (str): Predicted HTML table string.
            gt_html (str): Ground truth HTML table string.
            
        Returns:
            float: TEDS score between 0 and 1.
        """
        if (not pred_html) or (not gt_html):
            return 0.0
            
        try:
            parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
            pred = html.fromstring(pred_html, parser=parser)
            gt = html.fromstring(gt_html, parser=parser)
            
            if pred.xpath('body/table') and gt.xpath('body/table'):
                pred = pred.xpath('body/table')[0]
                gt = gt.xpath('body/table')[0]
                
                if self.ignore_nodes:
                    etree.strip_tags(pred, *self.ignore_nodes)
                    etree.strip_tags(gt, *self.ignore_nodes)
                
                n_nodes_pred = len(pred.xpath(".//*"))
                n_nodes_gt = len(gt.xpath(".//*"))
                n_nodes = max(n_nodes_pred, n_nodes_gt)
                
                if n_nodes == 0:
                    return 0.0
                
                tree_pred = self.load_html_tree(pred)
                tree_gt = self.load_html_tree(gt)
                
                edit_distance = APTED(tree_pred, tree_gt, CustomConfig()).compute_edit_distance()
                
                return 1.0 - (float(edit_distance) / n_nodes)
            else:
                return 0.0
                
        except Exception:
            return 0.0

    def _process_tokens_to_html(self, pred_tokens: str, pred_cells: list) -> str:
        """Process tokens and cells to HTML using mmocr_teds compatible method.
        
        Args:
            pred_tokens (str): Comma-separated structure tokens
            pred_cells (list): List of cell content strings
            
        Returns:
            str: Processed HTML string
        """
        # Convert text to token list
        master_token_list = text_to_list(pred_tokens)
        
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
    
    def process(self, data_batch: Sequence[Dict], data_samples: Sequence[Dict]) -> None:
        """Process one batch of data_samples. The processed results should be
        stored in ``self.results``, which will be used to compute the metrics
        when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of gts.
            data_samples (Sequence[Dict]): A batch of outputs from the model.
        """
        for i, data_sample in enumerate(data_samples):
            # Extract prediction HTML
            pred_html = self._extract_html_from_sample(data_sample, is_prediction=True)
            
            # Extract ground truth HTML
            # Ưu tiên lấy từ data_sample trước, nếu không có thì lấy từ data_batch
            gt_html = self._extract_html_from_sample(data_sample, is_prediction=False)

            # Nếu không có ground truth trong data_sample, thử lấy từ data_batch
            if not gt_html:
                if isinstance(data_batch, (list, tuple)) and i < len(data_batch):
                    gt_html = self._extract_html_from_sample(data_batch[i], is_prediction=False)
                elif isinstance(data_batch, dict):
                    gt_html = self._extract_html_from_sample(data_batch, is_prediction=False)

            # Lưu pred_html và gt_html vào result thay vì teds_score
            result = dict(pred_html=pred_html, gt_html=gt_html)
            self.results.append(result)

    def compute_metrics(self, results: Sequence[Dict]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[Dict]): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        prefix = self.prefix if self.prefix else 'TEDS'
        if not results:
            return {f'{prefix}_avg': 0.0, f'{prefix}_max': 0.0, f'{prefix}_min': 0.0}
        teds_scores = []
        for result in results:
            pred_html = result.get('pred_html', '')
            gt_html = result.get('gt_html', '')
            if pred_html and gt_html:
                teds_score = self.evaluate_single(pred_html, gt_html)
            else:
                teds_score = 0.0
            teds_scores.append(teds_score)
        avg_teds = sum(teds_scores) / len(teds_scores)
        max_teds = max(teds_scores)
        min_teds = min(teds_scores)
        eval_res = {
            f'{prefix}_avg': avg_teds,
            f'{prefix}_max': max_teds,
            f'{prefix}_min': min_teds,
        }
        return eval_res