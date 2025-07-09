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
    """

    default_prefix: Optional[str] = 'table'

    def __init__(self,
                 structure_only: bool = False,
                 ignore_nodes: Optional[list] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

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
    
    def process(self, data_batch: Sequence[Dict], data_samples: Sequence[Dict]) -> None:
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

            if pred_html and gt_html:
                teds_score = self.evaluate_single(pred_html, gt_html)
            else:
                teds_score = 0.0
            
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
            
        teds_scores = [result['teds_score'] for result in results]
        avg_teds = sum(teds_scores) / len(teds_scores)
        # Additional statistics
        max_teds = max(teds_scores)
        min_teds = min(teds_scores)
    
        eval_res = {
            'teds': avg_teds,
            'teds_max': max_teds,
            'teds_min': min_teds
        }

        return eval_res