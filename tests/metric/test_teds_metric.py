import os
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.metric.teds_metric import TEDSMetric


class TestTEDSMetric:
    """Test suite for TEDSMetric class."""
    
    @pytest.fixture
    def metric(self):
        """Create a basic TEDSMetric instance for testing."""
        return TEDSMetric()
    
    @pytest.fixture
    def metric_structure_only(self):
        """Create a TEDSMetric instance with structure_only=True."""
        return TEDSMetric(structure_only=True)
    
    def test_init_default_parameters(self):
        """Test TEDSMetric initialization with default parameters."""
        metric = TEDSMetric()
        assert metric.structure_only == False
        assert metric.n_jobs == 1
        assert metric.ignore_nodes is None
        assert metric.default_prefix == 'table'
        assert hasattr(metric, 'teds_evaluator')
    
    def test_init_custom_parameters(self):
        """Test TEDSMetric initialization with custom parameters."""
        metric = TEDSMetric(
            structure_only=True,
            n_jobs=4,
            ignore_nodes=['span', 'b'],
            collect_device='gpu',
            prefix='custom_table'
        )
        assert metric.structure_only == True
        assert metric.n_jobs == 4
        assert metric.ignore_nodes == ['span', 'b']
        assert hasattr(metric, 'teds_evaluator')
    
    def test_html_post_process(self, metric):
        """Test HTML post-processing functionality."""
        # Test HTML already wrapped
        html_wrapped = '<html><body><table><tr><td>test</td></tr></table></body></html>'
        result = metric._html_post_process(html_wrapped)
        assert result == html_wrapped
        
        # Test HTML that needs wrapping
        html_unwrapped = '<table><tr><td>test</td></tr></table>'
        result = metric._html_post_process(html_unwrapped)
        assert result.startswith('<html>')
        assert '<table><tr><td>test</td></tr></table>' in result
    
    def test_process_tokens_to_html(self, metric):
        """Test processing tokens and cells to HTML."""
        pred_text = "<td></td>,<td></td>,</tr>,<td></td>,<td></td>,</tr>,</tbody>"
        pred_cells = ["cell1", "cell2", "cell3", "cell4"]
        
        result = metric._process_tokens_to_html(pred_text, pred_cells)
        
        # Result should be properly formatted HTML
        assert result.startswith('<html>')
        assert 'cell1' in result
        assert 'cell2' in result
        assert 'cell3' in result
        assert 'cell4' in result
    
    def test_process_with_tokens_and_cells(self, metric):
        """Test process method with tokens and cells format (TableMASTER format)."""
        data_samples = [{
            'pred_text': "<td></td>,<td></td>,</tr>,</tbody>",
            'pred_cells': ["cell1", "cell2"],
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert 'teds_score' in metric.results[0]
        assert isinstance(metric.results[0]['teds_score'], float)
        assert 0.0 <= metric.results[0]['teds_score'] <= 1.0
    
    def test_process_with_dict_format(self, metric):
        """Test process method with dictionary format for pred_text and pred_cells."""
        data_samples = [{
            'pred_text': {'item': "<td></td>,<td></td>,</tr>,</tbody>"},
            'pred_cells': {'item': ["cell1", "cell2"]},
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert 'teds_score' in metric.results[0]
    
    def test_process_with_direct_html(self, metric):
        """Test process method with direct HTML format."""
        data_samples = [{
            'pred_table': {
                'html': '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>'
            },
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert 'teds_score' in metric.results[0]
    
    def test_process_with_pred_text_only(self, metric):
        """Test process method with pred_text only."""
        data_samples = [{
            'pred_text': '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>',
            'gt_text': '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>'
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert 'teds_score' in metric.results[0]
    
    def test_process_with_pred_instances(self, metric):
        """Test process method with pred_instances format."""
        class MockInstances:
            def __init__(self, html):
                self.html = html
        
        data_samples = [{
            'pred_instances': MockInstances('<table><tbody><tr><td>cell1</td></tr></tbody></table>'),
            'gt_instances': MockInstances('<table><tbody><tr><td>cell1</td></tr></tbody></table>')
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert 'teds_score' in metric.results[0]
    
    def test_process_with_missing_data(self, metric):
        """Test process method with missing prediction or ground truth."""
        # Missing prediction
        data_samples = [{
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td></tr></tbody></table></body></html>'
            }
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert metric.results[0]['teds_score'] == 0.0
        
        # Reset results for next test
        metric.results = []
        
        # Missing ground truth
        data_samples = [{
            'pred_table': {
                'html': '<table><tbody><tr><td>cell1</td></tr></tbody></table>'
            }
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert metric.results[0]['teds_score'] == 0.0
    
    def test_process_with_invalid_cells_format(self, metric):
        """Test process method with invalid cells format."""
        data_samples = [{
            'pred_text': "<td></td>,<td></td>,</tr>,</tbody>",
            'pred_cells': "invalid_format",  # Should be list but is string
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert 'teds_score' in metric.results[0]
    
    def test_compute_metrics_empty_results(self, metric):
        """Test compute_metrics with empty results."""
        result = metric.compute_metrics([])
        assert result == {'teds': 0.0}
    
    def test_compute_metrics_with_results(self, metric):
        """Test compute_metrics with sample results."""
        results = [
            {'teds_score': 0.8},
            {'teds_score': 0.9},
            {'teds_score': 0.7},
            {'teds_score': 0.6}
        ]
        
        computed = metric.compute_metrics(results)
        
        # Check required metrics
        assert 'teds' in computed
        assert 'teds_max' in computed
        assert 'teds_min' in computed
        
        # Check values
        expected_avg = (0.8 + 0.9 + 0.7 + 0.6) / 4
        assert computed['teds'] == float(f'{expected_avg:.4f}')
        assert computed['teds_max'] == 0.9
        assert computed['teds_min'] == 0.6
        
        # Check all values are floats
        assert all(isinstance(v, float) for v in computed.values())
    
    def test_compute_metrics_single_result(self, metric):
        """Test compute_metrics with single result."""
        results = [{'teds_score': 0.85}]
        
        computed = metric.compute_metrics(results)
        
        assert computed['teds'] == 0.85
        assert computed['teds_max'] == 0.85
        assert computed['teds_min'] == 0.85
    
    def test_perfect_match(self, metric):
        """Test with perfect matching HTML."""
        html = '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
        
        data_samples = [{
            'pred_table': {'html': html},
            'gt_table': {'html': html}
        }]
        
        metric.process([], data_samples)
        computed = metric.compute_metrics(metric.results)
        
        # Perfect match should have high TEDS score (close to 1.0)
        assert computed['teds'] > 0.9
    
    def test_completely_different_tables(self, metric):
        """Test with completely different table structures."""
        data_samples = [{
            'pred_table': {
                'html': '<html><body><table><tbody><tr><td>wrong</td></tr></tbody></table></body></html>'
            },
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr><tr><td>cell3</td><td>cell4</td></tr></tbody></table></body></html>'
            }
        }]
        
        metric.process([], data_samples)
        computed = metric.compute_metrics(metric.results)
        
        # Different structures should have lower TEDS score
        assert computed['teds'] < 1.0
    
    def test_structure_only_mode(self, metric_structure_only):
        """Test TEDSMetric in structure_only mode."""
        # Same structure, different content
        data_samples = [{
            'pred_table': {
                'html': '<html><body><table><tbody><tr><td>different_content</td><td>another_content</td></tr></tbody></table></body></html>'
            },
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        metric_structure_only.process([], data_samples)
        computed = metric_structure_only.compute_metrics(metric_structure_only.results)
        
        # In structure_only mode, content differences should be ignored
        # So score should be high (close to 1.0) for same structure
        assert computed['teds'] > 0.9
    
    def test_multiple_samples_processing(self, metric):
        """Test processing multiple samples in one batch."""
        data_samples = [
            {
                'pred_table': {'html': '<table><tr><td>cell1</td></tr></table>'},
                'gt_table': {'html': '<table><tr><td>cell1</td></tr></table>'}
            },
            {
                'pred_table': {'html': '<table><tr><td>cell2</td></tr></table>'},
                'gt_table': {'html': '<table><tr><td>cell2</td></tr></table>'}
            },
            {
                'pred_table': {'html': '<table><tr><td>different</td></tr></table>'},
                'gt_table': {'html': '<table><tr><td>cell3</td></tr></table>'}
            }
        ]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 3
        assert all('teds_score' in result for result in metric.results)
        
        computed = metric.compute_metrics(metric.results)
        assert 'teds' in computed
        assert 'teds_max' in computed
        assert 'teds_min' in computed


class TestTEDSMetricEdgeCases:
    """Test edge cases and error handling for TEDSMetric."""
    
    def test_empty_html_strings(self):
        """Test with empty HTML strings."""
        metric = TEDSMetric()
        
        data_samples = [{
            'pred_table': {'html': ''},
            'gt_table': {'html': ''}
        }]
        
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert metric.results[0]['teds_score'] == 0.0
    
    def test_malformed_html(self):
        """Test with malformed HTML."""
        metric = TEDSMetric()
        
        data_samples = [{
            'pred_table': {'html': '<table><tr><td>unclosed'},
            'gt_table': {'html': '<table><tr><td>cell1</td></tr></table>'}
        }]
        
        # Should handle malformed HTML gracefully
        metric.process([], data_samples)
        
        assert len(metric.results) == 1
        assert 'teds_score' in metric.results[0]
    
    def test_very_large_tables(self):
        """Test with larger table structures."""
        metric = TEDSMetric()
        
        # Generate a larger table
        large_table_rows = []
        for i in range(10):
            row = ''.join([f'<td>cell_{i}_{j}</td>' for j in range(5)])
            large_table_rows.append(f'<tr>{row}</tr>')
        
        large_table_html = f'<table><tbody>{"".join(large_table_rows)}</tbody></table>'
        
        data_samples = [{
            'pred_table': {'html': large_table_html},
            'gt_table': {'html': large_table_html}
        }]
        
        metric.process([], data_samples)
        computed = metric.compute_metrics(metric.results)
        
        # Should handle large tables and get perfect score for identical tables
        assert computed['teds'] > 0.9


if __name__ == '__main__':
    pytest.main([__file__])
