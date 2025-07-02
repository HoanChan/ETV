import os
import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.metric.batch_teds_metric import BatchTEDSMetric


class TestBatchTEDSMetric:
    """Test suite for BatchTEDSMetric class."""
    
    @pytest.fixture
    def metric(self):
        """Create a basic BatchTEDSMetric instance for testing."""
        return BatchTEDSMetric()
    
    @pytest.fixture
    def metric_structure_only(self):
        """Create a BatchTEDSMetric instance with structure_only=True."""
        return BatchTEDSMetric(structure_only=True)
    
    @pytest.fixture
    def metric_with_mock_teds(self):
        """Create a BatchTEDSMetric instance with mocked TEDS evaluator."""
        with patch('models.metric.batch_teds_metric.TEDS') as mock_teds:
            mock_instance = Mock()
            mock_instance.batch_evaluate.return_value = {
                'sample_0': 0.95,
                'sample_1': 0.87,
                'sample_2': 0.92
            }
            mock_teds.return_value = mock_instance
            metric = BatchTEDSMetric()
            yield metric, mock_instance
    
    def test_init_default_parameters(self):
        """Test BatchTEDSMetric initialization with default parameters."""
        metric = BatchTEDSMetric()
        assert metric.structure_only == False
        assert metric.n_jobs == 4
        assert metric.ignore_nodes is None
        assert metric.default_prefix == 'table'
        assert hasattr(metric, 'teds_evaluator')
        assert hasattr(metric, 'pred_samples')
        assert hasattr(metric, 'gt_samples')
        assert isinstance(metric.pred_samples, dict)
        assert isinstance(metric.gt_samples, dict)
    
    def test_init_custom_parameters(self):
        """Test BatchTEDSMetric initialization with custom parameters."""
        metric = BatchTEDSMetric(
            structure_only=True,
            n_jobs=8,
            ignore_nodes=['span', 'b'],
            collect_device='gpu',
            prefix='custom_table'
        )
        assert metric.structure_only == True
        assert metric.n_jobs == 8
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
        with patch('models.metric.batch_teds_metric.htmlPostProcess') as mock_post_process:
            mock_post_process.return_value = f'<html><body>{html_unwrapped}</body></html>'
            result = metric._html_post_process(html_unwrapped)
            mock_post_process.assert_called_once_with(html_unwrapped)
            assert result.startswith('<html>')
    
    def test_process_tokens_to_html(self, metric):
        """Test processing tokens and cells to HTML."""
        pred_text = "<td></td>,<td></td>,</tr>,<td></td>,<td></td>,</tr>,</tbody>"
        pred_cells = ["cell1", "cell2", "cell3", "cell4"]
        
        with patch('models.metric.batch_teds_metric.text_to_list') as mock_text_to_list, \
             patch('models.metric.batch_teds_metric.insert_text_to_token') as mock_insert_text, \
             patch('models.metric.batch_teds_metric.deal_bb') as mock_deal_bb, \
             patch.object(metric, '_html_post_process') as mock_html_post:
            
            mock_text_to_list.return_value = ['<td></td>', '<td></td>', '</tr>']
            mock_insert_text.return_value = '<table><tr><td>cell1</td><td>cell2</td></tr></table>'
            mock_deal_bb.side_effect = lambda html, tag: html  # Return unchanged
            mock_html_post.return_value = '<html><body><table><tr><td>cell1</td><td>cell2</td></tr></table></body></html>'
            
            result = metric._process_tokens_to_html(pred_text, pred_cells)
            
            mock_text_to_list.assert_called_once_with(pred_text)
            mock_insert_text.assert_called_once()
            assert mock_deal_bb.call_count == 2  # Called for 'thead' and 'tbody'
            mock_html_post.assert_called_once()
            assert result.startswith('<html>')
    
    def test_process_with_tokens_and_cells_format(self, metric):
        """Test process method with tokens and cells format (TableMASTER format)."""
        data_samples = [{
            'pred_text': "<td></td>,<td></td>,</tr>,</tbody>",
            'pred_cells': ["cell1", "cell2"],
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        with patch.object(metric, '_process_tokens_to_html') as mock_process_tokens:
            mock_process_tokens.return_value = '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            
            metric.process([], data_samples)
            
            mock_process_tokens.assert_called_once_with("<td></td>,<td></td>,</tr>,</tbody>", ["cell1", "cell2"])
            assert len(metric.pred_samples) == 1
            assert len(metric.gt_samples) == 1
            
            sample_id = list(metric.pred_samples.keys())[0]
            assert sample_id.startswith('sample_')
    
    def test_process_with_dict_format(self, metric):
        """Test process method with dictionary format for pred_text and pred_cells."""
        data_samples = [{
            'pred_text': {'item': "<td></td>,<td></td>,</tr>,</tbody>"},
            'pred_cells': {'item': ["cell1", "cell2"]},
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        with patch.object(metric, '_process_tokens_to_html') as mock_process_tokens:
            mock_process_tokens.return_value = '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            
            metric.process([], data_samples)
            
            mock_process_tokens.assert_called_once_with("<td></td>,<td></td>,</tr>,</tbody>", ["cell1", "cell2"])
            assert len(metric.pred_samples) == 1
            assert len(metric.gt_samples) == 1
    
    def test_process_with_direct_html_format(self, metric):
        """Test process method with direct HTML format."""
        data_samples = [{
            'pred_table': {
                'html': '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>'
            },
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        with patch.object(metric, '_html_post_process') as mock_html_post:
            mock_html_post.return_value = '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            
            metric.process([], data_samples)
            
            mock_html_post.assert_called_once()
            assert len(metric.pred_samples) == 1
            assert len(metric.gt_samples) == 1
    
    def test_process_with_pred_text_format(self, metric):
        """Test process method with pred_text format."""
        data_samples = [{
            'pred_text': '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>',
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        with patch.object(metric, '_html_post_process') as mock_html_post:
            mock_html_post.return_value = '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            
            metric.process([], data_samples)
            
            mock_html_post.assert_called_once()
            assert len(metric.pred_samples) == 1
            assert len(metric.gt_samples) == 1
    
    def test_process_with_pred_instances_format(self, metric):
        """Test process method with pred_instances format."""
        mock_instances = Mock()
        mock_instances.html = '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>'
        
        data_samples = [{
            'pred_instances': mock_instances,
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
        
        with patch.object(metric, '_html_post_process') as mock_html_post:
            mock_html_post.return_value = '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            
            metric.process([], data_samples)
            
            mock_html_post.assert_called_once()
            assert len(metric.pred_samples) == 1
            assert len(metric.gt_samples) == 1
    
    def test_process_with_multiple_samples(self, metric):
        """Test process method with multiple data samples."""
        data_samples = [
            {
                'pred_text': "<td></td>,<td></td>,</tr>,</tbody>",
                'pred_cells': ["cell1", "cell2"],
                'gt_table': {'html': '<html><body><table><tr><td>cell1</td><td>cell2</td></tr></table></body></html>'}
            },
            {
                'pred_table': {'html': '<table><tr><td>cell3</td><td>cell4</td></tr></table>'},
                'gt_table': {'html': '<html><body><table><tr><td>cell3</td><td>cell4</td></tr></table></body></html>'}
            }
        ]
        
        with patch.object(metric, '_process_tokens_to_html') as mock_process_tokens, \
             patch.object(metric, '_html_post_process') as mock_html_post:
            
            mock_process_tokens.return_value = '<html><body><table><tr><td>cell1</td><td>cell2</td></tr></table></body></html>'
            mock_html_post.return_value = '<html><body><table><tr><td>cell3</td><td>cell4</td></tr></table></body></html>'
            
            metric.process([], data_samples)
            
            assert len(metric.pred_samples) == 2
            assert len(metric.gt_samples) == 2
            
            sample_ids = list(metric.pred_samples.keys())
            assert all(sid.startswith('sample_') for sid in sample_ids)
    
    def test_process_with_gt_variants(self, metric):
        """Test process method with different GT format variants."""
        # Test with gt_text string
        data_samples = [{
            'pred_text': '<table><tr><td>test</td></tr></table>',
            'gt_text': '<table><tr><td>test</td></tr></table>'
        }]
        
        with patch.object(metric, '_html_post_process') as mock_html_post:
            mock_html_post.side_effect = lambda x: f'<html><body>{x}</body></html>'
            
            metric.process([], data_samples)
            
            assert len(metric.pred_samples) == 1
            assert len(metric.gt_samples) == 1
            
            gt_html = list(metric.gt_samples.values())[0]['html']
            assert gt_html.startswith('<html><body>')
    
    def test_compute_metrics_empty_samples(self, metric):
        """Test compute_metrics with empty samples."""
        result = metric.compute_metrics([])
        
        assert result == {'teds': 0.0}
    
    def test_compute_metrics_with_samples(self, metric_with_mock_teds):
        """Test compute_metrics with stored samples."""
        metric, mock_teds_evaluator = metric_with_mock_teds
        
        # Add some sample data
        metric.pred_samples = {
            'sample_0': '<html><body><table><tr><td>test1</td></tr></table></body></html>',
            'sample_1': '<html><body><table><tr><td>test2</td></tr></table></body></html>',
            'sample_2': '<html><body><table><tr><td>test3</td></tr></table></body></html>'
        }
        metric.gt_samples = {
            'sample_0': {'html': '<html><body><table><tr><td>test1</td></tr></table></body></html>'},
            'sample_1': {'html': '<html><body><table><tr><td>test2</td></tr></table></body></html>'},
            'sample_2': {'html': '<html><body><table><tr><td>test3</td></tr></table></body></html>'}
        }
        
        result = metric.compute_metrics([])
        
        # Verify TEDS evaluator was called
        mock_teds_evaluator.batch_evaluate.assert_called_once_with(
            metric.pred_samples, metric.gt_samples
        )
        
        # Check computed metrics
        assert 'teds' in result
        assert 'teds_max' in result
        assert 'teds_min' in result
        
        # Expected values based on mock return: [0.95, 0.87, 0.92]
        expected_avg = (0.95 + 0.87 + 0.92) / 3
        assert result['teds'] == float(f'{expected_avg:.4f}')
        assert result['teds_max'] == 0.95
        assert result['teds_min'] == 0.87
        
        # Verify samples were cleared
        assert len(metric.pred_samples) == 0
        assert len(metric.gt_samples) == 0
    
    def test_compute_metrics_single_sample(self):
        """Test compute_metrics with single sample."""
        with patch('models.metric.batch_teds_metric.TEDS') as mock_teds:
            mock_instance = Mock()
            mock_instance.batch_evaluate.return_value = {'sample_0': 0.88}
            mock_teds.return_value = mock_instance
            
            metric = BatchTEDSMetric()
            metric.pred_samples = {'sample_0': '<html><body><table><tr><td>test</td></tr></table></body></html>'}
            metric.gt_samples = {'sample_0': {'html': '<html><body><table><tr><td>test</td></tr></table></body></html>'}}
            
            result = metric.compute_metrics([])
            
            assert result['teds'] == 0.88
            assert result['teds_max'] == 0.88
            assert result['teds_min'] == 0.88
    
    def test_process_edge_cases(self, metric):
        """Test process method with edge cases and missing data."""
        # Test with empty pred_cells
        data_samples = [{
            'pred_text': "<td></td>,</tr>,</tbody>",
            'pred_cells': [],
            'gt_table': {'html': '<html><body><table><tr><td></td></tr></table></body></html>'}
        }]
        
        with patch.object(metric, '_process_tokens_to_html') as mock_process_tokens:
            mock_process_tokens.return_value = '<html><body><table><tr><td></td></tr></table></body></html>'
            
            metric.process([], data_samples)
            
            mock_process_tokens.assert_called_once_with("<td></td>,</tr>,</tbody>", [])
    
    def test_process_with_non_list_pred_cells(self, metric):
        """Test process method when pred_cells is not a list."""
        data_samples = [{
            'pred_text': "<td></td>,</tr>,</tbody>",
            'pred_cells': "not_a_list",
            'gt_table': {'html': '<html><body><table><tr><td></td></tr></table></body></html>'}
        }]
        
        with patch.object(metric, '_process_tokens_to_html') as mock_process_tokens:
            mock_process_tokens.return_value = '<html><body><table><tr><td></td></tr></table></body></html>'
            
            metric.process([], data_samples)
            
            # Should convert non-list to empty list
            mock_process_tokens.assert_called_once_with("<td></td>,</tr>,</tbody>", [])
    
    def test_sample_id_generation(self, metric):
        """Test that sample IDs are generated correctly and uniquely."""
        data_samples = [
            {'pred_text': 'test1', 'gt_table': {'html': 'test1'}},
            {'pred_text': 'test2', 'gt_table': {'html': 'test2'}},
            {'pred_text': 'test3', 'gt_table': {'html': 'test3'}}
        ]
        
        with patch.object(metric, '_html_post_process') as mock_html_post:
            mock_html_post.side_effect = lambda x: f'<html><body>{x}</body></html>'
            
            metric.process([], data_samples)
            
            sample_ids = list(metric.pred_samples.keys())
            assert len(sample_ids) == 3
            assert len(set(sample_ids)) == 3  # All unique
            assert all(sid.startswith('sample_') for sid in sample_ids)
    
    def test_integration_with_real_data_format(self, metric):
        """Test integration with realistic data formats."""
        # Simulate TableMASTER output format
        data_samples = [{
            'pred_text': {'item': '<thead>,<tr>,<td></td>,<td></td>,</tr>,</thead>,<tbody>,<tr>,<td></td>,<td></td>,</tr>,</tbody>'},
            'pred_cells': {'item': ['Header1', 'Header2', 'Cell1', 'Cell2']},
            'gt_table': {
                'html': '''<html><body><table>
                    <thead><tr><td>Header1</td><td>Header2</td></tr></thead>
                    <tbody><tr><td>Cell1</td><td>Cell2</td></tr></tbody>
                </table></body></html>'''
            }
        }]
        
        # Don't mock the internal methods to test real integration
        metric.process([], data_samples)
        
        assert len(metric.pred_samples) == 1
        assert len(metric.gt_samples) == 1
        
        # Verify the processed HTML contains expected content
        pred_html = list(metric.pred_samples.values())[0]
        assert 'Header1' in pred_html
        assert 'Header2' in pred_html
        assert 'Cell1' in pred_html
        assert 'Cell2' in pred_html
        assert pred_html.startswith('<html>')


class TestBatchTEDSMetricErrorHandling:
    """Test error handling and edge cases for BatchTEDSMetric."""
    
    def test_process_with_malformed_data(self):
        """Test process method with malformed or missing data."""
        metric = BatchTEDSMetric()
        
        # Test with completely empty sample
        data_samples = [{}]
        
        metric.process([], data_samples)
        
        # Should handle gracefully with empty strings
        assert len(metric.pred_samples) == 1
        assert len(metric.gt_samples) == 1
        
        sample_id = list(metric.pred_samples.keys())[0]
        assert metric.pred_samples[sample_id] == ""
        assert metric.gt_samples[sample_id]['html'] == ""
    
    def test_teds_evaluator_initialization_error(self):
        """Test handling of TEDS evaluator initialization errors."""
        with patch('models.metric.batch_teds_metric.TEDS') as mock_teds:
            mock_teds.side_effect = Exception("TEDS initialization failed")
            
            with pytest.raises(Exception, match="TEDS initialization failed"):
                BatchTEDSMetric()
    
    def test_compute_metrics_evaluator_error(self):
        """Test handling of TEDS evaluator errors during computation."""
        with patch('models.metric.batch_teds_metric.TEDS') as mock_teds:
            mock_instance = Mock()
            mock_instance.batch_evaluate.side_effect = Exception("Evaluation failed")
            mock_teds.return_value = mock_instance
            
            metric = BatchTEDSMetric()
            metric.pred_samples = {'sample_0': '<html>test</html>'}
            metric.gt_samples = {'sample_0': {'html': '<html>test</html>'}}
            
            with pytest.raises(Exception, match="Evaluation failed"):
                metric.compute_metrics([])
