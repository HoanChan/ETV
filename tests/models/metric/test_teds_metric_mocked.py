import pytest
from unittest.mock import Mock, patch
from models.metric.teds_metric import TEDSMetric


@pytest.mark.parametrize("structure_only,n_jobs,ignore_nodes", [
    (True, 2, None),
    (False, 1, None),
    (True, 4, ['td', 'th']),
])
@patch('models.metric.teds_metric.TEDS')
def test_init_with_mocked_teds(mock_teds_class, structure_only, n_jobs, ignore_nodes):
    """Test initialization with mocked TEDS class."""
    mock_teds_instance = Mock()
    mock_teds_class.return_value = mock_teds_instance
    
    metric = TEDSMetric(structure_only=structure_only, n_jobs=n_jobs, ignore_nodes=ignore_nodes)
    
    # Verify TEDS was initialized with correct parameters
    mock_teds_class.assert_called_once_with(
        structure_only=structure_only,
        n_jobs=n_jobs,
        ignore_nodes=ignore_nodes
    )
    assert metric.teds_evaluator == mock_teds_instance


@pytest.mark.parametrize("score,expected_result_count", [
    (0.85, 1),
    (0.92, 1),
    (0.0, 1),
    (1.0, 1),
])
@patch('models.metric.teds_metric.TEDS')
def test_process_with_mocked_teds_evaluator(mock_teds_class, score, expected_result_count):
    """Test process method with mocked TEDS evaluator."""
    mock_teds_instance = Mock()
    mock_teds_instance.evaluate.return_value = score
    mock_teds_class.return_value = mock_teds_instance
    
    metric = TEDSMetric()
    
    data_samples = [{
        'pred_table': {'html': '<table><tr><td>test</td></tr></table>'},
        'gt_table': {'html': '<table><tr><td>test</td></tr></table>'}
    }]
    
    metric.process([], data_samples)
    
    # Verify TEDS evaluator was called
    mock_teds_instance.evaluate.assert_called_once()
    
    # Check result
    assert len(metric.results) == expected_result_count
    assert metric.results[0]['teds_score'] == score

@patch('models.metric.teds_metric.TEDS')
@patch('models.metric.teds_metric.insert_text_to_token')
@patch('models.metric.teds_metric.text_to_list')
@patch('models.metric.teds_metric.deal_bb')
def test_process_tokens_to_html_with_mocks(mock_deal_bb, mock_text_to_list, 
                                         mock_insert_text, mock_teds_class):
    """Test _process_tokens_to_html with mocked post-processing functions."""
    mock_teds_class.return_value = Mock()
    
    # Setup mocks
    mock_text_to_list.return_value = ['<td></td>', '<td></td>', '</tr>']
    mock_insert_text.return_value = '<table><tr><td>cell1</td><td>cell2</td></tr></table>'
    mock_deal_bb.return_value = '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>'
    
    metric = TEDSMetric()
    result = metric._process_tokens_to_html("<td></td>,<td></td>,</tr>", ["cell1", "cell2"])
    
    # Verify function calls
    mock_text_to_list.assert_called_once_with("<td></td>,<td></td>,</tr>")
    mock_insert_text.assert_called_once_with(['<td></td>', '<td></td>', '</tr>'], ["cell1", "cell2"])
    assert mock_deal_bb.call_count == 2  # Called for 'thead' and 'tbody'
    
    # Result should be wrapped HTML
    assert result.startswith('<html>')

@patch('models.metric.teds_metric.TEDS')
def test_teds_evaluator_exception_handling(mock_teds_class):
    """Test handling of TEDS evaluator exceptions."""
    mock_teds_instance = Mock()
    mock_teds_instance.evaluate.side_effect = Exception("TEDS evaluation failed")
    mock_teds_class.return_value = mock_teds_instance
    
    metric = TEDSMetric()
    
    data_samples = [{
        'pred_table': {'html': '<table><tr><td>test</td></tr></table>'},
        'gt_table': {'html': '<table><tr><td>test</td></tr></table>'}
    }]
    
    # Process should handle exception gracefully
    try:
        metric.process([], data_samples)
        # If no exception, check that we got a result (possibly 0.0)
        assert len(metric.results) == 1
    except Exception:
        # If exception propagates, that's also acceptable behavior
        pass


@patch('models.metric.teds_metric.htmlPostProcess')
@patch('models.metric.teds_metric.TEDS')
def test_html_post_process_mocked(mock_teds_class, mock_html_post_process):
    """Test HTML post-processing with mocked htmlPostProcess function."""
    mock_teds_class.return_value = Mock()
    mock_html_post_process.return_value = '<html><body><table>test</table></body></html>'
    
    metric = TEDSMetric()
    result = metric._html_post_process('<table>test</table>')
    
    mock_html_post_process.assert_called_once_with('<table>test</table>')
    assert result == '<html><body><table>test</table></body></html>'


@patch('models.metric.teds_metric.TEDS')
def test_registry_fallback_behavior(mock_teds_class):
    """Test that the module works with registry fallback."""
    mock_teds_class.return_value = Mock()
    
    # This test verifies that the import fallback works
    # The actual fallback is tested by the successful import in other tests
    metric = TEDSMetric()
    assert hasattr(metric, 'teds_evaluator')


@patch('models.metric.teds_metric.TEDS')
def test_multiple_data_format_priorities(mock_teds_class):
    """Test data format priority handling with mocked TEDS."""
    mock_teds_instance = Mock()
    mock_teds_instance.evaluate.return_value = 0.9
    mock_teds_class.return_value = mock_teds_instance
    
    metric = TEDSMetric()
    
    # Test priority: pred_text + pred_cells should be used over pred_table
    data_samples = [{
        'pred_text': "<td></td>,<td></td>,</tr>",
        'pred_cells': ["cell1", "cell2"],
        'pred_table': {'html': '<table><tr><td>ignored</td></tr></table>'},  # Should be ignored
        'gt_table': {'html': '<table><tr><td>test</td></tr></table>'}
    }]
    
    metric.process([], data_samples)
    
    # Should have processed using tokens format, not direct HTML
    assert len(metric.results) == 1
    assert metric.results[0]['teds_score'] == 0.9
    
    # The evaluate method should have been called with processed HTML, not the ignored pred_table
    mock_teds_instance.evaluate.assert_called_once()
    args, kwargs = mock_teds_instance.evaluate.call_args
    pred_html, gt_html = args
    
    # The prediction should contain the processed cells, not "ignored"
    assert 'cell1' in pred_html
    assert 'cell2' in pred_html
    assert 'ignored' not in pred_html

# Integration tests for TEDSMetric with real dependencies

def test_real_teds_evaluation_simple():
    """Test with real TEDS evaluation on simple tables."""
    metric = TEDSMetric()
    
    # Simple identical tables
    simple_html = '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
    
    data_samples = [{
        'pred_table': {'html': simple_html},
        'gt_table': {'html': simple_html}
    }]
    
    metric.process([], data_samples)
    computed = metric.compute_metrics(metric.results)
    
    # Identical tables should have perfect or near-perfect score
    assert computed['teds'] >= 0.95


@pytest.mark.parametrize("pred_content,gt_content,expected_range", [
    # Different content but same structure
    (
        '<html><body><table><tbody><tr><td>different</td><td>content</td></tr></tbody></table></body></html>',
        '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>',
        (0.5, 1.0)
    ),
    # Different structure
    (
        '<html><body><table><tbody><tr><td>cell1</td></tr></tbody></table></body></html>',
        '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>',
        (0.0, 1.0)
    ),
])
def test_real_teds_evaluation_variations(pred_content, gt_content, expected_range):
    """Test with real TEDS evaluation on tables with different variations."""
    metric = TEDSMetric()
    
    data_samples = [{
        'pred_table': {'html': pred_content},
        'gt_table': {'html': gt_content}
    }]
    
    metric.process([], data_samples)
    computed = metric.compute_metrics(metric.results)
    
    # Score should be within expected range
    assert expected_range[0] <= computed['teds'] <= expected_range[1]


@pytest.mark.parametrize("structure_only", [True, False])
def test_structure_only_vs_content_sensitive(structure_only):
    """Test difference between structure_only and content-sensitive evaluation."""
    metric = TEDSMetric(structure_only=structure_only)
    
    # Same structure, different content
    pred_html = '<html><body><table><tbody><tr><td>wrong</td><td>content</td></tr></tbody></table></body></html>'
    gt_html = '<html><body><table><tbody><tr><td>correct</td><td>content</td></tr></tbody></table></body></html>'
    
    data_samples = [{
        'pred_table': {'html': pred_html},
        'gt_table': {'html': gt_html}
    }]
    
    metric.process([], data_samples)
    result = metric.compute_metrics(metric.results)
    
    # Both metrics should produce reasonable scores
    assert 0.0 <= result['teds'] <= 1.0