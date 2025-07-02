import os
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.metric.teds_metric import TEDSMetric


@pytest.fixture
def metric():
    """Create a basic TEDSMetric instance for testing."""
    return TEDSMetric()


@pytest.fixture
def metric_structure_only():
    """Create a TEDSMetric instance with structure_only=True."""
    return TEDSMetric(structure_only=True)


def test_init_default_parameters():
    """Test TEDSMetric initialization with default parameters."""
    metric = TEDSMetric()
    assert metric.structure_only == False
    assert metric.n_jobs == 1
    assert metric.ignore_nodes is None
    assert metric.default_prefix == 'table'
    assert hasattr(metric, 'teds_evaluator')


@pytest.mark.parametrize("structure_only,n_jobs,ignore_nodes,expected_structure,expected_jobs", [
    (True, 4, ['span', 'b'], True, 4),
    (False, 2, None, False, 2),
    (True, 1, ['div'], True, 1),
])
def test_init_custom_parameters(structure_only, n_jobs, ignore_nodes, expected_structure, expected_jobs):
    """Test TEDSMetric initialization with custom parameters."""
    metric = TEDSMetric(
        structure_only=structure_only,
        n_jobs=n_jobs,
        ignore_nodes=ignore_nodes,
        collect_device='gpu',
        prefix='custom_table'
    )
    assert metric.structure_only == expected_structure
    assert metric.n_jobs == expected_jobs
    assert metric.ignore_nodes == ignore_nodes
    assert hasattr(metric, 'teds_evaluator')
@pytest.mark.parametrize("html_input,expected_contains", [
    (
        '<html><body><table><tr><td>test</td></tr></table></body></html>',
        '<html><body><table><tr><td>test</td></tr></table></body></html>'
    ),
    (
        '<table><tr><td>test</td></tr></table>',
        '<table><tr><td>test</td></tr></table>'
    ),
])
def test_html_post_process(metric, html_input, expected_contains):
    """Test HTML post-processing functionality."""
    result = metric._html_post_process(html_input)
    if html_input.startswith('<html>'):
        assert result == html_input
    else:
        assert result.startswith('<html>')
        assert expected_contains in result


def test_process_tokens_to_html(metric):
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


@pytest.mark.parametrize("data_format,pred_key,gt_key", [
    ("tokens_and_cells", "pred_text", "gt_table"),
    ("dict_format", "pred_text", "gt_table"),
    ("direct_html", "pred_table", "gt_table"),
    ("pred_text_only", "pred_text", "gt_text"),
    ("pred_instances", "pred_instances", "gt_instances"),
])
def test_process_different_formats(metric, data_format, pred_key, gt_key):
    """Test process method with different data formats."""
    if data_format == "tokens_and_cells":
        data_samples = [{
            'pred_text': "<td></td>,<td></td>,</tr>,</tbody>",
            'pred_cells': ["cell1", "cell2"],
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
    elif data_format == "dict_format":
        data_samples = [{
            'pred_text': {'item': "<td></td>,<td></td>,</tr>,</tbody>"},
            'pred_cells': {'item': ["cell1", "cell2"]},
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
    elif data_format == "direct_html":
        data_samples = [{
            'pred_table': {
                'html': '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>'
            },
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table></body></html>'
            }
        }]
    elif data_format == "pred_text_only":
        data_samples = [{
            'pred_text': '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>',
            'gt_text': '<table><tbody><tr><td>cell1</td><td>cell2</td></tr></tbody></table>'
        }]
    elif data_format == "pred_instances":
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
    assert isinstance(metric.results[0]['teds_score'], float)
    assert 0.0 <= metric.results[0]['teds_score'] <= 1.0


@pytest.mark.parametrize("missing_data_type", ["missing_pred", "missing_gt"])
def test_process_with_missing_data(metric, missing_data_type):
    """Test process method with missing prediction or ground truth."""
    if missing_data_type == "missing_pred":
        data_samples = [{
            'gt_table': {
                'html': '<html><body><table><tbody><tr><td>cell1</td></tr></tbody></table></body></html>'
            }
        }]
    else:  # missing_gt
        data_samples = [{
            'pred_table': {
                'html': '<table><tbody><tr><td>cell1</td></tr></tbody></table>'
            }
        }]
    
    metric.process([], data_samples)
    
    assert len(metric.results) == 1
    assert metric.results[0]['teds_score'] == 0.0


def test_process_with_invalid_cells_format(metric):
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


def test_compute_metrics_empty_results(metric):
    """Test compute_metrics with empty results."""
    result = metric.compute_metrics([])
    assert result == {'teds': 0.0}


@pytest.mark.parametrize("results,expected_avg,expected_max,expected_min", [
    ([{'teds_score': 0.8}, {'teds_score': 0.9}, {'teds_score': 0.7}, {'teds_score': 0.6}], 0.75, 0.9, 0.6),
    ([{'teds_score': 0.85}], 0.85, 0.85, 0.85),
    ([{'teds_score': 1.0}, {'teds_score': 0.0}], 0.5, 1.0, 0.0),
])
def test_compute_metrics_with_results(metric, results, expected_avg, expected_max, expected_min):
    """Test compute_metrics with sample results."""
    computed = metric.compute_metrics(results)
    
    # Check required metrics
    assert 'teds' in computed
    assert 'teds_max' in computed
    assert 'teds_min' in computed
    
    # Check values
    assert computed['teds'] == float(f'{expected_avg:.4f}')
    assert computed['teds_max'] == expected_max
    assert computed['teds_min'] == expected_min
    
    # Check all values are floats
    assert all(isinstance(v, float) for v in computed.values())


def test_perfect_match(metric):
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


def test_completely_different_tables(metric):
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


def test_structure_only_mode(metric_structure_only):
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


def test_multiple_samples_processing(metric):
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


# Edge cases and error handling tests


@pytest.mark.parametrize("pred_html,gt_html,expected_score", [
    ('', '', 0.0),
    ('<table></table>', '', 0.0),
    ('', '<table></table>', 0.0),
])
def test_empty_html_strings(pred_html, gt_html, expected_score):
    """Test with empty HTML strings."""
    metric = TEDSMetric()
    
    data_samples = [{
        'pred_table': {'html': pred_html},
        'gt_table': {'html': gt_html}
    }]
    
    metric.process([], data_samples)
    
    assert len(metric.results) == 1
    assert metric.results[0]['teds_score'] == expected_score


def test_malformed_html():
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


def test_very_large_tables():
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


def test_malformed_html():
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


def test_very_large_tables():
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