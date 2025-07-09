import pytest
from typing import Dict, List, Tuple, Any, Optional
from models.metrics.TEDS.TEDS import TEDS
from models.metrics.teds_metric import TEDSMetric


# Test data samples
TEST_CASES = [
    # Format: (name, pred_html, gt_html, expected_score_range, structure_only, ignore_nodes)
    ('identical_simple', 
     '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
     '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
     (1.0, 1.0), False, None),
    
    ('different_content', 
     '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
     '<table><tr><td>X</td><td>Y</td></tr><tr><td>Z</td><td>W</td></tr></table>',
     (0.4, 0.6), False, None),
    
    ('empty_pred', '', '<table><tr><td>A</td><td>B</td></tr></table>', (0.0, 0.0), False, None),
    ('empty_gt', '<table><tr><td>A</td><td>B</td></tr></table>', '', (0.0, 0.0), False, None),
    ('both_empty', '', '', (0.0, 0.0), False, None),
    
    ('colspan_rowspan', 
     '<table><tr><td colspan="2">A</td></tr><tr><td>B</td><td>C</td></tr></table>',
     '<table><tr><td colspan="2">A</td></tr><tr><td>B</td><td>C</td></tr></table>',
     (1.0, 1.0), False, None),
    
    ('different_structure', 
     '<table><tr><td>A</td><td>B</td></tr></table>',
     '<table><tr><td>A</td></tr><tr><td>B</td></tr></table>',
     (0.0, 0.0), False, None),
    
    ('complex_table', 
     '<table><tr><td rowspan="2">A</td><td>B</td><td>C</td></tr><tr><td>D</td><td>E</td></tr></table>',
     '<table><tr><td rowspan="2">A</td><td>B</td><td>C</td></tr><tr><td>D</td><td>E</td></tr></table>',
     (1.0, 1.0), False, None),
    
    ('with_body_tags', 
     '<html><body><table><tr><td>A</td><td>B</td></tr></table></body></html>',
     '<html><body><table><tr><td>A</td><td>B</td></tr></table></body></html>',
     (1.0, 1.0), False, None),
    
    ('invalid_html', 
     '<table><tr><td>A</td><td>B</td></tr>',
     '<table><tr><td>A</td><td>B</td></tr></table>',
     (0.0, 0.0), False, None),
    
    # Structure-only test cases
    ('same_structure_diff_content_structure_only', 
     '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
     '<table><tr><td>X</td><td>Y</td></tr><tr><td>Z</td><td>W</td></tr></table>',
     (1.0, 1.0), True, None),
    
    ('same_structure_diff_content_normal', 
     '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
     '<table><tr><td>X</td><td>Y</td></tr><tr><td>Z</td><td>W</td></tr></table>',
     (0.4, 0.6), False, None),
    
    # Ignore nodes test cases
    ('ignore_b_i_tags', 
     '<table><tr><td><b>A</b></td><td><i>B</i></td></tr></table>',
     '<table><tr><td>A</td><td>B</td></tr></table>',
     (1.0, 1.0), False, ['b', 'i']),
    
    ('ignore_span_div_tags', 
     '<table><tr><td><span>A</span></td><td><div>B</div></td></tr></table>',
     '<table><tr><td>A</td><td>B</td></tr></table>',
     (1.0, 1.0), False, ['span', 'div']),
    
    ('ignore_nonexistent_tags', 
     '<table><tr><td>A</td><td>B</td></tr></table>',
     '<table><tr><td>A</td><td>B</td></tr></table>',
     (1.0, 1.0), False, ['nonexistent']),
]

EDGE_CASES = [
    ('malformed_pred', '<table><tr><td>A</td>', '<table><tr><td>A</td></tr></table>'),
    ('malformed_gt', '<table><tr><td>A</td></tr></table>', '<table><tr><td>A</td>'),
    ('invalid_pred', 'invalid html', '<table><tr><td>A</td></tr></table>'),
    ('invalid_gt', '<table><tr><td>A</td></tr></table>', 'invalid html'),
    ('both_invalid', 'invalid html', 'invalid html'),
    ('no_table_pred', '<div>content</div>', '<table><tr><td>A</td></tr></table>'),
    ('no_table_gt', '<table><tr><td>A</td></tr></table>', '<div>content</div>'),
    ('nested_tables', '<table><tr><td><table><tr><td>A</td></tr></table></td></tr></table>', 
     '<table><tr><td>A</td></tr></table>'),
    ('special_chars', '<table><tr><td>A&amp;B</td><td>C&lt;D</td></tr></table>',
     '<table><tr><td>A&B</td><td>C<D</td></tr></table>'),
    ('unicode_content', '<table><tr><td>你好</td><td>世界</td></tr></table>',
     '<table><tr><td>你好</td><td>世界</td></tr></table>'),
]

BATCH_TEST_DATA = [
    {'sample_id': 'sample1', 'pred': '<table><tr><td>A</td><td>B</td></tr></table>', 'gt': '<table><tr><td>A</td><td>B</td></tr></table>'},
    {'sample_id': 'sample2', 'pred': '<table><tr><td>X</td><td>Y</td></tr></table>', 'gt': '<table><tr><td>A</td><td>B</td></tr></table>'},
    {'sample_id': 'sample3', 'pred': '', 'gt': '<table><tr><td>A</td><td>B</td></tr></table>'},
]

CONFIG_COMBINATIONS = [
    (True, None),
    (False, None),
    (True, ['b', 'i']),
    (False, ['b', 'i']),
    (True, ['span', 'div']),
    (False, ['span', 'div']),
]


@pytest.mark.parametrize("name,pred,gt,expected_range,structure_only,ignore_nodes", TEST_CASES)
def test_compatibility_comprehensive(name, pred, gt, expected_range, structure_only, ignore_nodes):
    """Test compatibility between old and new TEDS implementations."""
    # Test with old TEDS
    old_teds = TEDS(structure_only=structure_only, ignore_nodes=ignore_nodes)
    old_score = old_teds.evaluate(pred, gt)
    
    # Test with new TEDSMetric
    new_teds = TEDSMetric(structure_only=structure_only, ignore_nodes=ignore_nodes)
    new_score = new_teds.evaluate_single(pred, gt)
    
    # Compare scores
    assert abs(old_score - new_score) < 1e-6, f"Score mismatch for {name}: old={old_score}, new={new_score}"
    
    # Check expected range
    min_expected, max_expected = expected_range
    assert min_expected <= old_score <= max_expected, f"Score {old_score} not in expected range {expected_range} for {name}"


@pytest.mark.parametrize("name,pred,gt", EDGE_CASES)
def test_edge_cases(name, pred, gt):
    """Test various edge cases for robustness."""
    old_teds = TEDS(structure_only=False)
    new_teds = TEDSMetric(structure_only=False)
    
    old_score = old_teds.evaluate(pred, gt)
    new_score = new_teds.evaluate_single(pred, gt)
    
    # Both should return valid scores
    assert isinstance(old_score, float) and isinstance(new_score, float)
    assert 0.0 <= old_score <= 1.0 and 0.0 <= new_score <= 1.0
    assert abs(old_score - new_score) < 1e-6, f"Score mismatch for {name}: old={old_score}, new={new_score}"


@pytest.mark.parametrize("structure_only,ignore_nodes", CONFIG_COMBINATIONS)
def test_configuration_combinations(structure_only, ignore_nodes):
    """Test various configuration combinations."""
    pred_html = '<table><tr><td><b>A</b></td><td><i>B</i></td></tr><tr><td>C</td><td>D</td></tr></table>'
    gt_html = '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>'
    
    old_teds = TEDS(structure_only=structure_only, ignore_nodes=ignore_nodes)
    new_teds = TEDSMetric(structure_only=structure_only, ignore_nodes=ignore_nodes)
    
    old_score = old_teds.evaluate(pred_html, gt_html)
    new_score = new_teds.evaluate_single(pred_html, gt_html)
    
    assert abs(old_score - new_score) < 1e-6, f"Config mismatch: old={old_score}, new={new_score}"


@pytest.mark.parametrize("n_samples", [1, 2, 3, 5, 10])
def test_batch_evaluation(n_samples):
    """Test batch evaluation with different sizes."""
    samples = (BATCH_TEST_DATA * ((n_samples // len(BATCH_TEST_DATA)) + 1))[:n_samples]
    
    # Prepare data for old TEDS
    pred_json = {sample['sample_id']: sample['pred'] for sample in samples}
    true_json = {sample['sample_id']: {'html': sample['gt']} for sample in samples}
    
    # Test with old TEDS
    old_teds = TEDS(structure_only=False, n_jobs=1)
    old_scores = old_teds.batch_evaluate(pred_json, true_json)
    
    # Test with new TEDSMetric
    new_teds = TEDSMetric(structure_only=False)
    data_samples = [
        {'pred_text': {'item': sample['pred']}, 'gt_text': {'item': sample['gt']}}
        for sample in samples
    ]
    
    new_teds.process([], data_samples)
    
    # Compare scores
    for i, sample in enumerate(samples):
        old_score = old_scores[sample['sample_id']]
        new_score = new_teds.results[i]['teds_score']
        assert abs(old_score - new_score) < 1e-6, f"Batch size {n_samples}, sample {i}: old={old_score}, new={new_score}"


@pytest.mark.parametrize("structure_only,ignore_nodes", CONFIG_COMBINATIONS)
def test_batch_with_config(structure_only, ignore_nodes):
    """Test batch evaluation with different configurations."""
    pred_json = {sample['sample_id']: sample['pred'] for sample in BATCH_TEST_DATA}
    true_json = {sample['sample_id']: {'html': sample['gt']} for sample in BATCH_TEST_DATA}
    
    old_teds = TEDS(structure_only=structure_only, ignore_nodes=ignore_nodes, n_jobs=1)
    old_scores = old_teds.batch_evaluate(pred_json, true_json)
    
    new_teds = TEDSMetric(structure_only=structure_only, ignore_nodes=ignore_nodes)
    data_samples = [
        {'pred_text': {'item': sample['pred']}, 'gt_text': {'item': sample['gt']}}
        for sample in BATCH_TEST_DATA
    ]
    
    new_teds.process([], data_samples)
    
    for i, sample in enumerate(BATCH_TEST_DATA):
        old_score = old_scores[sample['sample_id']]
        new_score = new_teds.results[i]['teds_score']
        assert abs(old_score - new_score) < 1e-6, f"Config ({structure_only}, {ignore_nodes}), sample {i}: old={old_score}, new={new_score}"


@pytest.mark.parametrize("results_count", [0, 1, 2, 5, 10])
def test_compute_metrics_variations(results_count):
    """Test compute_metrics with different result counts."""
    new_teds = TEDSMetric(structure_only=False)
    
    if results_count == 0:
        metrics = new_teds.compute_metrics([])
        assert metrics['teds'] == 0.0
    else:
        mock_results = [{'teds_score': 0.5 + (i * 0.1)} for i in range(results_count)]
        metrics = new_teds.compute_metrics(mock_results)
        
        assert isinstance(metrics, dict)
        assert 'teds' in metrics and 'teds_max' in metrics and 'teds_min' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        assert 0.0 <= metrics['teds'] <= 1.0
        
        expected_avg = sum(result['teds_score'] for result in mock_results) / len(mock_results)
        assert abs(metrics['teds'] - expected_avg) < 1e-6


# TEDSMetric specific tests
@pytest.fixture
def metric():
    return TEDSMetric()


@pytest.fixture
def metric_structure_only():
    return TEDSMetric(structure_only=True)


@pytest.mark.parametrize("structure_only,ignore_nodes", [
    (True, ['span', 'b']),
    (False, None),
    (True, ['div']),
])
def test_init_custom_parameters(structure_only, ignore_nodes):
    """Test TEDSMetric initialization with custom parameters."""
    metric = TEDSMetric(structure_only=structure_only, ignore_nodes=ignore_nodes)
    assert metric.structure_only == structure_only
    assert metric.ignore_nodes == ignore_nodes


@pytest.mark.parametrize("html_input,expected_starts_with", [
    ('<html><body><table><tr><td>test</td></tr></table></body></html>', '<html>'),
    ('<table><tr><td>test</td></tr></table>', '<html>'),
])
def test_html_post_process(metric, html_input, expected_starts_with):
    """Test HTML post-processing functionality."""
    result = metric._html_post_process(html_input)
    assert result.startswith(expected_starts_with)
    assert 'test' in result


@pytest.mark.parametrize("data_format", [
    "tokens_and_cells",
    "dict_format", 
    "direct_html",
    "pred_text_only",
    "pred_instances"
])
def test_process_different_formats(metric, data_format):
    """Test process method with different data formats."""
    if data_format == "tokens_and_cells":
        data_samples = [{
            'pred_text': "<td></td>,<td></td>,</tr>,</tbody>",
            'pred_cells': ["cell1", "cell2"],
            'gt_table': {'html': '<table><tr><td>cell1</td><td>cell2</td></tr></table>'}
        }]
    elif data_format == "dict_format":
        data_samples = [{
            'pred_text': {'item': "<td></td>,<td></td>,</tr>,</tbody>"},
            'pred_cells': {'item': ["cell1", "cell2"]},
            'gt_table': {'html': '<table><tr><td>cell1</td><td>cell2</td></tr></table>'}
        }]
    elif data_format == "direct_html":
        data_samples = [{
            'pred_table': {'html': '<table><tr><td>cell1</td><td>cell2</td></tr></table>'},
            'gt_table': {'html': '<table><tr><td>cell1</td><td>cell2</td></tr></table>'}
        }]
    elif data_format == "pred_text_only":
        data_samples = [{
            'pred_text': '<table><tr><td>cell1</td><td>cell2</td></tr></table>',
            'gt_text': '<table><tr><td>cell1</td><td>cell2</td></tr></table>'
        }]
    elif data_format == "pred_instances":
        class MockInstances:
            def __init__(self, html):
                self.html = html
        data_samples = [{
            'pred_instances': MockInstances('<table><tr><td>cell1</td></tr></table>'),
            'gt_instances': MockInstances('<table><tr><td>cell1</td></tr></table>')
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
        data_samples = [{'gt_table': {'html': '<table><tr><td>cell1</td></tr></table>'}}]
    else:
        data_samples = [{'pred_table': {'html': '<table><tr><td>cell1</td></tr></table>'}}]
    
    metric.process([], data_samples)
    assert len(metric.results) == 1
    assert metric.results[0]['teds_score'] == 0.0


@pytest.mark.parametrize("pred_html,gt_html,expected_score", [
    ('', '', 0.0),
    ('<table></table>', '', 0.0),
    ('', '<table></table>', 0.0),
])
def test_empty_html_handling(pred_html, gt_html, expected_score):
    """Test with empty HTML strings."""
    metric = TEDSMetric()
    data_samples = [{'pred_table': {'html': pred_html}, 'gt_table': {'html': gt_html}}]
    metric.process([], data_samples)
    assert metric.results[0]['teds_score'] == expected_score


@pytest.mark.parametrize("results,expected_avg", [
    ([{'teds_score': 0.8}, {'teds_score': 0.9}, {'teds_score': 0.7}], 0.8),
    ([{'teds_score': 0.85}], 0.85),
    ([{'teds_score': 1.0}, {'teds_score': 0.0}], 0.5),
])
def test_compute_metrics_with_results(metric, results, expected_avg):
    """Test compute_metrics with sample results."""
    computed = metric.compute_metrics(results)
    assert 'teds' in computed and 'teds_max' in computed and 'teds_min' in computed
    assert abs(computed['teds'] - expected_avg) < 1e-6
    assert all(isinstance(v, float) for v in computed.values())


def test_perfect_vs_different_tables(metric):
    """Test perfect match vs different tables."""
    perfect_html = '<table><tr><td>cell1</td><td>cell2</td></tr></table>'
    different_html = '<table><tr><td>wrong</td></tr></table>'
    
    # Test perfect match
    data_samples = [{'pred_table': {'html': perfect_html}, 'gt_table': {'html': perfect_html}}]
    metric.process([], data_samples)
    perfect_score = metric.results[0]['teds_score']
    
    # Test different tables
    metric.results = []
    data_samples = [{'pred_table': {'html': different_html}, 'gt_table': {'html': perfect_html}}]
    metric.process([], data_samples)
    different_score = metric.results[0]['teds_score']
    
    assert perfect_score > different_score
    assert perfect_score > 0.9
    assert different_score < 1.0


def test_structure_only_vs_normal_mode():
    """Test structure_only mode vs normal mode."""
    pred_html = '<table><tr><td>different_content</td><td>another_content</td></tr></table>'
    gt_html = '<table><tr><td>cell1</td><td>cell2</td></tr></table>'
    
    # Normal mode
    normal_metric = TEDSMetric(structure_only=False)
    data_samples = [{'pred_table': {'html': pred_html}, 'gt_table': {'html': gt_html}}]
    normal_metric.process([], data_samples)
    normal_score = normal_metric.results[0]['teds_score']
    
    # Structure only mode
    structure_metric = TEDSMetric(structure_only=True)
    structure_metric.process([], data_samples)
    structure_score = structure_metric.results[0]['teds_score']
    
    # Structure only should have higher score for same structure but different content
    assert structure_score > normal_score
    assert structure_score > 0.9


def test_multiple_samples_processing(metric):
    """Test processing multiple samples in one batch."""
    data_samples = [
        {'pred_table': {'html': '<table><tr><td>cell1</td></tr></table>'}, 
         'gt_table': {'html': '<table><tr><td>cell1</td></tr></table>'}},
        {'pred_table': {'html': '<table><tr><td>cell2</td></tr></table>'}, 
         'gt_table': {'html': '<table><tr><td>cell2</td></tr></table>'}},
        {'pred_table': {'html': '<table><tr><td>different</td></tr></table>'}, 
         'gt_table': {'html': '<table><tr><td>cell3</td></tr></table>'}},
    ]
    
    metric.process([], data_samples)
    
    assert len(metric.results) == 3
    assert all('teds_score' in result for result in metric.results)
    
    computed = metric.compute_metrics(metric.results)
    assert all(key in computed for key in ['teds', 'teds_max', 'teds_min'])


@pytest.mark.parametrize("processing_cycles", [1, 2, 3])
def test_multiple_processing_cycles(processing_cycles):
    """Test that the metric can be used multiple times."""
    metric = TEDSMetric(structure_only=False)
    
    for cycle in range(processing_cycles):
        metric.results = []
        
        sample_idx = cycle % len(TEST_CASES)
        name, pred, gt, expected_range, structure_only, ignore_nodes = TEST_CASES[sample_idx]
        
        data_samples = [{'pred_text': {'item': pred}, 'gt_text': {'item': gt}}]
        metric.process([], data_samples)
        metrics = metric.compute_metrics(metric.results)
        
        assert isinstance(metrics['teds'], float)
        assert 0.0 <= metrics['teds'] <= 1.0


def test_malformed_html_handling(metric):
    """Test with malformed HTML."""
    data_samples = [{'pred_table': {'html': '<table><tr><td>unclosed'}, 'gt_table': {'html': '<table><tr><td>cell1</td></tr></table>'}}]
    metric.process([], data_samples)
    assert len(metric.results) == 1
    assert 'teds_score' in metric.results[0]


def test_large_tables(metric):
    """Test with larger table structures."""
    large_table_rows = ''.join([f'<tr>{"".join([f"<td>cell_{i}_{j}</td>" for j in range(5)])}</tr>' for i in range(10)])
    large_table_html = f'<table>{large_table_rows}</table>'
    
    data_samples = [{'pred_table': {'html': large_table_html}, 'gt_table': {'html': large_table_html}}]
    metric.process([], data_samples)
    computed = metric.compute_metrics(metric.results)
    
    assert computed['teds'] > 0.9  # Should get perfect score for identical large tables