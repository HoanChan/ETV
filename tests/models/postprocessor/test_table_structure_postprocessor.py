import pytest
import torch
from unittest.mock import MagicMock
from models.postprocessors.table_structure_postprocessor import TableStructurePostprocessor

@pytest.fixture
def mock_dictionary():
    d = MagicMock()
    d.end_idx = 1
    d.start_idx = 0
    d.padding_idx = 2
    d.idx2str.side_effect = lambda idxs: [f'<tok{i}>' for i in idxs]
    return d

@pytest.fixture
def postprocessor(mock_dictionary):
    return TableStructurePostprocessor(dictionary=mock_dictionary, max_seq_len=10)

@pytest.mark.parametrize("probs,expected_idx", [
    (torch.eye(5)[[0, 3, 1]], [0, 3]),
    (torch.eye(5)[[2, 1]], [2]),
    (torch.eye(5)[[1]], []),
])
def test_get_single_prediction(postprocessor, probs, expected_idx):
    idx, _ = postprocessor.get_single_prediction(probs, None)
    assert idx == expected_idx

@pytest.mark.parametrize("strings,expected", [
    (["<td></td>,<td>,<tr>"], [[1, 1, 0]]),
    (["<tr>,<td></td>"], [[0, 1]]),
    (["<td>"], [[0]]),
])
def test_get_pred_bbox_mask(postprocessor, strings, expected):
    mask = postprocessor._get_pred_bbox_mask(strings)
    assert mask == expected

@pytest.mark.parametrize("scores,expected", [
    ([[0.5, 0.7, 0.9]], [0.7]),
    ([[]], [0.0]),
    ([[1.0]], [1.0]),
])
def test_get_avg_scores(postprocessor, scores, expected):
    assert postprocessor._get_avg_scores(scores) == expected
