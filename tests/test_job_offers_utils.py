import numpy as np

from job_offers_classifier.job_offers_utils import (
    create_hierarchy,
    get_parents,
    top_k_prediction,
)
from job_offers_classifier.load_save import load_to_df


def test_get_parents_for_leaf_code():
    assert get_parents("111101") == ["1", "11", "111", "1111"]


def test_create_hierarchy_builds_expected_levels(test_data_dir):
    hierarchy = create_hierarchy(load_to_df(str(test_data_dir / "classes.tsv")))
    assert hierarchy["111101"]["level"] == 4
    assert hierarchy["111101"]["parents"] == ["1", "11", "111", "1111"]


def test_top_k_prediction_returns_sorted_top_k():
    pred = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
    labels, probs = top_k_prediction(pred, 2)
    assert labels.tolist() == [[1, 2]]
    assert np.allclose(probs, [[0.7, 0.2]])

