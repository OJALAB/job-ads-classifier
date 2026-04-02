import numpy as np

from job_offers_classifier.job_offers_classfier import BaseHierarchicalJobOffersClassifier


def _build_classifier(sample_hierarchy):
    classifier = BaseHierarchicalJobOffersClassifier(hierarchy=sample_hierarchy)
    classifier._process_hierarchy()
    return classifier


def test_remap_labels_to_top_level(sample_hierarchy):
    classifier = _build_classifier(sample_hierarchy)
    mapping = {0: "1", 1: "2"}
    remapped = classifier.remap_labels_to_level(["111101", "211102"], mapping, 0)
    assert remapped == [0, 1]


def test_predict_for_level_bottom_up(sample_hierarchy):
    classifier = _build_classifier(sample_hierarchy)
    pred = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    level_pred, level_map = classifier.predict_for_level_bottom_up(
        pred,
        classifier.last_level_indices_map,
        0,
    )
    assert level_map == {0: "1", 1: "2"}
    assert np.allclose(level_pred, [[0.3, 0.7]])


def test_get_output_top_k_dataframe(sample_hierarchy):
    classifier = _build_classifier(sample_hierarchy)
    pred = np.array([[0.6, 0.2, 0.1, 0.1]], dtype=np.float32)
    df = classifier._get_output(pred, format="dataframe", top_k=2)
    assert list(df.columns) == ["class_1", "class_2", "prob_1", "prob_2"]
    assert df.iloc[0]["class_1"] == "111101"

