import importlib.util

import numpy as np
import pytest

from job_offers_classifier.job_offers_classfier import LinearJobOffersClassifier


pytestmark = pytest.mark.integration


def _has_linear_dependencies():
    modules = ["napkinxc", "pystempel", "sklearn", "stop_words"]
    return all(importlib.util.find_spec(module) is not None for module in modules)


@pytest.mark.skipif(not _has_linear_dependencies(), reason="linear model dependencies are not installed")
def test_linear_classifier_smoke(tmp_path, sample_hierarchy, sample_texts):
    model_dir = tmp_path / "linear-model"
    model = LinearJobOffersClassifier(
        model_dir=str(model_dir),
        hierarchy=sample_hierarchy,
        tfidf_vectorizer_min_df=1,
        threads=1,
        verbose=False,
    )
    model.fit(sample_texts["y_train"], sample_texts["x_train"])

    reloaded = LinearJobOffersClassifier(threads=1, verbose=False)
    reloaded.load(str(model_dir))
    pred, pred_map = reloaded.predict(sample_texts["x_test"])

    assert pred.shape == (4, 4)
    assert set(pred_map.values()) == {"111101", "111102", "211101", "211102"}
    assert np.all(pred.max(axis=1) > 0)

