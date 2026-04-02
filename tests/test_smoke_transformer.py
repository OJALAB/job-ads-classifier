import os

import pytest

from job_offers_classifier.job_offers_classfier import TransformerJobOffersClassifier


pytestmark = [pytest.mark.integration, pytest.mark.transformer]


@pytest.mark.skipif(os.environ.get("RUN_TRANSFORMER_SMOKE") != "1", reason="set RUN_TRANSFORMER_SMOKE=1 to enable")
def test_transformer_classifier_smoke(tmp_path, sample_hierarchy, sample_texts):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    pytest.importorskip("transformers")
    pytest.importorskip("torchmetrics")

    model_dir = tmp_path / "transformer-model"
    model_name = os.environ.get("TRANSFORMER_SMOKE_MODEL", "google/bert_uncased_L-2_H-128_A-2")

    model = TransformerJobOffersClassifier(
        model_dir=str(model_dir),
        hierarchy=sample_hierarchy,
        transformer_model=model_name,
        modeling_mode="bottom-up",
        learning_rate=5e-5,
        max_epochs=1,
        batch_size=2,
        max_sequence_length=32,
        devices=1,
        accelerator="cpu",
        precision="32-true",
        threads=1,
        verbose=False,
    )

    x_train = sample_texts["x_train"][:6]
    y_train = sample_texts["y_train"][:6]
    x_val = sample_texts["x_train"][6:]
    y_val = sample_texts["y_train"][6:]
    model.fit(y_train, x_train, y_val=y_val, X_val=x_val)

    reloaded = TransformerJobOffersClassifier(
        batch_size=2,
        devices=1,
        accelerator="cpu",
        precision="32-true",
        threads=1,
        verbose=False,
    )
    reloaded.load(str(model_dir))
    pred, pred_map = reloaded.predict(sample_texts["x_test"][:2])

    assert pred.shape[0] == 2
    assert set(pred_map.values()) == {"111101", "111102", "211101", "211102"}
