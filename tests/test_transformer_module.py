from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("transformers")
pytest.importorskip("torchmetrics")

from job_offers_classifier.transformer_module import TransformerClassifier


class _DummyMetrics:
    def __init__(self, values):
        self._values = values
        self.reset_called = False

    def compute(self):
        return self._values

    def reset(self):
        self.reset_called = True


def test_format_metric_values_converts_tensors_to_plain_python_values():
    metrics = {
        "val_acc_at_1": torch.tensor(1.0),
        "val_recall_at_3": torch.tensor([0.0, 1.0]),
        "already_plain": 7,
    }

    formatted = TransformerClassifier._format_metric_values(metrics)

    assert formatted == {
        "val_acc_at_1": 1.0,
        "val_recall_at_3": [0.0, 1.0],
        "already_plain": 7,
    }


def test_on_validation_epoch_end_prints_plain_metrics_and_resets(capsys):
    dummy_module = SimpleNamespace(
        hparams=SimpleNamespace(verbose=True),
        val_metrics=_DummyMetrics({"val_acc_at_1": torch.tensor(1.0)}),
    )

    TransformerClassifier.on_validation_epoch_end(dummy_module)

    captured = capsys.readouterr()
    assert "Validation performance:" in captured.out
    assert "'val_acc_at_1': 1.0" in captured.out
    assert "tensor(" not in captured.out
    assert dummy_module.val_metrics.reset_called
