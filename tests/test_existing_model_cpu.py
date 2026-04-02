import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("EXISTING_MODEL_DIR") or not os.environ.get("EXISTING_MODEL_INPUT"),
    reason="set EXISTING_MODEL_DIR and EXISTING_MODEL_INPUT to validate a real saved model",
)
def test_existing_model_can_predict_on_cpu(tmp_path):
    classifier = os.environ.get("EXISTING_MODEL_CLASSIFIER", "TransformerJobOffersClassifier")
    model_dir = os.environ["EXISTING_MODEL_DIR"]
    input_path = os.environ["EXISTING_MODEL_INPUT"]
    pred_path = tmp_path / "existing-model-predictions.txt"

    cmd = [
        sys.executable,
        "main.py",
        "predict",
        classifier,
        "-x",
        input_path,
        "-m",
        model_dir,
        "-p",
        str(pred_path),
        "-T",
        "1",
    ]

    if classifier == "TransformerJobOffersClassifier":
        cmd.extend(["-A", "cpu", "-P", "32"])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    assert pred_path.exists()
    assert Path(f"{pred_path}.map").exists()

