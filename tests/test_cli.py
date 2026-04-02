import subprocess
import sys

from click.testing import CliRunner

from main import main


def test_help_works_without_torch_installed():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "LinearJobOffersClassifier" in result.stdout
    assert "TransformerJobOffersClassifier" in result.stdout


def test_fit_requires_labels_and_hierarchy():
    runner = CliRunner()
    result = runner.invoke(main, ["fit", "LinearJobOffersClassifier", "-x", "tests/data/x_train.txt", "-m", "model"])
    assert result.exit_code != 0
    assert "--y-data is required for fit" in result.output


def test_predict_requires_output_path():
    runner = CliRunner()
    result = runner.invoke(main, ["predict", "LinearJobOffersClassifier", "-x", "tests/data/x_test.txt", "-m", "model"])
    assert result.exit_code != 0
    assert "--pred-path is required for predict" in result.output

