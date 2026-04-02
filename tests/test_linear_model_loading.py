from pathlib import Path

from job_offers_classifier.job_offers_classfier import LinearJobOffersClassifier


def test_linear_loader_defaults_to_bottom_up_without_tree(tmp_path):
    model_dir = tmp_path / "linear-bottom"
    model_dir.mkdir()

    classifier = LinearJobOffersClassifier(model_dir=str(model_dir), verbose=False)
    classifier.model_dir = str(model_dir)

    assert classifier._infer_modeling_mode_on_load() == "bottom-up"


def test_linear_loader_uses_tree_to_detect_top_down(tmp_path):
    model_dir = tmp_path / "linear-top"
    model_dir.mkdir()
    (model_dir / "tree.bin").write_bytes(b"tree")

    classifier = LinearJobOffersClassifier(model_dir=str(model_dir), verbose=False)
    classifier.model_dir = str(model_dir)

    assert classifier._infer_modeling_mode_on_load() == "top-down"
