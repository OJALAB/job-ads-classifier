from pathlib import Path

from job_offers_classifier import job_offers_classfier as joc


def test_load_stemmer_or_rebuild_falls_back_when_pickle_is_legacy(monkeypatch, tmp_path):
    stemmer_path = tmp_path / "stemmer.bin"
    stemmer_path.write_bytes(b"legacy")

    rebuilt = object()

    def fake_load_obj(_):
        raise ModuleNotFoundError("No module named 'stempel'")

    monkeypatch.setattr(joc, "load_obj", fake_load_obj)
    monkeypatch.setattr(joc, "_get_polish_stemmer", lambda verbose: rebuilt)

    assert joc._load_stemmer_or_rebuild(str(stemmer_path), verbose=False) is rebuilt


def test_load_stemmer_or_rebuild_uses_rebuild_when_file_missing(monkeypatch, tmp_path):
    stemmer_path = tmp_path / "missing-stemmer.bin"
    rebuilt = object()

    monkeypatch.setattr(joc, "_get_polish_stemmer", lambda verbose: rebuilt)

    assert joc._load_stemmer_or_rebuild(str(stemmer_path), verbose=False) is rebuilt
