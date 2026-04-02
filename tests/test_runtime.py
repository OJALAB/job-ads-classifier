from job_offers_classifier.runtime import normalize_threads, resolve_hardware


def test_normalize_threads_uses_all_cpus_for_minus_one():
    assert normalize_threads(-1) >= 1


def test_normalize_threads_keeps_positive_values():
    assert normalize_threads(3) == 3


def test_resolve_hardware_falls_back_to_cpu_without_torch():
    hardware = resolve_hardware()
    assert hardware["devices"] >= 1
    assert hardware["accelerator"] in {"cpu", "auto", "gpu"}
    if not hardware["torch_available"]:
        assert hardware["accelerator"] == "cpu"
        assert hardware["precision"] == "32-true"

