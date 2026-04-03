# Changelog

## 0.3.0 - 2026-04-03

- added a batched transformer tokenization path with dynamic per-batch padding
- added configurable transformer pooling with `cls` and `mean`
- added optional transformer gradient checkpointing
- reduced transformer prediction overhead by routing prediction through the Lightning datamodule path and disabling prediction logging
- added benchmark helpers for comparing the legacy lazy path with the new batched path
- refreshed the main README so the package can be installed and used directly with `pip` or `uv`
- moved the console entrypoint into the package so standard `pip install .` and wheel installs expose a working `job-ads-classifier` command
- documented the public RepOD pretrained models and the recommended compatibility-validation flows

## 0.2.0 - 2026-04-02

- upgraded the project to a modern Python 3.11/3.12 dependency stack
- migrated transformer training code to current `lightning` and `transformers` APIs
- made CPU and GPU runtime selection explicit and safe
- made optional dependencies lazy so the CLI can start without the full ML stack
- added compatibility handling for older transformer checkpoints
- added pytest unit tests and CPU-oriented smoke tests for server and Colab validation
- refreshed installation instructions and added a modern CPU reference Dockerfile
