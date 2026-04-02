# Changelog

## 0.2.0 - 2026-04-02

- upgraded the project to a modern Python 3.11/3.12 dependency stack
- migrated transformer training code to current `lightning` and `transformers` APIs
- made CPU and GPU runtime selection explicit and safe
- made optional dependencies lazy so the CLI can start without the full ML stack
- added compatibility handling for older transformer checkpoints
- added pytest unit tests and CPU-oriented smoke tests for server and Colab validation
- refreshed installation instructions and added a modern CPU reference Dockerfile
