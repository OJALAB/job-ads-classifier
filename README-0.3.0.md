# Hierarchical Job Ads Classifier 0.3.0

Release `0.3.0` focuses on performance and packaging.

Main changes:

- batched transformer tokenization
- dynamic per-batch padding
- configurable transformer pooling with `cls` and `mean`
- optional gradient checkpointing
- benchmark helpers for lazy vs batched tokenization
- refreshed installation and usage docs for `pip` and `uv`
- documented compatibility workflows for public RepOD models

Use the top-level `README.md` as the current installation and usage guide.

Release-specific companion files:

- `BENCHMARK-0.3.0.md`
- `requirements-0.3.0.txt`
- `requirements-colab-0.3.0.txt`

RepOD references for existing pretrained models:

- DOI landing page: [https://doi.org/10.18150/OCUTSI](https://doi.org/10.18150/OCUTSI)
- dataset page: [https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/OCUTSI](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/OCUTSI)

Recommended first compatibility targets:

- `linear-bottom-2024.zip`
- `transformer-bottom-base-2024.zip`
