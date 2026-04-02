# Colab

This folder contains a minimal Colab notebook for testing release `0.2.0` on CPU or GPU-enabled Colab runtimes.

Recommended starting point:

- `colab-smoke-test-0.2.0.ipynb`
- `colab-repod-pretrained-models-0.2.0.ipynb`

The notebook covers:

- repository clone
- dependency install
- CLI smoke check
- fast pytest suite
- linear smoke test
- optional transformer smoke test
- optional existing-model CPU compatibility test

The RepOD notebook covers:

- direct download from RepOD via Dataverse API
- validation of `linear-bottom-2024.zip`
- validation of `transformer-bottom-base-2024.zip`
