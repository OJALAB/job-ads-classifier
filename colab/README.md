# Colab

This folder contains Colab notebooks for testing the maintained `0.2.x` line on CPU or GPU-enabled Colab runtimes.

Recommended starting point:

- `colab-smoke-test-0.2.0.ipynb`
- `colab-repod-pretrained-models-0.2.0.ipynb`

Both notebooks currently target repository tag `v0.2.1`.

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
