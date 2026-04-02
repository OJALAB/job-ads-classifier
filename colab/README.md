# Colab

This folder contains Colab notebooks for testing the maintained `0.2.x` line on CPU or GPU-enabled Colab runtimes.

Recommended starting point:

- `colab-smoke-test-0.2.0.ipynb`
- `colab-repod-pretrained-models-0.2.0.ipynb`
- `colab-existing-model-gpu-predict-0.2.0.ipynb`
- `colab-short-transformer-training-gpu-0.2.0.ipynb`

The legacy smoke and RepOD notebooks currently target repository tag `v0.2.1`.

The GPU-specific notebooks clone `main` so they include the latest Colab fixes.

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

The existing-model GPU notebook covers:

- GPU runtime check on Colab
- prediction with an already extracted model directory
- preview of `predictions.txt` and `predictions.txt.map`

The short GPU training notebook covers:

- GPU runtime check on Colab
- one short transformer training run on `tests/data`
- prediction with the freshly trained model
- quick inspection of saved model artifacts
