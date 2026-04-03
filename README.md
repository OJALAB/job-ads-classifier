# Hierarchical Job Ads Classifier

Current maintained release: `0.3.0`

This repository provides a hierarchical job ads classifier with two backends:

- a `LinearJobOffersClassifier` based on TF-IDF plus `napkinxc`
- a `TransformerJobOffersClassifier` based on Hugging Face encoder models

The current `0.3.x` line focuses on:

- modern packaging and installation
- compatibility with existing saved models
- CPU and GPU-safe runtime handling
- a faster transformer data path with batched tokenization and dynamic padding

## Citation

If you use this repository, please cite:

Beręsewicz, M., Wydmuch, M., Cherniaiev, H., and Pater, R. (2026). *Multilingual Hierarchical Classification of Job Advertisements for Job Vacancy Statistics*. Journal of Official Statistics, 42(1), 23-61. https://doi.org/10.1177/0282423X251395400

GitHub citation metadata is also available in `CITATION.cff`.

## Python and platform support

- Python: `3.11` or `3.12`
- CPU: supported
- GPU: supported for the transformer backend when a matching PyTorch build is installed first

## Installation

### Recommended extras

The package uses optional dependency groups:

- `linear`: dependencies for the TF-IDF plus `napkinxc` backend
- `transformer`: dependencies for the Hugging Face plus Lightning backend
- `test`: pytest and test helpers

Typical full development install:

```bash
pip install -e ".[linear,transformer,test]"
```

### Install with `pip` from a local checkout

CPU-oriented full install:

```bash
git clone https://github.com/OJALAB/job-ads-classifier.git
cd job-ads-classifier
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[linear,transformer,test]"
```

If you prefer the fully pinned release files instead of extras:

```bash
pip install torch
pip install -r requirements-0.3.0.txt
pip install -e .
```

### Install with `pip` directly from GitHub

Until a dedicated PyPI release is published, install from GitHub or from a built wheel.

```bash
python -m pip install --upgrade pip
pip install "job-ads-classifier[linear,transformer] @ git+https://github.com/OJALAB/job-ads-classifier.git@main"
```

### Install with `uv`

Local editable install:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[linear,transformer,test]"
```

Install from GitHub:

```bash
uv venv
source .venv/bin/activate
uv pip install "job-ads-classifier[linear,transformer] @ git+https://github.com/OJALAB/job-ads-classifier.git@main"
```

### GPU install note

For GPU work, install the matching PyTorch build first and then install the package extras:

```bash
pip install torch  # replace with the correct CUDA build from pytorch.org
pip install -e ".[transformer]"
```

Use the selector at [pytorch.org](https://pytorch.org/get-started/locally/) for the exact CUDA build.

### Build a wheel locally

With `pip` tooling:

```bash
python -m pip install build
python -m build
pip install dist/job_ads_classifier-0.3.0-py3-none-any.whl
```

With `uv`:

```bash
uv build
uv pip install dist/job_ads_classifier-0.3.0-py3-none-any.whl
```

## Quick start

The repository keeps the `main.py` entry point and also exposes the installable console script:

```bash
job-ads-classifier --help
python main.py --help
```

### Linear fit

```bash
python main.py fit LinearJobOffersClassifier \
  -x tests/data/x_train.txt \
  -y tests/data/y_train.txt \
  --hierarchy-data tests/data/classes.tsv \
  -m models/linear-demo \
  --tfidf-vectorizer-min-df 1 \
  -T 1
```

### Linear predict

```bash
python main.py predict LinearJobOffersClassifier \
  -x tests/data/x_test.txt \
  -m models/linear-demo \
  -p predictions-linear.txt \
  -T 1
```

### Transformer fit

```bash
python main.py fit TransformerJobOffersClassifier \
  -x tests/data/x_train.txt \
  -y tests/data/y_train.txt \
  --hierarchy-data tests/data/classes.tsv \
  -m models/transformer-demo \
  -t google/bert_uncased_L-2_H-128_A-2 \
  -mm bottom-up \
  -e 1 \
  -b 2 \
  -s 32 \
  --pooling cls \
  --tokenization-mode batched \
  -A cpu \
  -P 32 \
  -T 1
```

### Transformer predict

```bash
python main.py predict TransformerJobOffersClassifier \
  -x tests/data/x_test.txt \
  -m models/transformer-demo \
  -p predictions-transformer.txt \
  --tokenization-mode batched \
  -A cpu \
  -P 32 \
  -T 1
```

### New transformer options in `0.3.0`

- `--pooling {cls,mean}`
- `--gradient-checkpointing`
- `--tokenization-mode {batched,lazy}`

Defaults:

- `pooling=cls`
- `gradient_checkpointing=False`
- `tokenization_mode=batched`

## What changed in `0.3.0`

- transformer tokenization can now run in a batched pre-encoded mode
- batch collation now uses dynamic padding instead of always padding every sample to `max_sequence_length`
- transformer pooling is now configurable with `cls` or `mean`
- gradient checkpointing is exposed for larger transformer runs
- prediction now goes through the Lightning datamodule path with less logging overhead
- a benchmark helper was added to measure lazy vs batched tokenization paths
- the main README now documents `pip` and `uv` installation flows

## Benchmarks

Release `0.3.0` includes a benchmark helper:

```bash
python scripts/benchmark_0_3_0.py --help
```

### Fast local benchmark for the transformer data path

This benchmark does not require a saved model. It compares the legacy lazy per-sample path with the new batched path using the current dataset and collator implementations.

```bash
python scripts/benchmark_0_3_0.py transformer-data \
  --text-count 5000 \
  --batch-size 32 \
  --repeats 3 \
  --max-seq-length 128
```

### Existing-model benchmark on a RepOD transformer model

Use an extracted RepOD transformer model and compare `lazy` vs `batched` on the same `predict` path:

```bash
python scripts/benchmark_0_3_0.py existing-model-compare \
  --classifier TransformerJobOffersClassifier \
  --model-dir /path/to/transformer-bottom-base-2024/transformer-bottom-base-2024 \
  --input-file tests/data/x_test.txt \
  --pred-path-prefix /tmp/job-ads-bench \
  --accelerator cpu \
  --precision 32 \
  --threads 1 \
  --batch-size 64 \
  --repeats 3
```

The benchmark report for this release is tracked in `BENCHMARK-0.3.0.md`.

### Server script for `0.2.0` vs `0.3.0`

To compare the old release tag with the current checkout on the same extracted model, use:

```bash
bash scripts/compare_server_0_2_0_vs_0_3_0.sh \
  --classifier TransformerJobOffersClassifier \
  --model-dir /path/to/transformer-bottom-base-2024/transformer-bottom-base-2024
```

The script:

- creates temporary git worktrees for `v0.2.0` and the current `HEAD`
- runs repeated `predict` timings on the same input data
- measures `0.3.0` in `batched` mode and, for transformers, also in `lazy` mode
- compares the prediction files and `.map` files
- writes both `summary.json` and `summary.md`

Current measured offline benchmark result for `0.3.0`:

- `transformer-data`, `10000` texts, `batch_size=32`, `repeats=3`, `tokenizer_model=local-fast`
- repeated local runs landed between `5.33x` and `5.89x`
- representative measured speedup: about `5.4x`

Measured end-to-end results on an existing public RepOD transformer were more conservative:

- CPU server predict with `transformer-bottom-base-2024`: `lazy` and `batched` were effectively tied, with no meaningful steady-state speedup
- Colab GPU predict on a larger replicated input: `batched` looked better on the full average because of a very slow first `lazy` run, but steady-state runs were again very close
- practical conclusion: `0.3.0` gives a clearly faster transformer data path in isolation, while full inference on an existing HerBERT-based model remains dominated by model execution rather than tokenization

## Public pretrained models on RepOD

Public pretrained classifiers are available in RepOD:

- DOI landing page: [https://doi.org/10.18150/OCUTSI](https://doi.org/10.18150/OCUTSI)
- dataset page: [https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/OCUTSI](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/OCUTSI)

Important artifacts include:

- `linear-bottom-2024.zip`
- `linear-top-2024.zip`
- `transformer-bottom-base-2024.zip`
- `transformer-top-base-2024.zip`
- `transformer-bottom-base-rob-2024.zip`
- `transformer-top-base-rob-2024.zip`
- multilingual `xlm-RoBERTa` variants
- split large-model archives stored as `.tar.gz.partaa` and `.partab`

Practical starting points:

- first CPU compatibility check:
  - `linear-bottom-2024.zip`
  - `linear-top-2024.zip`
  - `transformer-bottom-base-2024.zip`
  - `transformer-top-base-2024.zip`
- first GPU compatibility check:
  - `transformer-bottom-base-2024.zip`

If you use one of the split large-model archives, merge the parts first:

```bash
cat transformer-bottom-large-2024.tar.gz.partaa transformer-bottom-large-2024.tar.gz.partab > transformer-bottom-large-2024.tar.gz
tar -xzf transformer-bottom-large-2024.tar.gz
```

## Existing-model validation

### Fast CPU validation with pytest

Transformer model:

```bash
EXISTING_MODEL_DIR=/path/to/existing-model \
EXISTING_MODEL_INPUT=/path/to/input.txt \
EXISTING_MODEL_CLASSIFIER=TransformerJobOffersClassifier \
pytest tests/test_existing_model_cpu.py -q
```

Linear model:

```bash
EXISTING_MODEL_DIR=/path/to/existing-model \
EXISTING_MODEL_INPUT=/path/to/input.txt \
EXISTING_MODEL_CLASSIFIER=LinearJobOffersClassifier \
pytest tests/test_existing_model_cpu.py -q
```

### One-command server validation

Transformer example:

```bash
bash scripts/validate_existing_model_cpu.sh \
  TransformerJobOffersClassifier \
  /path/to/extracted-model \
  tests/data/x_test.txt \
  /tmp/job-ads-check
```

Linear example:

```bash
bash scripts/validate_existing_model_cpu.sh \
  LinearJobOffersClassifier \
  /path/to/extracted-model \
  tests/data/x_test.txt \
  /tmp/job-ads-check
```

The current compatibility path has been validated against old RepOD models, including:

- `linear-bottom-2024.zip`
- `transformer-bottom-base-2024.zip`

## Transformer backbones

The transformer backend expects Hugging Face encoder-style models through:

- `AutoTokenizer.from_pretrained(...)`
- `AutoModel.from_pretrained(...)`

Recommended starting points:

- Polish-focused: `allegro/herbert-base-cased`
- multilingual: `FacebookAI/xlm-roberta-base`
- modern general-purpose baseline: `answerdotai/ModernBERT-base`
- smallest smoke-test model: `google/bert_uncased_L-2_H-128_A-2`

Models that are not a drop-in fit for the current pipeline:

- decoder-only LLMs such as Gemma, Llama, Mistral, Qwen, GPT-style models
- sentence-transformers style pipelines that expect dedicated pooling heads

Why Gemma is not a recommended default here:

- the current code is built around encoder-style tokenization and `AutoModel`
- the classifier pools encoder hidden states and feeds them into a classification head
- decoder-only LLMs do not match that interface as cleanly as encoder backbones do

## Tests

Fast suite:

```bash
pytest -m "not integration"
```

Linear smoke test:

```bash
pytest tests/test_smoke_linear.py -q
```

Transformer smoke test:

```bash
RUN_TRANSFORMER_SMOKE=1 pytest tests/test_smoke_transformer.py -q
```

Optional model override:

```bash
RUN_TRANSFORMER_SMOKE=1 \
TRANSFORMER_SMOKE_MODEL=google/bert_uncased_L-2_H-128_A-2 \
pytest tests/test_smoke_transformer.py -q
```

## Colab

The `colab/` folder contains notebooks for the maintained release line:

- `colab/colab-smoke-test-0.2.0.ipynb`
- `colab/colab-repod-pretrained-models-0.2.0.ipynb`
- `colab/colab-existing-model-gpu-predict-0.2.0.ipynb`
- `colab/colab-short-transformer-training-gpu-0.2.0.ipynb`

The GPU-specific notebooks currently track `main` so they include the latest Colab fixes.

For a one-cell Colab benchmark from install to GPU timing, you can also run:

```bash
bash scripts/run_colab_transformer_benchmark_0_3_0.sh
```

This script:

- clones the repository
- installs the `0.3.0` Colab dependencies
- downloads `transformer-bottom-base-2024.zip` from RepOD
- extracts the model
- builds a larger benchmark input file from `tests/data/x_test.txt`
- runs `lazy` vs `batched` benchmark on GPU

## Versioned release docs

Historical and versioned READMEs remain in the repository:

- `README-0.1.0.md`
- `README-0.2.0.md`
- `README-0.3.0.md`

Default dependency pointers now target the maintained line:

- `requirements.txt` -> `requirements-0.3.0.txt`
- `requirements-colab.txt` -> `requirements-colab-0.3.0.txt`
