# Hierarchical Job Ads Classifier 0.2.0

Modernized release for current Python and ML libraries, with CPU/GPU-safe runtime handling and pytest coverage.

## Citation

If you use this repository, please cite:

Beręsewicz, M., Wydmuch, M., Cherniaiev, H., and Pater, R. (2026). *Multilingual Hierarchical Classification of Job Advertisements for Job Vacancy Statistics*. Journal of Official Statistics, 42(1), 23-61. https://doi.org/10.1177/0282423X251395400

GitHub citation metadata is also available in `CITATION.cff`.

## What changed in 0.2.0

- Python support is centered on Python 3.11 and 3.12.
- The transformer path was migrated to modern `lightning` and `transformers` APIs.
- The CLI now handles CPU fallback cleanly.
- Optional dependencies are loaded lazily, so the CLI can start without the full ML stack.
- The repository includes fast unit tests and optional smoke tests for the linear and transformer backends.
- Existing trained models are intended to remain usable, including compatibility handling for older transformer checkpoints.

## Recommended Environment

- Python: `3.11` or `3.12`
- CPU: supported
- GPU: supported when a matching PyTorch build is installed first

Why Python 3.11/3.12:

- the newest NumPy, pandas, and scikit-learn releases require Python 3.11+
- `pystempel` still overlaps cleanly with Python 3.12, which makes `3.11` and `3.12` the safest target range

## Installation

### AlmaLinux / server / local Linux

```bash
conda create -n job-ads-classifier python=3.12 -y
conda activate job-ads-classifier
python -m pip install --upgrade pip
```

Install PyTorch first:

- CPU only:

```bash
pip install torch
```

- GPU:

Use the selector at [pytorch.org](https://pytorch.org/get-started/locally/) and install the matching build first.

Then install the pinned environment for this release:

```bash
pip install -r requirements-0.2.0.txt
```

### Google Colab

```bash
!git clone https://github.com/OJALAB/job-ads-classifier.git
%cd job-ads-classifier
!python -m pip install --upgrade pip
!pip install -r requirements-colab-0.2.0.txt
```

There is also a ready notebook in:

- `colab/colab-smoke-test-0.2.0.ipynb`
- `colab/colab-repod-pretrained-models-0.2.0.ipynb`

## CLI Usage

The repository keeps the `main.py` entry point and also exposes an installable console script:

```bash
job-ads-classifier --help
python main.py --help
```

### Linear model fit

```bash
python main.py fit LinearJobOffersClassifier \
  -x tests/data/x_train.txt \
  -y tests/data/y_train.txt \
  --hierarchy-data tests/data/classes.tsv \
  -m models/linear-demo \
  --tfidf-vectorizer-min-df 1 \
  -T 1
```

### Linear model predict

```bash
python main.py predict LinearJobOffersClassifier \
  -x tests/data/x_test.txt \
  -m models/linear-demo \
  -p predictions-linear.txt \
  -T 1
```

### Transformer model fit

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
  -A cpu \
  -P 32 \
  -T 1
```

For GPU training you can use `-A auto` or `-A gpu`. If CUDA is unavailable, the CLI falls back to CPU safely.

## Tests

### Fast suite

```bash
pytest -m "not integration"
```

### Linear smoke test

```bash
pytest tests/test_smoke_linear.py -q
```

### Transformer smoke test

This is opt-in because it downloads a Hugging Face model:

```bash
RUN_TRANSFORMER_SMOKE=1 pytest tests/test_smoke_transformer.py -q
```

Optional model override:

```bash
RUN_TRANSFORMER_SMOKE=1 \
TRANSFORMER_SMOKE_MODEL=google/bert_uncased_L-2_H-128_A-2 \
pytest tests/test_smoke_transformer.py -q
```

## Existing Models On CPU

To validate an already trained model on a server without retraining:

```bash
conda activate job-ads-classifier
python main.py --help
pytest -m "not integration"
```

Then test the real model directory with CPU inference:

```bash
python main.py predict LinearJobOffersClassifier \
  -x /path/to/input.txt \
  -m /path/to/existing-model \
  -p /tmp/predictions.txt \
  -T 1
```

or, for transformers:

```bash
python main.py predict TransformerJobOffersClassifier \
  -x /path/to/input.txt \
  -m /path/to/existing-model \
  -p /tmp/predictions.txt \
  -A cpu \
  -P 32 \
  -T 1
```

The 0.2.0 loader includes extra compatibility handling for older transformer checkpoints.

### Public pretrained models

Public pretrained classifiers are available in RepOD:

- DOI landing page: `https://doi.org/10.18150/OCUTSI`
- dataset page: `https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/OCUTSI`

Available artifacts include:

- `linear-bottom-2024.zip`
- `linear-top-2024.zip`
- `transformer-bottom-base-2024.zip`
- `transformer-top-base-2024.zip`
- `transformer-bottom-base-rob-2024.zip`
- `transformer-top-base-rob-2024.zip`
- multilingual `xlm-RoBERTa` variants
- split large-model archives stored as `.tar.gz.partaa` and `.partab`

For the first CPU compatibility check, the safest starting point is one of:

- `linear-bottom-2024.zip`
- `linear-top-2024.zip`
- `transformer-bottom-base-2024.zip`
- `transformer-top-base-2024.zip`

If you use one of the split large-model archives, merge the parts first:

```bash
cat transformer-bottom-large-2024.tar.gz.partaa transformer-bottom-large-2024.tar.gz.partab > transformer-bottom-large-2024.tar.gz
tar -xzf transformer-bottom-large-2024.tar.gz
```

## Existing Saved Model CPU Test

On the server you can validate an already trained model directly through pytest:

```bash
EXISTING_MODEL_DIR=/path/to/existing-model \
EXISTING_MODEL_INPUT=/path/to/input.txt \
EXISTING_MODEL_CLASSIFIER=TransformerJobOffersClassifier \
pytest tests/test_existing_model_cpu.py -q
```

For a linear model:

```bash
EXISTING_MODEL_DIR=/path/to/existing-model \
EXISTING_MODEL_INPUT=/path/to/input.txt \
EXISTING_MODEL_CLASSIFIER=LinearJobOffersClassifier \
pytest tests/test_existing_model_cpu.py -q
```

### One-command server validation

There is also a helper script that runs:

- CLI smoke check
- fast pytest suite
- existing-model pytest validation
- one direct CPU prediction run

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
