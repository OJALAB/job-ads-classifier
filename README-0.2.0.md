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
- The Colab workflow now includes dedicated GPU notebooks for prediction with an existing model and for a short example training run.
- Old RepOD transformer models were validated both on CPU and GPU in Colab.
- Short Colab GPU training was validated end-to-end: fit, checkpoint save, reload, and predict.

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
- `colab/colab-existing-model-gpu-predict-0.2.0.ipynb`
- `colab/colab-short-transformer-training-gpu-0.2.0.ipynb`

Notebook roles:

- `colab-smoke-test-0.2.0.ipynb`: fast sanity check for the maintained `0.2.x` line
- `colab-repod-pretrained-models-0.2.0.ipynb`: download old public models from RepOD and validate them on CPU
- `colab-existing-model-gpu-predict-0.2.0.ipynb`: run GPU prediction with an already extracted model directory
- `colab-short-transformer-training-gpu-0.2.0.ipynb`: run a short example GPU transformer training and then predict with the trained model

Current notebook targets:

- the legacy smoke and RepOD notebooks clone `v0.2.1`
- the GPU-specific notebooks clone `main` so they include the latest Colab fixes

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

### Transformer model predict on GPU

```bash
python main.py predict TransformerJobOffersClassifier \
  -x tests/data/x_test.txt \
  -m models/transformer-demo \
  -p predictions-transformer-gpu.txt \
  -A gpu \
  -P 16-mixed \
  -T 1
```

For direct prediction with an existing pretrained transformer model on Colab or a CUDA server, `16`, `16-mixed`, or `bf16-mixed` are the most practical precision options. On CPU, use `-P 32` or `-P 32-true`.

## Transformer backbones

The transformer path currently uses:

- `AutoTokenizer.from_pretrained(...)`
- `AutoModel.from_pretrained(...)`
- the first token representation: `last_hidden_state[:, 0, :]`

This means the safest choices are encoder-style Hugging Face backbones such as:

- BERT
- RoBERTa
- HerBERT
- XLM-RoBERTa
- similar encoder-only models that work with `AutoTokenizer` and `AutoModel`

Practical guidance:

- models like `google/bert_uncased_L-2_H-128_A-2`, HerBERT, and RoBERTa-style encoders are good drop-in choices
- public Hugging Face models may emit warnings about unused task-specific weights when loaded through `AutoModel`; this is usually harmless
- an `HF_TOKEN` is optional for public models and mainly affects rate limits and download convenience

Models that are not a drop-in fit right now:

- decoder-only LLMs such as Gemma, Llama, Mistral, and similar causal language models
- sentence-embedding packages that expect a `sentence-transformers` style pooling stack
- setups where you want to feed precomputed embeddings instead of raw text into the current transformer training path

Why Gemma is not a safe default here:

- the code assumes encoder-style tokenization plus `AutoModel`
- it uses the first-token hidden state as the sentence representation
- decoder-only models do not match that assumption as cleanly as encoder backbones do

So in practice: you can use many other Hugging Face encoder models here, but Gemma is not currently a recommended drop-in replacement.

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

## Existing Models On GPU

For transformers, an already trained model can also be validated directly on GPU:

```bash
python main.py predict TransformerJobOffersClassifier \
  -x /path/to/input.txt \
  -m /path/to/existing-model \
  -p /tmp/predictions.txt \
  -A gpu \
  -P 16-mixed \
  -T 1
```

This path was validated in Colab with an old RepOD transformer model after extraction from:

- `transformer-bottom-base-2024.zip`

The linear path is still primarily a CPU-oriented validation scenario.

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

For the first GPU compatibility check, the most practical starting point is:

- `transformer-bottom-base-2024.zip`

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

## Colab Notes

Some Colab warnings are informational and expected:

- missing `HF_TOKEN` for public Hugging Face models
- Lightning notes about tiny runs, small logging intervals, or model summary precision
- warnings about unused task heads when `AutoModel` loads from a checkpoint that was originally packaged for a different task

Those warnings are usually not fatal if:

- the model downloads correctly
- training reaches checkpoint save
- prediction produces `predictions.txt` and `predictions.txt.map`

Recent Colab-specific fixes included:

- safer metric printing after validation and test, to avoid notebook recursion issues
- GPU notebook defaults aligned with RepOD extraction paths
- GPU training notebook switched to a short Python API fit path with an explicit validation split
