# Benchmark 0.3.0

This file tracks the performance evidence for release `0.3.0`.

## What is being measured

Primary release claim:

- the new transformer data path in `0.3.0` is faster than the legacy lazy per-sample tokenization path

Benchmark tools:

- `python scripts/benchmark_0_3_0.py transformer-data ...`
- `python scripts/benchmark_0_3_0.py existing-model-compare ...`

## Reproducible commands

### Synthetic transformer data-path benchmark

```bash
python scripts/benchmark_0_3_0.py transformer-data \
  --text-count 5000 \
  --batch-size 32 \
  --repeats 3 \
  --max-seq-length 128
```

For a more realistic no-network benchmark that still uses a true Hugging Face-style fast tokenizer backend, use the local fast tokenizer mode:

```bash
python scripts/benchmark_0_3_0.py transformer-data \
  --text-count 10000 \
  --batch-size 32 \
  --repeats 3 \
  --max-seq-length 128 \
  --tokenizer-model local-fast
```

### Existing RepOD transformer model benchmark

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

## Notes

- Measured local result for `0.3.0` on `2026-04-03`:
  - command:
    - `python scripts/benchmark_0_3_0.py transformer-data --text-count 10000 --batch-size 32 --repeats 3 --max-seq-length 128 --tokenizer-model local-fast`
  - environment:
    - Python `3.12.7`
    - `torch 2.11.0`
    - `transformers 5.5.0`
    - local temporary venv on macOS arm64
    - note: the benchmark only needed the tokenizer stack; the package release still pins `transformers` to the `5.4.x` line for the maintained classifier environment
  - repeated runs:
    - run 1:
      - legacy path: `6.7150s`
      - optimized batched path: `1.1394s`
      - speedup: `5.89x`
    - run 2:
      - legacy path: `5.1953s`
      - optimized batched path: `0.9754s`
      - speedup: `5.33x`
    - run 3:
      - legacy path: `5.1282s`
      - optimized batched path: `0.9582s`
      - speedup: `5.35x`
  - practical summary:
    - repeated local runs landed between `5.33x` and `5.89x`
    - a representative offline speedup for the new batched path is about `5.4x`
- The synthetic benchmark isolates the tokenizer plus dataloader path and is the quickest way to verify the `0.3.0` speedup.
- A toy fake tokenizer can understate or even hide the benefit because its per-call tokenization cost is unrealistically low. The `local-fast` mode is the recommended offline benchmark.
- The RepOD benchmark measures the real `predict` path on an extracted existing model.
- Linear-model support is kept stable in `0.3.0`; the main claimed runtime improvement in this release targets the transformer path.
