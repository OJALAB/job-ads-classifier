#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _load_texts(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _synthetic_texts(base_texts, count):
    texts = []
    for i in range(count):
        base = base_texts[i % len(base_texts)]
        texts.append(f"{base} benchmark sample {i} extra tokens {(i % 7) * ' lorem ipsum'}")
    return texts


def _runtime_metadata():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _build_local_fast_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "benchmark": 4,
        "sample": 5,
        "extra": 6,
        "tokens": 7,
        "lorem": 8,
        "ipsum": 9,
    }

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def run_transformer_pipeline_benchmark(text_count, batch_size, repeats, max_seq_length, tokenizer_model=""):
    import torch
    from torch.utils.data import DataLoader, Dataset

    from job_offers_classifier.collators import DynamicPaddingCollator
    from job_offers_classifier.datasets import TextDataset

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = None
        pad_token = "[PAD]"

        def __call__(self, text, add_special_tokens=True, max_length=None, truncation=True, return_attention_mask=True, padding=False, return_tensors=None):
            if isinstance(text, (list, tuple)):
                input_ids = [self._encode_one(item, max_length, padding) for item in text]
                attention_mask = [[1 if token != self.pad_token_id else 0 for token in ids] for ids in input_ids]
                if return_tensors == "pt":
                    return {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    }
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            input_ids = self._encode_one(text, max_length, padding)
            attention_mask = [1 if token != self.pad_token_id else 0 for token in input_ids]
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long),
                    "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                }
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def _encode_one(self, text, max_length, padding):
            tokens = [101]
            tokens.extend(1000 + len(token) for token in str(text).split())
            if max_length is not None:
                tokens = tokens[:max_length]
            if padding == "max_length" and max_length is not None:
                tokens = tokens + [self.pad_token_id] * max(0, max_length - len(tokens))
            return tokens

        def pad(self, features, padding=True, return_tensors="pt"):
            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids = []
            attention_mask = []
            for feature in features:
                ids = list(feature["input_ids"])
                pad_len = max_len - len(ids)
                input_ids.append(ids + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    class LegacyTextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_seq_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoded = self.tokenizer(
                [self.texts[idx]],
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "labels": self.labels[idx],
            }

    test_data = _load_texts(Path("tests/data/x_train.txt"))
    texts = _synthetic_texts(test_data, text_count)
    labels = [i % 4 for i in range(len(texts))]

    if tokenizer_model == "local-fast":
        tokenizer = _build_local_fast_tokenizer()
    elif tokenizer_model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = FakeTokenizer()

    legacy_dataset = LegacyTextDataset(texts, labels, tokenizer, max_seq_length=max_seq_length)
    new_dataset = TextDataset(texts, labels=labels, num_labels=4, lazy_encode=False)
    new_dataset.setup(tokenizer, max_seq_length=max_seq_length)
    new_loader = DataLoader(new_dataset, batch_size=batch_size, collate_fn=DynamicPaddingCollator(tokenizer))
    legacy_loader = DataLoader(legacy_dataset, batch_size=batch_size)

    def time_loader(loader):
        started = time.perf_counter()
        total_batches = 0
        total_tokens = 0
        for _ in range(repeats):
            for batch in loader:
                total_batches += 1
                total_tokens += int(batch["attention_mask"].sum().item())
        elapsed = time.perf_counter() - started
        return {
            "seconds": elapsed,
            "batches": total_batches,
            "tokens": total_tokens,
            "batches_per_second": total_batches / elapsed if elapsed else 0.0,
            "tokens_per_second": total_tokens / elapsed if elapsed else 0.0,
        }

    legacy = time_loader(legacy_loader)
    optimized = time_loader(new_loader)

    return {
        "benchmark": "transformer_data_pipeline",
        "runtime": _runtime_metadata(),
        "text_count": text_count,
        "batch_size": batch_size,
        "repeats": repeats,
        "max_seq_length": max_seq_length,
        "tokenizer_model": tokenizer_model or "fake-tokenizer",
        "legacy": legacy,
        "optimized": optimized,
        "speedup_vs_legacy": legacy["seconds"] / optimized["seconds"] if optimized["seconds"] else None,
    }


def run_existing_model_predict(
    classifier,
    model_dir,
    input_file,
    pred_path,
    accelerator="cpu",
    precision="32",
    threads=1,
    batch_size=64,
    tokenization_mode=None,
):
    command = [
        sys.executable,
        "main.py",
        "predict",
        classifier,
        "-x",
        str(input_file),
        "-m",
        str(model_dir),
        "-p",
        str(pred_path),
        "-A",
        accelerator,
        "-P",
        str(precision),
        "-T",
        str(threads),
        "-b",
        str(batch_size),
    ]
    if tokenization_mode:
        command.extend(["--tokenization-mode", str(tokenization_mode)])

    started = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return {
        "benchmark": "existing_model_predict",
        "runtime": _runtime_metadata(),
        "classifier": classifier,
        "model_dir": str(model_dir),
        "input_file": str(input_file),
        "tokenization_mode": tokenization_mode,
        "seconds": elapsed,
        "stdout_tail": completed.stdout.splitlines()[-5:],
    }


def run_existing_model_compare(
    classifier,
    model_dir,
    input_file,
    pred_path_prefix,
    accelerator="cpu",
    precision="32",
    threads=1,
    batch_size=64,
    repeats=1,
    mode_a="lazy",
    mode_b="batched",
):
    def run_many(mode):
        runs = []
        for repeat in range(repeats):
            runs.append(
                run_existing_model_predict(
                    classifier=classifier,
                    model_dir=model_dir,
                    input_file=input_file,
                    pred_path=Path(f"{pred_path_prefix}-{mode}-{repeat}.txt"),
                    accelerator=accelerator,
                    precision=precision,
                    threads=threads,
                    batch_size=batch_size,
                    tokenization_mode=mode if classifier == "TransformerJobOffersClassifier" else None,
                )
            )
        seconds = [run["seconds"] for run in runs]
        return {
            "mode": mode,
            "runs": seconds,
            "avg_seconds": sum(seconds) / len(seconds),
            "min_seconds": min(seconds),
            "max_seconds": max(seconds),
        }

    result_a = run_many(mode_a)
    result_b = run_many(mode_b)
    faster = result_a if result_a["avg_seconds"] < result_b["avg_seconds"] else result_b
    slower = result_b if faster is result_a else result_a

    return {
        "benchmark": "existing_model_compare",
        "runtime": _runtime_metadata(),
        "classifier": classifier,
        "model_dir": str(model_dir),
        "input_file": str(input_file),
        "repeats": repeats,
        "mode_a": result_a,
        "mode_b": result_b,
        "speedup": slower["avg_seconds"] / faster["avg_seconds"] if faster["avg_seconds"] else None,
        "faster_mode": faster["mode"],
    }


def main():
    parser = argparse.ArgumentParser(description="0.3.0 benchmark helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transformer_parser = subparsers.add_parser("transformer-data", help="benchmark old vs new transformer data pipeline")
    transformer_parser.add_argument("--text-count", type=int, default=5000)
    transformer_parser.add_argument("--batch-size", type=int, default=32)
    transformer_parser.add_argument("--repeats", type=int, default=3)
    transformer_parser.add_argument("--max-seq-length", type=int, default=128)
    transformer_parser.add_argument("--tokenizer-model", default="")

    predict_parser = subparsers.add_parser("existing-model-predict", help="benchmark predict time for an existing saved model")
    predict_parser.add_argument("--classifier", required=True)
    predict_parser.add_argument("--model-dir", required=True)
    predict_parser.add_argument("--input-file", required=True)
    predict_parser.add_argument("--pred-path", required=True)
    predict_parser.add_argument("--accelerator", default="cpu")
    predict_parser.add_argument("--precision", default="32")
    predict_parser.add_argument("--threads", type=int, default=1)
    predict_parser.add_argument("--batch-size", type=int, default=64)
    predict_parser.add_argument("--tokenization-mode", default="", choices=["", "batched", "lazy"])

    compare_parser = subparsers.add_parser("existing-model-compare", help="compare existing model predict runtimes, typically lazy vs batched tokenization for transformers")
    compare_parser.add_argument("--classifier", required=True)
    compare_parser.add_argument("--model-dir", required=True)
    compare_parser.add_argument("--input-file", required=True)
    compare_parser.add_argument("--pred-path-prefix", required=True)
    compare_parser.add_argument("--accelerator", default="cpu")
    compare_parser.add_argument("--precision", default="32")
    compare_parser.add_argument("--threads", type=int, default=1)
    compare_parser.add_argument("--batch-size", type=int, default=64)
    compare_parser.add_argument("--repeats", type=int, default=3)
    compare_parser.add_argument("--mode-a", default="lazy", choices=["lazy", "batched"])
    compare_parser.add_argument("--mode-b", default="batched", choices=["lazy", "batched"])

    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    if args.command == "transformer-data":
        result = run_transformer_pipeline_benchmark(
            text_count=args.text_count,
            batch_size=args.batch_size,
            repeats=args.repeats,
            max_seq_length=args.max_seq_length,
            tokenizer_model=args.tokenizer_model,
        )
    elif args.command == "existing-model-predict":
        result = run_existing_model_predict(
            classifier=args.classifier,
            model_dir=Path(args.model_dir),
            input_file=Path(args.input_file),
            pred_path=Path(args.pred_path),
            accelerator=args.accelerator,
            precision=args.precision,
            threads=args.threads,
            batch_size=args.batch_size,
            tokenization_mode=args.tokenization_mode or None,
        )
    else:
        result = run_existing_model_compare(
            classifier=args.classifier,
            model_dir=Path(args.model_dir),
            input_file=Path(args.input_file),
            pred_path_prefix=Path(args.pred_path_prefix),
            accelerator=args.accelerator,
            precision=args.precision,
            threads=args.threads,
            batch_size=args.batch_size,
            repeats=args.repeats,
            mode_a=args.mode_a,
            mode_b=args.mode_b,
        )

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
