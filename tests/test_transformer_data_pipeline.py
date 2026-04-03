import pytest


torch = pytest.importorskip("torch")
from job_offers_classifier.collators import DynamicPaddingCollator
from job_offers_classifier.datasets import TextDataset


class FakeTokenizer:
    pad_token_id = 0
    eos_token = None
    pad_token = "[PAD]"

    def __init__(self):
        self.calls = 0

    def _encode_one(self, text, max_length):
        tokens = [101]
        tokens.extend(1000 + len(token) for token in str(text).split())
        if max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def __call__(self, text, add_special_tokens=True, max_length=None, truncation=True, return_attention_mask=True):
        self.calls += 1
        if isinstance(text, (list, tuple)):
            input_ids = [self._encode_one(item, max_length) for item in text]
            attention_mask = [[1] * len(ids) for ids in input_ids]
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        input_ids = self._encode_one(text, max_length)
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def pad(self, features, padding=True, return_tensors="pt"):
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []

        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature["attention_mask"])
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def test_batched_dataset_pretokenizes_and_keeps_variable_lengths():
    tokenizer = FakeTokenizer()
    dataset = TextDataset(
        ["short text", "this is a much longer example text"],
        labels=[0, 1],
        num_labels=2,
        lazy_encode=False,
    )

    dataset.setup(tokenizer, max_seq_length=32)

    first_item = dataset[0]
    second_item = dataset[1]

    assert tokenizer.calls == 1
    assert first_item["input_ids"].ndim == 1
    assert second_item["input_ids"].ndim == 1
    assert first_item["input_ids"].numel() != second_item["input_ids"].numel()


def test_lazy_dataset_tokenizes_on_access():
    tokenizer = FakeTokenizer()
    dataset = TextDataset(
        ["short text", "another text"],
        labels=[0, 1],
        num_labels=2,
        lazy_encode=True,
    )

    dataset.setup(tokenizer, max_seq_length=32)
    assert tokenizer.calls == 0

    _ = dataset[0]
    _ = dataset[1]

    assert tokenizer.calls == 2


def test_dynamic_padding_collator_pads_only_to_batch_max():
    tokenizer = FakeTokenizer()
    dataset = TextDataset(
        ["short text", "this is a much longer example text"],
        labels=[0, 1],
        num_labels=2,
        lazy_encode=False,
    )
    dataset.setup(tokenizer, max_seq_length=32)

    collator = DynamicPaddingCollator(tokenizer)
    batch = collator([dataset[0], dataset[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].tolist() == [0, 1]
    assert batch["input_ids"].shape[1] == max(
        dataset[0]["input_ids"].numel(),
        dataset[1]["input_ids"].numel(),
    )
