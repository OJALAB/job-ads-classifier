import pytest

torch = pytest.importorskip("torch")

from job_offers_classifier.datasets import TextDataset


class DummyTokenizer:
    def __call__(self, text, add_special_tokens, max_length, padding, truncation, return_tensors):
        assert add_special_tokens is True
        assert max_length == 8
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"

        batch_size = len(text)
        return {
            "input_ids": torch.ones((batch_size, max_length), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),
        }


def test_text_dataset_uses_tokenizer_call_interface():
    dataset = TextDataset(["foo bar", "baz qux"], labels=[0, 1], lazy_encode=True)
    dataset.setup(DummyTokenizer(), max_seq_length=8)

    item = dataset[0]

    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["labels"] == 0
