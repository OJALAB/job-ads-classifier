from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional

class TransformerDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        model_name_or_path: str,
        max_seq_length: int = 256,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_shuffle = shuffle_train
        self.tokenizer = None
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing TransformerDataModule with model_name={model_name_or_path}, max_seq_length={max_seq_length}, train/eval_batch_size={train_batch_size}/{eval_batch_size}, num_workers={num_workers} ...")

    def _get_dataloader(self, dataset_key, batch_size=32, shuffle=False):
        if dataset_key in self.dataset:
            return DataLoader(self.dataset[dataset_key],
                              batch_size=batch_size,
                              num_workers=self.num_workers,
                              #persistent_workers=True,
                              #pin_memory=True,
                              shuffle=shuffle)
        else:
            return None

    def setup(self, stage=None):
        if self.verbose:
            print("Setting up TransformerDataModule ...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        for subset in self.dataset.values():
            subset.setup(self.tokenizer, self.max_seq_length)

    def train_dataloader(self):
        return self._get_dataloader("train", batch_size=self.train_batch_size, shuffle=self.train_shuffle)

    def val_dataloader(self):
        return self._get_dataloader("val", batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return self._get_dataloader("test", batch_size=self.train_batch_size)
