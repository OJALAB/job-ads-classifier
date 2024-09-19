import json
import torch
from torch.utils.data import Dataset

from job_offers_classifier.classification_utils import *


def get_num_labels(labels):
    if len(labels.shape) > 1:
        return labels.shape[1]
    else:
        return max(labels) + 1


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, num_labels=None, lazy_encode=False, labels_dense_vec=False, labels_groups=None):
        super().__init__()
        self.texts = texts
        self.labels = labels
        if self.labels is None:
            self.labels = np.zeros(len(texts))
        self.labels = np.array(self.labels)
        assert(len(self.texts) == self.labels.shape[0])

        self.lazy_encode = lazy_encode
        self.labels_dense_vec = labels_dense_vec
        # self.labels_groups = labels_groups
        # if labels_groups is not None:
        #     self.labels_groups_set = [set(group) for group in labels_groups]
        #     self.label_to_group_idx = {label: group_idx for group_idx, group in enumerate(labels_groups) for label in group}

        self.num_labels = get_num_labels(labels) if num_labels is None else num_labels
        if len(self.labels.shape) > 1 and self.labels.shape[1] != num_labels:
            self.labels.resize((self.labels[0], num_labels))

        self.encodings = None
        self.tokenizer = None
        self.max_seq_length = None

        # print(f"Initializing TextDataset with {self.labels.shape[0]} data points and {self.num_labels} labels, lazy_encode={lazy_encode}, labels_dense_vec={labels_dense_vec} ...")

    @staticmethod
    def _prepare_encodings(encodings, idx):
        return {key: val[idx] for key, val in encodings.items()}

    def _tokenize(self, text):
        return self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

    def setup(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if not self.lazy_encode:
            # print("Tokenizing dataset ...")
            self.encodings = self._tokenize(self.texts)

    def __len__(self):
        return len(self.texts)

    def get_num_labels(self):
        return self.num_labels

    def __getitem__(self, idx):
        if self.encodings:
            item = TextDataset._prepare_encodings(self.encodings, idx)
        else:
            item = TextDataset._prepare_encodings(self._tokenize([self.texts[idx]]), 0)

        if self.labels_dense_vec:
            item['labels'] = csr_vec_to_dense_tensor(self.labels[idx])
        # elif self.labels_groups is not None:
        #     group_labels = np.zeros(len(self.labels_groups))
        #     labels_group_mask = np.zeros(len(self.labels_groups))
        #     for l in self.labels[idx]:
        #         for g_idx, group in enumerate(self.labels_groups):
        #             l_idx = group.index(l)
        #             if l_idx >= 0:
        #                 group_labels[g_idx] = l_idx
        #                 labels_group_mask[g_idx] = 1
        #     item['labels'] = group_labels
        #     item['labels_group_mask'] = labels_group_mask
        else:
            item['labels'] = self.labels[idx]

        return item
