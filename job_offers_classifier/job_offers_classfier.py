import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
from stempel import StempelStemmer
from stempel.streams import DataInputStream
from stop_words import get_stop_words
import string
from napkinxc.models import HSM, OVR
import gzip

from job_offers_classifier.load_save import save_obj, load_obj
from job_offers_classifier.datasets import *
from job_offers_classifier.data_modules import TransformerDataModule
from job_offers_classifier.trainer import TrainerWrapper
from job_offers_classifier.transformer_module import TransformerClassifier


# Hack to silent StempelStemmer (via monkey patching)
@classmethod
def from_file_silent(cls, fpath):
    if fpath.endswith(".gz"):
        with gzip.open(fpath, "rb") as f:
            return cls.from_stream(DataInputStream(f, None))
    else:
        with open(fpath, "rb") as f:
            return cls.from_stream(DataInputStream(f, None))

from_file_verbose = StempelStemmer.from_file


class BaseHierarchicalJobOffersClassifier:
    def __init__(self,
                 model_dir=None,
                 hierarchy=None,
                 modeling_mode='bottom-up',
                 verbose=True):
        self.model_dir = model_dir
        self.hierarchy = hierarchy
        self.modeling_mode = modeling_mode
        self.base_model = None
        self.verbose = verbose

    def _init_fit(self):
        if self.model_dir is None:
            raise RuntimeError("Cannot fit with model_dir = None")

        if self.hierarchy is None:
            raise RuntimeError("Cannot fit with hierarchy = None")

        os.makedirs(self.model_dir, exist_ok=True)
        self.hierarchy_path = os.path.join(self.model_dir, "hierarchy.bin")
        save_obj(self.hierarchy_path, self.hierarchy)
        self._process_hierarchy()

    def _process_y(self, y):
        return [self.last_level_labels_map[y_i] for y_i in y]

    # def _bottom_up_process_y(self, y):
    #     return [self.last_level_labels_map[y_i] for y_i in y]
    
    # def _top_down_process_y(self, y):
    #     y = []
    #     for y_i in y:
    #         y.append([self.top_down_labels_map[y_p] for y_p in self.hierarchy[y_i]['parents'] + [y_i]])
    #     return y

    def _get_level_labels(self, level):
        level_labels = sorted([node['label'] for node in self.hierarchy.values() if node['level'] == level])
        return level_labels

    def _process_hierarchy(self):
        # Basic per level information
        self.levels = sorted({node['level'] for node in self.hierarchy.values()})
        self.levels_labels = {level: self._get_level_labels(level) for level in self.levels}
        self.levels_labels_map = {level: {label: i for i, label in enumerate(sorted(labels))} for level, labels in self.levels_labels.items()}
        self.levels_indices_map = {level: {i: label for label, i in labels_map.items()} for level, labels_map in self.levels_labels_map.items()}
        self.level_labels_count = {level: len(labels) for level, labels in self.levels_labels.items()}

        # Check if all mappings are correct
        for level in self.levels:
            assert len(self.levels_labels[level]) == self.level_labels_count[level]
            assert max(self.levels_labels_map[level].values()) == len(self.levels_labels_map[level]) - 1

        # Last level information
        self.last_level = max(self.levels)
        self.last_level_labels = self.levels_labels[self.last_level]
        self.last_level_labels_map = self.levels_labels_map[self.last_level]
        self.last_level_indices_map = self.levels_indices_map[self.last_level]
        self.last_level_labels_count = self.level_labels_count[self.last_level]
        
        # Direct children map
        direct_children_map = {}
        for node in self.hierarchy.values():
            parent = node['parents'][-1] if len(node['parents']) else '-1'
            if parent not in direct_children_map:
                direct_children_map[parent] = []
            direct_children_map[parent].append(node['label'])

        assert len(direct_children_map) == sum([v for k, v in self.level_labels_count.items() if k != self.last_level]) + 1

        # Siblings map
        siblings_map = {}
        for v in direct_children_map.values():
            for c in v:
                siblings_map[c] = v

        # Paths map for last level labels
        self.paths_map = [[] for _ in range(self.last_level_labels_count)]
        for label in self.last_level_labels:
            idx = self.last_level_labels_map[label]
            self.paths_map[idx] = self.hierarchy[label]['parents'] + [label]

        assert len(self.paths_map) == self.last_level_labels_count

        # For top-down modeling
        self.top_down_labels_map = {}
        prev_level_labels = 0
        for level in self.levels:
            labels_map = self.levels_labels_map[level]
            for label, i in labels_map.items():
                self.top_down_labels_map[label] = prev_level_labels + i
            prev_level_labels += self.level_labels_count[level]

        assert len(self.top_down_labels_map) == sum([v for v in self.level_labels_count.values()])
        assert max(self.top_down_labels_map.values()) == len(self.top_down_labels_map) - 1

        self.top_down_indices_map = {i: label for label, i in self.top_down_labels_map.items()}
        
        self.top_down_children_map = {}
        for parent, children in direct_children_map.items():
            self.top_down_children_map[parent] = [self.top_down_labels_map[c] for c in children]

        assert len(self.top_down_children_map) == len(direct_children_map)

        self.top_down_labels_groups = list(self.top_down_children_map.values())
        self.top_down_labels_groups_map = {}
        for i, group in enumerate(self.top_down_labels_groups):
            for label in group:
                self.top_down_labels_groups_map[label] = i

        self.top_down_sibling_map = {}
        for label, i in self.top_down_labels_map.items():
            self.top_down_sibling_map[i] = [self.top_down_labels_map[s] for s in siblings_map[label]]

        self.top_down_labels_paths = [
            [self.top_down_labels_map[p] for p in self.paths_map[i]] for i in range(self.last_level_labels_count)
        ]

        assert len(self.top_down_labels_paths) == self.last_level_labels_count

        self.top_down_labels_groups_mapping = np.full(
            (self.last_level_labels_count, len(self.top_down_labels_groups)), -1, dtype=np.float32
        )

        for label_idx, path in enumerate(self.top_down_labels_paths):
            for p_label in path:
                g_idx = self.top_down_labels_groups_map[p_label]
                group = self.top_down_labels_groups[g_idx]
                self.top_down_labels_groups_mapping[label_idx, g_idx] = group.index(p_label)

 
    def _init_load(self, model_dir):
        self.model_dir = model_dir
        self.hierarchy_path = os.path.join(self.model_dir, "hierarchy.bin")
        self.hierarchy = load_obj(self.hierarchy_path)
        self._process_hierarchy()

    def remap_labels_to_level(self, y, pred_mapping, output_level):
        if self.hierarchy is None:
            raise RuntimeError("Cannot process labels when hierarchy = None")

        if output_level == 'last':
            output_level = self.last_level

        if output_level > self.last_level:
            raise RuntimeError(f"Provided level {output_level} is larger than maximum hierachy level {self.last_level}")

        level_labels = self.levels_labels[output_level]
        pred_mapping_inv = {v: k for k, v in pred_mapping.items()}

        labels_level_parent_mapping = {}
        for node in self.hierarchy.values():
            level_parent = None
            for parent in node['parents'] + [node['label']]:
                if parent in level_labels:
                    level_parent = parent
                    break
            if level_parent is not None:
                labels_level_parent_mapping[node['label']] = pred_mapping_inv[level_parent]

        new_y = [labels_level_parent_mapping[y_i] for y_i in y]
        return new_y

    def predict_for_level_bottom_up(self, pred, pred_mapping, output_level):
        if self.hierarchy is None:
            raise RuntimeError("Cannot process prediction when hierarchy = None")

        level_labels = self.levels_labels[output_level]
        level_mapping = self.levels_labels_map[output_level]
        level_mapping_inv = self.levels_indices_map[output_level]

        #print("Predicting bottom-up ...")
        # Bottom-up accumulation of probabilities
        level_pred = np.zeros((pred.shape[0], len(level_labels)), dtype=np.float32)
        for i in range(pred.shape[1]):
            label = pred_mapping[i]
            level_parent = None
            for parent in self.hierarchy[label]['parents'] + [label]:
                if parent in level_mapping:
                    level_parent = level_mapping[parent]
                    break
            if level_parent is None:
                raise RuntimeError(f"Label {label} is not a child of {output_level} level label")

            level_pred[:,level_parent] += pred[:,i]

        return level_pred, level_mapping_inv

    
    def predict_for_level_top_down(self, factorized_pred, output_level):
        level_pred = np.ones((factorized_pred.shape[0], self.level_labels_count[output_level]), dtype=np.float32)
        level_mapping = self.levels_labels_map[output_level]
        level_mapping_inv = self.levels_indices_map[output_level]

        #print("Predicting for top-down ...")
        # Top-down accumulation of probabilities
        for label in self.levels_labels[output_level]:
            for parent in self.hierarchy[label]['parents'] + [label]:
                level_pred[:, level_mapping[label]] *= factorized_pred[:, self.top_down_labels_map[parent]]

        return level_pred, level_mapping_inv


    def _get_output(self, pred, output_level="last", format="array", top_k=None, pred_type="flat"):
        if output_level == "last":
            output_level = self.last_level
    
        if pred_type == "hierarchical":
            level_pred, level_map = self.predict_for_level_top_down(pred, output_level)
        else:
            if output_level != self.last_level:
                level_pred, level_map = self.predict_for_level_bottom_up(pred, self.last_level_indices_map, output_level)
            else:
                level_pred = pred
                level_map = self.last_level_indices_map

        # Get top_k labels
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError(f"top_k needs to be int > 0, is {top_k}")

            top_k_labels = np.flip(np.argsort(level_pred, axis=1), axis=1)[:, :top_k]
            top_k_prob = np.take_along_axis(level_pred, top_k_labels, axis=1)
            level_pred = (top_k_labels, top_k_prob)

        # Apply requested format
        if format == "array":
            return level_pred, level_map
        elif format == "dataframe":
            if top_k is None:
                # Sort level_map to be sure it's in correct order
                columns = [v for i, v in sorted(list(level_map.items()))]
                return pd.DataFrame(level_pred, columns=columns)
            else:
                columns = [f"class_{i + 1}" for i in range(top_k)] + [f"prob_{i + 1}" for i in range(top_k)]
                df = pd.DataFrame(np.hstack(level_pred), columns=columns)
                for c in columns[:top_k]:
                    df[c] = df[c].apply(lambda x: level_map[int(x)])
                return df
        else:
            raise ValueError(f"Unknown format {format}")


class LinearJobOffersClassifier(BaseHierarchicalJobOffersClassifier):
    def __init__(self,
                 model_dir=None,
                 hierarchy=None,
                 modeling_mode='top-down',
                 eps=0.001,
                 c=10,
                 ensemble=1,
                 threads=-1,
                 use_provided_hierarchy=True,
                 tfidf_vectorizer_min_df=2,
                 verbose=True):
        super().__init__(model_dir=model_dir, hierarchy=hierarchy, modeling_mode=modeling_mode, verbose=verbose)

        self.c = c
        self.eps = eps
        self.ensemble = ensemble
        self.threads = threads
        self.use_provided_hierarchy = use_provided_hierarchy
        self.modeling_mode = modeling_mode

        self.tfidf_vectorizer_path = None
        self.tfidf_vectorizer_min_df = tfidf_vectorizer_min_df
        self.tfidf_vectorizer = None

        self.stemmer_path = None
        self.stemmer = None
        self.verbose = verbose

    def _get_napkixc_hierarchy(self):
        napkixc_hierarchy = [(-1, 0, -1)]
        nodes_map = {node['label']: i + 1 for i, node in enumerate(self.hierarchy.values())}
        for node in self.hierarchy.values():
            node_id = nodes_map[node['label']]
            parent_id = 0
            if len(node['parents']):
                parent_id = nodes_map[node['parents'][-1]]
            label_id = self.last_level_labels_map.get(node['label'], -1)
            napkixc_hierarchy.append((parent_id, node_id, label_id))

        napkinxc_hierarchy_path = os.path.join(self.model_dir, "napkinxc_hierarchy.bin")
        save_obj(napkinxc_hierarchy_path, napkixc_hierarchy)

        return napkixc_hierarchy

    # Fit related methods
    def _process_text(self, X_text, lang='pl'):
        if lang == "pl":
            if self.stemmer is None:
                if self.verbose:
                    print("Loading stemmer ...")
                    StempelStemmer.from_file = from_file_verbose
                else:
                    StempelStemmer.from_file = from_file_silent
                self.stemmer = StempelStemmer.polimorf()
                save_obj(self.stemmer_path, self.stemmer)
            stop_words = set(get_stop_words(lang))
            punc_to_remove = {ord(p): ' ' for p in string.punctuation}
        else:
            raise RuntimeError(f"Language {lang} is not supported")

        if self.verbose:
            print("Processing text ...")
        X_proc_text = [""] * len(X_text)
        for i, x_i in enumerate(tqdm(X_text, disable=not self.verbose)):
            x_i = str(x_i).translate(punc_to_remove)
            x_i = [self.stemmer.stem(w.lower()) for w in x_i.split(' ') if len(w)]
            x_i = ' '.join([w for w in x_i if w is not None and not w in stop_words])
            x_i = x_i.replace("\n", " ").replace("\r", " ")
            X_proc_text[i] = x_i

        return X_proc_text

    def _vectorize_text(self, X_text):
        if self.tfidf_vectorizer is None:
            if self.verbose:
                print("Fitting tf-idf vectorizer ...")

            self.tfidf_vectorizer = TfidfVectorizer(lowercase=False, min_df=self.tfidf_vectorizer_min_df)
            self.tfidf_vectorizer.fit(X_text)
            save_obj(self.tfidf_vectorizer_path, self.tfidf_vectorizer)

        if self.verbose:
            print("Transforming text to tf-idf ...")

        return self.tfidf_vectorizer.transform(X_text)

    def _process_X(self, X_text=None, X_matrix=None):
        if X_text is not None and X_matrix is not None and len(X_text) != X_matrix.shape[0]:
            raise ValueError(f"Size of X_text ({len(X_text)}) does not match size of X_matrix ({X_matrix.shape[0]})")

        X = X_matrix

        if X_text is not None:
            if not isinstance(X_text, list):
                raise ValueError("X_text should be a list of strings or None")

            X_text = self._process_text(X_text)
            X_text = self._vectorize_text(X_text)
            X = X_text

        if X_text is not None and X_matrix is not None:
            X = csr_matrix(hstack([X_text, csr_matrix(X_matrix)]))

        return X

    def fit(self, y, X_text=None, X_matrix=None, save_train_data=True):
        if not isinstance(y, list):
            raise ValueError("y should be a list")

        self._init_fit()

        self.tfidf_vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.bin")
        self.stemmer_path = os.path.join(self.model_dir, "stemmer.bin")

        self.tfidf_vectorizer = None
        self.stemmer = None

        if self.verbose:
            print("Processing X ...")
        X = self._process_X(X_text, X_matrix)

        # print(f"Combined data: {type(X)}, {X.shape}")

        if self.verbose:
            print("Processing y ...")
        y = self._process_y(y)

        if save_train_data:
            train_data_path = os.path.join(self.model_dir, "train_data.libsvm")
            dump_svmlight_file(X, y, train_data_path)

            X_train_data_path = os.path.join(self.model_dir, "X_train.bin")
            save_obj(X_train_data_path, X)
            Y_train_data_path = os.path.join(self.model_dir, "Y_train.bin")
            save_obj(Y_train_data_path, y)

        if self.verbose:
            print("Initializing model ...")
        napkinxc_args = {
            'c': self.c,
            'eps': self.eps,
            'threads': self.threads,
            'verbose': self.verbose,
        }

        if self.ensemble > 1:
            napkinxc_args['ensemble'] = self.ensemble
        if self.modeling_mode == 'top-down' and self.use_provided_hierarchy:
            napkinxc_args['tree_structure'] = self._get_napkixc_hierarchy()

        if self.modeling_mode == 'top-down':
            self.base_model = HSM(self.model_dir, **napkinxc_args)
        elif self.modeling_mode == 'bottom-up':
            self.base_model = OVR(self.model_dir, **napkinxc_args)

        if self.verbose:
            print("Fitting model ...")
        self.base_model.fit(X, y)

    # Prediction related methods
    def load(self, model_dir):
        self._init_load(model_dir)

        self.tfidf_vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.bin")
        self.stemmer_path = os.path.join(self.model_dir, "stemmer.bin")

        self.tfidf_vectorizer = load_obj(self.tfidf_vectorizer_path)
        self.stemmer = load_obj(self.stemmer_path)

        if self.modeling_mode == 'top-down':
            self.base_model = HSM(self.model_dir)
        elif self.modeling_mode == 'bottom-up' or os.path.exists(os.path.join(self.model_dir, "tree.bin")):
            self.base_model = OVR(self.model_dir)
        self.base_model.load()

    def predict(self, X_text=None, X_matrix=None, output_level="last", format='array', top_k=None):
        if self.base_model is None:
            raise RuntimeError(f"Cannot predict, the {self.__class__.__name__} is not fitted")

        print("Processing X ...")
        X = self._process_X(X_text, X_matrix)
        pred = self.base_model.predict_proba(X, top_k=len(self.last_level_labels_map))

        print("Predicting ...")
        last_level_pred = np.zeros((X.shape[0], len(self.last_level_labels_map)), dtype=np.float32)
        for i, p in enumerate(tqdm(pred, disable=not self.verbose)):
            for p_i in p:
                last_level_pred[i, p_i[0]] = p_i[1]

        return self._get_output(last_level_pred, output_level=output_level, format=format, top_k=top_k)


class TransformerJobOffersClassifier(BaseHierarchicalJobOffersClassifier):
    def __init__(self,
                 model_dir=None,
                 hierarchy=None,
                 ckpt_path=None,
                 transformer_model="allegro/herbert-base-cased",
                 transformer_ckpt_path="",
                 modeling_mode='bottom-up',
                 adam_epsilon=1e-8,
                 learning_rate=1e-5,
                 weight_decay=0.01,
                 max_epochs=20,
                 batch_size=8,
                 max_sequence_length=512,
                 early_stopping=False,
                 early_stopping_delta=0.001,
                 early_stopping_patience=1,
                 devices=1,
                 accelerator="auto",
                 num_nodes=1,
                 threads=-1,
                 precision=16,
                 verbose=True):
        super().__init__(model_dir=model_dir, hierarchy=hierarchy, modeling_mode = modeling_mode, verbose=verbose)

        self.ckpt_path = ckpt_path
        self.transformer_model = transformer_model
        self.transformer_ckpt_path = transformer_ckpt_path
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.early_stopping = early_stopping
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.devices = devices
        self.accelerator = accelerator
        self.num_nodes = num_nodes
        self.threads = threads
        if self.threads == -1:
            self.threads = os.cpu_count()
        self.precision = precision
        self.fit_single = False

        self.num_devices = self.devices
        if isinstance(self.devices, (list, tuple)):
            self.num_devices = len(self.devices)


    def _create_text_dataset(self, y, X, labels_groups=None):
        if y is not None and isinstance(y[0], list):
            return TextDataset(X, labels=y,
                    num_labels=self.last_level_labels_count,
                    lazy_encode=True, labels_dense_vec=False, labels_groups=labels_groups)
        else:
            return TextDataset(X, labels=y,
                            num_labels=self.last_level_labels_count,
                            lazy_encode=True, labels_dense_vec=False)

    def _setup_data_module(self, dataset):
        text_dataset = {}

        if 'train' in dataset:
            text_dataset['train'] = self._create_text_dataset(*dataset['train'], labels_groups=dataset.get('labels_groups', None))

        if 'val' in dataset:
            text_dataset['val'] = self._create_text_dataset(*dataset['val'], labels_groups=dataset.get('labels_groups', None))

        if 'test' in dataset:
            text_dataset['test'] = self._create_text_dataset(*dataset['test'], labels_groups=dataset.get('labels_groups', None))

        data_module = TransformerDataModule(
            text_dataset,
            self.transformer_model,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            max_seq_length=self.max_sequence_length,
            num_workers=self.threads if self.num_devices < 2 else 0,
        )
        data_module.setup()
        return data_module

    def _fit(self, y, X, y_val=None, X_val=None, output_size=None, labels_groups=None, labels_paths=None, labels_groups_mapping=None):
        if output_size is None:
            raise RuntimeError("Output size is not provided, this should not happen")

        dataset = {"train": (y, X)}
        if y_val is not None and X_val is not None:
            dataset["val"] = (y_val, X_val)
        # if labels_groups is not None:
        #     dataset['labels_groups'] = labels_groups

        data_module = self._setup_data_module(dataset)
        trainer = TrainerWrapper(ckpt_dir=os.path.join(self.model_dir, "ckpts"),
                                 trainer_args={"max_epochs": self.max_epochs,
                                               "devices": self.devices,
                                               "num_nodes": self.num_nodes,
                                               "precision": self.precision,
                                               "accelerator": self.accelerator},
                                 early_stopping=self.early_stopping,
                                 early_stopping_args={"patience": self.early_stopping_patience,
                                                      "min_delta": self.early_stopping_delta},
                                 verbose=self.verbose)

        self.base_model = TransformerClassifier(
            model_name_or_path=self.transformer_ckpt_path
                if self.transformer_ckpt_path is not None and len(self.transformer_ckpt_path) > 0
                else self.transformer_model,
            output_size=output_size,
            adam_epsilon=self.adam_epsilon,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            verbose=self.verbose, 
            labels_groups=labels_groups, 
            labels_paths=labels_paths, 
            labels_groups_mapping=labels_groups_mapping
        )

        trainer.fit(self.base_model, datamodule=data_module, ckpt_path=self.ckpt_path)
        self.base_model.save_transformer(self.model_dir)

    def fit(self, y, X, y_val=None, X_val=None):
        self._init_fit()

        save_obj(os.path.join(self.model_dir, "transformer_arch.bin"), {
            'transformer_model': self.transformer_model,
            'transformer_ckpt': self.transformer_ckpt_path,
        })

        if self.modeling_mode in ["bottom-up", "bottom-up-flat", "top-down"]:
            y = self._process_y(y)
            if y_val is not None:
                y_val = self._process_y(y_val)

            labels_groups = self.top_down_labels_groups if self.modeling_mode == 'top-down' else None
            labels_paths = self.top_down_labels_paths if self.modeling_mode == 'top-down' else None
            labels_groups_mapping = self.top_down_labels_groups_mapping if self.modeling_mode == 'top-down' else None

            self._fit(y, X, y_val=y_val, X_val=X_val, output_size=self._get_output_size(), labels_groups=labels_groups, labels_paths=labels_paths, labels_groups_mapping=labels_groups_mapping)

        elif self.modeling_mode == "bottom-up-cascade":
            for l in range(self.last_level + 1):
                level_labels = self._get_level_labels(l)
                level_labels_map = {i: label for i, label in enumerate(level_labels)}
                level_y = self.remap_labels_to_level(y, level_labels_map, l)
                level_y_val = None
                if y_val is not None:
                    level_y_val = self.remap_labels_to_level(y_val, level_labels_map, l)

                self._fit(level_y, X, y_val=level_y_val, X_val=X_val, output_size=len(level_labels))
                self.transformer_ckpt_path = self.model_dir

        else:
            raise ValueError(f"Unknown training_mode={self.modeling_mode}")
        
    def _get_output_size(self):
        return len(self.top_down_labels_map) if self.modeling_mode == "top-down" else self.last_level_labels_count

    def load(self, model_dir):
        self._init_load(model_dir)

        transformer_arch = load_obj(os.path.join(self.model_dir, "transformer_arch.bin"))
        self.transformer_model = transformer_arch['transformer_model']
        self.transformer_ckpt_path = transformer_arch['transformer_ckpt']

        labels_groups = self.top_down_labels_groups if self.modeling_mode == 'top-down' else None
        labels_paths = self.top_down_labels_paths if self.modeling_mode == 'top-down' else None
        labels_groups_mapping = self.top_down_labels_groups_mapping if self.modeling_mode == 'top-down' else None

        self.base_model = TransformerClassifier(
            model_name_or_path=self.transformer_ckpt_path if self.transformer_ckpt_path is not None and len(self.transformer_ckpt_path) > 0 else self.transformer_model,
            output_size=self._get_output_size(),
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            verbose=self.verbose,
            labels_groups=labels_groups, 
            labels_paths=labels_paths, 
            labels_groups_mapping=labels_groups_mapping
        )

        # This is a bit hacky, should be done by the proper hook in the TrainerWrapper
        self.ckpt_path = os.path.join(self.model_dir, "transformer_classifier.ckpt")
        if not os.path.exists(self.ckpt_path):
            # Try to find checkpoint in ckpts directory
            ckpt_files = [f for f in os.listdir(os.path.join(self.model_dir, "ckpts")) if f.endswith(".ckpt")]
            if len(ckpt_files) == 0:
                raise RuntimeError(f"Cannot find checkpoint in {self.model_dir}")
            
            # Find the checkpoint with the highest epoch number
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("=")[1].split(".")[0].split("-")[0]), reverse=True)
            self.ckpt_path = os.path.join(self.model_dir, "ckpts", ckpt_files[0])


    def _predict(self, X):
        if self.base_model is None:
            raise RuntimeError(f"Cannot predict, the {self.__class__.__name__} is not fitted")

        dataset = {"test": (None, X)}
        data_module = self._setup_data_module(dataset)

        trainer = TrainerWrapper(ckpt_dir=self.ckpt_path, trainer_args={"devices": self.devices, "precision": self.precision},)
        pred = trainer.predict(self.base_model, dataloaders=data_module.test_dataloader(), ckpt_path=self.ckpt_path)
        return np.array(torch.vstack(pred))

    def predict(self, X, output_level="last", format='array', top_k=None):
        return self._get_output(self._predict(X), output_level=output_level, format=format, top_k=top_k, pred_type="hierarchical" if self.modeling_mode == "top-down" else "flat")

    def predict_hidden(self, X):
        if self.base_model is None:
            raise RuntimeError(f"Cannot predict, the {self.__class__.__name__} is not fitted")

        output_layer = self.base_model.output
        self.base_model.output = None
        hidden = self._predict(X)
        self.base_model.output = output_layer

        return hidden
