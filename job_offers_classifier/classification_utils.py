import numpy as np
from scipy.sparse import csr_matrix

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None


def to_csr_matrix(X, dtype=np.float32):
    if isinstance(X, list) and isinstance(X[0], list):
        first_elem = None
        size = 0
        for x in X:
            size += len(x)
            if len(x) and first_elem is None:
                first_elem = x[0]

        if first_elem is None:
            return csr_matrix((len(X), 0), dtype=dtype)

        indptr = np.zeros(len(X) + 1, dtype=np.int32)
        indices = np.zeros(size, dtype=np.int32)
        data = np.ones(size, dtype=dtype)
        cells = 0

        if isinstance(first_elem, int):
            for row, x in enumerate(X):
                indptr[row] = cells
                indices[cells:cells + len(x)] = x
                cells += len(x)
            indptr[len(X)] = cells

        elif isinstance(first_elem, tuple):
            for row, x in enumerate(X):
                indptr[row] = cells
                for x_i in x:
                    indices[cells] = x_i[0]
                    data[cells] = x_i[1]
                    cells += 1
            indptr[len(X)] = cells

        return csr_matrix((data, indices, indptr))
    elif isinstance(X, np.ndarray):
        return csr_matrix(X, dtype=dtype)
    else:
        raise TypeError('Cannot convert X to csr_matrix')


def _require_torch():
    if torch is None:
        raise ImportError(
            "This operation requires PyTorch. Install the transformer dependencies before using tensor helpers."
        )


def csr_vec_to_sparse_tensor(csr_vec):
    _require_torch()
    i = torch.LongTensor([list(csr_vec.indices)])
    v = torch.FloatTensor(csr_vec.data)
    tensor = torch.sparse.FloatTensor(i, v, torch.Size([csr_vec.shape[1]]))
    return tensor


def csr_vec_to_dense_tensor(csr_vec):
    _require_torch()
    tensor = torch.zeros(csr_vec.shape[1], dtype=torch.float)
    tensor[csr_vec.indices] = torch.tensor(csr_vec.data)
    return tensor


def tp_at_k(output, target, top_k):
    _require_torch()
    top_k_idx = torch.argsort(output, dim=1, descending=True)[:, :top_k]
    return target[top_k_idx].sum(dim=1)
