import sys
import os
import pickle
import json
import jsbeautifier
import pickle
import pyreadr
import pandas as pd
import numpy as np
import scipy


def load_obj(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save_obj(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load_to_df(filename):
    ext = filename.split(".")[-1]
    if ext in ["rds"]:
        # df is a dictionary where keys are the name of objects and the values python
        # objects. In the case of Rds there is only one object with None as key
        df = pyreadr.read_r(filename)[None]
    elif ext in ["xlsx", "xls", "odf", "ods", "odt"]:
        df = pd.read_excel(filename)
    elif ext in ["csv"]:
        df = pd.read_csv(filename, low_memory=False, delimiter=",", dtype=object)
    elif ext in ["tsv"]:
        df = pd.read_csv(filename, low_memory=False, delimiter="\t", dtype=object)
    elif ext in ["pkl"]:
        df = pd.read_pickle(filename)

    return df


def load_matrix(filename):
    ext = filename.split(".")[-1]
    if ext in ['npy']:
        mat = np.load(filename)
    elif ext in ['txt', 'gz']:
        mat = np.loadtxt(filename)
    elif ext in ['pkl']:
        mat = load_obj(filename)
    else:
        raise RuntimeError(f"Extension {ext} of {filename} is not supported")

    return mat


def save_matrix(filename, mat):
    ext = filename.split(".")[-1]
    if ext in ['npy']:
        np.save(filename, mat)
    elif ext in ['txt', 'gz']:
        np.savetxt(filename, mat)
    elif ext in ['pkl']:
        save_obj(filename, mat)
    else:
        raise RuntimeError(f"Extension {ext} of {filename} is not supported")


def load_texts(filename):
    texts = []
    with open(filename, encoding="utf-8") as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_as_text(filename, texts):
    with open(filename, "w", encoding="utf-8") as file:
        for line in texts:
            file.write(f"{line}\n")

