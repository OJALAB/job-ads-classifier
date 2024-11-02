import pandas as pd
import numpy as np


def concat_to_str(x):
    x = [str(x_i) for x_i in x if x_i == x_i]
    x = [x_i for x_i in x if len(x_i)]
    return ' '.join(x)


def fix_class_str(t):
    t = str(t)
    t = t.split('.')[0]
    t = '0' * (6 - len(t)) + t
    return t


def check_code(cl):
    return len(str(cl)) == 6


def class_to_digit(cl, digit=6):
    return str(cl)[:digit]


def get_parents(class_str):
    parents = []
    for digit in [1, 2, 3, 4, 6]:
        if len(class_str) > digit:
            parents.append(class_str[:digit])
    return parents


def create_hierarchy_node(class_str, name):
    parents = get_parents(class_str)
    return {"parents": parents, "label": class_str, "name": name, "level": len(parents)}


def create_hierarchy(df, class_str_field='class', name_field='name'):
    hierarchy = {}
    for i, row in df.iterrows():
        row[class_str_field]
        hierarchy[row[class_str_field]] = create_hierarchy_node(row[class_str_field], row[name_field])
    return hierarchy


def _return_new_dfs(new_dfs):
    if len(new_dfs) == 1:
        new_dfs = new_dfs[0]
    else:
        new_dfs = tuple(new_dfs)
    return new_dfs


def remove_classes(dfs, classes, class_field='class'):
    if not isinstance(dfs, (list, tuple)):
        dfs = [dfs]

    new_dfs = []
    for df in dfs:
        new_df = df.copy()
        for c in classes:
            index = new_df[class_field].astype(str).str.startswith(c)
            new_df = new_df[~index]
        new_dfs.append(new_df)
    return _return_new_dfs(new_dfs)


def remap_classes(dfs, classes_map, class_field='class'):
    if not isinstance(dfs, (list, tuple)):
        dfs = [dfs]

    new_dfs = []
    for df in dfs:
        new_df = df.copy()
        new_df[class_field] = new_df[class_field].apply(lambda x: classes_map.get(x, x))
        new_dfs.append(new_df)
    return _return_new_dfs(new_dfs)


def filter_classes(dfs, classes, class_field='class'):
    if not isinstance(dfs, (list, tuple)):
        dfs = [dfs]

    new_dfs = []
    for df in dfs:
        new_df = df.copy()
        index = new_df[class_field].isin(classes)
        new_df = new_df[index]
        new_dfs.append(new_df)
    return _return_new_dfs(new_dfs)


def top_k_prediction(pred, top_k):
    top_k_labels = np.flip(np.argsort(pred, axis=1))[:, top_k]
    top_k_prob = pred[:, top_k_labels]
    return top_k_labels, top_k_prob
