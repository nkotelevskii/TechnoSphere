import numpy as np


def matrix_multiply(x, y):
    out = np.dot(x, y)
    return out


def matrix_rowmean(x, weights=np.empty(0)):
    if weights.shape[0] == 0:
        out = x.mean(axis=1)
    else:
        s = weights.sum()
        out = np.sum(x * weights, axis=1) / s
    return out


def cosine_similarity(x, top_n=10, with_mean=True, with_std=True):
    if with_mean:
        x = x - x.mean(axis=1).reshape(x.shape[0], 1)
    if with_std:
        x = x / x.std(axis=1).reshape(x.shape[0], 1)
    y = np.arange(x.shape[0]).reshape((x.shape[0], 1))
    x[y, x.argsort()[:, :-top_n]] = 0
    dot = np.dot(x, x.T)
    z = np.sqrt(dot).diagonal()
    out = dot / np.dot(z.reshape((x.shape[0], 1)),
                       z.reshape((1, x.shape[0])))
    return out
