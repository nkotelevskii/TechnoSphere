from numba import jit
import numpy as np


@jit(nopython=True)
def matrix_multiply(x, y):
    out = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            for k in range(x.shape[1]):
                out[i, j] += x[i, k] * y[k, j]
    return out


@jit(nopython=True)
def matrix_rowmean(x, weights=np.empty(0)):
    out = np.zeros((x.shape[0]))
    if weights.shape[0] == 0:
        for i in range(x.shape[0]):
            s = 0.
            for j in range(x[i].shape[0]):
                s = s + x[i, j]
            out[i] = s / x[i].shape[0]
    else:
        y = np.empty_like(x)
        w = 0.
        for i in range(weights.shape[0]):
            w = w + weights[i]
        for i in range(x.shape[0]):
            for j in range(weights.shape[0]):
                y[i, j] = x[i, j] * weights[j]
        for i in range(y.shape[0]):
            s = 0.
            for j in range(y[i].shape[0]):
                s += y[i, j]
            out[i] = s / w
    return out


@jit(nopython=True)
def cosine_similarity(x, top_n=10, with_mean=True, with_std=True):
    if with_mean:
        for i in range(x.shape[0]):
            mean = 0.
            for j in range(x[i].shape[0]):
                mean = mean + x[i, j]
            mean = mean / x[i].shape[0]
            for j in range(x[i].shape[0]):
                x[i, j] = x[i, j] - mean
    if with_std:
        for i in range(x.shape[0]):
            std = 0.
            mean = 0.
            for j in range(x[i].shape[0]):
                mean = mean + x[i, j]
            mean = mean / x[i].shape[0]
            for j in range(x[i].shape[0]):
                std = std + (x[i, j] - mean) ** 2
            std = (std / x[i].shape[0]) ** 0.5
            for j in range(x[i].shape[0]):
                x[i, j] = x[i, j] / std

    for i in range(x.shape[0]):
        y = sorted(x[i])[:-top_n]
        for j in range(len(y)):
            for k in range(x[i].shape[0]):
                if x[i, k] == y[j]:
                    x[i, k] = 0
                    break

    dot = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            for k in range(x[i].shape[0]):
                dot[i, j] = dot[i, j] + x[i, k] * x[j, k]
    z = np.zeros((x.shape[0]))
    for i in range(x.shape[0]):
        z[i] = dot[i, i] ** 0.5
    norm = np.zeros((x.shape[0], x.shape[0]))
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            norm[i, j] = z[i] * z[j]
    out = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            out[i, j] = dot[i, j] / norm[i, j]
    return out
