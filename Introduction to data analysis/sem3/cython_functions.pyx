import numpy as np
cimport numpy as np

cpdef matrix_multiply(np.ndarray[np.float64_t, ndim = 2] x,
                      np.ndarray[np.float64_t, ndim = 2] y):
    cdef np.ndarray[np.float64_t, ndim = 2] out;
    cdef int sizer = x.shape[0];
    cdef int sizec = x.shape[1];
    out = np.zeros((sizer, y.shape[1]), dtype=np.float64)
    for i in range(sizer):
        for j in range(y.shape[1]):
            for k in range(sizec):
                out[i, j] += x[i, k] * y[k, j]
    return out


cpdef matrix_rowmean(np.ndarray[np.float64_t, ndim = 2] x,
                     weights = np.empty(0)):
    cdef np.ndarray[np.float64_t, ndim = 2] y;
    cdef int sizer = x.shape[0];
    cdef int sizec = x.shape[1];
    y = np.zeros((sizer, sizec), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] out;
    out = np.zeros((sizer), dtype=np.float64)
    cdef np.float64_t s = 0.0;
    cdef np.float64_t w = 0.0;
    if weights.shape[0] == 0:
        for i in range(sizer):
            s = 0.0
            for j in range(sizec):
                s += x[i, j]
            out[i] = s / sizec
    else:
        for i in range(sizec):
            w += weights[i]
        for i in range(sizer):
            for j in range(sizec):
                y[i, j] = x[i, j] * weights[j]
        for i in range(sizer):
            s = 0.0
            for j in range(sizec):
                s += y[i, j]
            out[i] = s / w
    return out



cpdef cosine_similarity(np.ndarray[np.float64_t, ndim = 2] x,
                        np.int_t top_n = 10,
                        with_mean = True,
                        with_std = True):
    cdef np.float64_t mean = 0.0;
    cdef np.float64_t std = 0.0;
    cdef int sizer = x.shape[0];
    cdef int sizec = x.shape[1];
    if with_mean:
        for i in range(sizer):
            mean = 0.0
            for j in range(sizec):
                mean = mean + x[i, j]
            mean = mean / sizec
            for j in range(sizec):
                x[i, j] = x[i, j] - mean
    if with_std:
        for i in range(sizer):
            std = 0.0
            mean = 0.0
            for j in range(sizec):
                mean = mean + x[i, j]
            mean = mean / sizec
            for j in range(sizec):
                std = std + (x[i, j] - mean) ** 2
            std = (std / sizec) ** 0.5
            for j in range(sizec):
                x[i, j] = x[i, j] / std

    for i in range(sizer):
        y = sorted(x[i])[:-top_n]
        for j in range(len(y)):
            for k in range(sizec):
                if x[i, k] == y[j]:
                    x[i, k] = 0
                    break
    cdef np.ndarray[np.float64_t, ndim = 2] dot;
    dot = np.zeros((sizer, sizer), dtype=np.float64)
    for i in range(sizer):
        for j in range(sizer):
            for k in range(x[i].shape[0]):
                dot[i, j] = dot[i, j] + x[i, k] * x[j, k]
    cdef np.ndarray[np.float64_t, ndim = 1] z;
    z = np.zeros((sizer), dtype=np.float64)
    for i in range(sizer):
        z[i] = dot[i, i] ** 0.5
    cdef np.ndarray[np.float64_t, ndim = 2] norm;
    norm = np.zeros((sizer, sizer), dtype=np.float64)
    for i in range(sizer):
        for j in range(sizer):
            norm[i, j] = z[i] * z[j]
    cdef np.ndarray[np.float64_t, ndim = 2] out;
    out = np.zeros((sizer, sizer), dtype=np.float64)
    for i in range(sizer):
        for j in range(sizer):
            out[i, j] = dot[i, j] / norm[i, j]
    return out
