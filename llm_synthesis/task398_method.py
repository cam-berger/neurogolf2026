import numpy as np

def method(grid):
    arr = np.asarray(grid).flatten()
    nz = int(np.count_nonzero(arr))
    N = 5 * nz
    out = np.zeros((N, N), dtype=arr.dtype)
    L = len(arr)
    for r in range(N):
        start = N - 1 - r
        for j in range(L):
            c = start + j
            if 0 <= c < N:
                out[r, c] = arr[j]
    return out
