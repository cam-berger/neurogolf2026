import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape

    def find_axis(dim):
        L = H if dim == 0 else W
        M = W if dim == 0 else H
        best_k, best_score = None, -1
        for k in range(L - 1, 2 * L):
            score = 0
            mismatch = 0
            for i in range(L):
                j = k - i
                if 0 <= j < L and j != i:
                    for m in range(M):
                        if dim == 0:
                            a, b = g[i, m], g[j, m]
                        else:
                            a, b = g[m, i], g[m, j]
                        if a != 9 and b != 9:
                            if a == b:
                                score += 1
                            else:
                                mismatch += 1
            if mismatch == 0 and score > best_score:
                best_score = score
                best_k = k
        return best_k

    kr = find_axis(0)
    kc = find_axis(1)

    changed = True
    while changed:
        changed = False
        for r in range(H):
            for c in range(W):
                if g[r, c] == 9:
                    cands = []
                    if kr is not None:
                        rr = kr - r
                        if 0 <= rr < H:
                            cands.append((rr, c))
                    if kc is not None:
                        cc = kc - c
                        if 0 <= cc < W:
                            cands.append((r, cc))
                    if kr is not None and kc is not None:
                        rr = kr - r; cc = kc - c
                        if 0 <= rr < H and 0 <= cc < W:
                            cands.append((rr, cc))
                    for rr, cc in cands:
                        if g[rr, cc] != 9:
                            g[r, c] = g[rr, cc]
                            changed = True
                            break
    return g
