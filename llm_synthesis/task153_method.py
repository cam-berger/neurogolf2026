import numpy as np

def method(grid):
    grid = np.asarray(grid)
    colors = [c for c in np.unique(grid) if c != 0]
    
    shapes = []
    for color in colors:
        mask = (grid == color)
        rs, cs = np.where(mask)
        minr, maxr = rs.min(), rs.max()
        minc, maxc = cs.min(), cs.max()
        sh = np.zeros((maxr - minr + 1, maxc - minc + 1), dtype=int)
        for r, c in zip(rs, cs):
            sh[r - minr, c - minc] = color
        shapes.append(sh)
    
    if len(shapes) != 2:
        return np.zeros((3, 3), dtype=int)
    
    A, B = shapes[0], shapes[1]
    hA, wA = A.shape
    hB, wB = B.shape
    
    for rA in range(4 - hA):
        for cA in range(4 - wA):
            for rB in range(4 - hB):
                for cB in range(4 - wB):
                    out = np.zeros((3, 3), dtype=int)
                    conflict = False
                    for i in range(hA):
                        for j in range(wA):
                            if A[i, j] != 0:
                                out[rA + i, cA + j] = A[i, j]
                    for i in range(hB):
                        for j in range(wB):
                            if B[i, j] != 0:
                                if out[rB + i, cB + j] != 0:
                                    conflict = True
                                    break
                                out[rB + i, cB + j] = B[i, j]
                        if conflict:
                            break
                    if not conflict and np.all(out != 0):
                        return out
    
    return np.zeros((3, 3), dtype=int)
