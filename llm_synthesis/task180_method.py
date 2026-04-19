import numpy as np

def method(grid):
    q4 = grid[0:4, 0:4]
    q5 = grid[0:4, 4:8]
    q6 = grid[4:8, 0:4]
    q9 = grid[4:8, 4:8]
    
    out = np.zeros((4, 4), dtype=grid.dtype)
    for i in range(4):
        for j in range(4):
            vals = [q4[i, j], q5[i, j], q6[i, j], q9[i, j]]
            nz = [v for v in vals if v != 0]
            if len(nz) == 0:
                out[i, j] = 0
            elif len(nz) == 1:
                out[i, j] = nz[0]
            else:
                if 5 in nz:
                    out[i, j] = 5
                elif 6 in nz:
                    out[i, j] = 6
                elif 9 in nz:
                    out[i, j] = 9
                else:
                    out[i, j] = 4
    return out
