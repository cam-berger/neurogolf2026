import numpy as np

def method(grid):
    h, w = grid.shape
    for r in range(h-2):
        for c in range(w-2):
            block = grid[r:r+3, c:c+3]
            border = [block[0,0], block[0,1], block[0,2],
                      block[1,0], block[1,2],
                      block[2,0], block[2,1], block[2,2]]
            center = block[1,1]
            if len(set(border)) == 1 and border[0] != 0 and center != 0 and center != border[0]:
                return np.array([[center]])
    return np.array([[0]])
