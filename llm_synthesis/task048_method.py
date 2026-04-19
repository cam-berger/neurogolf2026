def method(grid):
    import numpy as np
    g = np.asarray(grid)
    h, w = g.shape
    comp_id = -np.ones((h, w), dtype=int)
    comp = 0
    for i in range(h):
        for j in range(w):
            if g[i, j] in (2, 8) and comp_id[i, j] == -1:
                comp += 1
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < h and 0 <= x < w and g[y, x] in (2, 8) and comp_id[y, x] == -1:
                        comp_id[y, x] = comp
                        stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
    twos_comps = set()
    for i in range(h):
        for j in range(w):
            if g[i, j] == 2:
                twos_comps.add(int(comp_id[i, j]))
    if len(twos_comps) == 1:
        return np.array([[8]])
    else:
        return np.array([[0]])
