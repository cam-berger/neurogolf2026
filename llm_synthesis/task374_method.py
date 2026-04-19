import numpy as np

def method(grid):
    from scipy.ndimage import label
    g = grid.copy()
    mask = (g == 5)
    # find connected components (using 4-connectivity)
    lbl, n = label(mask)
    sizes = []
    for i in range(1, n+1):
        sizes.append((i, np.sum(lbl == i)))
    # sort by size
    sizes.sort(key=lambda x: -x[1])
    # largest -> 1, middle -> 4, smallest -> 2
    color_map = {}
    colors = [1, 4, 2]
    for idx, (comp_id, _) in enumerate(sizes):
        color_map[comp_id] = colors[idx]
    out = g.copy()
    for comp_id, c in color_map.items():
        out[lbl == comp_id] = c
    return out
