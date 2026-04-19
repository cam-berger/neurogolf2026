def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    # Find the template row: the fully populated (no zeros) row
    template_row = None
    for r in range(H):
        if np.all(g[r] != 0) and len(np.unique(g[r])) >= 2:
            template_row = r
            break
    if template_row is None:
        return g
    tmpl = g[template_row]
    for r in range(H):
        if r == template_row:
            continue
        if np.all(g[r] == 0):
            continue
        # Build mapping from template colors to row colors using nonzero cells
        mapping = {}
        for c in range(W):
            if g[r, c] != 0:
                mapping[tmpl[c]] = g[r, c]
        # Fill entire row based on template pattern
        for c in range(W):
            if tmpl[c] in mapping:
                g[r, c] = mapping[tmpl[c]]
    return g
