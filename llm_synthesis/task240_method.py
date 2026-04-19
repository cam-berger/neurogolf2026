import numpy as np

def method(grid):
    g = np.array(grid)
    n = g.shape[0]
    out = np.zeros_like(g)
    
    # Step 1: Apply 4-fold symmetry (mirror across center row and center column)
    for r in range(n):
        for c in range(n):
            v = g[r, c]
            if v != 0:
                for rr in (r, n - 1 - r):
                    for cc in (c, n - 1 - c):
                        out[rr, cc] = v
    
    # Step 2: For each concentric rectangle layer, if any edge cell is non-zero,
    # fill all four edges with that value.
    # Points live on odd rows/cols. Layer L has corners at (2L+1, 2L+1), (2L+1, n-2-2L), etc.
    max_layer = (n - 1) // 2  # number of odd positions per row
    for L in range(max_layer // 2 + 1):
        a = 2 * L + 1
        b = n - 1 - a
        if a >= b:
            continue
        
        # Find any edge value (non-corner position on the rectangle's perimeter)
        edge_val = 0
        # Top and bottom edges (between corners)
        for c in range(a + 2, b, 2):
            if out[a, c] != 0 and edge_val == 0:
                edge_val = out[a, c]
            if out[b, c] != 0 and edge_val == 0:
                edge_val = out[b, c]
        # Left and right edges
        for r in range(a + 2, b, 2):
            if out[r, a] != 0 and edge_val == 0:
                edge_val = out[r, a]
            if out[r, b] != 0 and edge_val == 0:
                edge_val = out[r, b]
        
        # Fill all four edges with edge_val
        if edge_val != 0:
            for c in range(a + 2, b, 2):
                out[a, c] = edge_val
                out[b, c] = edge_val
            for r in range(a + 2, b, 2):
                out[r, a] = edge_val
                out[r, b] = edge_val
    
    return out
