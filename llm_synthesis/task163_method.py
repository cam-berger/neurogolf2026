def method(grid):
    import numpy as np
    g = np.asarray(grid)
    out = np.zeros_like(g)
    # Restore separator lines
    out[g == 5] = 5
    
    # Find the 4
    pos = np.argwhere(g == 4)
    if len(pos) == 0:
        return out
    r, c = pos[0]
    
    # Block coords (using step 4: 3 cells + 1 separator)
    br, bc = r // 4, c // 4
    ir, ic = r % 4, c % 4
    
    # Extract source block
    sr0, sc0 = br * 4, bc * 4
    src = g[sr0:sr0+3, sc0:sc0+3]
    
    # Place at target block (ir, ic)
    tr0, tc0 = ir * 4, ic * 4
    out[tr0:tr0+3, tc0:tc0+3] = src
    
    return out
