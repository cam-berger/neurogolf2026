def method(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    layers = min(h, w) // 2
    # Extract ring colors from outermost to innermost
    ring_colors = [int(g[i, i]) for i in range(layers)]
    # Get unique colors preserving order
    seen = []
    for c in ring_colors:
        if c not in seen:
            seen.append(c)
    n = len(seen)
    # Build mapping: each color maps to its "outer neighbor" with wrap
    # (i.e., right cyclic rotation: seen[i] -> seen[(i-1) mod n])
    mapping = {seen[i]: seen[(i - 1) % n] for i in range(n)}
    out = g.copy()
    for c, nc in mapping.items():
        out[g == c] = nc
    return out
