def method(grid):
    import numpy as np
    from collections import Counter
    
    # Count cells per color and find leftmost column per color
    colors = {}
    leftmost = {}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = int(grid[r, c])
            if v != 0:
                colors[v] = colors.get(v, 0) + 1
                if v not in leftmost or c < leftmost[v]:
                    leftmost[v] = c
    
    max_count = max(colors.values())
    winners = [col for col, cnt in colors.items() if cnt == max_count]
    winners.sort(key=lambda x: leftmost[x])
    
    out = np.array([winners] * max_count, dtype=grid.dtype)
    return out
