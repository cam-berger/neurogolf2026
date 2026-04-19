import numpy as np

def method(grid):
    grid = np.asarray(grid)
    H, W = grid.shape
    out = grid.copy()
    
    # Find separator color (a color that fills at least one entire row)
    sep_color = None
    for r in range(H):
        if len(set(grid[r].tolist())) == 1:
            sep_color = int(grid[r, 0])
            break
    if sep_color is None:
        return out
    
    # Identify separator rows and cols
    sep_rows = [r for r in range(H) if np.all(grid[r] == sep_color)]
    sep_cols = [c for c in range(W) if np.all(grid[:, c] == sep_color)]
    
    # Compute cell row/col ranges
    def make_ranges(seps, N):
        ranges = []
        prev = 0
        for s in seps:
            if s > prev:
                ranges.append((prev, s))
            prev = s + 1
        if prev < N:
            ranges.append((prev, N))
        return ranges
    
    cell_row_ranges = make_ranges(sep_rows, H)
    cell_col_ranges = make_ranges(sep_cols, W)
    nR, nC = len(cell_row_ranges), len(cell_col_ranges)
    
    # Build cell grid
    cell_grid = np.zeros((nR, nC), dtype=int)
    for i, (r0, r1) in enumerate(cell_row_ranges):
        for j, (c0, c1) in enumerate(cell_col_ranges):
            cell_grid[i, j] = grid[r0, c0]
    
    # Connected components (8-connected) of non-zero cells
    visited = np.zeros((nR, nC), dtype=bool)
    components = []
    for i in range(nR):
        for j in range(nC):
            if cell_grid[i, j] != 0 and not visited[i, j]:
                comp = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < nR and 0 <= nc < nC and not visited[nr, nc] and cell_grid[nr, nc] != 0:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                components.append(comp)
    
    if not components:
        return out
    
    # Template = largest component; seeds = single-cell components
    template_comp = max(components, key=len)
    seeds = [c[0] for c in components if len(c) == 1]
    
    if not seeds or len(template_comp) <= 1:
        return out
    
    seed_color = cell_grid[seeds[0]]
    
    # Find template center (cell with seed color)
    center = None
    for (r, c) in template_comp:
        if cell_grid[r, c] == seed_color:
            center = (r, c)
            break
    if center is None:
        return out
    
    offsets = [(r - center[0], c - center[1], int(cell_grid[r, c])) for (r, c) in template_comp]
    
    # Stamp template at each seed
    for (sr, sc) in seeds:
        for (dr, dc, val) in offsets:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < nR and 0 <= nc < nC:
                r0, r1 = cell_row_ranges[nr]
                c0, c1 = cell_col_ranges[nc]
                out[r0:r1, c0:c1] = val
    
    return out
