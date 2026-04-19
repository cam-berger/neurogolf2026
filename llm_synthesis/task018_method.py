def method(grid):
    import numpy as np
    H, W = grid.shape
    out = np.zeros_like(grid)

    nz = grid[grid != 0]
    if len(nz) == 0:
        return out
    colors, counts = np.unique(nz, return_counts=True)
    filler = int(colors[np.argmax(counts)])

    # 8-connected components of non-zero cells
    visited = np.zeros((H, W), dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0 and not visited[r, c]:
                stack = [(r, c)]
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= H or cc < 0 or cc >= W:
                        continue
                    if visited[rr, cc] or grid[rr, cc] == 0:
                        continue
                    visited[rr, cc] = True
                    comp.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            stack.append((rr + dr, cc + dc))
                components.append(comp)

    templates = []
    template_cells_set = set()
    for comp in components:
        cols = set(int(grid[r, c]) for r, c in comp)
        if filler in cols:
            templates.append([(r, c, int(grid[r, c])) for r, c in comp])
            for r, c in comp:
                template_cells_set.add((r, c))

    # Remaining non-zero cells are sparse key cells
    non_template = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0 and (r, c) not in template_cells_set:
                non_template.append((r, c, int(grid[r, c])))
    cell_map = {(r, c): i for i, (r, c, v) in enumerate(non_template)}
    used = [False] * len(non_template)

    def T(r, c, t):
        if t == 0: return (r, c)
        if t == 1: return (-c, r)
        if t == 2: return (-r, -c)
        if t == 3: return (c, -r)
        if t == 4: return (r, -c)
        if t == 5: return (-c, -r)
        if t == 6: return (-r, c)
        if t == 7: return (c, r)

    for template in templates:
        t_keys = [(r, c, v) for r, c, v in template if v != filler]
        if not t_keys:
            continue
        ref_r, ref_c, ref_v = t_keys[0]

        changed = True
        while changed:
            changed = False
            for i, (ar, ac, av) in enumerate(non_template):
                if used[i] or av != ref_v:
                    continue
                matched = False
                for t in range(8):
                    tr, tc = T(ref_r, ref_c, t)
                    off_r = ar - tr
                    off_c = ac - tc
                    idxs = [i]
                    ok = True
                    for (r, c, v) in t_keys[1:]:
                        tr2, tc2 = T(r, c, t)
                        nr, nc = tr2 + off_r, tc2 + off_c
                        if (nr, nc) not in cell_map:
                            ok = False; break
                        j = cell_map[(nr, nc)]
                        if used[j] or non_template[j][2] != v:
                            ok = False; break
                        idxs.append(j)
                    if ok:
                        for (r, c, v) in template:
                            tr2, tc2 = T(r, c, t)
                            nr, nc = tr2 + off_r, tc2 + off_c
                            if 0 <= nr < H and 0 <= nc < W:
                                out[nr, nc] = v
                        for j in idxs:
                            used[j] = True
                        matched = True
                        changed = True
                        break
                if matched:
                    break

    return out
