def method(grid):
    import numpy as np
    grid = np.array(grid)
    H, W = grid.shape
    result = grid.copy()

    visited = np.zeros_like(grid, dtype=bool)
    shapes = []
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 4 and not visited[i, j]:
                stack = [(i, j)]
                fours = []
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= H or c < 0 or c >= W:
                        continue
                    if visited[r, c] or grid[r, c] != 4:
                        continue
                    visited[r, c] = True
                    fours.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((r + dr, c + dc))
                decos = {}
                for r, c in fours:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < H and 0 <= cc < W:
                                v = grid[rr, cc]
                                if v != 0 and v != 4:
                                    decos[(rr, cc)] = int(v)
                shapes.append((fours, decos))

    template = None
    best = -1
    for s in shapes:
        fours, decos = s
        colors = set(decos.values())
        score = len(decos) + 100 * (len(colors) - 1)
        if len(colors) > 1 and score > best:
            best = score
            template = s

    if template is None:
        return result

    t_fours, t_decos = template
    t_pattern = {}
    for r, c in t_fours:
        t_pattern[(r, c)] = 4
    for (r, c), v in t_decos.items():
        t_pattern[(r, c)] = v

    def transform(pattern, tidx):
        new = {}
        for (r, c), v in pattern.items():
            if tidx == 0: nr, nc = r, c
            elif tidx == 1: nr, nc = c, -r
            elif tidx == 2: nr, nc = -r, -c
            elif tidx == 3: nr, nc = -c, r
            elif tidx == 4: nr, nc = r, -c
            elif tidx == 5: nr, nc = -r, c
            elif tidx == 6: nr, nc = c, r
            elif tidx == 7: nr, nc = -c, -r
            new[(nr, nc)] = v
        return new

    for s in shapes:
        if s is template:
            continue
        s_fours, s_decos = s
        s_fours_set = set(s_fours)
        s_pat = {}
        for p in s_fours:
            s_pat[p] = 4
        for p, v in s_decos.items():
            s_pat[p] = v

        applied = False
        for tidx in range(8):
            if applied:
                break
            tp = transform(t_pattern, tidx)
            tp_fours = [p for p, v in tp.items() if v == 4]
            if not tp_fours:
                continue
            t0 = tp_fours[0]
            for s0 in s_fours:
                dr = s0[0] - t0[0]
                dc = s0[1] - t0[1]
                translated_fours = set((r + dr, c + dc) for r, c in tp_fours)
                if translated_fours != s_fours_set:
                    continue
                ok = True
                for (r, c), v in s_decos.items():
                    tr, tc = r - dr, c - dc
                    if tp.get((tr, tc)) != v:
                        ok = False
                        break
                if not ok:
                    continue
                for (r, c), v in tp.items():
                    if v == 4:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if result[nr, nc] == 0:
                            result[nr, nc] = v
                applied = True
                break

    return result
