def method(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    vals, counts = np.unique(g, return_counts=True)
    bg = vals[np.argmax(counts)]

    visited = np.zeros_like(g, dtype=bool)
    components = []
    for i in range(H):
        for j in range(W):
            if not visited[i, j] and g[i, j] != bg:
                stack = [(i, j)]
                cells = []
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= H or c < 0 or c >= W:
                        continue
                    if visited[r, c] or g[r, c] == bg:
                        continue
                    visited[r, c] = True
                    cells.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((r + dr, c + dc))
                components.append(cells)

    template = None
    markers = []
    for comp in components:
        colors = set(g[r, c] for r, c in comp)
        if len(colors) >= 2:
            template = comp
        else:
            markers.append(comp)

    if template is None:
        return g

    trs = [r for r, c in template]
    tcs = [c for r, c in template]
    tr0, tr1 = min(trs), max(trs)
    tc0, tc1 = min(tcs), max(tcs)
    th = tr1 - tr0 + 1
    tw = tc1 - tc0 + 1
    tpat = np.full((th, tw), bg, dtype=g.dtype)
    for r, c in template:
        tpat[r - tr0, c - tc0] = g[r, c]

    flat = tpat[tpat != bg]
    uvals, ucounts = np.unique(flat, return_counts=True)
    fill_color = uvals[np.argmax(ucounts)]
    anchor_colors = [int(v) for v in uvals if v != fill_color]
    if len(anchor_colors) < 2:
        return g
    anchor_pos = {}
    for ac in anchor_colors:
        positions = [(r, c) for r in range(th) for c in range(tw) if tpat[r, c] == ac]
        anchor_pos[ac] = positions[0]

    marker_info = []
    for m in markers:
        rs = [r for r, c in m]
        cs = [c for r, c in m]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        color = int(g[m[0]])
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        marker_info.append({'color': color, 'r0': r0, 'c0': c0, 'size': size})

    ac1, ac2 = anchor_colors[0], anchor_colors[1]
    used = [False] * len(marker_info)
    pairs = []
    for i in range(len(marker_info)):
        if used[i]:
            continue
        mi = marker_info[i]
        if mi['color'] not in anchor_colors:
            continue
        other_color = ac1 if mi['color'] == ac2 else ac2
        best = -1
        best_dist = float('inf')
        for j in range(len(marker_info)):
            if i == j or used[j]:
                continue
            mj = marker_info[j]
            if mj['color'] != other_color:
                continue
            if mj['size'] != mi['size']:
                continue
            d = abs(mi['r0'] - mj['r0']) + abs(mi['c0'] - mj['c0'])
            if d < best_dist:
                best_dist = d
                best = j
        if best >= 0:
            used[i] = used[best] = True
            pairs.append((i, best))

    out = g.copy()
    a1_pos = anchor_pos[ac1]
    a2_pos = anchor_pos[ac2]

    for i, j in pairs:
        mi = marker_info[i]
        mj = marker_info[j]
        if mi['color'] == ac1:
            m_a1, m_a2 = mi, mj
        else:
            m_a1, m_a2 = mj, mi
        scale = m_a1['size']
        mv = ((m_a2['r0'] - m_a1['r0']) // scale, (m_a2['c0'] - m_a1['c0']) // scale)

        chosen = None
        for flipV in [False, True]:
            for flipH in [False, True]:
                for transpose in [False, True]:
                    pat = tpat.copy()
                    a1r, a1c = a1_pos
                    a2r, a2c = a2_pos
                    ph, pw = th, tw
                    if flipV:
                        pat = pat[::-1]
                        a1r = ph - 1 - a1r
                        a2r = ph - 1 - a2r
                    if flipH:
                        pat = pat[:, ::-1]
                        a1c = pw - 1 - a1c
                        a2c = pw - 1 - a2c
                    if transpose:
                        pat = pat.T
                        a1r, a1c = a1c, a1r
                        a2r, a2c = a2c, a2r
                        ph, pw = pw, ph
                    new_tv = (a2r - a1r, a2c - a1c)
                    if new_tv == mv:
                        chosen = (pat, a1r, a1c)
                        break
                if chosen:
                    break
            if chosen:
                break

        if not chosen:
            continue
        pat, a1r, a1c = chosen
        ph, pw = pat.shape
        orig_r = m_a1['r0'] - a1r * scale
        orig_c = m_a1['c0'] - a1c * scale

        for pr in range(ph):
            for pc in range(pw):
                v = pat[pr, pc]
                if v == bg:
                    continue
                for dr in range(scale):
                    for dc in range(scale):
                        rr = orig_r + pr * scale + dr
                        cc = orig_c + pc * scale + dc
                        if 0 <= rr < H and 0 <= cc < W:
                            out[rr, cc] = v

    return out
