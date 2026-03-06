"""
paths_4d.py — Diagonal Hamiltonian and Golden-Ratio path constructors for 4D
              toroidal grids.
"""

import sys
sys.setrecursionlimit(200_000)

from utils import PHI, get_nb4, is_adjacent_torus


# ── Diagonal Hamiltonian (C0) ─────────────────────────────────────────────────

def diagonal4d(m):
    """
    Build a Hamiltonian path on the m×m×m×m toroidal grid using a
    Warnsdorff-guided diagonal sweep.

    The diagonal rank dr(i,j,k,l) = s·m³ + i·m² + j·m + k  where
    s = (i+j+k+l) % m enforces strict fiber-plane ordering while
    Warnsdorff degree tiebreaking (prefer vertices with fewest unvisited
    neighbours) prevents dead ends.

    Structural property: vertices with high coordinate sums are visited last,
    making the tail of this path the natural steal target for scouts.

    Args:
        m: Grid side length.

    Returns:
        List of (i,j,k,l) tuples, length m⁴, all distinct.
    """
    total = m**4

    # Warnsdorff degree: number of unvisited neighbours
    wdeg = {
        (i, j, k, l): 4
        for i in range(m) for j in range(m)
        for k in range(m) for l in range(m)
    }

    def dr(i, j, k, l):
        """Diagonal rank — primary sort key."""
        return ((i+j+k+l) % m) * m**3 + i*m**2 + j*m + k

    vis = set()
    cur = (0, 0, 0, 0)
    path = [cur]
    vis.add(cur)
    for nb in get_nb4(*cur, m):
        wdeg[nb] -= 1
    cr = dr(*cur)

    for _ in range(total - 1):
        cands = [v for v in get_nb4(*cur, m) if v not in vis]
        if not cands:
            break
        # Sort by (fiber advance, Warnsdorff degree)
        cands.sort(key=lambda v: ((dr(*v) - cr) % total, wdeg[v]))
        cur = cands[0]
        path.append(cur)
        vis.add(cur)
        cr = dr(*cur)
        for nb in get_nb4(*cur, m):
            wdeg[nb] -= 1

    assert len(set(path)) == total, "diagonal4d: path is not Hamiltonian"
    return path


# ── Golden-Ratio Path (scouts) ────────────────────────────────────────────────

def _build_golden4d(ranges, m):
    """
    Recursive φ-split Hamiltonian path builder for a 4D sub-box.

    Args:
        ranges: List of (lo, hi) pairs for each axis.
        m:      Grid side length (for toroidal adjacency check).

    Returns:
        List of (i,j,k,l) tuples covering the sub-box exactly once.
    """
    lo = [r[0] for r in ranges]
    hi = [r[1] for r in ranges]
    dims = [hi[a] - lo[a] + 1 for a in range(4)]

    # Base case — 1D line along the one non-unit axis
    if sum(1 for d in dims if d > 1) == 1:
        axis = next(a for a in range(4) if dims[a] > 1)
        v = list(lo)
        out = []
        for x in range(lo[axis], hi[axis]+1):
            v2 = list(v)
            v2[axis] = x
            out.append(tuple(v2))
        return out

    # Split along longest axis at golden ratio
    axis = dims.index(max(dims))
    size = dims[axis]
    split = max(1, round(size / PHI))
    sv = lo[axis] + split - 1

    left_ranges  = [(lo[a], sv       if a == axis else hi[a]) for a in range(4)]
    right_ranges = [(sv+1  if a == axis else lo[a], hi[a])    for a in range(4)]

    lp = _build_golden4d(left_ranges,  m)
    rp = _build_golden4d(right_ranges, m)

    # Try all 4 endpoint orientations to find an adjacent join
    for l, r in [(lp, rp), (lp, rp[::-1]),
                 (lp[::-1], rp), (lp[::-1], rp[::-1])]:
        if is_adjacent_torus(l[-1], r[0], m):
            return l + r

    return lp + rp   # fallback


def make_golden4d(m, rot=0):
    """
    Build a golden-ratio Hamiltonian path on the full m×m×m×m grid.

    Args:
        m:   Grid side length.
        rot: Axis rotation (0–3) to produce structurally distinct variants.

    Returns:
        List of (i,j,k,l) tuples, length m⁴.
    """
    path = _build_golden4d([(0, m-1)] * 4, m)
    if rot > 0:
        path = [tuple(v[(a + rot) % 4] for a in range(4)) for v in path]
    return path
