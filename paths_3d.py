"""
paths_3d.py — Diagonal Hamiltonian and Golden-Ratio path constructors for 3D
              toroidal grids.
"""

import sys
sys.setrecursionlimit(200_000)

from utils import PHI, get_nb3, is_adjacent_torus


# ── Diagonal Hamiltonian (C0) ─────────────────────────────────────────────────

def _getD(i, j, k, m):
    """
    Direction selector based on fiber s = (i+j+k) % m.
    Returns a 3-character string whose first character is the axis index
    of the next step (Knuth arc-decomposition for odd m).
    """
    s = (i + j + k) % m
    if s == 0:
        return "012" if j == m - 1 else "210"
    elif s == m - 1:
        return "120" if i > 0 else "210"
    else:
        return "201" if i == m - 1 else "102"


def diagonal3d(m):
    """
    Build a Hamiltonian path on the m×m×m toroidal grid using the
    Knuth arc-decomposition diagonal sweep.

    The path visits fiber planes s = (i+j+k) % m in strict order 0,1,…,m−1,
    cycling until all m³ vertices are covered. Vertices with high coordinate
    sums are visited last, making their tail the natural steal target for scouts.

    Args:
        m: Grid side length (works best for odd m).

    Returns:
        List of (i,j,k) tuples, length m³, all distinct.
    """
    i, j, k = 0, 0, 0
    path = [(0, 0, 0)]
    for _ in range(m**3 - 1):
        d  = _getD(i, j, k, m)
        nb = get_nb3(i, j, k, m)
        i, j, k = nb[int(d[0])]
        path.append((i, j, k))
    assert len(set(path)) == m**3, "diagonal3d: path is not Hamiltonian"
    return path


# ── Golden-Ratio Path (scouts) ────────────────────────────────────────────────

def _build_golden3d(ranges, m):
    """
    Recursive φ-split Hamiltonian path builder for a 3D sub-box.

    Args:
        ranges: (ilo, ihi, jlo, jhi, klo, khi)
        m:      Grid side length (for toroidal adjacency check).

    Returns:
        List of (i,j,k) tuples covering the sub-box exactly once.
    """
    ilo, ihi, jlo, jhi, klo, khi = ranges
    dims = [ihi-ilo+1, jhi-jlo+1, khi-klo+1]

    # Base cases — 1D lines
    if dims[1] == 1 and dims[2] == 1:
        return [(i, jlo, klo) for i in range(ilo, ihi+1)]
    if dims[0] == 1 and dims[2] == 1:
        return [(ilo, j, klo) for j in range(jlo, jhi+1)]
    if dims[0] == 1 and dims[1] == 1:
        return [(ilo, jlo, k) for k in range(klo, khi+1)]

    # Split along longest axis at golden ratio
    axis = dims.index(max(dims))
    if axis == 0:
        sv = ilo + max(1, round(dims[0] / PHI)) - 1
        lr = (ilo, sv,   jlo, jhi, klo, khi)
        rr = (sv+1, ihi, jlo, jhi, klo, khi)
    elif axis == 1:
        sv = jlo + max(1, round(dims[1] / PHI)) - 1
        lr = (ilo, ihi, jlo, sv,   klo, khi)
        rr = (ilo, ihi, sv+1, jhi, klo, khi)
    else:
        sv = klo + max(1, round(dims[2] / PHI)) - 1
        lr = (ilo, ihi, jlo, jhi, klo, sv)
        rr = (ilo, ihi, jlo, jhi, sv+1, khi)

    lp = _build_golden3d(lr, m)
    rp = _build_golden3d(rr, m)

    # Try all 4 endpoint orientations to find an adjacent join
    for l, r in [(lp, rp), (lp, rp[::-1]),
                 (lp[::-1], rp), (lp[::-1], rp[::-1])]:
        if is_adjacent_torus(l[-1], r[0], m):
            return l + r

    return lp + rp   # fallback: non-adjacent stitch (rare)


def make_golden3d(m, rot=0):
    """
    Build a golden-ratio Hamiltonian path on the full m×m×m grid.

    Args:
        m:   Grid side length.
        rot: Axis rotation (0, 1, or 2) to produce structurally distinct
             variants for different scouts.

    Returns:
        List of (i,j,k) tuples, length m³.
    """
    path = _build_golden3d((0, m-1, 0, m-1, 0, m-1), m)
    if rot > 0:
        path = [(v[rot % 3], v[(rot+1) % 3], v[(rot+2) % 3]) for v in path]
    return path
