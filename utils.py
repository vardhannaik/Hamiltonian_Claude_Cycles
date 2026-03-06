"""
utils.py — shared helpers for cooperative cycle coverage on toroidal grids.
"""

import math

PHI = (1 + math.sqrt(5)) / 2


# ── Adjacency ─────────────────────────────────────────────────────────────────

def get_nb3(i, j, k, m):
    """3D toroidal neighbours of (i,j,k)."""
    return [
        ((i+1)%m, j, k),
        (i, (j+1)%m, k),
        (i, j, (k+1)%m),
    ]


def get_nb4(i, j, k, l, m):
    """4D toroidal neighbours of (i,j,k,l)."""
    return [
        ((i+1)%m, j, k, l),
        (i, (j+1)%m, k, l),
        (i, j, (k+1)%m, l),
        (i, j, k, (l+1)%m),
    ]


def is_adjacent_torus(v1, v2, m):
    """True if v1 and v2 are adjacent on the m^n torus."""
    return sum(
        min(abs(v1[a] - v2[a]), m - abs(v1[a] - v2[a]))
        for a in range(len(v1))
    ) == 1


# ── Lateness helper ───────────────────────────────────────────────────────────

def lateness(smap, v, cur_step, total):
    """
    Steps until leader visits v from cur_step in its cyclic schedule.
    Returns 0 if v is not in the leader's future (already passed or absent).
    """
    vi = smap.get(v, -1)
    if vi < 0:
        return 0
    l = (vi - cur_step) % total
    return l if l > 0 else 0
