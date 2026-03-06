"""
simulate_3d.py — V3 (pure existence) and V6 (existence + late-steal)
                 cooperative cycle coverage for 3D toroidal grids.

Usage:
    python simulate_3d.py --m 5 --late-frac 0.095
    python simulate_3d.py --m 5 --sweep          # sweep LATE_FRAC 0.01–0.50
"""

import argparse
import sys
sys.setrecursionlimit(200_000)

from utils import get_nb3, lateness
from paths_3d import diagonal3d, make_golden3d


# ── V3: Pure Existence Scoring ────────────────────────────────────────────────

def simulate_3d_v3(m, s0, s1, s2):
    """
    3-agent cooperative coverage — V3 pure-existence scoring.

    Tier hierarchy:
      C1:  tier 2 = C0 will never visit v  (gap)
           tier 1 = C0 will visit v
      C2:  tier 3 = neither C0 nor C1 visits  (pure gap)
           tier 2 = C0 skips, C1 will come
           tier 1 = C0 will visit

    Args:
        m:          Grid side length.
        s0, s1, s2: Pre-planned Hamiltonian sequences for C0, C1, C2.

    Returns:
        (history, coverage)
        history:  list of (pos_C0, pos_C1, pos_C2) at each step
        coverage: list of int — cumulative nodes covered at each step
    """
    total = m**3
    c0s = {v: t for t, v in enumerate(s0)}
    c1s = {v: t for t, v in enumerate(s1)}

    pos = [s0[0], s1[0], s2[0]]
    joint = set(pos)
    history = [tuple(pos)]
    coverage = [len(joint)]
    idx = [0] * 3

    for t in range(total):
        new_pos = [None] * 3
        claimed = set()

        # C0 — strict leader
        n0 = s0[(t + 1) % total]
        new_pos[0] = n0
        claimed.add(n0)
        c0_step = t + 1

        idx[1] = (idx[1] + 1) % total
        idx[2] = (idx[2] + 1) % total

        # C1
        nb1 = get_nb3(*pos[1], m)
        best = -1; bc = 0
        for tc, d in enumerate(nb1):
            if d in joint or d in claimed:
                s = -1
            else:
                s = 2 if lateness(c0s, d, c0_step, total) == 0 else 1
            if s > best: best = s; bc = tc
        c1n = nb1[bc]; new_pos[1] = c1n; claimed.add(c1n)
        c1_cur = c1s.get(c1n, idx[1])

        # C2
        nb2 = get_nb3(*pos[2], m)
        best = -1; bc = 0
        for tc, d in enumerate(nb2):
            if d in joint or d in claimed:
                s = -1
            else:
                l0 = lateness(c0s, d, c0_step, total)
                l1 = lateness(c1s, d, c1_cur,  total)
                if   l0 == 0 and l1 == 0: s = 3
                elif l0 == 0:             s = 2
                else:                     s = 1
            if s > best: best = s; bc = tc
        new_pos[2] = nb2[bc]; claimed.add(nb2[bc])

        for c in range(3):
            pos[c] = new_pos[c]
            joint.add(new_pos[c])
        history.append(tuple(pos))
        coverage.append(len(joint))
        if len(joint) == total:
            break

    return history, coverage


# ── V6: Existence Tiers + Late-Stealing ──────────────────────────────────────

def simulate_3d_v6(m, s0, s1, s2, late_frac=0.095):
    """
    3-agent cooperative coverage — V6 existence tiers + late-steal sub-tier.

    Builds on V3: the pure-gap tier remains the highest priority.  Within the
    "leader will visit" region, vertices in the last `late_frac` of a leader's
    *remaining* schedule are classified as "late" and given a steal sub-tier
    above "C0-soon" vertices.

    Tier hierarchy:
      C1:  3 = gap(C0)
           2 = C0-late  (last late_frac of C0's remaining schedule)
           1 = C0-soon

      C2:  4 = pure gap (neither C0 nor C1)
           3 = C0-skip, C1-late
           2 = C0-skip C1-soon  OR  C0-late
           1 = C0-soon

    Args:
        m:          Grid side length.
        s0, s1, s2: Pre-planned Hamiltonian sequences.
        late_frac:  Fraction of remaining schedule that counts as "late".
                    Optimal: ~0.095 for 3D m=5.

    Returns:
        (history, coverage)
    """
    total = m**3
    c0s = {v: t for t, v in enumerate(s0)}
    c1s = {v: t for t, v in enumerate(s1)}

    pos = [s0[0], s1[0], s2[0]]
    joint = set(pos)
    history = [tuple(pos)]
    coverage = [len(joint)]
    idx = [0] * 3

    for t in range(total):
        new_pos = [None] * 3
        claimed = set()

        # C0 — strict leader
        n0 = s0[(t + 1) % total]
        new_pos[0] = n0
        claimed.add(n0)
        c0_step = t + 1
        c0_rem  = total - c0_step
        thr = int(c0_rem * (1 - late_frac))   # lateness threshold (dynamic)

        idx[1] = (idx[1] + 1) % total
        idx[2] = (idx[2] + 1) % total

        # ── C1 ────────────────────────────────────────────────────────────────
        nb1 = get_nb3(*pos[1], m)
        best = -1; bc = 0
        for tc, d in enumerate(nb1):
            if d in joint or d in claimed:
                s = -1
            else:
                l0 = lateness(c0s, d, c0_step, total)
                if   l0 == 0:     s = 3   # gap — C0 never comes
                elif l0 >= thr:   s = 2   # C0-late — steal the tail
                else:             s = 1   # C0-soon — let C0 handle it
            if s > best: best = s; bc = tc
        c1n = nb1[bc]; new_pos[1] = c1n; claimed.add(c1n)
        c1_cur = c1s.get(c1n, idx[1])

        # ── C2 ────────────────────────────────────────────────────────────────
        nb2 = get_nb3(*pos[2], m)
        best = -1; bc = 0
        for tc, d in enumerate(nb2):
            if d in joint or d in claimed:
                s = -1
            else:
                l0 = lateness(c0s, d, c0_step, total)
                l1 = lateness(c1s, d, c1_cur,  total)
                if   l0 == 0 and l1 == 0:     s = 4   # pure gap
                elif l0 == 0 and l1 >= thr:   s = 3   # C0-skip, C1-late
                elif l0 == 0:                 s = 2   # C0-skip, C1-soon
                elif l0 >= thr:               s = 2   # C0-late
                else:                         s = 1   # C0-soon
            if s > best: best = s; bc = tc
        new_pos[2] = nb2[bc]; claimed.add(nb2[bc])

        for c in range(3):
            pos[c] = new_pos[c]
            joint.add(new_pos[c])
        history.append(tuple(pos))
        coverage.append(len(joint))
        if len(joint) == total:
            break

    return history, coverage


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cover_step(coverage, total):
    return next((i for i, c in enumerate(coverage) if c == total), len(coverage)-1)


def main():
    parser = argparse.ArgumentParser(description="3D cooperative cycle coverage")
    parser.add_argument("--m",         type=int,   default=5,     help="Grid side length")
    parser.add_argument("--late-frac", type=float, default=0.095, help="Late-steal fraction (V6)")
    parser.add_argument("--sweep",     action="store_true",        help="Sweep late_frac 0.01–0.50")
    args = parser.parse_args()

    m     = args.m
    total = m**3

    print(f"Building sequences for m={m} (total={total})...")
    d0 = diagonal3d(m)
    g1 = make_golden3d(m, rot=0)
    g2 = make_golden3d(m, rot=1)

    # V3 baseline
    _, cov_v3 = simulate_3d_v3(m, d0, g1, g2)
    cs_v3 = _cover_step(cov_v3, total)
    print(f"\nV3 (pure existence): {cs_v3}/{total}  ({cs_v3/total*100:.2f}%)")

    if args.sweep:
        print("\nSweeping late_frac 0.01 – 0.50:")
        best_r = total; best_f = 0.0
        for num in range(1, 51):
            frac = num / 100
            _, cov = simulate_3d_v6(m, d0, g1, g2, late_frac=frac)
            cs = _cover_step(cov, total)
            marker = " ← best" if cs < best_r else ""
            if cs < best_r: best_r = cs; best_f = frac
            print(f"  frac={frac:.2f}: {cs}/{total} ({cs/total*100:.1f}%){marker}")
        print(f"\nBest: frac={best_f:.3f} → {best_r}/{total} ({best_r/total*100:.2f}%)")
        print(f"V6 saves {cs_v3 - best_r} steps over V3")
    else:
        _, cov_v6 = simulate_3d_v6(m, d0, g1, g2, late_frac=args.late_frac)
        cs_v6 = _cover_step(cov_v6, total)
        print(f"V6 (late_frac={args.late_frac}): {cs_v6}/{total}  ({cs_v6/total*100:.2f}%)")
        print(f"V6 saves {cs_v3 - cs_v6} steps over V3")


if __name__ == "__main__":
    main()
