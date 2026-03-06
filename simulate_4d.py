"""
simulate_4d.py — V3 (pure existence) and V6 (existence + late-steal)
                 cooperative cycle coverage for 4D toroidal grids.

Usage:
    python simulate_4d.py --m 5 --late-frac 0.187
    python simulate_4d.py --m 5 --sweep          # sweep LATE_FRAC 0.01–0.50
"""

import argparse
import sys
sys.setrecursionlimit(200_000)

from utils import get_nb4, lateness
from paths_4d import diagonal4d, make_golden4d


# ── V3: Pure Existence Scoring ────────────────────────────────────────────────

def simulate_4d_v3(m, s0, s1, s2, s3):
    """
    4-agent cooperative coverage — V3 pure-existence scoring.

    Tier hierarchy:
      C1:  2 = gap(C0)  |  1 = C0 visits
      C2:  3 = gap(C0+C1)  |  2 = C0-skip  |  1 = C0 visits
      C3:  4 = gap(C0+C1+C2)  |  3 = C0+C1-skip  |  2 = C0-skip  |  1 = C0 visits

    Returns:
        (history, coverage)
    """
    total = m**4
    c0s = {v: t for t, v in enumerate(s0)}
    c1s = {v: t for t, v in enumerate(s1)}
    c2s = {v: t for t, v in enumerate(s2)}

    pos   = [s0[0], s1[0], s2[0], s3[0]]
    joint = set(pos)
    history  = [tuple(pos)]
    coverage = [len(joint)]
    idx = [0] * 4

    for t in range(total):
        new_pos = [None] * 4
        claimed = set()

        n0 = s0[(t + 1) % total]
        new_pos[0] = n0; claimed.add(n0)
        c0_step = t + 1
        idx[1] = (idx[1]+1) % total
        idx[2] = (idx[2]+1) % total
        idx[3] = (idx[3]+1) % total

        # C1
        nb1 = get_nb4(*pos[1], m); best=-1; bc=0
        for tc, d in enumerate(nb1):
            if d in joint or d in claimed: s=-1
            else: s = 2 if lateness(c0s, d, c0_step, total)==0 else 1
            if s>best: best=s; bc=tc
        c1n=nb1[bc]; new_pos[1]=c1n; claimed.add(c1n)
        c1_cur=c1s.get(c1n, idx[1])

        # C2
        nb2=get_nb4(*pos[2],m); best=-1; bc=0
        for tc,d in enumerate(nb2):
            if d in joint or d in claimed: s=-1
            else:
                l0=lateness(c0s,d,c0_step,total); l1=lateness(c1s,d,c1_cur,total)
                if   l0==0 and l1==0: s=3
                elif l0==0:           s=2
                else:                 s=1
            if s>best: best=s; bc=tc
        c2n=nb2[bc]; new_pos[2]=c2n; claimed.add(c2n)
        c2_cur=c2s.get(c2n,idx[2])

        # C3
        nb3=get_nb4(*pos[3],m); best=-1; bc=0
        for tc,d in enumerate(nb3):
            if d in joint or d in claimed: s=-1
            else:
                l0=lateness(c0s,d,c0_step,total)
                l1=lateness(c1s,d,c1_cur, total)
                l2=lateness(c2s,d,c2_cur, total)
                if   l0==0 and l1==0 and l2==0: s=4
                elif l0==0 and l1==0:            s=3
                elif l0==0:                      s=2
                else:                            s=1
            if s>best: best=s; bc=tc
        new_pos[3]=nb3[bc]; claimed.add(nb3[bc])

        for c in range(4): pos[c]=new_pos[c]; joint.add(new_pos[c])
        history.append(tuple(pos)); coverage.append(len(joint))
        if len(joint)==total: break

    return history, coverage


# ── V6: Existence Tiers + Late-Stealing ──────────────────────────────────────

def simulate_4d_v6(m, s0, s1, s2, s3, late_frac=0.187):
    """
    4-agent cooperative coverage — V6 existence tiers + late-steal sub-tier.

    Each scout Ck first checks for pure gaps (no leader visits v), then for
    vertices in the last `late_frac` of each leader's remaining schedule —
    stealing the tail to compress the cover time.

    Tier hierarchy:
      C1:  3 = gap(C0)  |  2 = C0-late  |  1 = C0-soon
      C2:  4 = gap(C0+C1)  |  3 = C0-skip,C1-late  |  2 = C0-skip/C0-late  |  1 = soon
      C3:  5 = gap(all)  |  4 = C0+C1-skip,C2-late  |  3 = C0-skip,C1-late  |
           2 = C0-late  |  1 = soon

    Args:
        m:          Grid side length.
        s0–s3:      Pre-planned Hamiltonian sequences for C0–C3.
        late_frac:  Fraction of remaining schedule that counts as "late".
                    Optimal: ~0.187 for 4D m=5.

    Returns:
        (history, coverage)
    """
    total = m**4
    c0s = {v: t for t, v in enumerate(s0)}
    c1s = {v: t for t, v in enumerate(s1)}
    c2s = {v: t for t, v in enumerate(s2)}

    pos   = [s0[0], s1[0], s2[0], s3[0]]
    joint = set(pos)
    history  = [tuple(pos)]
    coverage = [len(joint)]
    idx = [0] * 4

    for t in range(total):
        new_pos = [None] * 4
        claimed = set()

        n0 = s0[(t + 1) % total]
        new_pos[0] = n0; claimed.add(n0)
        c0_step = t + 1
        c0_rem  = total - c0_step
        thr = int(c0_rem * (1 - late_frac))

        idx[1] = (idx[1]+1) % total
        idx[2] = (idx[2]+1) % total
        idx[3] = (idx[3]+1) % total

        # ── C1 ────────────────────────────────────────────────────────────────
        nb1=get_nb4(*pos[1],m); best=-1; bc=0
        for tc,d in enumerate(nb1):
            if d in joint or d in claimed: s=-1
            else:
                l0=lateness(c0s,d,c0_step,total)
                if   l0==0:   s=3
                elif l0>=thr: s=2
                else:         s=1
            if s>best: best=s; bc=tc
        c1n=nb1[bc]; new_pos[1]=c1n; claimed.add(c1n)
        c1_cur=c1s.get(c1n,idx[1])

        # ── C2 ────────────────────────────────────────────────────────────────
        nb2=get_nb4(*pos[2],m); best=-1; bc=0
        for tc,d in enumerate(nb2):
            if d in joint or d in claimed: s=-1
            else:
                l0=lateness(c0s,d,c0_step,total)
                l1=lateness(c1s,d,c1_cur, total)
                if   l0==0 and l1==0:     s=4
                elif l0==0 and l1>=thr:   s=3
                elif l0==0:               s=2
                elif l0>=thr:             s=2
                else:                     s=1
            if s>best: best=s; bc=tc
        c2n=nb2[bc]; new_pos[2]=c2n; claimed.add(c2n)
        c2_cur=c2s.get(c2n,idx[2])

        # ── C3 ────────────────────────────────────────────────────────────────
        nb3=get_nb4(*pos[3],m); best=-1; bc=0
        for tc,d in enumerate(nb3):
            if d in joint or d in claimed: s=-1
            else:
                l0=lateness(c0s,d,c0_step,total)
                l1=lateness(c1s,d,c1_cur, total)
                l2=lateness(c2s,d,c2_cur, total)
                if   l0==0 and l1==0 and l2==0:     s=5
                elif l0==0 and l1==0 and l2>=thr:   s=4
                elif l0==0 and l1==0:                s=3
                elif l0==0 and l1>=thr:              s=3
                elif l0==0:                          s=2
                elif l0>=thr:                        s=2
                else:                                s=1
            if s>best: best=s; bc=tc
        new_pos[3]=nb3[bc]; claimed.add(nb3[bc])

        for c in range(4): pos[c]=new_pos[c]; joint.add(new_pos[c])
        history.append(tuple(pos)); coverage.append(len(joint))
        if len(joint)==total: break

    return history, coverage


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cover_step(coverage, total):
    return next((i for i, c in enumerate(coverage) if c == total), len(coverage)-1)


def main():
    parser = argparse.ArgumentParser(description="4D cooperative cycle coverage")
    parser.add_argument("--m",         type=int,   default=5,     help="Grid side length")
    parser.add_argument("--late-frac", type=float, default=0.187, help="Late-steal fraction (V6)")
    parser.add_argument("--sweep",     action="store_true",        help="Sweep late_frac 0.01–0.50")
    args = parser.parse_args()

    m     = args.m
    total = m**4

    print(f"Building sequences for m={m} (total={total})...")
    d0 = diagonal4d(m)
    g1 = make_golden4d(m, rot=0)
    g2 = make_golden4d(m, rot=1)
    g3 = make_golden4d(m, rot=2)

    _, cov_v3 = simulate_4d_v3(m, d0, g1, g2, g3)
    cs_v3 = _cover_step(cov_v3, total)
    print(f"\nV3 (pure existence): {cs_v3}/{total}  ({cs_v3/total*100:.2f}%)")

    if args.sweep:
        print("\nSweeping late_frac 0.01 – 0.50:")
        best_r = total; best_f = 0.0
        for num in range(1, 51):
            frac = num / 100
            _, cov = simulate_4d_v6(m, d0, g1, g2, g3, late_frac=frac)
            cs = _cover_step(cov, total)
            marker = " ← best" if cs < best_r else ""
            if cs < best_r: best_r = cs; best_f = frac
            print(f"  frac={frac:.2f}: {cs}/{total} ({cs/total*100:.1f}%){marker}")
        print(f"\nBest: frac={best_f:.3f} → {best_r}/{total} ({best_r/total*100:.2f}%)")
        print(f"V6 saves {cs_v3 - best_r} steps over V3")
    else:
        _, cov_v6 = simulate_4d_v6(m, d0, g1, g2, g3, late_frac=args.late_frac)
        cs_v6 = _cover_step(cov_v6, total)
        print(f"V6 (late_frac={args.late_frac}): {cs_v6}/{total}  ({cs_v6/total*100:.2f}%)")
        print(f"V6 saves {cs_v3 - cs_v6} steps over V3")


if __name__ == "__main__":
    main()
