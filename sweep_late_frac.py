"""
sweep_late_frac.py — benchmark LATE_FRAC across a range for 3D or 4D.

Usage:
    python sweep_late_frac.py --dim 3 --m 5
    python sweep_late_frac.py --dim 4 --m 5
    python sweep_late_frac.py --dim 3 --m 5 --fine 0.08 0.12
"""

import argparse
import sys
sys.setrecursionlimit(200_000)

from paths_3d import diagonal3d, make_golden3d
from paths_4d import diagonal4d, make_golden4d
from simulate_3d import simulate_3d_v3, simulate_3d_v6
from simulate_4d import simulate_4d_v3, simulate_4d_v6


def _cover_step(coverage, total):
    return next((i for i, c in enumerate(coverage) if c == total), len(coverage) - 1)


def sweep_3d(m, lo=0.01, hi=0.50, steps=50):
    total = m ** 3
    print(f"Building 3D sequences (m={m}, total={total})...")
    d0 = diagonal3d(m)
    g1 = make_golden3d(m, rot=0)
    g2 = make_golden3d(m, rot=1)

    _, cov_v3 = simulate_3d_v3(m, d0, g1, g2)
    cs_v3 = _cover_step(cov_v3, total)
    print(f"V3 baseline: {cs_v3}/{total} ({cs_v3/total*100:.2f}%)\n")
    print(f"{'frac':>8}  {'cover':>8}  {'pct':>7}  {'vs V3':>8}")
    print("-" * 40)

    best_r = total; best_f = lo
    frac_range = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    for frac in frac_range:
        _, cov = simulate_3d_v6(m, d0, g1, g2, late_frac=frac)
        cs = _cover_step(cov, total)
        marker = " ◀" if cs < best_r else ""
        if cs < best_r: best_r = cs; best_f = frac
        print(f"  {frac:.4f}  {cs:>5}/{total}  {cs/total*100:>6.2f}%  {cs-cs_v3:>+6}{marker}")

    print(f"\nBest: frac={best_f:.4f}  →  {best_r}/{total} ({best_r/total*100:.2f}%)")
    print(f"V6 saves {cs_v3 - best_r} steps over V3")
    return best_f, best_r


def sweep_4d(m, lo=0.01, hi=0.50, steps=50):
    total = m ** 4
    print(f"Building 4D sequences (m={m}, total={total})...")
    d0 = diagonal4d(m)
    g1 = make_golden4d(m, rot=0)
    g2 = make_golden4d(m, rot=1)
    g3 = make_golden4d(m, rot=2)

    _, cov_v3 = simulate_4d_v3(m, d0, g1, g2, g3)
    cs_v3 = _cover_step(cov_v3, total)
    print(f"V3 baseline: {cs_v3}/{total} ({cs_v3/total*100:.2f}%)\n")
    print(f"{'frac':>8}  {'cover':>10}  {'pct':>7}  {'vs V3':>8}")
    print("-" * 44)

    best_r = total; best_f = lo
    frac_range = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    for frac in frac_range:
        _, cov = simulate_4d_v6(m, d0, g1, g2, g3, late_frac=frac)
        cs = _cover_step(cov, total)
        marker = " ◀" if cs < best_r else ""
        if cs < best_r: best_r = cs; best_f = frac
        print(f"  {frac:.4f}  {cs:>7}/{total}  {cs/total*100:>6.2f}%  {cs-cs_v3:>+6}{marker}")

    print(f"\nBest: frac={best_f:.4f}  →  {best_r}/{total} ({best_r/total*100:.2f}%)")
    print(f"V6 saves {cs_v3 - best_r} steps over V3")
    return best_f, best_r


def main():
    parser = argparse.ArgumentParser(description="Sweep LATE_FRAC for cooperative coverage")
    parser.add_argument("--dim",  type=int,   default=3,    help="3 or 4")
    parser.add_argument("--m",    type=int,   default=5,    help="Grid side length")
    parser.add_argument("--lo",   type=float, default=0.01, help="Sweep lower bound")
    parser.add_argument("--hi",   type=float, default=0.50, help="Sweep upper bound")
    parser.add_argument("--steps",type=int,   default=50,   help="Number of frac values")
    parser.add_argument("--fine", type=float, nargs=2,      help="Fine-tune range: --fine 0.08 0.12")
    args = parser.parse_args()

    lo, hi = (args.fine[0], args.fine[1]) if args.fine else (args.lo, args.hi)

    if args.dim == 3:
        sweep_3d(args.m, lo=lo, hi=hi, steps=args.steps)
    elif args.dim == 4:
        sweep_4d(args.m, lo=lo, hi=hi, steps=args.steps)
    else:
        print("--dim must be 3 or 4"); sys.exit(1)


if __name__ == "__main__":
    main()
