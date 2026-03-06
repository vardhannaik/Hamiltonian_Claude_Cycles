"""
visualise_4d.py — animated MP4 of V6 cooperative coverage on a 4D toroidal grid.

The 4th dimension is shown as 5 side-by-side 3D panels, one per l-slice (l=0..m-1).
Agents appear in the panel matching their current l-coordinate.

Usage:
    python visualise_4d.py --m 5 --late-frac 0.187 --out dg_4d_m5_v6.mp4
    python visualise_4d.py --m 5 --fps 15 --dpi 140

Requires: matplotlib, numpy, ffmpeg
"""

import argparse
import sys
import math
sys.setrecursionlimit(200_000)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from utils import get_nb4, lateness
from paths_4d import diagonal4d, make_golden4d
from simulate_4d import simulate_4d_v3, simulate_4d_v6

BG    = "#08090c"
GC    = "#151b25"
COLS  = ["#34d399", "#818cf8", "#f59e0b", "#f87171"]  # C0..C3
TRAIL = 8


def _cover_step(coverage, total):
    return next((i for i, c in enumerate(coverage) if c == total), len(coverage) - 1)


def render(m, late_frac, out_path, fps, dpi, stride=2):
    total = m ** 4
    print(f"Building 4D sequences (m={m}, total={total})...")
    d0 = diagonal4d(m)
    g1 = make_golden4d(m, rot=0)
    g2 = make_golden4d(m, rot=1)
    g3 = make_golden4d(m, rot=2)

    _, cov_v3 = simulate_4d_v3(m, d0, g1, g2, g3)
    cs_v3 = _cover_step(cov_v3, total)

    print(f"Simulating V6 (late_frac={late_frac})...")
    history, coverage = simulate_4d_v6(m, d0, g1, g2, g3, late_frac=late_frac)
    cover_step = _cover_step(coverage, total)
    print(f"  V3: {cs_v3}/{total}  V6: {cover_step}/{total}  saves {cs_v3-cover_step} steps")

    n_frames_total = min(cover_step + 5, len(history))
    all_visited = []
    joint_set = set()
    for t in range(n_frames_total):
        joint_set = joint_set | set(history[t])
        all_visited.append(frozenset(joint_set))

    frame_indices = list(range(0, n_frames_total, stride))
    if (n_frames_total - 1) not in frame_indices:
        frame_indices.append(n_frames_total - 1)
    n_frames = len(frame_indices)
    print(f"Rendering {n_frames} frames (stride={stride}) → {out_path}")

    # ── Figure: 1×5 strip of 3D panels (one per l-slice) ────────────────────
    fig = plt.figure(figsize=(22, 5.5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    axes = []
    for l_val in range(m):
        ax = fig.add_subplot(1, m, l_val + 1, projection="3d")
        ax.set_facecolor(BG)
        axes.append(ax)

    # Coverage axis below (use inset)
    ax_cov = fig.add_axes([0.08, 0.04, 0.84, 0.10])
    ax_cov.set_facecolor("#0e1015")
    for sp in ax_cov.spines.values(): sp.set_color("#1c2130")
    ax_cov.set_xlim(0, cover_step + 3)
    ax_cov.set_ylim(0, total * 1.06)
    ax_cov.tick_params(colors="#374151", labelsize=6)
    ax_cov.axhline(total, color="#2a3550", lw=1, ls="--")
    ax_cov.axhline(cs_v3, color="#7c3f3f", lw=1, ls=":", alpha=0.5)
    ax_cov.text(1, total * 1.01, f"n={total}", color="#374151",
                fontsize=6, fontfamily="monospace")
    ax_cov.text(1, cs_v3 + total*0.01, f"V3={cs_v3}",
                color="#7c3f3f", fontsize=6, fontfamily="monospace", alpha=0.6)
    cov_line, = ax_cov.plot([], [], color="#34d399", lw=1.5)
    cov_dot   = ax_cov.scatter([], [], color="white", s=20, zorder=5)

    fig.text(0.5, 0.99,
             f"D+G 4D V6  ·  m={m}  ·  m⁴={total}  ·  "
             f"5 l-slices  ·  late-{late_frac:.1%} steal",
             color="white", fontsize=9, fontfamily="monospace",
             fontweight="bold", ha="center", va="top")
    step_txt = fig.text(0.5, 0.96, "", color="#6b7280", fontsize=8,
                        fontfamily="monospace", ha="center", va="top")

    seqs    = [d0, g1, g2, g3]
    c0s_map = {v: t for t, v in enumerate(d0)}

    verts_by_l = {l_val: [(i,j,k) for i in range(m) for j in range(m) for k in range(m)]
                  for l_val in range(m)}

    def make_frame(fi):
        t       = frame_indices[fi]
        tc      = min(t, len(all_visited) - 1)
        visited = all_visited[tc]
        pos_t   = history[min(t, len(history) - 1)]

        c0_step_now = tc + 1
        c0_rem = total - c0_step_now
        thr    = int(c0_rem * (1 - late_frac))

        for l_val, ax in enumerate(axes):
            ax.cla(); ax.set_facecolor(BG)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False; pane.set_edgecolor(GC)
            ax.set_xlim(-0.5, m-0.5); ax.set_ylim(-0.5, m-0.5); ax.set_zlim(-0.5, m-0.5)
            ax.set_xticks(range(m)); ax.set_yticks(range(m)); ax.set_zticks(range(m))
            ax.tick_params(colors="#374151", labelsize=5)
            ax.set_xlabel("i", color="#374151", fontsize=6, labelpad=0)
            ax.set_ylabel("j", color="#374151", fontsize=6, labelpad=0)
            ax.set_zlabel("k", color="#374151", fontsize=6, labelpad=0)
            ax.set_title(f"l = {l_val}", color="#4b5563",
                         fontsize=7, fontfamily="monospace", pad=2)
            ax.view_init(elev=22, azim=42 + fi * 0.5)

            vis_pts=[]; late_pts=[]; soon_pts=[]; unv_pts=[]
            for ijk in verts_by_l[l_val]:
                v = (*ijk, l_val)
                if v in visited: vis_pts.append(ijk)
                else:
                    l0 = c0s_map.get(v, -1)
                    if l0 < 0: unv_pts.append(ijk)
                    else:
                        rem = (l0 - c0_step_now) % total
                        if rem > 0 and rem >= thr: late_pts.append(ijk)
                        elif rem > 0:              soon_pts.append(ijk)
                        else:                      unv_pts.append(ijk)

            if unv_pts:  xs,ys,zs=zip(*unv_pts);  ax.scatter(xs,ys,zs, c="#1a2030",s=10,alpha=0.3,depthshade=True)
            if soon_pts: xs,ys,zs=zip(*soon_pts); ax.scatter(xs,ys,zs, c="#1e3a5f",s=12,alpha=0.5,depthshade=True)
            if late_pts: xs,ys,zs=zip(*late_pts); ax.scatter(xs,ys,zs, c="#7c3aed",s=16,alpha=0.65,depthshade=True)
            if vis_pts:  xs,ys,zs=zip(*vis_pts);  ax.scatter(xs,ys,zs, c="#2a3a55",s=14,alpha=0.6,depthshade=True)

            # Agents in this l-slice
            for cidx, (seq, col) in enumerate(zip(seqs, COLS)):
                cp = pos_t[cidx]
                if cp[3] != l_val: continue  # agent is in a different l-slice
                cp3 = cp[:3]
                trail = []
                for ts in range(max(0, tc - TRAIL), tc + 1):
                    p = history[min(ts, len(history)-1)][cidx]
                    if p[3] == l_val: trail.append(p[:3])
                for ti in range(len(trail) - 1):
                    v0, v1 = trail[ti], trail[ti+1]
                    frac = (ti + 1) / max(len(trail), 1)
                    ax.plot([v0[0],v1[0]], [v0[1],v1[1]], [v0[2],v1[2]],
                            color=col, alpha=frac*0.7, lw=1.8)
                ax.scatter(*cp3, c=col, s=120, alpha=1.0, depthshade=False,
                           edgecolors="white", linewidths=0.5, zorder=10)
                ax.scatter(*cp3, c=col, s=400, alpha=0.10, depthshade=False, zorder=9)

        cv = coverage[min(tc, len(coverage)-1)]
        cov_line.set_data(range(min(tc+1, len(coverage))), coverage[:min(tc+1, len(coverage))])
        cov_dot.set_offsets([[tc, cv]])
        step_txt.set_text(
            f"step {tc:3d}/{cover_step}  ·  {cv}/{total} nodes ({cv/total*100:.1f}%)"
            + ("  ✓ COMPLETE" if cv == total else ""))

    ani = animation.FuncAnimation(
        fig, make_frame, frames=n_frames, interval=int(1000/fps), blit=False)
    writer = animation.FFMpegWriter(
        fps=fps, bitrate=2000,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"])
    ani.save(out_path, writer=writer, dpi=dpi)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualise 4D V6 coverage")
    parser.add_argument("--m",         type=int,   default=5,                  help="Grid side length")
    parser.add_argument("--late-frac", type=float, default=0.187,              help="Late-steal fraction")
    parser.add_argument("--out",       type=str,   default="dg_4d_m5_v6.mp4", help="Output MP4 path")
    parser.add_argument("--fps",       type=int,   default=15,                 help="Frames per second")
    parser.add_argument("--dpi",       type=int,   default=140,                help="Resolution (DPI)")
    parser.add_argument("--stride",    type=int,   default=2,                  help="Frame stride (1=every frame)")
    args = parser.parse_args()
    render(args.m, args.late_frac, args.out, args.fps, args.dpi, args.stride)


if __name__ == "__main__":
    main()
