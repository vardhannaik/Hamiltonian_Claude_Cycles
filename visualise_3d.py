"""
visualise_3d.py — animated MP4 of V6 cooperative coverage on a 3D toroidal grid.

Usage:
    python visualise_3d.py --m 5 --late-frac 0.095 --out dg_3d_m5_v6.mp4
    python visualise_3d.py --m 5 --fps 12 --dpi 150

Requires: matplotlib, numpy, ffmpeg (for MP4 output)
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

from utils import get_nb3, lateness
from paths_3d import diagonal3d, make_golden3d
from simulate_3d import simulate_3d_v3, simulate_3d_v6

PHI = (1 + math.sqrt(5)) / 2

BG     = "#08090c"
GC     = "#151b25"
COLS   = ["#34d399", "#818cf8", "#f59e0b"]   # C0 green, C1 violet, C2 amber
TRAIL  = 14


def _cover_step(coverage, total):
    return next((i for i, c in enumerate(coverage) if c == total), len(coverage) - 1)


def render(m, late_frac, out_path, fps, dpi):
    total = m ** 3
    print(f"Building sequences (m={m})...")
    d0 = diagonal3d(m)
    g1 = make_golden3d(m, rot=0)
    g2 = make_golden3d(m, rot=1)

    _, cov_v3 = simulate_3d_v3(m, d0, g1, g2)
    cs_v3 = _cover_step(cov_v3, total)

    print(f"Simulating V6 (late_frac={late_frac})...")
    history, coverage = simulate_3d_v6(m, d0, g1, g2, late_frac=late_frac)
    cover_step = _cover_step(coverage, total)
    print(f"  V3: {cs_v3}/{total}  V6: {cover_step}/{total}  saves {cs_v3-cover_step} steps")

    # Pre-compute visited sets per frame
    n_frames_total = min(cover_step + 5, len(history))
    all_visited = []
    joint_set = set()
    for t in range(n_frames_total):
        joint_set = joint_set | set(history[t])
        all_visited.append(frozenset(joint_set))
    n_frames = n_frames_total

    print(f"Rendering {n_frames} frames → {out_path}")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs = GridSpec(2, 3, figure=fig,
                  left=0.03, right=0.97, top=0.88, bottom=0.07,
                  hspace=0.3, wspace=0.15,
                  height_ratios=[3, 1], width_ratios=[3, 1.2, 1])

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.set_facecolor(BG)

    ax_cov = fig.add_subplot(gs[1, 0])
    ax_cov.set_facecolor("#0e1015")
    for sp in ax_cov.spines.values(): sp.set_color("#1c2130")
    ax_cov.set_xlim(0, cover_step + 3)
    ax_cov.set_ylim(0, total * 1.06)
    ax_cov.set_xlabel("step",    color="#6b7280", fontsize=8, fontfamily="monospace")
    ax_cov.set_ylabel("covered", color="#6b7280", fontsize=8, fontfamily="monospace")
    ax_cov.tick_params(colors="#374151", labelsize=7)
    ax_cov.axhline(total, color="#2a3550", lw=1, ls="--")
    ax_cov.axhline(cs_v3, color="#7c3f3f", lw=1, ls=":", alpha=0.6)
    ax_cov.text(1, total * 1.02, f"n=m³={total}", color="#374151",
                fontsize=7, fontfamily="monospace")
    ax_cov.text(1, cs_v3 + 1, f"V3={cs_v3}", color="#7c3f3f",
                fontsize=7, fontfamily="monospace", alpha=0.7)
    cov_line, = ax_cov.plot([], [], color="#34d399", lw=2.0)
    cov_dot   = ax_cov.scatter([], [], color="white", s=35, zorder=5)

    ax_tier = fig.add_subplot(gs[0, 1])
    ax_tier.set_facecolor("#0d0f12"); ax_tier.axis("off")

    ax_leg = fig.add_subplot(gs[1, 1])
    ax_leg.set_facecolor("#0d0f12"); ax_leg.axis("off")
    for i, (col, lbl) in enumerate(zip(COLS, ["C0  diagonal", "C1  gap|late", "C2  gap|late"])):
        ax_leg.plot([0.04, 0.18], [0.80 - i*0.28]*2, color=col, lw=3,
                    transform=ax_leg.transAxes)
        ax_leg.text(0.22, 0.80 - i*0.28, lbl, color="#9ca3af", fontsize=8,
                    va="center", transform=ax_leg.transAxes, fontfamily="monospace")

    ax_stat = fig.add_subplot(gs[:, 2])
    ax_stat.set_facecolor("#0d0f12"); ax_stat.axis("off")

    fig.text(0.5, 0.96,
             f"D+G 3D V6  ·  m={m}  ·  m³={total}  ·  "
             f"V3 existence + late-{late_frac:.1%} steal",
             color="white", fontsize=10, fontfamily="monospace",
             fontweight="bold", ha="center", va="top")
    step_txt = fig.text(0.5, 0.91, "", color="#6b7280", fontsize=9,
                        fontfamily="monospace", ha="center", va="top")

    seqs     = [d0, g1, g2]
    all_verts = [(i, j, k) for i in range(m) for j in range(m) for k in range(m)]
    c0s_map  = {v: t for t, v in enumerate(d0)}
    c1s_map  = {v: t for t, v in enumerate(g1)}

    # ── Frame function ───────────────────────────────────────────────────────
    def make_frame(fi):
        tc      = min(fi, len(all_visited) - 1)
        visited = all_visited[tc]
        pos_t   = history[min(fi, len(history) - 1)]

        # 3D scene
        ax3d.cla(); ax3d.set_facecolor(BG)
        for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor(GC)
        ax3d.set_xlim(-0.5, m-0.5); ax3d.set_ylim(-0.5, m-0.5); ax3d.set_zlim(-0.5, m-0.5)
        ax3d.set_xticks(range(m)); ax3d.set_yticks(range(m)); ax3d.set_zticks(range(m))
        ax3d.tick_params(colors="#374151", labelsize=7)
        ax3d.set_xlabel("i", color="#4b5563", fontsize=9, labelpad=1)
        ax3d.set_ylabel("j", color="#4b5563", fontsize=9, labelpad=1)
        ax3d.set_zlabel("k", color="#4b5563", fontsize=9, labelpad=1)
        ax3d.view_init(elev=24, azim=35 + fi * 1.5)

        c0_step_now = tc + 1
        c0_rem = total - c0_step_now
        thr = int(c0_rem * (1 - late_frac))

        vis_pts = []; late0_pts = []; soon0_pts = []; unv_pts = []
        for v in all_verts:
            if v in visited:
                vis_pts.append(v)
            else:
                l0 = c0s_map.get(v, -1)
                if l0 < 0:
                    unv_pts.append(v)
                else:
                    rem = (l0 - c0_step_now) % total
                    if rem > 0 and rem >= thr: late0_pts.append(v)
                    elif rem > 0:              soon0_pts.append(v)
                    else:                      unv_pts.append(v)

        if unv_pts:
            xs,ys,zs = zip(*unv_pts); ax3d.scatter(xs,ys,zs, c="#1a2030", s=14, alpha=0.3, depthshade=True)
        if soon0_pts:
            xs,ys,zs = zip(*soon0_pts); ax3d.scatter(xs,ys,zs, c="#1e3a5f", s=16, alpha=0.5, depthshade=True)
        if late0_pts:
            xs,ys,zs = zip(*late0_pts); ax3d.scatter(xs,ys,zs, c="#7c3aed", s=20, alpha=0.65, depthshade=True)
        if vis_pts:
            xs,ys,zs = zip(*vis_pts); ax3d.scatter(xs,ys,zs, c="#2a3a55", s=22, alpha=0.6, depthshade=True)

        for cidx, (seq, col) in enumerate(zip(seqs, COLS)):
            trail = [history[min(ts, len(history)-1)][cidx]
                     for ts in range(max(0, tc - TRAIL), tc + 1)]
            for ti in range(len(trail) - 1):
                v0, v1 = trail[ti], trail[ti+1]
                frac = (ti + 1) / max(len(trail), 1)
                ax3d.plot([v0[0],v1[0]], [v0[1],v1[1]], [v0[2],v1[2]],
                          color=col, alpha=frac*0.7, lw=2.2)
            cp = pos_t[cidx]
            ax3d.scatter(*cp, c=col, s=170, alpha=1.0, depthshade=False,
                         edgecolors="white", linewidths=0.6, zorder=10)
            ax3d.scatter(*cp, c=col, s=550, alpha=0.10, depthshade=False, zorder=9)

        ax3d.scatter([],[],[], c="#7c3aed", s=20, alpha=0.65, label="C0-late (steal)")
        ax3d.scatter([],[],[], c="#1e3a5f", s=16, alpha=0.5,  label="C0-soon")
        ax3d.scatter([],[],[], c="#1a2030", s=14, alpha=0.3,  label="gap")
        ax3d.scatter([],[],[], c="#2a3a55", s=22, alpha=0.6,  label="visited")
        ax3d.legend(loc="upper left", fontsize=6.5, framealpha=0.3,
                    labelcolor="white", facecolor="#0a0c0f", edgecolor="#1c2130")

        # Tier readout
        ax_tier.cla(); ax_tier.set_facecolor("#0d0f12"); ax_tier.axis("off")
        ax_tier.text(0.5, 0.97, "scout tiers", color="#4b5563", fontsize=8.5,
                     ha="center", va="top", transform=ax_tier.transAxes,
                     fontfamily="monospace")

        c1_cur = c1s_map.get(pos_t[1], 0)
        tc_colors = {-1:"#374151", 1:"#1e3a5f", 2:"#1d4ed8", 3:"#065f46", 4:"#14532d"}
        tc_labels = {-1:"blocked", 1:"C0-soon", 2:"C0-late/skip",
                     3:"gap/C1-late", 4:"pure gap"}
        for ci, (cname, col) in enumerate(zip(["C1","C2"], COLS[1:])):
            nbs = get_nb3(*pos_t[ci+1], m)
            scores = []
            for nb in nbs:
                if nb in visited: scores.append(-1); continue
                l0 = lateness(c0s_map, nb, c0_step_now, total)
                if ci == 0:
                    s = 3 if l0==0 else (2 if l0>=thr else 1)
                else:
                    l1 = lateness(c1s_map, nb, c1_cur, total)
                    if l0==0 and l1==0:   s=4
                    elif l0==0 and l1>=thr: s=3
                    elif l0==0:           s=2
                    elif l0>=thr:         s=2
                    else:                 s=1
                scores.append(s)
            best_s = max(scores) if scores else 0
            y = 0.72 - ci * 0.38
            ax_tier.add_patch(plt.Rectangle(
                (0.05, y-0.12), 0.90, 0.22,
                facecolor=tc_colors.get(best_s, "#0d0f12"), alpha=0.4,
                transform=ax_tier.transAxes))
            ax_tier.plot([0.07, 0.20], [y, y], color=col, lw=3,
                         transform=ax_tier.transAxes)
            ax_tier.text(0.24, y+0.04, cname, color=col, fontsize=9, va="center",
                         transform=ax_tier.transAxes,
                         fontfamily="monospace", fontweight="bold")
            ax_tier.text(0.24, y-0.06,
                         f"tier {best_s}: {tc_labels.get(best_s,'?')}",
                         color="#9ca3af", fontsize=7, va="center",
                         transform=ax_tier.transAxes, fontfamily="monospace")

        # Stats
        ax_stat.cla(); ax_stat.set_facecolor("#0d0f12"); ax_stat.axis("off")
        cv = coverage[min(tc, len(coverage)-1)]
        stats = [
            ("step",    f"{tc}/{cover_step}"),
            ("covered", f"{cv}/{total}"),
            ("pct",     f"{cv/total*100:.1f}%"),
            ("",""),
            ("V3 cover", f"{cs_v3}/{total}"),
            ("V6 cover", f"{cover_step}/{total}"),
            ("savings",  f"{cs_v3-cover_step} steps"),
            ("",""),
            ("late_frac",  f"{late_frac:.1%}"),
            ("threshold",  f"{thr} steps"),
            ("late nodes", f"{len(late0_pts)}"),
            ("gap nodes",  f"{len(unv_pts)}"),
        ]
        for si, (k, v) in enumerate(stats):
            if not k: continue
            ax_stat.text(0.05, 0.95 - si*0.075, k+":", color="#4b5563",
                         fontsize=8, va="top", transform=ax_stat.transAxes,
                         fontfamily="monospace")
            ax_stat.text(0.95, 0.95 - si*0.075, v, color="#e5e7eb",
                         fontsize=8, va="top", ha="right",
                         transform=ax_stat.transAxes, fontfamily="monospace")
        if cv == total:
            ax_stat.text(0.5, 0.08, "✓ COMPLETE", color="#34d399", fontsize=11,
                         ha="center", va="bottom", transform=ax_stat.transAxes,
                         fontfamily="monospace", fontweight="bold")

        cov_line.set_data(range(min(tc+1, len(coverage))),
                          coverage[:min(tc+1, len(coverage))])
        cov_dot.set_offsets([[tc, cv]])
        step_txt.set_text(
            f"step {tc:3d}/{cover_step}  ·  {cv}/{total} nodes ({cv/total*100:.1f}%)"
            + ("  ✓ COMPLETE" if cv == total else ""))

    ani = animation.FuncAnimation(
        fig, make_frame, frames=n_frames, interval=int(1000/fps), blit=False)
    writer = animation.FFMpegWriter(
        fps=fps, bitrate=2200,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "19"])
    ani.save(out_path, writer=writer, dpi=dpi)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualise 3D V6 coverage")
    parser.add_argument("--m",         type=int,   default=5,                  help="Grid side length")
    parser.add_argument("--late-frac", type=float, default=0.095,              help="Late-steal fraction")
    parser.add_argument("--out",       type=str,   default="dg_3d_m5_v6.mp4", help="Output MP4 path")
    parser.add_argument("--fps",       type=int,   default=12,                 help="Frames per second")
    parser.add_argument("--dpi",       type=int,   default=150,                help="Resolution (DPI)")
    args = parser.parse_args()
    render(args.m, args.late_frac, args.out, args.fps, args.dpi)


if __name__ == "__main__":
    main()
