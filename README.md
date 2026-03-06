# Cooperative Cycle Coverage on Toroidal Grids
## Diagonal Leaders, Golden Scouts, and Late-Stealing in nD

> **Abstract.** We present a multi-agent cooperative coverage algorithm for n-dimensional toroidal grids where each agent follows a pre-planned path but dynamically steals vertices from its leaders' future schedules. A diagonal Hamiltonian (C0) sweeps the grid in uniform fiber planes; golden-ratio scouts (C1, C2, … Cn−1) first fill pure gaps — vertices no leader will ever visit — then prioritise stealing the *latest-scheduled* vertices from their leaders, since those are visited last and cause the most delay. Applied to 3D and 4D grids at m=5, the algorithm achieves **50.4% efficiency in 3D** (vs 73.6% for pure-existence scoring) and **56.6% efficiency in 4D** (vs 69.1%), covering m³=125 and m⁴=625 vertices in 63 and 354 cooperative steps respectively.

---

## Table of Contents

1. [Background](#background)
2. [Path Construction](#path-construction)
   - [Diagonal Hamiltonian (C0)](#diagonal-hamiltonian-c0)
   - [Golden-Ratio Path (C1…Cn−1)](#golden-ratio-path-c1cn1)
3. [Role-Aware Simulation](#role-aware-simulation)
   - [V3: Pure Existence Scoring](#v3-pure-existence-scoring)
   - [V6: Existence Tiers + Late-Stealing](#v6-existence-tiers--late-stealing)
4. [Results](#results)
5. [3D Implementation (m=5)](#3d-implementation-m5)
6. [4D Implementation (m=5)](#4d-implementation-m5)
7. [Full Code](#full-code)

---

## Background

Consider an m×m×…×m toroidal grid (each axis wraps around). Multiple agents must collectively visit every vertex exactly once as fast as possible — minimising the *cover step*, the first timestep at which all vertices have been visited by at least one agent.

Each agent moves one step per timestep to an adjacent vertex (taxicab distance 1, toroidal). The challenge: coordinate their paths so they spread out and cover different regions rather than clustering.

The key insight is **role asymmetry**:
- **C0** is a strict leader following a pre-planned diagonal Hamiltonian path
- **C1, C2, …** are scouts that observe C0's (and each other's) future schedule and adaptively choose which vertex to visit next based on where the leaders are *not* going

This asymmetry allows scouts to make informed decisions without any runtime communication — they only need read-access to the leaders' pre-planned sequences.

---

## Path Construction

### Diagonal Hamiltonian (C0)

The diagonal path sweeps fiber planes `s = (i+j+k+…) % m` in strict order. Within each fiber, a Warnsdorff tiebreaker (prefer vertices with fewest unvisited neighbours) resolves ambiguity.

**Key property:** C0 visits vertices with coordinate sum `s=0` first, then `s=1`, …, `s=m−1`, then repeats. This means the *tail* of C0's schedule contains vertices with the largest coordinate values — concentrated in the high-index corner of the grid.

```python
# 3D Diagonal Hamiltonian
def getD(i, j, k, m):
    """Direction selector based on fiber s=(i+j+k)%m."""
    s = (i + j + k) % m
    if s == 0:
        return "012" if j == m - 1 else "210"
    elif s == m - 1:
        return "120" if i > 0 else "210"
    else:
        return "201" if i == m - 1 else "102"

def diagonal3d(m):
    i, j, k = 0, 0, 0
    path = [(0, 0, 0)]
    for _ in range(m**3 - 1):
        d = getD(i, j, k, m)
        nb = get_nb3(i, j, k, m)
        i, j, k = nb[int(d[0])]
        path.append((i, j, k))
    return path
```

```python
# 4D Diagonal Hamiltonian (Warnsdorff-guided)
def diagonal4d(m):
    total = m**4
    wdeg = {(i,j,k,l): 4
            for i in range(m) for j in range(m)
            for k in range(m) for l in range(m)}

    def dr(i, j, k, l):
        return ((i+j+k+l) % m) * m**3 + i*m**2 + j*m + k

    vis = set()
    cur = (0, 0, 0, 0)
    path = [cur]; vis.add(cur)
    for nb in get_nb4(*cur, m):
        wdeg[nb] -= 1
    cr = dr(*cur)

    for _ in range(total - 1):
        cands = [v for v in get_nb4(*cur, m) if v not in vis]
        if not cands: break
        cands.sort(key=lambda v: ((dr(*v) - cr) % total, wdeg[v]))
        cur = cands[0]
        path.append(cur); vis.add(cur); cr = dr(*cur)
        for nb in get_nb4(*cur, m):
            wdeg[nb] -= 1
    return path
```

### Golden-Ratio Path (C1…Cn−1)

The golden path recursively splits the grid along its longest axis at the golden ratio φ = (1+√5)/2, then stitches the two halves together at an adjacent boundary.

```python
PHI = (1 + 5**0.5) / 2

def build_golden3d(ranges, m):
    ilo, ihi, jlo, jhi, klo, khi = ranges
    dims = [ihi-ilo+1, jhi-jlo+1, khi-klo+1]

    # Base cases: 1D lines
    if dims[1]==1 and dims[2]==1:
        return [(i, jlo, klo) for i in range(ilo, ihi+1)]
    if dims[0]==1 and dims[2]==1:
        return [(ilo, j, klo) for j in range(jlo, jhi+1)]
    if dims[0]==1 and dims[1]==1:
        return [(ilo, jlo, k) for k in range(klo, khi+1)]

    axis = dims.index(max(dims))
    if axis == 0:
        sv = ilo + max(1, round(dims[0] / PHI)) - 1
        lr = (ilo, sv, jlo, jhi, klo, khi)
        rr = (sv+1, ihi, jlo, jhi, klo, khi)
    elif axis == 1:
        sv = jlo + max(1, round(dims[1] / PHI)) - 1
        lr = (ilo, ihi, jlo, sv, klo, khi)
        rr = (ilo, ihi, sv+1, jhi, klo, khi)
    else:
        sv = klo + max(1, round(dims[2] / PHI)) - 1
        lr = (ilo, ihi, jlo, jhi, klo, sv)
        rr = (ilo, ihi, jlo, jhi, sv+1, khi)

    lp = build_golden3d(lr, m)
    rp = build_golden3d(rr, m)

    # Try all 4 orientations to find an adjacent join
    for l, r in [(lp,rp),(lp,rp[::-1]),(lp[::-1],rp),(lp[::-1],rp[::-1])]:
        if sum(min(abs(l[-1][a]-r[0][a]), m-abs(l[-1][a]-r[0][a]))
               for a in range(3)) == 1:
            return l + r
    return lp + rp  # fallback: non-adjacent stitch
```

---

## Role-Aware Simulation

### V3: Pure Existence Scoring

Each scout checks whether its neighbouring vertices will ever be visited by any leader. Vertices that no leader will visit are pure gaps — highest priority.

```python
# V3 scoring for C2 (aware of C0 and C1)
def score_c2_v3(d, joint, claimed, c0s, c1s, c0_step, c1_cur, total):
    if d in joint or d in claimed:
        return -1

    def will_visit(smap, v, cur):
        vi = smap.get(v, -1)
        if vi < 0: return False
        return (vi - cur) % total > 0

    c0_future = will_visit(c0s, d, c0_step)
    c1_future = will_visit(c1s, d, c1_cur)

    if not c0_future and not c1_future:
        return 3   # pure gap — neither leader visits
    elif not c0_future:
        return 2   # C0 skips it, C1 will come
    else:
        return 1   # C0 will visit — lowest priority
```

### V6: Existence Tiers + Late-Stealing

V6 keeps V3's existence tiers as the primary signal and adds a **late-steal sub-tier**: vertices in the last `LATE_FRAC` of a leader's remaining schedule get a higher priority than vertices the leader visits soon.

**Rationale:** C0's diagonal path visits high-coordinate vertices last. Stealing them early saves the maximum number of steps, since otherwise C0 would only reach them near the end of its schedule.

```
Tier hierarchy for C1 (aware of C0):
  tier 3 = C0 will never visit v          ← pure gap
  tier 2 = C0 will visit v, but LATE      ← steal the tail
  tier 1 = C0 will visit v soon           ← let C0 handle it

Tier hierarchy for C2 (aware of C0 + C1):
  tier 4 = neither C0 nor C1 visits       ← pure gap
  tier 3 = C0 skips, C1 visits LATE       ← steal C1's tail
  tier 2 = C0 skips C1-soon, OR C0-late   ← secondary steal
  tier 1 = C0 visits soon                 ← lowest priority
```

```python
LATE_FRAC = 0.095   # 3D tuned  (use 0.187 for 4D)

def simulate_v6(m, s0, s1, s2, late_frac=LATE_FRAC):
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

        # C0 advances strictly along its pre-planned path
        n0 = s0[(t + 1) % total]
        new_pos[0] = n0; claimed.add(n0)
        c0_step = t + 1
        c0_rem = total - c0_step
        thr = int(c0_rem * (1 - late_frac))   # lateness threshold

        idx[1] = (idx[1] + 1) % total
        idx[2] = (idx[2] + 1) % total

        def late(smap, v, cur):
            vi = smap.get(v, -1)
            if vi < 0: return 0
            l = (vi - cur) % total
            return l if l > 0 else 0

        # ── C1: gap(C0) | C0-late | C0-soon ──────────────────────────
        nb1 = get_nb3(*pos[1], m)
        best = -1; bc = 0
        for tc in range(3):
            d = nb1[tc]
            if d in joint or d in claimed:
                s = -1
            else:
                l0 = late(c0s, d, c0_step)
                if l0 == 0:       s = 3   # gap
                elif l0 >= thr:   s = 2   # C0-late → steal
                else:             s = 1   # C0-soon → skip
            if s > best: best = s; bc = tc
        c1n = nb1[bc]; new_pos[1] = c1n; claimed.add(c1n)
        c1_cur = c1s.get(c1n, idx[1])

        # ── C2: gap | C1-late | C0-late | soon ───────────────────────
        nb2 = get_nb3(*pos[2], m)
        best = -1; bc = 0
        for tc in range(3):
            d = nb2[tc]
            if d in joint or d in claimed:
                s = -1
            else:
                l0 = late(c0s, d, c0_step)
                l1 = late(c1s, d, c1_cur)
                if l0 == 0 and l1 == 0:       s = 4   # pure gap
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
```

---

## Results

### Summary Table

| Configuration | Cover Step | Efficiency | Savings vs V3 |
|---|---|---|---|
| 3D m=5, V3 (pure existence) | 92 / 125 | 73.6% | — |
| **3D m=5, V6 (+ late-steal)** | **63 / 125** | **50.4%** | **−29 steps** |
| 4D m=5, V3 (pure existence) | 432 / 625 | 69.1% | — |
| **4D m=5, V6 (+ late-steal)** | **354 / 625** | **56.6%** | **−78 steps** |

### Optimal LATE_FRAC by Dimension

| Dimension | Optimal LATE_FRAC | Tail concentration (avg coord sum Δ) |
|---|---|---|
| 3D m=5 | 0.095 | +1.3 (late vs early) |
| 4D m=5 | 0.187 | +4.0 (late vs early) |

Higher-dimensional grids have stronger tail concentration in the diagonal path, requiring a larger late window to capture the right vertices.

### Why Late-Stealing Works

The diagonal C0 path starts at (0,0,0,…) and visits vertices in order of increasing coordinate sum. The high-index corner — vertices near (m−1, m−1, m−1, …) — is visited last. These are exactly the vertices that:

1. Are hardest to reach from the path's starting point
2. Would be covered last if C0 were the only agent
3. Can be reached early by a scout positioned anywhere in the grid

Stealing them shifts coverage of the most delayed vertices to earlier timesteps, compressing the cover time.

---

## 3D Implementation (m=5)

**Grid:** 5×5×5 = 125 vertices, 3 agents (C0 diagonal + C1 + C2 golden)

```python
# Full 3D simulation with V6 scoring
import math, sys
sys.setrecursionlimit(200000)

PHI = (1 + 5**0.5) / 2
m = 5
LATE_FRAC_3D = 0.095

def get_nb3(i, j, k, m):
    return [((i+1)%m, j, k), (i, (j+1)%m, k), (i, j, (k+1)%m)]

# Build sequences
d0 = diagonal3d(m)                 # C0: Knuth arc diagonal
g1 = make_golden3d(m, rot=0)       # C1: golden, axis rot 0
g2 = make_golden3d(m, rot=1)       # C2: golden, axis rot 1

# Run simulation
history, coverage = simulate_v6(m, d0, g1, g2, LATE_FRAC_3D)
cover_step = next(i for i, c in enumerate(coverage) if c == m**3)
print(f"Cover step: {cover_step}/{m**3} ({cover_step/m**3*100:.1f}%)")
# → Cover step: 63/125 (50.4%)
```

---

## 4D Implementation (m=5)

**Grid:** 5×5×5×5 = 625 vertices, 4 agents (C0 diagonal + C1 + C2 + C3 golden)

The 4D version extends the tier hierarchy with one extra scout (C3) that is aware of all three leaders' futures.

```python
LATE_FRAC_4D = 0.187

def get_nb4(i, j, k, l, m):
    return [((i+1)%m,j,k,l),(i,(j+1)%m,k,l),
            (i,j,(k+1)%m,l),(i,j,k,(l+1)%m)]

def simulate_4d_v6(m, s0, s1, s2, s3, late_frac=LATE_FRAC_4D):
    total = m**4
    c0s = {v: t for t, v in enumerate(s0)}
    c1s = {v: t for t, v in enumerate(s1)}
    c2s = {v: t for t, v in enumerate(s2)}
    pos = [s0[0], s1[0], s2[0], s3[0]]
    joint = set(pos)
    history = [tuple(pos)]; coverage = [len(joint)]; idx = [0]*4

    for t in range(total):
        new_pos = [None]*4; claimed = set()
        n0 = s0[(t+1) % total]; new_pos[0] = n0; claimed.add(n0)
        c0_step = t + 1; c0_rem = total - c0_step
        thr = int(c0_rem * (1 - late_frac))
        idx[1] = (idx[1]+1)%total
        idx[2] = (idx[2]+1)%total
        idx[3] = (idx[3]+1)%total

        def late(smap, v, cur):
            vi = smap.get(v, -1)
            if vi < 0: return 0
            l = (vi - cur) % total
            return l if l > 0 else 0

        # C1: gap(C0) | C0-late | C0-soon
        nb1 = get_nb4(*pos[1], m); best=-1; bc=0
        for tc in range(4):
            d = nb1[tc]
            if d in joint or d in claimed: s=-1
            else:
                l0 = late(c0s, d, c0_step)
                s = 3 if l0==0 else (2 if l0>=thr else 1)
            if s > best: best=s; bc=tc
        c1n = nb1[bc]; new_pos[1]=c1n; claimed.add(c1n)
        c1_cur = c1s.get(c1n, idx[1])

        # C2: gap | C1-late(C0-skip) | C0-late | soon
        nb2 = get_nb4(*pos[2], m); best=-1; bc=0
        for tc in range(4):
            d = nb2[tc]
            if d in joint or d in claimed: s=-1
            else:
                l0 = late(c0s, d, c0_step)
                l1 = late(c1s, d, c1_cur)
                if l0==0 and l1==0:       s=4
                elif l0==0 and l1>=thr:   s=3
                elif l0==0:               s=2
                elif l0>=thr:             s=2
                else:                     s=1
            if s > best: best=s; bc=tc
        c2n = nb2[bc]; new_pos[2]=c2n; claimed.add(c2n)
        c2_cur = c2s.get(c2n, idx[2])

        # C3: gap | C2-late | C1-late | C0-late | soon
        nb3 = get_nb4(*pos[3], m); best=-1; bc=0
        for tc in range(4):
            d = nb3[tc]
            if d in joint or d in claimed: s=-1
            else:
                l0 = late(c0s, d, c0_step)
                l1 = late(c1s, d, c1_cur)
                l2 = late(c2s, d, c2_cur)
                if l0==0 and l1==0 and l2==0:      s=5
                elif l0==0 and l1==0 and l2>=thr:  s=4
                elif l0==0 and l1==0:               s=3
                elif l0==0 and l1>=thr:             s=3
                elif l0==0:                         s=2
                elif l0>=thr:                       s=2
                else:                               s=1
            if s > best: best=s; bc=tc
        new_pos[3] = nb3[bc]; claimed.add(nb3[bc])

        for c in range(4): pos[c]=new_pos[c]; joint.add(new_pos[c])
        history.append(tuple(pos)); coverage.append(len(joint))
        if len(joint) == total: break

    return history, coverage

# Build 4D sequences
d0 = diagonal4d(m)
g1 = make_golden4d(m, rot=0)
g2 = make_golden4d(m, rot=1)
g3 = make_golden4d(m, rot=2)

history, coverage = simulate_4d_v6(m, d0, g1, g2, g3)
cover_step = next(i for i, c in enumerate(coverage) if c == m**4)
print(f"Cover step: {cover_step}/{m**4} ({cover_step/m**4*100:.1f}%)")
# → Cover step: 354/625 (56.6%)
```

---

## Full Code

See [`simulate_3d.py`](simulate_3d.py) and [`simulate_4d.py`](simulate_4d.py) for complete implementations including:
- Path constructors (diagonal + golden) for 3D and 4D
- V3 (pure existence) and V6 (existence + late-steal) simulators
- Animation code for rendering MP4 visualisations
- Benchmark sweep over `LATE_FRAC` values

### Helper: Adjacency Check

```python
def is_adjacent_torus(v1, v2, m, ndim):
    """Check if two vertices are adjacent on the m^ndim torus."""
    return sum(
        min(abs(v1[a] - v2[a]), m - abs(v1[a] - v2[a]))
        for a in range(ndim)
    ) == 1
```

### Helper: Lateness Function

```python
def lateness(smap, v, cur_step, total):
    """Steps until leader visits v from current position. 0 if never."""
    vi = smap.get(v, -1)
    if vi < 0:
        return 0
    l = (vi - cur_step) % total
    return l if l > 0 else 0
```

---

## Citation

```bibtex
@misc{cooperative-cycle-coverage-2025,
  title  = {Cooperative Cycle Coverage on Toroidal Grids:
             Diagonal Leaders, Golden Scouts, and Late-Stealing},
  year   = {2025},
  note   = {Preprint. Code: github.com/username/cooperative-cycle-coverage}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
