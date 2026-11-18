"""MAE 4700 HW10 Problem 2(b): torsion of square, 1:4 rectangle and circular shafts.

Uses the Problem 2(a) solver (main.solve_problem) and BoxGrid_2D.box_grid_2d unchanged.

    python study_2b.py [circle_mesh_dir]

circle_mesh_dir (default: ./ansysC5_inputs, then ./circle_mesh) must hold the four
standard input files of the provided ANSYS quarter-circle mesh (coordinates in metres).
Outputs (next to this file): results_2b.txt, convergence_2b.png, shapes_2b.png.
Units are SI throughout: m, Pa, N, N*m.
"""
import ast
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from main import solve_problem
from BoxGrid_2D import box_grid_2d

HERE = Path(__file__).resolve().parent
G, ALPHA = 77.2e9, 0.3
GA = G * ALPHA                      # G*alpha = 2.316e10 Pa/m
SIDE_SQUARE = 0.01                  # 1 cm^2 square
SHORT, LONG = 0.005, 0.02           # 1 cm^2 rectangle, aspect ratio 1:4


# ------------------------------------------------------------ exact solutions
def rect_exact(a, b, n_terms=40):
    """Series solution for a solid rectangle, long side a >= short side b.
    Returns (T, tau_max, vM_max); tau_max is at the midpoint of the long side."""
    k = np.arange(1, 2 * n_terms, 2)
    J = a * b**3 * (1/3 - 64*b/(math.pi**5*a) * np.sum(np.tanh(k*math.pi*a/(2*b)) / k**5))
    tau = GA * b * (1 - 8/math.pi**2 * np.sum(1/(k**2*np.cosh(k*math.pi*a/(2*b)))))
    return GA*J, tau, math.sqrt(3)*tau


def circle_exact(R):
    """Solid circle: J = pi R^4/2, tau_max = G alpha R (uniform on the boundary)."""
    return GA*math.pi*R**4/2, GA*R, math.sqrt(3)*GA*R


# --------------------------------------------------------------------- solves
def run_box(Lx, Ly, nodes_per_element, nx, ny):
    nodes, conn, _ = box_grid_2d(0.0, Lx, 0.0, Ly, nodes_per_element, nx, ny)
    tol = 1e-12 * max(Lx, Ly)
    bc = [0.0 if min(x, Lx - x, y, Ly - y) < tol else None for x, y in nodes]  # phi=0 on all exterior nodes
    phi, stress, T = solve_problem(nodes, conn, bc, [GA]*len(conn))
    return dict(nodes=np.array(nodes), conn=conn, phi=np.array(phi), stress=np.array(stress),
                T=T, vm=max(s[2] for s in stress), nel=len(conn), nn=len(nodes))


def run_circle(mesh_dir):
    read = lambda f: ast.literal_eval((Path(mesh_dir)/f).read_text())
    nodes = read("nodal_coordinates.txt"); conn = read("connectivity_array.txt")
    bc = read("essential_BCs.txt"); par = read("problem_parameters.txt")
    phi, stress, Tq = solve_problem(nodes, conn, bc, par)
    nodes = np.array(nodes)
    R = float(np.max(np.hypot(nodes[:, 0], nodes[:, 1])))
    return dict(nodes=nodes, conn=conn, phi=np.array(phi), stress=np.array(stress),
                T=4*Tq, Tq=Tq, vm=max(s[2] for s in stress), nel=len(conn), nn=len(nodes), R=R)


# ------------------------------------------------------------ convergence aids
def richardson(f, r=2.0):
    """f: three finest values (r = refinement ratio). Returns (p, f_inf, GCI_fine) or NaNs."""
    f1, f2, f3 = f[-3:]
    d1, d2 = f2 - f1, f3 - f2
    if d1*d2 <= 0 or d2 == 0:
        return float("nan"), float("nan"), float("nan")
    p = math.log(abs(d1/d2))/math.log(r)
    f_inf = f3 + d2/(r**p - 1)
    gci = 1.25*abs(d2/f3)/(r**p - 1)
    return p, f_inf, gci


def sweep(Lx, Ly, levels, ratio, etype):
    rows = []
    for nx in levels:
        print(f"    solving {'T3' if etype == 3 else 'Q4'} mesh {nx} x {ratio*nx} ...", end=" ", flush=True)
        r = run_box(Lx, Ly, etype, nx, ratio*nx)
        print("done", flush=True)
        rows.append((nx, r["nel"], r["nn"], r["T"], r["vm"]))
    return rows


# ----------------------------------------------------------------------- main
def main():
    quick = "--quick" in sys.argv          # skip the largest meshes (nx > 16)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mesh_dir = Path(args[0]) if args else next(
        (HERE/d for d in ("ansysC5_inputs", "circle_mesh") if (HERE/d).is_dir()), None)
    if mesh_dir is None:
        sys.exit("Circle mesh directory not found: pass it as the first argument.")

    log = []
    def P(s=""):
        print(s, flush=True); log.append(s)

    cases = [
        ("Square 10 x 10 mm",        SIDE_SQUARE, SIDE_SQUARE, [2, 4, 8, 16, 32, 64], 1, rect_exact(SIDE_SQUARE, SIDE_SQUARE)),
        ("Rectangle 5 x 20 mm (1:4)", SHORT,       LONG,        [2, 4, 8, 16, 32],     4, rect_exact(LONG, SHORT)),
    ]
    if quick:
        cases = [(n, a, b, [l for l in lv if l <= 16], r, e) for n, a, b, lv, r, e in cases]
    extrap = {}
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7))
    for c, (name, Lx, Ly, levels, ratio, ex) in enumerate(cases):
        P(f"=== {name}   exact (series): T = {ex[0]:.5f} N*m, max vM = {ex[2]/1e6:.3f} MPa")
        for etype in (3, 4):
            rows = sweep(Lx, Ly, levels, ratio, etype)
            lab = "T3" if etype == 3 else "Q4"
            P(f"  {lab}:  nx   elems  nodes     T [N*m]   vM_max [MPa]   |dT/T|     |dvM/vM|  (change vs previous level)")
            for i, (nx, ne, nn, T, vm) in enumerate(rows):
                dT = abs(T/rows[i-1][3]-1) if i else float("nan")
                dv = abs(vm/rows[i-1][4]-1) if i else float("nan")
                P(f"      {nx:4d} {ne:6d} {nn:6d}  {T:11.5f}  {vm/1e6:12.4f}  {dT:9.2e}  {dv:9.2e}")
            pT, Ti, gT = richardson([r[3] for r in rows]); pv, vi, gv = richardson([r[4] for r in rows])
            P(f"      torque: observed order p = {pT:.2f}, extrapolated T = {Ti:.5f} N*m, GCI(fine) = {gT*100:.3f}%")
            P(f"      max vM: observed order p = {pv:.2f}, extrapolated vM = {vi/1e6:.3f} MPa, GCI(fine) = {gv*100:.2f}%")
            extrap[(name, etype)] = (Ti, vi)
            h = [1/r[0] for r in rows]
            axs[0, c].loglog(h, [abs(r[3]-ex[0])/ex[0] for r in rows], "o-", label=lab)
            axs[1, c].loglog(h, [abs(r[4]-ex[2])/ex[2] for r in rows], "o-", label=lab)
        P()
        axs[0, c].loglog(h, [0.5*x**2 for x in h], "k--", lw=.8, label=r"slope 2")
        axs[1, c].loglog(h, [0.5*x for x in h], "k--", lw=.8, label=r"slope 1")
        axs[0, c].set_title(name + ": torque error", fontsize=9)
        axs[1, c].set_title(name + ": max vM error", fontsize=9)
        for r_ in (0, 1):
            axs[r_, c].set_xlabel("h/L = 1/nx"); axs[r_, c].grid(True, which="both", alpha=.3); axs[r_, c].legend(fontsize=8)
        axs[0, c].set_ylabel("relative error vs series solution")
    fig.tight_layout(); fig.savefig(HERE/"convergence_2b.png", dpi=200); plt.close(fig)

    # ---- circle (provided ANSYS mesh, quarter model)
    circ = run_circle(mesh_dir)
    ce = circle_exact(circ["R"])
    P(f"=== Circle, provided ANSYS quarter mesh ({circ['nel']} Q4 elements, {circ['nn']} nodes, R = {circ['R']*1e3:.3f} mm)")
    P(f"  quarter-model torque = {circ['Tq']:.5f} N*m   ->  full-section torque 4T = {circ['T']:.5f} N*m")
    P(f"  exact: T = {ce[0]:.5f} N*m, max vM = {ce[2]/1e6:.3f} MPa;  FE max vM = {circ['vm']/1e6:.3f} MPa")
    P(f"  (check only) error: T {abs(circ['T']/ce[0]-1)*100:.3f}%, vM {abs(circ['vm']/ce[2]-1)*100:.2f}%")
    P()

    sq = extrap[("Square 10 x 10 mm", 4)]; rc = extrap[("Rectangle 5 x 20 mm (1:4)", 4)]
    shapes = [("Circle", circ["T"], circ["vm"]), ("Square", *sq), ("Rect 1:4", *rc)]
    P("=== Comparison at equal area 1 cm^2 (box shapes: Q4 Richardson-extrapolated; circle: ANSYS mesh)")
    P("  shape        T [N*m]   T/T_circ   vM_max [MPa]   vM/vM_circ   T/vM_max [nN*m/Pa]")
    for nm, T, v in shapes:
        P(f"  {nm:9s} {T:9.3f}   {T/circ['T']:7.3f}   {v/1e6:11.2f}   {v/circ['vm']:9.3f}   {T/v*1e9:10.1f}")
    (HERE/"results_2b.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    # ---- field plots (phi contours, vM at element centroids)
    fine = [("Square", run_box(SIDE_SQUARE, SIDE_SQUARE, 4, 40, 40)),
            ("Rectangle 1:4", run_box(SHORT, LONG, 4, 20, 80)),
            ("Circle (quarter model)", circ)]
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.5))
    for k, (nm, r) in enumerate(fine):
        nd, cn = r["nodes"]*1e3, r["conn"]
        tris = [[e[0], e[1], e[2]] for e in cn] + [[e[0], e[2], e[3]] for e in cn]
        tri = mtri.Triangulation(nd[:, 0], nd[:, 1], tris)
        ax = axs[0, k]; cs = ax.tricontourf(tri, r["phi"]/1e3, 20, cmap="viridis")
        fig.colorbar(cs, ax=ax, label=r"$\phi$ [kN/m]"); ax.set_title(nm); ax.set_aspect("equal")
        cen = np.array([nd[list(e)].mean(axis=0) for e in cn])
        ax = axs[1, k]; sc = ax.scatter(cen[:, 0], cen[:, 1], c=r["stress"][:, 2]/1e6, s=7, cmap="inferno")
        fig.colorbar(sc, ax=ax, label="von Mises at centroids [MPa]"); ax.set_aspect("equal")
        for a in (axs[0, k], axs[1, k]): a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
    fig.tight_layout(); fig.savefig(HERE/"shapes_2b.png", dpi=180); plt.close(fig)

    # ---- bar chart of the comparison
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    names = [s[0] for s in shapes]
    ax[0].bar(names, [s[1] for s in shapes], color=["tab:green", "tab:blue", "tab:orange"]); ax[0].set_ylabel("Torque T [N*m]")
    ax[1].bar(names, [s[2]/1e6 for s in shapes], color=["tab:green", "tab:blue", "tab:orange"]); ax[1].set_ylabel("Max von Mises [MPa]")
    for a in ax: a.grid(axis="y", alpha=.3)
    fig.suptitle("Equal area 1 cm$^2$, G alpha = 2.316e10 Pa/m", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE/"comparison_2b.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
