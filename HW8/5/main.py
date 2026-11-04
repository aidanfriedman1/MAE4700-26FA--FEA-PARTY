import os
# Keep BLAS single-threaded (set BEFORE importing numpy); avoids some silent Windows crashes.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import faulthandler
faulthandler.enable()  # prints a Python traceback even on a native crash

import ast
import traceback
from pathlib import Path

import numpy as np

# All seven input files must be in the same folder as this main.py file.
HERE = Path(__file__).resolve().parent


def mm(a, b):
    """Matrix product without BLAS."""
    return np.einsum("ij,jk->ik", a, b)


def mv(a, x):
    """Matrix-vector product without BLAS."""
    return np.einsum("ij,j->i", a, x)


def solve_dense(A, b):
    """Gaussian elimination with partial pivoting (no LAPACK). Returns x, min |pivot|/max |A|."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    amax = np.max(np.abs(A))
    minpiv = np.inf
    for k in range(n):
        m = k + int(np.argmax(np.abs(A[k:, k])))
        if m != k:
            A[[k, m]] = A[[m, k]]
            b[[k, m]] = b[[m, k]]
        piv = A[k, k]
        minpiv = min(minpiv, abs(piv))
        if abs(piv) < 1e-12 * amax:
            raise ValueError("Reduced stiffness matrix is singular; check supports and connectivity")
        f = A[k+1:, k] / piv
        A[k+1:, k:] -= f[:, None] * A[k, k:][None, :]
        b[k+1:] -= f * b[k]
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = (b[k] - np.dot(A[k, k+1:], x[k+1:])) / A[k, k]
    return x, minpiv / amax


def read_list(filename):
    path = HERE / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def element_geometry(x1, x2):
    delta = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
    L = float(np.linalg.norm(delta))
    if L <= 0.0:
        raise ValueError("An element has zero length.")
    return L, delta[0] / L, delta[1] / L


def dofs_for(nodes):
    i, j = nodes
    return np.array([3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2], dtype=int)


def transformation(c, s):
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.zeros((6, 6))
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


def local_stiffness(L, EA_over_L, EI):
    k = np.zeros((6, 6), dtype=float)
    ka = EA_over_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k[np.ix_([0, 3], [0, 3])] = ka
    kb = (EI / L**3) * np.array([
        [12.0, 6.0*L, -12.0, 6.0*L],
        [6.0*L, 4.0*L**2, -6.0*L, 2.0*L**2],
        [-12.0, -6.0*L, 12.0, -6.0*L],
        [6.0*L, 2.0*L**2, -6.0*L, 4.0*L**2],
    ])
    k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = kb
    return k


def local_load(L, p):
    # p is constant load per unit length in the local +y direction.
    return (p * L / 2.0) * np.array([0.0, 1.0, L/6.0, 0.0, 1.0, -L/6.0])


def interpolate(L, c, s, dl, xi):
    """Global displacement (ux, uy) at natural coordinate xi in [0,1] along an element."""
    ulocal = (1.0 - xi)*dl[0] + xi*dl[3]
    N1 = 1.0 - 3.0*xi**2 + 2.0*xi**3
    N2 = L*(xi - 2.0*xi**2 + xi**3)
    N3 = 3.0*xi**2 - 2.0*xi**3
    N4 = L*(-xi**2 + xi**3)
    wlocal = N1*dl[1] + N2*dl[2] + N3*dl[4] + N4*dl[5]
    return c*ulocal - s*wlocal, s*ulocal + c*wlocal


def main():
    print("Reading inputs...", flush=True)
    coords = np.asarray(read_list("nodal_coordinates.txt"), dtype=float)
    conn = read_list("connectivity_array.txt")
    EA_L = np.asarray(read_list("element_stiffnesses.txt"), dtype=float)
    EI = np.asarray(read_list("element_flexural_rigidities.txt"), dtype=float)
    p = np.asarray(read_list("distributed_loads.txt"), dtype=float)
    nodal_loads = np.asarray(read_list("external_nodal_forces.txt"), dtype=float)
    bcs = read_list("displacement_BCs.txt")

    nnode = len(coords)
    nelem = len(conn)
    if coords.shape != (nnode, 2) or nnode < 2:
        raise ValueError("nodal_coordinates.txt must be a list of [x,y] coordinates")
    if EA_L.shape != (nelem,) or EI.shape != (nelem,) or p.shape != (nelem,):
        raise ValueError("Each element input file must contain exactly one value per element")
    if nodal_loads.shape != (nnode, 3):
        raise ValueError("external_nodal_forces.txt must have one [Fx,Fy,M] row per node")
    if len(bcs) != nnode or any(len(row) != 3 for row in bcs):
        raise ValueError("displacement_BCs.txt must have one [ux,uy,theta] row per node")
    if np.any(EA_L <= 0) or np.any(EI <= 0):
        raise ValueError("All EA/L and EI values must be positive")
    for e, pair in enumerate(conn):
        if len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= nnode:
            raise ValueError(f"Invalid connectivity for element {e}: {pair}")

    print(f"Input validation: PASS ({nnode} nodes, {nelem} frame elements)", flush=True)

    # ---------------- Assembly (each element assembled exactly once) ----------------
    ndof = 3 * nnode
    K = np.zeros((ndof, ndof))
    Fdist = np.zeros(ndof)
    element_data = []

    for e, nodes in enumerate(conn):
        i, j = nodes
        L, c, s = element_geometry(coords[i], coords[j])
        T = transformation(c, s)
        kl = local_stiffness(L, EA_L[e], EI[e])
        fl = local_load(L, p[e])
        kg = mm(mm(T.T, kl), T)
        fg = mv(T.T, fl)
        edofs = dofs_for(nodes)
        K[np.ix_(edofs, edofs)] += kg
        Fdist[edofs] += fg
        element_data.append((L, c, s, T, kl, fl, edofs))
    print("Assembly complete.", flush=True)

    # ---------------- Boundary conditions and solve ----------------
    F = nodal_loads.reshape(-1) + Fdist
    u = np.zeros(ndof)
    fixed, free = [], []
    for node, row in enumerate(bcs):
        for local_dof, value in enumerate(row):
            gdof = 3*node + local_dof
            if value is None:
                free.append(gdof)
            else:
                fixed.append(gdof)
                u[gdof] = float(value)
    fixed = np.asarray(fixed, dtype=int)
    free = np.asarray(free, dtype=int)
    if len(free) == 0:
        raise ValueError("No free degrees of freedom: check displacement_BCs.txt")

    Kff = K[np.ix_(free, free)]
    rhs = F[free] - mv(K[np.ix_(free, fixed)], u[fixed])
    print("Solving system...", flush=True)
    u[free], pivrat = solve_dense(Kff, rhs)
    print(f"Solve complete (min pivot / max stiffness = {pivrat:.3e})", flush=True)

    reactions = mv(K, u) - F
    reactions[free] = 0.0

    # ---------------- Checks ----------------
    residual = (mv(K, u) - F)[free]
    scale = max(1.0, np.max(np.abs(F)))
    if np.max(np.abs(residual)) > 1e-8 * scale:
        raise RuntimeError(f"Free-DOF residual is not zero: max = {np.max(np.abs(residual)):.6e}")
    if not np.allclose(K, K.T, atol=1e-7, rtol=1e-10):
        raise RuntimeError("Global stiffness matrix is not symmetric")

    # ---------------- Element end forces: k*d - f ----------------
    local_end_forces = []
    for L, c, s, T, kl, fl, edofs in element_data:
        dlocal = mv(T, u[edofs])
        local_end_forces.append((mv(kl, dlocal) - fl).tolist())

    disp = u.reshape(nnode, 3)
    react = reactions.reshape(nnode, 3)
    endf = np.asarray(local_end_forces, dtype=float)

    (HERE / "nodal_displacements.txt").write_text(repr(disp.tolist()), encoding="utf-8")
    (HERE / "reaction_forces.txt").write_text(repr(react.tolist()), encoding="utf-8")
    (HERE / "internal_forces.txt").write_text(repr(local_end_forces), encoding="utf-8")

    np.set_printoptions(precision=6, suppress=False, linewidth=140)
    print("\nCHECKS: stiffness symmetry PASS; free-DOF residual PASS")
    print("\nNodal displacements [ux (m), uy (m), theta (rad)]:")
    for n, row in enumerate(disp, start=1):
        print(f"Node {n}: {row}")
    print("\nSupport reactions [Rx (N), Ry (N), Mz (N m)]:")
    for n, row in enumerate(react, start=1):
        print(f"Node {n}: {row}")
    print("\nMember end forces [N1, V1, M1, N2, V2, M2]:")
    for e, row in enumerate(endf, start=1):
        print(f"Element {e}: {row}")
        print(f"  max |axial| = {max(abs(row[0]), abs(row[3])):.6g} N")
        print(f"  max |shear| = {max(abs(row[1]), abs(row[4])):.6g} N")
        print(f"  end moments = {row[2]:.6g}, {row[5]:.6g} N m")

    # ---------------- Max interpolated displacement ----------------
    best = {"mag": -1.0, "element": None, "xi": None, "xy": None}
    for e, (nodes, data) in enumerate(zip(conn, element_data), start=1):
        L, c, s, T, kl, fl, edofs = data
        i, j = nodes
        dl = mv(T, u[edofs])
        for xi in np.linspace(0.0, 1.0, 1001):
            ux, uy = interpolate(L, c, s, dl, xi)
            mag = float(np.hypot(ux, uy))
            if mag > best["mag"]:
                best = {"mag": mag, "element": e, "xi": float(xi),
                        "xy": (coords[i] + xi*(coords[j]-coords[i])).tolist()}
    print(f"\nMaximum interpolated translational displacement: {best['mag']:.8e} m "
          f"on element {best['element']} at xi = {best['xi']:.4f}, "
          f"global position = {best['xy']} m")

    # ---------------- Optional deformed-shape plot ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        umax = max(np.max(np.abs(disp[:, :2])), 1e-300)
        span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
        scale_plot = 0.1 * span / umax
        fig, ax = plt.subplots(figsize=(6, 7))
        for (nodes, data) in zip(conn, element_data):
            L, c, s, T, kl, fl, edofs = data
            i, j = nodes
            ax.plot(*zip(coords[i], coords[j]), "--", color="gray", lw=1)
            dl = mv(T, u[edofs])
            pts = []
            for xi in np.linspace(0, 1, 60):
                ux, uy = interpolate(L, c, s, dl, xi)
                base = coords[i] + xi*(coords[j]-coords[i])
                pts.append([base[0] + scale_plot*ux, base[1] + scale_plot*uy])
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], "b-", lw=2)
        for n, (x, y) in enumerate(coords, start=1):
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(-10, 4))
        ax.set_aspect("equal")
        ax.set_title(f"Deformed shape (displacements x {scale_plot:.0f})")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.savefig(HERE / "deformed_shape.png", dpi=150, bbox_inches="tight")
        print("Wrote deformed_shape.png")
    except Exception as exc:  # plotting is optional
        print(f"(Skipped deformed-shape plot: {exc})")

    print("Wrote nodal_displacements.txt, reaction_forces.txt, and internal_forces.txt")
    print(f"Output directory: {HERE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nSOLVER FAILED. Full error details:", flush=True)
        traceback.print_exc()
        raise
