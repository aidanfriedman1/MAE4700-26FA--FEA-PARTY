from ast import literal_eval
from pathlib import Path

import numpy as np


INPUT_FILES = (
    "nodal coordinates.txt",
    "connectivity array.txt",
    "element flexural rigidities.txt",
    "distributed loads.txt",
    "applied loads.txt",
    "essential BCs.txt",
)


def read_literal(filename):
    """Read one Python list literal from the current working directory."""
    try:
        return literal_eval(Path(filename).read_text(encoding="utf-8").strip())
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Cannot read valid Python literal from '{filename}': {exc}") from exc


def _finite_number(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and bool(np.isfinite(value)))


def _node_index(value, count, where):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{where}: node index must be an integer")
    if not 0 <= value < count:
        raise ValueError(f"{where}: node index {value} outside 0 through {count - 1}")


def validate_inputs(coords, connectivity, element_EI, distributed_loads,
                    applied_loads, essential_bcs):
    """Validate raw input lists and ensure that the mesh forms a continuous beam."""
    if not isinstance(coords, list) or len(coords) < 2 or not all(_finite_number(x) for x in coords):
        raise ValueError("nodal coordinates.txt must be a list of at least two finite numbers")
    n = len(coords)
    if len(set(coords)) != n:
        raise ValueError("Nodal coordinates must be distinct")

    if not isinstance(connectivity, list) or len(connectivity) == 0:
        raise ValueError("connectivity array.txt must be a nonempty list")
    for e, nodes in enumerate(connectivity):
        if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
            raise ValueError(f"Element {e}: connectivity must have exactly two nodes")
        i, j = nodes
        _node_index(i, n, f"Element {e}")
        _node_index(j, n, f"Element {e}")
        if coords[j] <= coords[i]:
            raise ValueError(f"Element {e}: right coordinate must exceed left coordinate")
        if e and coords[i] < coords[connectivity[e - 1][0]]:
            raise ValueError("Elements must be listed from left to right")
    # Elements must form one connected beam without gaps, overlaps or unused nodes.
    ordered = sorted(range(n), key=lambda i: coords[i])
    expected = [[ordered[e], ordered[e + 1]] for e in range(n - 1)]
    if connectivity != expected:
        raise ValueError("Connectivity must connect each neighboring coordinate pair once, left to right")

    for name, values, positive in (
        ("element flexural rigidities.txt", element_EI, True),
        ("distributed loads.txt", distributed_loads, False),
    ):
        if not isinstance(values, list) or len(values) != len(connectivity):
            raise ValueError(f"{name} needs one value for each element")
        for e, value in enumerate(values):
            if not _finite_number(value) or (positive and value <= 0):
                raise ValueError(f"{name}, element {e}: value must be " +
                                 ("finite and positive" if positive else "finite"))

    for name, records in (("applied loads.txt", applied_loads),
                          ("essential BCs.txt", essential_bcs)):
        if not isinstance(records, list):
            raise ValueError(f"{name} must be a list")
        for k, record in enumerate(records):
            if not isinstance(record, (list, tuple)) or len(record) != 3:
                raise ValueError(f"{name}, row {k}: expected [node, dof, value]")
            node, dof, value = record
            _node_index(node, n, f"{name}, row {k}")
            if not isinstance(dof, (int, np.integer)) or isinstance(dof, (bool, np.bool_)) or dof not in (0, 1):
                raise ValueError(f"{name}, row {k}: dof must be 0 (w) or 1 (theta)")
            if not _finite_number(value):
                raise ValueError(f"{name}, row {k}: value must be finite")

    seen = set()
    for node, dof, _ in essential_bcs:
        global_dof = 2 * node + dof
        if global_dof in seen:
            raise ValueError(f"essential BCs.txt prescribes global DOF {global_dof} more than once")
        seen.add(global_dof)


def element_dofs(nodes):
    """Global indices for the local ordering [w1, theta1, w2, theta2]."""
    i, j = nodes
    return [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]


def beam_element(x1, x2, EI, p):
    """Return cubic Hermite element stiffness and consistent distributed load."""
    L = x2 - x1
    if L <= 0:
        raise ValueError("Beam element must have x2 > x1")
    if not _finite_number(EI) or EI <= 0:
        raise ValueError("Beam element EI must be finite and positive")
    if not _finite_number(p):
        raise ValueError("Beam element distributed load must be finite")
    ke = (EI / L**3) * np.array([
        [12,       6*L,     -12,       6*L],
        [6*L,   4*L**2,    -6*L,   2*L**2],
        [-12,     -6*L,      12,      -6*L],
        [6*L,   2*L**2,    -6*L,   4*L**2],
    ], dtype=float)
    fe = (p * L / 2) * np.array([1, L / 6, 1, -L / 6], dtype=float)
    return ke, fe


def assemble_system(coords, connectivity, element_EI, distributed_loads, applied_loads):
    """Assemble K and F, accumulating overlapping element and point contributions."""
    ndof = 2 * len(coords)
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)
    for e, nodes in enumerate(connectivity):
        i, j = nodes
        gdofs = element_dofs(nodes)
        ke, fe = beam_element(coords[i], coords[j], element_EI[e], distributed_loads[e])
        for a in range(4):
            A = gdofs[a]
            F[A] += fe[a]
            for b in range(4):
                B = gdofs[b]
                K[A, B] += ke[a, b]
    for node, dof, value in applied_loads:
        F[2*node + dof] += value
    return K, F


def solve_partitioned(K, F, essential_bcs):
    """Solve the free DOFs and return d, reactions, residual, free indices."""
    constrained = np.array([2*node + dof for node, dof, _ in essential_bcs], dtype=int)
    d_E = np.array([value for _, _, value in essential_bcs], dtype=float)
    free = np.setdiff1d(np.arange(len(F)), constrained)
    d = np.zeros(len(F), dtype=float)
    d[constrained] = d_E
    if len(free):
        Kff = K[np.ix_(free, free)]
        Kfe = K[np.ix_(free, constrained)]
        rhs = F[free] - Kfe @ d_E
        try:
            d[free] = np.linalg.solve(Kff, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Constrained stiffness matrix is singular; check supports and connectivity") from exc
        if not np.all(np.isfinite(d)) or not np.allclose(Kff @ d[free], rhs, rtol=1e-8, atol=1e-8):
            raise ValueError("Constrained beam system is singular/ill-conditioned or cannot be solved reliably")
    residual = K @ d - F
    reactions = [float(residual[g]) for g in constrained]
    return d, reactions, residual, free


def beam_fields(x1, x2, EI, d_e, r):
    """Return w, M=EI*w'', V=-EI*w''' at dimensionless position r."""
    L = x2 - x1
    N = np.array([1 - 3*r**2 + 2*r**3,
                  L*(r - 2*r**2 + r**3),
                  3*r**2 - 2*r**3,
                  L*(-r**2 + r**3)])
    N_xx = np.array([(-6 + 12*r)/L**2, (-4 + 6*r)/L,
                     (6 - 12*r)/L**2, (-2 + 6*r)/L])
    N_xxx = np.array([12/L**3, 6/L**2, -12/L**3, 6/L**2])
    w = float(N @ d_e)
    M = float(EI * (N_xx @ d_e))
    V = float(-EI * (N_xxx @ d_e))
    return w, M, V


def postprocess_element(x1, x2, EI, d_e):
    """Recover exactly three values per element, including both endpoints."""
    L = x2 - x1
    rows_w, rows_M, rows_V = [], [], []
    for r in (0.0, 0.5, 1.0):
        x = float(x1 + r * L)
        w, M, V = beam_fields(x1, x2, EI, d_e, r)
        rows_w.append([x, w])
        rows_M.append([x, M])
        rows_V.append([x, V])
    return rows_w, rows_M, rows_V


def solve_beam(coords, connectivity, element_EI, distributed_loads,
               applied_loads, essential_bcs):
    """Complete in-memory FEM pipeline for any valid beam input."""
    validate_inputs(coords, connectivity, element_EI, distributed_loads,
                    applied_loads, essential_bcs)
    K, F = assemble_system(coords, connectivity, element_EI, distributed_loads, applied_loads)
    d, reactions, residual, free = solve_partitioned(K, F, essential_bcs)
    rows_w, rows_M, rows_V = [], [], []
    for e, nodes in enumerate(connectivity):
        i, j = nodes
        d_e = d[element_dofs(nodes)]
        w, M, V = postprocess_element(coords[i], coords[j], element_EI[e], d_e)
        rows_w.extend(w)
        rows_M.extend(M)
        rows_V.extend(V)
    return {
        "K": K, "F": F, "d": d, "reactions": reactions,
        "residual": residual, "free": free,
        "nodal": [[float(value) for value in pair] for pair in d.reshape((-1, 2))],
        "displacement": rows_w, "moment": rows_M, "shear": rows_V,
    }


def write_list(filename, values):
    """Write raw Python list syntax, ensuring built-in Python floats."""
    def plain(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [plain(v) for v in value]
        return float(value)
    Path(filename).write_text(repr(plain(values)) + "\n", encoding="utf-8")


def midspan_displacement(coords, connectivity, element_EI, d):
    """Interpolate w at the geometric midspan, even if it is not a mesh node."""
    xmid = (min(coords) + max(coords)) / 2.0
    for e, (i, j) in enumerate(connectivity):
        x1, x2 = coords[i], coords[j]
        if x1 <= xmid <= x2:
            r = (xmid - x1) / (x2 - x1)
            return beam_fields(x1, x2, element_EI[e],
                               d[element_dofs((i, j))], r)[0]
    raise ValueError("Could not locate the beam midspan in the element mesh")


def refine_mesh(coords, connectivity, element_EI, distributed_loads,
                applied_loads, essential_bcs):
    """Split each existing element into two without changing the physical model."""
    refined_coords = []
    refined_EI = []
    refined_loads = []
    old_to_new = {}

    for e, (i, j) in enumerate(connectivity):
        refined_coords.extend([float(coords[i]), float((coords[i] + coords[j]) / 2)])
        old_to_new[i] = 2*e
        old_to_new[j] = 2*e + 2
        refined_EI.extend([element_EI[e], element_EI[e]])
        refined_loads.extend([distributed_loads[e], distributed_loads[e]])

    refined_coords.append(float(coords[connectivity[-1][1]]))
    refined_conn = [[i, i + 1] for i in range(len(refined_coords) - 1)]
    refined_applied = [[old_to_new[n], dof, value]
                       for n, dof, value in applied_loads]
    refined_bcs = [[old_to_new[n], dof, value]
                   for n, dof, value in essential_bcs]
    return (refined_coords, refined_conn, refined_EI, refined_loads,
            refined_applied, refined_bcs)


def check_numerics(coords, connectivity, element_EI, distributed_loads,
                   applied_loads, essential_bcs, out):
    """Six Section 14 numerical checks, using the user's model where applicable.

    The exact-solution mesh-convergence check applies only when the supplied
    input is a uniformly loaded, constant-EI, simply supported beam.
    No example loading or prescribed test case is built into the solver.
    """
    # 1. Each element stiffness matrix must be symmetric.
    for e, (i, j) in enumerate(connectivity):
        ke, _ = beam_element(coords[i], coords[j], element_EI[e], distributed_loads[e])
        scale = max(1.0, float(np.max(np.abs(ke))))
        if not np.allclose(ke, ke.T, rtol=1e-10, atol=1e-12*scale):
            raise ValueError(f"Numerical check failed: element {e} stiffness is not symmetric")

    # 2. Assembly must preserve global stiffness symmetry.
    K, F, d = out["K"], out["F"], out["d"]
    kscale = max(1.0, float(np.max(np.abs(K))))
    if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12*kscale):
        raise ValueError("Numerical check failed: assembled K is not symmetric")

    # 3. Every free DOF must have negligible equilibrium residual (Kd-F).
    residual = out["residual"][out["free"]]
    loadscale = max(1.0, float(np.max(np.abs(F))),
                    float(np.max(np.abs(K @ d))))
    if not np.allclose(residual, 0.0, rtol=0.0, atol=1e-8*loadscale):
        raise ValueError("Numerical check failed: nonzero free-DOF residual")

    # 4-5. Rigid translation and linear displacement/rotation interpolation.
    # These are mathematical properties of the cubic Hermite interpolation,
    # checked using each element length and EI in the supplied mesh.
    c = 1.0
    for e, (i, j) in enumerate(connectivity):
        x1, x2 = coords[i], coords[j]
        EI = element_EI[e]
        L = x2 - x1
        for r in (0.0, 0.5, 1.0):
            # A constant displacement field has no curvature or shear.
            w, M, V = beam_fields(x1, x2, EI, [c, 0.0, c, 0.0], r)
            mscale = max(1.0, abs(EI*c/L**2))
            vscale = max(1.0, abs(EI*c/L**3))
            if (not np.isclose(w, c, rtol=1e-10, atol=1e-12) or
                abs(M) > 1e-10*mscale or abs(V) > 1e-10*vscale):
                raise ValueError(f"Numerical check failed: element {e} rigid translation")

            # w(x)=a+b*(x-x1): constant slope and zero curvature/shear.
            a, b = 0.5, 0.25
            d_linear = [a, b, a+b*L, b]
            w, M, V = beam_fields(x1, x2, EI, d_linear, r)
            expected_w = a+b*r*L
            wscale = max(1.0, abs(a), abs(b*L))
            mscale = max(1.0, abs(EI*wscale/L**2))
            vscale = max(1.0, abs(EI*wscale/L**3))
            if (not np.isclose(w, expected_w, rtol=1e-10, atol=1e-12*wscale) or
                abs(M) > 1e-10*mscale or abs(V) > 1e-10*vscale):
                raise ValueError(f"Numerical check failed: element {e} linear interpolation")

    print("PASS: element symmetry, global symmetry, free-DOF residual,")
    print("      rigid translation interpolation, linear rotation interpolation")

    # 6. Mesh refinement against the exact midspan deflection applies only to
    # the special class of input models for which that exact answer is known.
    left_node = connectivity[0][0]
    right_node = connectivity[-1][1]
    simple_supports = (
        len(essential_bcs) == 2
        and {2*n + dof for n, dof, _ in essential_bcs}
            == {2*left_node, 2*right_node}
        and all(value == 0.0 for _, _, value in essential_bcs)
    )
    uniform_EI = np.allclose(element_EI, element_EI[0], rtol=1e-12, atol=0)
    uniform_p = np.allclose(distributed_loads, distributed_loads[0],
                           rtol=1e-12, atol=0)
    applicable = (simple_supports and uniform_EI and uniform_p
                  and not applied_loads and distributed_loads[0] != 0
                  and len(connectivity) <= 64)

    if applicable:
        # Compare the current mesh and two successively refined meshes.
        length = max(coords) - min(coords)
        exact = 5*distributed_loads[0]*length**4/(384*element_EI[0])
        values = [midspan_displacement(coords, connectivity, element_EI, d)]
        model = (coords, connectivity, element_EI, distributed_loads,
                 applied_loads, essential_bcs)
        for _ in range(2):
            model = refine_mesh(*model)
            finer = solve_beam(*model)
            values.append(midspan_displacement(model[0], model[1], model[2], finer["d"]))

        errors = [abs(value-exact) for value in values]
        tolerance = 1e-8*max(1.0, abs(exact))
        if (errors[-1] > errors[0] + tolerance or
            errors[-1] > 0.05*abs(exact) + tolerance):
            raise ValueError("Numerical check failed: midspan displacement does not converge")
        print("PASS: midspan mesh refinement toward the analytical solution")
    else:
        print("SKIP: analytical mesh-refinement check (requires uniform-load, ")
        print("      constant-EI, simply supported beam, with <=64 elements)")


def main():
    # Read six input files from the current working directory.
    coords, connectivity, element_EI, distributed_loads, applied_loads, essential_bcs = (
        read_literal(filename) for filename in INPUT_FILES
    )

    # Validate, assemble K/F, solve with prescribed DOFs, recover reactions,
    # and postprocess displacement, bending moment and shear force.
    out = solve_beam(coords, connectivity, element_EI, distributed_loads,
                     applied_loads, essential_bcs)

    # Check the numerical consistency of the current solution.
    check_numerics(coords, connectivity, element_EI, distributed_loads,
                   applied_loads, essential_bcs, out)

    # Write five raw Python lists. Shared-element interface points are retained.
    write_list("nodal displacements.txt", out["nodal"])
    write_list("reaction forces.txt", out["reactions"])
    write_list("displacement.txt", out["displacement"])
    write_list("bending moment.txt", out["moment"])
    write_list("shear force.txt", out["shear"])
    print("Beam solution complete. Wrote all five required output files.")


if __name__ == "__main__":
    main()



