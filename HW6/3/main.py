
from ast import literal_eval
from pathlib import Path
import numpy as np


def read_literal(filename):
    """Read the raw Python literal contained in an input file."""
    return literal_eval(Path(filename).read_text(encoding="utf-8").strip())


def read_inputs():
    """Load and check the eight standardized problem inputs."""
    x_end = np.asarray(read_literal("nodal_coordinates.txt"), dtype=float)
    end_connectivity = read_literal("connectivity_array.txt")
    nnpe = read_literal("nodes_per_element.txt")
    nq = read_literal("num_quadrature_points.txt")
    p_expr = read_literal("p_function.txt")
    f_expr = read_literal("f_function.txt")
    point_sources = read_literal("point_sources.txt")
    boundary_conditions = read_literal("boundary_conditions.txt")

    if x_end.ndim != 1 or len(x_end) < 2 or not np.all(np.isfinite(x_end)):
        raise ValueError("Provide at least two finite end-node coordinates.")
    if len(np.unique(x_end)) != len(x_end):
        raise ValueError("Input end-node coordinates must be distinct.")
    if type(nnpe) is not int or nnpe not in (2, 3, 4):
        raise ValueError("nodes per element must be 2, 3, or 4.")
    if type(nq) is not int or not 1 <= nq <= 5:
        raise ValueError("num quadrature points must be an integer from 1 to 5.")
    if not isinstance(p_expr, str) or not isinstance(f_expr, str):
        raise ValueError("p and f must be NumPy expression strings.")
    if not isinstance(end_connectivity, list) or not end_connectivity:
        raise ValueError("connectivity array must be a nonempty list.")
    for pair in end_connectivity:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(i) is not int or i < 0 or i >= len(x_end) for i in pair)):
            raise ValueError("Each connectivity entry must contain two valid zero-based indices.")
        if not x_end[pair[1]] > x_end[pair[0]]:
            raise ValueError("Each element must list its left endpoint before its right endpoint.")

    if not isinstance(point_sources, list):
        raise ValueError("point sources must be a list.")
    for source in point_sources:
        if not isinstance(source, (list, tuple)) or len(source) != 2:
            raise ValueError("Each point source must be [location, value].")
        xp, value = (float(v) for v in source)
        if not np.isfinite(xp) or not np.isfinite(value):
            raise ValueError("Point sources must contain finite numbers.")
        if xp < min(x_end) or xp > max(x_end):
            raise ValueError("Point source lies outside the domain.")

    if not isinstance(boundary_conditions, list) or len(boundary_conditions) != 2:
        raise ValueError("Provide one boundary condition for each domain endpoint.")
    for bc in boundary_conditions:
        if (not isinstance(bc, (list, tuple)) or len(bc) != 2
                or bc[0] not in ("essential", "natural")
                or not np.isfinite(float(bc[1]))):
            raise ValueError("Boundary conditions must be ['essential' or 'natural', value].")

    return (x_end, end_connectivity, nnpe, nq, p_expr, f_expr,
            point_sources, boundary_conditions)


def build_computational_mesh(x_end, end_connectivity, nnpe):
    """Keep input end-node indices; append evenly spaced interior nodes."""
    x_all = list(x_end)
    full_connectivity = []
    for left, right in end_connectivity:
        local = [left]
        for x in np.linspace(x_end[left], x_end[right], nnpe)[1:-1]:
            local.append(len(x_all))
            x_all.append(float(x))
        local.append(right)
        full_connectivity.append(local)
    return np.asarray(x_all, dtype=float), full_connectivity


def reference_nodes(nnpe):
    """Equally spaced Lagrange nodes on the reference interval [-1, 1]."""
    return np.linspace(-1.0, 1.0, nnpe)


def shape_functions(xi, nnpe):
    """Lagrange interpolation functions N_i(xi)."""
    nodes = reference_nodes(nnpe)
    N = np.ones(nnpe)
    for i in range(nnpe):
        for j in range(nnpe):
            if j != i:
                N[i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])
    return N


def shape_derivatives(xi, nnpe):
    """Analytical derivatives dN_i/dxi for the Lagrange functions."""
    nodes = reference_nodes(nnpe)
    dN = np.zeros(nnpe)
    for i in range(nnpe):
        for k in range(nnpe):
            if k == i:
                continue
            term = 1.0 / (nodes[i] - nodes[k])
            for j in range(nnpe):
                if j != i and j != k:
                    term *= (xi - nodes[j]) / (nodes[i] - nodes[j])
            dN[i] += term
    return dN


def gauss_rule(nq):
    """Gauss-Legendre points and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(nq)


def evaluate_function(expr, x):
    """Evaluate a supplied NumPy expression, as specified in the code guide."""
    value = np.asarray(eval(expr, {"np": np, "x": x}), dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Nonfinite function value at x={x}: {expr}")
    return float(value) if value.ndim == 0 else value


def element_matrices(xe, nnpe, xi_q, w_q, p_expr, f_expr):
    """Integrate Ke = int(B.T*p*B dx), Fe = int(N.T*f dx)."""
    ke = np.zeros((nnpe, nnpe))
    fe = np.zeros(nnpe)
    for xi, w in zip(xi_q, w_q):
        N = shape_functions(xi, nnpe)
        dN_dxi = shape_derivatives(xi, nnpe)
        J = float(dN_dxi @ xe)
        if J <= 0:
            raise ValueError("Element Jacobian must be positive.")
        B = dN_dxi / J
        x = float(N @ xe)
        p = evaluate_function(p_expr, x)
        f = evaluate_function(f_expr, x)
        ke += np.outer(B, B) * p * J * w
        fe += N * f * J * w
    check_symmetric(ke, "Element stiffness matrix Ke")
    return ke, fe


def add_point_sources(F, x_end, end_connectivity, full_connectivity,
                      nnpe, point_sources):
    """Apply each point source to just one containing element."""
    tol = 1.0e-12 * max(1.0, np.max(np.abs(x_end)))
    for xp, magnitude in point_sources:
        owner = None
        for e, (left, right) in enumerate(end_connectivity):
            if x_end[left] - tol <= xp <= x_end[right] + tol:
                owner = e
                break  # Prevent double counting at a shared endpoint.
        if owner is None:
            raise ValueError(f"No element contains point source at x={xp}.")
        left, right = end_connectivity[owner]
        xi_p = 2.0 * (xp - x_end[left]) / (x_end[right] - x_end[left]) - 1.0
        xi_p = float(np.clip(xi_p, -1.0, 1.0))
        N = shape_functions(xi_p, nnpe)
        F[np.asarray(full_connectivity[owner], dtype=int)] += N * magnitude


def assemble(x_all, x_end, end_connectivity, full_connectivity,
             nnpe, nq, p_expr, f_expr, point_sources):
    """Assemble global stiffness matrix and all source contributions."""
    K = np.zeros((len(x_all), len(x_all)))
    F = np.zeros(len(x_all))
    xi_q, w_q = gauss_rule(nq)
    for gdofs in full_connectivity:
        ids = np.asarray(gdofs, dtype=int)
        xe = x_all[ids]
        ke, fe = element_matrices(xe, nnpe, xi_q, w_q, p_expr, f_expr)
        K[np.ix_(ids, ids)] += ke
        F[ids] += fe
    add_point_sources(F, x_end, end_connectivity, full_connectivity,
                      nnpe, point_sources)
    check_symmetric(K, "Assembled stiffness matrix K")
    return K, F


def apply_natural_bcs(F, x_end, boundary_conditions, p_expr):
    """Add -p(left)*u'(left) and +p(right)*u'(right) to F."""
    left_dof = int(np.argmin(x_end))
    right_dof = int(np.argmax(x_end))
    if boundary_conditions[0][0] == "natural":
        F[left_dof] -= evaluate_function(p_expr, x_end[left_dof]) * float(boundary_conditions[0][1])
    if boundary_conditions[1][0] == "natural":
        F[right_dof] += evaluate_function(p_expr, x_end[right_dof]) * float(boundary_conditions[1][1])
    return left_dof, right_dof


def solve_partitioned(K, F, essential_dofs, prescribed_values):
    """Reuse the standard partitioned FEM linear system."""
    if not essential_dofs:
        raise ValueError("At least one essential BC is needed for a unique solution.")
    essential = np.asarray(essential_dofs, dtype=int)
    free = np.setdiff1d(np.arange(len(F)), essential)
    d = np.zeros(len(F))
    d[essential] = prescribed_values
    if len(free):
        Kff = K[np.ix_(free, free)]
        Kfe = K[np.ix_(free, essential)]
        try:
            d[free] = np.linalg.solve(Kff, F[free] - Kfe @ d[essential])
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular FEM system: check the mesh, quadrature, and BCs.") from exc
    return d


def recover_end_derivatives(x_all, full_connectivity, d, nnpe):
    """Return du/dx at xi=-1 and +1 INSIDE every element."""
    derivatives = []
    for gdofs in full_connectivity:
        ids = np.asarray(gdofs, dtype=int)
        xe = x_all[ids]
        de = d[ids]
        pair = []
        for xi in (-1.0, 1.0):
            dN_dxi = shape_derivatives(xi, nnpe)
            J = float(dN_dxi @ xe)
            pair.append(float((dN_dxi / J) @ de))
        derivatives.append(pair)
    return derivatives


def write_outputs(nodal_solution, reactions, derivatives):
    """Write only plain Python lists, with the exact required filenames."""
    outputs = {
        "nodal_solution.txt": nodal_solution,
        "reactions.txt": reactions,
        "element_derivatives.txt": derivatives,
    }
    for filename, value in outputs.items():
        Path(filename).write_text(repr(value) + "\n", encoding="utf-8")



def check_symmetric(matrix, description):
    """Verify that a local or global FEM stiffness matrix is symmetric."""
    if not np.allclose(matrix, matrix.T, rtol=1e-11, atol=1e-11):
        raise AssertionError(f"{description} is not symmetric.")


def run_unit_tests():
    """Section 18.5: test shape functions, quadrature, and stiffness symmetry.

    Test all three element orders and all five supported Gauss rules.
    These checks are independent of the eight user-supplied input files.
    """
    for nnpe in (2, 3, 4):
        nodes = reference_nodes(nnpe)

        # N_i(xi_j) = delta_ij: each shape function is 1 at its own
        # reference node and 0 at the other reference nodes.
        for j, xi in enumerate(nodes):
            expected = np.zeros(nnpe)
            expected[j] = 1.0
            np.testing.assert_allclose(shape_functions(xi, nnpe),
                                       expected, rtol=0, atol=1e-12)

        # Partition of unity, derivative sum, and finite values.
        for xi in np.linspace(-1.0, 1.0, 21):
            N = shape_functions(xi, nnpe)
            dN_dxi = shape_derivatives(xi, nnpe)
            if not (np.all(np.isfinite(N)) and np.all(np.isfinite(dN_dxi))):
                raise AssertionError("Nonfinite shape function or derivative.")
            np.testing.assert_allclose(np.sum(N), 1.0,
                                       rtol=0, atol=1e-12)
            np.testing.assert_allclose(np.sum(dN_dxi), 0.0,
                                       rtol=0, atol=1e-12)

    # n-point Gauss-Legendre quadrature integrates every polynomial
    # of degree <= 2*n - 1 exactly over [-1, 1].
    for nq in range(1, 6):
        xi_q, w_q = gauss_rule(nq)
        for degree in range(2 * nq):
            actual = np.dot(w_q, xi_q ** degree)
            exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
            np.testing.assert_allclose(actual, exact, rtol=0, atol=1e-12)

        # Verify local Ke and assembled K on a nonuniform two-element mesh,
        # using every supported element order and quadrature rule.
        x_end = np.array([0.0, 0.35, 1.0])
        connectivity = [[0, 1], [1, 2]]
        for nnpe in (2, 3, 4):
            x_all, full_connectivity = build_computational_mesh(
                x_end, connectivity, nnpe)
            for gdofs in full_connectivity:
                xe = x_all[np.asarray(gdofs, dtype=int)]
                ke, _ = element_matrices(xe, nnpe, xi_q, w_q,
                                         "1.0 + 0.25*x", "x**2")
                check_symmetric(ke, "Element stiffness matrix Ke")
            K, _ = assemble(x_all, x_end, connectivity, full_connectivity,
                            nnpe, nq, "1.0 + 0.25*x", "x**2", [])
            check_symmetric(K, "Assembled stiffness matrix K")


def main():
    run_unit_tests()  # Section 18.5 checks before the BVP solve.
    print("Section 18.5 unit tests passed.")

    (x_end, end_connectivity, nnpe, nq, p_expr, f_expr,
     point_sources, boundary_conditions) = read_inputs()

    x_all, full_connectivity = build_computational_mesh(
        x_end, end_connectivity, nnpe)
    K, F = assemble(x_all, x_end, end_connectivity, full_connectivity,
                    nnpe, nq, p_expr, f_expr, point_sources)
    left_dof, right_dof = apply_natural_bcs(
        F, x_end, boundary_conditions, p_expr)

    essential_dofs = []
    prescribed_values = []
    for dof, bc in zip((left_dof, right_dof), boundary_conditions):
        if bc[0] == "essential":
            essential_dofs.append(dof)
            prescribed_values.append(float(bc[1]))

    d = solve_partitioned(K, F, essential_dofs, prescribed_values)
    residual = K @ d - F
    reactions = [
        float(residual[left_dof]) if boundary_conditions[0][0] == "essential" else 0.0,
        float(residual[right_dof]) if boundary_conditions[1][0] == "essential" else 0.0,
    ]
    derivatives = recover_end_derivatives(x_all, full_connectivity, d, nnpe)
    nodal_solution = [float(value) for value in d[:len(x_end)]]
    write_outputs(nodal_solution, reactions, derivatives)


if __name__ == "__main__":
    main()
