import ast
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parent


def read_python_list(filename):
    """Safely read a file whose entire contents are a Python list."""
    with (DIRECTORY / filename).open("r", encoding="utf-8") as file:
        result = ast.literal_eval(file.read())
    if not isinstance(result, list):
        raise ValueError(f"{filename} must contain a Python list.")
    return result


def write_python_list(filename, values):
    """Write an actual Python list, not NumPy's array representation."""
    with (DIRECTORY / filename).open("w", encoding="utf-8") as file:
        file.write(repr(values) + "\n")


def shape_functions(element_type, xi, eta):
    """Return N (n,) and natural derivatives dN/d(xi,eta) (2,n)."""
    if element_type == "T3":
        # Reference triangle: (0,0), (1,0), (0,1).
        N = np.array([1.0 - xi - eta, xi, eta], dtype=float)
        dN_natural = np.array([
            [-1.0, 1.0, 0.0],  # dN/dxi
            [-1.0, 0.0, 1.0],  # dN/deta
        ])
    elif element_type == "Q4":
        # Reference square: nodes (-1,-1),(1,-1),(1,1),(-1,1).
        N = 0.25 * np.array([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ])
        dN_natural = 0.25 * np.array([
            [-(1.0 - eta),  (1.0 - eta),
               (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi),  -(1.0 + xi),
               (1.0 + xi),   (1.0 - xi)],
        ])
    else:
        raise ValueError("Element type must be T3 or Q4.")
    return N, dN_natural


def gauss_rule(element_type):
    """Return (xi, eta, weight) tuples on the parent element."""
    if element_type == "T3":
        return [(1.0 / 3.0, 1.0 / 3.0, 0.5)]

    point = 1.0 / np.sqrt(3.0)
    return [(xi, eta, 1.0)
            for xi in (-point, point)
            for eta in (-point, point)]


def physical_shape_gradients(coords_element, element_type, xi, eta):
    """Compute N, B, and det(J) at a parent-element point.

    J = [[dx/dxi, dy/dxi], [dx/deta, dy/deta]]
    B = [[dN1/dx, ...], [dN1/dy, ...]] = J^(-1) dN_natural.
    """
    N, dN_natural = shape_functions(element_type, xi, eta)
    J = dN_natural @ coords_element
    det_J = float(np.linalg.det(J))
    if not np.isfinite(det_J) or det_J <= 0.0:
        raise ValueError(
            "An element has det(J) <= 0: check counterclockwise "
            "connectivity and element geometry."
        )
    B = np.linalg.solve(J, dN_natural)
    return N, B, det_J


def validate_inputs(nodes, connectivity, essential_BCs, problem_parameters):
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) == 0:
        raise ValueError("nodal_coordinates.txt must contain [[x, y], ...].")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("Nodal coordinates must be finite.")
    if not connectivity:
        raise ValueError("Connectivity must contain at least one element.")

    n_per_element = len(connectivity[0])
    if n_per_element not in (3, 4):
        raise ValueError("Elements must have either 3 (T3) or 4 (Q4) nodes.")
    for element in connectivity:
        if len(element) != n_per_element:
            raise ValueError("All elements must be of the same type.")
        if any(type(index) is not int for index in element):
            raise ValueError("Connectivity node indices must be integers.")
        if len(set(element)) != n_per_element:
            raise ValueError("An element repeats a node index.")
        if any(index < 0 or index >= len(nodes) for index in element):
            raise ValueError("Connectivity references an out-of-range node.")

    if len(essential_BCs) != len(nodes):
        raise ValueError("essential_BCs.txt must have one entry per node.")
    for value in essential_BCs:
        if value is not None:
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("Each essential BC must be a number or None.")

    if len(problem_parameters) != len(connectivity):
        raise ValueError("problem_parameters.txt must have one G*alpha per element.")
    if not np.all(np.isfinite(np.asarray(problem_parameters, dtype=float))):
        raise ValueError("G*alpha parameters must be finite numbers.")
    return "T3" if n_per_element == 3 else "Q4"


def solve_problem(nodal_coordinates, connectivity_array, essential_BCs,
                  problem_parameters):
    """Solve torsion problem; return nodal phi, centroid stresses, torque.

    The right-hand side is +2*G*alpha because -laplacian(phi)=2*G*alpha.
    No extra minus sign is applied to the source vector.
    """
    nodes = np.asarray(nodal_coordinates, dtype=float)
    connectivity = connectivity_array
    element_type = validate_inputs(nodes, connectivity, essential_BCs,
                                   problem_parameters)
    n_nodes = len(nodes)
    K_global = np.zeros((n_nodes, n_nodes), dtype=float)
    f_global = np.zeros(n_nodes, dtype=float)

    # Assemble elemental contributions using the specified quadrature.
    for element_id, element in enumerate(connectivity):
        indices = np.asarray(element, dtype=int)
        coords_element = nodes[indices]
        n_local = len(element)
        K_element = np.zeros((n_local, n_local), dtype=float)
        f_element = np.zeros(n_local, dtype=float)
        G_alpha = float(problem_parameters[element_id])

        for xi, eta, weight in gauss_rule(element_type):
            N, B, det_J = physical_shape_gradients(
                coords_element, element_type, xi, eta)
            d_area = weight * det_J
            K_element += (B.T @ B) * d_area
            f_element += N * (2.0 * G_alpha) * d_area

        K_global[np.ix_(indices, indices)] += K_element
        f_global[indices] += f_element

    # Enforce essential conditions using partitioned elimination:
    # K_ff phi_f = f_f - K_fc phi_c.
    fixed = np.array([value is not None for value in essential_BCs])
    free = ~fixed
    phi = np.zeros(n_nodes, dtype=float)
    if np.any(fixed):
        phi[fixed] = np.asarray(
            [value for value in essential_BCs if value is not None],
            dtype=float)
    if np.any(free):
        if not np.any(fixed):
            raise ValueError(
                "No essential boundary conditions: pure-Neumann problem "
                "has an undetermined additive constant."
            )
        K_ff = K_global[np.ix_(free, free)]
        right_hand_side = f_global[free] - K_global[np.ix_(free, fixed)] @ phi[fixed]
        try:
            phi[free] = np.linalg.solve(K_ff, right_hand_side)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The reduced system is singular. Check disconnected nodes, "
                "mesh connectivity, and essential boundary conditions."
            ) from exc

    # Postprocess at the (iso)parametric centroid and integrate torque.
    element_stresses = []
    total_torque = 0.0
    for element in connectivity:
        indices = np.asarray(element, dtype=int)
        coords_element = nodes[indices]
        phi_element = phi[indices]
        center_xi, center_eta = ((1.0 / 3.0, 1.0 / 3.0)
                                 if element_type == "T3" else (0.0, 0.0))
        _, B_center, _ = physical_shape_gradients(
            coords_element, element_type, center_xi, center_eta)
        dphi_dx, dphi_dy = B_center @ phi_element
        tau_xz = float(dphi_dy)
        tau_yz = float(-dphi_dx)
        sigma_vm = float(np.sqrt(3.0 * (tau_xz**2 + tau_yz**2)))
        element_stresses.append([tau_xz, tau_yz, sigma_vm])

        for xi, eta, weight in gauss_rule(element_type):
            N, _, det_J = physical_shape_gradients(
                coords_element, element_type, xi, eta)
            phi_at_gauss = float(N @ phi_element)
            total_torque += 2.0 * phi_at_gauss * weight * det_J

    return phi.tolist(), element_stresses, float(total_torque)


def main():
    nodes = read_python_list("nodal_coordinates.txt")
    connectivity = read_python_list("connectivity_array.txt")
    essential_BCs = read_python_list("essential_BCs.txt")
    problem_parameters = read_python_list("problem_parameters.txt")

    nodal_phi, element_stresses, torque = solve_problem(
        nodes, connectivity, essential_BCs, problem_parameters)

    write_python_list("nodal_phi.txt", nodal_phi)
    write_python_list("element_stresses.txt", element_stresses)
    write_python_list("torsion_results.txt", [torque])


if __name__ == "__main__":
    main()
