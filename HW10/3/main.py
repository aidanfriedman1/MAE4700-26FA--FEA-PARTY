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


def validate_inputs(nodes, connectivity, essential_BCs, problem_parameters, convection_BCs, point_sources):
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
        raise ValueError("problem_parameters.txt must have one conductivity per element.")
    
    conductivities = np.asarray(problem_parameters, dtype=float)
    if not np.all(np.isfinite(conductivities)) or np.any(conductivities <= 0):
        raise ValueError("Element conductivities must be finite and positive.")

    if len(point_sources) != len(nodes):
     raise ValueError("point_sources.txt must have one value per node.")
    if not np.all(np.isfinite(np.asarray(point_sources, dtype=float))):
        raise ValueError("Point sources must contain finite numbers.")

    edge_counts = {}

    for element in connectivity:
        for i in range(len(element)):
            edge = tuple(sorted((element[i], element[(i + 1) % len(element)])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    used_edges = set()

    for bc in convection_BCs:
        if len(bc) != 4:
            raise ValueError("Each convection BC must be [node1, node2, h, T_inf].")

        node1, node2, h, T_inf = bc

        if type(node1) is not int or type(node2) is not int:
            raise ValueError("Convection edge node indices must be integers.")

        if not np.all(np.isfinite([h, T_inf])) or h < 0:
            raise ValueError("Convection parameters must be finite and h nonnegative.")

        edge = tuple(sorted((node1, node2)))

        if edge_counts.get(edge, 0) != 1:
            raise ValueError("Convection must be applied to an exterior element edge.")

        if edge in used_edges:
            raise ValueError("Duplicate convection edge.")

        used_edges.add(edge)

    return "T3" if n_per_element == 3 else "Q4"


def solve_problem(nodal_coordinates, connectivity_array, essential_BCs,
                  problem_parameters, convection_BCs, point_sources):
    """Solve torsion problem; return nodal phi, centroid stresses, torque.

    The right-hand side is +2*G*alpha because -laplacian(phi)=2*G*alpha.
    No extra minus sign is applied to the source vector.
    """
    nodes = np.asarray(nodal_coordinates, dtype=float)
    connectivity = connectivity_array
    element_type = validate_inputs(nodes, connectivity, essential_BCs,
                                   problem_parameters, convection_BCs, point_sources)
    n_nodes = len(nodes)
    K_global = np.zeros((n_nodes, n_nodes), dtype=float)
    f_global = np.zeros(n_nodes, dtype=float)

    # Assemble elemental contributions using the specified quadrature.
    for element_id, element in enumerate(connectivity):
        indices = np.asarray(element, dtype=int)
        coords_element = nodes[indices]
        n_local = len(element)
        K_element = np.zeros((n_local, n_local), dtype=float)
        kappa = float(problem_parameters[element_id])

        for xi, eta, weight in gauss_rule(element_type):
            N, B, det_J = physical_shape_gradients(
                coords_element, element_type, xi, eta)
            d_area = weight * det_J
            K_element += kappa * (B.T @ B) * d_area 

        K_global[np.ix_(indices, indices)] += K_element

    # Assemble point source contributions.
    for node1, node2, h, T_inf in convection_BCs:
        indices = np.array([node1, node2], dtype=int)

        edge_length = float(np.linalg.norm(nodes[node2] - nodes[node1]))

        K_edge = (h * edge_length / 6.0) * np.array([[2, 1], [1, 2]], dtype=float)

        f_edge = (h * T_inf * edge_length / 2.0) * np.array([1, 1], dtype=float)

        K_global[np.ix_(indices, indices)] += K_edge
        f_global[indices] += f_edge

    # Add prescribed nodal point sources. 
    f_global += np.asarray(point_sources, dtype=float)

    # Enforce essential conditions using partitioned elimination:
    # K_ff T_f = f_f - K_fc T_c.
    fixed = np.array([value is not None for value in essential_BCs])
    free = ~fixed
    temperatures = np.zeros(n_nodes, dtype=float)
    if np.any(fixed):
        temperatures[fixed] = np.asarray(
            [value for value in essential_BCs if value is not None],
            dtype=float)
    if np.any(free):
        if not np.any(fixed) and not any (h > 0 for _, _, h, _ in convection_BCs):
            raise ValueError(
                "The Problem has no prescribed essential boundary conditions or "
                "positive convection coefficients."
            )
        
        K_ff = K_global[np.ix_(free, free)]
        right_hand_side = f_global[free] - K_global[np.ix_(free, fixed)] @ temperatures[fixed]
        try:
            temperatures[free] = np.linalg.solve(K_ff, right_hand_side)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The reduced system is singular. Check disconnected nodes, "
                "mesh connectivity, and essential boundary conditions."
            ) from exc

    # Postprocess heat-flux compoenents at each element's (iso)parametric centroid and integrate total heat transfer.
    element_heat_fluxes = []
  
    for element_id, element in enumerate(connectivity):
        indices = np.asarray(element, dtype=int)
        coords_element = nodes[indices]
        temperatures_element = temperatures[indices]
        kappa = float(problem_parameters[element_id])

        center_xi, center_eta = ((1.0 / 3.0, 1.0 / 3.0)
                                 if element_type == "T3" else (0.0, 0.0))
        
        _, B_center, _ = physical_shape_gradients(
            coords_element, element_type, center_xi, center_eta)
        
        q_x, q_y = -kappa * (B_center @ temperatures_element)

        element_heat_fluxes.append([float(q_x), float(q_y)])

    return temperatures.tolist(), element_heat_fluxes


def main():
    nodes = read_python_list("nodal_coordinates.txt")
    connectivity = read_python_list("connectivity_array.txt")
    problem_parameters = read_python_list("problem_parameters.txt")
    essential_BCs = read_python_list("essential_BCs.txt")
    convection_BCs = read_python_list("convection_BCs.txt")
    point_sources = read_python_list("point_sources.txt")

    nodal_temperatures, element_heat_fluxes = solve_problem(
        nodes, connectivity, essential_BCs, problem_parameters,
        convection_BCs, point_sources)

    write_python_list("nodal_temperatures.txt", nodal_temperatures)
    write_python_list("element_heat_fluxes.txt", element_heat_fluxes)


if __name__ == "__main__":
    main()
