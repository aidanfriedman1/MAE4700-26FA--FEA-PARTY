import ast
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parent


def read_input(filename):
    with (DIRECTORY / filename).open() as file:
        return ast.literal_eval(file.read())


def thermal_inputs_present():
    return (DIRECTORY / "thermal_expansion_coeff.txt").exists() and (
        DIRECTORY / "temperature_change.txt"
    ).exists()


def validate_inputs(coords, connectivity, k_values, loads, bcs, alpha=None, delta_t=None):
    # Coordinate width determines whether this is a 2D or 3D problem.
    if coords.ndim != 2 or coords.shape[1] not in (2, 3) or not len(coords):
        raise ValueError("nodal coordinates must have shape (num_nodes, 2 or 3)")
    num_nodes, ndim = coords.shape

    if not np.all(np.isfinite(coords)):
        raise ValueError("nodal coordinates must be finite")
    if not isinstance(connectivity, list) or not connectivity:
        raise ValueError("connectivity array must contain element [i, j] lists")
    if len(k_values) != len(connectivity) or not np.all(np.isfinite(k_values)) or np.any(k_values <= 0):
        raise ValueError("element stiffnesses must contain one positive finite value per element")
    if loads.shape != (num_nodes, ndim) or not np.all(np.isfinite(loads)):
        raise ValueError("external nodal forces must match coordinate dimensions")
    if not isinstance(bcs, list) or len(bcs) != num_nodes:
        raise ValueError("displacement BCs must contain one list per node")

    for node, values in enumerate(bcs):
        if not isinstance(values, list) or len(values) != ndim:
            raise ValueError(f"node {node} BC must have {ndim} entries")
        for value in values:
            if value is not None and (
                not isinstance(value, (int, float)) or not np.isfinite(value)
            ):
                raise ValueError("BC entries must be finite numbers or None")

    for e, nodes in enumerate(connectivity):
        if (
            not isinstance(nodes, list)
            or len(nodes) != 2
            or any(type(n) is not int or n < 0 or n >= num_nodes for n in nodes)
        ):
            raise ValueError(f"element {e} must connect two valid zero-based integer nodes")
        element_geometry(coords[nodes[0]], coords[nodes[1]])

    if alpha is not None or delta_t is not None:
        if alpha is None or delta_t is None:
            raise ValueError(
                "Both thermal_expansion_coeff.txt and temperature_change.txt "
                "must be present for thermal analysis."
            )
        if len(alpha) != len(connectivity) or len(delta_t) != len(connectivity):
            raise ValueError(
                "Thermal expansion coefficient and temperature change lists "
                "must contain one value per element."
            )
        if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(delta_t)):
            raise ValueError("Thermal inputs must contain finite values.")

    return num_nodes, ndim


def node_dofs(node, ndim):
    return np.arange(node * ndim, (node + 1) * ndim, dtype=int)


def element_dofs(nodes, ndim):
    return np.concatenate((node_dofs(nodes[0], ndim), node_dofs(nodes[1], ndim)))


def element_geometry(x1, x2):
    dx = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
    length = np.linalg.norm(dx)
    if length <= 0:
        raise ValueError("Truss element has zero length")
    return length, dx / length


def element_operator(x1, x2):
    length, direction = element_geometry(x1, x2)
    B = np.concatenate((-direction, direction))
    return length, direction, B


def element_stiffness(x1, x2, k):
    _, _, B = element_operator(x1, x2)
    return k * np.outer(B, B)


def assemble_global_stiffness(num_nodes, ndim, coords, connectivity, k_values):
    K = np.zeros((num_nodes * ndim, num_nodes * ndim))
    for e, nodes in enumerate(connectivity):
        gdofs = element_dofs(nodes, ndim)
        ke = element_stiffness(coords[nodes[0]], coords[nodes[1]], k_values[e])
        if not np.allclose(ke, ke.T):
            raise ValueError(f"element {e} stiffness is not symmetric")
        K[np.ix_(gdofs, gdofs)] += ke
    return K


def build_force_vector(loads):
    return loads.reshape(-1).copy()


def assemble_thermal_force(num_nodes, ndim, coords, connectivity, k_values, alpha, delta_t):
    # For each element:
    # N = k * (B u - alpha*DeltaT*L)
    # Therefore K u = F_external + F_thermal,
    # where F_thermal = sum[k * B^T * alpha*DeltaT*L].
    F_thermal = np.zeros(num_nodes * ndim)

    for e, nodes in enumerate(connectivity):
        length, _, B = element_operator(coords[nodes[0]], coords[nodes[1]])
        gdofs = element_dofs(nodes, ndim)
        F_thermal[gdofs] += (
            k_values[e] * B * alpha[e] * delta_t[e] * length
        )

    return F_thermal


def solve_system(K, F, bcs, ndim):
    u = np.zeros(len(F))
    fixed = []
    free = []

    for node, values in enumerate(bcs):
        for direction, value in enumerate(values):
            dof = node * ndim + direction
            if value is None:
                free.append(dof)
            else:
                fixed.append(dof)
                u[dof] = float(value)

    fixed = np.array(fixed, dtype=int)
    free = np.array(free, dtype=int)

    if len(free):
        K_ff = K[np.ix_(free, free)]
        K_fe = K[np.ix_(free, fixed)]
        condition_number = np.linalg.cond(K_ff)
        print("K_FF condition number:", condition_number)

        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise ValueError(
                "K_FF is singular or ill-conditioned. "
                "Check constraints, mechanisms, or disconnected nodes."
            )

        try:
            u[free] = np.linalg.solve(
                K_ff, F[free] - K_fe @ u[fixed]
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "K_FF is singular. Check constraints and mechanisms."
            ) from exc

    return u, free, fixed


def recover_reactions(K, effective_F, u, fixed):
    # effective_F = external_F + thermal_equivalent_F
    # Thus K*u - effective_F gives the support reactions.
    reactions = np.zeros(len(effective_F))
    reactions[fixed] = (K @ u - effective_F)[fixed]
    return reactions


def recover_element_results(
    coords, connectivity, k_values, u, ndim, alpha=None, delta_t=None
):
    forces = []
    stresses = []
    total_strains = []
    thermal_strains = []
    mechanical_strains = []

    thermal = alpha is not None and delta_t is not None

    for e, nodes in enumerate(connectivity):
        length, _, B = element_operator(coords[nodes[0]], coords[nodes[1]])
        ue = u[element_dofs(nodes, ndim)]

        delta_L = float(B @ ue)
        total_strain = delta_L / length
        thermal_strain = float(alpha[e] * delta_t[e]) if thermal else 0.0
        mechanical_strain = total_strain - thermal_strain

        # k = EA/L, so N = k*L*mechanical_strain.
        force = float(k_values[e] * length * mechanical_strain)

        forces.append(force)
        total_strains.append(total_strain)
        thermal_strains.append(thermal_strain)
        mechanical_strains.append(mechanical_strain)

    return (
        forces,
        np.asarray(total_strains),
        np.asarray(thermal_strains),
        np.asarray(mechanical_strains),
    )


def write_output(filename, values):
    if isinstance(values, np.ndarray):
        values = values.tolist()
    (DIRECTORY / filename).write_text(str(values))


def check_solution(K, effective_F, u, reactions, free, ndim):
    if not np.allclose(K, K.T):
        raise ValueError("Global stiffness symmetry check failed")

    residual = K @ u - effective_F
    if not np.allclose(residual[free], 0, atol=1e-7):
        raise ValueError("Free DOF residual check failed")

    if not np.allclose(
        np.sum((effective_F + reactions).reshape(-1, ndim), axis=0),
        0,
        atol=1e-7,
    ):
        raise ValueError("Global force equilibrium check failed")

    print("Stiffness symmetry, free DOF residual, and global equilibrium: PASS")


def main():
    coords = np.asarray(read_input("nodal_coordinates.txt"), dtype=float)
    connectivity = read_input("connectivity_array.txt")
    k_values = np.asarray(read_input("element_stiffnesses.txt"), dtype=float)
    loads = np.asarray(read_input("external_nodal_forces.txt"), dtype=float)
    bcs = read_input("displacement_BCs.txt")

    thermal = thermal_inputs_present()

    if thermal:
        alpha = np.asarray(read_input("thermal_expansion_coeff.txt"), dtype=float)
        delta_t = np.asarray(read_input("temperature_change.txt"), dtype=float)
    else:
        alpha = None
        delta_t = None

    num_nodes, ndim = validate_inputs(
        coords, connectivity, k_values, loads, bcs, alpha, delta_t
    )

    analysis_type = "thermal + mechanical" if thermal else "mechanical only"
    print(
        f"Input validation: PASS ({ndim}D, {num_nodes} nodes, "
        f"{len(connectivity)} elements, {analysis_type})"
    )

    F_external = build_force_vector(loads)
    K = assemble_global_stiffness(
        num_nodes, ndim, coords, connectivity, k_values
    )

    if thermal:
        F_thermal = assemble_thermal_force(
            num_nodes, ndim, coords, connectivity, k_values, alpha, delta_t
        )
    else:
        F_thermal = np.zeros_like(F_external)

    effective_F = F_external + F_thermal

    u, free, fixed = solve_system(K, effective_F, bcs, ndim)
    reactions = recover_reactions(K, effective_F, u, fixed)

    (
        forces,
        total_strains,
        thermal_strains,
        mechanical_strains,
    ) = recover_element_results(
        coords, connectivity, k_values, u, ndim, alpha, delta_t
    )

    check_solution(K, effective_F, u, reactions, free, ndim)

    displacements = u.reshape(num_nodes, ndim)
    reaction_forces = reactions.reshape(num_nodes, ndim)

    write_output("nodal_displacements.txt", displacements)
    write_output("reaction_forces.txt", reaction_forces)
    write_output("internal_forces.txt", forces)

    # Additional Q4 outputs. These do not replace the three required Q3 outputs.
    write_output("element_total_strains.txt", total_strains)
    write_output("element_thermal_strains.txt", thermal_strains)
    write_output("element_mechanical_strains.txt", mechanical_strains)

    # Stress = axial force / area. Q4 uses A = 1 cm^2 = 1e-4 m^2.
    # For generality, change AREA below if a different area is used.
    AREA = 1.0e-4
    stresses = np.asarray(forces) / AREA
    write_output("element_stresses.txt", stresses)

    print("\nGlobal stiffness matrix:\n", K)
    print("\nNodal displacements:\n", displacements)
    print("\nReaction forces:\n", reaction_forces)
    print("\nInternal element forces:\n", forces)

    if thermal:
        print("\nTotal element strains:\n", total_strains)
        print("\nThermal element strains:\n", thermal_strains)
        print("\nMechanical element strains:\n", mechanical_strains)
        print("\nElement stresses [Pa]:\n", stresses)


if __name__ == "__main__":
    main()
