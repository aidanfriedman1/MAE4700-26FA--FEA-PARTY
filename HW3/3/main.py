import ast
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parent


def read_input(filename):
    with (DIRECTORY / filename).open() as file:
        return ast.literal_eval(file.read())


def validate_inputs(coords, connectivity, k_values, loads, bcs):
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
            if value is not None and (not isinstance(value, (int, float)) or not np.isfinite(value)):
                raise ValueError("BC entries must be finite numbers or None")
    for e, nodes in enumerate(connectivity):
        if (not isinstance(nodes, list) or len(nodes) != 2
                or any(type(n) is not int or n < 0 or n >= num_nodes for n in nodes)):
            raise ValueError(f"element {e} must connect two valid zero-based integer nodes")
        element_geometry(coords[nodes[0]], coords[nodes[1]])
    return num_nodes, ndim


def node_dofs(node, ndim):
    # Each node contributes ndim consecutive global displacement DOFs.
    return np.arange(node * ndim, (node + 1) * ndim, dtype=int)


def element_dofs(nodes, ndim):
    return np.concatenate((node_dofs(nodes[0], ndim), node_dofs(nodes[1], ndim)))


def element_geometry(x1, x2):
    # The unit direction vector works for both 2D and 3D bars.
    dx = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
    length = np.linalg.norm(dx)
    if length <= 0:
        raise ValueError("Truss element has zero length")
    return length, dx / length


def element_operator(x1, x2):
    # B projects element nodal displacements onto axial extension.
    length, direction = element_geometry(x1, x2)
    B = np.concatenate((-direction, direction))
    return length, direction, B


def element_stiffness(x1, x2, k):
    # The input k is EA/L; the outer product gives the global element matrix.
    _, _, B = element_operator(x1, x2)
    return k * np.outer(B, B)


def assemble_global_stiffness(num_nodes, ndim, coords, connectivity, k_values):
    # Add each element matrix into its corresponding global DOFs.
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


def solve_system(K, F, bcs, ndim):
    # None marks a free DOF; a number prescribes its displacement.
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
        # Solve K_ff u_f = F_f - K_fe u_e, allowing nonzero prescribed motion.
        K_ff = K[np.ix_(free, free)]
        K_fe = K[np.ix_(free, fixed)]
        condition_number = np.linalg.cond(K_ff)
        print("K_FF condition number:", condition_number)
        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise ValueError("K_FF is singular or ill-conditioned. Check constraints, mechanisms, or disconnected nodes.")
        try:
            u[free] = np.linalg.solve(K_ff, F[free] - K_fe @ u[fixed])
        except np.linalg.LinAlgError as exc:
            raise ValueError("K_FF is singular. Check constraints and mechanisms.") from exc
    return u, free, fixed


def recover_reactions(K, F, u, fixed):
    # Reactions are the residual at prescribed DOFs only.
    reactions = np.zeros(len(F))
    reactions[fixed] = (K @ u - F)[fixed]
    return reactions


def recover_element_forces(coords, connectivity, k_values, u, ndim):
    # Positive axial force denotes tension; negative denotes compression.
    forces = []
    for e, nodes in enumerate(connectivity):
        _, _, B = element_operator(coords[nodes[0]], coords[nodes[1]])
        forces.append(float(k_values[e] * (B @ u[element_dofs(nodes, ndim)])))
    return forces


def check_solution(K, F, u, reactions, free, ndim):
    # Check symmetry, the free-DOF equations, and overall force balance.
    if not np.allclose(K, K.T):
        raise ValueError("Global stiffness symmetry check failed")
    residual = K @ u - F
    if not np.allclose(residual[free], 0, atol=1e-7):
        raise ValueError("Free DOF residual check failed")
    if not np.allclose(np.sum((F + reactions).reshape(-1, ndim), axis=0), 0, atol=1e-7):
        raise ValueError("Global force equilibrium check failed")
    print("Stiffness symmetry, free DOF residual, and global equilibrium: PASS")


def write_output(filename, values):
    if isinstance(values, np.ndarray):
        values = values.tolist()
    (DIRECTORY / filename).write_text(str(values))


def main():
    # Read the same five raw-Python-list files used by the HW2 solver.
    coords = np.asarray(read_input("nodal_coordinates.txt"), dtype=float)
    connectivity = read_input("connectivity_array.txt")
    k_values = np.asarray(read_input("element_stiffnesses.txt"), dtype=float)
    loads = np.asarray(read_input("external_nodal_forces.txt"), dtype=float)
    bcs = read_input("displacement_BCs.txt")
    num_nodes, ndim = validate_inputs(coords, connectivity, k_values, loads, bcs)
    # Every later DOF count and output shape uses the detected dimension.
    print(f"Input validation: PASS ({ndim}D, {num_nodes} nodes, {len(connectivity)} elements)")
    F = build_force_vector(loads)
    K = assemble_global_stiffness(num_nodes, ndim, coords, connectivity, k_values)
    u, free, fixed = solve_system(K, F, bcs, ndim)
    reactions = recover_reactions(K, F, u, fixed)
    forces = recover_element_forces(coords, connectivity, k_values, u, ndim)
    check_solution(K, F, u, reactions, free, ndim)
    displacements = u.reshape(num_nodes, ndim)
    reaction_forces = reactions.reshape(num_nodes, ndim)
    write_output("nodal_displacements.txt", displacements)
    # Write lists so the output can be read back with ast.literal_eval.
    write_output("reaction_forces.txt", reaction_forces)
    write_output("internal_forces.txt", forces)
    print("\nGlobal stiffness matrix:\n", K)
    print("\nNodal displacements:\n", displacements)
    print("\nReaction forces:\n", reaction_forces)
    print("\nInternal element forces:\n", forces)


if __name__ == "__main__":
    main()
