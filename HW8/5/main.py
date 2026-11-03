import ast
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Input and validation
# -----------------------------------------------------------------------------
def read_input(filename):
    """Read one raw Python literal from the current run directory."""
    with (DIRECTORY / filename).open("r", encoding="utf-8") as file:
        return ast.literal_eval(file.read())


def validate_inputs(coords, connectivity, axial_stiffnesses, flexural_rigidities,
                    distributed_loads, nodal_loads, bcs):
    """Validate all seven HW8 Question 5 input files."""
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
        raise ValueError("nodal coordinates must have shape (num_nodes, 2)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("nodal coordinates must contain only finite values")

    num_nodes = len(coords)
    if not isinstance(connectivity, list) or not connectivity:
        raise ValueError("connectivity array must be a nonempty list")
    num_elements = len(connectivity)

    for e, nodes in enumerate(connectivity):
        if (not isinstance(nodes, list) or len(nodes) != 2
                or any(type(node) is not int for node in nodes)):
            raise ValueError(f"element {e} connectivity must be [node1, node2] integers")
        if any(node < 0 or node >= num_nodes for node in nodes):
            raise ValueError(f"element {e} contains an invalid node index")
        if nodes[0] == nodes[1]:
            raise ValueError(f"element {e} connects a node to itself")
        element_geometry(coords[nodes[0]], coords[nodes[1]])

    if axial_stiffnesses.shape != (num_elements,):
        raise ValueError("element stiffnesses must contain one EA/L value per element")
    if (not np.all(np.isfinite(axial_stiffnesses))
            or np.any(axial_stiffnesses <= 0.0)):
        raise ValueError("all element axial stiffnesses EA/L must be positive and finite")

    if flexural_rigidities.shape != (num_elements,):
        raise ValueError("element flexural rigidities must contain one EI value per element")
    if (not np.all(np.isfinite(flexural_rigidities))
            or np.any(flexural_rigidities <= 0.0)):
        raise ValueError("all element flexural rigidities EI must be positive and finite")

    if distributed_loads.shape != (num_elements,) or not np.all(np.isfinite(distributed_loads)):
        raise ValueError("distributed loads must contain one finite constant p value per element")

    if nodal_loads.shape != (num_nodes, 3) or not np.all(np.isfinite(nodal_loads)):
        raise ValueError("external nodal forces must have shape (num_nodes, 3) and be finite")

    if not isinstance(bcs, list) or len(bcs) != num_nodes:
        raise ValueError("displacement BCs must contain one [ux, uy, theta] list per node")
    for node, values in enumerate(bcs):
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"node {node} boundary conditions must contain [ux, uy, theta]")
        for value in values:
            if value is not None:
                if (not isinstance(value, (int, float))
                        or not np.isfinite(value)):
                    raise ValueError("boundary-condition entries must be finite numbers or None")

    return num_nodes, num_elements


# -----------------------------------------------------------------------------
# Element geometry, DOFs, and coordinate transformation
# -----------------------------------------------------------------------------
def element_geometry(x1, x2):
    """Return length L and direction cosines c=cos(angle), s=sin(angle)."""
    delta = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("frame element has zero or invalid length")
    c = float(delta[0] / length)
    s = float(delta[1] / length)
    return length, c, s


def element_dofs(nodes):
    """Global DOF indices for [ux1, uy1, theta1, ux2, uy2, theta2]."""
    i, j = nodes
    return np.array([3*i, 3*i + 1, 3*i + 2,
                     3*j, 3*j + 1, 3*j + 2], dtype=int)


def transformation_matrix(c, s):
    """Map global element displacements to local element displacements: d' = T d."""
    node_transform = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
    T = np.zeros((6, 6))
    T[:3, :3] = node_transform
    T[3:, 3:] = node_transform
    return T


# -----------------------------------------------------------------------------
# Local frame element stiffness and consistent distributed-load vector
# -----------------------------------------------------------------------------
def local_element_stiffness(length, axial_stiffness, EI):
    """Build the 6x6 local frame stiffness in [u1,w1,th1,u2,w2,th2] order.

    axial_stiffness is the input EA/L (not EA). EI is the flexural rigidity.
    """
    L = length
    k_local = np.zeros((6, 6))

    # Axial truss contribution, local DOFs u1 and u2 (indices 0 and 3).
    axial = axial_stiffness * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_local[np.ix_([0, 3], [0, 3])] = axial

    # Euler-Bernoulli bending contribution, local DOFs [w1, th1, w2, th2].
    bending = (EI / L**3) * np.array([
        [12.0,       6.0*L,  -12.0,       6.0*L],
        [6.0*L,  4.0*L**2,   -6.0*L,  2.0*L**2],
        [-12.0,     -6.0*L,   12.0,      -6.0*L],
        [6.0*L,  2.0*L**2,   -6.0*L,  4.0*L**2]
    ])
    bending_dofs = [1, 2, 4, 5]
    k_local[np.ix_(bending_dofs, bending_dofs)] = bending
    return k_local


def local_distributed_load(length, p):
    """Consistent local load for constant p acting in the local +y' direction."""
    L = length
    return (p * L / 2.0) * np.array([
        0.0, 1.0, L / 6.0,
        0.0, 1.0, -L / 6.0
    ])


def frame_element_matrices(x1, x2, axial_stiffness, EI, p):
    """Return local/global stiffness, transform, and local/global element loads."""
    L, c, s = element_geometry(x1, x2)
    T = transformation_matrix(c, s)
    k_local = local_element_stiffness(L, axial_stiffness, EI)
    f_local = local_distributed_load(L, p)
    k_global = T.T @ k_local @ T
    f_global = T.T @ f_local
    return L, T, k_local, f_local, k_global, f_global


# -----------------------------------------------------------------------------
# Global assembly and solution
# -----------------------------------------------------------------------------
def assemble_global_system(num_nodes, coords, connectivity, axial_stiffnesses,
                           flexural_rigidities, distributed_loads):
    """Assemble global K and the equivalent nodal loads from element p loads."""
    num_dofs = 3 * num_nodes
    K = np.zeros((num_dofs, num_dofs))
    F_distributed = np.zeros(num_dofs)

    for e, nodes in enumerate(connectivity):
        gdofs = element_dofs(nodes)
        _, _, _, _, ke_global, fe_global = frame_element_matrices(
            coords[nodes[0]], coords[nodes[1]],
            axial_stiffnesses[e], flexural_rigidities[e], distributed_loads[e]
        )
        if not np.allclose(ke_global, ke_global.T, rtol=1e-10, atol=1e-10):
            raise ValueError(f"element {e} global stiffness matrix is not symmetric")
        K[np.ix_(gdofs, gdofs)] += ke_global
        F_distributed[gdofs] += fe_global

    return K, F_distributed


def build_global_force_vector(nodal_loads, F_distributed):
    """Combine applied nodal forces/moments and equivalent distributed loads."""
    return nodal_loads.reshape(-1).copy() + F_distributed


def solve_system(K, F, bcs):
    """Solve the partitioned system, allowing nonzero prescribed DOF values."""
    u = np.zeros(len(F))
    fixed = []
    free = []

    for node, values in enumerate(bcs):
        for local_dof, value in enumerate(values):
            gdof = 3 * node + local_dof
            if value is None:
                free.append(gdof)
            else:
                fixed.append(gdof)
                u[gdof] = float(value)

    fixed = np.asarray(fixed, dtype=int)
    free = np.asarray(free, dtype=int)

    if len(free):
        K_ff = K[np.ix_(free, free)]
        K_fe = K[np.ix_(free, fixed)]
        condition_number = np.linalg.cond(K_ff)
        print("K_FF condition number:", condition_number)
        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise ValueError(
                "K_FF is singular or ill-conditioned. Check supports, mechanisms, "
                "and disconnected nodes."
            )
        try:
            u[free] = np.linalg.solve(K_ff, F[free] - K_fe @ u[fixed])
        except np.linalg.LinAlgError as exc:
            raise ValueError("K_FF is singular; check the frame constraints and connectivity") from exc

    return u, free, fixed


def recover_reactions(K, F, u, fixed):
    """Return reactions at prescribed DOFs; entries at free DOFs are zero."""
    reactions = np.zeros(len(F))
    reactions[fixed] = (K @ u - F)[fixed]
    return reactions


# -----------------------------------------------------------------------------
# Element force recovery and numerical checks
# -----------------------------------------------------------------------------
def recover_element_forces(coords, connectivity, axial_stiffnesses,
                           flexural_rigidities, distributed_loads, u):
    """Recover local end forces: k_local d_local - f_local."""
    element_forces = []
    for e, nodes in enumerate(connectivity):
        gdofs = element_dofs(nodes)
        L, T, k_local, f_local, _, _ = frame_element_matrices(
            coords[nodes[0]], coords[nodes[1]],
            axial_stiffnesses[e], flexural_rigidities[e], distributed_loads[e]
        )
        d_global = u[gdofs]
        d_local = T @ d_global
        q_local = k_local @ d_local - f_local
        element_forces.append([float(value) for value in q_local])
    return element_forces


def check_solution(K, F, u, reactions, free, coords, num_nodes):
    """Check stiffness symmetry, free-DOF residual, and global force/moment balance."""
    if not np.allclose(K, K.T, rtol=1e-9, atol=1e-8):
        raise ValueError("global stiffness symmetry check failed")

    residual = K @ u - F
    if len(free) and not np.allclose(residual[free], 0.0, rtol=1e-7, atol=1e-7):
        raise ValueError("free-DOF equilibrium residual check failed")

    # F includes the transformed consistent element loads, so its nodal
    # force/moment resultant can be checked about the global origin.
    total = (F + reactions).reshape(num_nodes, 3)
    sum_fx = float(np.sum(total[:, 0]))
    sum_fy = float(np.sum(total[:, 1]))
    sum_mz = float(np.sum(total[:, 2] + coords[:, 0] * total[:, 1]
                           - coords[:, 1] * total[:, 0]))
    resultant = np.array([sum_fx, sum_fy, sum_mz])
    scale = max(1.0, float(np.linalg.norm(F)), float(np.linalg.norm(reactions)))
    if not np.allclose(resultant, 0.0, rtol=1e-8, atol=1e-7 * scale):
        raise ValueError(f"global force/moment equilibrium check failed: {resultant}")

    print("Global stiffness symmetry: PASS")
    print("Free-DOF residual: PASS")
    print("Global force and moment equilibrium: PASS")


def write_output(filename, values):
    """Write output as a raw Python list of built-in floats."""
    if isinstance(values, np.ndarray):
        values = values.tolist()
    (DIRECTORY / filename).write_text(repr(values), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------
def main():
    # Read the seven standard HW8 Question 5 input files.
    coords = np.asarray(read_input("nodal_coordinates.txt"), dtype=float)
    connectivity = read_input("connectivity_array.txt")
    axial_stiffnesses = np.asarray(read_input("element_stiffnesses.txt"), dtype=float)
    flexural_rigidities = np.asarray(read_input("element_flexural_rigidities.txt"), dtype=float)
    distributed_loads = np.asarray(read_input("distributed_loads.txt"), dtype=float)
    nodal_loads = np.asarray(read_input("external_nodal_forces.txt"), dtype=float)
    bcs = read_input("displacement_BCs.txt")

    num_nodes, num_elements = validate_inputs(
        coords, connectivity, axial_stiffnesses, flexural_rigidities,
        distributed_loads, nodal_loads, bcs
    )
    print(f"Input validation: PASS ({num_nodes} nodes, {num_elements} frame elements)")

    K, F_distributed = assemble_global_system(
        num_nodes, coords, connectivity, axial_stiffnesses,
        flexural_rigidities, distributed_loads
    )
    F = build_global_force_vector(nodal_loads, F_distributed)

    u, free, fixed = solve_system(K, F, bcs)
    reactions = recover_reactions(K, F, u, fixed)
    internal_forces = recover_element_forces(
        coords, connectivity, axial_stiffnesses,
        flexural_rigidities, distributed_loads, u
    )
    check_solution(K, F, u, reactions, free, coords, num_nodes)

    nodal_displacements = u.reshape(num_nodes, 3)
    reaction_forces = reactions.reshape(num_nodes, 3)

    # Exact required output filenames and list formats from the HW8 handout.
    write_output("nodal_displacements.txt", nodal_displacements)
    write_output("reaction_forces.txt", reaction_forces)
    write_output("internal_forces.txt", internal_forces)

    print("\nNodal displacements [ux, uy, theta]:\n", nodal_displacements)
    print("\nReaction forces/moments [Rx, Ry, Mz]:\n", reaction_forces)
    print("\nLocal element end forces [N1, V1, M1, N2, V2, M2]:\n",
          np.asarray(internal_forces))


if __name__ == "__main__":
    main()
