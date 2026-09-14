import numpy as np
import ast


# ------------------------------------------------------------
# Reading input files
# ------------------------------------------------------------

def read_input(filename):
    """
    Read a raw Python list from a text file.
    """

    with open(filename, "r") as file:
        return ast.literal_eval(file.read())


# ------------------------------------------------------------
# Element geometry
# ------------------------------------------------------------

def element_geometry(x1, x2):
    """
    Calculate element length and direction cosines.

    Parameters
    ----------
    x1 : array-like
        [x, y] coordinates of the first node.

    x2 : array-like
        [x, y] coordinates of the second node.

    Returns
    -------
    L : float
        Element length.

    c : float
        Cosine of the element angle with the global x-axis.

    s : float
        Sine of the element angle with the global x-axis.
    """

    delta_x = x2[0] - x1[0]
    delta_y = x2[1] - x1[1]

    L = np.sqrt(delta_x**2 + delta_y**2)

    if L <= 0:
        raise ValueError("Truss element has zero length.")

    c = delta_x / L
    s = delta_y / L

    return L, c, s


# ------------------------------------------------------------
# Element DOF mapping
# ------------------------------------------------------------

def element_dofs(nodes):
    """
    Map the two nodes of one element to the four global DOFs.

    Zero-based node numbering is used.

    Node n:
        x DOF = 2*n
        y DOF = 2*n + 1
    """

    i = nodes[0]
    j = nodes[1]

    return np.array([
        2 * i,
        2 * i + 1,
        2 * j,
        2 * j + 1
    ], dtype=int)


# ------------------------------------------------------------
# Element stiffness matrix
# ------------------------------------------------------------

def element_stiffness(x1, x2, k):
    """
    Return the 4x4 truss element stiffness matrix
    expressed in global coordinates.

    k = EA / L
    """

    L, c, s = element_geometry(x1, x2)

    ke = k * np.array([
        [ c**2,   c*s,   -c**2,  -c*s ],
        [ c*s,    s**2,  -c*s,   -s**2],
        [-c**2,  -c*s,    c**2,   c*s ],
        [-c*s,   -s**2,   c*s,    s**2]
    ])

    return ke


# ------------------------------------------------------------
# Assemble global stiffness matrix
# ------------------------------------------------------------

def assemble_global_stiffness(
    num_nodes,
    coordinates,
    connectivity,
    k_values
):
    """
    Assemble the global stiffness matrix K.
    """

    num_dofs = 2 * num_nodes

    K = np.zeros((num_dofs, num_dofs))

    for e in range(len(connectivity)):

        # Global nodes for this element
        nodes = connectivity[e]

        # Coordinates of the two element nodes
        x1 = coordinates[nodes[0]]
        x2 = coordinates[nodes[1]]

        # Element stiffness matrix
        ke = element_stiffness(
            x1,
            x2,
            k_values[e]
        )

        # Global DOFs for the element
        gdofs = element_dofs(nodes)

        # Map local element stiffness matrix
        # into global stiffness matrix
        for local_row in range(4):

            global_row = gdofs[local_row]

            for local_col in range(4):

                global_col = gdofs[local_col]

                K[global_row, global_col] += \
                    ke[local_row, local_col]

    return K


# ------------------------------------------------------------
# Build global force vector
# ------------------------------------------------------------

def build_force_vector(num_nodes, loads):
    """
    Create the global external force vector.

    Input format:
        [
            [Fx0, Fy0],
            [Fx1, Fy1],
            ...
        ]

    Global vector format:
        [Fx0, Fy0, Fx1, Fy1, ...]
    """

    F = np.zeros(2 * num_nodes)

    for node in range(num_nodes):

        F[2 * node] = loads[node][0]
        F[2 * node + 1] = loads[node][1]

    return F


# ------------------------------------------------------------
# Solve FEM system
# ------------------------------------------------------------

def solve_system(K, F, prescribed_displacements):
    """
    Apply displacement boundary conditions and solve
    for the global nodal displacement vector.
    """

    num_nodes = len(prescribed_displacements)
    num_dofs = 2 * num_nodes

    fixed_dofs = []
    free_dofs = []

    # Determine free and constrained DOFs
    for node in range(num_nodes):

        for direction in range(2):

            global_dof = 2 * node + direction

            if prescribed_displacements[node][direction] is None:
                free_dofs.append(global_dof)
            else:
                fixed_dofs.append(global_dof)

    fixed_dofs = np.array(
        fixed_dofs,
        dtype=int
    )

    free_dofs = np.array(
        free_dofs,
        dtype=int
    )

    # Initialize global displacement vector
    u = np.zeros(num_dofs)

    # Apply prescribed displacements
    for node in range(num_nodes):

        for direction in range(2):

            value = prescribed_displacements[node][direction]

            if value is not None:

                global_dof = 2 * node + direction

                u[global_dof] = float(value)

    # Partition stiffness matrix
    K_FF = K[np.ix_(free_dofs, free_dofs)]

    K_FE = K[np.ix_(free_dofs, fixed_dofs)]

    # Partition force vector
    F_F = F[free_dofs]

    # Prescribed displacement vector
    u_E = u[fixed_dofs]

    # Solve:
    #
    # K_FF u_F = F_F - K_FE u_E
    #
    u_F = np.linalg.solve(
        K_FF,
        F_F - K_FE @ u_E
    )

    # Insert free displacements
    # into the global displacement vector
    u[free_dofs] = u_F

    return u, free_dofs, fixed_dofs


# ------------------------------------------------------------
# Recover reaction forces
# ------------------------------------------------------------

def recover_reactions(K, F, u, fixed_dofs):
    """
    Calculate reaction forces using:

        R = K u - F

    Reaction values at free DOFs are explicitly set to zero.
    """

    full_residual = K @ u - F

    reactions = np.zeros(len(F))

    for dof in fixed_dofs:
        reactions[dof] = full_residual[dof]

    return reactions


# ------------------------------------------------------------
# Recover element internal forces
# ------------------------------------------------------------

def recover_element_forces(
    coordinates,
    connectivity,
    k_values,
    u
):
    """
    Calculate the axial internal force in each truss element.

    Positive = tension
    Negative = compression
    """

    internal_forces = []

    for e in range(len(connectivity)):

        nodes = connectivity[e]

        # Element coordinates
        x1 = coordinates[nodes[0]]
        x2 = coordinates[nodes[1]]

        # Element geometry
        L, c, s = element_geometry(x1, x2)

        # Element global DOFs
        gdofs = element_dofs(nodes)

        # Element displacement vector
        u_e = u[gdofs]

        # Axial change in length
        #
        # delta = [-c, -s, c, s] * u_e
        #
        axial_extension = (
            -c * u_e[0]
            -s * u_e[1]
            +c * u_e[2]
            +s * u_e[3]
        )

        # Internal axial force
        force = k_values[e] * axial_extension

        internal_forces.append(force)

    return internal_forces


# ------------------------------------------------------------
# Check solution
# ------------------------------------------------------------

def check_solution(
    K,
    F,
    u,
    reactions,
    free_dofs
):
    """
    Check symmetry, residual, free DOF reactions,
    and global equilibrium.
    """

    # Check that global stiffness matrix is symmetric
    if np.allclose(K, K.T):
        print("Stiffness matrix symmetry check: PASS")
    else:
        print("Stiffness matrix symmetry check: FAIL")

    # Check residual
    residual = K @ u - F - reactions

    if np.allclose(residual, 0):
        print("Residual check: PASS")
    else:
        print("Residual check: FAIL")

    # Free DOFs should have zero reactions
    if np.allclose(reactions[free_dofs], 0):
        print("Free DOF reaction check: PASS")
    else:
        print("Free DOF reaction check: FAIL")

    # Global equilibrium
    force_pairs = F.reshape((-1, 2))
    reaction_pairs = reactions.reshape((-1, 2))

    total_external_force = np.sum(
        force_pairs,
        axis=0
    )

    total_reaction_force = np.sum(
        reaction_pairs,
        axis=0
    )

    equilibrium = (
        total_external_force
        + total_reaction_force
    )

    if np.allclose(equilibrium, 0):
        print("Global equilibrium check: PASS")
    else:
        print("Global equilibrium check: FAIL")

    print(
        "Net force [Fx, Fy]:",
        equilibrium
    )


# ------------------------------------------------------------
# Write output files
# ------------------------------------------------------------

def write_output(filename, values):
    """
    Write values as a raw Python list.
    """

    if isinstance(values, np.ndarray):
        values = values.tolist()

    else:
        values = [
            float(value)
            for value in values
        ]

    with open(filename, "w") as file:
        file.write(str(values))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    # --------------------------------------------------------
    # Read input files
    # --------------------------------------------------------

    coordinates = read_input(
        "nodal coordinates.txt"
    )

    connectivity = read_input(
        "connectivity array.txt"
    )

    k_values = read_input(
        "element stiffnesses.txt"
    )

    loads = read_input(
        "external nodal forces.txt"
    )

    prescribed_displacements = read_input(
        "displacement BCs.txt"
    )

    # Convert numerical inputs to NumPy arrays
    coordinates = np.array(
        coordinates,
        dtype=float
    )

    connectivity = np.array(
        connectivity,
        dtype=int
    )

    k_values = np.array(
        k_values,
        dtype=float
    )

    loads = np.array(
        loads,
        dtype=float
    )

    # Number of nodes
    num_nodes = len(coordinates)

    # --------------------------------------------------------
    # Build force vector
    # --------------------------------------------------------

    F = build_force_vector(
        num_nodes,
        loads
    )

    # --------------------------------------------------------
    # Assemble global stiffness matrix
    # --------------------------------------------------------

    K = assemble_global_stiffness(
        num_nodes,
        coordinates,
        connectivity,
        k_values
    )

    # --------------------------------------------------------
    # Solve for nodal displacements
    # --------------------------------------------------------

    u, free_dofs, fixed_dofs = solve_system(
        K,
        F,
        prescribed_displacements
    )

    # --------------------------------------------------------
    # Calculate reaction forces
    # --------------------------------------------------------

    reactions = recover_reactions(
        K,
        F,
        u,
        fixed_dofs
    )

    # --------------------------------------------------------
    # Calculate element forces
    # --------------------------------------------------------

    internal_forces = recover_element_forces(
        coordinates,
        connectivity,
        k_values,
        u
    )

    # --------------------------------------------------------
    # Check solution
    # --------------------------------------------------------

    check_solution(
        K,
        F,
        u,
        reactions,
        free_dofs
    )

    # --------------------------------------------------------
    # Reshape displacement and reaction outputs
    # --------------------------------------------------------

    nodal_displacements = u.reshape(
        (num_nodes, 2)
    )

    reaction_forces = reactions.reshape(
        (num_nodes, 2)
    )

    # --------------------------------------------------------
    # Write output files
    # --------------------------------------------------------

    write_output(
        "nodal displacements.txt",
        nodal_displacements
    )

    write_output(
        "reaction forces.txt",
        reaction_forces
    )

    write_output(
        "internal forces.txt",
        internal_forces
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nGlobal stiffness matrix:")
    print(K)

    print("\nNodal displacements:")
    print(nodal_displacements)

    print("\nReaction forces:")
    print(reaction_forces)

    print("\nInternal element forces:")
    print(internal_forces)


if __name__ == "__main__":
    main()
