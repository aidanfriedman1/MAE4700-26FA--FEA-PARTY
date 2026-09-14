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
# Input validation
# ------------------------------------------------------------

def validate_inputs(
    coordinates,
    connectivity_raw,
    k_values,
    loads,
    prescribed_displacements
):
    """
    Check that all input data has the correct size,
    shape, type, and connectivity.
    """

    num_nodes = len(coordinates)
    num_elements = len(connectivity_raw)

    # --------------------------------------------------------
    # Check coordinates shape
    # --------------------------------------------------------

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "nodal coordinates must have shape (num_nodes, 2)"
        )

    # --------------------------------------------------------
    # Check connectivity shape before converting to int
    # --------------------------------------------------------

    if not isinstance(connectivity_raw, list):
        raise ValueError(
            "connectivity array must be a list"
        )

    for element in connectivity_raw:

        if not isinstance(element, list) or len(element) != 2:
            raise ValueError(
                "connectivity array must have shape "
                "(num_elements, 2)"
            )

    # --------------------------------------------------------
    # Check that every connectivity entry is an integer
    # --------------------------------------------------------

    for element in connectivity_raw:

        for node in element:

            if not isinstance(node, int):
                raise ValueError(
                    "every connectivity entry must be an integer"
                )

            if node < 0 or node >= num_nodes:
                raise ValueError(
                    "connectivity contains an invalid node number"
                )

    # --------------------------------------------------------
    # Check number of element stiffnesses
    # --------------------------------------------------------

    if len(k_values) != num_elements:
        raise ValueError(
            "element stiffnesses must contain one value per element"
        )

    # --------------------------------------------------------
    # Check external nodal forces shape
    # --------------------------------------------------------

    if loads.ndim != 2 or loads.shape != (num_nodes, 2):
        raise ValueError(
            "external nodal forces must have shape (num_nodes, 2)"
        )

    # --------------------------------------------------------
    # Check displacement BCs
    # --------------------------------------------------------

    if len(prescribed_displacements) != num_nodes:
        raise ValueError(
            "displacement BCs must contain one [ux, uy] pair per node"
        )

    for node in range(num_nodes):

        pair = prescribed_displacements[node]

        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                "each displacement BC must contain [ux, uy]"
            )

        for value in pair:

            if value is not None and not isinstance(
                value,
                (int, float)
            ):
                raise ValueError(
                    "each displacement BC entry must be "
                    "a number or None"
                )

    # --------------------------------------------------------
    # Check for coincident end nodes
    # --------------------------------------------------------

    for e in range(num_elements):

        node_i = connectivity_raw[e][0]
        node_j = connectivity_raw[e][1]

        if np.allclose(
            coordinates[node_i],
            coordinates[node_j]
        ):
            raise ValueError(
                f"element {e} has coincident end nodes"
            )

    print("Input validation: PASS")


# ------------------------------------------------------------
# Element geometry
# ------------------------------------------------------------

def element_geometry(x1, x2):
    """
    Calculate element length and direction cosines.
    """

    delta_x = x2[0] - x1[0]
    delta_y = x2[1] - x1[1]

    L = np.sqrt(
        delta_x**2 + delta_y**2
    )

    if L <= 0:
        raise ValueError(
            "Truss element has zero length."
        )

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

    L, c, s = element_geometry(
        x1,
        x2
    )

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

    K = np.zeros(
        (num_dofs, num_dofs)
    )

    for e in range(len(connectivity)):

        # Global nodes for this element
        nodes = connectivity[e]

        # Coordinates of the two nodes
        x1 = coordinates[nodes[0]]
        x2 = coordinates[nodes[1]]

        # Element stiffness matrix
        ke = element_stiffness(
            x1,
            x2,
            k_values[e]
        )

        # Global DOFs
        gdofs = element_dofs(
            nodes
        )

        # Add local contributions to global matrix
        for local_row in range(4):

            global_row = gdofs[local_row]

            for local_col in range(4):

                global_col = gdofs[local_col]

                K[
                    global_row,
                    global_col
                ] += ke[
                    local_row,
                    local_col
                ]

    return K


# ------------------------------------------------------------
# Build global force vector
# ------------------------------------------------------------

def build_force_vector(
    num_nodes,
    loads
):
    """
    Create the global external force vector.

    Input:
        [
            [Fx0, Fy0],
            [Fx1, Fy1],
            ...
        ]

    Output:
        [Fx0, Fy0, Fx1, Fy1, ...]
    """

    F = np.zeros(
        2 * num_nodes
    )

    for node in range(num_nodes):

        F[2 * node] = \
            loads[node][0]

        F[2 * node + 1] = \
            loads[node][1]

    return F


# ------------------------------------------------------------
# Solve FEM system
# ------------------------------------------------------------

def solve_system(
    K,
    F,
    prescribed_displacements
):
    """
    Apply displacement boundary conditions and solve
    for the global nodal displacement vector.
    """

    num_nodes = len(
        prescribed_displacements
    )

    num_dofs = 2 * num_nodes

    fixed_dofs = []
    free_dofs = []

    # --------------------------------------------------------
    # Identify fixed and free DOFs
    # --------------------------------------------------------

    for node in range(num_nodes):

        for direction in range(2):

            global_dof = \
                2 * node + direction

            if prescribed_displacements[
                node
            ][direction] is None:

                free_dofs.append(
                    global_dof
                )

            else:

                fixed_dofs.append(
                    global_dof
                )

    fixed_dofs = np.array(
        fixed_dofs,
        dtype=int
    )

    free_dofs = np.array(
        free_dofs,
        dtype=int
    )

    # --------------------------------------------------------
    # Initialize global displacement vector
    # --------------------------------------------------------

    u = np.zeros(
        num_dofs
    )

    # --------------------------------------------------------
    # Insert prescribed displacements
    # --------------------------------------------------------

    for node in range(num_nodes):

        for direction in range(2):

            value = prescribed_displacements[
                node
            ][direction]

            if value is not None:

                global_dof = \
                    2 * node + direction

                u[global_dof] = \
                    float(value)

    # --------------------------------------------------------
    # Partition system
    # --------------------------------------------------------

    K_FF = K[
        np.ix_(
            free_dofs,
            free_dofs
        )
    ]

    K_FE = K[
        np.ix_(
            free_dofs,
            fixed_dofs
        )
    ]

    F_F = F[
        free_dofs
    ]

    u_E = u[
        fixed_dofs
    ]

    # --------------------------------------------------------
    # Solve:
    #
    # K_FF u_F = F_F - K_FE u_E
    # --------------------------------------------------------

    u_F = np.linalg.solve(
        K_FF,
        F_F - K_FE @ u_E
    )

    # Put free displacements back
    # into full displacement vector
    u[free_dofs] = u_F

    return (
        u,
        free_dofs,
        fixed_dofs
    )


# ------------------------------------------------------------
# Recover reaction forces
# ------------------------------------------------------------

def recover_reactions(
    K,
    F,
    u,
    fixed_dofs
):
    """
    Calculate reactions using:

        R = K u - F

    Free DOFs are explicitly set to zero.
    """

    full_residual = \
        K @ u - F

    reactions = np.zeros(
        len(F)
    )

    for dof in fixed_dofs:

        reactions[dof] = \
            full_residual[dof]

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
    Calculate the axial internal force
    in each truss element.

    Positive = tension
    Negative = compression
    """

    internal_forces = []

    for e in range(
        len(connectivity)
    ):

        nodes = connectivity[e]

        # Element coordinates
        x1 = coordinates[
            nodes[0]
        ]

        x2 = coordinates[
            nodes[1]
        ]

        # Element geometry
        L, c, s = element_geometry(
            x1,
            x2
        )

        # Element global DOFs
        gdofs = element_dofs(
            nodes
        )

        # Element displacement vector
        u_e = u[
            gdofs
        ]

        # Axial extension:
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
        force = (
            k_values[e]
            * axial_extension
        )

        internal_forces.append(
            force
        )

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
    Check stiffness matrix symmetry,
    residual, free DOF reactions,
    and global equilibrium.
    """

    # --------------------------------------------------------
    # Symmetry check
    # --------------------------------------------------------

    if np.allclose(
        K,
        K.T
    ):
        print(
            "Stiffness matrix symmetry check: PASS"
        )

    else:
        print(
            "Stiffness matrix symmetry check: FAIL"
        )

    # --------------------------------------------------------
    # Residual check
    # --------------------------------------------------------

    residual = (
        K @ u
        - F
        - reactions
    )

    if np.allclose(
        residual,
        0
    ):
        print(
            "Residual check: PASS"
        )

    else:
        print(
            "Residual check: FAIL"
        )

    # --------------------------------------------------------
    # Free DOF reaction check
    # --------------------------------------------------------

    if np.allclose(
        reactions[free_dofs],
        0
    ):
        print(
            "Free DOF reaction check: PASS"
        )

    else:
        print(
            "Free DOF reaction check: FAIL"
        )

    # --------------------------------------------------------
    # Global equilibrium check
    # --------------------------------------------------------

    force_pairs = F.reshape(
        (-1, 2)
    )

    reaction_pairs = \
        reactions.reshape(
            (-1, 2)
        )

    total_external_force = \
        np.sum(
            force_pairs,
            axis=0
        )

    total_reaction_force = \
        np.sum(
            reaction_pairs,
            axis=0
        )

    equilibrium = (
        total_external_force
        + total_reaction_force
    )

    if np.allclose(
        equilibrium,
        0
    ):
        print(
            "Global equilibrium check: PASS"
        )

    else:
        print(
            "Global equilibrium check: FAIL"
        )

    print(
        "Net force [Fx, Fy]:",
        equilibrium
    )


# ------------------------------------------------------------
# Write output files
# ------------------------------------------------------------

def write_output(
    filename,
    values
):
    """
    Write values as a raw Python list.
    """

    if isinstance(
        values,
        np.ndarray
    ):
        values = values.tolist()

    else:
        values = [
            float(value)
            for value in values
        ]

    with open(
        filename,
        "w"
    ) as file:

        file.write(
            str(values)
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    # --------------------------------------------------------
    # Read raw input files
    # --------------------------------------------------------

    coordinates_raw = read_input(
        "nodal coordinates.txt"
    )

    connectivity_raw = read_input(
        "connectivity array.txt"
    )

    k_values_raw = read_input(
        "element stiffnesses.txt"
    )

    loads_raw = read_input(
        "external nodal forces.txt"
    )

    prescribed_displacements = read_input(
        "displacement BCs.txt"
    )

    # --------------------------------------------------------
    # Convert numerical inputs
    # --------------------------------------------------------

    coordinates = np.array(
        coordinates_raw,
        dtype=float
    )

    k_values = np.array(
        k_values_raw,
        dtype=float
    )

    loads = np.array(
        loads_raw,
        dtype=float
    )

    # --------------------------------------------------------
    # Validate raw and converted inputs
    # --------------------------------------------------------

    validate_inputs(
        coordinates,
        connectivity_raw,
        k_values,
        loads,
        prescribed_displacements
    )

    # Only convert connectivity to integer
    # after verifying every entry is actually an integer
    connectivity = np.array(
        connectivity_raw,
        dtype=int
    )

    # Number of nodes
    num_nodes = len(
        coordinates
    )

    # --------------------------------------------------------
    # Build global force vector
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

    u, free_dofs, fixed_dofs = \
        solve_system(
            K,
            F,
            prescribed_displacements
        )

    # --------------------------------------------------------
    # Recover reaction forces
    # --------------------------------------------------------

    reactions = \
        recover_reactions(
            K,
            F,
            u,
            fixed_dofs
        )

    # --------------------------------------------------------
    # Recover internal forces
    # --------------------------------------------------------

    internal_forces = \
        recover_element_forces(
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
    # Reshape output data
    # --------------------------------------------------------

    nodal_displacements = \
        u.reshape(
            (num_nodes, 2)
        )

    reaction_forces = \
        reactions.reshape(
            (num_nodes, 2)
        )

    # --------------------------------------------------------
    # Write required output files
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

    print(
        "\nGlobal stiffness matrix:"
    )

    print(
        K
    )

    print(
        "\nNodal displacements:"
    )

    print(
        nodal_displacements
    )

    print(
        "\nReaction forces:"
    )

    print(
        reaction_forces
    )

    print(
        "\nInternal element forces:"
    )

    print(
        internal_forces
    )


if __name__ == "__main__":
    main()
