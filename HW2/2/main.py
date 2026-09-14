import numpy as np
import ast
from pathlib import Path


# ============================================================
# Reading input files
# ============================================================

def read_input(filename):
    filepath = Path(__file__).parent / filename

    with open(filepath, "r") as file:
        return ast.literal_eval(file.read())


# ============================================================
# Input validation
# ============================================================

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
    # Nodal coordinates must have shape (num_nodes, 2)
    # --------------------------------------------------------

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "nodal coordinates must have shape (num_nodes, 2)"
        )

    # --------------------------------------------------------
    # Connectivity must have shape (num_elements, 2)
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
    # Connectivity entries must be integers from
    # 0 to num_nodes - 1
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
    # One element stiffness per element
    # --------------------------------------------------------

    if len(k_values) != num_elements:
        raise ValueError(
            "element stiffnesses must contain one value per element"
        )

    # --------------------------------------------------------
    # External nodal forces must have shape (num_nodes, 2)
    # --------------------------------------------------------

    if loads.ndim != 2 or loads.shape != (num_nodes, 2):
        raise ValueError(
            "external nodal forces must have shape (num_nodes, 2)"
        )

    # --------------------------------------------------------
    # One displacement BC pair per node
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
    # No element may have coincident end nodes
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


# ============================================================
# Element geometry
# ============================================================

def element_geometry(x1, x2):
    """
    Calculate the element length and direction cosines.
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


# ============================================================
# Element DOF mapping
# ============================================================

def element_dofs(nodes):
    """
    Map the two nodes of one element to four global DOFs.

    Zero-based node numbering:

        Node n x DOF = 2*n
        Node n y DOF = 2*n + 1
    """

    i = nodes[0]
    j = nodes[1]

    return np.array([
        2 * i,
        2 * i + 1,
        2 * j,
        2 * j + 1
    ], dtype=int)


# ============================================================
# Element stiffness matrix
# ============================================================

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


# ============================================================
# Element-level numerical checks
# ============================================================

def check_element_stiffness(
    ke,
    c,
    s,
    element_number
):
    """
    Check:

    1. Element stiffness symmetry.
    2. Horizontal element stiffness acts only in x.
    3. Vertical element stiffness acts only in y.
    """

    # --------------------------------------------------------
    # Element symmetry
    # --------------------------------------------------------

    if not np.allclose(
        ke,
        ke.T
    ):
        raise ValueError(
            f"Element {element_number} stiffness matrix "
            f"is not symmetric."
        )

    # --------------------------------------------------------
    # Horizontal element check
    #
    # If s = 0, the element has no stiffness in global y.
    # Therefore rows/columns associated with y DOFs
    # should be approximately zero.
    # --------------------------------------------------------

    if np.isclose(s, 0.0):

        y_dofs = [1, 3]

        if not np.allclose(
            ke[y_dofs, :],
            0.0
        ):
            raise ValueError(
                f"Horizontal element {element_number} "
                f"has unexpected y-direction stiffness."
            )

        if not np.allclose(
            ke[:, y_dofs],
            0.0
        ):
            raise ValueError(
                f"Horizontal element {element_number} "
                f"has unexpected y-direction stiffness."
            )

    # --------------------------------------------------------
    # Vertical element check
    #
    # If c = 0, the element has no stiffness in global x.
    # Therefore rows/columns associated with x DOFs
    # should be approximately zero.
    # --------------------------------------------------------

    if np.isclose(c, 0.0):

        x_dofs = [0, 2]

        if not np.allclose(
            ke[x_dofs, :],
            0.0
        ):
            raise ValueError(
                f"Vertical element {element_number} "
                f"has unexpected x-direction stiffness."
            )

        if not np.allclose(
            ke[:, x_dofs],
            0.0
        ):
            raise ValueError(
                f"Vertical element {element_number} "
                f"has unexpected x-direction stiffness."
            )


# ============================================================
# Assemble global stiffness matrix
# ============================================================

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

        # Global nodes
        nodes = connectivity[e]

        # Node coordinates
        x1 = coordinates[nodes[0]]
        x2 = coordinates[nodes[1]]

        # Geometry
        L, c, s = element_geometry(
            x1,
            x2
        )

        # Element stiffness matrix
        ke = element_stiffness(
            x1,
            x2,
            k_values[e]
        )

        # ----------------------------------------------------
        # Check element symmetry and orientation behavior
        # ----------------------------------------------------

        check_element_stiffness(
            ke,
            c,
            s,
            e
        )

        # Global DOFs
        gdofs = element_dofs(
            nodes
        )

        # ----------------------------------------------------
        # Assemble local matrix into global matrix
        # ----------------------------------------------------

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

    print(
        "Element stiffness symmetry check: PASS"
    )

    print(
        "Horizontal/vertical element stiffness check: PASS"
    )

    return K


# ============================================================
# Build global force vector
# ============================================================

def build_force_vector(
    num_nodes,
    loads
):
    """
    Create the global external force vector:

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


# ============================================================
# Solve FEM system
# ============================================================

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
    # Initialize displacement vector
    # --------------------------------------------------------

    u = np.zeros(
        num_dofs
    )

    # --------------------------------------------------------
    # Apply prescribed displacements
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
    # Partition stiffness matrix
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

    # --------------------------------------------------------
    # Partition force vector
    # --------------------------------------------------------

    F_F = F[
        free_dofs
    ]

    u_E = u[
        fixed_dofs
    ]

    # --------------------------------------------------------
    # Insufficient constraints / singularity check
    # --------------------------------------------------------

    if len(free_dofs) > 0:

        condition_number = np.linalg.cond(
            K_FF
        )

        print(
            "K_FF condition number:",
            condition_number
        )

        if (
            not np.isfinite(condition_number)
            or condition_number > 1e12
        ):
            raise ValueError(
                "K_FF is singular or ill-conditioned. "
                "The structure may have insufficient constraints, "
                "a mechanism, or a disconnected node."
            )

    # --------------------------------------------------------
    # Solve:
    #
    # K_FF u_F = F_F - K_FE u_E
    # --------------------------------------------------------

    try:

        u_F = np.linalg.solve(
            K_FF,
            F_F - K_FE @ u_E
        )

    except np.linalg.LinAlgError:

        raise ValueError(
            "The system could not be solved because K_FF "
            "is singular. Check for insufficient constraints "
            "or a structural mechanism."
        )

    print(
        "Insufficient-constraint / singularity check: PASS"
    )

    # Insert free displacements
    u[free_dofs] = u_F

    return (
        u,
        free_dofs,
        fixed_dofs
    )


# ============================================================
# Recover reaction forces
# ============================================================

def recover_reactions(
    K,
    F,
    u,
    fixed_dofs
):
    """
    Calculate reaction forces:

        R = K u - F

    Free DOF reaction entries are set to zero.
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


# ============================================================
# Recover element internal forces
# ============================================================

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

        # Node coordinates
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

        # Global DOFs
        gdofs = element_dofs(
            nodes
        )

        # Element displacement vector
        u_e = u[
            gdofs
        ]

        # ----------------------------------------------------
        # Axial extension
        #
        # delta = [-c, -s, c, s] u_e
        # ----------------------------------------------------

        axial_extension = (
            -c * u_e[0]
            -s * u_e[1]
            +c * u_e[2]
            +s * u_e[3]
        )

        # ----------------------------------------------------
        # Axial element force
        # ----------------------------------------------------

        force = (
            k_values[e]
            * axial_extension
        )

        internal_forces.append(
            force
        )

    return internal_forces


# ============================================================
# Numerical solution checks
# ============================================================

def check_solution(
    K,
    F,
    u,
    reactions,
    free_dofs
):
    """
    Perform numerical checks:

    1. Global stiffness symmetry
    2. Free-DOF residual
    3. Free DOF reaction values
    4. Global force equilibrium
    """

    # --------------------------------------------------------
    # 1. Global symmetry
    #
    # ||K - K^T|| should be near zero
    # --------------------------------------------------------

    global_symmetry_error = \
        np.linalg.norm(
            K - K.T
        )

    print(
        "\nGlobal symmetry error ||K - K^T||:",
        global_symmetry_error
    )

    if np.allclose(
        K,
        K.T
    ):
        print(
            "Global stiffness symmetry check: PASS"
        )

    else:
        print(
            "Global stiffness symmetry check: FAIL"
        )

    # --------------------------------------------------------
    # 2. Free-DOF residual
    #
    # (K u - F)_F should be near zero
    # --------------------------------------------------------

    full_residual = \
        K @ u - F

    free_residual = \
        full_residual[
            free_dofs
        ]

    print(
        "Free-DOF residual:",
        free_residual
    )

    if np.allclose(
        free_residual,
        0
    ):
        print(
            "Free-DOF residual check: PASS"
        )

    else:
        print(
            "Free-DOF residual check: FAIL"
        )

    # --------------------------------------------------------
    # Free DOF reaction output should be zero
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
    # 3. Global equilibrium
    #
    # Applied loads + reactions = 0
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

    print(
        "Global equilibrium residual [Fx, Fy]:",
        equilibrium
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


# ============================================================
# Write output files
# ============================================================

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


# ============================================================
# Main
# ============================================================

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
    # Convert numerical quantities
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
    # Validate inputs
    # --------------------------------------------------------

    validate_inputs(
        coordinates,
        connectivity_raw,
        k_values,
        loads,
        prescribed_displacements
    )

    # Convert connectivity only after validating
    # that all entries are integers
    connectivity = np.array(
        connectivity_raw,
        dtype=int
    )

    num_nodes = len(
        coordinates
    )

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
    # Solve system
    # --------------------------------------------------------

    u, free_dofs, fixed_dofs = \
        solve_system(
            K,
            F,
            prescribed_displacements
        )

    # --------------------------------------------------------
    # Recover reactions
    # --------------------------------------------------------

    reactions = \
        recover_reactions(
            K,
            F,
            u,
            fixed_dofs
        )

    # --------------------------------------------------------
    # Recover internal element forces
    # --------------------------------------------------------

    internal_forces = \
        recover_element_forces(
            coordinates,
            connectivity,
            k_values,
            u
        )

    # --------------------------------------------------------
    # Numerical validation checks
    # --------------------------------------------------------

    check_solution(
        K,
        F,
        u,
        reactions,
        free_dofs
    )

    # --------------------------------------------------------
    # Reshape nodal output data
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
