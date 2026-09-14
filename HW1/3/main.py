import numpy as np
import ast


# Reading input files

def read_input(filename):
    with open(filename, "r") as file:
        return ast.literal_eval(file.read())


# Assembling Element stiffness matrix

def element_stiffness(k):
    """
    Return the 2x2 stiffness matrix for one spring.
    """

    return k * np.array([
        [1, -1],
        [-1, 1]
    ])


# Assembling global stiffness matrix

def assemble_global_stiffness(num_nodes, connectivity, k_values):
    """
    Assemble the global stiffness matrix K.
    """

    K = np.zeros((num_nodes, num_nodes))

    for e in range(len(connectivity)):

        # Global nodes for this element
        nodes = connectivity[e]

        # Element stiffness matrix
        ke = element_stiffness(k_values[e])

        # Map local nodes to global nodes
        for local_row in range(2):
            global_row = nodes[local_row]

            for local_col in range(2):
                global_col = nodes[local_col]

                # Add local contribution to global matrix
                K[global_row, global_col] += ke[local_row, local_col]

    return K



# Building global force vector

def build_force_vector(num_nodes, loads):
    """
    Create the global external force vector F.
    """

    F = np.zeros(num_nodes)

    for i in range(num_nodes):
        F[i] = loads[i]

    return F



# Solving FEM system

def solve_system(K, F, prescribed_displacements):
    """
    Apply displacement boundary conditions and solve
    for the nodal displacements.
    """

    num_nodes = len(F)

    # Find prescribed and free DOFs
    fixed_dofs = []
    free_dofs = []

    for i in range(num_nodes):

        if prescribed_displacements[i] is None:
            free_dofs.append(i)
        else:
            fixed_dofs.append(i)

    # Convert to numpy arrays
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    free_dofs = np.array(free_dofs, dtype=int)

    # Initialize displacement vector
    u = np.zeros(num_nodes)

    # Apply prescribed displacements
    for i in fixed_dofs:
        u[i] = prescribed_displacements[i]

    # Partition stiffness matrix
    K_FF = K[np.ix_(free_dofs, free_dofs)]
    K_FE = K[np.ix_(free_dofs, fixed_dofs)]

    # Partition force vector
    F_F = F[free_dofs]

    # Prescribed displacements
    u_E = u[fixed_dofs]

    # Solve:
    #
    # K_FF u_F = F_F - K_FE u_E
    #
    u_F = np.linalg.solve(
        K_FF,
        F_F - K_FE @ u_E
    )

    # Put free displacements into global vector
    u[free_dofs] = u_F

    return u, free_dofs, fixed_dofs


# Recovering reaction forces

def recover_reactions(K, F, u, fixed_dofs):
    """
    Calculate reaction forces using:

        R = K u - F
    """

    reactions = K @ u - F

    # Only reactions at prescribed DOFs
    reaction_output = np.zeros(len(F))

    for i in fixed_dofs:
        reaction_output[i] = reactions[i]

    return reaction_output



# Recovering element internal forces


def recover_element_forces(u, connectivity, k_values):
    """
    Calculate extension/compression and internal force
    for every spring element.
    """

    internal_forces = []

    for e in range(len(connectivity)):

        # Global nodes for this element
        i = connectivity[e][0]
        j = connectivity[e][1]

        # Element displacement
        delta_u = u[j] - u[i]

        # Internal spring force
        force = k_values[e] * delta_u

        internal_forces.append(force)

    return internal_forces


# Checking solution


def check_solution(K, F, u, reactions, free_dofs):
    """
    Check symmetry and equilibrium.
    """

    # Check that K is symmetric
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

    # Check free DOFs have approximately zero reaction
    if np.allclose(reactions[free_dofs], 0):
        print("Free DOF reaction check: PASS")
    else:
        print("Free DOF reaction check: FAIL")


# Writing output files

def write_output(filename, values):

    with open(filename, "w") as file:
        file.write(str([float(value) for value in values]))


# Main

def main():

    # Read input files
    connectivity = read_input("connectivity_array.txt")
    k_values = read_input("element_stiffnesses.txt")
    loads = read_input("external_nodal_forces.txt")
    prescribed_displacements = read_input(
        "displacement_BCs.txt"
    )

    # Number of nodes
    num_nodes = len(loads)

    # Build force vector
    F = build_force_vector(num_nodes, loads)

    # Assemble global stiffness matrix
    K = assemble_global_stiffness(
        num_nodes,
        connectivity,
        k_values
    )

    # Solve for nodal displacements
    u, free_dofs, fixed_dofs = solve_system(
        K,
        F,
        prescribed_displacements
    )

    # Calculate reactions
    reactions = recover_reactions(
        K,
        F,
        u,
        fixed_dofs
    )

    # Calculate element forces
    internal_forces = recover_element_forces(
        u,
        connectivity,
        k_values
    )

    # Check solution
    check_solution(
        K,
        F,
        u,
        reactions,
        free_dofs
    )

    # Write output files
    write_output(
        "nodal displacements.txt",
        u
    )

    write_output(
        "reaction forces.txt",
        reactions
    )

    write_output(
        "internal forces.txt",
        internal_forces
    )

    # Print results
    print("\nGlobal stiffness matrix:")
    print(K)

    print("\nNodal displacements:")
    print(u)

    print("\nReaction forces:")
    print(reactions)

    print("\nInternal element forces:")
    print(internal_forces)


if __name__ == "__main__":
    main()
