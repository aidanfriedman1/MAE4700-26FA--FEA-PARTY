"""
main.py

Solves 1D systems of springs (linear-elastic spring finite elements).

Reads four input files (raw Python list literals):
    connectivity_array.txt   -> list of [node_i, node_j] per element
    element_stiffnesses.txt  -> list of float stiffnesses per element
    external_nodal_forces.txt-> list of float forces per node
    displacement_BCs.txt     -> list of float-or-None per node

Writes three output files (raw Python list literals):
    nodal_displacements.txt  -> list of float displacements per node
    reaction_forces.txt      -> list of float reaction forces per node
    internal_forces.txt      -> list of float internal (axial) forces per element

All file paths are resolved relative to the location of this script, so the
code runs identically regardless of the current working directory or OS.
"""

import ast
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path handling (relative to this file, per assignment requirement)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILES = {
    "connectivity": SCRIPT_DIR / "connectivity array.txt",
    "stiffnesses": SCRIPT_DIR / "element stiffnesses.txt",
    "forces": SCRIPT_DIR / "external nodal forces.txt",
    "bcs": SCRIPT_DIR / "displacement BCs.txt",
}

OUTPUT_FILES = {
    "displacements": SCRIPT_DIR / "nodal displacements.txt",
    "reactions": SCRIPT_DIR / "reaction forces.txt",
    "internal_forces": SCRIPT_DIR / "internal forces.txt",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_list_file(path: Path) -> list:
    """Read a text file whose entire content is a raw Python list literal."""
    text = path.read_text().strip()
    data = ast.literal_eval(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path.name}, got {type(data)}")
    return data


def write_list_file(path: Path, data: list) -> None:
    """Write a Python list of floats to a text file as a raw list literal."""
    # Ensure plain python floats (not numpy scalar types) for a clean, valid
    # Python list literal in the output file.
    clean = [float(x) for x in data]
    path.write_text(str(clean))


# ---------------------------------------------------------------------------
# Finite element assembly and solve
# ---------------------------------------------------------------------------
def assemble_global_stiffness(connectivity, stiffnesses, n_nodes):
    """Assemble the global stiffness matrix for a 1D spring system."""
    K = np.zeros((n_nodes, n_nodes))
    for (node_i, node_j), k in zip(connectivity, stiffnesses):
        k_local = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
        dofs = [node_i, node_j]
        for a in range(2):
            for b in range(2):
                K[dofs[a], dofs[b]] += k_local[a, b]
    return K


def solve_displacements(K, forces, bcs):
    """
    Solve K u = F for nodal displacements, given prescribed displacement
    boundary conditions (None entries in `bcs` are free DOFs).
    """
    n_nodes = len(forces)
    forces = np.asarray(forces, dtype=float)

    is_prescribed = np.array([bc is not None for bc in bcs])
    is_free = ~is_prescribed

    u = np.zeros(n_nodes)
    u[is_prescribed] = np.array(
        [bc for bc in bcs if bc is not None], dtype=float
    )

    if np.any(is_free):
        K_ff = K[np.ix_(is_free, is_free)]
        K_fp = K[np.ix_(is_free, is_prescribed)]
        F_f = forces[is_free]
        u_p = u[is_prescribed]

        rhs = F_f - K_fp @ u_p
        u_f = np.linalg.solve(K_ff, rhs)
        u[is_free] = u_f

    return u


def compute_reactions(K, u, forces):
    """Reaction forces at every node: R = K u - F_applied."""
    forces = np.asarray(forces, dtype=float)
    return K @ u - forces


def compute_internal_forces(connectivity, stiffnesses, u):
    """
    Axial (internal) force in each spring element:
        f_elem = k * (u_j - u_i)
    Positive value = element is in tension.
    """
    internal_forces = []
    for (node_i, node_j), k in zip(connectivity, stiffnesses):
        f_elem = k * (u[node_j] - u[node_i])
        internal_forces.append(f_elem)
    return internal_forces


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    connectivity = read_list_file(INPUT_FILES["connectivity"])
    stiffnesses = read_list_file(INPUT_FILES["stiffnesses"])
    forces = read_list_file(INPUT_FILES["forces"])
    bcs = read_list_file(INPUT_FILES["bcs"])

    n_nodes = len(forces)

    K = assemble_global_stiffness(connectivity, stiffnesses, n_nodes)
    u = solve_displacements(K, forces, bcs)
    reactions = compute_reactions(K, u, forces)
    internal_forces = compute_internal_forces(connectivity, stiffnesses, u)

    write_list_file(OUTPUT_FILES["displacements"], u)
    write_list_file(OUTPUT_FILES["reactions"], reactions)
    write_list_file(OUTPUT_FILES["internal_forces"], internal_forces)


if __name__ == "__main__":
    main()