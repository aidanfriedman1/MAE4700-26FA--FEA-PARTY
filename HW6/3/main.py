from ast import literal_eval
from pathlib import Path
import numpy as np


def read_literal(filename):
    """Read the raw Python literal contained in an input file."""
    return literal_eval(Path(filename).read_text(encoding="utf-8").strip())


def read_inputs():
    """Load and check the eight standardized problem inputs."""
    x_end = np.asarray(read_literal("nodal coordinates.txt"), dtype=float)
    end_connectivity = read_literal("connectivity array.txt")
    nnpe = read_literal("nodes per element.txt")
    nq = read_literal("num quadrature points.txt")
    p_expr = read_literal("p function.txt")
    f_expr = read_literal("f function.txt")
    point_sources = read_literal("point sources.txt")
    boundary_conditions = read_literal("boundary conditions.txt")

    if x_end.ndim != 1 or len(x_end) < 2 or not np.all(np.isfinite(x_end)):
        raise ValueError("Provide at least two finite end-node coordinates.")
    if len(np.unique(x_end)) != len(x_end):
        raise ValueError("Input end-node coordinates must be distinct.")
    if type(nnpe) is not int or nnpe not in (2, 3, 4):
        raise ValueError("nodes per element must be 2, 3, or 4.")
    if type(nq) is not int or not 1 <= nq <= 5:
        raise ValueError("num quadrature points must be an integer from 1 to 5.")
    if not isinstance(p_expr, str) or not isinstance(f_expr, str):
        raise ValueError("p and f must be NumPy expression strings.")
    if not isinstance(end_connectivity, list) or not end_connectivity:
        raise ValueError("connectivity array must be a nonempty list.")
    for pair in end_connectivity:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(i) is not int or i < 0 or i >= len(x_end) for i in pair)):
            raise ValueError("Each connectivity entry must contain two valid zero-based indices.")
        if not x_end[pair[1]] > x_end[pair[0]]:
            raise ValueError("Each element must list its left endpoint before its right endpoint.")

    if not isinstance(point_sources, list):
        raise ValueError("point sources must be a list.")
    for source in point_sources:
        if not isinstance(source, (list, tuple)) or len(source) != 2:
            raise ValueError("Each point source must be [location, value].")
        xp, value = (float(v) for v in source)
        if not np.isfinite(xp) or not np.isfinite(value):
            raise ValueError("Point sources must contain finite numbers.")
        if xp < min(x_end) or xp > max(x_end):
            raise ValueError("Point source lies outside the domain.")

    if not isinstance(boundary_conditions, list) or len(boundary_conditions) != 2:
        raise ValueError("Provide one boundary condition for each domain endpoint.")
    for bc in boundary_conditions:
        if (not isinstance(bc, (list, tuple)) or len(bc) != 2
                or bc[0] not in ("essential", "natural")
                or not np.isfinite(float(bc[1]))):
            raise ValueError("Boundary conditions must be ['essential' or 'natural', value].")

    return (x_end, end_connectivity, nnpe, nq, p_expr, f_expr,
            point_sources, boundary_conditions)


def build_computational_mesh(x_end, end_connectivity, nnpe):
    """Keep input end-node indices; append evenly spaced interior nodes."""
    x_all = list(x_end)
    full_connectivity = []
    for left, right in end_connectivity:
        local = [left]
        for x in np.linspace(x_end[left], x_end[right], nnpe)[1:-1]:
            local.append(len(x_all))
            x_all.append(float(x))
        local.append(right)
        full_connectivity.append(local)
    return np.asarray(x_all, dtype=float), full_connectivity


def reference_nodes(nnpe):
    """Equally spaced Lagrange nodes on the reference interval [-1, 1]."""
    return np.linspace(-1.0, 1.0, nnpe)


def shape_functions(xi, nnpe):
    """Lagrange interpolation functions N_i(xi)."""
    nodes = reference_nodes(nnpe)
    N = np.ones(nnpe)
    for i in range(nnpe):
        for j in range(nnpe):
            if j != i:
