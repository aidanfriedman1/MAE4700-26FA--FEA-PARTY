"""Explore Jacobian variation in four- and eight-node quadrilateral elements.

Complete each location marked by ``NotImplementedError`` before running the
script. The completed script plots each element and the determinant of its
Jacobian in both the parent and physical domains.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def main():
    plt.rcParams.update(
        {
            "lines.linewidth": 3,
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.labelweight": "bold",
        }
    )

    # Q4 element: show that det(J) can be negative for a concave element.
    xy_q4 = np.array(
        [[2.0, 1.0], [0.0, 0.0], [1.0, 0.0], [2.0, -1.0]]
    )
    calculate_and_plot(xy_q4, "Q4 element")

    # Q8 element: move corner or midside nodes to explore the mapping.
    xy_q8 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ]
    )
    calculate_and_plot(xy_q8, "Q8 element")

    plt.show()


def calculate_and_plot(xy, title):
    """Calculate det(J) on a grid and make the plots for one element."""
    nodes_per_element = xy.shape[0]
    plot_element_shape(xy, title)

    points = np.linspace(-1.0, 1.0, 100)
    xi_grid, eta_grid = np.meshgrid(points, points)
    det_j = np.zeros_like(xi_grid)
    x_grid = np.zeros_like(xi_grid)
    y_grid = np.zeros_like(xi_grid)

    for row in range(xi_grid.shape[0]):
        for col in range(xi_grid.shape[1]):
            xi = xi_grid[row, col]
            eta = eta_grid[row, col]

            jacobian = jacobian_matrix(xi, eta, xy)
            det_j[row, col] = np.linalg.det(jacobian)

            shape = shape_functions(xi, eta, nodes_per_element)
            x_grid[row, col], y_grid[row, col] = physical_coordinates(shape, xy)

    plot_jacobian(
        xi_grid, eta_grid, x_grid, y_grid, det_j, title
    )


def plot_element_shape(xy, title):
    """Plot the element boundary, including curved sides for a Q8 element."""
    nodes_per_element = xy.shape[0]
    fig, ax = plt.subplots()

    if nodes_per_element == 4:
        closed_xy = np.vstack((xy, xy[0]))
        ax.plot(closed_xy[:, 0], closed_xy[:, 1], "o-")
    elif nodes_per_element == 8:
        edge = np.linspace(-1.0, 1.0, 10)
        boundary_xi = np.concatenate(
            (edge, np.ones_like(edge), -edge, -np.ones_like(edge))
        )
        boundary_eta = np.concatenate(
            (-np.ones_like(edge), edge, np.ones_like(edge), -edge)
        )

        boundary_xy = np.zeros((boundary_xi.size, 2))
        for index, (xi, eta) in enumerate(zip(boundary_xi, boundary_eta)):
            shape = shape_functions(xi, eta, nodes_per_element)
            boundary_xy[index, :] = shape @ xy
        ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], "o-")
    else:
        raise ValueError("Only Q4 and Q8 elements are supported.")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} shape")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()


def physical_coordinates(shape, xy):
    """Map one parent-domain point to physical coordinates."""
    # x = sum_i N_i x_i,  y = sum_i N_i y_i
    # shape has length n, xy is n-by-2 -> returns [x, y]
    return shape @ xy


def shape_functions(xi, eta, nodes_per_element):
    """Return the Q4 or Q8 shape-function array."""
    if nodes_per_element == 4:
        return 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        )

    if nodes_per_element == 8:
        # Serendipity Q8: corners 1-4, then midsides 5 (bottom), 6 (right),
        # 7 (top), 8 (left).
        return np.array(
            [
                -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
                0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
                0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
                0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
                0.5 * (1.0 - xi**2) * (1.0 - eta),
                0.5 * (1.0 + xi) * (1.0 - eta**2),
                0.5 * (1.0 - xi**2) * (1.0 + eta),
                0.5 * (1.0 - xi) * (1.0 - eta**2),
            ]
        )

    raise ValueError("Only Q4 and Q8 elements are supported.")


def shape_function_gradients(xi, eta, nodes_per_element):
    """Return shape-function gradients with respect to xi and eta."""
    if nodes_per_element == 4:
        # Each row is [dN_i/dxi, dN_i/deta]; transpose to get 2-by-4.
        return 0.25 * np.array(
            [
                [-(1.0 - eta), -(1.0 - xi)],
                [(1.0 - eta), -(1.0 + xi)],
                [(1.0 + eta), (1.0 + xi)],
                [-(1.0 + eta), (1.0 - xi)],
            ]
        ).T

    if nodes_per_element == 8:
        # Each row below is [dN_i/dxi, dN_i/deta]; transpose to obtain
        # the 2-by-8 gradient matrix used in the Jacobian calculation.
        return np.array(
            [
                [
                    -0.25 * (-1.0 + eta) * (2.0 * xi + eta),
                    -0.25 * (-1.0 + xi) * (xi + 2.0 * eta),
                ],
                [
                    0.25 * (-1.0 + eta) * (-2.0 * xi + eta),
                    0.25 * (1.0 + xi) * (-xi + 2.0 * eta),
                ],
                [
                    0.25 * (1.0 + eta) * (2.0 * xi + eta),
                    0.25 * (1.0 + xi) * (xi + 2.0 * eta),
                ],
                [
                    -0.25 * (1.0 + eta) * (-2.0 * xi + eta),
                    -0.25 * (-1.0 + xi) * (-xi + 2.0 * eta),
                ],
                [xi * (-1.0 + eta), 0.5 * (1.0 + xi) * (-1.0 + xi)],
                [-0.5 * (1.0 + eta) * (-1.0 + eta), -eta * (1.0 + xi)],
                [-xi * (1.0 + eta), -0.5 * (1.0 + xi) * (-1.0 + xi)],
                [0.5 * (1.0 + eta) * (-1.0 + eta), eta * (-1.0 + xi)],
            ]
        ).T

    raise ValueError("Only Q4 and Q8 elements are supported.")


def jacobian_matrix(xi, eta, xy):
    """Return the Jacobian matrix at one parent-domain point."""
    gradients = shape_function_gradients(xi, eta, xy.shape[0])
    # (2 x n) @ (n x 2) -> [[dx/dxi, dy/dxi], [dx/deta, dy/deta]]
    return gradients @ xy


def plot_jacobian(xi, eta, x, y, det_j, title):
    """Plot det(J) in the parent and physical domains."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    parent_plot = axes[0].pcolormesh(xi, eta, det_j, shading="auto")
    fig.colorbar(parent_plot, ax=axes[0])
    axes[0].set_xlabel(r"$\xi$")
    axes[0].set_ylabel(r"$\eta$")
    axes[0].set_title("Parent domain")
    axes[0].set_aspect("equal", adjustable="box")

    # Construct the mapped cells explicitly. This preserves folded or inverted
    # cells, which are precisely the invalid mappings this script explores.
    polygons = []
    cell_values = []
    for row in range(x.shape[0] - 1):
        for col in range(x.shape[1] - 1):
            polygons.append(
                [
                    (x[row, col], y[row, col]),
                    (x[row, col + 1], y[row, col + 1]),
                    (x[row + 1, col + 1], y[row + 1, col + 1]),
                    (x[row + 1, col], y[row + 1, col]),
                ]
            )
            cell_values.append(
                np.mean(det_j[row : row + 2, col : col + 2])
            )

    physical_plot = PolyCollection(
        polygons,
        array=np.asarray(cell_values),
        edgecolors="none",
    )
    physical_plot.set_clim(det_j.min(), det_j.max())
    axes[1].add_collection(physical_plot)
    axes[1].autoscale_view()
    fig.colorbar(physical_plot, ax=axes[1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Physical domain")
    axes[1].set_aspect("equal", adjustable="box")

    fig.suptitle(f"{title}: determinant of the Jacobian")
    fig.tight_layout()


if __name__ == "__main__":
    main()