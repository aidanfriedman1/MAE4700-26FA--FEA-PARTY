"""Structured T3 or Q4 meshes for rectangular two-dimensional domains.
This is the Python counterpart of ``BoxGrid_2D.m``. Node and element indices
are zero-based. The command-line interface writes the standard input files
used by Homework 10, Problem 2.
"""
from argparse import ArgumentParser
from pathlib import Path
def box_grid_2d(
x_left,
x_right,
y_bottom,
y_top,
nodes_per_element,
num_x,
num_y,
):
"""Return coordinates, connectivity, and boundary data for a rectangle.
``num_x`` and ``num_y`` are the numbers of rectangular subdivisions.
A Q4 mesh contains ``num_x * num_y`` elements. A T3 mesh contains twice
that number because each rectangle is divided along its southwest-to-
northeast diagonal.
"""
if nodes_per_element not in (3, 4):
raise ValueError("nodes_per_element must be 3 (T3) or 4 (Q4).")
if num_x < 1 or num_y < 1:
raise ValueError("num_x and num_y must both be positive.")
if x_right <= x_left or y_top <= y_bottom:
raise ValueError("The upper coordinate limits must exceed the lower
limits.")
x_values = [
x_left + (x_right - x_left) * i / num_x
for i in range(num_x + 1)
]
y_values = [
y_bottom + (y_top - y_bottom) * j / num_y
for j in range(num_y + 1)
]
nodal_coordinates = [
[x_values[i], y_values[j]]
for j in range(num_y + 1)
for i in range(num_x + 1)
]
connectivity_array = []
for j in range(num_y):
for i in range(num_x):
southwest = j * (num_x + 1) + i
southeast = southwest + 1
northwest = (j + 1) * (num_x + 1) + i
northeast = northwest + 1
if nodes_per_element == 4:
connectivity_array.append(
[southwest, southeast, northeast, northwest]
)
else:
connectivity_array.append([southwest, southeast, northwest])
connectivity_array.append([northeast, northwest, southeast])
boundary_nodes = [
list(range(num_x + 1)),
[(j + 1) * (num_x + 1) - 1 for j in range(num_y + 1)],
[num_y * (num_x + 1) + i for i in range(num_x + 1)],
[j * (num_x + 1) for j in range(num_y + 1)],
]
if nodes_per_element == 4:
boundary_elements = [
list(range(num_x)),
[(j + 1) * num_x - 1 for j in range(num_y)],
[(num_y - 1) * num_x + i for i in range(num_x)],
[j * num_x for j in range(num_y)],
]
surface_indicators = [
[-2] * num_x,
[1] * num_y,
[2] * num_x,
[-1] * num_y,
]
else:
boundary_elements = [
[2 * i for i in range(num_x)],
[2 * (j + 1) * num_x - 1 for j in range(num_y)],
[2 * ((num_y - 1) * num_x + i) + 1 for i in range(num_x)],
[2 * j * num_x for j in range(num_y)],
]
surface_indicators = [
[-2] * num_x,
[-1] * num_y,
[-2] * num_x,
[-1] * num_y,
]
boundary_data = {
"nodes": boundary_nodes,
"elements": boundary_elements,
"surface_indicators": surface_indicators,
}
return nodal_coordinates, connectivity_array, boundary_data
def write_torsion_input_files(
output_directory,
nodal_coordinates,
connectivity_array,
boundary_data,
g_alpha,
):
"""Write the standard Problem 2 inputs for a rectangular shaft.
Every exterior node receives the required value ``phi = 0``. Interior
entries in ``essential_BCs.txt`` are written as ``None``.
"""
output_path = Path(output_directory)
output_path.mkdir(parents=True, exist_ok=True)
prescribed_values = [None] * len(nodal_coordinates)
for side_nodes in boundary_data["nodes"]:
for node in side_nodes:
prescribed_values[node] = 0.0
files = {
"nodal_coordinates.txt": nodal_coordinates,
"connectivity_array.txt": connectivity_array,
"essential_BCs.txt": prescribed_values,
"problem_parameters.txt": [float(g_alpha)] * len(connectivity_array),
}
for filename, values in files.items():
(output_path / filename).write_text(repr(values) + "\n", encoding="utf-8")
def parse_arguments():
parser = ArgumentParser(
description="Generate a rectangular T3 or Q4 mesh for HW10 Problem 2."
)
parser.add_argument("--x-left", type=float, required=True)
parser.add_argument("--x-right", type=float, required=True)
parser.add_argument("--y-bottom", type=float, required=True)
parser.add_argument("--y-top", type=float, required=True)
parser.add_argument("--nodes-per-element", type=int, choices=(3, 4),
required=True)
parser.add_argument("--num-x", type=int, required=True)
parser.add_argument("--num-y", type=int, required=True)
parser.add_argument("--g-alpha", type=float, default=2.316e10)
parser.add_argument("--output-directory", default=".")
return parser.parse_args()
def main():
arguments = parse_arguments()
mesh = box_grid_2d(
arguments.x_left,
arguments.x_right,
arguments.y_bottom,
arguments.y_top,
arguments.nodes_per_element,
arguments.num_x,
arguments.num_y,
)
write_torsion_input_files(
arguments.output_directory,
*mesh,
arguments.g_alpha,
)
if __name__ == "__main__":
mai
