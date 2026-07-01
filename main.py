import numpy as np
import json
from scipy.spatial import Voronoi, Delaunay
from scipy.ndimage import binary_dilation, gaussian_filter

from map_manager import load_current_map
from voronoi import generate_voronoi_diagram
from marker import isolate_buildable_plot, mark_path_and_perimeter, generate_zones
from visualizer_server import force_serve
from terraformer import apply_terraforming
from plotter import find_modular_plots, visualize_house_volumes
from builder import deploy_settlement

if __name__ == "__main__":
    load_current_map()
    generate_voronoi_diagram()
    if not isolate_buildable_plot():
        raise RuntimeError("No buildable settlement core was found; stopping before path marking.")
    generate_zones(num_zones=4)
    mark_path_and_perimeter()
    apply_terraforming(plot_setback=2)
    find_modular_plots(module_size=8, setback=2)
    visualize_house_volumes(module_size=8, max_floors=3)
    force_serve("data/settlement_viz.npz")
    deploy_settlement()
