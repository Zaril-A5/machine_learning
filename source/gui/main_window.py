import sys
import os
import json

# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import tkinter as tk
from tkinter import ttk

from input_panel import InputPanel
from result_panel import ResultPanel
from visualization import VisualizationPanel

from integration.traffic_route_finder import TrafficRouteFinder
from data_processing.graph_builder import GraphBuilder
from pathfinding.route_finder import TopKRouteFinder

from data_processing.data_loader import TrafficLocationLoader
from ml_models.predictor import TrafficPredictor
from data_processing.traffic_flow_loader import TrafficFlowLoader


class MainWindow:
    def __init__(self, root):
        self.root = root

        self.root.title("Traffic Route Finder")
        self.root.geometry("900x900")

        # --- Backend Initialization ---

        try:
            # Load traffic locations
            self.loader = TrafficLocationLoader(
                "../data/raw/Traffic_Count_Locations_with_LONG_LAT.csv"
            )

            self.loader.load_locations()

            # Load historical traffic flow data
            self.flow_loader = TrafficFlowLoader(
                "../data/raw/Scats Data October 2006.xls"
            )

            self.flow_loader.load_flow_data()

            # Build graph
            self.graph = GraphBuilder(self.loader)

            # Add graph nodes
            self.graph.add_nodes_from_locations()

            # Connect nearby intersections
            self.graph.add_edges_between_nearby_sites(20.0)

            self.graph.display_graph_info()

            # Load predictor
            self.predictor = TrafficPredictor()

            # Give historical traffic data to predictor
            self.predictor.historical_data = self.flow_loader.flow_data

            # Load config file
            with open("../config.json", "r") as f:
                config = json.load(f)

            # Traffic route integration
            self.traffic_finder = TrafficRouteFinder(
                config,
                self.graph
            )

            self.traffic_finder.set_ml_predictor(
                self.predictor
            )

            # Top-K route finder
            self.route_finder = TopKRouteFinder(
                self.graph,
                self.traffic_finder
            )

            print("[OK] Backend initialized successfully.")

        except Exception as e:
            print(f"[ERROR] Backend initialization failed: {e}")

        # --- Main Container ---
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # --- Input Panel ---
        self.input_panel = InputPanel(
            main_frame,
            self.find_route
        )

        self.input_panel.pack(fill="x", pady=10)

        # --- Result Panel ---
        self.result_panel = ResultPanel(main_frame)
        self.result_panel.pack(fill="both", expand=True)

        # --- Visualization Panel ---
        self.visual_panel = VisualizationPanel(main_frame)
        self.visual_panel.pack(fill="both", expand=True, pady=10)

        # --- Status Bar ---
        self.status_label = ttk.Label(
            root,
            text="Ready",
            anchor="w"
        )

        self.status_label.pack(fill="x", side="bottom")

    def find_route(self, data):
        """
        Route-finding logic.
        """

        origin = data["origin"]
        destination = data["destination"]
        time = data["time"]
        model = data["model"]

        self.status_label.config(text="Finding routes...")

        try:
            real_paths = self.route_finder.find_top_5_paths(
                int(origin),
                int(destination)
            )

            formatted_paths = [
                (path, round(cost, 2))
                for path, cost in real_paths
            ]

            print("REAL PATHS:", formatted_paths)
            print(f"Origin: {origin}")
            print(f"Destination: {destination}")

            routes = []

            for path, cost in real_paths:
                route_string = " → ".join(
                    [str(node) for node in path]
                )

                routes.append({
                    "route": route_string,
                    "time": f"{cost:.2f} sec",
                    "model": model
                })

        except Exception as e:
            print(f"[ROUTING ERROR] {e}")

            self.status_label.config(
                text=f"Routing Error: {str(e)}"
            )

            return

        self.result_panel.display_results(routes)

        if routes:
            first_route = routes[0]["route"]

            route_nodes = [
                node.strip()
                for node in first_route.split("→")
            ]

            self.visual_panel.draw_route(route_nodes)

        self.status_label.config(
            text=f"Showing routes from {origin} to {destination}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()