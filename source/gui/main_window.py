import tkinter as tk
from tkinter import ttk

from input_panel import InputPanel
from result_panel import ResultPanel
from visualization import VisualizationPanel

from ml_models.predictor import TrafficPredictor
from integration.traffic_route_finder import TrafficRouteFinder
from data_processing.graph_builder import GraphBuilder
from pathfinding.route_finder import TopKRouteFinder


class MainWindow:
    def __init__(self, root):
        self.root = root

        self.root.title("Traffic Route Finder")
        self.root.geometry("900x900")

        # --- Backend Initialization ---

        # TODO:
        # Replace with real graph loader from Member 1

        self.graph = None
        self.traffic_finder = None
        self.route_finder = None
        self.predictor = None

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
        Placeholder route-finding logic.
        """

        origin = data["origin"]
        destination = data["destination"]
        time = data["time"]
        model = data["model"]

        self.status_label.config(text="Finding routes...")

        # Placeholder until backend fully connected
        routes = []

        try:
            # TODO:
            # Replace with real backend integration

            # Example future integration:
            #
            # real_routes = self.route_finder.find_top_k_paths(
            #     int(origin),
            #     int(destination),
            #     timestamp=time
            # )

            # Temporary mock result
            routes = [
                {
                    "route": f"{origin} → 3001 → 3005 → {destination}",
                    "time": "12 mins",
                    "model": model
                }
            ]

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
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