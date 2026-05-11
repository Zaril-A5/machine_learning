import tkinter as tk
from tkinter import ttk

from input_panel import InputPanel
from result_panel import ResultPanel
from visualization import VisualizationPanel


class MainWindow:
    def __init__(self, root):
        self.root = root

        self.root.title("Traffic Route Finder")
        self.root.geometry("900x900")

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

        # TODO:
        # Replace this with real integration:
        # - predictor.py
        # - route_finder.py
        # - travel_time_estimator.py

        routes = [
            {
                "route": f"{origin} → 3001 → 3005 → {destination}",
                "time": "12 mins",
                "model": model
            },
            {
                "route": f"{origin} → 3010 → {destination}",
                "time": "15 mins",
                "model": model
            },
            {
                "route": f"{origin} → 3020 → 3025 → {destination}",
                "time": "18 mins",
                "model": model
            },
            {
                "route": f"{origin} → 3030 → {destination}",
                "time": "20 mins",
                "model": model
            },
            {
                "route": f"{origin} → 3040 → 3050 → {destination}",
                "time": "25 mins",
                "model": model
            }
        ]

        self.result_panel.display_results(routes)

        # Visualize first route (placeholder)
        sample_route_nodes = [
            origin,
            "3001",
            "3005",
            destination
        ]

        self.visual_panel.draw_route(sample_route_nodes)

        self.status_label.config(
            text=f"Showing routes from {origin} to {destination}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()