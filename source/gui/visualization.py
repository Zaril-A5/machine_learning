import tkinter as tk
from tkinter import ttk


class VisualizationPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # --- Title ---
        ttk.Label(
            self,
            text="Route Visualization",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        # --- Canvas ---
        self.canvas = tk.Canvas(
            self,
            width=700,
            height=250,
            bg="white"
        )

        self.canvas.pack(padx=10, pady=10)

    def draw_route(self, route_nodes):
        """
        Draw route visually on canvas.

        Parameters:
        - route_nodes: list of node names
          Example: ["3001", "3005", "3010"]
        """

        # Clear previous drawing
        self.canvas.delete("all")

        if not route_nodes:
            return

        # Layout settings
        start_x = 80
        y = 120
        spacing = 120
        radius = 20

        positions = []

        # --- Draw Nodes ---
        for i, node in enumerate(route_nodes):
            x = start_x + (i * spacing)

            positions.append((x, y))

            # Node circle
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="lightblue",
                outline="black",
                width=2
            )

            # Node label
            self.canvas.create_text(
                x,
                y,
                text=node,
                font=("Arial", 10, "bold")
            )

        # --- Draw Connections ---
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]

            self.canvas.create_line(
                x1 + radius,
                y1,
                x2 - radius,
                y2,
                arrow=tk.LAST,
                width=2
            )