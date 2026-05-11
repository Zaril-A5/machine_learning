import tkinter as tk
from tkinter import ttk


class ResultPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # --- Title ---
        ttk.Label(
            self,
            text="Top 5 Routes",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        # --- Table Frame ---
        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True)

        # --- Scrollbar ---
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side="right", fill="y")

        # --- Treeview Table ---
        self.tree = ttk.Treeview(
            table_frame,
            columns=("Route", "Time", "Model"),
            show="headings",
            yscrollcommand=scrollbar.set,
            height=8
        )

        scrollbar.config(command=self.tree.yview)

        # --- Columns ---
        self.tree.heading("Route", text="Route")
        self.tree.heading("Time", text="Estimated Time")
        self.tree.heading("Model", text="Model")

        self.tree.column("Route", width=400)
        self.tree.column("Time", width=120)
        self.tree.column("Model", width=100)

        self.tree.pack(fill="both", expand=True)

    def display_results(self, routes):
        """
        Display route results.

        routes = list of dictionaries:
        [
            {
                "route": "...",
                "time": "...",
                "model": "..."
            }
        ]
        """

        # Clear old results
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert new results
        for route in routes:
            self.tree.insert(
                "",
                "end",
                values=(
                    route["route"],
                    route["time"],
                    route["model"]
                )
            )