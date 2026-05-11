from tkinter import ttk
from tkinter import messagebox


class InputPanel(ttk.Frame):
    def __init__(self, parent, on_find_route):
        super().__init__(parent, padding=15)

        self.on_find_route = on_find_route

        # --- Title ---
        ttk.Label(
            self,
            text="Traffic Route Finder",
            font=("Arial", 18, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=15)

        # --- Origin ---
        ttk.Label(self, text="Origin SCATS:").grid(
            row=1,
            column=0,
            sticky="w",
            pady=5
        )

        self.origin_entry = ttk.Entry(self, width=30)
        self.origin_entry.grid(row=1, column=1, pady=5)

        # --- Destination ---
        ttk.Label(self, text="Destination SCATS:").grid(
            row=2,
            column=0,
            sticky="w",
            pady=5
        )

        self.destination_entry = ttk.Entry(self, width=30)
        self.destination_entry.grid(row=2, column=1, pady=5)

        # --- Time ---
        ttk.Label(self, text="Departure Time (HH:MM):").grid(
            row=3,
            column=0,
            sticky="w",
            pady=5
        )

        self.time_entry = ttk.Entry(self, width=30)
        self.time_entry.insert(0, "08:00")
        self.time_entry.grid(row=3, column=1, pady=5)

        # --- Model Selection ---
        ttk.Label(self, text="Model:").grid(
            row=4,
            column=0,
            sticky="w",
            pady=5
        )

        self.model_combo = ttk.Combobox(
            self,
            values=["LSTM", "GRU", "XGBoost"],
            state="readonly",
            width=27
        )

        self.model_combo.current(2)
        self.model_combo.grid(row=4, column=1, pady=5)

        # --- Find Route Button ---
        self.find_button = ttk.Button(
            self,
            text="Find Route",
            command=self.handle_find_route
        )

        self.find_button.grid(
            row=5,
            column=0,
            columnspan=2,
            pady=20
        )

    def validate_inputs(self, origin, destination, time):
        """
        Validate user inputs.
        """

        if not origin.strip():
            messagebox.showerror("Input Error", "Origin SCATS is required.")
            return False

        if not destination.strip():
            messagebox.showerror("Input Error", "Destination SCATS is required.")
            return False

        if ":" not in time:
            messagebox.showerror(
                "Input Error",
                "Time must be in HH:MM format."
            )
            return False

        return True

    def handle_find_route(self):
        """
        Handle route search button.
        """

        origin = self.origin_entry.get()
        destination = self.destination_entry.get()
        time = self.time_entry.get()
        model = self.model_combo.get()

        # Validate input
        if not self.validate_inputs(origin, destination, time):
            return

        data = {
            "origin": origin,
            "destination": destination,
            "time": time,
            "model": model
        }

        self.on_find_route(data)