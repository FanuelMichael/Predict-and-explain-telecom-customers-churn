import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt

class CustomerSelectionForm:
    def __init__(self, processor, indices, max_index):
        self.processor = processor
        self.indices = indices
        self.selected_idx = None

        max_index = 100
        self.max_index = max_index

        # Create main window
        self.window = tk.Tk()
        self.window.title("Customer selection and profile")
        self.window.geometry("850x690")
        self.window.resizable(True, True)

        # Frame for ID input
        input_frame = ttk.LabelFrame(self.window, text="Customer sample Id", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(input_frame, text=f"Enter index (0 to 40):").pack(side="left", padx=5)

        self.id_var = tk.StringVar()
        self.id_entry = ttk.Entry(input_frame, textvariable=self.id_var, width=10)
        self.id_entry.pack(side="left", padx=5)

        self.ok_button = ttk.Button(input_frame, text="OK", command=self.on_ok)
        self.ok_button.pack(side="left", padx=5)

        # Frame for profile display
        profile_frame = ttk.LabelFrame(self.window, text="Customer original profile", padding=10)
        profile_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Table (Treeview)
        columns = ("Feature", "Value")
        self.tree = ttk.Treeview(profile_frame, columns=columns, show="headings", height=20)
        self.tree.heading("Feature", text="Feature Name")
        self.tree.heading("Value", text="Original Value")
        self.tree.column("Feature", width=150)
        self.tree.column("Value", width=150)
        self.tree.pack(side="left", fill="both", expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(profile_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Bottom buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill="x", padx=10, pady=10)

        self.predict_button = ttk.Button(button_frame, text="Predict this customer",
                                         command=self.on_predict, state="disabled")
        self.predict_button.pack(side="right", padx=5)

        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.on_cancel)
        self.cancel_button.pack(side="right", padx=5)

        # Center window
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def on_ok(self):
        try:
            idx = int(self.id_var.get())
            if idx < 0 or idx > self.max_index:
                raise ValueError(f"Index must be between 0 and {self.max_index}")

            raw = self.processor.get_customer_original_data(idx, self.indices)

            # Clear table
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Insert rows
            for key, value in raw.items():
                self.tree.insert("", "end", values=(key, value))
            self.predict_button.config(state="normal")
            self.selected_idx = idx

        except Exception as e:
            messagebox.showerror("Error", f"Invalid sample Id: {e}")

    def on_predict(self):
        raw = self.processor.get_customer_original_data(self.selected_idx, self.indices)
        self.window.destroy()
        self.selected_raw = raw

    def on_cancel(self):
        self.selected_idx = None
        self.window.destroy()

    def get_selected_index(self):
        self.window.mainloop()
        return self.selected_idx, getattr(self, "selected_raw", None)

    def customer_profile_figure(customer_dict):
        # Adjust figure size and high DPI
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.axis("off")

        keys = list(customer_dict.keys())
        values = [str(v) for v in customer_dict.values()]
        table_data = list(zip(keys, values))
        # ----------Create the base table----------------------------
        table = ax.table(cellText=table_data, colLabels=["Feature", "Value"], loc="center", cellLoc="left")
        # ----------Configure baseline sizing------------------------
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.6)  # Generous vertical padding for a modern look

        # Iterate through cells to apply elegant custom styling
        for (row, col), cell in table.get_celld().items():
            # Style the Header Row
            if row == 0:
                cell.set_text_props(weight='bold', color='white', fontsize=11)
                cell.set_facecolor('#1E293B')
                cell.set_edgecolor('#1E293B')
            else:
                # Apply Zebra Striping to rows
                if row % 2 == 0:
                    cell.set_facecolor('#F8FAFC')  # Soft cool gray
                else:
                    cell.set_facecolor('#FFFFFF')  # Pure white
                cell.set_edgecolor('#E2E8F0')  # Subtle, soft border color
                cell.set_text_props(color='#334155')
                # Make the feature names bold for better visual hierarchy
                if col == 0:
                    cell.set_text_props(weight='bold', color='#0F172A')
        plt.close(fig)
        return fig