from modules.ui.BaseConvertModelUIView import BaseConvertModelUIView
from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkConvertModelUIView(BaseConvertModelUIView, ctk.CTkToplevel):
    def __init__(self, parent, controller: ConvertModelUIController, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseConvertModelUIView.__init__(self, ctk_components)

        self.controller = controller
        self.ui_state = CtkUIState(self, controller.convert_model_args)
        self._dynamic_widgets = []

        self.title("Convert models")
        self.geometry("550x350")
        self.resizable(True, True)

        self.frame = ctk.CTkFrame(self, width=600, height=300)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)

        self.build_content(self.frame, controller, self.ui_state, self._rebuild_dynamic_ui)
        self._rebuild_dynamic_ui()
        self.frame.pack(fill="both", expand=True)

        self.wait_visibility()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _rebuild_dynamic_ui(self, *args):
        for widget in self._dynamic_widgets:
            widget.destroy()

        before = set(self.frame.winfo_children())
        self.build_dynamic_content(self.frame, self.controller, self.ui_state)
        self._dynamic_widgets = [c for c in self.frame.winfo_children() if c not in before]

    def set_converting(self, active):
        self.button.configure(state="disabled" if active else "normal")
