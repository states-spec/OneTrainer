from modules.ui.BaseSettingsTabView import BaseSettingsTabView
from modules.ui.SettingsTabController import SettingsTabController
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QWidget


class PySide6SettingsTabView(BaseSettingsTabView, QWidget):

    def __init__(self, master, controller: SettingsTabController, ui_state):
        QWidget.__init__(self, master)
        BaseSettingsTabView.__init__(self, pyside6_components)

        frame = QWidget(self)
        pyside6_components._layout(self).addWidget(frame, 0, 0)
        lo = pyside6_components._layout(frame)
        lo.setContentsMargins(pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD)
        self.build(frame, controller, ui_state)
        pyside6_components._pack_form(frame)
        controller.watch(ui_state)
