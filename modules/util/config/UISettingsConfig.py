import json
import os
from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.enum.UITheme import UITheme
from modules.util.path_util import write_json_atomic

# per-user UI preferences, kept apart from the training config so presets and saved configs don't carry them
UI_SETTINGS_PATH = os.path.join("training_user_settings", "ui_settings.json")


class UISettingsConfig(BaseConfig):
    theme: UITheme
    ui_scale: float
    font_size: int
    remember_window_size: bool
    start_maximized: bool
    window_width: int
    window_height: int
    system_file_dialogs: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values() -> 'UISettingsConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("theme", UITheme.DARK, UITheme, False))
        data.append(("ui_scale", 1.0, float, False))
        data.append(("font_size", 0, int, False))  # point size, 0 = the toolkit default
        data.append(("remember_window_size", True, bool, False))
        data.append(("start_maximized", False, bool, False))
        data.append(("window_width", 0, int, False))  # last window size, 0 = the default size
        data.append(("window_height", 0, int, False))
        # Linux, Qt UI: the desktop's own file dialog (GTK, portal or KDE) instead of Qt's. Off by default, because it
        # can freeze or crash OneTrainer on some desktops (the pip Qt build loads the desktop's dialog code)
        data.append(("system_file_dialogs", False, bool, False))

        return UISettingsConfig(data)

    @staticmethod
    def load(path: str = UI_SETTINGS_PATH) -> 'UISettingsConfig':
        settings = UISettingsConfig.default_values()
        try:
            with open(path, "r") as f:
                settings.from_dict(json.load(f))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Could not read the UI settings from {path}, using the defaults: {e}")
        return settings

    def save(self, path: str = UI_SETTINGS_PATH):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_json_atomic(path, self.to_dict())
        except OSError as e:
            print(f"Could not save the UI settings to {path}: {e}")
