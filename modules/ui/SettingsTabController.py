from collections.abc import Callable

from modules.util.config.UISettingsConfig import UI_SETTINGS_PATH, UISettingsConfig
from modules.util.enum.UITheme import UITheme


class SettingsTabController:
    SETTING_NAMES = ("theme", "ui_scale", "font_size", "remember_window_size", "start_maximized", "system_file_dialogs")

    def __init__(
            self,
            settings: UISettingsConfig,
            apply_settings: Callable[[UISettingsConfig], None],
            needs_restart: tuple[str, ...],
            supports_font_size: bool,
            supports_system_file_dialogs: bool = False,
    ):
        self.settings = settings
        self.apply_settings = apply_settings
        self.needs_restart = needs_restart  # names of the settings the toolkit can only apply at startup
        self.supports_font_size = supports_font_size
        self.supports_system_file_dialogs = supports_system_file_dialogs  # Qt on Linux: Qt's own or the desktop's dialog

    def get_themes(self) -> list[tuple[str, UITheme]]:
        return [
            ("Dark", UITheme.DARK),
            ("Light", UITheme.LIGHT),
            ("System", UITheme.SYSTEM),
        ]

    def get_ui_scales(self) -> list[tuple[str, float]]:
        scales = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
        if self.settings.ui_scale not in scales:  # keep a hand-edited value selectable instead of resetting it
            scales = sorted(scales + [self.settings.ui_scale])
        return [(f"{round(scale * 100)}%", scale) for scale in scales]

    def get_font_sizes(self) -> list[tuple[str, int]]:
        sizes = [8, 9, 10, 11, 12, 13, 14, 16, 18]
        if self.settings.font_size not in sizes and self.settings.font_size > 0:
            sizes = sorted(sizes + [self.settings.font_size])
        return [("Default", 0)] + [(f"{size} pt", size) for size in sizes]

    def get_settings_path(self) -> str:
        return UI_SETTINGS_PATH

    def watch(self, ui_state):
        # apply and save on every change; the traces only fire on changes, not when the widgets are built
        for name in self.SETTING_NAMES:
            ui_state.add_var_trace(name, self.__on_change)

    def reset(self, ui_state):
        # writes the defaults through the widgets' vars into self.settings; each change applies and saves
        defaults = UISettingsConfig.default_values()
        defaults.window_width = self.settings.window_width
        defaults.window_height = self.settings.window_height
        ui_state.update(defaults)

    def __on_change(self):
        self.apply_settings(self.settings)
        self.settings.save()
