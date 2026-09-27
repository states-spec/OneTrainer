import locale
import os
import signal
import sys
from abc import ABCMeta

from modules.util.config.UISettingsConfig import UISettingsConfig
from modules.util.enum.UITheme import UITheme

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QToolTip, QWidget


class QtABCMeta(type(QWidget), ABCMeta):
    # Combined metaclass that resolves the conflict between Qt's Shiboken metaclass and ABCMeta.
    pass


_STYLESHEET = """
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
        padding: 2px 2px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    QProgressBar {
        background-color: %(progress_background)s;
    }
    QToolButton {
        padding-top: 0px;
        padding-bottom: 0px;
        padding-right: 40px;
    }
    QToolButton::menu-indicator {
        subcontrol-origin: padding;
        subcontrol-position: right center;
        width: 12px;
        height: 12px;
        right: 10px;
    }
"""

# the light look OneTrainer always had, captured in create_application; the font size the platform picked
_light_palette: QPalette | None = None
_default_font_point_size: float | None = None
_current_theme: UITheme | None = None


def _dark_palette() -> QPalette:
    window = QColor(45, 45, 45)
    button = QColor(53, 53, 53)
    base = QColor(30, 30, 30)
    text = QColor(230, 230, 230)
    disabled_text = QColor(128, 128, 128)
    highlight = QColor(42, 130, 218)

    palette = QPalette()
    for role, color in (
            (QPalette.ColorRole.Window, window),
            (QPalette.ColorRole.WindowText, text),
            (QPalette.ColorRole.Base, base),
            (QPalette.ColorRole.AlternateBase, button),
            (QPalette.ColorRole.ToolTipBase, button),
            (QPalette.ColorRole.ToolTipText, text),
            (QPalette.ColorRole.PlaceholderText, disabled_text),
            (QPalette.ColorRole.Text, text),
            (QPalette.ColorRole.Button, button),
            (QPalette.ColorRole.ButtonText, text),
            (QPalette.ColorRole.BrightText, QColor(255, 80, 80)),
            (QPalette.ColorRole.Link, QColor(86, 160, 240)),
            (QPalette.ColorRole.Highlight, highlight),
            (QPalette.ColorRole.HighlightedText, QColor("white")),
            (QPalette.ColorRole.Light, QColor(75, 75, 75)),
            (QPalette.ColorRole.Midlight, QColor(62, 62, 62)),
            (QPalette.ColorRole.Mid, QColor(40, 40, 40)),
            (QPalette.ColorRole.Dark, QColor(25, 25, 25)),
            (QPalette.ColorRole.Shadow, QColor(10, 10, 10)),
    ):
        palette.setColor(role, color)

    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(52, 52, 52))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(48, 48, 48))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, disabled_text)
    return palette


def _is_dark(app: QApplication, theme: UITheme) -> bool:
    if theme == UITheme.SYSTEM:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    return theme == UITheme.DARK


def apply_theme(app: QApplication, theme: UITheme):
    global _current_theme
    _current_theme = theme

    # also tells the platform, e.g. for a dark title bar on Windows
    if theme == UITheme.SYSTEM:
        app.styleHints().unsetColorScheme()
    else:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark if theme == UITheme.DARK else Qt.ColorScheme.Light)

    dark = _is_dark(app, theme)
    palette = _dark_palette() if dark else QPalette(_light_palette)
    app.setPalette(palette)
    QToolTip.setPalette(palette)
    app.setStyleSheet(_STYLESHEET % {"progress_background": "#505050" if dark else "#c8c8c8"})


def apply_font_size(app: QApplication, point_size: int):
    font = app.font()
    font.setPointSizeF(point_size if point_size > 0 else _default_font_point_size)
    if font != app.font():
        app.setFont(font)
        # widgets already polished by the application stylesheet keep their old font until it is set again
        app.setStyleSheet(app.styleSheet())


def apply_ui_settings(settings: UISettingsConfig):
    # the UI scale is applied by create_application, Qt can't change it while running
    app = QApplication.instance()
    if settings.theme != _current_theme:
        apply_theme(app, settings.theme)
    apply_font_size(app, settings.font_size)


def _on_color_scheme_changed(_scheme):
    if _current_theme == UITheme.SYSTEM:
        apply_theme(QApplication.instance(), UITheme.SYSTEM)


def create_application() -> QApplication:
    global _light_palette, _default_font_point_size

    # Restore the OS default SIGINT handler so Ctrl+C terminates the process
    # directly at the C level. Qt's event loop blocks inside C++, so Python's
    # own SIGINT handler would never get a chance to run while app.exec() is
    # active and Ctrl+C would be ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    settings = UISettingsConfig.load()
    # Qt reads the scale factor once, when the application is created. A value set in the environment wins.
    if settings.ui_scale != 1.0 and "QT_SCALE_FACTOR" not in os.environ:
        os.environ["QT_SCALE_FACTOR"] = str(settings.ui_scale)

    app = QApplication(sys.argv)
    # QApplication initializes the C locale from the environment (setlocale(LC_ALL, "")), which sets LC_NUMERIC
    # to a locale whose decimal separator may be a comma. C libraries then misparse '.' floats: protobuf's upb
    # backend rejects sentencepiece's schema ("Invalid default '0.9995'"), breaking every sentencepiece tokenizer
    # (T5/Chroma/Flux), and the Kineto profiler writes comma decimals into its JSON traces. Restore the C numeric
    # locale, which Python itself uses by default. Qt's own display uses QLocale, which is independent of this and
    # keeps formatting numbers per the system locale.
    locale.setlocale(locale.LC_NUMERIC, "C")
    # Force Fusion everywhere: native styles (e.g. windowsvista) draw standard
    # controls via OS theme APIs, which breaks once an application stylesheet
    # is set, producing a flatter look than Fusion's own stylesheet-aware painting.
    app.setStyle(QStyleFactory.create("Fusion"))
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    _light_palette = QPalette(app.palette())
    _light_palette.setColor(QPalette.ColorRole.Base, QColor("white"))
    _light_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    _default_font_point_size = app.font().pointSizeF()

    apply_theme(app, settings.theme)
    apply_font_size(app, settings.font_size)
    app.styleHints().colorSchemeChanged.connect(_on_color_scheme_changed)

    return app
