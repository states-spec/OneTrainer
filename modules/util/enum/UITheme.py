from enum import Enum


class UITheme(Enum):
    DARK = 'DARK'
    LIGHT = 'LIGHT'
    SYSTEM = 'SYSTEM'

    def __str__(self):
        return self.value
