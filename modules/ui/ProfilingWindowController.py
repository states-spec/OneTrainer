import faulthandler
import os

from modules.util import path_util


class ProfilingWindowController:
    def __init__(self):
        self.view = None

    def create_window(self, parent, view_cls):
        self.view = view_cls(parent, self)
        return self.view

    def dump_stack(self):
        path = path_util.diagnostics_path('stacks.txt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            faulthandler.dump_traceback(f)
        self.view.set_message(f'Stack dumped to {path}')

    def start_profiler(self):
        from scalene import scalene_profiler
        scalene_profiler.start()
        self.view.set_profiling_active(True)

    def end_profiler(self):
        from scalene import scalene_profiler
        scalene_profiler.stop()
        self.view.set_profiling_active(False)
