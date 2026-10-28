"""Cooperative main-thread operations, with synchronous script compatibility."""

from functools import wraps
from types import GeneratorType
from time import perf_counter
import bpy


class RenderRequest:
    """Suspend a render transaction until Blender's native render job ends."""

    def __init__(self, *, write_still=True):
        # Compositor File Output nodes write independently of the main colour
        # result. Guidance exports can omit that otherwise unused extra image.
        self.write_still = write_still


def operation(function):
    @wraps(function)
    def run(*args, **kwargs):
        steps = function(*args, **kwargs)
        return run_sync(steps)

    @wraps(function)
    def dispatch(*args, **kwargs):
        if kwargs.pop('_steps', False):
            return function(*args, **kwargs)
        return run(*args, **kwargs)
    return dispatch


def run_sync(steps):
    """Existing background, debug and script callers retain blocking semantics."""
    if not isinstance(steps, GeneratorType):
        return steps
    try:
        while True:
            request = next(steps)
            if isinstance(request, RenderRequest):
                result = bpy.ops.render.render(write_still=request.write_still)
                if 'CANCELLED' in result:
                    raise RuntimeError('StableGen render cancelled')
    except StopIteration as complete:
        return complete.value
    finally:
        steps.close()


class MainThreadOperation:
    """Advance transactions only on the main thread and after render teardown.

    Handlers only mark completion. The timer waits until the RENDER job has
    actually released its scene before resuming or unwinding the transaction.
    """

    def __init__(self, steps, done, cancelled=lambda: False):
        self.steps = steps
        self.done = done
        self.cancelled = cancelled
        self.rendering = False
        self.outcome = None
        self.scene = None
        self.handlers = []

    def _detach(self):
        for collection, callback in self.handlers:
            if callback in collection:
                collection.remove(callback)
        self.handlers.clear()

    def _start_render(self, request):
        self.scene = bpy.context.scene
        self.outcome = None
        def complete(scene, *args):
            if scene == self.scene:
                self.outcome = 'complete'
        def cancel(scene, *args):
            if scene == self.scene:
                self.outcome = 'cancel'
        for collection, callback in ((bpy.app.handlers.render_complete, complete),
                                     (bpy.app.handlers.render_cancel, cancel)):
            collection.append(callback)
            self.handlers.append((collection, callback))
        try:
            # Guidance exports should not open a Render Result window or steal
            # the panel's context. Restore the preference immediately after
            # invoking; it is only consulted when the job is started.
            view = bpy.context.preferences.view
            display = view.render_display_type
            try:
                view.render_display_type = 'NONE'
                result = bpy.ops.render.render('INVOKE_DEFAULT', write_still=request.write_still)
            finally:
                view.render_display_type = display
            if 'CANCELLED' in result:
                raise RuntimeError('StableGen render could not start')
            if 'FINISHED' in result and self.outcome is None:
                self.outcome = 'complete'
            self.rendering = True
        except Exception:
            self._detach()
            raise

    def tick(self):
        try:
            # Do not enter or unwind a scene transaction while another render
            # owns Blender's scene. Completion handlers alone are insufficient.
            if not self.rendering and bpy.app.is_job_running('RENDER'):
                return 0.05
            if self.rendering:
                if bpy.app.is_job_running('RENDER'):
                    # Completion precedes native job teardown. Poll more
                    # closely only during that short tail, without touching
                    # the scene before Blender has actually released it.
                    return 0.01 if self.outcome is not None else 0.05
                if self.outcome is None:
                    raise RuntimeError('StableGen render ended without completion')
                self.rendering = False
                self._detach()
                if self.outcome == 'cancel':
                    raise RuntimeError('StableGen render cancelled')
            # Tiny graph/layout steps should share a bounded turn instead of
            # each paying a timer delay. A render always ends the batch; never
            # resume its transaction until the native job has released it.
            deadline = perf_counter() + 0.005
            while True:
                if self.cancelled():
                    raise RuntimeError('Generation cancelled')
                request = next(self.steps)
                if isinstance(request, RenderRequest):
                    if bpy.app.is_job_running('RENDER'):
                        raise RuntimeError('Another render job is already running')
                    self._start_render(request)
                    break
                if perf_counter() >= deadline:
                    break
            return 0.01
        except StopIteration:
            self._detach()
            self.done(None)
        except Exception as error:
            self._detach()
            try:
                self.steps.close()
            except Exception as cleanup_error:
                error = RuntimeError(f'{error}; cleanup failed: {cleanup_error}')
            self.done(error)
        return None
