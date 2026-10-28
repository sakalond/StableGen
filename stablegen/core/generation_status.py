"""Server status presentation without Blender or workflow dependencies."""

from contextlib import contextmanager
import os
import time


WAITING_STAGES = (
    'Uploading Images', 'Submitting Workflow', 'Queued on Server',
    'Preparing Inputs', 'Loading Models', 'Preparing Conditioning',
    'Starting Sampler', 'Decoding Image', 'Saving Result', 'Receiving Result',
    'Processing on Server',
)


def server_status_active(operator):
    return bool(getattr(operator, '_server_stage', None)) and (
        operator._server_stage == getattr(operator, '_stage', None)
    )


def set_server_stage(operator, stage, detail='', progress=None):
    """Track elapsed time per node; percentages belong to that node only."""
    if (not server_status_active(operator)
            or operator._server_stage != stage
            or getattr(operator, '_server_stage_detail', '') != detail):
        operator._server_stage_started = time.monotonic()
        operator._server_step_text = ''
    operator._stage = stage
    operator._server_stage = stage
    operator._server_stage_detail = detail
    operator._server_node_progress = progress
    if hasattr(operator, '_detail_stage'):
        operator._detail_stage = stage
        if hasattr(operator, '_detail_progress'):
            operator._detail_progress = progress or 0.0


@contextmanager
def reporting_upload(operator, image_path):
    """Show upload status only around a real transfer, restoring caller state."""
    if operator is None:
        yield
        return
    missing = object()
    names = ('_stage', '_server_stage', '_server_stage_started',
             '_server_stage_detail', '_server_node_progress',
             '_server_step_text',
             '_detail_stage', '_detail_progress')
    previous = {name: getattr(operator, name, missing) for name in names}
    set_server_stage(operator, 'Uploading Images', os.path.basename(image_path))
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                if hasattr(operator, name):
                    delattr(operator, name)
            else:
                setattr(operator, name, value)


def node_stage(node_id, node_info, node_map):
    """Classify execution, excluding sampler selectors and model patchers."""
    kind = node_info.get('class_type', '')
    if (str(node_id) == str(node_map.get('sampler', ''))
            or kind in {'KSampler', 'KSamplerAdvanced', 'SamplerCustom',
                        'SamplerCustomAdvanced'}):
        return 'Starting Sampler'
    if kind == 'LoadImage':
        return 'Preparing Inputs'
    if 'Loader' in kind or kind.startswith('Load'):
        return 'Loading Models'
    if 'Decode' in kind:
        return 'Decoding Image'
    if 'Encode' in kind or 'Conditioning' in kind:
        return 'Preparing Conditioning'
    if kind.startswith('Save'):
        return 'Saving Result'
    return 'Processing on Server'


def stage_progress(operator):
    if server_status_active(operator):
        return getattr(operator, '_server_node_progress', None) or 0.0
    return getattr(operator, '_progress', 0.0)


def stage_text(operator, default='Generating'):
    stage = getattr(operator, '_stage', '') or default
    if server_status_active(operator):
        elapsed = max(0, int(time.monotonic() - operator._server_stage_started))
        duration = f'{elapsed // 60}m {elapsed % 60:02d}s' if elapsed >= 60 else f'{elapsed}s'
        label = 'Generating' if stage == 'Generating Image' else stage
        steps = getattr(operator, '_server_step_text', '').removeprefix('Step ')
        text = ' · '.join(part for part in (label, steps, duration) if part)
        progress = getattr(operator, '_server_node_progress', None)
        return f'{text} ({progress:.0f}%)' if progress is not None else text
    return f"{stage} ({getattr(operator, '_progress', 0.0):.0f}%)"


def draw_stage_progress(layout, operator, default='Generating'):
    """Used by the native panel and its captured Studio presentation."""
    bar = layout.progress(text=stage_text(operator, default),
                          factor=max(0.0, min(stage_progress(operator) / 100.0, 1.0)))
    # Native UILayout.progress returns None. The Studio adapter returns its
    # captured control so hover help stays metadata, outside Blender's API.
    if bar is not None and server_status_active(operator):
        context = (getattr(operator, '_server_context_stage', '')
                   if getattr(operator, '_pbr_active', False) else '')
        detail = getattr(operator, '_server_stage_detail', '')
        bar.tooltip = '\n'.join(part for part in (context, detail) if part)
