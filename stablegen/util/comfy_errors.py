"""Readable reasons for a workflow ComfyUI refuses to queue.

ComfyUI answers a /prompt it cannot run with HTTP 400 and a JSON body: {"error": {...}, "node_errors": {node id:
{"errors": [...], "class_type": ...}}}. Each error has a type, message, details and extra_info. StableGen used to
report only "HTTP Error 400: Bad Request", which hid the usual cause, a model or custom node that is not installed.
"""
import json

# Placeholders the model lists hold while ComfyUI has not provided real models.
PLACEHOLDERS = {'NO_SERVER', 'NONE_AVAILABLE', 'NONE_FOUND', 'NO_PREFS', 'REFRESH', 'NO_ASSIGNED'}
# Loader inputs whose choices are model files.
MODEL_INPUTS = {'ckpt_name', 'unet_name', 'clip_name', 'clip_name1', 'clip_name2', 'clip_name3', 'vae_name',
                'lora_name', 'control_net_name', 'model_name', 'style_model_name', 'clip_vision_name',
                'upscale_model_name', 'ipadapter_file'}
MODEL_EXTENSIONS = ('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.sft')
SHOWN = 3
DETAIL_LIMIT = 200


def _clip(text, limit=DETAIL_LIMIT):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + '…'


def _reason(node_id, class_type, error):
    """One readable line, and whether it is a model missing from ComfyUI."""
    kind = error.get('type', '')
    extra = error.get('extra_info') or {}
    if kind == 'value_not_in_list':
        value = extra.get('received_value')
        name = extra.get('input_name', '')
        if isinstance(value, str) and value in PLACEHOLDERS:
            return f"no model is selected for {class_type} {name} (refresh the model lists)", False
        if isinstance(value, str) and (name in MODEL_INPUTS or value.lower().endswith(MODEL_EXTENSIONS)):
            return f"the model '{value}' is not installed ({class_type} {name})", True
    if kind == 'missing_node_type':
        missing = extra.get('class_type') or class_type
        return (f"the node '{missing}' is not available: install its custom node or update ComfyUI, "
                "then restart ComfyUI"), False
    where = f"{class_type} (node {node_id})" if class_type else ''
    text = ': '.join(part for part in (error.get('message'), error.get('details')) if part)
    return _clip(f"{where}: {text}" if where else text), False


def describe_rejection(body):
    """A readable reason from ComfyUI's /prompt rejection body (bytes or str), or None when it has none."""
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    reasons = []
    node_errors = data.get('node_errors')
    if isinstance(node_errors, dict):
        for node_id, node in node_errors.items():
            if isinstance(node, dict):
                for error in node.get('errors') or []:
                    if isinstance(error, dict):
                        reasons.append(_reason(node_id, node.get('class_type', ''), error))
    error = data.get('error')
    if not reasons and isinstance(error, dict):
        extra = error.get('extra_info') or {}
        reasons.append(_reason(extra.get('node_id', ''), extra.get('class_type', ''), error))
    elif not reasons and isinstance(error, str) and error:
        reasons.append((_clip(error), False))
    if not reasons:
        return None
    lines = list(dict.fromkeys(line for line, _ in reasons))
    text = '; '.join(lines[:SHOWN]) + (f'; and {len(lines) - SHOWN} more' if len(lines) > SHOWN else '')
    if any(missing for _, missing in reasons):
        text += ". Install the StableGen installer package for this architecture, then refresh the model lists"
    return "ComfyUI rejected the workflow: " + text
