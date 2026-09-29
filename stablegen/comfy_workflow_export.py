"""Create editable ComfyUI workflow files from StableGen Qwen 2.1 prompts."""

from __future__ import annotations

from copy import deepcopy
import uuid


_OUTPUTS = {
    'UNETLoader': [('MODEL', 'MODEL')],
    'CLIPLoader': [('CLIP', 'CLIP')],
    'VAELoader': [('VAE', 'VAE')],
    'QwenImage21Cache': [('MODEL', 'MODEL')],
    'TextEncodeQwenImage21': [
        ('positive', 'CONDITIONING'),
        ('negative', 'CONDITIONING'),
        ('latent', 'LATENT'),
    ],
    'KSampler': [('LATENT', 'LATENT')],
    'VAEDecode': [('IMAGE', 'IMAGE')],
    'SaveImage': [('images', 'IMAGE')],
    'SaveImageWebsocket': [],
    'LoadImage': [('IMAGE', 'IMAGE'), ('MASK', 'MASK')],
    'ImageScaleToTotalPixels': [('IMAGE', 'IMAGE')],
}

_WIDGETS = {
    'UNETLoader': ['unet_name', 'weight_dtype'],
    'CLIPLoader': ['clip_name', 'type', 'device'],
    'VAELoader': ['vae_name'],
    'QwenImage21Cache': ['device', 'dtype'],
    'TextEncodeQwenImage21': ['prompt', 'negative_prompt', 'resolution'],
    'KSampler': [
        'seed', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'
    ],
    'SaveImage': ['filename_prefix'],
    'LoadImage': ['image'],
    'ImageScaleToTotalPixels': [
        'upscale_method', 'megapixels', 'resolution_steps'
    ],
}

_LINK_INPUT_TYPES = {
    'QwenImage21Cache': {'model': 'MODEL'},
    'TextEncodeQwenImage21': {
        'clip': 'CLIP',
        'vae': 'VAE',
        **{f'images.image_{index}': 'IMAGE' for index in range(1, 17)},
    },
    'KSampler': {
        'model': 'MODEL',
        'positive': 'CONDITIONING',
        'negative': 'CONDITIONING',
        'latent_image': 'LATENT',
    },
    'VAEDecode': {'samples': 'LATENT', 'vae': 'VAE'},
    'SaveImage': {'images': 'IMAGE'},
    'ImageScaleToTotalPixels': {'image': 'IMAGE'},
}

_INPUT_ORDER = {
    'QwenImage21Cache': ['model', 'device', 'dtype'],
    'TextEncodeQwenImage21': [
        'clip', 'images.image_1', 'vae',
        'prompt', 'negative_prompt', 'resolution',
        *[f'images.image_{index}' for index in range(2, 17)],
    ],
    'KSampler': [
        'model', 'seed', 'steps', 'cfg', 'sampler_name', 'scheduler',
        'positive', 'negative', 'latent_image', 'denoise',
    ],
    'VAEDecode': ['samples', 'vae'],
    'SaveImage': ['images', 'filename_prefix'],
    'ImageScaleToTotalPixels': [
        'upscale_method', 'megapixels', 'resolution_steps', 'image'
    ],
}

_WIDGET_TYPES = {
    'seed': 'INT',
    'steps': 'INT',
    'cfg': 'FLOAT',
    'denoise': 'FLOAT',
    'resolution': 'INT',
    'megapixels': 'FLOAT',
    'resolution_steps': 'INT',
    'prompt': 'STRING',
    'negative_prompt': 'STRING',
    'filename_prefix': 'STRING',
}

_POSITIONS = {
    'UNETLoader': (0, 0),
    'CLIPLoader': (0, 180),
    'VAELoader': (0, 360),
    'LoadImage': (0, 580),
    'ImageScaleToTotalPixels': (330, 650),
    'QwenImage21Cache': (360, 0),
    'TextEncodeQwenImage21': (360, 220),
    'KSampler': (920, 0),
    'VAEDecode': (1260, 0),
    'SaveImage': (1540, 0),
}


def _is_link(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _integer_node_id(node_id):
    try:
        return int(node_id)
    except (TypeError, ValueError):
        return None


def _widget_value(class_type, name, api_inputs):
    if class_type == 'SaveImage' and name == 'filename_prefix':
        return api_inputs.get(name, 'StableGen_test')
    return api_inputs.get(name)


def _node_size(class_type):
    if class_type == 'TextEncodeQwenImage21':
        return [500, 610]
    if class_type == 'LoadImage':
        return [300, 300]
    if class_type == 'KSampler':
        return [310, 270]
    return [300, 120]


def qwen_image_21_ui_workflow(prompt):
    """Return a frontend workflow for a Qwen 2.1 API prompt, or ``None``.

    ComfyUI API prompts normally import through ``loadApiJson``. Frontend
    1.53.x currently throws while reconstructing Qwen 2.1's COMFY_AUTOGROW_V3
    image sockets. An explicit UI workflow bypasses that conversion path and
    remains editable and runnable in the browser.
    """
    if not any(
        node.get('class_type') == 'TextEncodeQwenImage21'
        for node in prompt.values()
        if isinstance(node, dict)
    ):
        return None

    api_graph = deepcopy(prompt)
    numeric_ids = [
        value for value in (_integer_node_id(key) for key in api_graph)
        if value is not None
    ]
    next_id = max(numeric_ids or [0]) + 1

    # The websocket-only output is useful to StableGen but inconvenient in an
    # interactive graph. Replace it with a normal SaveImage node.
    for node in api_graph.values():
        if node.get('class_type') == 'SaveImageWebsocket':
            node['class_type'] = 'SaveImage'
            node.setdefault('inputs', {})['filename_prefix'] = 'StableGen_test'
            node.setdefault('_meta', {})['title'] = 'Save Image'

    nodes = []
    node_by_id = {}
    type_offsets = {}
    for raw_id, api_node in sorted(
        api_graph.items(),
        key=lambda item: (_integer_node_id(item[0]) is None,
                          _integer_node_id(item[0]) or 0),
    ):
        node_id = _integer_node_id(raw_id)
        if node_id is None:
            node_id = next_id
            next_id += 1
        class_type = api_node.get('class_type', '')
        if class_type not in _OUTPUTS:
            continue
        api_inputs = api_node.get('inputs', {})
        widgets = _WIDGETS.get(class_type, [])
        input_order = _INPUT_ORDER.get(class_type, [])
        link_types = _LINK_INPUT_TYPES.get(class_type, {})
        ui_inputs = []
        for name in input_order:
            if name in link_types:
                ui_inputs.append({
                    'name': name,
                    'type': link_types[name],
                    'link': None,
                    **({'shape': 7} if name.startswith('images.') else {}),
                })
            elif name in widgets:
                ui_inputs.append({
                    'name': name,
                    'type': _WIDGET_TYPES.get(name, 'COMBO'),
                    'widget': {'name': name},
                    'link': None,
                })

        count = type_offsets.get(class_type, 0)
        type_offsets[class_type] = count + 1
        base_x, base_y = _POSITIONS.get(class_type, (0, 0))
        base_y += count * (360 if class_type == 'LoadImage' else 160)

        outputs = [
            {'name': name, 'type': output_type, 'links': []}
            for name, output_type in _OUTPUTS[class_type]
        ]
        widget_values = [
            _widget_value(class_type, name, api_inputs) for name in widgets
        ]
        if class_type == 'KSampler':
            widget_values.insert(1, 'fixed')
        if class_type == 'LoadImage':
            widget_values.append('image')

        title = api_node.get('_meta', {}).get('title')
        ui_node = {
            'id': node_id,
            'type': class_type,
            'pos': [base_x, base_y],
            'size': _node_size(class_type),
            'flags': {},
            'order': len(nodes),
            'mode': 0,
            'inputs': ui_inputs,
            'outputs': outputs,
            'properties': {
                'Node name for S&R': class_type,
                # Anything Everywhere releases written for the old group-node
                # API call GroupNodeHandler.getGroupData for every ordinary
                # node unless it opts out. Current ComfyUI removed that API,
                # so imported test workflows otherwise produce a continuous
                # frontend error even though the graph itself is valid.
                'rejects_ue_links': True,
            },
            'widgets_values': widget_values,
            'widgets_values_named': {
                name: _widget_value(class_type, name, api_inputs)
                for name in widgets
            },
        }
        if title and title != class_type:
            ui_node['title'] = title
        nodes.append(ui_node)
        node_by_id[str(raw_id)] = ui_node

    links = []
    next_link_id = 1
    for raw_id, api_node in api_graph.items():
        target = node_by_id.get(str(raw_id))
        if target is None:
            continue
        class_type = api_node.get('class_type', '')
        for input_name, value in api_node.get('inputs', {}).items():
            if not _is_link(value):
                continue
            source = node_by_id.get(str(value[0]))
            if source is None:
                continue
            target_slot = next((
                index for index, spec in enumerate(target['inputs'])
                if spec['name'] == input_name
            ), None)
            source_slot = int(value[1])
            if target_slot is None or source_slot >= len(source['outputs']):
                continue
            link_type = _LINK_INPUT_TYPES.get(class_type, {}).get(input_name)
            if link_type is None:
                continue
            link_id = next_link_id
            next_link_id += 1
            target['inputs'][target_slot]['link'] = link_id
            source['outputs'][source_slot]['links'].append(link_id)
            links.append([
                link_id, source['id'], source_slot,
                target['id'], target_slot, link_type,
            ])

    return {
        'id': str(uuid.uuid4()),
        'revision': 0,
        'last_node_id': max((node['id'] for node in nodes), default=0),
        'last_link_id': max((link[0] for link in links), default=0),
        'nodes': nodes,
        'links': links,
        'groups': [],
        'config': {},
        'extra': {
            'ds': {'scale': 0.75, 'offset': [120.0, 120.0]},
            'frontendVersion': '1.53.6',
        },
        'version': 0.4,
    }
