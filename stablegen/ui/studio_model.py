"""Live RNA bindings for the custom Studio sidebar; no generation logic."""
from dataclasses import dataclass
import json
from pathlib import Path

import bpy

from .presentation import Layout, Control, capture_panel
from .presets import get_preset_items
from .model_units import update_model_list, get_lora_models, get_controlnet_models
from ..core.architectures import (architecture_items, model_architecture_items,
                                    trellis_texture_items, trellis_initial_image_items)


@dataclass
class LiveControl:
    control: Control
    enabled: bool = True
    alert: bool = False


def controls(node, enabled=True, alert=False):
    """Keep inherited enable/alert states when moving native controls."""
    enabled = enabled and node.attributes.get('enabled', True) and node.attributes.get('active', True)
    alert = alert or node.attributes.get('alert', False)
    for child in node.children:
        if isinstance(child, Layout):
            yield from controls(child, enabled, alert)
        else:
            yield LiveControl(child, enabled, alert)


def operator(name):
    namespace, identifier = name.split('.', 1)
    return getattr(getattr(bpy.ops, namespace), identifier)


@dataclass(frozen=True)
class Binding:
    """Resolve RNA again after undo instead of holding invalid datablocks."""
    root: str
    name: str
    path: str
    property: str

    @classmethod
    def from_data(cls, data, property):
        owner = data.id_data
        if isinstance(owner, bpy.types.Scene):
            root, name = 'scene', owner.name
        elif isinstance(owner, bpy.types.WindowManager):
            root, name = 'wm', ''
        elif isinstance(owner, bpy.types.Object):
            root, name = 'object', owner.name
        elif isinstance(owner, bpy.types.Material):
            root, name = 'material', owner.name
        else:
            raise TypeError('This RNA control needs its native editor')
        path = '' if data.as_pointer() == owner.as_pointer() else data.path_from_id()
        return cls(root, name, path, property)

    def data(self, context):
        if self.root == 'scene':
            owner = context.scene if context.scene.name == self.name else bpy.data.scenes.get(self.name)
        elif self.root == 'wm':
            owner = context.window_manager
        elif self.root == 'object':
            owner = bpy.data.objects.get(self.name)
        else:
            owner = bpy.data.materials.get(self.name)
        if owner is None:
            raise ReferenceError('The edited datablock no longer exists')
        return owner.path_resolve(self.path) if self.path else owner

    def value(self, context):
        return getattr(self.data(context), self.property)

    def rna(self, context):
        return self.data(context).bl_rna.properties[self.property]

    def set(self, context, value):
        data = self.data(context)
        prop = data.bl_rna.properties[self.property]
        if prop.type in {'INT', 'FLOAT'} and not prop.is_array:
            value = max(prop.hard_min, min(prop.hard_max, value))
            value = int(round(value)) if prop.type == 'INT' else float(value)
        setattr(data, self.property, value)


def enum_items(binding, context):
    data, key = binding.data(context), binding.property
    callbacks = {'architecture_mode': architecture_items,
                 'model_architecture': model_architecture_items,
                 'trellis2_texture_mode': trellis_texture_items,
                 'trellis2_initial_image_arch': trellis_initial_image_items,
                 'stablegen_preset': get_preset_items,
                 'sequential_ipadapter_mode': None}
    if key == 'sequential_ipadapter_mode':
        from ..core.properties import _get_ipadapter_mode_items
        return _get_ipadapter_mode_items(data, context)
    if key in callbacks:
        return callbacks[key](data, context)
    if key == 'model_name':
        if isinstance(data, bpy.types.Scene):
            return update_model_list(data, context)
        if data.bl_rna.identifier == 'LoRAUnit':
            return get_lora_models(data, context)
        if data.bl_rna.identifier == 'ControlNetUnit':
            return get_controlnet_models(context, data.unit_type)
    return [(item.identifier, item.name, item.description)
            for item in binding.rna(context).enum_items]


def enum_label(binding, context):
    value = binding.value(context)
    return next((item[1] for item in enum_items(binding, context)
                 if item and item[0] == value), str(value))


def document(context):
    return capture_panel(context)


def first_property(document, name):
    for section in document.sections:
        for item in controls(section.content):
            c = item.control
            if c.method == 'prop' and c.args[1] == name:
                return item
    return None


def action(document, *names):
    for node in document.actions:
        for item in controls(node):
            if item.control.method == 'operator' and item.control.args[0] in names:
                return item
    return None


def projected_images(context):
    """Read existing projection textures using their saved camera identities."""
    from ..cameras.images import projected_images as resolve
    return resolve(context)
