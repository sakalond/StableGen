"""StableGen addon - registration entry point.

This module is intentionally thin: it assembles the *classes* list from
submodules, registers them with Blender, and delegates property
registration to :pymod:`core.properties`.
"""

import bpy  # pylint: disable=import-error

# -- UI / Operator classes ------------------------------------------------
from .ui.presets import (
    ApplyPreset, SavePreset, DeletePreset,
    ResetQwenPrompt, SwitchToMeshGeneration,
)
from .ui.panel import StableGenPanel
from .ui.queue import (
    SG_UL_SceneQueueList, SceneQueueAdd, SceneQueueRemove, SceneQueueClear,
    SceneQueueMoveUp, SceneQueueMoveDown, SceneQueueOpenResult,
    SceneQueueInvalidate, SceneQueueProcess,
)
from .texturing.rendering import (
    BakeTextures, SwitchMaterial,
)
from .texturing.orbit_export import ExportOrbitGIF
from .texturing.game_export import ExportForGameEngine
from .cameras import (
    AddCameras, ApplyAutoAspect, CloneCamera, MirrorCamera, ToggleCameraLabels,
    CollectCameraPrompts,
    CameraPromptItem, CameraOrderItem, SG_UL_CameraOrderList, SyncCameraOrder,
    MoveCameraOrder, ApplyCameraOrderPreset,
)
from .debug_tools import debug_classes as _debug_classes
from .utils import AddHDRI, ApplyModifiers, CurvesToMesh
from .texturing.generator import (
    ComfyUIGenerate, Reproject, Regenerate, MirrorReproject,
)
from .mesh_gen.trellis2 import Trellis2Generate
from .mesh_gen.batch import (
    TRELLIS2_OT_BatchSelectFolder, TRELLIS2_OT_BatchGenerate,
    TRELLIS2_OT_BatchCancel, TRELLIS2_OT_BatchClear,
    TRELLIS2_OT_BatchBakeSettings,
)
from .dae_import import DAE_IMPORT_CLASSES

# -- Core / UI sub-packages ----------------------------------------------
from .core.preferences import (
    CheckServerStatus, StableGenAddonPreferences,
    ControlNetModelMappingItem, STABLEGEN_UL_ControlNetMappingList,
    RefreshControlNetMappings,
)
from .core.load_handlers import load_handler
from .core.properties import register_properties, unregister_properties
from .ui.model_units import (
    ControlNetUnit, LoRAUnit,
    RefreshCheckpointList, RefreshLoRAList,
    AddControlNetUnit, RemoveControlNetUnit, AddLoRAUnit, RemoveLoRAUnit,
    update_model_list,
)
from .ui.queue import SceneQueueItem, _sg_queue_load_handler, _sg_queue_load

bl_info = {
    "name": "StableGen",
    "category": "Object",
    "author": "Ondrej Sakala",
    "version": (0, 3, 0),
    'blender': (4, 2, 0)
}

# ---------------------------------------------------------------------------
# Class registration list
# ---------------------------------------------------------------------------
# PropertyGroups first (they are referenced by CollectionProperty in
# register_properties), then operators and preferences, then the panel last.
classes = [
    # PropertyGroups
    CameraPromptItem,
    CameraOrderItem,
    ControlNetModelMappingItem,
    ControlNetUnit,
    LoRAUnit,
    SceneQueueItem,
    # Preferences
    STABLEGEN_UL_ControlNetMappingList,
    RefreshControlNetMappings,
    StableGenAddonPreferences,
    # Operators - server / model management
    CheckServerStatus,
    RefreshCheckpointList,
    RefreshLoRAList,
    AddControlNetUnit,
    RemoveControlNetUnit,
    AddLoRAUnit,
    RemoveLoRAUnit,
    # Operators - generation
    ComfyUIGenerate,
    Reproject,
    Regenerate,
    MirrorReproject,
    Trellis2Generate,
    TRELLIS2_OT_BatchSelectFolder,
    TRELLIS2_OT_BatchGenerate,
    TRELLIS2_OT_BatchCancel,
    TRELLIS2_OT_BatchClear,
    TRELLIS2_OT_BatchBakeSettings,
    # Operators - render tools
    BakeTextures,
    AddCameras,
    ApplyAutoAspect,
    CloneCamera,
    MirrorCamera,
    ToggleCameraLabels,
    SwitchMaterial,
    ExportOrbitGIF,
    ExportForGameEngine,
    CollectCameraPrompts,
    SG_UL_CameraOrderList,
    SyncCameraOrder,
    MoveCameraOrder,
    ApplyCameraOrderPreset,
    # Operators - utilities
    AddHDRI,
    ApplyModifiers,
    CurvesToMesh,
    # Operators - DAE import
    *DAE_IMPORT_CLASSES,
    # Operators - presets / panel
    ApplyPreset,
    SavePreset,
    DeletePreset,
    ResetQwenPrompt,
    SwitchToMeshGeneration,
    # Queue operators
    SG_UL_SceneQueueList,
    SceneQueueAdd,
    SceneQueueRemove,
    SceneQueueClear,
    SceneQueueMoveUp,
    SceneQueueMoveDown,
    SceneQueueOpenResult,
    SceneQueueInvalidate,
    SceneQueueProcess,
    # Main panel (must be last – it uses all the above)
    StableGenPanel,
]
# Append debug classes (defined in debug_tools.py)
classes.extend(_debug_classes)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    register_properties(
        update_model_list=update_model_list,
        ControlNetUnit=ControlNetUnit,
        LoRAUnit=LoRAUnit,
        SceneQueueItem=SceneQueueItem,
        load_handler=load_handler,
        _sg_queue_load_handler=_sg_queue_load_handler,
        _sg_queue_load=_sg_queue_load,
    )


def unregister():
    from .mesh_gen.batch import unregister_batch
    unregister_batch()

    unregister_properties(
        load_handler=load_handler,
        _sg_queue_load_handler=_sg_queue_load_handler,
    )

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()