"""DAE (COLLADA) import and preprocessing for StableGen.

Provides a custom XML-based COLLADA importer that works across all Blender
versions (including 5.x where the built-in Collada importer was removed),
plus geometry cleanup utilities tailored for SketchUp exports.
"""

from .operators import DAE_IMPORT_CLASSES as _DAE_IMPORT_CLASSES
from .batch_texture import BATCH_TEXTURE_CLASSES

DAE_IMPORT_CLASSES = [*_DAE_IMPORT_CLASSES, *BATCH_TEXTURE_CLASSES]

__all__ = ["DAE_IMPORT_CLASSES"]
