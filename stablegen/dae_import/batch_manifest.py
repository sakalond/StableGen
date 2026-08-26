"""Pure-Python helpers for StableGen model batch manifests.

The Blender operator imports this module, but it intentionally has no ``bpy``
dependency so manifest validation can be tested outside Blender.
"""

from __future__ import annotations

import csv
import os
import re


SUPPORTED_MODEL_EXTENSIONS = {".dae", ".glb"}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
}

_REQUIRED_COLUMNS = {"model_path", "reference_path", "output_path"}


def _resolve_inside(root: str, relative_path: str, label: str) -> str:
    """Resolve a manifest path while preventing absolute/traversal paths."""
    value = (relative_path or "").strip().replace("/", os.sep)
    if not value:
        raise ValueError(f"{label} is empty")
    if os.path.isabs(value) or os.path.splitdrive(value)[0]:
        raise ValueError(f"{label} must be relative: {relative_path!r}")

    root_real = os.path.realpath(root)
    resolved = os.path.realpath(os.path.join(root_real, value))
    try:
        inside = os.path.commonpath((root_real, resolved)) == root_real
    except ValueError as exc:
        raise ValueError(f"{label} escapes the manifest directory") from exc
    if not inside:
        raise ValueError(f"{label} escapes the manifest directory")
    return resolved


def load_batch_manifest(manifest_path: str) -> tuple[str, list[dict[str, str]]]:
    """Load and validate a CSV manifest.

    Paths are relative to the directory containing the manifest. Input files
    must exist. Output paths are validated for containment and a ``.glb``
    extension, but they do not need to exist.
    """
    manifest_abs = os.path.realpath(manifest_path)
    if not os.path.isfile(manifest_abs):
        raise ValueError(f"Batch manifest does not exist: {manifest_path}")
    root = os.path.dirname(manifest_abs)

    with open(manifest_abs, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                "Batch manifest is missing column(s): " + ", ".join(missing)
            )
        source_rows = list(reader)

    if not source_rows:
        raise ValueError("Batch manifest contains no model rows")

    entries: list[dict[str, str]] = []
    seen_models: set[str] = set()
    seen_outputs: set[str] = set()
    for row_number, row in enumerate(source_rows, start=2):
        model_path = _resolve_inside(
            root, row.get("model_path", ""), f"row {row_number} model_path"
        )
        reference_path = _resolve_inside(
            root,
            row.get("reference_path", ""),
            f"row {row_number} reference_path",
        )
        output_path = _resolve_inside(
            root, row.get("output_path", ""), f"row {row_number} output_path"
        )

        model_ext = os.path.splitext(model_path)[1].lower()
        if model_ext not in SUPPORTED_MODEL_EXTENSIONS:
            raise ValueError(
                f"row {row_number} has unsupported model format: {model_ext}"
            )
        reference_ext = os.path.splitext(reference_path)[1].lower()
        if reference_ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"row {row_number} has unsupported reference format: "
                f"{reference_ext}"
            )
        if os.path.splitext(output_path)[1].lower() != ".glb":
            raise ValueError(f"row {row_number} output_path must end in .glb")
        if not os.path.isfile(model_path):
            raise ValueError(f"row {row_number} model is missing: {model_path}")
        if not os.path.isfile(reference_path):
            raise ValueError(
                f"row {row_number} reference is missing: {reference_path}"
            )

        model_key = os.path.normcase(model_path)
        output_key = os.path.normcase(output_path)
        if model_key in seen_models:
            raise ValueError(f"row {row_number} repeats model_path")
        if output_key in seen_outputs:
            raise ValueError(f"row {row_number} repeats output_path")
        seen_models.add(model_key)
        seen_outputs.add(output_key)

        entries.append({
            "model_path": model_path,
            "relative_path": os.path.relpath(model_path, root),
            "reference_path": reference_path,
            "output_relative_path": os.path.relpath(output_path, root),
        })

    return root, entries


def discover_model_files(root: str, recursive: bool = True) -> list[dict[str, str]]:
    """Discover DAE and GLB inputs in deterministic path order."""
    root_abs = os.path.realpath(root)
    paths: list[str] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root_abs):
            dirnames.sort()
            for filename in sorted(filenames):
                if os.path.splitext(filename)[1].lower() in SUPPORTED_MODEL_EXTENSIONS:
                    paths.append(os.path.join(dirpath, filename))
    else:
        for filename in sorted(os.listdir(root_abs)):
            path = os.path.join(root_abs, filename)
            if (os.path.isfile(path) and
                    os.path.splitext(filename)[1].lower() in
                    SUPPORTED_MODEL_EXTENSIONS):
                paths.append(path)

    entries = []
    for path in paths:
        relative = os.path.relpath(path, root_abs)
        output_relative = os.path.splitext(relative)[0] + ".glb"
        entries.append({
            "model_path": path,
            "relative_path": relative,
            "reference_path": "",
            "output_relative_path": output_relative,
        })
    return entries


def filename_contains_model_stem(filename: str, model_stem: str) -> bool:
    """Match a model stem without allowing house 25 to match house 250."""
    if not model_stem:
        return False
    pattern = rf"(?<![0-9A-Za-z]){re.escape(model_stem)}(?![0-9A-Za-z])"
    return re.search(pattern, filename, flags=re.IGNORECASE) is not None
