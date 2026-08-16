"""Blender-independent helpers for preserving submitted ComfyUI workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Mapping


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename_component(value: object, fallback: str) -> str:
    text = _UNSAFE_FILENAME_CHARS.sub("_", str(value or "").strip())
    text = text.strip("._-")
    return text or fallback


def _write_json_atomic(path: Path, prompt: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(prompt, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def archive_prompt(
    prompt: Mapping,
    output_dir: str | os.PathLike,
    *,
    workflow: str = "workflow",
    view_index: int | None = None,
    camera_name: str | None = None,
) -> tuple[Path, Path]:
    """Save the latest prompt plus a non-overwriting per-submission copy.

    ``prompt.json`` remains a compatibility/convenience alias for the latest
    graph. Every call also creates a JSON file under ``prompts/``. Repeated
    submissions for the same view receive a numeric suffix.
    """
    output_path = Path(output_dir)
    latest_path = output_path / "prompt.json"
    archive_dir = output_path / "prompts"

    workflow_part = _safe_filename_component(workflow, "workflow")
    stem_parts = []
    if view_index is not None:
        stem_parts.append(f"view_{int(view_index):03d}")
    else:
        stem_parts.append("submission")
    if camera_name:
        stem_parts.append(_safe_filename_component(camera_name, "camera"))
    stem_parts.append(workflow_part)
    stem = "_".join(stem_parts)

    archive_path = archive_dir / f"{stem}.json"
    suffix = 2
    while archive_path.exists():
        archive_path = archive_dir / f"{stem}_{suffix:02d}.json"
        suffix += 1

    _write_json_atomic(latest_path, prompt)
    _write_json_atomic(archive_path, prompt)
    return latest_path, archive_path
