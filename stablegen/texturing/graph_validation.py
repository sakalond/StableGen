"""Pure validation helpers for dynamically edited ComfyUI workflow graphs."""

from __future__ import annotations

from collections.abc import Mapping


def _is_node_reference(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def validate_workflow_graph(prompt: Mapping, semantic_node_ids=()) -> list[str]:
    """Return graph errors without changing the workflow.

    Checks every ComfyUI link target and, for Qwen text encoders, verifies
    that image slots are consistent and do not accidentally alias one another.
    """
    issues: list[str] = []
    node_ids = {str(node_id) for node_id in prompt}
    for node_id, node in prompt.items():
        for input_name, value in node.get("inputs", {}).items():
            if _is_node_reference(value) and str(value[0]) not in node_ids:
                issues.append(
                    f"node {node_id} input {input_name} references missing node {value[0]}"
                )

    semantic_refs = {}
    for node_id in semantic_node_ids:
        node = prompt.get(str(node_id))
        if node is None:
            issues.append(f"semantic text encoder node {node_id} is missing")
            continue
        inputs = node.get("inputs", {})
        slots = {
            key: tuple(inputs[key])
            for key in ("image1", "image2", "image3")
            if key in inputs and _is_node_reference(inputs[key])
        }
        reverse = {}
        for slot, reference in slots.items():
            reverse.setdefault(reference, []).append(slot)
        for reference, aliases in reverse.items():
            if len(aliases) > 1:
                issues.append(
                    f"semantic node {node_id} aliases {', '.join(aliases)} to {reference[0]}"
                )
        semantic_refs[str(node_id)] = slots

    if len(semantic_refs) > 1:
        reference_slots = next(iter(semantic_refs.values()))
        for node_id, slots in list(semantic_refs.items())[1:]:
            if slots != reference_slots:
                issues.append(
                    f"semantic image slots differ between text encoders and node {node_id}"
                )
    return issues
