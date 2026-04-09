"""
schema_export — export the live RunConfig JSON schema as formatted text.

Generated programmatically from the RunConfig Pydantic model so the AI context
stays in sync with code changes automatically.  No manual update step required.

Usage
-----
    from ai_layer.knowledge_base.schema_export import get_schema_text
    text = get_schema_text()   # inject into agent system prompt
"""
from __future__ import annotations

import json
import sys
import os

# engine/ is in the project root — add it to path if needed.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def get_schema_text() -> str:
    """
    Return the RunConfig JSON schema as a readable text block for injection
    into an agent system prompt.

    Returns a formatted string showing all fields, types, and descriptions.
    Falls back to a short error notice if the engine cannot be imported
    (e.g. in a deployment where engine/ is not installed).
    """
    try:
        from engine.config.run_config import RunConfig
        schema = RunConfig.model_json_schema()
        return _format_schema(schema)
    except ImportError as exc:
        return f"[RunConfig schema unavailable — engine not importable: {exc}]"


def _format_schema(schema: dict) -> str:
    """Format a JSON schema dict as readable text for injection into a prompt."""
    lines = ["## RunConfig JSON Schema", ""]

    title = schema.get("title", "RunConfig")
    description = schema.get("description", "")
    lines.append(f"**{title}**")
    if description:
        lines.append(description)
    lines.append("")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if properties:
        lines.append("### Top-level fields")
        lines.append("")
        for field_name, field_def in properties.items():
            req_marker = " *(required)*" if field_name in required else " *(optional)*"
            field_type = _resolve_type(field_def, schema)
            description = field_def.get("description", "")
            default = field_def.get("default")
            default_str = f"  Default: `{default}`" if default is not None else ""
            lines.append(f"- **`{field_name}`**{req_marker} — `{field_type}`")
            if description:
                lines.append(f"  {description}{default_str}")
            elif default_str:
                lines.append(f"  {default_str.strip()}")

    # Append full schema as JSON for completeness.
    lines.append("")
    lines.append("### Full schema (JSON)")
    lines.append("```json")
    lines.append(json.dumps(schema, indent=2))
    lines.append("```")

    return "\n".join(lines)


def _resolve_type(field_def: dict, root_schema: dict) -> str:
    """Extract a human-readable type string from a field definition."""
    if "$ref" in field_def:
        ref = field_def["$ref"].split("/")[-1]
        return ref
    if "anyOf" in field_def:
        types = [_resolve_type(t, root_schema) for t in field_def["anyOf"]]
        return " | ".join(types)
    if "type" in field_def:
        t = field_def["type"]
        if t == "array":
            items = field_def.get("items", {})
            return f"list[{_resolve_type(items, root_schema)}]"
        return t
    return "any"
