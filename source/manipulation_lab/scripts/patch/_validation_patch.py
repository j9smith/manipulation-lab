"""
Modification of the original _validate function from the Isaac Lab Project.

Patches the _validate function of isaaclab.utils.configclass
to skip objects with single underscore prefixes to avoid recursion
errors when loading a scene via Hydra config.
"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import importlib
import sys

def _patched_validate(obj: object, prefix: str="", _seen=None) -> list[str]:
    """
    Store and skip visited objects to avoid recursion errors
    """
    # Patch
    if _seen is None:
        _seen = set()
    object_id = id(obj)

    if object_id in _seen:
        return []
    _seen.add(object_id)

    # Everything else remains the same
    missing_fields = []

    if type(obj) is type(MISSING):
        missing_fields.append(prefix)
        return missing_fields
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            current_path = f"{prefix}[{index}]"
            missing_fields.extend(_patched_validate(item, prefix=current_path))
        return missing_fields
    elif isinstance(obj, dict):
        obj_dict = obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return missing_fields

    for key, value in obj_dict.items():
        if key.startswith("__") or key.startswith("_"):
            continue
        current_path = f"{prefix}.{key}" if prefix else key
        missing_fields.extend(_patched_validate(value, prefix=current_path))

    if prefix == "" and missing_fields:
        formatted_message = "\n".join(f"  - {field}" for field in missing_fields)
        raise TypeError(
            f"Missing values detected in object {obj.__class__.__name__} for the following"
            f" fields:\n{formatted_message}\n"
        )
    return missing_fields

configclass = importlib.import_module("isaaclab.utils.configclass")
_orig_validate = configclass._validate
configclass._validate = _patched_validate
