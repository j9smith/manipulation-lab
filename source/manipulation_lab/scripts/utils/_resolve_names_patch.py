"""
Modification of the original resolve_matching_names_values function from the Isaac Lab Project.

Patches the resolve_matching_names_values function of isaaclab.utils.string
to cast Hydra DictConfig objects to dictionaries before passing to the original function.

This circumvents an unexpected type error when loading a scene via Hydra config.
"""

import isaaclab.utils.string as string_utils
from collections.abc import Mapping

_original_resolve_names_values = string_utils.resolve_matching_names_values

def _patched_resolve_names_values(data, strings, preserve_order=False):
    if isinstance(data, Mapping) and not isinstance(data, dict):
        data = dict(data)
    return _original_resolve_names_values(data, strings, preserve_order)

string_utils.resolve_matching_names_values = _patched_resolve_names_values