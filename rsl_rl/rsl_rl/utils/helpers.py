# SPDX-License-Identifier: BSD-3-Clause
# Utility functions for rsl_rl (copied to avoid circular import with legged_gym)

def class_to_dict(obj) -> dict:
    """Convert class attributes to dict recursively"""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result
