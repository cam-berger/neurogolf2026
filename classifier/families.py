"""Transformation families recognized by the classifier."""

from __future__ import annotations

from enum import Enum


class TransformFamily(str, Enum):
    IDENTITY       = "identity"        # output == input
    COLOR_REMAP    = "color_remap"     # bijective color permutation, no spatial change
    ROT90          = "rot90"           # 90 deg clockwise rotation
    ROT180         = "rot180"          # 180 deg rotation
    ROT270         = "rot270"          # 270 deg clockwise (== 90 CCW)
    FLIP_H         = "flip_h"          # left-right flip
    FLIP_V         = "flip_v"          # up-down flip
    TRANSPOSE      = "transpose"       # matrix transpose
    SCALE_UP       = "scale_up"        # integer upscale (nearest neighbor)
    LOCAL_RULE_3x3 = "local_rule_3x3"  # output depends on 3x3 neighborhood
    LOCAL_RULE_5x5 = "local_rule_5x5"  # output depends on 5x5 neighborhood
    UNKNOWN        = "unknown"         # could not classify


# Most specific -> least specific. Classifier walks this list and returns the
# first family whose gate in classifier.rules fires.
FAMILY_PRIORITY: list[TransformFamily] = [
    TransformFamily.IDENTITY,
    TransformFamily.ROT90,
    TransformFamily.ROT180,
    TransformFamily.ROT270,
    TransformFamily.FLIP_H,
    TransformFamily.FLIP_V,
    TransformFamily.TRANSPOSE,
    TransformFamily.COLOR_REMAP,
    TransformFamily.SCALE_UP,
    TransformFamily.LOCAL_RULE_3x3,
    TransformFamily.LOCAL_RULE_5x5,
    TransformFamily.UNKNOWN,
]


__all__ = ["TransformFamily", "FAMILY_PRIORITY"]
