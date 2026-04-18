"""ONNX network generators, one per transformation family."""

from classifier.families import TransformFamily
from generators.base import NetworkGenerator
from generators.color_remap import ColorRemapGenerator
from generators.geometric import GeometricGenerator
from generators.identity import IdentityGenerator
from generators.local_rule import LocalRuleGenerator
from generators.tiling import TilingGenerator

GENERATORS: dict[TransformFamily, NetworkGenerator] = {
    TransformFamily.IDENTITY: IdentityGenerator(),
    TransformFamily.COLOR_REMAP: ColorRemapGenerator(),
    TransformFamily.ROT90: GeometricGenerator("rot90"),
    TransformFamily.ROT180: GeometricGenerator("rot180"),
    TransformFamily.ROT270: GeometricGenerator("rot270"),
    TransformFamily.FLIP_H: GeometricGenerator("flip_h"),
    TransformFamily.FLIP_V: GeometricGenerator("flip_v"),
    TransformFamily.TRANSPOSE: GeometricGenerator("transpose"),
    TransformFamily.SCALE_UP: TilingGenerator(),
    TransformFamily.LOCAL_RULE_3x3: LocalRuleGenerator(kernel=3),
    TransformFamily.LOCAL_RULE_5x5: LocalRuleGenerator(kernel=5),
}


__all__ = [
    "GENERATORS",
    "NetworkGenerator",
    "ColorRemapGenerator",
    "GeometricGenerator",
    "IdentityGenerator",
    "LocalRuleGenerator",
    "TilingGenerator",
]
