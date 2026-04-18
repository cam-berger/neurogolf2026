"""Identity generator: output == input via a single Identity op."""

from __future__ import annotations

import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, NetworkGenerator, make_model


class IdentityGenerator(NetworkGenerator):
    family = "identity"

    def can_generate(self, task: dict, features: dict) -> bool:
        return bool(features.get("is_identity"))

    def generate(self, task: dict, features: dict) -> onnx.ModelProto | None:
        node = helper.make_node("Identity", [INPUT_NAME], [OUTPUT_NAME], name="identity")
        return make_model([node], initializers=[], doc="identity: output = input")


__all__ = ["IdentityGenerator"]
