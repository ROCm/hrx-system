# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

from build_tools.spirv.registry.dependency_expression import (
    format_dependency_expression,
    parse_dependency_expression,
)

_REGISTRY_PATH: Path | None = None


class DependencyExpressionRegistryTest(unittest.TestCase):
    def test_every_pinned_registry_dependency_round_trips(self):
        self.assertIsNotNone(_REGISTRY_PATH)
        registry_path = cast(Path, _REGISTRY_PATH)

        root = ET.parse(registry_path).getroot()
        expression_count = 0
        saw_conjunction = False
        saw_disjunction = False
        saw_grouping = False
        saw_member_reference = False
        for element in root.iter():
            source_expression = element.get("depends")
            if source_expression is None:
                continue
            expression_count += 1
            identity = (
                element.get("name")
                or element.get("struct")
                or element.get("extension")
                or element.get("feature")
                or str(expression_count)
            )
            source = f"{registry_path}:{element.tag}[{identity}]@depends"
            with self.subTest(source=source, expression=source_expression):
                expression = parse_dependency_expression(
                    source_expression,
                    source=source,
                )
                formatted = format_dependency_expression(expression)
                self.assertEqual(
                    parse_dependency_expression(formatted, source=source),
                    expression,
                )
            saw_conjunction |= "+" in source_expression
            saw_disjunction |= "," in source_expression
            saw_grouping |= "(" in source_expression
            saw_member_reference |= "::" in source_expression

        self.assertGreater(expression_count, 0)
        self.assertTrue(saw_conjunction)
        self.assertTrue(saw_disjunction)
        self.assertTrue(saw_grouping)
        self.assertTrue(saw_member_reference)


def _registry_path_from_argv(argv: list[str]) -> Path:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <vk.xml>")
    registry_path = Path(argv[1])
    if not registry_path.is_file():
        raise SystemExit(f"Vulkan registry does not exist: {registry_path}")
    return registry_path


if __name__ == "__main__":
    _REGISTRY_PATH = _registry_path_from_argv(sys.argv)
    unittest.main(argv=[sys.argv[0]])
