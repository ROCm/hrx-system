# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import unittest

from build_tools.spirv.registry.dependency_expression import (
    DependencyAllOf,
    DependencyAnyOf,
    DependencyAtom,
    DependencyExpressionParseError,
    dependency_expression_names,
    evaluate_dependency_expression,
    format_dependency_expression,
    parse_dependency_expression,
)


def _atom(name: str) -> DependencyAtom:
    return DependencyAtom(name)


class DependencyExpressionTest(unittest.TestCase):
    def test_parses_atom_categories(self):
        for name in (
            "VK_VERSION_1_4",
            "VK_KHR_shader_bfloat16",
            "VkPhysicalDeviceVulkan12Features::descriptorIndexing",
            "VK_EXT_shader_object2",
        ):
            with self.subTest(name=name):
                self.assertEqual(parse_dependency_expression(name), _atom(name))

    def test_normalizes_same_operator_chains_in_source_order(self):
        self.assertEqual(
            parse_dependency_expression("A+B+(C+D)"),
            DependencyAllOf((_atom("A"), _atom("B"), _atom("C"), _atom("D"))),
        )
        self.assertEqual(
            parse_dependency_expression("D,C,(B,A)"),
            DependencyAnyOf((_atom("D"), _atom("C"), _atom("B"), _atom("A"))),
        )

    def test_mixed_operators_have_equal_left_associative_precedence(self):
        self.assertEqual(
            parse_dependency_expression("A,B+C"),
            DependencyAllOf((DependencyAnyOf((_atom("A"), _atom("B"))), _atom("C"))),
        )
        self.assertEqual(
            parse_dependency_expression("A+B,C"),
            DependencyAnyOf((DependencyAllOf((_atom("A"), _atom("B"))), _atom("C"))),
        )

    def test_parentheses_override_mixed_operator_association(self):
        self.assertEqual(
            parse_dependency_expression("A,(B+C)"),
            DependencyAnyOf((_atom("A"), DependencyAllOf((_atom("B"), _atom("C"))))),
        )
        self.assertEqual(
            parse_dependency_expression("A+(B,C)"),
            DependencyAllOf((_atom("A"), DependencyAnyOf((_atom("B"), _atom("C"))))),
        )

    def test_accepts_ascii_whitespace(self):
        self.assertEqual(
            parse_dependency_expression(" \tA +\n(B , C)\r"),
            DependencyAllOf((_atom("A"), DependencyAnyOf((_atom("B"), _atom("C"))))),
        )

    def test_canonical_format_round_trips_structural_expression(self):
        sources = (
            "A",
            "A+B+C",
            "A,B,C",
            "A+(B,C)+D",
            "(A+B),(C+D)",
            "((((A,B)+C),D)+E),F",
        )
        for source in sources:
            with self.subTest(source=source):
                expression = parse_dependency_expression(source)
                formatted = format_dependency_expression(expression)
                self.assertEqual(parse_dependency_expression(formatted), expression)

    def test_evaluation_matches_boolean_semantics(self):
        cases = (
            ("A+B", lambda a, b, c: a and b),
            ("A,B", lambda a, b, c: a or b),
            ("A,B+C", lambda a, b, c: (a or b) and c),
            ("A,(B+C)", lambda a, b, c: a or (b and c)),
        )
        for source, expected in cases:
            expression = parse_dependency_expression(source)
            for a, b, c in itertools.product((False, True), repeat=3):
                supported = {"A": a, "B": b, "C": c}
                with self.subTest(source=source, supported=supported):
                    self.assertEqual(
                        evaluate_dependency_expression(
                            expression, supported.__getitem__
                        ),
                        expected(a, b, c),
                    )

    def test_collects_referenced_names(self):
        expression = parse_dependency_expression("A+(B,C)+A")
        self.assertEqual(
            dependency_expression_names(expression),
            frozenset(("A", "B", "C")),
        )

    def test_rejects_malformed_expressions(self):
        malformed = (
            "",
            " \t\n",
            "+A",
            "A+",
            "A++B",
            "A,,B",
            "()",
            "(A+B",
            "A+B)",
            "A B",
            "A-B",
            "!A",
        )
        for source in malformed:
            with self.subTest(source=source):
                with self.assertRaises(DependencyExpressionParseError):
                    parse_dependency_expression(source)

    def test_parse_error_carries_source_coordinates_and_excerpt(self):
        with self.assertRaises(DependencyExpressionParseError) as raised:
            parse_dependency_expression(
                "A+\n)",
                source="vk.xml:extension[VK_TEST]@depends",
            )
        self.assertEqual(raised.exception.position, 3)
        self.assertEqual(raised.exception.line, 2)
        self.assertEqual(raised.exception.column, 1)
        self.assertIn(
            "vk.xml:extension[VK_TEST]@depends:2:1:",
            str(raised.exception),
        )
        self.assertTrue(str(raised.exception).endswith(")\n^"))

    def test_direct_construction_rejects_noncanonical_nodes(self):
        with self.assertRaises(ValueError):
            DependencyAtom("")
        with self.assertRaises(ValueError):
            DependencyAtom("VK-TEST")
        with self.assertRaises(ValueError):
            DependencyAllOf((_atom("A"),))
        with self.assertRaises(ValueError):
            DependencyAnyOf((_atom("A"),))
        with self.assertRaises(ValueError):
            DependencyAllOf((DependencyAllOf((_atom("A"), _atom("B"))), _atom("C")))
        with self.assertRaises(TypeError):
            DependencyAnyOf([_atom("A"), _atom("B")])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
