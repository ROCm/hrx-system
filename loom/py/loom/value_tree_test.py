# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Coverage of static operand grouping independent of an operation schema."""

import pytest

from loom import value_tree


def test_ragged_barriers_preserve_association() -> None:
    first = [("source", ["a"]), ("target", ["b", "c"])]
    second = [("source", ["a", "b"]), ("target", ["c"])]
    first_values, first_shape = value_tree.flatten(first)
    second_values, second_shape = value_tree.flatten(second)
    assert first_shape != second_shape
    assert (
        value_tree.format_tree(first_shape, first_values)
        == "[(source, [a]), (target, [b, c])]"
    )
    assert (
        value_tree.format_tree(second_shape, second_values)
        == "[(source, [a, b]), (target, [c])]"
    )


def test_absent_empty_record_and_empty_range_remain_distinct() -> None:
    values, shape = value_tree.flatten((None, (), [], [("semaphore", "value")]))
    assert values == ["semaphore", "value"]
    assert (
        value_tree.format_tree(shape, values) == "(none, (), [], [(semaphore, value)])"
    )


def test_shared_containers_are_not_cycles() -> None:
    shared = ["value"]
    values, shape = value_tree.flatten((shared, shared))
    assert values == ["value", "value"]
    assert value_tree.format_tree(shape, values) == "([value], [value])"
    shared.append(shared)
    with pytest.raises(ValueError, match="cycle"):
        value_tree.flatten(shared)


def test_deep_groups_do_not_use_python_recursion() -> None:
    tree = "value"
    for _ in range(4096):
        tree = [tree]
    values, shape = value_tree.flatten(tree)
    assert value_tree.format_tree(shape, values) == "[" * 4096 + "value" + "]" * 4096


def test_pending_children_are_bounded_before_expansion() -> None:
    tree = [None] * value_tree.MAX_COUNT
    with pytest.raises(ValueError, match="shape domain"):
        value_tree.flatten(tree)


def test_flat_operands_do_not_determine_group_association() -> None:
    first_values, first_shape = value_tree.flatten([["a"], ["b", "c"]])
    second_values, second_shape = value_tree.flatten([["a", "b"], ["c"]])
    assert first_values == second_values
    assert first_shape != second_shape


def test_result_leaves_preserve_independent_input_and_result_order() -> None:
    values, shape = value_tree.flatten(
        ([("first", value_tree.OUT), ("second", value_tree.OUT)], [])
    )
    assert values == ["first", "second"]
    assert (
        value_tree.format_tree(shape, values, 2)
        == "([(first, out), (second, out)], [])"
    )
    with pytest.raises(ValueError, match="results"):
        value_tree.validate(shape, 2, 1)


@pytest.mark.parametrize(
    ("shape", "count"),
    [
        ([], 0),
        ([0, 0], 2),
        ([6], 0),
        ([4], 1),
        ([5], 0),
        ([-1], 0),
        ([True], 0),
        ([0], 0),
        ([1], 1),
        ([1 << 63], 0),
    ],
)
def test_malformed_shapes_are_rejected(shape, count) -> None:
    with pytest.raises(ValueError, match="value tree"):
        value_tree.validate(shape, count)
