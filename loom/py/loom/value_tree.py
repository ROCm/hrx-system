# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Static grouping of ordinary SSA operands.

Tuples are records, lists are ranges, None is absence, OUT consumes the next
result, and other objects are input leaves. The preorder shape stores no value
identities: replacing or remapping SSA operands does not change it. Each token's
low two bits identify its kind; the remaining bits count immediate children.
Leaf and absent tokens have no children. A value token's payload distinguishes
input (0) and result (1).
Both the shape and the flattened values fit Loom's u16 field counts.
"""

from collections.abc import Sequence
from typing import Any

LEAF = 0
ABSENT = 1
RECORD = 2
RANGE = 3
RESULT = 4
OUT = object()
MAX_COUNT = 65535


def is_flat(shape: Sequence[int], value_count: int) -> bool:
    """Tests a constructed tree for the implicit flat input record spelling."""
    return len(shape) == value_count + 1 and shape[0] == ((value_count << 2) | RECORD)


def validate(shape: Sequence[int], value_count: int, result_count: int = 0) -> None:
    """Validates an external shape and its exact operand coverage."""
    if (
        not 0 < len(shape) <= MAX_COUNT
        or not 0 <= value_count <= MAX_COUNT
        or not 0 <= result_count <= MAX_COUNT
    ):
        raise ValueError("value tree exceeds the u16 shape or operand domain")
    pending = 1
    leaves = 0
    results = 0
    for position, token in enumerate(shape):
        if not pending:
            raise ValueError("value tree has trailing nodes")
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise ValueError("value tree token must be a nonnegative integer")
        kind = token & 3
        children = token >> 2
        if token == RESULT:
            results += 1
            children = 0
        if kind < RECORD and children:
            raise ValueError("value tree leaf or absence cannot have children")
        pending += children - 1
        if pending > len(shape) - position - 1:
            raise ValueError("value tree has missing children")
        leaves += token == LEAF
    if pending:
        raise ValueError("value tree is incomplete")
    if leaves != value_count:
        raise ValueError(
            f"value tree describes {leaves} operands, expected {value_count}"
        )
    if results != result_count:
        raise ValueError(
            f"value tree describes {results} results, expected {result_count}"
        )


def flatten(
    tree: Any, *, maximum_count: int = MAX_COUNT
) -> tuple[list[Any], list[int]]:
    """Flattens a tree, allowing a temporary implicit root beyond stored limits."""
    values: list[Any] = []
    shape: list[int] = []
    # Exit entries distinguish legal shared subtrees from container cycles.
    pending = [(tree, False)]
    pending_node_count = 1
    active: set[int] = set()
    while pending:
        node, exiting = pending.pop()
        if exiting:
            active.remove(id(node))
            continue
        pending_node_count -= 1
        if len(shape) == maximum_count:
            raise ValueError("value tree exceeds the u16 shape domain")
        if isinstance(node, (tuple, list)):
            if id(node) in active:
                raise ValueError("value tree contains a container cycle")
            if len(node) > MAX_COUNT:
                raise ValueError("value tree group exceeds the u16 child domain")
            active.add(id(node))
            kind = RECORD if isinstance(node, tuple) else RANGE
            shape.append((len(node) << 2) | kind)
            pending_node_count += len(node)
            if len(shape) + pending_node_count > maximum_count:
                raise ValueError("value tree exceeds the u16 shape domain")
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(node))
        elif node is None:
            shape.append(ABSENT)
        elif node is OUT:
            shape.append(RESULT)
        else:
            shape.append(LEAF)
            values.append(node)
    return values, shape


def format_tree(
    shape: Sequence[int], values: Sequence[str], result_count: int = 0
) -> str:
    """Renders validated grouping without a Python recursion-depth limit."""
    validate(shape, len(values), result_count)
    pieces: list[str] = []
    # Each entry owns the remaining immediate children and closing delimiter.
    groups: list[tuple[int, str]] = []
    value_position = 0
    for token in shape:
        if groups:
            remaining, close = groups.pop()
            groups.append((remaining - 1, close))
        kind = token & 3
        children = token >> 2
        if token == RESULT:
            pieces.append("out")
        elif kind == LEAF:
            pieces.append(values[value_position])
            value_position += 1
        elif kind == ABSENT:
            pieces.append("none")
        else:
            pieces.append("(" if kind == RECORD else "[")
            groups.append((children, ")" if kind == RECORD else "]"))
            if children:
                continue
        while groups and groups[-1][0] == 0:
            pieces.append(groups.pop()[1])
        if groups:
            pieces.append(", ")
    return "".join(pieces)
