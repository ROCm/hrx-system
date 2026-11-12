# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source expression demand across structured value transport.

The immutable input module owns the SSA identities; this per-render plan owns
their C++ materialization requirement. A complete source value transports its
dimensions and layout without separate expressions. Independent operations,
constructors, and predicates still require their referenced scalar values.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

from loom.format.cxx.context import walk_operations
from loom.ir import (
    DynamicDim,
    DynamicEncoding,
    Module,
    Operation,
    ParameterizedType,
    PredicateListAttr,
    ShapedType,
)

# Source constructors observe these parts of their result's dependent type.
# View copies and ordinary value uses carry the same information implicitly.
_CONSTRUCTOR_COMPONENTS = {
    "buffer.view": (True, True),
    "vector.refine": (True, False),
    "vector.load": (True, False),
    "vector.splat": (True, False),
    "vector.slice": (True, False),
    "view.subview": (True, False),
    "command.parameter": (True, True),
    "channel.bind": (False, True),
}


def _constructor_operands(module: Module, operation: Operation) -> Iterable[int]:
    components = _CONSTRUCTOR_COMPONENTS.get(operation.name)
    if components is None:
        return
    dimensions, encoding = components
    value_type = module.values[operation.results[0]].type
    if isinstance(value_type, ParameterizedType):
        value_type = value_type.get("payload")
    value_type = cast(ShapedType, value_type)
    if dimensions:
        for dimension in value_type.dims:
            if isinstance(dimension, DynamicDim) and dimension.value_id is not None:
                yield dimension.value_id
    if encoding and isinstance(value_type.encoding, DynamicEncoding):
        yield value_type.encoding.value_id


def plan_value_transport(
    module: Module,
    resolve: Callable[[Operation], Any],
    parents: Mapping[int, Operation],
    source_operands: Callable[[Operation], Sequence[int]],
) -> frozenset[int]:
    """Returns structured identities requiring no independent C++ carrier.

    Each transfer edge is retained once, then a backward worklist propagates
    observation demand. The result is valid until the input module changes.
    Rendering consumes it directly, without following nested forwarding chains.
    Producers and effects remain executable even when a carrier is omitted.
    """

    sources: dict[int, set[int]] = defaultdict(set)
    observed: set[int] = set()

    def connect(targets: Sequence[int], operands: Sequence[int]) -> None:
        for target, operand in zip(targets, operands, strict=True):
            sources[target].add(operand)

    def conditional(operation: Operation) -> Sequence[int]:
        return [resolve(operation).value_id("condition")]

    def counted(operation: Operation) -> Sequence[int]:
        fields = resolve(operation)
        body = fields.region("body").blocks[0]
        arguments = body.arg_ids[1:]
        connect(arguments, fields.value_ids("iter_args"))
        # The body argument and final result share one source loop variable.
        connect(operation.results, arguments)
        connect(arguments, operation.results)
        return [
            value
            for field in (
                "lower_bound",
                "upper_bound",
                "step",
                "unroll_factor",
                "pipeline_depth",
            )
            for value in fields.value_ids(field)
        ]

    def condition_loop(operation: Operation) -> Sequence[int]:
        fields = resolve(operation)
        before = fields.region("before").blocks[0]
        after = fields.region("after").blocks[0]
        connect(before.arg_ids, fields.value_ids("iter_args"))
        connect(after.arg_ids, operation.results)
        connect(operation.results, after.arg_ids)
        # An identity header keeps the same source variables even when only
        # the next iteration observes a carried value, not the final result.
        terminator = before.ops[-1]
        if (
            terminator.name == "scf.condition"
            and resolve(terminator).value_ids("forwarded") == before.arg_ids
        ):
            connect(before.arg_ids, operation.results)
        return ()

    def condition(operation: Operation) -> Sequence[int]:
        fields = resolve(operation)
        connect(parents[id(operation)].results, fields.value_ids("forwarded"))
        return [fields.value_id("condition")]

    def yielded(operation: Operation) -> Sequence[int]:
        parent = parents[id(operation)]
        targets = parent.results
        if parent.name == "scf.while":
            targets = resolve(parent).region("before").blocks[0].arg_ids
        connect(targets, operation.operands)
        return ()

    def declaration(operation: Operation) -> Sequence[int]:
        return ()

    projections = {
        "scf.if": conditional,
        "scf.for": counted,
        "scf.while": condition_loop,
        "scf.condition": condition,
        "scf.yield": yielded,
        "func.decl": declaration,
        "command.decl": declaration,
        "pipeline.decl": declaration,
    }
    for operation in walk_operations(module.body):
        project = projections.get(operation.name, source_operands)
        observed.update(project(operation))
        observed.update(_constructor_operands(module, operation))
        for attribute in operation.attributes.values():
            if isinstance(attribute, PredicateListAttr):
                observed.update(
                    argument.value
                    for predicate in attribute
                    for argument in predicate.args
                    if argument.tag == "value"
                )

    pending = list(observed)
    while pending:
        for source in sources.get(pending.pop(), ()):
            if source not in observed:
                observed.add(source)
                pending.append(source)
    return frozenset(sources.keys() - observed)
