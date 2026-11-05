# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SSA identity traversal and translation for complete types and attributes."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

from loom.ir import (
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    FunctionType,
    GroupType,
    ParameterizedAttr,
    ParameterizedAttrArray,
    ParameterizedType,
    Predicate,
    PredicateArg,
    PredicateListAttr,
    RegisterType,
    ShapedType,
)


def binding_children(value: Any) -> Iterable[Any]:
    """Immediate children that can carry SSA identity, without flattening a DAG."""
    match value:
        case ShapedType(dims=dims, encoding=encoding):
            return (*dims, encoding)
        case GroupType(dims=dimensions):
            return dimensions
        case FunctionType(arg_types=args, result_types=results):
            return (*args, *results)
        case DialectType(params=parameters):
            return parameters
        case RegisterType(value_type=child):
            return (child,) if child is not None else ()
        case ParameterizedType() | ParameterizedAttr():
            return value.slots
        case ParameterizedAttrArray(values=values) | PredicateListAttr(values=values):
            return values
        case Predicate(args=args):
            return args
        case Mapping():
            return value.values()
        case list() | tuple():
            return value
        case _:
            return ()


def iter_value_bindings(
    value: Any,
    *,
    visited: set[int] | None = None,
) -> Iterator[DynamicDim | DynamicEncoding | PredicateArg]:
    """Visit each reachable binding once, including nested TYPE attributes.

    The traversal owns only an invocation-local identity set and stack. Shared
    children are visited once; no transitive reference lists are retained on IR.
    Unbound DynamicDim entries are yielded for the verifier to diagnose.
    """
    pending = [value]
    if visited is None:
        visited = set()
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if isinstance(value, DynamicDim | DynamicEncoding):
            yield value
        elif isinstance(value, PredicateArg):
            if value.tag == "value":
                yield value
        else:
            pending.extend(binding_children(value))


def iter_value_binding_occurrences(
    value: Any,
) -> Iterator[DynamicDim | DynamicEncoding | PredicateArg]:
    """Visits every SSA binding occurrence in structural order.

    Unlike :func:`iter_value_bindings`, this preserves repeated references and
    does not collapse shared subgraphs. It mirrors ``loom_type_walk_value_refs``
    for contracts whose positional representation must match the type's exact
    SSA-reference sequence.
    """
    pending = [value]
    while pending:
        value = pending.pop()
        if isinstance(value, DynamicDim | DynamicEncoding):
            yield value
        elif isinstance(value, PredicateArg):
            if value.tag == "value":
                yield value
        else:
            pending.extend(reversed(tuple(binding_children(value))))


def remap_value_bindings(
    roots: Iterable[Any], value_ids: Mapping[int, int]
) -> list[Any]:
    """Translate a type/attribute graph, preserving shared and unchanged nodes.

    IDs outside value_ids keep their identities. Completion is iterative and
    memoized by source identity across all roots, so shared subgraphs are not
    copied or traversed repeatedly. The memo is released with this call.
    """
    roots = list(roots)
    completed: dict[int, Any] = {}
    pending = [(root, False) for root in roots]
    while pending:
        value, expanded = pending.pop()
        identity = id(value)
        if identity in completed:
            continue
        if isinstance(value, DynamicDim | DynamicEncoding):
            source = value.value_id
            target = value_ids.get(source, source) if source is not None else None
            completed[identity] = (
                value if target == source else replace(value, value_id=target)
            )
            continue
        if isinstance(value, PredicateArg):
            target = (
                value_ids.get(value.value, value.value)
                if value.tag == "value"
                else value.value
            )
            completed[identity] = (
                value if target == value.value else replace(value, value=target)
            )
            continue
        children = tuple(binding_children(value))
        if children and not expanded:
            pending.append((value, True))
            pending.extend((child, False) for child in children)
            continue
        mapped = tuple(completed[id(child)] for child in children)
        if all(a is b for a, b in zip(children, mapped, strict=True)):
            completed[identity] = value
            continue
        match value:
            case ShapedType():
                result = replace(value, dims=mapped[:-1], encoding=mapped[-1])
            case GroupType():
                result = replace(value, dims=mapped)
            case FunctionType(arg_types=args):
                result = FunctionType(mapped[: len(args)], mapped[len(args) :])
            case DialectType(name=name):
                result = DialectType(name, mapped)
            case RegisterType():
                result = replace(value, value_type=mapped[0]) if mapped else value
            case ParameterizedType():
                result = ParameterizedType(
                    value.definition,
                    {
                        parameter.name: slot
                        for parameter, slot in zip(
                            value.definition.params, mapped, strict=True
                        )
                        if slot is not None
                    },
                )
            case ParameterizedAttr():
                result = ParameterizedAttr(
                    value.definition,
                    {
                        parameter.name: slot
                        for parameter, slot in zip(
                            value.definition.parameters, mapped, strict=True
                        )
                        if slot is not None
                    },
                )
            case ParameterizedAttrArray():
                result = ParameterizedAttrArray(mapped)
            case PredicateListAttr():
                result = PredicateListAttr(mapped)
            case Predicate(kind=kind):
                result = Predicate(kind, mapped)
            case CanonicalAttrDict():
                result = CanonicalAttrDict.from_sorted_items(
                    zip(value, mapped, strict=True)
                )
            case Mapping():
                result = dict(zip(value, mapped, strict=True))
            case list():
                result = list(mapped)
            case tuple():
                result = mapped
            case _:
                result = value
        completed[identity] = result
    return [completed[id(root)] for root in roots]
