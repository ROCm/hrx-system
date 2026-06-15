# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang/TIR assumption import helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from loom.builder import TiedResultSpec, ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import TileLangConverter
from loom.importers.tilelang.nodes import node_kind, node_text, source_name
from loom.importers.tilelang.ops.topology import integer_value
from loom.ir import INDEX, Predicate, PredicateArg, Type

ASSUME_ATTR_KEYS = {
    "tl.assume",
    "tilelang_assume",
}


@dataclass(frozen=True, slots=True)
class _AssumeValue:
    source: object
    ref: ValueRef


@dataclass(frozen=True, slots=True)
class _ExtractedPredicate:
    predicate: Predicate
    values: tuple[_AssumeValue, ...]


@dataclass(slots=True)
class _AssumeGroup:
    domain: str
    values: list[_AssumeValue]
    predicates: list[Predicate]


_NormalizedComparison = tuple[
    PredicateArg,
    _AssumeValue | None,
    PredicateArg,
    _AssumeValue | None,
]


def convert_assume_attr_stmt(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> bool:
    """Import a scoped TileLang assumption AttrStmt."""

    child = context.fork(preview_block=context.preview_block)
    condition = getattr(stmt, "node", None)
    if apply_assumption(condition, child, converter, owner=stmt):
        converter.convert_stmt(getattr(stmt, "body", None), child)
        child.record_converted(
            node_text(stmt),
            f"AttrStmt `{getattr(stmt, 'attr_key', '')}` scoped to Loom assume",
        )
        context.merge_child_records(child)
        return True
    context.merge_child_records(child)
    return False


def convert_assume_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> bool:
    """Import a tir.assume effect call into the current statement scope."""

    args = tuple(getattr(expr, "args", ()))
    if len(args) != 1:
        context.record_blocked(node_text(expr), "tir.assume expects one condition")
        return False
    return apply_assumption(args[0], context, converter, owner=expr)


def apply_assumption(
    condition: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    owner: object,
) -> bool:
    """Build Loom assume ops for a supported TileLang/TIR predicate."""

    extracted = _extract_condition(condition, context, converter)
    if extracted is None:
        context.record_blocked(
            node_text(owner),
            "assume condition is not a supported integer predicate",
        )
        return False
    if not extracted:
        context.record_converted(node_text(owner), "constant true Loom assume")
        return True
    groups = _group_predicates(extracted, context, owner=owner)
    if groups is None:
        return False
    for group in groups:
        _build_assume_group(group, context)
    context.record_converted(node_text(owner), "Loom assume")
    return True


def _extract_condition(
    condition: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> tuple[_ExtractedPredicate, ...] | None:
    constant = _condition_static_bool(condition)
    if constant is not None:
        return () if constant else None
    kind = node_kind(condition)
    if kind == "And":
        source_lhs = getattr(condition, "a", None)
        source_rhs = getattr(condition, "b", None)
        lhs_constant = _condition_static_bool(source_lhs)
        rhs_constant = _condition_static_bool(source_rhs)
        if lhs_constant is False or rhs_constant is False:
            return None
        if lhs_constant is True:
            return _extract_condition(source_rhs, context, converter)
        if rhs_constant is True:
            return _extract_condition(source_lhs, context, converter)
        lhs = _extract_condition(source_lhs, context, converter)
        rhs = _extract_condition(source_rhs, context, converter)
        if lhs is None or rhs is None:
            return None
        return lhs + rhs
    if kind == "Or":
        source_lhs = getattr(condition, "a", None)
        source_rhs = getattr(condition, "b", None)
        lhs_constant = _condition_static_bool(source_lhs)
        rhs_constant = _condition_static_bool(source_rhs)
        if lhs_constant is True or rhs_constant is True:
            return ()
        if lhs_constant is False:
            return _extract_condition(source_rhs, context, converter)
        if rhs_constant is False:
            return _extract_condition(source_lhs, context, converter)
        return None
    if kind in _COMPARISON_PREDICATES:
        return _extract_comparison(condition, context, converter)
    return None


def _condition_static_bool(condition: object) -> bool | None:
    if isinstance(condition, bool):
        return condition
    value = integer_value(condition)
    if value in (0, 1):
        return bool(value)
    return None


def _extract_comparison(
    condition: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> tuple[_ExtractedPredicate, ...] | None:
    kind = node_kind(condition)
    lhs = getattr(condition, "a", None)
    rhs = getattr(condition, "b", None)
    divisibility = _extract_divisibility(kind, lhs, rhs, context, converter)
    if divisibility is not None:
        return (divisibility,)
    lhs_arg = _predicate_arg(lhs, context, converter)
    rhs_arg = _predicate_arg(rhs, context, converter)
    if lhs_arg is None or rhs_arg is None:
        return None
    lhs_predicate_arg, lhs_value = lhs_arg
    rhs_predicate_arg, rhs_value = rhs_arg
    normalized = _normalize_mixed_address_comparison(
        lhs_predicate_arg,
        lhs_value,
        rhs_predicate_arg,
        rhs_value,
        context,
    )
    if normalized is None:
        return None
    lhs_predicate_arg, lhs_value, rhs_predicate_arg, rhs_value = normalized
    if lhs_value is None and rhs_value is None:
        return _constant_comparison(kind, lhs_predicate_arg, rhs_predicate_arg)
    values = tuple(
        _dedupe_values(value for value in (lhs_value, rhs_value) if value is not None)
    )
    return (
        _ExtractedPredicate(
            predicate=Predicate(
                _COMPARISON_PREDICATES[kind],
                (lhs_predicate_arg, rhs_predicate_arg),
            ),
            values=values,
        ),
    )


def _normalize_mixed_address_comparison(
    lhs_arg: PredicateArg,
    lhs_value: _AssumeValue | None,
    rhs_arg: PredicateArg,
    rhs_value: _AssumeValue | None,
    context: TileLangConversionContext,
) -> _NormalizedComparison | None:
    lhs_domain = _assume_value_domain(lhs_value)
    rhs_domain = _assume_value_domain(rhs_value)
    if lhs_domain is None or rhs_domain is None or lhs_domain == rhs_domain:
        return lhs_arg, lhs_value, rhs_arg, rhs_value
    if {lhs_domain, rhs_domain} != {"address", "scalar"}:
        return None
    if lhs_domain == "scalar":
        lhs_value = _assume_value_as_index(lhs_value, context)
        if lhs_value is None:
            return None
        lhs_arg = _value_predicate_arg(lhs_value.ref, context)
    if rhs_domain == "scalar":
        rhs_value = _assume_value_as_index(rhs_value, context)
        if rhs_value is None:
            return None
        rhs_arg = _value_predicate_arg(rhs_value.ref, context)
    return lhs_arg, lhs_value, rhs_arg, rhs_value


def _assume_value_domain(value: _AssumeValue | None) -> str | None:
    if value is None:
        return None
    return _assume_domain(value.ref)


def _assume_value_as_index(
    value: _AssumeValue | None,
    context: TileLangConversionContext,
) -> _AssumeValue | None:
    if value is None:
        return None
    if _assume_domain(value.ref) == "address":
        return value
    if not _is_integer_scalar(value.ref):
        return None
    existing = context.mapped_index_value(value.source)
    if existing is not None:
        return _AssumeValue(source=value.source, ref=existing)
    result = context.builder.index.cast(
        input=value.ref,
        results=[INDEX],
        name=context.fresh_name(
            f"{source_name(value.source, fallback='value')}_idx",
        ),
    )
    context.map_index_value(value.source, result)
    context.record_converted(
        node_text(value.source),
        f"{context.ssa(result)} = index.cast",
    )
    return _AssumeValue(source=value.source, ref=result)


def _extract_divisibility(
    kind: str,
    lhs: object,
    rhs: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> _ExtractedPredicate | None:
    if kind != "EQ":
        return None
    mod_expr: object | None = None
    if integer_value(rhs) == 0 and node_kind(lhs) == "FloorMod":
        mod_expr = lhs
    elif integer_value(lhs) == 0 and node_kind(rhs) == "FloorMod":
        mod_expr = rhs
    if mod_expr is None:
        return None
    value = _assume_value(getattr(mod_expr, "a", None), context, converter)
    modulus = integer_value(getattr(mod_expr, "b", None))
    if value is None or modulus is None:
        return None
    if modulus <= 0:
        return None
    return _ExtractedPredicate(
        predicate=Predicate(
            "mul",
            (_value_predicate_arg(value.ref, context), PredicateArg("const", modulus)),
        ),
        values=(value,),
    )


def _constant_comparison(
    kind: str,
    lhs: PredicateArg,
    rhs: PredicateArg,
) -> tuple[_ExtractedPredicate, ...] | None:
    if (
        lhs.tag != "const"
        or rhs.tag != "const"
        or not isinstance(lhs.value, int)
        or not isinstance(rhs.value, int)
    ):
        return None
    match kind:
        case "EQ":
            value = lhs.value == rhs.value
        case "NE":
            value = lhs.value != rhs.value
        case "LT":
            value = lhs.value < rhs.value
        case "LE":
            value = lhs.value <= rhs.value
        case "GT":
            value = lhs.value > rhs.value
        case "GE":
            value = lhs.value >= rhs.value
        case _:
            return None
    return () if value else None


def _predicate_arg(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> tuple[PredicateArg, _AssumeValue | None] | None:
    value = integer_value(expr)
    if value is not None:
        return PredicateArg("const", value), None
    assume_value = _assume_value(expr, context, converter)
    if assume_value is None:
        return None
    return _value_predicate_arg(assume_value.ref, context), assume_value


def _assume_value(
    source: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> _AssumeValue | None:
    ref = context.mapped(source)
    if ref is None:
        ref = converter.convert_expr(source, context)
    if ref is None:
        return None
    return _AssumeValue(source=source, ref=ref)


def _value_predicate_arg(
    ref: ValueRef,
    context: TileLangConversionContext,
) -> PredicateArg:
    return PredicateArg("value", context.ssa(ref).removeprefix("%"))


def _group_predicates(
    extracted: tuple[_ExtractedPredicate, ...],
    context: TileLangConversionContext,
    *,
    owner: object,
) -> tuple[_AssumeGroup, ...] | None:
    groups_by_domain: dict[str, _AssumeGroup] = {}
    for extracted_predicate in extracted:
        value_domains = tuple(
            _assume_domain(value.ref) for value in extracted_predicate.values
        )
        if any(domain is None for domain in value_domains):
            context.record_blocked(
                node_text(owner),
                "assume predicate has non-integer or boolean operands",
            )
            return None
        domains = {domain for domain in value_domains if domain is not None}
        if len(domains) != 1:
            context.record_blocked(
                node_text(owner),
                "assume predicate spans scalar and address domains",
            )
            return None
        domain = next(iter(domains))
        group = groups_by_domain.setdefault(
            domain,
            _AssumeGroup(domain=domain, values=[], predicates=[]),
        )
        group.predicates.append(extracted_predicate.predicate)
        for value in extracted_predicate.values:
            if all(existing.ref.id != value.ref.id for existing in group.values):
                group.values.append(value)
    return tuple(groups_by_domain.values())


def _build_assume_group(
    group: _AssumeGroup,
    context: TileLangConversionContext,
) -> None:
    values = [value.ref for value in group.values]
    names = [
        context.fresh_name(
            f"{source_name(value.source, fallback='value')}_assumed",
        )
        for value in group.values
    ]
    result_types: list[Type | TiedResultSpec] = [
        value.ref.type for value in group.values
    ]
    if group.domain == "address":
        results = context.builder.index.assume(
            values=values,
            predicates=group.predicates,
            results=result_types,
            names=names,
        )
    else:
        results = context.builder.scalar.assume(
            values=values,
            predicates=group.predicates,
            results=result_types,
            names=names,
        )
    for source_value, result in zip(group.values, results, strict=True):
        if _maps_as_index_value(source_value, result, context):
            context.map_index_value(source_value.source, result)
        else:
            context.map_value(source_value.source, result, str(result.type))
        buffer_access_key = context.buffer_access_source_key(source_value.source)
        if buffer_access_key is not None:
            context.map_buffer_access_key(buffer_access_key, result)


def _maps_as_index_value(
    source_value: _AssumeValue,
    result: ValueRef,
    context: TileLangConversionContext,
) -> bool:
    if _assume_domain(result) != "address":
        return False
    mapped = context.mapped(source_value.source)
    return mapped is not None and _assume_domain(mapped) == "scalar"


def _assume_domain(ref: ValueRef) -> str | None:
    value_type = str(ref.type)
    if value_type in ("index", "offset"):
        return "address"
    if value_type != "i1" and value_type.startswith("i"):
        return "scalar"
    return None


def _is_integer_scalar(ref: ValueRef) -> bool:
    return _assume_domain(ref) == "scalar"


def _dedupe_values(values: Iterable[_AssumeValue]) -> tuple[_AssumeValue, ...]:
    deduped: list[_AssumeValue] = []
    for value in values:
        if any(existing.ref.id == value.ref.id for existing in deduped):
            continue
        deduped.append(value)
    return tuple(deduped)


_COMPARISON_PREDICATES = {
    "EQ": "eq",
    "NE": "ne",
    "LT": "lt",
    "LE": "le",
    "GT": "gt",
    "GE": "ge",
}
