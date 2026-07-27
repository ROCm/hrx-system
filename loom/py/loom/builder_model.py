# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared model for typed Loom Python builders.

This module derives the public Python builder surface from op declarations.
Both the dynamic runtime builders and generated `.pyi` stubs consume this
model so builder behavior and type tooling stay aligned.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass
from enum import Enum, auto

from loom.assembly import (
    Attr,
    AttrDict,
    AttrTable,
    BindingList,
    BlockArgs,
    BlockRef,
    Clause,
    DescriptorRef,
    Flags,
    FormatElement,
    FuncArgs,
    Glue,
    IndexList,
    Keyword,
    OperandDict,
    OpRef,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    RegionTable,
    ResultType,
    ResultTypeList,
    Scope,
    StableKeyRef,
    SymbolRef,
    TemplateParam,
    TemplateParamFlags,
    TypedRefs,
    TypeOf,
    TypesOf,
)
from loom.assembly import (
    Region as RegionFmt,
)
from loom.dsl import AttrDef, Op
from loom.fields import FieldKind, compute_layout

__all__ = [
    "BuilderParam",
    "BuilderParamKind",
    "BuilderSignature",
    "attr_type_hint",
    "builder_method_names",
    "signature_for_op",
    "signatures_for_ops",
]


_PYTHON_KEYWORDS = frozenset(keyword.kwlist)


class BuilderParamKind(Enum):
    """Kind of public builder parameter and its lowering behavior."""

    ATTR = auto()
    BLOCK_ARGS = auto()
    DESCRIPTOR_REF = auto()
    FLAGS = auto()
    FUNC_ARGS = auto()
    INDEX_LIST = auto()
    OPERAND = auto()
    OPERAND_DICT = auto()
    OPERAND_VARIADIC = auto()
    PREDICATE_LIST = auto()
    REGION = auto()
    REGION_TABLE_CASES = auto()
    REGION_TABLE_DEFAULT = auto()
    RESULT_TYPES = auto()
    SUCCESSOR = auto()


@dataclass(frozen=True, slots=True)
class BuilderParam:
    """One public parameter in an op builder signature."""

    name: str
    kind: BuilderParamKind
    type_hint: str
    required: bool = True
    doc: str = ""
    attr_def: AttrDef | None = None
    binding_kind: str | None = None
    names_field: str | None = None
    region_field: str | None = None
    region_optional: bool = False
    stable_id_field: str | None = None
    ordinal_field: str | None = None
    static_field: str | None = None

    @property
    def py_name(self) -> str:
        """Python-safe public parameter name."""
        return python_name(self.name)


@dataclass(frozen=True, slots=True)
class BuilderSignature:
    """Complete public builder signature for one op."""

    op: Op
    method_name: str
    params: tuple[BuilderParam, ...]
    return_hint: str


def attr_type_hint(attr_def: AttrDef | None) -> str:
    """Return the public Python type hint for an op attribute."""
    if attr_def is None:
        return "Any"
    match attr_def.attr_type:
        case "i64":
            return "int"
        case "f64":
            return "float"
        case "string":
            return "str"
        case "bool":
            return "bool"
        case "enum":
            return "str"
        case "symbol":
            return "str"
        case "type":
            return "Type"
        case "i64_array":
            return "list[int]"
        case "any":
            return "Any"
        case "dict":
            return "Mapping[str, Any]"
        case _:
            return "Any"


def builder_method_names(ops: tuple[Op, ...] | list[Op]) -> list[str]:
    """Return stable Python method names for a dialect's ops.

    Most ops use the final op-name segment (`scalar.addi` -> `addi`). When
    that segment collides within a dialect, use the full dialect-local suffix
    (`vector.load.mask` -> `load_mask`) so dotted op families remain distinct.
    """
    short_name_counts: dict[str, int] = {}
    for op in ops:
        short_name_counts[op.short_name] = short_name_counts.get(op.short_name, 0) + 1

    method_names: list[str] = []
    used_by: dict[str, str] = {}
    for op in ops:
        if op.builder_name is not None:
            method_name = op.builder_name
        elif short_name_counts[op.short_name] == 1:
            method_name = op.short_name
        else:
            dialect_separator = op.name.find(".")
            local_name = (
                op.name[dialect_separator + 1 :] if dialect_separator >= 0 else op.name
            )
            method_name = local_name.replace(".", "_")
        if method_name in _PYTHON_KEYWORDS:
            method_name = python_name(method_name)
        prior_op_name = used_by.get(method_name)
        if prior_op_name is not None:
            raise ValueError(
                "Python builder method name collision: "
                f"{prior_op_name!r} and {op.name!r} both map to {method_name!r}"
            )
        used_by[method_name] = op.name
        method_names.append(method_name)
    return method_names


def python_name(name: str) -> str:
    """Return a Python-safe spelling for a dialect, op, or parameter name."""
    if name in _PYTHON_KEYWORDS:
        return name + "_"
    return name


def signature_for_op(op: Op, method_name: str | None = None) -> BuilderSignature:
    """Derive the public builder signature for one op."""
    params = _extract_params(op)
    has_result_types = any(
        param.kind == BuilderParamKind.RESULT_TYPES for param in params
    )
    if op.results and not has_result_types:
        params.append(
            BuilderParam(
                name="results",
                kind=BuilderParamKind.RESULT_TYPES,
                type_hint="list[Type | TiedResultSpec]",
                doc="Result types.",
            )
        )
    _check_duplicate_param_names(op, params)
    return BuilderSignature(
        op=op,
        method_name=method_name or _single_op_method_name(op),
        params=tuple(params),
        return_hint=_return_hint(op),
    )


def signatures_for_ops(ops: tuple[Op, ...] | list[Op]) -> dict[str, BuilderSignature]:
    """Derive method-name keyed builder signatures for a dialect's ops."""
    method_names = builder_method_names(ops)
    return {
        method_name: signature_for_op(op, method_name)
        for op, method_name in zip(ops, method_names, strict=True)
    }


def _single_op_method_name(op: Op) -> str:
    method_name = op.short_name.replace(".", "_")
    return python_name(method_name)


def _check_duplicate_param_names(op: Op, params: list[BuilderParam]) -> None:
    seen: dict[str, str] = {}
    for param in params:
        prior = seen.get(param.py_name)
        if prior is not None:
            raise ValueError(
                f"Op '{op.name}': builder parameters {prior!r} and "
                f"{param.name!r} both map to Python name {param.py_name!r}"
            )
        seen[param.py_name] = param.name


def _return_hint(op: Op) -> str:
    if not op.results:
        return "None"
    if len(op.results) == 1:
        if any(result.variadic for result in op.results):
            return "list[ValueRef]"
        return "ValueRef"
    return "list[ValueRef]"


def _extract_params(op: Op) -> list[BuilderParam]:  # noqa: C901
    """Extract public builder parameters from an op's assembly format."""
    layout = compute_layout(op)
    params: list[BuilderParam] = []
    covered_attrs: set[str] = set()
    region_defs = {region.name: region for region in op.regions}

    def append_attr_param(name: str) -> None:
        attr_def = op.attr(name)
        params.append(
            BuilderParam(
                name=name,
                kind=BuilderParamKind.ATTR,
                type_hint=attr_type_hint(attr_def),
                required=not (attr_def.optional if attr_def else False),
                attr_def=attr_def,
                doc=attr_def.doc if attr_def else "",
            )
        )
        covered_attrs.add(name)

    def append_ref_param(name: str) -> None:
        if name in ("iv",):
            return
        desc = layout.fields.get(name)
        if desc and desc.kind == FieldKind.OPERAND:
            params.append(
                BuilderParam(
                    name=name,
                    kind=BuilderParamKind.OPERAND,
                    type_hint="ValueRef | None" if desc.optional else "ValueRef",
                    required=not desc.optional,
                    doc=f"Operand: {name}",
                )
            )

    def append_block_args_param(name: str) -> None:
        region_def = region_defs.get(name)
        has_derived_args = bool(
            region_def is not None
            and (region_def.arg_source or region_def.implicit_args)
        )
        if has_derived_args:
            return
        params.append(
            BuilderParam(
                name=f"{name}_args",
                kind=BuilderParamKind.BLOCK_ARGS,
                type_hint="Sequence[tuple[str, Type]]",
                required=False,
                region_field=name,
                doc=f"Entry block args for region: {name}",
            )
        )

    def walk(elements: tuple[FormatElement, ...] | list[FormatElement]) -> None:
        for element in elements:
            match element:
                case Ref(field=name):
                    append_ref_param(name)

                case Refs(field=name) | TypedRefs(field=name):
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.OPERAND_VARIADIC,
                            type_hint="list[ValueRef]",
                            required=False,
                            doc=f"Variadic operands: {name}",
                        )
                    )

                case BlockRef(field=name):
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.SUCCESSOR,
                            type_hint="Block",
                            doc=f"Successor: {name}",
                        )
                    )

                case (
                    Attr(field=name)
                    | SymbolRef(field=name)
                    | OpRef(field=name)
                    | TemplateParam(field=name)
                ):
                    append_attr_param(name)

                case TemplateParamFlags(param=param_name, flags=flags_name):
                    append_attr_param(param_name)
                    attr_def = op.attr(flags_name)
                    params.append(
                        BuilderParam(
                            name=flags_name,
                            kind=BuilderParamKind.FLAGS,
                            type_hint="str",
                            required=False,
                            attr_def=attr_def,
                            doc=attr_def.doc if attr_def else "Instance flags.",
                        )
                    )
                    covered_attrs.add(flags_name)

                case IndexList(dynamic=dynamic_field, static=static_field):
                    params.append(
                        BuilderParam(
                            name=dynamic_field,
                            kind=BuilderParamKind.INDEX_LIST,
                            type_hint="list[int | ValueRef]",
                            static_field=static_field,
                            doc=f"Index list: {dynamic_field}",
                        )
                    )
                    covered_attrs.add(static_field)

                case BindingList(field=name, kind=binding_kind):
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.OPERAND_VARIADIC,
                            type_hint="list[ValueRef]",
                            required=False,
                            binding_kind=binding_kind,
                            doc=f"Binding list: {name}",
                        )
                    )

                case OperandDict(operands=operand_field, names=names_field):
                    params.append(
                        BuilderParam(
                            name=operand_field,
                            kind=BuilderParamKind.OPERAND_DICT,
                            type_hint="dict[str, ValueRef]",
                            required=False,
                            names_field=names_field,
                            doc=f"Operand dictionary: {operand_field}",
                        )
                    )
                    covered_attrs.add(names_field)

                case AttrTable(keys=keys_field, values=values_field):
                    append_attr_param(keys_field)
                    params.append(
                        BuilderParam(
                            name=values_field,
                            kind=BuilderParamKind.OPERAND_VARIADIC,
                            type_hint="list[ValueRef]",
                            required=False,
                            doc=f"Attribute-keyed table values: {values_field}",
                        )
                    )
                    covered_attrs.add(keys_field)

                case RegionTable(
                    keys=keys_field,
                    case_regions=case_regions_field,
                    default_region=default_region_field,
                ):
                    append_attr_param(keys_field)
                    params.append(
                        BuilderParam(
                            name=default_region_field,
                            kind=BuilderParamKind.REGION_TABLE_DEFAULT,
                            type_hint="Region",
                            doc=f"Default region: {default_region_field}",
                        )
                    )
                    params.append(
                        BuilderParam(
                            name=case_regions_field,
                            kind=BuilderParamKind.REGION_TABLE_CASES,
                            type_hint="list[Region]",
                            doc=f"Case regions: {case_regions_field}",
                        )
                    )
                    covered_attrs.add(keys_field)

                case ResultType(field=name):
                    params.append(
                        BuilderParam(
                            name="results",
                            kind=BuilderParamKind.RESULT_TYPES,
                            type_hint="list[Type | TiedResultSpec]",
                            doc=f"Result type: {name}",
                        )
                    )

                case ResultTypeList(field=name):
                    params.append(
                        BuilderParam(
                            name="results",
                            kind=BuilderParamKind.RESULT_TYPES,
                            type_hint="list[Type | TiedResultSpec]",
                            doc=f"Result types: {name}",
                        )
                    )

                case RegionFmt(field=name):
                    region_def = region_defs.get(name)
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.REGION,
                            type_hint="Region | None",
                            required=False,
                            region_optional=region_def.optional
                            if region_def is not None
                            else False,
                            doc=f"Region: {name}",
                        )
                    )

                case OptionalGroup(elements=inner, anchor=_anchor):
                    walk(inner)

                case Scope(elements=inner):
                    walk(inner)

                case Clause(elements=inner):
                    walk(inner)

                case FuncArgs(field=name):
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.FUNC_ARGS,
                            type_hint="list[ValueRef]",
                            required=False,
                            doc=f"Function signature args: {name}",
                        )
                    )

                case BlockArgs(region=name):
                    append_block_args_param(name)

                case Flags(field=name):
                    attr_def = op.attr(name)
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.FLAGS,
                            type_hint="str",
                            required=False,
                            attr_def=attr_def,
                            doc=attr_def.doc if attr_def else "Instance flags.",
                        )
                    )
                    covered_attrs.add(name)

                case DescriptorRef(key=name, ordinal=ordinal):
                    attr_def = op.attr(name)
                    ordinal_attr = op.attr(ordinal)
                    if attr_def is None or attr_def.attr_type != "string":
                        raise ValueError(
                            f"Op '{op.name}': DescriptorRef key field "
                            f"'{name}' must be a string attr"
                        )
                    if ordinal_attr is None or ordinal_attr.attr_type != "i64":
                        raise ValueError(
                            f"Op '{op.name}': DescriptorRef ordinal field "
                            f"'{ordinal}' must be an i64 attr"
                        )
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.DESCRIPTOR_REF,
                            type_hint=attr_type_hint(attr_def),
                            attr_def=attr_def,
                            ordinal_field=ordinal,
                            doc=attr_def.doc,
                        )
                    )
                    covered_attrs.add(name)
                    covered_attrs.add(ordinal)

                case StableKeyRef(key=name, stable_id=stable_id):
                    attr_def = op.attr(name)
                    stable_id_attr = op.attr(stable_id)
                    if attr_def is None or attr_def.attr_type != "string":
                        raise ValueError(
                            f"Op '{op.name}': StableKeyRef key field "
                            f"'{name}' must be a string attr"
                        )
                    if stable_id_attr is None or stable_id_attr.attr_type != "i64":
                        raise ValueError(
                            f"Op '{op.name}': StableKeyRef stable ID field "
                            f"'{stable_id}' must be an i64 attr"
                        )
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.DESCRIPTOR_REF,
                            type_hint=attr_type_hint(attr_def),
                            attr_def=attr_def,
                            stable_id_field=stable_id,
                            doc=attr_def.doc,
                        )
                    )
                    covered_attrs.add(name)
                    covered_attrs.add(stable_id)

                case PredicateList(field=name):
                    attr_def = op.attr(name)
                    params.append(
                        BuilderParam(
                            name=name,
                            kind=BuilderParamKind.PREDICATE_LIST,
                            type_hint="list[Predicate]",
                            required=not (attr_def.optional if attr_def else False),
                            attr_def=attr_def,
                            doc=f"Predicate list: {name}",
                        )
                    )
                    covered_attrs.add(name)

                case AttrDict(field=name):
                    if name:
                        append_attr_param(name)
                    else:
                        for attr_def in op.attrs:
                            if attr_def.attr_type == "flags":
                                continue
                            if attr_def.name in covered_attrs:
                                continue
                            append_attr_param(attr_def.name)

                case Keyword() | TypeOf() | TypesOf() | Glue():
                    pass

                case _:
                    raise ValueError(
                        f"Op '{op.name}': unsupported builder format element "
                        f"{type(element).__name__}"
                    )

    walk(op.format)
    return params
