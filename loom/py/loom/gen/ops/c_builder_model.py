# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builder parameter and pattern model for generated C builders."""

from __future__ import annotations

from typing import Any

from loom.assembly import (
    AlignedRefs,
    Attr,
    AttrDict,
    AttrParams,
    AttrTable,
    BindingList,
    BlockArgs,
    BlockRef,
    Clause,
    Flags,
    FormatElement,
    FuncArgs,
    Glue,
    IndexList,
    KeyRef,
    Keyword,
    OperandDict,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    RegionTable,
    ResultType,
    ResultTypeList,
    Scope,
    ScopedEnumRef,
    StableKeyRef,
    SymbolRef,
    TemplateParam,
    TemplateParamFlags,
    TypedRefs,
    TypeOf,
    TypesOf,
)
from loom.assembly import Region as RegionFmt
from loom.builder_model import fixed_result_type_constraints
from loom.dsl import ATTR_TYPE_FLAGS, AttrDef, FuncLikeInterface, Op, RegionDef, TiedResult, TypeConstraint
from loom.fields import FieldKind, FieldLayout, compute_layout
from loom.gen.ops import c_queries
from loom.gen.ops.c_enum_attrs import SharedEnumMap
from loom.gen.ops.c_enum_attrs import enum_c_type as _enum_c_type
from loom.gen.support.c import c_identifier as _c_identifier


def detect_builder_pattern(op: Op) -> str | None:
    """Detects if an op matches a standard builder macro pattern.

    Returns the macro name suffix or None for complex ops.
    """
    # The shared macros accept caller-provided result types. Exact result
    # constraints instead use the general generator so the public builder
    # signature cannot contradict the op declaration.
    has_explicit_result_type = any(isinstance(element, ResultType | ResultTypeList) for element in flatten_format(op.format))
    if fixed_result_type_constraints(op) is not None and not has_explicit_result_type:
        return None

    layout = compute_layout(op)
    non_flags = c_queries.non_flags_attrs(op)
    has_flags = c_queries.has_flags_attr(op)
    has_template_param = any(isinstance(e, AttrParams | TemplateParam | TemplateParamFlags) for e in flatten_format(op.format))
    operand_names = tuple(operand.name for operand in op.operands)

    # Binary: 2 fixed operands, 1 fixed result, no real attrs/regions.
    # Only use the generic lhs/rhs builder macro when the op uses those exact
    # field names; otherwise the generated C API should preserve the op's
    # semantic operand names.
    if (
        operand_names == ("lhs", "rhs")
        and layout.fixed_operand_count == 2
        and not layout.variadic_operand
        and layout.fixed_result_count == 1
        and not layout.variadic_result
        and len(non_flags) == 0
        and len(op.regions) == 0
    ):
        return "BINARY_WITH_FLAGS" if has_flags else "BINARY"

    # Unary: 1 fixed operand, 1 fixed result, no real attrs/regions.
    if (
        operand_names == ("input",)
        and layout.fixed_operand_count == 1
        and not layout.variadic_operand
        and layout.fixed_result_count == 1
        and not layout.variadic_result
        and len(non_flags) == 0
        and len(op.regions) == 0
    ):
        # Distinguish cast from unary by checking for "to" keyword.
        has_to = any(isinstance(e, Keyword) and e.text == "to" for e in op.format)
        if has_to:
            return "CAST"
        return "UNARY_WITH_FLAGS" if has_flags else "UNARY"

    # Comparison: 2 fixed operands, 1 fixed result, 1 enum attr, no regions.
    if (
        operand_names == ("lhs", "rhs")
        and layout.fixed_operand_count == 2
        and not layout.variadic_operand
        and layout.fixed_result_count == 1
        and not layout.variadic_result
        and len(non_flags) == 1
        and non_flags[0].attr_type == "enum"
        and not has_template_param
        and len(op.regions) == 0
    ):
        return "COMPARISON_WITH_FLAGS" if has_flags else "COMPARISON"

    return None


# ============================================================================
# C builder parameter extraction
# ============================================================================


_C_ATTR_TYPE_MAP: dict[str, str] = {
    "i64": "int64_t",
    "f64": "double",
    "string": "loom_string_id_t",
    "bool": "bool",
    "enum": "uint8_t",
    "enum_array": "loom_enum_array_t",
    "signed_enum_set": "loom_signed_enum_set_t",
    "symbol": "loom_symbol_ref_t",
    "symbol_array": "loom_symbol_ref_array_t",
    "symbol_set": "loom_symbol_ref_array_t",
    "i64_array": "const int64_t*",
    "bytes": "iree_const_byte_span_t",
    "type": "uint32_t",
    "encoding": "uint16_t",
    "dict": "loom_named_attr_slice_t",
    "parameterized": "loom_attribute_t",
    "parameterized_array": "loom_parameterized_attr_array_t",
    "any": "loom_attribute_t",
}

# Maps implicit block arg type keywords to C scalar type constants.
IMPLICIT_ARG_TYPE_MAP: dict[str, str] = {
    "index": "LOOM_SCALAR_TYPE_INDEX",
    "i32": "LOOM_SCALAR_TYPE_I32",
    "i64": "LOOM_SCALAR_TYPE_I64",
    "f32": "LOOM_SCALAR_TYPE_F32",
    "f64": "LOOM_SCALAR_TYPE_F64",
}

_FIXED_RESULT_TYPE_CONSTRAINTS = frozenset(
    {
        TypeConstraint.I1,
        TypeConstraint.I32,
        TypeConstraint.INDEX,
        TypeConstraint.OFFSET,
    }
)

_BUILD_FLAG_OPTIONAL_ATTR_TYPES = frozenset(
    {
        "i64",
        "f64",
        "string",
        "bool",
        "enum",
        "enum_array",
        "signed_enum_set",
        "symbol",
        "symbol_array",
        "symbol_set",
        "i64_array",
        "bytes",
        "type",
        "encoding",
        "dict",
        "parameterized_array",
    }
)

_TRAILING_BUILD_FLAG_OPTIONAL_ATTR_TYPES = frozenset(
    {
        "i64_array",
        "symbol_array",
        "symbol_set",
        "bytes",
        "dict",
        "parameterized_array",
    }
)


def c_parameter_name(name: object) -> str:
    """Projects a semantic op field name to a C/C++ parameter identifier."""
    return _c_identifier(str(name))


def flatten_format(
    elements: tuple[FormatElement, ...],
) -> list[FormatElement]:
    """Recursively collects all format elements, flattening groups."""
    result: list[FormatElement] = []
    for element in elements:
        result.append(element)
        if isinstance(element, OptionalGroup | Scope | Clause):
            result.extend(flatten_format(element.elements))
    return result


def static_tied_results(op: Op) -> list[tuple[int, int]]:
    """Returns [(result_index, operand_index)] for statically-tied results.

    Statically-tied results are declared with TiedResult in the op definition,
    meaning the tie is part of the op's semantics and the builder emits it
    automatically without caller input.
    """
    ties: list[tuple[int, int]] = []
    operand_names = [o.name for o in op.operands]
    for i, result in enumerate(op.results):
        if isinstance(result, TiedResult):
            operand_index = operand_names.index(result.tied_to)
            ties.append((i, operand_index))
    return ties


def inferred_variadic_result_type_source(op: Op) -> str | None:
    """Returns the operand field defining the complete variadic result list.

    IterArgsMatchResults makes both the result count and each result type a
    function of the corresponding iter operand. Builders for that exact shape
    should preserve the schema contract instead of accepting a redundant
    parallel type array from every caller.
    """
    if len(op.results) != 1 or not op.results[0].variadic:
        return None
    result_name = op.results[0].name
    matches = [constraint for constraint in op.constraints if constraint.name == "IterArgsMatchResults" and len(constraint.args) == 2 and constraint.args[1] == result_name]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Op '{op.name}': result field '{result_name}' has multiple IterArgsMatchResults constraints")
    source_name = matches[0].args[0]
    source = next((operand for operand in op.operands if operand.name == source_name), None)
    if source is None or not source.variadic or source.optional:
        raise ValueError(f"Op '{op.name}': IterArgsMatchResults source '{source_name}' must be a required variadic operand field")
    return source_name


def fixed_result_types_fit_single_builder_type(op: Op) -> bool:
    """Returns true when fixed results can be built from one dynamic type.

    Fixed non-variadic ResultTypeList builders expose a compact C signature
    with one caller-provided result_type when at most one result type is not
    statically known from the op declaration. Remaining results must be
    concrete scalar constraints that the generated builder can synthesize.
    """
    layout = compute_layout(op)
    if layout.variadic_result or layout.fixed_result_count <= 1:
        return True
    dynamic_result_count = 0
    for result in op.results:
        if result.type_constraint in _FIXED_RESULT_TYPE_CONSTRAINTS:
            continue
        dynamic_result_count += 1
    return dynamic_result_count <= 1


def _c_attr_param_type(
    op: Op,
    attr_def: AttrDef,
    shared_enums: SharedEnumMap,
) -> str:
    """Returns the C builder parameter type for an attribute."""
    if attr_def.attr_type == "enum" and attr_def.enum_def is not None and not attr_def.optional:
        return _enum_c_type(op, attr_def, shared_enums)
    return _C_ATTR_TYPE_MAP.get(attr_def.attr_type, "loom_attribute_t")


def extract_c_params(op: Op, shared_enums: SharedEnumMap) -> list[dict[str, Any]]:
    """Extracts builder parameters for a C function from format specs.

    Mirrors the Python _extract_params but produces C-oriented descriptors.
    Each param has: name, kind, c_type, and kind-specific extras.
    """
    layout = compute_layout(op)
    inferred_result_source = inferred_variadic_result_type_source(op)
    params: list[dict[str, Any]] = []
    implicit_fields = {"iv", "args"}
    covered_attrs: set[str] = {
        boundary_attr for element in flatten_format(op.format) if isinstance(element, FuncArgs) for boundary_attr in (element.start_attr, element.end_attr) if boundary_attr is not None
    }

    def append_attr_param(name: str) -> None:
        attr_def = op.attr(name)
        if attr_def is None:
            return
        c_type = _c_attr_param_type(op, attr_def, shared_enums)
        params.append(
            {
                "name": name,
                "kind": "attr",
                "c_type": c_type,
                "attr_type": attr_def.attr_type,
                "optional": attr_def.optional,
                "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
            }
        )
        covered_attrs.add(name)

    # Collect field names that are anchors of OptionalGroups.
    optional_anchors: set[str] = set()

    def _collect_optional_anchors(elements: tuple[FormatElement, ...]) -> None:
        for element in elements:
            if isinstance(element, OptionalGroup):
                optional_anchors.add(element.anchor)
                _collect_optional_anchors(element.elements)
            elif isinstance(element, Scope | Clause):
                _collect_optional_anchors(element.elements)

    _collect_optional_anchors(op.format)

    # Ops with ResultType or ResultTypeList can have tied results, meaning
    # any operand may be consumed. Track for loom_may_consume annotation.
    has_result_type_list = any(isinstance(e, ResultType | ResultTypeList) for e in flatten_format(op.format))

    # Static ties are declared in the op definition (TiedResult). Dynamic
    # ties require explicit builder parameters.
    static_ties = static_tied_results(op)
    has_dynamic_ties = any(isinstance(e, ResultTypeList) for e in flatten_format(op.format)) and not static_ties

    # Track the most recent BindingList for association with the next Region.
    _pending_binding: dict[str, Any] | None = None
    # Track body-backed FuncArgs groups available to projected/body regions.
    _pending_func_args: list[dict[str, Any]] = []
    _body_func_args: list[dict[str, Any]] = []
    # Region fields whose explicit block args are not derivable from operands.
    explicit_block_args_by_region: dict[str, str] = {}
    func_args_field_names = c_queries.func_args_field_names(op)
    func_like_body_region_name: str | None = None
    for interface in op.interfaces:
        if isinstance(interface, FuncLikeInterface):
            func_like_body_region_name = interface.body
            break

    def _find_region_def(target_op: Op, region_name: str) -> RegionDef | None:
        for r in target_op.regions:
            if r.name == region_name:
                return r
        return None

    def walk(elements: tuple[FormatElement, ...]) -> None:
        nonlocal _pending_binding
        for element in elements:
            match element:
                case Ref(field=name):
                    if name in implicit_fields:
                        continue
                    desc = layout.fields.get(name)
                    if desc and desc.kind == FieldKind.OPERAND:
                        params.append(
                            {
                                "name": name,
                                "kind": "operand",
                                "c_type": "loom_value_id_t",
                                "index": desc.index,
                                "optional": desc.optional,
                                "may_consume": has_result_type_list,
                            }
                        )

                case Refs(field=name) | TypedRefs(field=name):
                    params.append(
                        {
                            "name": name,
                            "kind": "operand_variadic",
                            "c_type": "const loom_value_id_t*",
                            "may_consume": has_result_type_list,
                        }
                    )

                case AlignedRefs(refs=refs_field, alignments=alignments_field):
                    refs_desc = layout.fields.get(refs_field)
                    if refs_desc is None or refs_desc.kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': AlignedRefs refs field '{refs_field}' is not an operand field")
                    if not refs_desc.variadic:
                        raise ValueError(f"Op '{op.name}': AlignedRefs refs field '{refs_field}' must be variadic")
                    params.append(
                        {
                            "name": refs_field,
                            "kind": "operand_variadic",
                            "c_type": "const loom_value_id_t*",
                            "may_consume": has_result_type_list,
                        }
                    )
                    append_attr_param(alignments_field)

                case BlockRef(field=name):
                    desc = layout.fields.get(name)
                    if desc is None or desc.kind != FieldKind.SUCCESSOR:
                        kind_name = desc.kind.name if desc else "UNKNOWN"
                        raise ValueError(f"Op '{op.name}': BlockRef('{name}') references {kind_name}, expected SUCCESSOR")
                    params.append(
                        {
                            "name": name,
                            "kind": "successor",
                            "c_type": "loom_block_t*",
                            "index": desc.index,
                        }
                    )

                case OperandDict(operands=operand_field, names=names_field):
                    params.append(
                        {
                            "name": operand_field,
                            "kind": "operand_dict",
                            "c_type": "const loom_named_value_t*",
                            "operand_index": layout.fields[operand_field].index,
                            "names_attr_index": c_queries.resolve_attr_index(op, names_field, "builder"),
                            "names_field": names_field,
                            "may_consume": has_result_type_list,
                        }
                    )
                    covered_attrs.add(names_field)

                case AttrTable(keys=keys_field, values=values_field):
                    append_attr_param(keys_field)
                    values_desc = layout.fields.get(values_field)
                    if values_desc is None or values_desc.kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': AttrTable values field '{values_field}' is not an operand field")
                    if not values_desc.variadic:
                        raise ValueError(f"Op '{op.name}': AttrTable values field '{values_field}' must be variadic")
                    params.append(
                        {
                            "name": values_field,
                            "kind": "operand_variadic",
                            "c_type": "const loom_value_id_t*",
                            "may_consume": has_result_type_list,
                            "index": values_desc.index,
                        }
                    )
                    covered_attrs.add(keys_field)

                case RegionTable(keys=keys_field, case_regions=case_regions_field, default_region=default_region_field):
                    append_attr_param(keys_field)
                    case_region_def = _find_region_def(op, case_regions_field)
                    default_region_def = _find_region_def(op, default_region_field)
                    params.append(
                        {
                            "name": "region_table",
                            "kind": "auto_region_table",
                            "case_region_index": layout.fields[case_regions_field].index,
                            "default_region_index": layout.fields[default_region_field].index,
                            "case_region_name": case_regions_field,
                            "default_region_name": default_region_field,
                            "keys_field": keys_field,
                            "case_implicit_args": (case_region_def.implicit_args if case_region_def else ()),
                            "default_implicit_args": (default_region_def.implicit_args if default_region_def else ()),
                        }
                    )
                    covered_attrs.add(keys_field)

                case Attr(field=name):
                    append_attr_param(name)

                case SymbolRef(field=name):
                    attr_def = op.attr(name)
                    params.append(
                        {
                            "name": name,
                            "kind": "symbol",
                            "c_type": "loom_symbol_ref_t",
                            "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
                            "optional": (attr_def.optional if attr_def else False),
                        }
                    )
                    covered_attrs.add(name)

                case IndexList(dynamic=dynamic_field, static=static_field):
                    static_desc = layout.fields.get(static_field)
                    params.append(
                        {
                            "name": dynamic_field,
                            "kind": "index_list",
                            "dynamic_field": dynamic_field,
                            "static_field": static_field,
                            "static_attr_index": (c_queries.resolve_attr_index(op, static_field, "builder") if static_desc else 0),
                        }
                    )
                    covered_attrs.add(static_field)

                case BindingList(field=name, kind=binding_kind):
                    params.append(
                        {
                            "name": name,
                            "kind": "binding_list",
                            "c_type": "const loom_value_id_t*",
                            "binding_kind": binding_kind,
                            "may_consume": has_result_type_list,
                        }
                    )
                    # Track as pending binding for the next Region.
                    nonlocal _pending_binding
                    _pending_binding = {
                        "name": name,
                        "binding_kind": binding_kind,
                    }

                case BlockArgs(region=name):
                    region_def = _find_region_def(op, name)
                    has_derived_args = bool(region_def is not None and (region_def.arg_source or region_def.implicit_args))
                    if not has_derived_args:
                        param_name = f"{name}_arg_types"
                        params.append(
                            {
                                "name": param_name,
                                "kind": "block_args",
                                "region": name,
                            }
                        )
                        explicit_block_args_by_region[name] = param_name

                case ResultType(field=name):
                    # When ResultType references a variadic result, the
                    # builder needs result_types/result_count (same as
                    # ResultTypeList) since the op can have multiple results.
                    field_desc = layout.fields.get(name)
                    is_variadic = field_desc and field_desc.kind == FieldKind.RESULT and layout.variadic_result
                    if is_variadic:
                        params.append(
                            {
                                "name": "result_types",
                                "kind": "result_types",
                            }
                        )
                    else:
                        params.append(
                            {
                                "name": "result_type",
                                "kind": "result_type",
                            }
                        )

                case ResultTypeList(field=name):
                    if inferred_result_source is None:
                        params.append(
                            {
                                "name": "result_types",
                                "kind": "result_types",
                            }
                        )
                    if has_dynamic_ties:
                        params.append(
                            {
                                "name": "tied_results",
                                "kind": "tied_results",
                            }
                        )

                case RegionFmt(field=name):
                    region_def = _find_region_def(op, name)
                    binding = _pending_binding
                    _pending_binding = None
                    arg_source = region_def.arg_source if region_def else None
                    if arg_source in func_args_field_names or name == func_like_body_region_name:
                        func_args = tuple(_body_func_args)
                    else:
                        func_args = tuple(_pending_func_args)
                    _pending_func_args.clear()
                    if arg_source in func_args_field_names:
                        arg_source = None
                    params.append(
                        {
                            "name": name,
                            "kind": "auto_region",
                            "region_index": layout.fields[name].index,
                            "optional": (region_def.optional if region_def else False),
                            "binding": binding,
                            "arg_source": arg_source,
                            "block_args": explicit_block_args_by_region.pop(name, None),
                            "implicit_args": (region_def.implicit_args if region_def else ()),
                            "func_args": func_args,
                        }
                    )

                case Flags(field=name):
                    params.append(
                        {
                            "name": "instance_flags",
                            "kind": "instance_flags",
                            "c_type": "uint8_t",
                            "attr_name": name,
                        }
                    )
                    covered_attrs.add(name)

                case TemplateParam(field=name):
                    append_attr_param(name)

                case AttrParams(field=name):
                    append_attr_param(name)

                case TemplateParamFlags(param=param_name, flags=flags_name):
                    append_attr_param(param_name)
                    params.append(
                        {
                            "name": "instance_flags",
                            "kind": "instance_flags",
                            "c_type": "uint8_t",
                            "attr_name": flags_name,
                        }
                    )
                    covered_attrs.add(flags_name)

                case OptionalGroup(elements=inner, anchor=_anchor):
                    walk(inner)

                case Scope(elements=inner):
                    walk(inner)

                case Clause(elements=inner):
                    walk(inner)

                case FuncArgs(
                    field=name,
                    group=group,
                    start_attr=start_attr,
                    end_attr=end_attr,
                ):
                    field_desc = layout.fields.get(name)
                    operand_field = bool(field_desc is not None and field_desc.kind == FieldKind.OPERAND)
                    if group is not None:
                        param_name = f"{group}_types"
                    elif len(func_args_field_names) == 1:
                        param_name = "arg_types"
                    else:
                        param_name = f"{name}_types"
                    param = {
                        "name": param_name,
                        "kind": "func_args",
                        "field": name,
                        "operand_field": operand_field,
                        "start_attr_index": c_queries.resolve_attr_index(op, start_attr, "FuncArgs"),
                        "end_attr_index": c_queries.resolve_attr_index(op, end_attr, "FuncArgs"),
                    }
                    params.append(param)
                    if not operand_field:
                        _pending_func_args.append(param)
                        _body_func_args.append(param)
                    if start_attr is not None:
                        covered_attrs.add(start_attr)
                    if end_attr is not None:
                        covered_attrs.add(end_attr)

                case PredicateList(field=name):
                    attr_def = op.attr(name)
                    if attr_def is not None:
                        params.append(
                            {
                                "name": name,
                                "kind": "predicate_list",
                                "optional": attr_def.optional,
                                "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
                            }
                        )
                        covered_attrs.add(name)

                case AttrDict(field=name):
                    if name:
                        append_attr_param(name)
                    else:
                        for attr_def in op.attrs:
                            if attr_def.attr_type == ATTR_TYPE_FLAGS:
                                continue
                            if attr_def.name in covered_attrs:
                                continue
                            append_attr_param(attr_def.name)

                case KeyRef(field=name):
                    append_attr_param(name)

                case ScopedEnumRef(field=name):
                    raise ValueError(f"Op '{op.name}': scoped enum field '{name}' requires a domain-aware handwritten C builder")

                case StableKeyRef(key=name, stable_id=stable_id):
                    attr_def = op.attr(name)
                    if attr_def is None:
                        continue
                    params.append(
                        {
                            "name": name,
                            "kind": "stable_key_ref",
                            "c_type": _c_attr_param_type(op, attr_def, shared_enums),
                            "attr_type": attr_def.attr_type,
                            "attr_index": c_queries.resolve_attr_index(op, name, "builder"),
                            "stable_id_attr_index": c_queries.resolve_attr_index(op, stable_id, "builder"),
                        }
                    )
                    covered_attrs.add(name)
                    covered_attrs.add(stable_id)

                case Keyword() | TypeOf() | TypesOf() | Glue():
                    pass

    walk(op.format)

    # Results not spelled by the assembly format normally require explicit
    # builder parameters. Exact fixed result constraints are synthesized by
    # the generated implementation instead.
    has_result_param = any(p["kind"] in ("result_type", "result_types") for p in params)
    if not has_result_param and len(op.results) > 0 and inferred_result_source is None and fixed_result_type_constraints(op) is None:
        if layout.variadic_result:
            params.append(
                {
                    "name": "result_types",
                    "kind": "result_types",
                }
            )
        else:
            params.append(
                {
                    "name": "result_types",
                    "kind": "result_types",
                }
            )

    return params


def optional_param_uses_build_flag(param: dict[str, object]) -> bool:
    """Returns true if an optional builder param needs an explicit presence bit."""
    if not param.get("optional"):
        return False
    if param["kind"] == "symbol":
        return True
    if param["kind"] == "operand":
        return True
    if param["kind"] == "auto_region":
        return True
    if param["kind"] == "predicate_list":
        return True
    if param["kind"] != "attr":
        return False
    return param.get("attr_type") in _BUILD_FLAG_OPTIONAL_ATTR_TYPES


def build_flag_params(params: list[dict[str, object]]) -> list[dict[str, object]]:
    """Returns optional builder parameters controlled by build_flags."""
    flagged_params = [param for param in params if optional_param_uses_build_flag(param)]
    existing_params: list[dict[str, object]] = []
    aggregate_params: list[dict[str, object]] = []
    for param in flagged_params:
        is_aggregate = param["kind"] == "predicate_list" or (param["kind"] == "attr" and param.get("attr_type") in _TRAILING_BUILD_FLAG_OPTIONAL_ATTR_TYPES)
        # Newly flag-controlled aggregates trail fields with established public
        # bit ordinals so enabling explicit presence does not renumber them.
        if is_aggregate:
            aggregate_params.append(param)
        else:
            existing_params.append(param)
    return existing_params + aggregate_params


def build_flags_type_name(prefix: str) -> str:
    return f"{prefix}_build_flags_t"


def build_flags_storage_type(flag_count: int) -> str:
    """Returns the smallest public integer type that can hold all build flags."""
    if flag_count <= 32:
        return "uint32_t"
    if flag_count <= 64:
        return "uint64_t"
    raise ValueError(f"build has {flag_count} optional fields, which exceeds the 64-bit build flag capacity")


def build_flag_bit_literal(flag_count: int) -> str:
    """Returns the C integer literal used for build flag enumerators."""
    return "UINT64_C(1)" if flag_count > 32 else "1u"


def build_flag_bit_name(prefix: str, param: dict[str, object]) -> str:
    return f"{prefix.upper()}_BUILD_FLAG_HAS_{str(param['name']).upper()}"


def build_c_param_list(op: Op, params: list[dict[str, object]], layout: FieldLayout, prefix: str) -> list[str]:
    """Builds the C parameter string list from extracted params.

    Adds loom_optional annotations for optional attrs and regions.
    """
    c_params = ["loom_builder_t* builder"]
    if build_flag_params(params):
        c_params.append(f"{build_flags_type_name(prefix)} build_flags")
    for param in params:
        opt = "loom_optional " if param.get("optional") else ""
        consume = "loom_may_consume " if param.get("may_consume") else ""
        name = c_parameter_name(param["name"])
        match param["kind"]:
            case "operand":
                c_params.append(f"{opt}{consume}loom_value_id_t {name}")
            case "successor":
                c_params.append(f"loom_block_t* {name}")
            case "operand_variadic":
                c_params.append(f"{consume}const loom_value_id_t* {name}")
                c_params.append(f"iree_host_size_t {name}_count")
            case "operand_dict":
                c_params.append(f"{consume}const loom_named_value_t* {name}")
                c_params.append(f"iree_host_size_t {name}_count")
            case "attr":
                c_params.append(f"{opt}{param['c_type']} {name}")
                if param["attr_type"] == "i64_array":
                    c_params.append(f"iree_host_size_t {name}_count")
            case "stable_key_ref":
                c_params.append(f"{param['c_type']} {name}")
            case "symbol":
                c_params.append(f"{opt}loom_symbol_ref_t {name}")
            case "index_list":
                dynamic_name = c_parameter_name(param["dynamic_field"])
                static_name = c_parameter_name(param["static_field"])
                c_params.append(f"const loom_value_id_t* {dynamic_name}")
                c_params.append(f"iree_host_size_t {dynamic_name}_count")
                c_params.append(f"const int64_t* {static_name}")
                c_params.append(f"iree_host_size_t {static_name}_count")
            case "binding_list":
                c_params.append(f"{consume}const loom_value_id_t* {name}")
                c_params.append(f"iree_host_size_t {name}_count")
            case "result_type":
                c_params.append("loom_type_t result_type")
            case "result_types":
                if layout.variadic_result:
                    c_params.append("const loom_type_t* result_types")
                    c_params.append("iree_host_size_t result_count")
                elif fixed_result_types_fit_single_builder_type(op):
                    c_params.append("loom_type_t result_type")
                else:
                    c_params.append("const loom_type_t* result_types")
            case "tied_results":
                c_params.append("const loom_tied_result_t* tied_results")
                c_params.append("iree_host_size_t tied_result_count")
            case "predicate_list":
                c_params.append(f"{opt}const loom_predicate_t* {name}")
                c_params.append(f"iree_host_size_t {name}_count")
            case "instance_flags":
                c_params.append(f"uint8_t {name}")
            case "func_args":
                c_params.append(f"const loom_type_t* {name}")
                c_params.append(f"iree_host_size_t {name}_count")
            case "block_args":
                c_params.append(f"const loom_type_t* {name}")
                c_params.append(f"iree_host_size_t {name}_count")
            case "auto_region":
                pass  # Auto-created by builder, no parameter.
    c_params.append("loom_location_id_t location")
    c_params.append("loom_op_t** out_op")
    return c_params
