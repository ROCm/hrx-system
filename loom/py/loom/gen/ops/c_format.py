# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Assembly format translation for generated C op metadata."""

from __future__ import annotations

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
from loom.dsl import Op
from loom.fields import FieldKind, compute_layout
from loom.gen.assembly.tokens import KEYWORD_MAP, REGION_SYNTAX_MAP
from loom.gen.ops import c_queries


def translate_format_elements(op: Op) -> list[tuple[str, int, str]]:
    """Translates an op's format spec to C format element initializers.

    Returns a list of (kind_str, field_index, data_str) triples that
    become C struct initializers: {kind, field_index, data}.
    """
    layout = compute_layout(op)
    elements: list[tuple[str, int, str]] = []

    # Implicit fields are created by the printer/parser from context, not
    # declared as operands/results/attrs/regions.
    implicit_fields = {"iv", "args"}

    def resolve_field(name: str) -> tuple[FieldKind, int]:
        desc = layout.fields.get(name)
        if desc is None:
            raise ValueError(f"Op '{op.name}': format references undeclared field '{name}'")
        if desc.kind == FieldKind.ATTR:
            return desc.kind, c_queries.resolve_attr_index(op, name, "format")
        return desc.kind, desc.index

    def resolve_region_syntax(syntax: str) -> str:
        c_name = REGION_SYNTAX_MAP.get(syntax)
        if c_name is None:
            known = ", ".join(repr(name) for name in sorted(REGION_SYNTAX_MAP))
            raise ValueError(f"Op '{op.name}': unknown region syntax {syntax!r}. Known syntaxes: {known}")
        return c_name

    def walk(format_elements: tuple[FormatElement, ...]) -> None:
        for element in format_elements:
            match element:
                case Ref(field=name):
                    if name in implicit_fields:
                        # Implicit field (iv, etc.) is emitted as operand ref
                        # with a special index. The printer/parser handles
                        # implicit fields specially.
                        elements.append(("LOOM_FORMAT_KIND_OPERAND_REF", 0xFF, "0"))
                    else:
                        ref_kind, ref_index = resolve_field(name)
                        if ref_kind == FieldKind.OPERAND:
                            elements.append(("LOOM_FORMAT_KIND_OPERAND_REF", ref_index, "0"))
                        else:
                            raise ValueError(f"Op '{op.name}': Ref('{name}') references {ref_kind.name}, expected OPERAND")

                case Refs(field=name):
                    kind, index = resolve_field(name)
                    if kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': Refs('{name}') references {kind.name}, expected OPERAND")
                    elements.append(("LOOM_FORMAT_KIND_OPERAND_REFS", index, "0"))

                case TypedRefs(field=name):
                    kind, index = resolve_field(name)
                    if kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': TypedRefs('{name}') references {kind.name}, expected OPERAND")
                    elements.append(("LOOM_FORMAT_KIND_OPERAND_TYPED_REFS", index, "0"))

                case BlockRef(field=name):
                    kind, index = resolve_field(name)
                    if kind != FieldKind.SUCCESSOR:
                        raise ValueError(f"Op '{op.name}': BlockRef('{name}') references {kind.name}, expected SUCCESSOR")
                    elements.append(("LOOM_FORMAT_KIND_SUCCESSOR_REF", index, "0"))

                case Attr(field=name):
                    kind, index = resolve_field(name)
                    if kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': Attr('{name}') references {kind.name}, expected ATTR")
                    elements.append(("LOOM_FORMAT_KIND_ATTR_VALUE", index, "0"))

                case SymbolRef(field=name):
                    kind, index = resolve_field(name)
                    if kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': SymbolRef('{name}') references {kind.name}, expected ATTR")
                    elements.append(("LOOM_FORMAT_KIND_SYMBOL_REF", index, "0"))

                case TypeOf(field=name):
                    kind, index = resolve_field(name)
                    if kind == FieldKind.OPERAND:
                        elements.append(("LOOM_FORMAT_KIND_OPERAND_TYPE", index, "0"))
                    elif kind == FieldKind.RESULT:
                        elements.append(("LOOM_FORMAT_KIND_RESULT_TYPE", index, "0"))
                    else:
                        raise ValueError(f"Op '{op.name}': TypeOf('{name}') references {kind.name}")

                case TypesOf(field=name):
                    kind, index = resolve_field(name)
                    if kind == FieldKind.OPERAND:
                        elements.append(("LOOM_FORMAT_KIND_OPERAND_TYPES", index, "0"))
                    elif kind == FieldKind.RESULT:
                        elements.append(("LOOM_FORMAT_KIND_RESULT_TYPE_LIST", index, "0"))
                    else:
                        raise ValueError(f"Op '{op.name}': TypesOf('{name}') references {kind.name}")

                case ResultType(field=name):
                    kind, index = resolve_field(name)
                    if kind != FieldKind.RESULT:
                        raise ValueError(f"Op '{op.name}': ResultType('{name}') references {kind.name}, expected RESULT")
                    elements.append(("LOOM_FORMAT_KIND_RESULT_TYPE_SINGLE", index, "0"))

                case ResultTypeList(field=name, parens=parens):
                    kind, index = resolve_field(name)
                    if kind != FieldKind.RESULT:
                        raise ValueError(f"Op '{op.name}': ResultTypeList('{name}') references {kind.name}, expected RESULT")
                    payload = "LOOM_RESULT_TYPE_LIST_PARENS" if parens else "0"
                    elements.append(("LOOM_FORMAT_KIND_RESULT_TYPE_LIST", index, payload))

                case Clause(name=name, elements=inner):
                    kw_enum = KEYWORD_MAP.get(name)
                    if kw_enum is None:
                        raise ValueError(f"Op '{op.name}': unknown clause name '{name}'. Add it to KEYWORD_MAP and the C keyword enum.")
                    elements.append(("LOOM_FORMAT_KIND_KEYWORD", 0, kw_enum))
                    elements.append(("LOOM_FORMAT_KIND_GLUE", 0, "0"))
                    elements.append(("LOOM_FORMAT_KIND_KEYWORD", 0, KEYWORD_MAP["("]))
                    walk(inner)
                    elements.append(("LOOM_FORMAT_KIND_KEYWORD", 0, KEYWORD_MAP[")"]))

                case Keyword(text=text):
                    kw_enum = KEYWORD_MAP.get(text)
                    if kw_enum is None:
                        raise ValueError(f"Op '{op.name}': unknown keyword '{text}'. Add it to KEYWORD_MAP and the C keyword enum.")
                    elements.append(("LOOM_FORMAT_KIND_KEYWORD", 0, kw_enum))

                case AttrDict(field=field_name):
                    attr_index = 0
                    attr_dict_flags = "0"
                    if field_name:
                        non_flags = c_queries.non_flags_attrs(op)
                        for idx, non_flags_attr in enumerate(non_flags):
                            if non_flags_attr.name == field_name:
                                attr_index = idx
                                break
                    else:
                        attr_dict_flags = "LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS"
                    elements.append(("LOOM_FORMAT_KIND_ATTR_DICT", attr_index, attr_dict_flags))

                case OperandDict(operands=operand_field, names=names_field):
                    operand_kind, operand_index = resolve_field(operand_field)
                    names_kind, names_index = resolve_field(names_field)
                    if operand_kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': OperandDict operands field '{operand_field}' is not an operand field")
                    if names_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': OperandDict names field '{names_field}' is not an attr field")
                    operand_desc = layout.fields[operand_field]
                    if not operand_desc.variadic:
                        raise ValueError(f"Op '{op.name}': OperandDict operands field '{operand_field}' must be variadic")
                    names_attr = op.attr(names_field)
                    if names_attr is None or names_attr.attr_type != "dict":
                        raise ValueError(f"Op '{op.name}': OperandDict names field '{names_field}' must be a dict attr")
                    elements.append(("LOOM_FORMAT_KIND_OPERAND_DICT", operand_index, str(names_index)))

                case AttrTable(keys=keys_field, values=values_field):
                    values_kind, values_index = resolve_field(values_field)
                    keys_kind, keys_index = resolve_field(keys_field)
                    if values_kind != FieldKind.OPERAND:
                        raise ValueError(f"Op '{op.name}': AttrTable values field '{values_field}' is not an operand field")
                    if keys_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': AttrTable keys field '{keys_field}' is not an attr field")
                    values_desc = layout.fields[values_field]
                    if not values_desc.variadic:
                        raise ValueError(f"Op '{op.name}': AttrTable values field '{values_field}' must be variadic")
                    keys_attr = op.attr(keys_field)
                    if keys_attr is None or keys_attr.attr_type != "i64_array":
                        raise ValueError(f"Op '{op.name}': AttrTable keys field '{keys_field}' must be an i64_array attr")
                    elements.append(("LOOM_FORMAT_KIND_ATTR_TABLE", values_index, str(keys_index)))

                case RegionTable(keys=keys_field, case_regions=case_regions_field, default_region=default_region_field):
                    case_kind, case_index = resolve_field(case_regions_field)
                    default_kind, default_index = resolve_field(default_region_field)
                    keys_kind, keys_index = resolve_field(keys_field)
                    if case_kind != FieldKind.REGION:
                        raise ValueError(f"Op '{op.name}': RegionTable case field '{case_regions_field}' is not a region field")
                    if default_kind != FieldKind.REGION:
                        raise ValueError(f"Op '{op.name}': RegionTable default field '{default_region_field}' is not a region field")
                    if keys_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': RegionTable keys field '{keys_field}' is not an attr field")
                    case_desc = layout.fields[case_regions_field]
                    if not case_desc.variadic:
                        raise ValueError(f"Op '{op.name}': RegionTable case field '{case_regions_field}' must be variadic")
                    keys_attr = op.attr(keys_field)
                    if keys_attr is None or keys_attr.attr_type != "i64_array":
                        raise ValueError(f"Op '{op.name}': RegionTable keys field '{keys_field}' must be an i64_array attr")
                    payload = f"LOOM_FORMAT_REGION_TABLE_DATA({keys_index}, {default_index})"
                    elements.append(("LOOM_FORMAT_KIND_REGION_TABLE", case_index, payload))

                case RegionFmt(field=name, syntax=syntax):
                    kind, index = resolve_field(name)
                    if layout.fields[name].variadic:
                        raise ValueError(f"Op '{op.name}': variadic Region '{name}' must use RegionTable or another table format")
                    elements.append(("LOOM_FORMAT_KIND_REGION", index, resolve_region_syntax(syntax)))

                case IndexList(dynamic=dynamic_field, static=static_field, glue=glue):
                    _dyn_kind, dyn_index = resolve_field(dynamic_field)
                    _sta_kind, sta_index = resolve_field(static_field)
                    if sta_index > 0x7FFF:
                        raise ValueError(f"Op '{op.name}': IndexList static attr index {sta_index} exceeds packed format limit")
                    payload = str(sta_index) if glue else f"LOOM_FORMAT_INDEX_LIST_DATA({sta_index}, false)"
                    elements.append(("LOOM_FORMAT_KIND_INDEX_LIST", dyn_index, payload))

                case BindingList(field=name, kind=binding_kind):
                    _field_kind, index = resolve_field(name)
                    binding_kind_name = "LOOM_BINDING_ELEMENT" if binding_kind == "element" else "LOOM_BINDING_CAPTURE"
                    elements.append(("LOOM_FORMAT_KIND_BINDING_LIST", index, binding_kind_name))

                case BlockArgs(region=name):
                    kind, index = resolve_field(name)
                    if kind != FieldKind.REGION:
                        raise ValueError(f"Op '{op.name}': BlockArgs region field '{name}' is not a region field")
                    elements.append(("LOOM_FORMAT_KIND_BLOCK_ARGS", index, "0"))

                case FuncArgs(field=name):
                    elements.append(("LOOM_FORMAT_KIND_FUNC_ARGS", 0, "0"))

                case PredicateList(field=name):
                    _field_kind, index = resolve_field(name)
                    elements.append(("LOOM_FORMAT_KIND_PREDICATE_LIST", index, "0"))

                case OptionalGroup(elements=inner, anchor=anchor):
                    # Resolve anchor to determine category.
                    anchor_desc = layout.fields.get(anchor)
                    if anchor_desc is None:
                        # Implicit field (predicates, etc.) with no backing
                        # storage on the op. Use LOOM_ANCHOR_ATTR with an index
                        # beyond any real attr so the printer's bounds check
                        # evaluates to "not present."
                        anchor_category = 1
                        anchor_index = 0xFF
                    else:
                        anchor_index = anchor_desc.index
                        if anchor_desc.kind == FieldKind.OPERAND:
                            anchor_category = 0
                        elif anchor_desc.kind == FieldKind.ATTR:
                            anchor_category = 1
                        elif anchor_desc.kind == FieldKind.REGION:
                            anchor_category = 2
                        elif anchor_desc.kind == FieldKind.RESULT:
                            anchor_category = 3
                        else:
                            anchor_category = 1

                    inner_start = len(elements)
                    elements.append(("LOOM_FORMAT_KIND_OPTIONAL_GROUP", 0, "0"))
                    walk(inner)
                    inner_count = len(elements) - inner_start - 1
                    data = f"({inner_count} << 2) | {anchor_category}"
                    elements[inner_start] = (
                        "LOOM_FORMAT_KIND_OPTIONAL_GROUP",
                        anchor_index,
                        data,
                    )

                case Scope(elements=inner):
                    inner_start = len(elements)
                    elements.append(("LOOM_FORMAT_KIND_SCOPE", 0, "0"))
                    walk(inner)
                    inner_count = len(elements) - inner_start - 1
                    elements[inner_start] = (
                        "LOOM_FORMAT_KIND_SCOPE",
                        0,
                        str(inner_count),
                    )

                case Flags(field=name):
                    elements.append(("LOOM_FORMAT_KIND_FLAGS", 0, "0"))

                case OpRef(field=name):
                    kind, index = resolve_field(name)
                    elements.append(("LOOM_FORMAT_KIND_OP_REF", index, "0"))

                case DescriptorRef(key=key_name, ordinal=ordinal_name):
                    key_kind, key_index = resolve_field(key_name)
                    ordinal_kind, ordinal_index = resolve_field(ordinal_name)
                    key_attr = op.attr(key_name)
                    ordinal_attr = op.attr(ordinal_name)
                    if key_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': DescriptorRef key field '{key_name}' is not an attr field")
                    if key_attr is None or key_attr.attr_type != "string":
                        raise ValueError(f"Op '{op.name}': DescriptorRef key field '{key_name}' must be a string attr")
                    if ordinal_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': DescriptorRef ordinal field '{ordinal_name}' is not an attr field")
                    if ordinal_attr is None or ordinal_attr.attr_type != "i64":
                        raise ValueError(f"Op '{op.name}': DescriptorRef ordinal field '{ordinal_name}' must be an i64 attr")
                    elements.append(
                        (
                            "LOOM_FORMAT_KIND_DESCRIPTOR_REF",
                            key_index,
                            str(ordinal_index),
                        )
                    )

                case StableKeyRef(key=key_name, stable_id=stable_id_name):
                    key_kind, key_index = resolve_field(key_name)
                    stable_id_kind, stable_id_index = resolve_field(stable_id_name)
                    key_attr = op.attr(key_name)
                    stable_id_attr = op.attr(stable_id_name)
                    if key_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': StableKeyRef key field '{key_name}' is not an attr field")
                    if key_attr is None or key_attr.attr_type != "string":
                        raise ValueError(f"Op '{op.name}': StableKeyRef key field '{key_name}' must be a string attr")
                    if stable_id_kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': StableKeyRef stable ID field '{stable_id_name}' is not an attr field")
                    if stable_id_attr is None or stable_id_attr.attr_type != "i64":
                        raise ValueError(f"Op '{op.name}': StableKeyRef stable ID field '{stable_id_name}' must be an i64 attr")
                    elements.append(
                        (
                            "LOOM_FORMAT_KIND_STABLE_KEY_REF",
                            key_index,
                            str(stable_id_index),
                        )
                    )

                case TemplateParam(field=name):
                    kind, index = resolve_field(name)
                    elements.append(("LOOM_FORMAT_KIND_TEMPLATE_PARAM", index, "0"))

                case TemplateParamFlags(param=param_name, flags=flags_name):
                    kind, index = resolve_field(param_name)
                    flags_attr = op.attr(flags_name)
                    if kind != FieldKind.ATTR:
                        raise ValueError(f"Op '{op.name}': TemplateParamFlags parameter field '{param_name}' is not an attr field")
                    if flags_attr is None or flags_attr.attr_type != "flags":
                        raise ValueError(f"Op '{op.name}': TemplateParamFlags flags field '{flags_name}' must be a flags attr")
                    elements.append(("LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS", index, "0"))

                case Glue():
                    elements.append(("LOOM_FORMAT_KIND_GLUE", 0, "0"))

    walk(op.format)
    return elements
