# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Format-driven text printer for loom IR.

Walks an op's assembly format element list and emits text tokens.
Each format element type has a handler that reads data from the
resolved fields and appends tokens to the output.

Canonical formatting rules:
  - One op per line (no line-length wrapping).
  - Regions: '{' at end of op line, body ops indented 2 spaces,
    '}' on its own line. Region separators at outer indentation.
  - Single space between tokens, suppressed before , ) ] } [
    and after ( [ {.
  - Result types always use parens: -> (type).

Value naming:
  - Value.name stores the bare name without sigil ("x", not "%x").
  - The printer adds the '%' sigil when emitting: "x" -> %x.
  - Unnamed values (name == "") get auto-names using their value ID: %0, %1, %2.
  - Digit-only names and identifier names occupy separate syntactic
    namespaces, so no collision avoidance is needed.
  - Stable: same IR always produces identical output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from loom.assembly import (
    AlignedRefs,
    Attr,
    AttrDict,
    AttrParams,
    AttrTable,
    BindingList,
    BlockArgs,
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
    Param,
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
from loom.assembly import (
    Region as RegionFmt,
)
from loom.dsl import (
    AttrDef,
    EncodingAliasDef,
    EncodingFamilyDef,
    Op,
    ParameterizedAttrDef,
    TypeDef,
)
from loom.fields import (
    FieldLayout,
    FormatFields,
    ResolvedFields,
    compute_layout,
    resolve_fields,
)
from loom.ir import (
    Block,
    BufferType,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingType,
    EnumArrayAttr,
    FunctionType,
    Module,
    NoneType,
    Operation,
    ParameterizedAttr,
    ParameterizedAttrArray,
    ParameterizedType,
    PoolType,
    Predicate,
    PredicateArg,
    Region,
    RegisterType,
    ScalarType,
    ShapedType,
    StaticDim,
    StorageType,
    SymbolName,
    Type,
    Value,
)
from loom.location_tag import builtin_location_tag_name

_IR_TYPE_CLASSES = (
    ScalarType,
    ShapedType,
    BufferType,
    FunctionType,
    RegisterType,
    StorageType,
    DialectType,
    ParameterizedType,
    EncodingType,
    PoolType,
    NoneType,
)


@dataclass(frozen=True, slots=True)
class _EncodingAliasSelector:
    """Canonical source-name lookup state for one structural family."""

    # Fixed enum parameter selecting an alias row.
    discriminator_name: str
    # Aliases indexed by the discriminator parameter value.
    aliases_by_discriminator: Mapping[Any, EncodingAliasDef]
    # Static family parameter descriptors indexed by source name.
    parameters_by_name: Mapping[str, AttrDef]


_EncodingAliasSelectors = Mapping[str, _EncodingAliasSelector]

__all__ = [
    "Printer",
    "print_type",
]


# ============================================================================
# Type printing
# ============================================================================


class TypePrintContext:
    """Context for printing types with dim names and encodings.

    When printing a type that belongs to a specific Value, the context
    provides dim name resolution (DynamicDim -> [%M]) and encoding
    resolution (encoding_instance -> #enc or
    #encoding.operand<element_format=i8, payload_elements=32,
    payload_packing=dense_lanes>).

    Without context, dynamic dims print as '?' and encodings are omitted.
    """

    __slots__ = (
        "_dim_bindings",
        "_encoding_alias_selectors",
        "_encoding_binding",
        "_module",
        "_use_aliases",
    )

    def __init__(
        self,
        dim_bindings: dict[int, int],
        module: Module,
        encoding_binding: int = -1,
        use_aliases: bool = True,
        encoding_alias_selectors: _EncodingAliasSelectors | None = None,
    ) -> None:
        self._dim_bindings = dim_bindings
        self._encoding_alias_selectors = encoding_alias_selectors
        self._encoding_binding = encoding_binding
        self._module = module
        self._use_aliases = use_aliases

    def dim_name(self, position: int) -> str | None:
        """Get the name for a dynamic dim at the given position.

        Returns %name if a binding is present, or None.
        """
        value_id = self._dim_bindings.get(position)
        if value_id is None:
            return None
        return resolve_value_name(self._module, value_id)

    def encoding_binding_name(self) -> str | None:
        """Get the SSA name for a dynamic encoding binding.

        Returns %name when the value has a DynamicEncoding and a valid
        encoding_binding. Returns None if no binding is set.
        """
        if self._encoding_binding < 0:
            return None
        return resolve_value_name(self._module, self._encoding_binding)


def _select_canonical_encoding_alias(
    encoding: EncodingInstance,
    selectors: _EncodingAliasSelectors | None,
) -> EncodingAliasDef | None:
    if selectors is None:
        return None
    selector = selectors.get(encoding.name)
    if selector is None:
        return None
    parameters = dict(encoding.params)
    alias = selector.aliases_by_discriminator.get(
        parameters.get(selector.discriminator_name)
    )
    if alias is None:
        return None

    # Re-parsing the alias must reproduce the complete structural encoding.
    # Every contributed parameter must be present and fixed values must match.
    for parameter_name, parameter_value in alias.parameters:
        if parameter_name not in parameters:
            return None
        if (parameter_name, parameter_value) in alias.fixed_parameters and parameters[
            parameter_name
        ] != parameter_value:
            return None
    return alias


def _format_encoding_instance(
    encoding: EncodingInstance,
    *,
    use_alias: bool,
    encoding_alias_selectors: _EncodingAliasSelectors | None = None,
) -> str:
    """Format a static encoding as #alias or #family<name = value, ...>."""
    if use_alias and encoding.alias:
        return f"#{encoding.alias}"

    canonical_alias = (
        _select_canonical_encoding_alias(encoding, encoding_alias_selectors)
        if use_alias
        else None
    )
    name = canonical_alias.name if canonical_alias is not None else encoding.name
    parameters = encoding.params
    if canonical_alias is not None:
        alias_parameters = dict(canonical_alias.parameters)
        fixed_parameter_names = {
            parameter_name for parameter_name, _ in canonical_alias.fixed_parameters
        }
        parameters = tuple(
            (parameter_name, parameter_value)
            for parameter_name, parameter_value in parameters
            if parameter_name not in fixed_parameter_names
            and alias_parameters.get(parameter_name) != parameter_value
        )
    if not parameters:
        return f"#{name}"
    selector = (
        encoding_alias_selectors.get(encoding.name)
        if encoding_alias_selectors is not None
        else None
    )
    parameter_descriptors = selector.parameters_by_name if selector is not None else {}
    param_strs = []
    for parameter_name, value in parameters:
        formatted_value = _format_attr_value(
            value,
            parameter_descriptors.get(parameter_name),
            encoding_alias_selectors=encoding_alias_selectors,
            use_encoding_aliases=use_alias,
        )
        param_strs.append(f"{parameter_name}={formatted_value}")
    return f"#{name}<{', '.join(param_strs)}>"


def print_type(
    ir_type: Type,
    context: TypePrintContext | None = None,
    type_registry: dict[Any, Any] | None = None,
) -> str:
    """Print a loom type in canonical text form.

    With context: dynamic dims print as [%name], encodings print as #alias.
    Without context: dynamic dims print as ?, encodings omitted.
    With type_registry: DialectType printing walks the TypeDef format spec.
    Without type_registry: DialectType uses comma-separated params fallback.
    """
    compact_type_def = _compact_type_definition(ir_type, type_registry)
    if compact_type_def is not None:
        return _print_descriptor_backed_type(
            ir_type, compact_type_def, context, type_registry
        )

    match ir_type:
        case ScalarType():
            return repr(ir_type)
        case ShapedType():
            return _print_shaped_type(
                ir_type, _compact_shape_type_definition(ir_type), context
            )
        case BufferType():
            return "buffer"
        case PoolType():
            return _print_pool_type(
                ir_type, _compact_shape_type_definition(ir_type), context
            )
        case FunctionType(arg_types=args, result_types=results):
            arg_strs = ", ".join(print_type(t, context, type_registry) for t in args)
            result_strs = ", ".join(
                print_type(t, context, type_registry) for t in results
            )
            return f"({arg_strs}) -> ({result_strs})"
        case RegisterType(
            name=name,
            descriptor_set_stable_id=descriptor_set_stable_id,
            register_class_id=register_class_id,
            unit_count=unit_count,
            value_type=value_type,
        ):
            reg_class = (
                name
                if name is not None
                else f"0x{descriptor_set_stable_id:x}:{register_class_id}"
            )
            unit_suffix = "" if unit_count == 1 else f" x{unit_count}"
            value_suffix = (
                ""
                if value_type is None
                else f" : {print_type(value_type, context, type_registry)}"
            )
            return f"reg<{reg_class}{unit_suffix}{value_suffix}>"
        case NoneType():
            return "none"
        case ParameterizedType():
            return _print_descriptor_backed_type(
                ir_type, ir_type.definition, context, type_registry
            )
        case DialectType(name=_name, params=_params):
            return _print_dialect_type(ir_type, context, type_registry)
    raise ValueError(f"Unknown type: {ir_type}")


def _compact_type_definition(
    ir_type: Type, type_registry: dict[Any, Any] | None
) -> TypeDef | None:
    """Resolves a compact Python value class to its registered declaration."""

    if type_registry is not None:
        type_def = type_registry.get(type(ir_type))
        if isinstance(type_def, TypeDef) and type_def.uses_inline_enum_parameter:
            return type_def

    from loom.builtin_types import BUILTIN_TYPE_BY_PYTHON_TYPE

    return BUILTIN_TYPE_BY_PYTHON_TYPE.get(type(ir_type))


def _compact_shape_type_definition(ir_type: ShapedType | PoolType) -> TypeDef:
    """Resolves a compact shape representation to its declaration."""

    from loom.builtin_types import BUILTIN_COMPACT_SHAPE_TYPE_BY_KIND

    return BUILTIN_COMPACT_SHAPE_TYPE_BY_KIND[ir_type.type_kind]


def _print_dialect_type(
    dialect_type: DialectType,
    context: TypePrintContext | None,
    type_registry: dict[Any, Any] | None,
) -> str:
    """Print a dialect type, walking TypeDef format if available."""
    from loom.assembly import (
        Attr as AsmAttr,
    )
    from loom.assembly import (
        Clause as AsmClause,
    )
    from loom.assembly import (
        Glue as AsmGlue,
    )
    from loom.assembly import (
        Keyword as AsmKeyword,
    )
    from loom.assembly import (
        OptionalGroup as AsmOptionalGroup,
    )
    from loom.assembly import (
        TypeOf as AsmTypeOf,
    )

    name: str = dialect_type.name
    params = dialect_type.params

    if not params:
        return name

    # Try to use TypeDef format spec for printing.
    type_def = type_registry.get(name) if type_registry else None
    if type_def is None or not type_def.format:
        # Fallback: comma-separated params.
        param_strs = ", ".join(print_type(p, context, type_registry) for p in params)
        return f"{name}<{param_strs}>"

    # Walk the TypeDef format to produce the interior.
    parts: list[str] = []
    param_index = 0

    def walk_type_format(elements: tuple[Any, ...]) -> None:
        nonlocal param_index, parts
        for element in elements:
            match element:
                case AsmTypeOf():
                    if param_index < len(params):
                        parts.append(
                            print_type(params[param_index], context, type_registry)
                        )
                        param_index += 1
                case AsmAttr(field=_field_name):
                    if param_index < len(params):
                        p = params[param_index]
                        if isinstance(p, DialectType):
                            parts.append(p.name)
                        else:
                            parts.append(str(p))
                        param_index += 1
                case AsmKeyword(text=text):
                    parts.append(text)
                case AsmClause(name=name, elements=inner):
                    outer_parts = parts
                    parts = []
                    walk_type_format(inner)
                    inner_text = " ".join(parts)
                    parts = outer_parts
                    parts.append(f"{name}({inner_text})")
                case AsmOptionalGroup(elements=inner):
                    if param_index < len(params):
                        walk_type_format(inner)
                case AsmGlue():
                    pass

    walk_type_format(type_def.format)
    interior = " ".join(parts)
    return f"{name}<{interior}>"


def _print_descriptor_backed_type(
    value: Type,
    type_def: TypeDef,
    context: TypePrintContext | None,
    type_registry: dict[Any, Any] | None,
) -> str:
    """Prints an indirect or compact type through its declarative format."""
    stream = TokenStream()

    def has(parameter: AttrDef) -> bool:
        if isinstance(value, ParameterizedType):
            return value.has(parameter.name)
        parameter_value = getattr(value, parameter.name)
        return not (parameter.optional and int(parameter_value) == 0)

    def get(parameter: AttrDef) -> Any:
        if isinstance(value, ParameterizedType):
            return value.get(parameter.name)
        parameter_value = getattr(value, parameter.name)
        return (
            int(parameter_value) if parameter.attr_type == "enum" else parameter_value
        )

    if type_def.omits_empty_parameter_list and all(
        isinstance(parameter, AttrDef) and not has(parameter)
        for parameter in type_def.params
    ):
        return type_def.name

    def walk(elements: tuple[FormatElement, ...]) -> None:
        for element in elements:
            match element:
                case Param(field=field):
                    parameter = type_def.param(field)
                    assert isinstance(parameter, AttrDef)
                    parameter_value = get(parameter)
                    stream.emit(
                        _format_attr_value(
                            parameter_value,
                            parameter,
                            type_context=context,
                            type_registry=type_registry,
                        )
                    )
                case Keyword(text=text):
                    stream.emit(text)
                case Clause(name=name, elements=inner):
                    stream.emit(name)
                    stream.emit("(")
                    walk(inner)
                    stream.emit(")")
                case OptionalGroup(elements=inner, anchor=anchor):
                    parameter = type_def.param(anchor)
                    assert isinstance(parameter, AttrDef)
                    if has(parameter):
                        walk(inner)
                case Glue():
                    stream.set_glue()
                case _:
                    raise ValueError(
                        f"unsupported parameterized type format element: {element!r}"
                    )

    walk(type_def.format)
    return f"{type_def.name}<{stream.join()}>"


def _print_shaped_type(
    shaped: ShapedType,
    type_def: TypeDef,
    context: TypePrintContext | None = None,
) -> str:
    """Print shaped types with optional dim names and encodings/layouts."""
    if shaped.rank == 0:
        inner = repr(shaped.element_type)
    else:
        dim_parts: list[str] = []
        for i, dim in enumerate(shaped.dims):
            match dim:
                case StaticDim(size=size):
                    dim_parts.append(str(size))
                case DynamicDim():
                    dim_name = context.dim_name(i) if context else None
                    if dim_name is not None:
                        dim_parts.append(f"[{dim_name}]")
                    else:
                        dim_parts.append("?")
        inner = "x".join(dim_parts) + "x" + repr(shaped.element_type)

    # Encoding suffix.
    if shaped.has_encoding:
        enc = shaped.encoding
        if isinstance(enc, DynamicEncoding):
            # SSA encoding — resolve through context.
            enc_name = context.encoding_binding_name() if context else None
            if enc_name is not None:
                inner += f", {enc_name}"
            else:
                inner += ", ?"
        elif isinstance(enc, EncodingInstance):
            use_aliases = context._use_aliases if context else True
            inner += ", " + _format_encoding_instance(
                enc,
                use_alias=use_aliases,
                encoding_alias_selectors=(
                    context._encoding_alias_selectors if context else None
                ),
            )

    return f"{type_def.name}<{inner}>"


def _print_pool_type(
    pool: PoolType,
    type_def: TypeDef,
    context: TypePrintContext | None = None,
) -> str:
    """Print pool<[%block_size]> or pool<N>."""
    match pool.block_size:
        case StaticDim(size=size):
            return f"{type_def.name}<{size}>"
        case DynamicDim():
            dim_name = context.dim_name(0) if context else None
            if dim_name is not None:
                return f"{type_def.name}<[{dim_name}]>"
            return f"{type_def.name}<?>"
        case _:
            raise TypeError(f"unexpected dim type: {type(pool.block_size)}")


# ============================================================================
# Token stream
# ============================================================================

# Punctuation that always backward-glues (no space before them).
_BACKWARD_GLUE = frozenset(",)]}")
# Punctuation that forward-glues (no space after them).
# When a token ends with one of these, the next token glues.
_FORWARD_GLUE = frozenset("([{")


class TokenStream:
    """Collects tokens with explicit spacing control.

    Spacing model:
      - Default: space before each token.
      - Backward-glue punctuation (, ) ] }): always suppress space before.
      - Explicit Glue element: marks the next token to suppress space.
      - Composite elements with built-in glue (BindingList, BlockArgs,
        FuncArgs, and non-leading IndexList when requested): their output is
        emitted with glue=True.

    The token joiner is trivial: check the glue flag, emit space or not.
    No character-level heuristics.
    """

    __slots__ = ("_parts", "_glue_next")

    def __init__(self) -> None:
        self._parts: list[tuple[str, bool]] = []  # (text, glue_before)
        self._glue_next: bool = False

    def emit(self, text: str, glue: bool = False) -> None:
        """Append a token.

        glue=True suppresses the space before this token.
        Also suppressed if the token starts with backward-glue
        punctuation (, ) ] }), if the previous token ended with
        forward-glue punctuation ( [ {, or if a prior set_glue()
        call is pending.
        """
        if not text:
            return
        # Check if previous token ends with forward-glue punctuation.
        prev_forward = self._parts and self._parts[-1][0][-1] in _FORWARD_GLUE
        actual_glue = (
            glue or self._glue_next or prev_forward or text[0] in _BACKWARD_GLUE
        )
        self._parts.append((text, actual_glue))
        self._glue_next = False

    def set_glue(self) -> None:
        """Mark the next emitted token to glue (no space before it)."""
        self._glue_next = True

    def join(self) -> str:
        """Join all tokens into a string with resolved spacing."""
        if not self._parts:
            return ""
        result: list[str] = [self._parts[0][0]]
        for text, glue_before in self._parts[1:]:
            if not glue_before:
                result.append(" ")
            result.append(text)
        return "".join(result)


# ============================================================================
# Value naming
# ============================================================================


def resolve_value_name(module: Module, value_id: int) -> str:
    """Returns the SSA name for a value with '%' sigil.

    User-assigned names are returned with % prefix added.
    Unnamed values get %N where N is the value ID. These occupy
    separate syntactic namespaces (identifiers vs digit-only), so
    no collision is possible.
    """
    if value_id < len(module.values):
        name: str = module.values[value_id].name
        if name:
            return "%" + name
    return f"%{value_id}"


# ============================================================================
# Attribute formatting
# ============================================================================


def _format_string_literal(value: str) -> str:
    """Format a decoded string payload as one canonical JSON string literal."""
    escaped_chunks: list[str] = ['"']
    for character in value:
        codepoint = ord(character)
        if character == '"':
            escaped_chunks.append('\\"')
        elif character == "\\":
            escaped_chunks.append("\\\\")
        elif character == "\b":
            escaped_chunks.append("\\b")
        elif character == "\f":
            escaped_chunks.append("\\f")
        elif character == "\n":
            escaped_chunks.append("\\n")
        elif character == "\r":
            escaped_chunks.append("\\r")
        elif character == "\t":
            escaped_chunks.append("\\t")
        elif codepoint < 0x20:
            escaped_chunks.append(f"\\u{codepoint:04X}")
        elif 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError(f"invalid surrogate codepoint U+{codepoint:04X}")
        else:
            escaped_chunks.append(character)
    escaped_chunks.append('"')
    return "".join(escaped_chunks)


def _format_attr_value(
    value: Any,
    attr_def: AttrDef | None = None,
    *,
    type_context: TypePrintContext | None = None,
    type_registry: dict[Any, Any] | None = None,
    encoding_alias_selectors: _EncodingAliasSelectors | None = None,
    use_encoding_aliases: bool = True,
) -> str:
    """Format an attribute value for text output.

    Enum attributes print as bare keywords (lt, not "lt").
    String attributes print quoted. Everything else prints as literals.
    """
    if encoding_alias_selectors is None and type_context is not None:
        encoding_alias_selectors = type_context._encoding_alias_selectors
    if type_context is not None:
        use_encoding_aliases = type_context._use_aliases
    if attr_def is not None and attr_def.attr_type == "enum":
        return _format_enum_attr_value(value, attr_def)
    if attr_def is not None and attr_def.attr_type == "enum_array":
        if not isinstance(value, EnumArrayAttr):
            raise TypeError(
                f"enum-array attribute value must be EnumArrayAttr: {value!r}"
            )
        assert attr_def.enum_def is not None
        keyword_by_value = {
            case.value: case.keyword for case in attr_def.enum_def.cases
        }
        return (
            "["
            + ", ".join(
                _format_enum_attr_value(element, attr_def, keyword_by_value)
                for element in value
            )
            + "]"
        )
    if attr_def is not None and attr_def.attr_type == "symbol":
        if not isinstance(value, str | SymbolName):
            raise TypeError(f"symbol attribute value must be a symbol: {value!r}")
        symbol_name = str(value)
        if symbol_name.startswith("@"):
            raise ValueError(f"symbol attribute value must not include '@': {value!r}")
        return "@" + symbol_name
    if attr_def is not None and attr_def.attr_type == "type":
        if not isinstance(value, _IR_TYPE_CLASSES):
            raise TypeError(f"type attribute value must be a Type: {value!r}")
        return print_type(cast(Type, value), type_context, type_registry)
    if attr_def is not None and attr_def.attr_type == "bytes":
        if not isinstance(value, bytes | bytearray):
            raise TypeError(f"bytes attribute value must be bytes: {value!r}")
        return f'bytes("{bytes(value).hex()}")'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, SymbolName):
        return "@" + str(value)
    if isinstance(value, bytes | bytearray):
        return f'bytes("{bytes(value).hex()}")'
    if isinstance(value, EnumArrayAttr):
        raise ValueError("enum arrays require a descriptor-backed field")
    if isinstance(value, ParameterizedAttr):
        if attr_def is not None:
            if attr_def.attr_type != "parameterized":
                raise TypeError(
                    "parameterized attribute value does not match field "
                    f"kind {attr_def.attr_type!r}"
                )
            expected_family = attr_def.parameterized_attr
            if (
                expected_family is not None
                and value.family_name != expected_family.name
            ):
                raise ValueError(
                    f"parameterized attribute family '{value.family_name}' "
                    f"does not match field contract '{expected_family.name}'"
                )
        parameters = _format_parameterized_attr_parameters(
            value,
            value.definition,
            type_context=type_context,
            type_registry=type_registry,
            encoding_alias_selectors=encoding_alias_selectors,
            use_encoding_aliases=use_encoding_aliases,
        )
        return f"#{value.family_name}<{parameters}>"
    if isinstance(value, ParameterizedAttrArray):
        if attr_def is None or attr_def.attr_type != "parameterized_array":
            raise ValueError(
                "parameterized attribute arrays require a descriptor-backed field"
            )
        element_descriptor = AttrDef(
            attr_def.name,
            "parameterized",
            parameterized_attr=attr_def.parameterized_attr,
        )
        return (
            "["
            + ", ".join(
                _format_attr_value(
                    element,
                    element_descriptor,
                    type_context=type_context,
                    type_registry=type_registry,
                    encoding_alias_selectors=encoding_alias_selectors,
                    use_encoding_aliases=use_encoding_aliases,
                )
                for element in value
            )
            + "]"
        )
    if isinstance(value, str):
        return _format_string_literal(value)
    if isinstance(value, list | tuple):
        parts = [
            _format_attr_value(
                v,
                type_context=type_context,
                type_registry=type_registry,
                encoding_alias_selectors=encoding_alias_selectors,
                use_encoding_aliases=use_encoding_aliases,
            )
            for v in value
        ]
        return "[" + ", ".join(parts) + "]"
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            formatted_item = _format_attr_value(
                item,
                type_context=type_context,
                type_registry=type_registry,
                encoding_alias_selectors=encoding_alias_selectors,
                use_encoding_aliases=use_encoding_aliases,
            )
            parts.append(f"{key} = {formatted_item}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, EncodingInstance):
        return _format_encoding_instance(
            value,
            use_alias=use_encoding_aliases,
            encoding_alias_selectors=encoding_alias_selectors,
        )
    return str(value)


def _format_parameterized_attr_parameters(
    value: Any,
    definition: ParameterizedAttrDef,
    *,
    type_context: TypePrintContext | None = None,
    type_registry: dict[Any, Any] | None = None,
    encoding_alias_selectors: _EncodingAliasSelectors | None = None,
    use_encoding_aliases: bool = True,
) -> str:
    """Formats one known-family parameterized attribute payload."""
    if not isinstance(value, ParameterizedAttr):
        raise TypeError(
            f"expected parameterized attribute '{definition.name}', got {value!r}"
        )
    if value.family_name != definition.name:
        raise ValueError(
            f"parameterized attribute family '{value.family_name}' does not "
            f"match field contract '{definition.name}'"
        )

    parameters = []
    slots = value.slots
    primary_parameter_index = definition.primary_parameter_index
    if primary_parameter_index is not None:
        primary_parameter = definition.parameters[primary_parameter_index]
        primary_slot = slots[primary_parameter_index]
        if primary_slot is not None:
            parameters.append(
                _format_attr_value(
                    primary_slot,
                    primary_parameter,
                    type_context=type_context,
                    type_registry=type_registry,
                    encoding_alias_selectors=encoding_alias_selectors,
                    use_encoding_aliases=use_encoding_aliases,
                )
            )
    for parameter_index, (parameter, slot) in enumerate(
        zip(definition.parameters, slots, strict=True)
    ):
        if slot is None or parameter_index == primary_parameter_index:
            continue
        formatted_slot = _format_attr_value(
            slot,
            parameter,
            type_context=type_context,
            type_registry=type_registry,
            encoding_alias_selectors=encoding_alias_selectors,
            use_encoding_aliases=use_encoding_aliases,
        )
        parameters.append(f"{parameter.name} = {formatted_slot}")
    return ", ".join(parameters)


def _format_enum_attr_value(
    value: Any,
    attr_def: AttrDef,
    keyword_by_value: dict[int, str] | None = None,
) -> str:
    """Format one descriptor-backed enum keyword or raw open value."""
    assert attr_def.enum_def is not None
    if keyword_by_value is None:
        keyword_by_value = {
            case.value: case.keyword for case in attr_def.enum_def.cases
        }
    if isinstance(value, str):
        if value not in attr_def.enum_def.keywords:
            raise ValueError(f"unknown {attr_def.enum_def.name} keyword {value!r}")
        return value
    if type(value) is not int or not 0 <= value <= 0xFF:
        raise ValueError(
            f"{attr_def.enum_def.name} value must be a byte, got {value!r}"
        )
    keyword = keyword_by_value.get(value)
    if keyword is not None:
        return keyword
    if not attr_def.open_enum:
        raise ValueError(f"closed {attr_def.enum_def.name} enum has no value {value}")
    return f"<{value}>"


def _is_pipeline_printable_name(value: Any, *, allow_dot: bool) -> bool:
    """Returns true when value can be printed as a pipeline syntax identifier."""
    if not isinstance(value, str) or not value:
        return False
    first = value[0]
    if not (first.isascii() and (first.isalpha() or first in "_$")):
        return False
    for character in value[1:]:
        if not character.isascii():
            return False
        if (
            character.isalnum()
            or character in "_$-"
            or (allow_dot and character == ".")
        ):
            continue
        return False
    return True


def _is_pipeline_printable_attr_value(
    value: Any, attr_def: AttrDef | None = None
) -> bool:
    if attr_def is not None and attr_def.attr_type in ("enum", "enum_array"):
        return True
    if attr_def is not None and attr_def.attr_type == "symbol":
        return _is_pipeline_printable_name(str(value), allow_dot=False)
    if isinstance(value, ParameterizedAttr):
        return all(
            slot is None or _is_pipeline_printable_attr_value(slot, parameter)
            for parameter, slot in zip(
                value.definition.parameters, value.slots, strict=True
            )
        )
    if isinstance(value, ParameterizedAttrArray):
        if attr_def is None or attr_def.attr_type != "parameterized_array":
            return False
        element_descriptor = AttrDef(
            attr_def.name,
            "parameterized",
            parameterized_attr=attr_def.parameterized_attr,
        )
        return all(
            _is_pipeline_printable_attr_value(element, element_descriptor)
            for element in value
        )
    if isinstance(value, bool | int | float | str | bytes | bytearray):
        return True
    if isinstance(value, list | tuple):
        return all(_is_pipeline_printable_attr_value(item) for item in value)
    if isinstance(value, Mapping):
        return _is_pipeline_printable_attr_dict(value)
    if isinstance(value, (*_IR_TYPE_CLASSES, EncodingInstance)):
        return True
    return False


def _is_pipeline_printable_attr_dict(value: Mapping[str, Any] | None) -> bool:
    if value is None:
        return True
    return all(
        _is_pipeline_printable_name(key, allow_dot=False)
        and _is_pipeline_printable_attr_value(item)
        for key, item in value.items()
    )


def _format_float(value: float) -> str:
    """Format a float with enough precision to round-trip."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0.0 else "inf"
    text = f"{value:.17g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def _format_predicate_arg(arg: PredicateArg) -> str:
    """Format a single predicate argument."""
    match arg.tag:
        case "value":
            return f"%{arg.value}"
        case "const":
            return str(arg.value)
        case _:
            raise ValueError(f"unknown predicate arg tag: {arg.tag!r}")


def _format_predicate(predicate: Predicate) -> str:
    """Format a single predicate: kind(arg, arg, ...)."""
    arg_strs = [_format_predicate_arg(a) for a in predicate.args]
    return f"{predicate.kind}({', '.join(arg_strs)})"


def _format_predicate_list(predicates: list[Predicate]) -> str:
    """Format a predicate list: [pred(...), pred(...)]."""
    parts = [_format_predicate(p) for p in predicates]
    return "[" + ", ".join(parts) + "]"


def _is_symbol_define(op_decl: Op) -> bool:
    """Check if an op declaration has the SymbolDefine trait."""
    return any(t.name == "SymbolDefine" for t in op_decl.traits)


def _implicit_terminator_name(op_decl: Op) -> str | None:
    """Returns the implicit terminator op name for regions owned by op_decl."""
    traits = [trait for trait in op_decl.traits if trait.name == "ImplicitTerminator"]
    if not traits:
        return None
    if len(traits) != 1 or len(traits[0].args) != 1:
        raise ValueError(
            f"Op '{op_decl.name}' has malformed ImplicitTerminator trait: {traits!r}"
        )
    return traits[0].args[0]


# ============================================================================
# Printer
# ============================================================================


class Printer:
    """Format-driven text printer for loom IR.

    Usage:
        printer = Printer()
        printer.register_ops(ALL_TEST_OPS)
        text = printer.print_module(module)

    The printer maintains:
      - An op registry (Op declarations for format-driven printing).
      - A field layout cache (computed once per op kind).
      - A name table (built per function, maps value IDs to SSA names).
    """

    def __init__(
        self,
        *,
        print_locations: bool = False,
        use_aliases: bool = True,
        indent: bool = True,
        print_regions: bool = True,
    ) -> None:
        self._registry: dict[str, Op] = {}
        self._type_registry: dict[Any, Any] = {}  # TypeDef by name and value class.
        self._encoding_alias_selectors: dict[str, _EncodingAliasSelector] = {}
        self._layouts: dict[str, FieldLayout] = {}
        self._module: Module | None = None
        self._indent: int = 0
        self._lines: list[str] = []
        self._print_locations: bool = print_locations
        self._use_aliases: bool = use_aliases
        self._indent_enabled: bool = indent
        self._print_regions: bool = print_regions

    def register_ops(self, ops: Sequence[Op]) -> None:
        """Register op declarations for format-driven printing."""
        for op in ops:
            self._registry[op.name] = op

    def register_types(self, types: Sequence[Any]) -> None:
        """Register type declarations for format-driven type printing."""
        for td in types:
            self._type_registry[td.name] = td
            if td.python_type is not None:
                self._type_registry[td.python_type] = td

    def register_encoding_families(self, families: Sequence[EncodingFamilyDef]) -> None:
        """Register canonical source spellings for encoding families."""
        for family in families:
            if not family.aliases:
                continue
            discriminator_name = cast(AttrDef, family.alias_discriminator).name
            aliases_by_discriminator = {
                dict(alias.fixed_parameters)[discriminator_name]: alias
                for alias in family.aliases
            }
            self._encoding_alias_selectors[family.name] = _EncodingAliasSelector(
                discriminator_name=discriminator_name,
                aliases_by_discriminator=aliases_by_discriminator,
                parameters_by_name={
                    parameter.name: parameter for parameter in family.parameters
                },
            )

    # --- Layout cache ---

    def _layout(self, op_decl: Op) -> FieldLayout:
        """Get or compute the field layout for an op kind."""
        layout = self._layouts.get(op_decl.name)
        if layout is None:
            layout = compute_layout(op_decl)
            self._layouts[op_decl.name] = layout
        return layout

    # --- Name and type resolution ---

    def _value_name(self, value_id: int) -> str:
        """Resolve a value ID to its SSA name."""
        assert self._module is not None
        return resolve_value_name(self._module, value_id)

    def _type_context(self, value: Value, module: Module) -> TypePrintContext:
        """Create a TypePrintContext for a value's dim and encoding bindings."""
        return TypePrintContext(
            value.dim_bindings,
            module,
            encoding_binding=value.encoding_binding,
            use_aliases=self._use_aliases,
            encoding_alias_selectors=self._encoding_alias_selectors,
        )

    def _format_attr_value(
        self,
        value: Any,
        attr_def: AttrDef | None = None,
        *,
        type_context: TypePrintContext | None = None,
    ) -> str:
        return _format_attr_value(
            value,
            attr_def,
            type_context=type_context,
            type_registry=self._type_registry,
            encoding_alias_selectors=self._encoding_alias_selectors,
            use_encoding_aliases=self._use_aliases,
        )

    def _format_parameterized_attr_parameters(
        self,
        value: Any,
        definition: ParameterizedAttrDef,
    ) -> str:
        return _format_parameterized_attr_parameters(
            value,
            definition,
            type_registry=self._type_registry,
            encoding_alias_selectors=self._encoding_alias_selectors,
            use_encoding_aliases=self._use_aliases,
        )

    def _print_value_type(self, value_id: int, module: Module) -> str:
        """Print the type of a value with dim names and encodings."""
        value = module.values[value_id]
        context = self._type_context(value, module)
        return print_type(value.type, context, self._type_registry)

    # --- Output ---

    def _emit(self, line: str) -> None:
        """Append an indented line to the output."""
        if self._indent_enabled:
            self._lines.append("  " * self._indent + line)
        else:
            self._lines.append(line)

    def _emit_comments(self, comments: tuple[str, ...]) -> None:
        """Emit leading line comments at the current indentation."""
        for comment in comments:
            self._emit("//" + comment)

    # --- Module printing ---

    def print_module(self, module: Module) -> str:
        """Print a complete module to canonical text."""
        self._lines = []
        self._module = module
        for encoding in module.encodings:
            if encoding.alias:
                self._emit(
                    f"#{encoding.alias} = "
                    f"{_format_encoding_instance(encoding, use_alias=False)}"
                )
        for symbol in module.symbols:
            if symbol.op is not None:
                self._print_top_level_op(symbol.op, module)
                self._lines.append("")
        # Remove trailing blank line.
        while self._lines and self._lines[-1] == "":
            self._lines.pop()
        return "\n".join(self._lines) + "\n" if self._lines else ""

    def _print_top_level_op(self, op: Operation, module: Module) -> None:
        """Print a top-level symbol-defining op using the format-element-driven walker.

        All func-like ops (func.def, func.decl, func.template, func.ukernel)
        and future symbol-defining ops are printed through this path using the
        same format walker as body ops.
        """
        self._module = module
        self._emit_comments(op.comments)

        op_decl = self._registry.get(op.name)
        if op_decl is None:
            # Fallback for ops with no registered format.
            self._emit(f"{op.name}()")
            return

        layout = self._layout(op_decl)
        fields = resolve_fields(layout, op, module)
        stream = TokenStream()
        covered_attrs: set[str] = set()

        # Op name (no LHS results — SYMBOL_DEFINE trait).
        stream.emit(op.name)

        stream = self._walk_format_inline(
            op_decl.format,
            op_decl,
            fields,
            module,
            stream,
            covered_attrs,
            self._print_regions,
        )

        if self._print_locations:
            loc_str = self._format_location(op.location_id, module)
            if loc_str:
                stream.emit(loc_str)

        self._emit(stream.join())

    # --- Region body printing ---

    def _print_region_body(
        self,
        region: Region,
        module: Module,
        implicit_terminator_name: str | None = None,
    ) -> None:
        """Print the blocks of a region."""
        for block in region.blocks:
            # Block label (if named and not the entry block).
            if block.label:
                if block.leading_blank_line:
                    self._lines.append("")
                saved_indent = self._indent
                self._indent = max(self._indent - 1, 0)
                self._emit_comments(block.comments)
                arg_strs = ""
                if block.arg_ids:
                    args = []
                    for arg_id in block.arg_ids:
                        name = self._value_name(arg_id)
                        arg_type = print_type(module.values[arg_id].type)
                        args.append(f"{name}: {arg_type}")
                    arg_strs = "(" + ", ".join(args) + ")"
                self._emit(f"^{block.label}{arg_strs}:")
                self._indent = saved_indent
            final_live_op_index = self._printable_final_op_index(block)
            for i, op in enumerate(block.ops):
                if not op.is_dead:
                    if (
                        i == final_live_op_index
                        and self._should_elide_implicit_terminator(
                            op, implicit_terminator_name
                        )
                    ):
                        continue
                    if op.leading_blank_line:
                        self._lines.append("")
                    self._print_op(op, module)

    def _printable_final_op_index(self, block: Block) -> int:
        """Returns the index of the final non-dead op, or -1 if none."""
        for i in range(len(block.ops) - 1, -1, -1):
            if not block.ops[i].is_dead:
                return i
        return -1

    def _can_print_pipeline_region(
        self,
        region: Region,
        module: Module,
        implicit_terminator_name: str | None,
    ) -> bool:
        """Returns true when the region has an exact friendly pipeline spelling."""
        if self._print_locations:
            return False
        if len(region.blocks) != 1:
            return False
        block = region.blocks[0]
        if block.label or block.arg_ids or block.comments:
            return False

        final_live_op_index = self._printable_final_op_index(block)
        if implicit_terminator_name is None or final_live_op_index < 0:
            return False
        if not self._should_elide_implicit_terminator(
            block.ops[final_live_op_index], implicit_terminator_name
        ):
            return False
        for i, op in enumerate(block.ops):
            if op.is_dead:
                continue
            if i == final_live_op_index and self._should_elide_implicit_terminator(
                op, implicit_terminator_name
            ):
                continue
            if op.name == "pass.yield":
                return False
            if not self._can_print_pipeline_statement(
                op, module, implicit_terminator_name
            ):
                return False
        return True

    def _can_print_pipeline_statement(
        self,
        op: Operation,
        module: Module,
        implicit_terminator_name: str | None,
    ) -> bool:
        match op.name:
            case "pass.run":
                options = op.attributes.get("options")
                return _is_pipeline_printable_name(
                    op.attributes.get("key"), allow_dot=True
                ) and (
                    "options" not in op.attributes
                    or (
                        isinstance(options, Mapping)
                        and _is_pipeline_printable_attr_dict(options)
                    )
                )
            case "pass.for":
                return (
                    op.attributes.get("anchor") in ("module", "func")
                    and len(op.regions) == 1
                    and self._can_print_pipeline_region(
                        op.regions[0], module, implicit_terminator_name
                    )
                )
            case "pass.where":
                attrs = op.attributes.get("attrs")
                return (
                    _is_pipeline_printable_name(
                        op.attributes.get("predicate"), allow_dot=True
                    )
                    and (
                        "attrs" not in op.attributes
                        or (
                            isinstance(attrs, Mapping)
                            and _is_pipeline_printable_attr_dict(attrs)
                        )
                    )
                    and len(op.regions) == 1
                    and self._can_print_pipeline_region(
                        op.regions[0], module, implicit_terminator_name
                    )
                )
            case "pass.repeat":
                attrs = op.attributes
                return (
                    attrs.get("mode") in ("fixed", "until_converged")
                    and all(key in ("mode", "count", "max_iterations") for key in attrs)
                    and (
                        "count" not in attrs
                        or (
                            isinstance(attrs.get("count"), int)
                            and not isinstance(attrs.get("count"), bool)
                        )
                    )
                    and (
                        "max_iterations" not in attrs
                        or (
                            isinstance(attrs.get("max_iterations"), int)
                            and not isinstance(attrs.get("max_iterations"), bool)
                        )
                    )
                    and len(op.regions) == 1
                    and self._can_print_pipeline_region(
                        op.regions[0], module, implicit_terminator_name
                    )
                )
            case "pass.if_changed":
                return (
                    not op.attributes
                    and len(op.regions) == 1
                    and self._can_print_pipeline_region(
                        op.regions[0], module, implicit_terminator_name
                    )
                )
            case "pass.call":
                return _is_pipeline_printable_name(
                    op.attributes.get("callee"), allow_dot=False
                )
            case "pass.fail" | "pass.halt":
                return isinstance(op.attributes.get("message"), str)
            case _:
                return False

    def _print_pipeline_region_body(
        self,
        region: Region,
        module: Module,
        *,
        implicit_terminator_name: str | None,
    ) -> None:
        if not region.blocks:
            return
        block = region.blocks[0]
        final_live_op_index = self._printable_final_op_index(block)
        for i, op in enumerate(block.ops):
            if op.is_dead:
                continue
            if i == final_live_op_index and self._should_elide_implicit_terminator(
                op, implicit_terminator_name
            ):
                continue
            if op.leading_blank_line:
                self._lines.append("")
            self._print_pipeline_statement(
                op,
                module,
                implicit_terminator_name=implicit_terminator_name,
            )

    def _print_pipeline_statement(
        self,
        op: Operation,
        module: Module,
        *,
        implicit_terminator_name: str | None,
    ) -> None:
        self._emit_comments(op.comments)
        match op.name:
            case "pass.run":
                key = cast(str, op.attributes["key"])
                options = op.attributes.get("options")
                suffix = (
                    self._format_pipeline_attr_parens(cast(Mapping[str, Any], options))
                    if isinstance(options, Mapping)
                    else ""
                )
                self._emit(f"{key}{suffix}")
            case "pass.for":
                anchor = cast(str, op.attributes["anchor"])
                self._emit(f"for {anchor} {{")
                self._indent += 1
                self._print_pipeline_region_body(
                    op.regions[0],
                    module,
                    implicit_terminator_name=implicit_terminator_name,
                )
                self._indent -= 1
                self._emit("}")
            case "pass.where":
                predicate = cast(str, op.attributes["predicate"])
                attrs = op.attributes.get("attrs")
                suffix = (
                    self._format_pipeline_attr_parens(cast(Mapping[str, Any], attrs))
                    if isinstance(attrs, Mapping)
                    else ""
                )
                self._emit(f"where {predicate}{suffix} {{")
                self._indent += 1
                self._print_pipeline_region_body(
                    op.regions[0],
                    module,
                    implicit_terminator_name=implicit_terminator_name,
                )
                self._indent -= 1
                self._emit("}")
            case "pass.repeat":
                mode = cast(str, op.attributes["mode"])
                repeat_attrs: list[tuple[str, Any]] = []
                if "count" in op.attributes:
                    repeat_attrs.append(("count", op.attributes["count"]))
                if "max_iterations" in op.attributes:
                    repeat_attrs.append(
                        ("max_iterations", op.attributes["max_iterations"])
                    )
                suffix = self._format_pipeline_attr_parens(
                    CanonicalAttrDict(repeat_attrs)
                )
                self._emit(f"repeat {mode}{suffix} {{")
                self._indent += 1
                self._print_pipeline_region_body(
                    op.regions[0],
                    module,
                    implicit_terminator_name=implicit_terminator_name,
                )
                self._indent -= 1
                self._emit("}")
            case "pass.if_changed":
                self._emit("if changed {")
                self._indent += 1
                self._print_pipeline_region_body(
                    op.regions[0],
                    module,
                    implicit_terminator_name=implicit_terminator_name,
                )
                self._indent -= 1
                self._emit("}")
            case "pass.call":
                callee = cast(str, op.attributes["callee"])
                self._emit(f"call @{callee}")
            case "pass.fail":
                message = _format_string_literal(cast(str, op.attributes["message"]))
                self._emit(f"fail {message}")
            case "pass.halt":
                message = _format_string_literal(cast(str, op.attributes["message"]))
                self._emit(f"halt {message}")
            case _:
                self._print_op(op, module)

    def _format_pipeline_attr_parens(self, attrs: Mapping[str, Any]) -> str:
        if not attrs:
            return ""
        body = ", ".join(
            f"{key} = {self._format_attr_value(value)}" for key, value in attrs.items()
        )
        return f"({body})"

    def _should_elide_implicit_terminator(
        self,
        op: Operation,
        implicit_terminator_name: str | None,
    ) -> bool:
        """Returns true if op is the final empty implicit terminator."""
        return (
            implicit_terminator_name is not None
            and op.name == implicit_terminator_name
            and not op.operands
            and not op.results
            and not op.tied_results
            and not op.attributes
            and not op.regions
        )

    def _implicit_region_arg_id(
        self,
        op_decl: Op,
        fields: FormatFields,
        name: str,
    ) -> int | None:
        """Returns the entry-block arg ID for an implicit region Ref field."""
        for region_decl in op_decl.regions:
            for arg_index, (arg_name, _arg_type) in enumerate(
                region_decl.implicit_args
            ):
                if arg_name != name:
                    continue
                region = fields.region(region_decl.name)
                if (
                    region is None
                    or not region.blocks
                    or arg_index >= len(region.blocks[0].arg_ids)
                ):
                    raise ValueError(
                        f"Op '{op_decl.name}' region '{region_decl.name}' has no "
                        f"entry block arg for implicit field '{name}'"
                    )
                return region.blocks[0].arg_ids[arg_index]
        return None

    # --- Op printing ---

    def print_operation(
        self, op: Operation, module: Module, print_regions: bool = True
    ) -> str:
        """Print a single operation (may be multi-line for ops with regions)."""
        self._module = module
        saved_lines = self._lines
        saved_indent = self._indent
        self._lines = []
        self._indent = 0
        self._print_op(op, module, print_regions=print_regions)
        result = "\n".join(self._lines)
        self._lines = saved_lines
        self._indent = saved_indent
        return result

    def _print_op(
        self, op: Operation, module: Module, print_regions: bool | None = None
    ) -> None:
        """Print an op as one or more indented lines."""
        self._emit_comments(op.comments)
        if print_regions is None:
            print_regions = self._print_regions
        op_decl = self._registry.get(op.name)
        if op_decl is None:
            self._emit(self._generic_op_string(op, module))
            return

        layout = self._layout(op_decl)
        fields = resolve_fields(layout, op, module)

        stream = TokenStream()
        covered_attrs: set[str] = set()

        # SYMBOL_DEFINE ops (functions) don't print LHS results.
        if op.results and not _is_symbol_define(op_decl):
            result_names = [self._value_name(vid) for vid in op.results]
            stream.emit(", ".join(result_names))
            stream.emit("=")

        stream.emit(op.name)

        # Walk format elements. Regions are printed inline: when a
        # Region element is encountered, the current tokens are flushed
        # as a line ending with " {", the body is printed indented, and
        # a new token accumulation starts with "}".
        stream = self._walk_format_inline(
            op_decl.format,
            op_decl,
            fields,
            module,
            stream,
            covered_attrs,
            print_regions,
        )

        # Location annotation (omitted for LOCATION_UNKNOWN = 0).
        if self._print_locations:
            loc_str = self._format_location(op.location_id, module)
            if loc_str:
                stream.emit(loc_str)

        self._emit(stream.join())

    def _format_location(self, location_id: int, module: Module) -> str:
        """Format a location annotation, or return empty string for unknown."""
        from loom.ir import (
            LOCATION_UNKNOWN,
            FileLocation,
            FusedLocation,
            OpaqueLocation,
            TaggedLocation,
        )

        def format_file_body(loc: FileLocation, *, always_print_range: bool) -> str:
            source = (
                module.sources[loc.source_id]
                if loc.source_id < len(module.sources)
                else "?"
            )
            text = f"{_format_string_literal(source)}:{loc.start_line}:{loc.start_col}"
            if (
                always_print_range
                or loc.end_line != loc.start_line
                or loc.end_col != loc.start_col
            ):
                text += f" to {loc.end_line}:{loc.end_col}"
            return text

        def format_body(body_location_id: int) -> str:
            body = module.locations.get(body_location_id)
            if body is None:
                return ""
            if isinstance(body, FileLocation):
                return format_file_body(body, always_print_range=False)
            if isinstance(body, FusedLocation):
                parts = [format_body(child_id) for child_id in body.children]
                return f"fused<{', '.join(parts)}>"
            if isinstance(body, OpaqueLocation):
                tag = (
                    module.sources[body.source_id]
                    if body.source_id < len(module.sources)
                    else "?"
                )
                try:
                    data = body.data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("opaque location data is not valid UTF-8") from exc
                return (
                    f"opaque<{_format_string_literal(tag)}, "
                    f"{_format_string_literal(data)}>"
                )
            if isinstance(body, TaggedLocation):
                if body.tag <= 0 or body.tag > 0xFFFF:
                    raise ValueError("tagged location tag must be in [1, 65535]")
                tag = builtin_location_tag_name(body.tag)
                if tag is None:
                    tag = str(body.tag)
                text = f"tagged<{tag}, {_format_string_literal(body.data.hex())}"
                if body.child != LOCATION_UNKNOWN:
                    text += f", {format_body(body.child)}"
                return text + ">"
            return ""

        if location_id == LOCATION_UNKNOWN:
            return ""
        loc = module.locations.get(location_id)
        if loc is None:
            return ""
        if isinstance(loc, FileLocation):
            return f"loc({format_file_body(loc, always_print_range=True)})"
        return f"loc({format_body(location_id)})"

    def _walk_format_inline(
        self,
        elements: tuple[FormatElement, ...],
        op_decl: Op,
        fields: FormatFields,
        module: Module,
        stream: TokenStream,
        covered_attrs: set[str],
        print_regions: bool = True,
    ) -> TokenStream:
        """Walk format elements with inline region printing.

        Returns the current token stream (may differ from the input stream
        if a region was printed — the old stream is flushed as a line and
        a new one starts with '}').
        """
        for element_index, element in enumerate(elements):
            match element:
                case Ref(field=name):
                    try:
                        vid = fields.value_id(name)
                    except KeyError:
                        implicit_arg_id = self._implicit_region_arg_id(
                            op_decl, fields, name
                        )
                        if implicit_arg_id is None:
                            raise
                        vid = implicit_arg_id
                    stream.emit(self._value_name(vid))

                case Refs(field=name):
                    vids = fields.value_ids(name)
                    if vids:
                        names = [self._value_name(vid) for vid in vids]
                        stream.emit(", ".join(names))

                case TypedRefs(field=name):
                    vids = fields.value_ids(name)
                    if vids:
                        entries = [
                            (
                                f"{self._value_name(vid)}: "
                                f"{self._print_value_type(vid, module)}"
                            )
                            for vid in vids
                        ]
                        stream.emit(", ".join(entries))

                case Attr(field=name):
                    covered_attrs.add(name)
                    value = fields.attr(name)
                    if value is not None:
                        attr_def = op_decl.attr(name)
                        stream.emit(self._format_attr_value(value, attr_def))

                case SymbolRef(field=name):
                    covered_attrs.add(name)
                    value = fields.attr(name)
                    if value is not None:
                        stream.emit("@" + str(value))

                case TypeOf(field=name):
                    vid = fields.value_id(name)
                    stream.emit(self._print_value_type(vid, module))

                case TypesOf(field=name):
                    vids = fields.value_ids(name)
                    if vids:
                        type_strs = [
                            self._print_value_type(vid, module) for vid in vids
                        ]
                        stream.emit(", ".join(type_strs))

                case ResultType(field=name):
                    assert isinstance(fields, ResolvedFields)
                    result_id = fields.value_id(name)
                    stream.emit(self._print_value_type(result_id, module))

                case ResultTypeList(field=name, parens=parens, uniform=uniform):
                    assert isinstance(fields, ResolvedFields)
                    stream.emit(
                        self._format_result_types(
                            fields,
                            name,
                            op_decl,
                            parens=parens,
                            uniform=uniform,
                        )
                    )

                case Keyword(text=text):
                    stream.emit(text)

                case Clause(name=name, elements=inner):
                    stream.emit(name)
                    stream.emit("(", glue=True)
                    stream = self._walk_format_inline(
                        inner,
                        op_decl,
                        fields,
                        module,
                        stream,
                        covered_attrs,
                        print_regions,
                    )
                    stream.emit(")")

                case AttrDict(field=dict_field):
                    if dict_field and hasattr(fields, "_op"):
                        # Named dict attribute: read the dict value directly.
                        covered_attrs.add(dict_field)
                        dict_value = fields._op.attributes.get(dict_field)
                        if isinstance(dict_value, Mapping):
                            stream.emit(self._format_named_dict(dict_value, op_decl))
                    elif hasattr(fields, "_op"):
                        # Legacy: uncovered attributes from the op's dict.
                        attr_str = self._format_attr_dict(
                            fields._op.attributes,
                            covered_attrs,
                            op_decl,
                        )
                        if attr_str:
                            stream.emit(attr_str)

                case AttrTable(keys=keys_field, values=values_field):
                    assert isinstance(fields, ResolvedFields)
                    covered_attrs.add(keys_field)
                    stream = self._emit_attr_table(
                        stream, fields, keys_field, values_field
                    )

                case AlignedRefs(refs=refs_field, alignments=alignments_field):
                    assert isinstance(fields, ResolvedFields)
                    covered_attrs.add(alignments_field)
                    stream.emit(
                        self._format_aligned_refs(fields, refs_field, alignments_field)
                    )

                case RegionTable(
                    keys=keys_field,
                    case_regions=case_regions_field,
                    default_region=default_region_field,
                ):
                    assert isinstance(fields, ResolvedFields)
                    covered_attrs.add(keys_field)
                    if not print_regions:
                        stream.emit("{ ... }")
                    else:
                        self._emit(stream.join() + " {")
                        self._indent += 1
                        self._format_region_table(
                            fields,
                            keys_field,
                            case_regions_field,
                            default_region_field,
                            op_decl,
                            module,
                        )
                        self._indent -= 1
                        stream = TokenStream()
                        stream.emit("}")

                case OperandDict(operands=operand_field, names=names_field):
                    assert isinstance(fields, ResolvedFields)
                    covered_attrs.add(names_field)
                    operand_dict_str = self._format_operand_dict(
                        fields, operand_field, names_field, module
                    )
                    if operand_dict_str:
                        stream.emit(operand_dict_str)

                case RegionFmt(field=name, syntax=syntax):
                    region = fields.region(name)
                    implicit_terminator_name = _implicit_terminator_name(op_decl)
                    if not print_regions:
                        # Declaration mode: placeholder braces.
                        if syntax == "pipeline":
                            stream.emit("pipeline { ... }")
                        elif syntax == "test.do":
                            stream.emit("do { ... }")
                        else:
                            stream.emit("{ ... }")
                    elif (
                        syntax == "pipeline"
                        and region is not None
                        and (
                            self._can_print_pipeline_region(
                                region, module, implicit_terminator_name
                            )
                        )
                    ):
                        self._emit(stream.join() + " pipeline {")
                        self._indent += 1
                        self._print_pipeline_region_body(
                            region,
                            module,
                            implicit_terminator_name=implicit_terminator_name,
                        )
                        self._indent -= 1
                        stream = TokenStream()
                        stream.emit("}")
                    elif syntax == "test.do":
                        self._emit(stream.join() + " do {")
                        if region is not None:
                            self._indent += 1
                            self._print_region_body(
                                region,
                                module,
                                implicit_terminator_name=implicit_terminator_name,
                            )
                            self._indent -= 1
                        stream = TokenStream()
                        stream.emit("}")
                    else:
                        # Flush current tokens + " {" as a line.
                        self._emit(stream.join() + " {")
                        # Print region body indented.
                        if region is not None:
                            self._indent += 1
                            self._print_region_body(
                                region,
                                module,
                                implicit_terminator_name=implicit_terminator_name,
                            )
                            self._indent -= 1
                        # Start new token accumulation with "}".
                        stream = TokenStream()
                        stream.emit("}")

                case IndexList(dynamic=dynamic_field, static=static_field, glue=glue):
                    assert isinstance(fields, ResolvedFields)
                    covered_attrs.add(static_field)
                    stream.emit(
                        self._format_index_list(fields, dynamic_field, static_field),
                        glue=element_index != 0 and glue,
                    )

                case BindingList(field=name):
                    assert isinstance(fields, ResolvedFields)
                    stream.emit(
                        self._format_binding_list(fields, name, module), glue=True
                    )

                case BlockArgs(region=name):
                    assert isinstance(fields, ResolvedFields)
                    stream.emit(
                        self._format_block_args(fields, name, module), glue=True
                    )

                case FuncArgs(
                    field=name,
                    start_attr=start_attr,
                    end_attr=end_attr,
                ):
                    assert isinstance(fields, ResolvedFields)
                    if start_attr is not None:
                        covered_attrs.add(start_attr)
                    if end_attr is not None:
                        covered_attrs.add(end_attr)
                    arg_names, _arg_types, arg_value_ids = fields.func_args(name)
                    start = fields.attr(start_attr) if start_attr is not None else 0
                    end = (
                        fields.attr(end_attr)
                        if end_attr is not None
                        else len(arg_value_ids)
                    )
                    if (
                        not isinstance(start, int)
                        or not isinstance(end, int)
                        or start < 0
                        or end < start
                        or end > len(arg_value_ids)
                    ):
                        raise ValueError(
                            f"Op '{op_decl.name}' function argument slice "
                            f"[{start}, {end}) is outside its "
                            f"{len(arg_value_ids)} arguments."
                        )
                    arg_names = arg_names[start:end]
                    arg_value_ids = arg_value_ids[start:end]
                    arg_strs: list[str] = []
                    for i, arg_value_id in enumerate(arg_value_ids):
                        type_str = self._print_value_type(arg_value_id, module)
                        arg_name = arg_names[i] if i < len(arg_names) else ""
                        if arg_name:
                            arg_strs.append(f"%{arg_name}: {type_str}")
                        else:
                            arg_strs.append(type_str)
                    stream.emit("(" + ", ".join(arg_strs) + ")", glue=True)

                case PredicateList(field=name):
                    predicates = fields.attr(name) if hasattr(fields, "attr") else None
                    if predicates is None and hasattr(fields, "_op"):
                        predicates = fields._op.attributes.get(name)
                    if predicates is not None:
                        stream.emit(_format_predicate_list(predicates))

                case OptionalGroup(elements=inner, anchor=anchor):
                    if fields.is_present(anchor):
                        stream = self._walk_format_inline(
                            inner,
                            op_decl,
                            fields,
                            module,
                            stream,
                            covered_attrs,
                            print_regions,
                        )

                case Scope(elements=inner):
                    stream = self._walk_format_inline(
                        inner,
                        op_decl,
                        fields,
                        module,
                        stream,
                        covered_attrs,
                        print_regions,
                    )

                case Flags(field=name):
                    covered_attrs.add(name)
                    value = fields.attr(name)
                    if value:
                        stream.emit(f"<{value}>", glue=True)

                case KeyRef(field=name):
                    covered_attrs.add(name)
                    value = fields.attr(name)
                    if value:
                        stream.emit(f"<{value}>", glue=True)

                case ScopedEnumRef(field=name):
                    covered_attrs.add(name)
                    value = fields.attr(name)
                    if value:
                        stream.emit(f"<{value}>", glue=True)

                case StableKeyRef(key=key, stable_id=stable_id):
                    covered_attrs.add(key)
                    covered_attrs.add(stable_id)
                    value = fields.attr(key)
                    if value:
                        stream.emit(f"<{value}>", glue=True)

                case TemplateParam(field=name):
                    covered_attrs.add(name)
                    attr_def = op_decl.attr(name)
                    value = fields.attr(name)
                    stream.emit(
                        f"<{self._format_attr_value(value, attr_def)}>",
                        glue=True,
                    )

                case AttrParams(field=name):
                    covered_attrs.add(name)
                    attr_def = cast(AttrDef, op_decl.attr(name))
                    value = fields.attr(name)
                    definition = cast(ParameterizedAttrDef, attr_def.parameterized_attr)
                    parameters = self._format_parameterized_attr_parameters(
                        value, definition
                    )
                    stream.emit(
                        f"<{parameters}>",
                        glue=True,
                    )

                case TemplateParamFlags(param=param_name, flags=flags_name):
                    covered_attrs.add(param_name)
                    covered_attrs.add(flags_name)
                    param_attr_def = op_decl.attr(param_name)
                    param_value = fields.attr(param_name)
                    flags_value = fields.attr(flags_name)
                    param_text = self._format_attr_value(param_value, param_attr_def)
                    if flags_value:
                        stream.emit(f"<{param_text}, {flags_value}>", glue=True)
                    else:
                        stream.emit(f"<{param_text}>", glue=True)

                case Glue():
                    stream.set_glue()

        return stream

    # --- Formatting helpers ---

    def _format_result_types(
        self,
        fields: ResolvedFields,
        name: str,
        op_decl: Op,
        *,
        parens: bool = True,
        uniform: bool = False,
    ) -> str:
        """Format result types with optional parentheses."""
        desc = fields._layout.fields.get(name)
        if desc is None:
            return "()" if parens else ""

        if uniform:
            result_ids = list(fields._op.results)
        elif desc.variadic:
            result_ids = fields.value_ids(name)
        else:
            result_ids = [fields.value_id(name)]

        if not result_ids:
            return "()" if parens else ""

        tied_map = fields.tied_result_map()
        parts: list[str] = []

        for result_id in result_ids[:1] if uniform else result_ids:
            type_str = self._print_value_type(result_id, fields._module)
            value = fields._module.values[result_id]

            # Find this result's position in the op's result list.
            try:
                result_position = fields._op.results.index(result_id)
            except ValueError:
                result_position = -1

            if result_position in tied_map:
                tied = tied_map[result_position]
                operand_name = fields.operand_name_for_tied(tied)
                parts.append(f"{operand_name} as {type_str}")
            elif value.name:
                # Named result: %name: type.
                # Omit the name if it matches the LHS result name (to avoid
                # redundancy in func.call and other body ops).
                # Symbol-defining ops have no LHS results in the printed format,
                # so we always print the name there.
                is_symbol = _is_symbol_define(op_decl)
                if is_symbol:
                    parts.append(f"%{value.name}: {type_str}")
                else:
                    parts.append(type_str)
            else:
                parts.append(type_str)

        text = ", ".join(parts)
        if parens:
            return "(" + text + ")"
        return text

    def _format_aligned_refs(
        self,
        fields: ResolvedFields,
        refs_field: str,
        alignments_field: str,
    ) -> str:
        """Format [align(N) %value, ...] from paired attrs and operands."""
        alignments = fields.attr(alignments_field) or []
        value_ids = fields.value_ids(refs_field)
        if len(alignments) != len(value_ids):
            raise ValueError(
                f"AlignedRefs field '{refs_field}' has {len(value_ids)} values "
                f"but '{alignments_field}' has {len(alignments)} alignments."
            )
        parts = [
            f"align({alignment}) {self._value_name(value_id)}"
            for alignment, value_id in zip(alignments, value_ids, strict=True)
        ]
        return "[" + ", ".join(parts) + "]"

    def _format_index_list(
        self, fields: ResolvedFields, dynamic_field: str, static_field: str
    ) -> str:
        """Format [0, %x, 4]. Starts with '[' for gluing."""
        static_values = fields.attr(static_field) or []
        dynamic_ids = fields.value_ids(dynamic_field)

        sentinel = -(2**63)
        parts: list[str] = []
        dynamic_index = 0

        if static_values:
            for static_val in static_values:
                if static_val == sentinel and dynamic_index < len(dynamic_ids):
                    parts.append(self._value_name(dynamic_ids[dynamic_index]))
                    dynamic_index += 1
                else:
                    parts.append(str(static_val))
        else:
            parts.extend(self._value_name(vid) for vid in dynamic_ids)

        return "[" + ", ".join(parts) + "]"

    def _format_binding_list(
        self, fields: ResolvedFields, name: str, module: Module
    ) -> str:
        """Format (%block_arg = %operand : type, ...)."""
        operand_ids = fields.value_ids(name)
        if not operand_ids:
            return "()"

        # Binding args are appended after any implicit region args, so use the
        # trailing block args that correspond to the binding operands.
        block_arg_names: list[str] = []
        op = fields._op
        if op.regions:
            first_region = op.regions[0]
            if first_region.blocks:
                entry_block = first_region.blocks[0]
                binding_arg_ids = entry_block.arg_ids
                if len(binding_arg_ids) > len(operand_ids):
                    binding_arg_ids = binding_arg_ids[-len(operand_ids) :]
                block_arg_names.extend(
                    self._value_name(arg_id) for arg_id in binding_arg_ids
                )

        parts: list[str] = []
        for i, operand_id in enumerate(operand_ids):
            operand_name = self._value_name(operand_id)
            operand_type = print_type(module.values[operand_id].type)

            if i < len(block_arg_names):
                parts.append(f"{block_arg_names[i]} = {operand_name} : {operand_type}")
            else:
                parts.append(f"{operand_name} : {operand_type}")

        return "(" + ", ".join(parts) + ")"

    def _format_block_args(
        self, fields: ResolvedFields, name: str, module: Module
    ) -> str:
        """Format (%block_arg: type, ...)."""
        region = fields.region(name)
        entry_block = region.blocks[0] if region and region.blocks else None
        arg_value_ids = list(entry_block.arg_ids) if entry_block else []
        parts: list[str] = []
        for arg_value_id in arg_value_ids:
            arg_type = self._print_value_type(arg_value_id, module)
            parts.append(f"{self._value_name(arg_value_id)}: {arg_type}")
        return "(" + ", ".join(parts) + ")"

    def _format_operand_dict(
        self,
        fields: ResolvedFields,
        operand_field: str,
        names_field: str,
        module: Module,
    ) -> str:
        """Format {key = %value : type, ...} from keyed variadic operands."""
        names = fields.attr(names_field)
        if not names:
            return ""
        if not isinstance(names, Mapping):
            raise TypeError(
                f"OperandDict names field '{names_field}' must be a mapping, "
                f"got {type(names).__name__}."
            )

        operand_ids = fields.value_ids(operand_field)
        parts: list[str] = []
        for key, ordinal in names.items():
            if not isinstance(ordinal, int):
                raise TypeError(
                    f"OperandDict entry '{key}' must map to an integer ordinal, "
                    f"got {type(ordinal).__name__}."
                )
            if ordinal < 0 or ordinal >= len(operand_ids):
                raise ValueError(
                    f"OperandDict entry '{key}' maps to ordinal {ordinal}, "
                    f"but field '{operand_field}' has {len(operand_ids)} operands."
                )
            operand_id = operand_ids[ordinal]
            operand_name = self._value_name(operand_id)
            operand_type = self._print_value_type(operand_id, module)
            parts.append(f"{key} = {operand_name} : {operand_type}")
        return "{" + ", ".join(parts) + "}"

    def _attr_table_rows(
        self,
        fields: ResolvedFields,
        keys_field: str,
        values_field: str,
    ) -> tuple[list[int], list[list[int]], list[int]]:
        """Return normalized rows for a static-keyed value table."""
        keys = fields.attr(keys_field)
        if keys is None:
            keys = []
        if not isinstance(keys, Sequence) or isinstance(keys, str | bytes):
            raise TypeError(
                f"AttrTable keys field '{keys_field}' must be a sequence of ints, "
                f"got {type(keys).__name__}."
            )
        key_values = list(keys)
        for key in key_values:
            if not isinstance(key, int):
                raise TypeError(
                    f"AttrTable key field '{keys_field}' contains "
                    f"{type(key).__name__}, expected int."
                )

        value_ids = fields.value_ids(values_field)
        row_count = len(key_values) + 1
        if len(value_ids) % row_count != 0:
            raise ValueError(
                f"AttrTable values field '{values_field}' has {len(value_ids)} "
                f"values for {row_count} rows."
            )
        row_width = len(value_ids) // row_count

        rows = [
            value_ids[row_index * row_width : (row_index + 1) * row_width]
            for row_index in range(len(key_values))
        ]
        default_row = value_ids[len(key_values) * row_width : row_count * row_width]
        return key_values, rows, default_row

    def _format_attr_table_row(self, value_ids: Sequence[int]) -> str:
        return "(" + ", ".join(self._value_name(v) for v in value_ids) + ")"

    def _format_attr_table(
        self,
        fields: ResolvedFields,
        keys_field: str,
        values_field: str,
        _module: Module,
    ) -> str:
        """Format {key = (%row), ...} default(%row) from flattened operands."""
        key_values, rows, default_row = self._attr_table_rows(
            fields, keys_field, values_field
        )

        parts = [
            f"{key} = {self._format_attr_table_row(row)}"
            for key, row in zip(key_values, rows, strict=True)
        ]
        return (
            "{"
            + ", ".join(parts)
            + "} default"
            + self._format_attr_table_row(default_row)
        )

    def _emit_attr_table(
        self,
        stream: TokenStream,
        fields: ResolvedFields,
        keys_field: str,
        values_field: str,
    ) -> TokenStream:
        """Emit a value table, using one line per non-empty case row."""
        key_values, rows, default_row = self._attr_table_rows(
            fields, keys_field, values_field
        )
        if not key_values:
            stream.emit("{} default" + self._format_attr_table_row(default_row))
            return stream

        self._emit(stream.join() + " {")
        self._indent += 1
        for row_index, (key, row) in enumerate(zip(key_values, rows, strict=True)):
            suffix = "," if row_index + 1 < len(key_values) else ""
            self._emit(f"{key} = {self._format_attr_table_row(row)}{suffix}")
        self._indent -= 1
        stream = TokenStream()
        stream.emit("} default" + self._format_attr_table_row(default_row))
        return stream

    def _format_region_table(
        self,
        fields: ResolvedFields,
        keys_field: str,
        case_regions_field: str,
        default_region_field: str,
        op_decl: Op,
        module: Module,
    ) -> None:
        """Format case/default regions inside an enclosing table block."""
        keys = fields.attr(keys_field)
        if keys is None:
            keys = []
        if not isinstance(keys, Sequence) or isinstance(keys, str | bytes):
            raise TypeError(
                f"RegionTable keys field '{keys_field}' must be a sequence of "
                f"ints, got {type(keys).__name__}."
            )
        key_values = list(keys)
        for key in key_values:
            if not isinstance(key, int):
                raise TypeError(
                    f"RegionTable key field '{keys_field}' contains "
                    f"{type(key).__name__}, expected int."
                )

        case_regions = fields.regions(case_regions_field)
        if len(case_regions) != len(key_values):
            raise ValueError(
                f"RegionTable case region field '{case_regions_field}' has "
                f"{len(case_regions)} regions for {len(key_values)} keys."
            )

        implicit_terminator_name = _implicit_terminator_name(op_decl)
        for key, region in zip(key_values, case_regions, strict=True):
            self._emit(f"case {key} {{")
            self._indent += 1
            self._print_region_body(
                region,
                module,
                implicit_terminator_name=implicit_terminator_name,
            )
            self._indent -= 1
            self._emit("}")

        self._emit("default {")
        self._indent += 1
        default_region = fields.region(default_region_field)
        if default_region is None:
            raise ValueError(
                f"RegionTable default region field '{default_region_field}' is absent."
            )
        self._print_region_body(
            default_region,
            module,
            implicit_terminator_name=implicit_terminator_name,
        )
        self._indent -= 1
        self._emit("}")

    def _format_named_dict(self, dict_value: Mapping[str, Any], op_decl: Op) -> str:
        """Format {key = value, ...} from a named dict attribute."""
        parts: list[str] = []
        for key, value in dict_value.items():
            parts.append(f"{key} = {self._format_attr_value(value)}")
        return "{" + ", ".join(parts) + "}"

    def _format_attr_dict(
        self, attributes: Mapping[str, Any], covered: set[str], op_decl: Op
    ) -> str:
        """Format {key = value, ...} for uncovered attributes."""
        extras = {}
        for key, value in attributes.items():
            if key in covered:
                continue
            attr_def = op_decl.attr(key) if op_decl else None
            if (
                attr_def is not None
                and attr_def.elide_default
                and value == attr_def.default
            ):
                continue
            extras[key] = value
        if not extras:
            return ""

        parts: list[str] = []
        for key, value in extras.items():
            attr_def = op_decl.attr(key) if op_decl else None
            parts.append(f"{key} = {self._format_attr_value(value, attr_def)}")
        return "{" + ", ".join(parts) + "}"

    def _generic_op_string(self, op: Operation, module: Module) -> str:
        """Fallback for ops without a registered declaration."""
        stream = TokenStream()
        if op.results:
            names = [self._value_name(vid) for vid in op.results]
            stream.emit(", ".join(names))
            stream.emit("=")
        stream.emit(op.name)
        if op.operands:
            names = [self._value_name(vid) for vid in op.operands]
            stream.emit(", ".join(names))
        return stream.join()
