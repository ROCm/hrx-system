# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Format-element-driven parser for loom IR text format.

Mirrors the printer: walks an op's format element list consuming
tokens instead of emitting them. Types are parsed through a type
registry (TypeDef declarations), not hardcoded tables.

Architecture (bottom to top):
  1. NameScope — SSA name → value ID mapping with parent chain
  2. Type parser — registry-driven, walks TypeDef format specs
  3. Format walk — mirrors printer's _walk_format, consumes tokens
  4. Structure parser — module/function/block/op orchestration
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum, unique
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
    FuncLikeInterface,
    LoopLikeInterface,
    Op,
    ParameterizedAttrDef,
    TypeConstraint,
    TypeDef,
)
from loom.fields import FieldKind, FieldLayout, compute_layout
from loom.format.text.tokenizer import (
    ParseError,
    SourceLocation,
    Token,
    Tokenizer,
    TokenKind,
)
from loom.ir import (
    ATTR_AGGREGATE_MAX_NESTING_DEPTH,
    BUFFER_TYPE,
    ENCODING_LAYOUT_TYPE,
    ENCODING_SCHEMA_TYPE,
    ENCODING_STORAGE_TYPE,
    ENCODING_TRANSFORM_TYPE,
    ENCODING_TYPE,
    I1,
    I32,
    INDEX,
    NONE_TYPE,
    OFFSET,
    PREDICATE_KINDS,
    VALUE_DEF_BLOCK_NONE,
    Block,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingType,
    EnumArrayAttr,
    FileLocation,
    FunctionType,
    FusedLocation,
    Module,
    OpaqueLocation,
    Operation,
    ParameterizedAttr,
    ParameterizedAttrArray,
    PlaceholderType,
    PoolType,
    Predicate,
    PredicateArg,
    Region,
    RegisterType,
    ScalarType,
    ShapedType,
    SignedEnumSetAttr,
    StaticDim,
    SymbolName,
    SymbolNameArray,
    SymbolNameSet,
    TaggedLocation,
    Type,
    TypeKind,
    Value,
    binding_element_type,
    parse_scalar_type_kind,
    rebuild_value_metadata,
    record_operation_value_metadata,
    symbol_from_operation,
)
from loom.ir import (
    TiedResult as IRTiedResult,
)
from loom.location_tag import parse_builtin_location_tag
from loom.stable_id import stable_id_from_string
from loom.target.descriptor_sets import DESCRIPTOR_SET_REGISTRATIONS

__all__ = [
    "ParseError",
    "Parser",
    "NameScope",
    "parse_type_string",
]


# ============================================================================
# Parser-level state (set by Parser during parse(), read by module-level fns)
# ============================================================================

# These are set by Parser.parse() and read by _parse_encoding() during type
# parsing. This avoids threading alias/encoding state through every function
# in the call chain. Reset to None/empty between parses.
_CURRENT_ALIASES: dict[str, EncodingInstance] | None = None
_CURRENT_KNOWN_ENCODINGS: set[str] | None = None
_CURRENT_IMPLICIT_SHAPED_ATTACHMENTS: set[str] | None = None
_CURRENT_CANONICAL_ENCODING_ALIASES: dict[str, tuple[str, EncodingAliasDef]] | None = (
    None
)


def _parse_special_float(text: str) -> float | None:
    match text:
        case "nan" | "-nan":
            return float("nan")
        case "inf":
            return float("inf")
        case "-inf":
            return float("-inf")
        case _:
            return None


def _parse_float_literal(text: str) -> float:
    unsigned_text = text[1:] if text.startswith("-") else text
    if unsigned_text.startswith(("0x", "0X")):
        try:
            return float.fromhex(text)
        except OverflowError:
            return float("-inf" if text.startswith("-") else "inf")
    return float(text)


def _concrete_type_for_constraint(constraint: TypeConstraint) -> Type | None:
    """Returns the concrete type implied by a singleton type constraint."""
    match constraint:
        case TypeConstraint.I1:
            return I1
        case TypeConstraint.I32:
            return I32
        case TypeConstraint.INDEX:
            return INDEX
        case TypeConstraint.OFFSET:
            return OFFSET
        case TypeConstraint.BUFFER:
            return BUFFER_TYPE
        case TypeConstraint.ANY_ENCODING:
            return ENCODING_TYPE
        case TypeConstraint.ENCODING_LAYOUT:
            return ENCODING_LAYOUT_TYPE
        case TypeConstraint.ENCODING_SCHEMA:
            return ENCODING_SCHEMA_TYPE
        case TypeConstraint.ENCODING_STORAGE:
            return ENCODING_STORAGE_TYPE
        case TypeConstraint.ENCODING_TRANSFORM:
            return ENCODING_TRANSFORM_TYPE
        case _:
            return None


def _parse_generic_attr_value_from_tokens(
    tokenizer: Tokenizer,
    module: Module,
    filename: str,
    *,
    attr_dict_nesting_depth: int = 0,
    scope: NameScope | None = None,
    type_registry: dict[str, TypeDef] | None = None,
    mode: TypeParseMode | None = None,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None = None,
    aliases: dict[str, EncodingInstance] | None = None,
    known_encodings: set[str] | None = None,
) -> Any:
    """Parse an untyped attr value from the current token stream."""
    if tokenizer.at(TokenKind.INTEGER):
        return int(tokenizer.next().text)
    if tokenizer.at(TokenKind.FLOAT):
        return _parse_float_literal(tokenizer.next().text)
    if tokenizer.at(TokenKind.STRING):
        return tokenizer.next().text
    if (
        tokenizer.at(TokenKind.BARE_IDENT, "bytes")
        and tokenizer.peek_n(1).kind == TokenKind.LPAREN
    ):
        return _parse_bytes_attr_value_from_tokens(tokenizer, filename)
    if tokenizer.at(TokenKind.BARE_IDENT):
        text = tokenizer.next().text
        special_float = _parse_special_float(text)
        if special_float is not None:
            return special_float
        if text == "true":
            return True
        if text == "false":
            return False
        return text
    if tokenizer.at(TokenKind.SYMBOL):
        return SymbolName(tokenizer.next().text)
    if tokenizer.at(TokenKind.HASH_ATTR):
        family = (
            parameterized_attr_registry.get(tokenizer.peek().text)
            if parameterized_attr_registry is not None
            else None
        )
        if family is not None:
            return _parse_parameterized_attr_from_tokens(
                tokenizer,
                module,
                filename,
                family,
                scope=scope,
                type_registry=type_registry,
                mode=mode,
                parameterized_attr_registry=parameterized_attr_registry,
                aliases=aliases,
                known_encodings=known_encodings,
                aggregate_nesting_depth=attr_dict_nesting_depth,
            )
        return _parse_static_encoding_from_tokens(
            tokenizer,
            module,
            filename,
            attr_dict_nesting_depth=attr_dict_nesting_depth,
            aliases=aliases if aliases is not None else _CURRENT_ALIASES,
            known_encodings=(
                known_encodings
                if known_encodings is not None
                else _CURRENT_KNOWN_ENCODINGS
            ),
        )
    if tokenizer.at(TokenKind.LBRACKET):
        tokenizer.next()
        values: list[int] = []
        if not tokenizer.at(TokenKind.RBRACKET):
            values.append(int(tokenizer.expect(TokenKind.INTEGER).text))
            while tokenizer.try_consume(TokenKind.COMMA):
                values.append(int(tokenizer.expect(TokenKind.INTEGER).text))
        tokenizer.expect(TokenKind.RBRACKET)
        return values
    if tokenizer.at(TokenKind.LBRACE):
        open_brace_token = tokenizer.next()
        if attr_dict_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
            raise ParseError(
                "aggregate attribute nesting exceeds maximum depth "
                f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}",
                open_brace_token.location,
                filename,
            )
        entries: list[tuple[str, Any]] = []
        seen_keys: set[str] = set()
        while not tokenizer.at(TokenKind.RBRACE):
            key_token = tokenizer.expect(TokenKind.BARE_IDENT)
            if key_token.text in seen_keys:
                raise ParseError(
                    f"duplicate attribute dict key '{key_token.text}'",
                    key_token.location,
                    filename,
                )
            seen_keys.add(key_token.text)
            tokenizer.expect(TokenKind.EQUALS)
            value = _parse_generic_attr_value_from_tokens(
                tokenizer,
                module,
                filename,
                attr_dict_nesting_depth=attr_dict_nesting_depth + 1,
                scope=scope,
                type_registry=type_registry,
                mode=mode,
                parameterized_attr_registry=parameterized_attr_registry,
                aliases=aliases,
                known_encodings=known_encodings,
            )
            entries.append((key_token.text, value))
            tokenizer.try_consume(TokenKind.COMMA)
        tokenizer.expect(TokenKind.RBRACE)
        return CanonicalAttrDict(entries)

    token = tokenizer.peek()
    raise ParseError(
        f"expected attribute value, got {token.kind.name}",
        token.location,
        filename,
    )


def _parse_bytes_attr_value_from_tokens(tokenizer: Tokenizer, filename: str) -> bytes:
    """Parse bytes("001122ff") as raw bytes."""
    bytes_token = tokenizer.expect(TokenKind.BARE_IDENT, "bytes")
    tokenizer.expect(TokenKind.LPAREN)
    hex_text = tokenizer.expect(TokenKind.STRING).text
    tokenizer.expect(TokenKind.RPAREN)
    if len(hex_text) % 2 != 0:
        raise ParseError(
            "bytes attribute hex string must have even length",
            bytes_token.location,
            filename,
        )
    if not all(character in "0123456789abcdefABCDEF" for character in hex_text):
        raise ParseError(
            "bytes attribute hex string must contain only hexadecimal digits",
            bytes_token.location,
            filename,
        )
    return bytes(
        int(hex_text[index : index + 2], 16) for index in range(0, len(hex_text), 2)
    )


def _parse_static_encoding_params_from_tokens(
    tokenizer: Tokenizer,
    module: Module,
    filename: str,
    *,
    attr_dict_nesting_depth: int = 0,
) -> tuple[tuple[str, Any], ...]:
    """Parse '<name = value, ...>' after a static encoding family name."""
    tokenizer.expect(TokenKind.LANGLE)
    entries: list[tuple[str, Any]] = []
    seen_names: set[str] = set()
    if not tokenizer.at(TokenKind.RANGLE):
        while True:
            name_token = tokenizer.expect(TokenKind.BARE_IDENT)
            if name_token.text in seen_names:
                raise ParseError(
                    f"duplicate encoding parameter '{name_token.text}'",
                    name_token.location,
                    filename,
                )
            seen_names.add(name_token.text)
            tokenizer.expect(TokenKind.EQUALS)
            if tokenizer.at(TokenKind.SSA_VALUE):
                raise ParseError(
                    f"static encoding parameter '{name_token.text}' cannot use "
                    "an SSA value; pass dynamic parameters with "
                    "encoding.define #family<static attrs> "
                    "{param = %value : type} : encoding<role>",
                    tokenizer.peek().location,
                    filename,
                )
            value = _parse_generic_attr_value_from_tokens(
                tokenizer,
                module,
                filename,
                attr_dict_nesting_depth=attr_dict_nesting_depth,
            )
            entries.append((name_token.text, value))
            if not tokenizer.try_consume(TokenKind.COMMA):
                break
    tokenizer.expect(TokenKind.RANGLE)
    return tuple(entries)


def _parse_static_encoding_from_tokens(
    tokenizer: Tokenizer,
    module: Module,
    filename: str,
    *,
    attr_dict_nesting_depth: int = 0,
    aliases: dict[str, EncodingInstance] | None = None,
    known_encodings: set[str] | None = None,
) -> EncodingInstance:
    """Parse '#alias', '#family', or '#family<name = value, ...>'."""
    token = tokenizer.expect(TokenKind.HASH_ATTR)
    if aliases is not None:
        aliased = aliases.get(token.text)
        if aliased is not None:
            module.add_encoding(aliased)
            return aliased

    params: tuple[tuple[str, Any], ...] = ()
    if tokenizer.at(TokenKind.LANGLE):
        params = _parse_static_encoding_params_from_tokens(
            tokenizer,
            module,
            filename,
            attr_dict_nesting_depth=attr_dict_nesting_depth,
        )

    if known_encodings is not None and token.text not in known_encodings:
        raise ParseError(
            f"unknown encoding '{token.text}'. "
            f"Known encodings: {sorted(known_encodings)}",
            token.location,
            filename,
        )

    canonical_alias = (
        _CURRENT_CANONICAL_ENCODING_ALIASES.get(token.text)
        if _CURRENT_CANONICAL_ENCODING_ALIASES is not None
        else None
    )
    if canonical_alias is not None:
        family_name, alias = canonical_alias
        authored_parameters = dict(params)
        for parameter_name, _ in alias.fixed_parameters:
            if parameter_name in authored_parameters:
                raise ParseError(
                    f"encoding alias '{alias.name}' fixes parameter "
                    f"'{parameter_name}'; the parameter cannot be restated",
                    token.location,
                    filename,
                )
        expanded_parameters = dict(alias.parameters)
        expanded_parameters.update(authored_parameters)
        instance = EncodingInstance(
            name=family_name,
            params=tuple(expanded_parameters.items()),
        )
    else:
        instance = EncodingInstance(name=token.text, params=params)
    module.add_encoding(instance)
    return instance


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
# Name scope
# ============================================================================


class NameScope:
    """Maps SSA names to value IDs during parsing.

    Parent-chain for nested regions: inner scopes can see outer
    names, outer scopes cannot see inner names after the region ends.
    """

    __slots__ = ("_names", "_parent", "_placeholder_locations")

    def __init__(self, parent: NameScope | None = None) -> None:
        self._names: dict[str, int] = {}
        self._parent = parent
        self._placeholder_locations: dict[str, SourceLocation] = {}

    def define(self, name: str, value_id: int) -> None:
        """Define a new SSA name in this scope."""
        if name in self._names:
            raise ValueError(
                f"SSA name '%{name}' already defined in this scope "
                f"(existing value ID {self._names[name]}, "
                f"new value ID {value_id})"
            )
        self._names[name] = value_id

    def define_placeholder(
        self, name: str, value_id: int, location: SourceLocation
    ) -> None:
        """Define a forward placeholder and retain its original source location."""
        self.define(name, value_id)
        self._placeholder_locations[name] = location

    def placeholder_location(self, name: str) -> SourceLocation | None:
        """Returns the source location of a local forward placeholder."""
        return self._placeholder_locations.get(name)

    def lookup(self, name: str) -> int:
        """Look up an SSA name, searching parent scopes."""
        if name in self._names:
            return self._names[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        raise KeyError(f"undefined SSA value '%{name}'")

    def push(self) -> NameScope:
        """Create a child scope for entering a region."""
        return NameScope(parent=self)


# ============================================================================
# Type parse mode
# ============================================================================


@unique
class TypeParseMode(Enum):
    """Context for how dynamic dim references are resolved."""

    BODY = "body"  # Op body: [%M] looks up existing values.
    SIGNATURE = "signature"  # Function signature: [%M] creates placeholders.


def _parse_enum_attr_value_from_tokens(
    tokenizer: Tokenizer,
    descriptor: AttrDef,
) -> str | int:
    """Parses one descriptor-backed enum keyword or open byte value."""
    assert descriptor.enum_def is not None
    value_by_keyword = {
        enum_case.keyword: enum_case.value for enum_case in descriptor.enum_def.cases
    }
    if tokenizer.at(TokenKind.LANGLE):
        opening = tokenizer.next()
        if not descriptor.open_enum:
            raise ParseError(
                f"enum attribute '{descriptor.name}' is closed and does not "
                "admit raw values",
                opening.location,
                tokenizer._filename,
            )
        integer = tokenizer.expect(TokenKind.INTEGER)
        value = int(integer.text)
        if not 0 <= value <= 0xFF:
            raise ParseError(
                f"enum value {value} is outside the byte domain [0, 255]",
                integer.location,
                tokenizer._filename,
            )
        tokenizer.expect(TokenKind.RANGLE)
        return value

    if tokenizer.at(TokenKind.BARE_IDENT) or tokenizer.at(TokenKind.OP_NAME):
        ident = tokenizer.next()
    else:
        ident = tokenizer.expect(TokenKind.BARE_IDENT)
    if ident.text not in value_by_keyword:
        raise ParseError(
            f"invalid enum value '{ident.text}', expected one of "
            f"{descriptor.enum_def.keywords}",
            ident.location,
            tokenizer._filename,
        )
    return ident.text


def _parse_parameterized_attr_from_tokens(
    tokenizer: Tokenizer,
    module: Module,
    filename: str,
    expected_definition: ParameterizedAttrDef | None,
    *,
    scope: NameScope | None,
    type_registry: dict[str, TypeDef] | None,
    mode: TypeParseMode | None,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None,
    aliases: dict[str, EncodingInstance] | None,
    known_encodings: set[str] | None,
    aggregate_nesting_depth: int,
) -> ParameterizedAttr:
    """Parses one descriptor-backed parameterized attribute value."""
    family_token = tokenizer.expect(TokenKind.HASH_ATTR)
    definition = (
        parameterized_attr_registry.get(family_token.text)
        if parameterized_attr_registry is not None
        else None
    )
    if (
        definition is None
        and expected_definition is not None
        and expected_definition.name == family_token.text
    ):
        definition = expected_definition
    if definition is None:
        raise ParseError(
            f"unknown parameterized attribute family '{family_token.text}'",
            family_token.location,
            filename,
        )
    if expected_definition is not None and definition.name != expected_definition.name:
        raise ParseError(
            f"expected parameterized attribute family "
            f"'{expected_definition.name}', got '{definition.name}'",
            family_token.location,
            filename,
        )

    return _parse_parameterized_attr_parameters_from_tokens(
        tokenizer,
        module,
        filename,
        definition,
        scope=scope,
        type_registry=type_registry,
        mode=mode,
        parameterized_attr_registry=parameterized_attr_registry,
        aliases=aliases,
        known_encodings=known_encodings,
        aggregate_nesting_depth=aggregate_nesting_depth,
    )


def _parse_parameterized_attr_parameters_from_tokens(
    tokenizer: Tokenizer,
    module: Module,
    filename: str,
    definition: ParameterizedAttrDef,
    *,
    scope: NameScope | None,
    type_registry: dict[str, TypeDef] | None,
    mode: TypeParseMode | None,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None,
    aliases: dict[str, EncodingInstance] | None,
    known_encodings: set[str] | None,
    aggregate_nesting_depth: int,
) -> ParameterizedAttr:
    """Parses a known parameterized attribute family's angle payload."""

    opening = tokenizer.expect(TokenKind.LANGLE)
    if aggregate_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
        raise ParseError(
            "aggregate attribute nesting exceeds maximum depth "
            f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}",
            opening.location,
            filename,
        )

    descriptors = {parameter.name: parameter for parameter in definition.parameters}
    parameters: dict[str, Any] = {}
    if not tokenizer.at(TokenKind.RANGLE):
        while True:
            parameter_name: str
            parameter: AttrDef
            is_named = True
            if (
                not parameters
                and definition.primary_parameter is not None
                and not (
                    tokenizer.at(TokenKind.BARE_IDENT)
                    and tokenizer.peek_n(1).kind == TokenKind.EQUALS
                )
            ):
                parameter = definition.primary_parameter
                parameter_name = parameter.name
                is_named = False
                parameter_location = tokenizer.peek().location
            else:
                name_token = tokenizer.expect(TokenKind.BARE_IDENT)
                parameter_name = name_token.text
                parameter_location = name_token.location
                declared_parameter = descriptors.get(parameter_name)
                if declared_parameter is None:
                    raise ParseError(
                        f"unknown parameter '{parameter_name}' for '{definition.name}'",
                        name_token.location,
                        filename,
                    )
                parameter = declared_parameter
            if parameter_name in parameters:
                raise ParseError(
                    f"duplicate parameter '{parameter_name}' for '{definition.name}'",
                    parameter_location,
                    filename,
                )
            if is_named:
                tokenizer.expect(TokenKind.EQUALS)
            value = _parse_descriptor_attr_value_from_tokens(
                tokenizer,
                module,
                filename,
                parameter,
                scope=scope,
                type_registry=type_registry,
                mode=mode,
                parameterized_attr_registry=parameterized_attr_registry,
                aliases=aliases,
                known_encodings=known_encodings,
                attr_dict_nesting_depth=aggregate_nesting_depth + 1,
            )
            if parameter.attr_type == "symbol":
                value = SymbolName(value)
            parameters[parameter_name] = value
            if not tokenizer.try_consume(TokenKind.COMMA):
                break
    closing = tokenizer.expect(TokenKind.RANGLE)
    try:
        return ParameterizedAttr(definition, parameters)
    except (TypeError, ValueError) as error:
        raise ParseError(str(error), closing.location, filename) from error


def _parse_descriptor_attr_value_from_tokens(
    tokenizer: Tokenizer,
    module: Module,
    filename: str,
    descriptor: AttrDef,
    *,
    scope: NameScope | None,
    type_registry: dict[str, TypeDef] | None,
    mode: TypeParseMode | None,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None,
    aliases: dict[str, EncodingInstance] | None,
    known_encodings: set[str] | None,
    attr_dict_nesting_depth: int,
) -> Any:
    """Parses one value according to its descriptor schema."""
    match descriptor.attr_type:
        case "i64":
            return int(tokenizer.expect(TokenKind.INTEGER).text)
        case "f64":
            if tokenizer.at(TokenKind.FLOAT):
                return _parse_float_literal(tokenizer.next().text)
            if tokenizer.at(TokenKind.BARE_IDENT):
                special_float = _parse_special_float(tokenizer.peek().text)
                if special_float is not None:
                    tokenizer.next()
                    return special_float
            return float(tokenizer.expect(TokenKind.INTEGER).text)
        case "string":
            return tokenizer.expect(TokenKind.STRING).text
        case "bool":
            ident = tokenizer.expect(TokenKind.BARE_IDENT)
            if ident.text == "true":
                return True
            if ident.text == "false":
                return False
            raise ParseError(
                f"expected 'true' or 'false', got '{ident.text}'",
                ident.location,
                filename,
            )
        case "enum":
            return _parse_enum_attr_value_from_tokens(tokenizer, descriptor)
        case "enum_array":
            tokenizer.expect(TokenKind.LBRACKET)
            assert descriptor.enum_def is not None
            value_by_keyword = {
                enum_case.keyword: enum_case.value
                for enum_case in descriptor.enum_def.cases
            }
            values: list[int] = []

            def append_value() -> None:
                value = _parse_enum_attr_value_from_tokens(tokenizer, descriptor)
                values.append(
                    value if isinstance(value, int) else value_by_keyword[value]
                )

            if not tokenizer.at(TokenKind.RBRACKET):
                append_value()
                while tokenizer.try_consume(TokenKind.COMMA):
                    if len(values) == 0xFFFF:
                        raise ParseError(
                            "enum-array length exceeds UINT16_MAX",
                            tokenizer.peek().location,
                            filename,
                        )
                    append_value()
            tokenizer.expect(TokenKind.RBRACKET)
            return EnumArrayAttr(values)
        case "signed_enum_set":
            tokenizer.expect(TokenKind.LBRACKET)
            assert descriptor.enum_def is not None
            value_by_keyword = {
                enum_case.keyword: enum_case.value
                for enum_case in descriptor.enum_def.cases
            }
            positive_values: list[int] = []
            negative_values: list[int] = []
            seen_values: set[int] = set()

            def append_value() -> None:
                is_negative = tokenizer.try_consume(TokenKind.MINUS)
                parsed_value = _parse_enum_attr_value_from_tokens(tokenizer, descriptor)
                stable_value = (
                    parsed_value
                    if isinstance(parsed_value, int)
                    else value_by_keyword[parsed_value]
                )
                if stable_value in seen_values:
                    raise ParseError(
                        f"signed enum value {stable_value} appears more than once",
                        tokenizer.peek().location,
                        filename,
                    )
                seen_values.add(stable_value)
                (negative_values if is_negative else positive_values).append(
                    stable_value
                )

            if not tokenizer.at(TokenKind.RBRACKET):
                append_value()
                while tokenizer.try_consume(TokenKind.COMMA):
                    append_value()
            tokenizer.expect(TokenKind.RBRACKET)
            return SignedEnumSetAttr(positive_values, negative_values)
        case "symbol":
            return tokenizer.expect(TokenKind.SYMBOL).text
        case "symbol_array" | "symbol_set":
            collection_name = descriptor.attr_type.replace("_", "-")
            tokenizer.expect(TokenKind.LBRACKET)
            names: list[SymbolName] = []
            seen_names: set[SymbolName] | None = (
                set() if descriptor.attr_type == "symbol_set" else None
            )
            if not tokenizer.at(TokenKind.RBRACKET):
                while True:
                    if len(names) == 0xFFFF:
                        raise ParseError(
                            f"{collection_name} length exceeds UINT16_MAX",
                            tokenizer.peek().location,
                            filename,
                        )
                    token = tokenizer.expect(TokenKind.SYMBOL)
                    name = SymbolName(token.text)
                    if seen_names is not None:
                        if name in seen_names:
                            raise ParseError(
                                f"duplicate symbol name '@{name}' in symbol set",
                                token.location,
                                filename,
                            )
                        seen_names.add(name)
                    names.append(name)
                    if not tokenizer.try_consume(TokenKind.COMMA):
                        break
            tokenizer.expect(TokenKind.RBRACKET)
            if descriptor.attr_type == "symbol_set":
                return SymbolNameSet(names)
            return SymbolNameArray(names)
        case "type":
            if scope is None or type_registry is None:
                raise ValueError("type attribute parsing requires a type context")
            parsed_type, _ = parse_type_from_tokens(
                tokenizer,
                scope,
                module,
                type_registry,
                mode or TypeParseMode.BODY,
                parameterized_attr_registry=parameterized_attr_registry,
            )
            return parsed_type
        case "i64_array":
            tokenizer.expect(TokenKind.LBRACKET)
            values: list[int] = []
            if not tokenizer.at(TokenKind.RBRACKET):
                values.append(int(tokenizer.expect(TokenKind.INTEGER).text))
                while tokenizer.try_consume(TokenKind.COMMA):
                    values.append(int(tokenizer.expect(TokenKind.INTEGER).text))
            tokenizer.expect(TokenKind.RBRACKET)
            return values
        case "bytes":
            return _parse_bytes_attr_value_from_tokens(tokenizer, filename)
        case "encoding":
            return _parse_static_encoding_from_tokens(
                tokenizer,
                module,
                filename,
                aliases=aliases,
                known_encodings=known_encodings,
            )
        case "dict":
            if not tokenizer.at(TokenKind.LBRACE):
                token = tokenizer.peek()
                raise ParseError(
                    f"expected attribute dictionary, got {token.kind.name}",
                    token.location,
                    filename,
                )
            return _parse_generic_attr_value_from_tokens(
                tokenizer,
                module,
                filename,
                attr_dict_nesting_depth=attr_dict_nesting_depth,
                scope=scope,
                type_registry=type_registry,
                mode=mode,
                parameterized_attr_registry=parameterized_attr_registry,
                aliases=aliases,
                known_encodings=known_encodings,
            )
        case "parameterized":
            return _parse_parameterized_attr_from_tokens(
                tokenizer,
                module,
                filename,
                descriptor.parameterized_attr,
                scope=scope,
                type_registry=type_registry,
                mode=mode,
                parameterized_attr_registry=parameterized_attr_registry,
                aliases=aliases,
                known_encodings=known_encodings,
                aggregate_nesting_depth=attr_dict_nesting_depth,
            )
        case "parameterized_array":
            opening = tokenizer.expect(TokenKind.LBRACKET)
            if attr_dict_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
                raise ParseError(
                    "aggregate attribute nesting exceeds maximum depth "
                    f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}",
                    opening.location,
                    filename,
                )
            values: list[ParameterizedAttr] = []
            if not tokenizer.at(TokenKind.RBRACKET):
                while True:
                    if len(values) == 0xFFFF:
                        raise ParseError(
                            "parameterized attribute array length exceeds UINT16_MAX",
                            tokenizer.peek().location,
                            filename,
                        )
                    values.append(
                        _parse_parameterized_attr_from_tokens(
                            tokenizer,
                            module,
                            filename,
                            descriptor.parameterized_attr,
                            scope=scope,
                            type_registry=type_registry,
                            mode=mode,
                            parameterized_attr_registry=parameterized_attr_registry,
                            aliases=aliases,
                            known_encodings=known_encodings,
                            aggregate_nesting_depth=attr_dict_nesting_depth + 1,
                        )
                    )
                    if not tokenizer.try_consume(TokenKind.COMMA):
                        break
            tokenizer.expect(TokenKind.RBRACKET)
            return ParameterizedAttrArray(values)
        case "any":
            return _parse_generic_attr_value_from_tokens(
                tokenizer,
                module,
                filename,
                attr_dict_nesting_depth=attr_dict_nesting_depth,
                scope=scope,
                type_registry=type_registry,
                mode=mode,
                parameterized_attr_registry=parameterized_attr_registry,
                aliases=aliases,
                known_encodings=known_encodings,
            )
        case _:
            return _parse_generic_attr_value_from_tokens(
                tokenizer,
                module,
                filename,
                attr_dict_nesting_depth=attr_dict_nesting_depth,
                scope=scope,
                type_registry=type_registry,
                mode=mode,
                parameterized_attr_registry=parameterized_attr_registry,
                aliases=aliases,
                known_encodings=known_encodings,
            )


# ============================================================================
# Type reference resolution
# ============================================================================


def _resolve_type_value(
    name: str,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
    token: Token,
    filename: str,
) -> int:
    """Resolve an SSA value reference in a type context.

    |name| is the bare name (without '%' sigil). In SIGNATURE mode,
    creates a PlaceholderType value if not found. In BODY mode, raises
    ParseError for undefined names.
    """
    try:
        return scope.lookup(name)
    except KeyError:
        pass

    if mode == TypeParseMode.SIGNATURE:
        value_id = module.add_value(Value(name=name, type=PlaceholderType()))
        scope.define_placeholder(name, value_id, token.location)
        return value_id

    raise ParseError(
        f"undefined SSA value '%{name}'",
        token.location,
        filename,
    )


def _parse_dim_from_tokens(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
    filename: str,
) -> tuple[StaticDim | DynamicDim, int | None]:
    """Parse a single dimension: INTEGER (static) or [SSA_VALUE] (dynamic).

    Returns (dim, binding_value_id). binding_value_id is None for static dims.
    """
    token = tokenizer.peek()
    if token.kind == TokenKind.INTEGER:
        tokenizer.next()
        return StaticDim(int(token.text)), None
    if token.kind == TokenKind.LBRACKET:
        tokenizer.next()  # consume [
        name_token = tokenizer.expect(TokenKind.SSA_VALUE)
        value_id = _resolve_type_value(
            name_token.text, scope, module, mode, name_token, filename
        )
        tokenizer.expect(TokenKind.RBRACKET)
        return DynamicDim(), value_id
    raise ParseError(
        f"expected integer or '[' for dimension, got {token.kind.name} {token.text!r}",
        token.location,
        filename,
    )


def _parse_type_encoding_from_tokens(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
    filename: str,
) -> tuple[EncodingInstance | DynamicEncoding | None, int]:
    """Parse a type encoding after the comma in a shaped type.

    Returns (encoding, encoding_value_id). encoding_value_id is -1
    for static encodings, or the SSA value ID for dynamic (%enc).
    """
    token = tokenizer.peek()
    if token.kind == TokenKind.SSA_VALUE:
        tokenizer.next()
        value_id = _resolve_type_value(token.text, scope, module, mode, token, filename)
        # Verify the referenced value has EncodingType.
        if value_id < len(module.values):
            value = module.values[value_id]
            if not isinstance(value.type, EncodingType | PlaceholderType):
                raise ParseError(
                    f"encoding reference '%{token.text}' has type "
                    f"'{value.type}', expected encoding",
                    token.location,
                    filename,
                )
        return DynamicEncoding(), value_id

    if token.kind == TokenKind.HASH_ATTR:
        instance = _parse_static_encoding_from_tokens(
            tokenizer,
            module,
            filename,
            aliases=_CURRENT_ALIASES,
            known_encodings=_CURRENT_KNOWN_ENCODINGS,
        )
        return instance, -1

    raise ParseError(
        f"expected encoding (%name or #name), got {token.kind.name} {token.text!r}",
        token.location,
        filename,
    )


# ============================================================================
# Type parser — registry-driven
# ============================================================================


def _is_type_start(token: Token, type_registry: dict[str, TypeDef]) -> bool:
    """Check if a token could start a type expression."""
    if token.kind == TokenKind.BARE_IDENT:
        if parse_scalar_type_kind(token.text) is not None:
            return True
        if token.text == "encoding":
            return True
        if token.text == "reg":
            return True
        if token.text in type_registry:
            return True
        return False
    if token.kind == TokenKind.OP_NAME:
        # Dotted type names like hal.buffer and test.ref.
        return True
    if token.kind == TokenKind.LPAREN:
        return True  # Function type.
    return False


def parse_type_from_tokens(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    type_registry: dict[str, TypeDef],
    mode: TypeParseMode = TypeParseMode.BODY,
    *,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None = None,
) -> tuple[Type, dict[int, int]]:
    """Parse a type from the token stream.

    Dispatches through the type registry for structured types.
    Returns (type, dim_bindings).
    """
    token = tokenizer.peek()

    # Scalar type keyword?
    if token.kind == TokenKind.BARE_IDENT:
        scalar_kind = parse_scalar_type_kind(token.text)
        if scalar_kind is not None:
            tokenizer.next()
            return ScalarType(scalar_kind), {}

    # Register type keyword?
    if token.kind == TokenKind.BARE_IDENT and token.text == "reg":
        return _parse_register_type(
            tokenizer,
            scope,
            module,
            type_registry,
            mode,
            parameterized_attr_registry,
        )

    # Registered type (BARE_IDENT like "tile" or OP_NAME like "hal.buffer")?
    if token.kind in (TokenKind.BARE_IDENT, TokenKind.OP_NAME):
        type_def = type_registry.get(token.text)
        if type_def is not None:
            tokenizer.next()
            if type_def.ir_kind == "buffer" and type_def.is_opaque:
                return BUFFER_TYPE, {}
            if type_def.is_opaque:
                return DialectType(type_def.name), {}
            if type_def.omits_empty_parameter_list and not tokenizer.at(
                TokenKind.LANGLE
            ):
                try:
                    return type_def(), {}
                except (TypeError, ValueError) as error:
                    raise ParseError(
                        str(error), token.location, tokenizer._filename
                    ) from error
            tokenizer.expect(TokenKind.LANGLE)
            # Compact shape types parse from the token stream using in_dim_list
            # for 'x' separators. Other types use the interior tokenizer.
            if type_def.uses_compact_shape_format:
                result = _parse_compact_shape_type_from_tokens(
                    type_def,
                    tokenizer,
                    scope,
                    module,
                    mode,
                )
                tokenizer.in_dim_list = False  # cleanup on error paths
                return result
            interior = tokenizer.scan_to_matching_angle_bracket()
            return _parse_type_interior(
                type_def,
                interior,
                scope,
                module,
                type_registry,
                mode,
                token.location,
                tokenizer._filename,
                parameterized_attr_registry,
            )
        if token.kind == TokenKind.OP_NAME:
            tokenizer.next()
            return DialectType(token.text), {}

    # Function type: (types) -> (types)
    if token.kind == TokenKind.LPAREN:
        return _parse_function_type(
            tokenizer,
            scope,
            module,
            type_registry,
            mode,
            parameterized_attr_registry,
        )

    raise ParseError(
        f"expected type, got {token.kind.name} {token.text!r}",
        token.location,
        tokenizer._filename,
    )


def _resolve_register_type(
    register_class_name: str,
    unit_count: int,
    value_type: Type | None,
    location: SourceLocation,
    filename: str,
) -> RegisterType:
    namespace = register_class_name.split(".", 1)[0]
    matches: list[RegisterType] = []
    for registration in DESCRIPTOR_SET_REGISTRATIONS:
        if not registration.key.startswith(f"{namespace}."):
            continue
        descriptor_set = registration.load()
        for register_class_id, register_class in enumerate(descriptor_set.reg_classes):
            if register_class.name == register_class_name:
                matches.append(
                    RegisterType(
                        stable_id_from_string(descriptor_set.key),
                        register_class_id,
                        unit_count,
                        register_class_name,
                        value_type,
                    )
                )
    if not matches:
        raise ParseError(
            "register class is not defined by a registered descriptor set: "
            f"{register_class_name}",
            location,
            filename,
        )
    if len(matches) > 1:
        raise ParseError(
            "register class is ambiguous across descriptor sets: "
            f"{register_class_name}",
            location,
            filename,
        )
    return matches[0]


def _parse_register_type(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    type_registry: dict[str, TypeDef],
    mode: TypeParseMode,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None,
) -> tuple[RegisterType, dict[int, int]]:
    """Parse reg<namespace.class [xN] [: value_type]>."""
    tokenizer.expect(TokenKind.BARE_IDENT, "reg")
    tokenizer.expect(TokenKind.LANGLE)
    class_token = tokenizer.expect(TokenKind.OP_NAME)
    unit_count = 1
    if not tokenizer.at(TokenKind.RANGLE) and not tokenizer.at(TokenKind.COLON):
        suffix_token = tokenizer.expect(TokenKind.BARE_IDENT)
        if suffix_token.text == "x":
            count_token = tokenizer.expect(TokenKind.INTEGER)
            count_text = count_token.text
            location = count_token.location
        elif suffix_token.text.startswith("x") and suffix_token.text[1:].isdigit():
            count_text = suffix_token.text[1:]
            location = suffix_token.location
        else:
            raise ParseError(
                "expected register unit suffix 'xN'",
                suffix_token.location,
                tokenizer._filename,
            )
        unit_count = int(count_text, 10)
        if unit_count < 1:
            raise ParseError(
                "register unit count must be >= 1",
                location,
                tokenizer._filename,
            )
    value_type: Type | None = None
    dim_bindings: dict[int, int] = {}
    if tokenizer.try_consume(TokenKind.COLON) is not None:
        value_type, dim_bindings = parse_type_from_tokens(
            tokenizer,
            scope,
            module,
            type_registry,
            mode,
            parameterized_attr_registry=parameterized_attr_registry,
        )
    tokenizer.expect(TokenKind.RANGLE)
    try:
        return (
            _resolve_register_type(
                class_token.text,
                unit_count,
                value_type,
                class_token.location,
                tokenizer._filename,
            ),
            dim_bindings,
        )
    except ValueError as err:
        raise ParseError(str(err), class_token.location, tokenizer._filename) from err


def parse_type_string(
    text: str,
    type_registry: dict[str, TypeDef] | None = None,
    scope: NameScope | None = None,
    module: Module | None = None,
    mode: TypeParseMode | None = None,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None = None,
) -> tuple[Type, dict[int, int]]:
    """Parse a type from a string. Convenience for testing."""
    tokenizer = Tokenizer(text)
    if scope is None:
        scope = NameScope()
    if module is None:
        module = Module()
    if type_registry is None:
        from loom.builtin_types import ALL_BUILTIN_TYPES

        type_registry = {td.name: td for td in ALL_BUILTIN_TYPES}
    if mode is None:
        mode = TypeParseMode.BODY
    result = parse_type_from_tokens(
        tokenizer,
        scope,
        module,
        type_registry,
        mode,
        parameterized_attr_registry=parameterized_attr_registry,
    )
    tokenizer.expect(TokenKind.EOF)
    return result


def _parse_function_type(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    type_registry: dict[str, TypeDef],
    mode: TypeParseMode,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None,
) -> tuple[FunctionType, dict[int, int]]:
    """Parse (arg_types) -> (result_types)."""
    tokenizer.expect(TokenKind.LPAREN)
    arg_types: list[Type] = []
    if not tokenizer.at(TokenKind.RPAREN):
        arg_type, _ = parse_type_from_tokens(
            tokenizer,
            scope,
            module,
            type_registry,
            mode,
            parameterized_attr_registry=parameterized_attr_registry,
        )
        arg_types.append(arg_type)
        while tokenizer.try_consume(TokenKind.COMMA):
            arg_type, _ = parse_type_from_tokens(
                tokenizer,
                scope,
                module,
                type_registry,
                mode,
                parameterized_attr_registry=parameterized_attr_registry,
            )
            arg_types.append(arg_type)
    tokenizer.expect(TokenKind.RPAREN)
    tokenizer.expect(TokenKind.ARROW)
    tokenizer.expect(TokenKind.LPAREN)
    result_types: list[Type] = []
    if not tokenizer.at(TokenKind.RPAREN):
        result_type, _ = parse_type_from_tokens(
            tokenizer,
            scope,
            module,
            type_registry,
            mode,
            parameterized_attr_registry=parameterized_attr_registry,
        )
        result_types.append(result_type)
        while tokenizer.try_consume(TokenKind.COMMA):
            result_type, _ = parse_type_from_tokens(
                tokenizer,
                scope,
                module,
                type_registry,
                mode,
                parameterized_attr_registry=parameterized_attr_registry,
            )
            result_types.append(result_type)
    tokenizer.expect(TokenKind.RPAREN)
    return FunctionType(tuple(arg_types), tuple(result_types)), {}


# ============================================================================
# Type interior parser — format-driven
# ============================================================================


_KEYWORD_TOKEN_KINDS = {
    ",": TokenKind.COMMA,
    ":": TokenKind.COLON,
    "->": TokenKind.ARROW,
    "=": TokenKind.EQUALS,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    "<": TokenKind.LANGLE,
    ">": TokenKind.RANGLE,
    "x": TokenKind.DIM_X,
}


def _token_is_keyword(token: Token, text: str) -> bool:
    """Returns whether a token matches a keyword spelling."""
    kind = _KEYWORD_TOKEN_KINDS.get(text)
    if kind is not None:
        return token.kind == kind
    return token.kind == TokenKind.BARE_IDENT and token.text == text


def _at_keyword(tokenizer: Tokenizer, text: str) -> bool:
    """Returns whether the tokenizer is positioned at a keyword."""
    return _token_is_keyword(tokenizer.peek(), text)


def _expect_keyword(tokenizer: Tokenizer, text: str) -> None:
    """Consume a keyword token, handling both punctuation and words."""
    kind = _KEYWORD_TOKEN_KINDS.get(text)
    if kind is None:
        tokenizer.expect(TokenKind.BARE_IDENT, text)
        return
    if kind == TokenKind.DIM_X:
        tokenizer.in_dim_list = True
        try:
            tokenizer.expect(kind)
        finally:
            tokenizer.in_dim_list = False
        return
    tokenizer.expect(kind)


def _type_optional_present(
    tokenizer: Tokenizer,
    inner_elements: tuple[FormatElement, ...],
    type_def: TypeDef,
    type_registry: dict[str, TypeDef],
) -> bool:
    """Peek to decide if an OptionalGroup is present in a type interior."""
    if not inner_elements:
        return False
    token_offset = 0
    for element in inner_elements:
        if isinstance(element, Glue):
            continue
        token = tokenizer.peek_n(token_offset)
        match element:
            case Keyword(text=text):
                if not _token_is_keyword(token, text):
                    return False
                token_offset += 1
            case Clause(name=name):
                return (
                    _token_is_keyword(token, name)
                    and tokenizer.peek_n(token_offset + 1).kind == TokenKind.LPAREN
                )
            case TypeOf() | TypesOf():
                return _is_type_start(token, type_registry)
            case Param(field=field):
                descriptor = type_def.param(field)
                assert isinstance(descriptor, AttrDef)
                match descriptor.attr_type:
                    case "i64":
                        return token.kind == TokenKind.INTEGER
                    case "f64":
                        return token.kind in (
                            TokenKind.INTEGER,
                            TokenKind.FLOAT,
                            TokenKind.BARE_IDENT,
                        )
                    case "string":
                        return token.kind == TokenKind.STRING
                    case "enum":
                        return token.kind in (
                            TokenKind.BARE_IDENT,
                            TokenKind.LANGLE,
                        )
                    case (
                        "enum_array"
                        | "i64_array"
                        | "symbol_array"
                        | "symbol_set"
                        | "parameterized_array"
                    ):
                        return token.kind == TokenKind.LBRACKET
                    case "bool":
                        return token.kind == TokenKind.BARE_IDENT
                    case "symbol":
                        return token.kind == TokenKind.SYMBOL
                    case "type":
                        return _is_type_start(token, type_registry)
                    case "bytes":
                        return _token_is_keyword(token, "bytes")
                    case "encoding" | "parameterized":
                        return token.kind == TokenKind.HASH_ATTR
                    case "dict":
                        return token.kind == TokenKind.LBRACE
                    case _:
                        return token.kind != TokenKind.EOF
            case _:
                return token.kind != TokenKind.EOF
    return True


def _parse_type_interior(
    type_def: TypeDef,
    interior: str,
    scope: NameScope,
    module: Module,
    type_registry: dict[str, TypeDef],
    mode: TypeParseMode,
    location: SourceLocation,
    filename: str,
    parameterized_attr_registry: Mapping[str, ParameterizedAttrDef] | None,
) -> tuple[Type, dict[int, int]]:
    """Parse the interior of a parameterized type.

    The type_def's format spec drives the parse for dialect types
    such as test.ref<T>. Built-in shaped types use
    _parse_compact_shape_type_from_tokens directly.
    """
    interior_tokenizer = Tokenizer(interior, filename)
    parsed_params: list[Type] = []
    parsed_attrs: dict[str, Any] = {}

    def walk_type_format(elements: tuple[FormatElement, ...]) -> None:
        """Walk type format elements, consuming tokens."""
        for element in elements:
            match element:
                case TypeOf():
                    param_type, _ = parse_type_from_tokens(
                        interior_tokenizer,
                        scope,
                        module,
                        type_registry,
                        mode,
                        parameterized_attr_registry=parameterized_attr_registry,
                    )
                    parsed_params.append(param_type)
                case TypesOf():
                    # Comma-separated types.
                    if _is_type_start(interior_tokenizer.peek(), type_registry):
                        t, _ = parse_type_from_tokens(
                            interior_tokenizer,
                            scope,
                            module,
                            type_registry,
                            mode,
                            parameterized_attr_registry=parameterized_attr_registry,
                        )
                        parsed_params.append(t)
                        while interior_tokenizer.try_consume(TokenKind.COMMA):
                            t, _ = parse_type_from_tokens(
                                interior_tokenizer,
                                scope,
                                module,
                                type_registry,
                                mode,
                                parameterized_attr_registry=(
                                    parameterized_attr_registry
                                ),
                            )
                            parsed_params.append(t)
                case Attr(field=name):
                    tok = interior_tokenizer.expect(TokenKind.BARE_IDENT)
                    parsed_attrs[name] = tok.text
                case Param(field=name):
                    descriptor = type_def.param(name)
                    assert isinstance(descriptor, AttrDef)
                    value = _parse_descriptor_attr_value_from_tokens(
                        interior_tokenizer,
                        module,
                        filename,
                        descriptor,
                        scope=scope,
                        type_registry=type_registry,
                        mode=mode,
                        parameterized_attr_registry=parameterized_attr_registry,
                        aliases=_CURRENT_ALIASES,
                        known_encodings=_CURRENT_KNOWN_ENCODINGS,
                        attr_dict_nesting_depth=0,
                    )
                    if descriptor.attr_type == "symbol":
                        value = SymbolName(value)
                    parsed_attrs[name] = value
                case SymbolRef(field=name):
                    tok = interior_tokenizer.expect(TokenKind.SYMBOL)
                    parsed_attrs[name] = tok.text
                case Keyword(text=text):
                    _expect_keyword(interior_tokenizer, text)
                case Clause(name=name, elements=inner):
                    _expect_keyword(interior_tokenizer, name)
                    interior_tokenizer.expect(TokenKind.LPAREN)
                    walk_type_format(inner)
                    interior_tokenizer.expect(TokenKind.RPAREN)
                case OptionalGroup(elements=inner, anchor=_anchor):
                    if _type_optional_present(
                        interior_tokenizer, inner, type_def, type_registry
                    ):
                        walk_type_format(inner)
                case Glue():
                    pass
                case Ref(field=name):
                    tok = interior_tokenizer.expect(TokenKind.SSA_VALUE)
                    parsed_attrs[name] = tok.text
                case Refs(field=name):
                    if interior_tokenizer.at(TokenKind.SSA_VALUE):
                        refs = [interior_tokenizer.next().text]
                        while interior_tokenizer.try_consume(TokenKind.COMMA):
                            refs.append(
                                interior_tokenizer.expect(TokenKind.SSA_VALUE).text
                            )
                        parsed_attrs[name] = refs

    walk_type_format(type_def.format)
    interior_tokenizer.expect(TokenKind.EOF)

    if type_def.uses_attribute_parameters:
        try:
            return type_def(**parsed_attrs), {}
        except (TypeError, ValueError) as error:
            raise ParseError(str(error), location, filename) from error

    return DialectType(type_def.name, tuple(parsed_params)), {}


def _parse_compact_shape_type_from_tokens(
    type_def: TypeDef,
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
) -> tuple[ShapedType | PoolType, dict[int, int]]:
    """Parse a shaped type (tile, tensor, vector, view, pool) from the token stream.

    Called after LANGLE has been consumed. Consumes tokens through
    RANGLE. Uses in_dim_list on the tokenizer to handle 'x' as a
    dimension separator.
    """
    filename = tokenizer._filename

    # Pool: single dim, no element type, no encoding.
    if type_def.ir_kind == "pool":
        token = tokenizer.peek()
        if token.kind not in (TokenKind.INTEGER, TokenKind.LBRACKET):
            raise ParseError(
                f"expected integer or '[' for pool dim, "
                f"got {token.kind.name} {token.text!r}",
                token.location,
                filename,
            )
        dim, binding_id = _parse_dim_from_tokens(
            tokenizer, scope, module, mode, filename
        )
        tokenizer.expect(TokenKind.RANGLE)
        dim_bindings: dict[int, int] = {}
        if binding_id is not None:
            dim_bindings[0] = binding_id
        return PoolType(block_size=dim), dim_bindings

    # TypeDef construction validates the compact representation kind.
    type_kind = TypeKind[type_def.ir_kind.upper()]

    # Parse dimensions. in_dim_list must be true from the start so
    # that '0x' in '0xf32' is scanned as INTEGER(0) + DIM_X(x) +
    # BARE_IDENT(f32), not as hex INTEGER(0xf32).
    dims: list[StaticDim | DynamicDim] = []
    dim_bindings = {}
    tokenizer.in_dim_list = True
    token = tokenizer.peek()
    if token.kind in (TokenKind.INTEGER, TokenKind.LBRACKET):
        dim, binding_id = _parse_dim_from_tokens(
            tokenizer, scope, module, mode, filename
        )
        dims.append(dim)
        if binding_id is not None:
            dim_bindings[len(dims) - 1] = binding_id

        while tokenizer.at(TokenKind.DIM_X):
            tokenizer.next()  # consume 'x'
            tokenizer.in_dim_list = False
            token = tokenizer.peek()
            if token.kind not in (TokenKind.INTEGER, TokenKind.LBRACKET):
                break  # element type follows
            tokenizer.in_dim_list = True
            dim, binding_id = _parse_dim_from_tokens(
                tokenizer, scope, module, mode, filename
            )
            dims.append(dim)
            if binding_id is not None:
                dim_bindings[len(dims) - 1] = binding_id
    else:
        # Rank 0 — no dims. Clear in_dim_list before element type.
        tokenizer.in_dim_list = False

    if type_kind == TypeKind.VECTOR and not dims:
        raise ParseError(
            "vector types must have rank >= 1",
            token.location,
            filename,
        )

    # Parse element type (in_dim_list is false here).
    element_token = tokenizer.expect(TokenKind.BARE_IDENT)
    scalar_kind = parse_scalar_type_kind(element_token.text)
    if scalar_kind is None:
        raise ParseError(
            f"unknown element type '{element_token.text}' in shaped type",
            element_token.location,
            filename,
        )
    element_type = ScalarType(scalar_kind)

    # Parse optional encoding.
    encoding: EncodingInstance | DynamicEncoding | None = None
    encoding_binding = -1
    if tokenizer.try_consume(TokenKind.COMMA):
        if len(type_def.params) < 3:
            raise ParseError(
                f"{type_def.name} types must not carry encoding or layout attachments",
                tokenizer.peek().location,
                filename,
            )
        encoding, encoding_binding = _parse_type_encoding_from_tokens(
            tokenizer, scope, module, mode, filename
        )
        if (
            isinstance(encoding, EncodingInstance)
            and _CURRENT_IMPLICIT_SHAPED_ATTACHMENTS is not None
            and encoding.name in _CURRENT_IMPLICIT_SHAPED_ATTACHMENTS
        ):
            encoding = None
    if encoding_binding >= 0:
        dim_bindings[-1] = encoding_binding

    tokenizer.expect(TokenKind.RANGLE)

    shaped = ShapedType(
        type_kind=type_kind,
        element_type=element_type,
        dims=tuple(dims),
        encoding=encoding,
    )
    return shaped, dim_bindings


# ============================================================================
# Parser (placeholder — format walk and structure parsing go here)
# ============================================================================


class ParsedFields:
    """Mutable accumulator filled during the format walk.

    Mirror of ResolvedFields: where ResolvedFields reads from an
    existing Operation, ParsedFields builds one.

    func_arg_ids holds the logical function ABI value IDs created by FuncArgs
    parsing. Body ops consume them as region entry-block pre-args. Declaration
    ops also record them in their declared operand field while retaining this
    logical list for tied-result indexing.

    implicit_values holds parser-created region block args such as loop IVs.
    RegionFmt defines those names in the child scope when the region starts.

    definition_scope_block_args holds region entry args that remain visible to
    adjacent signature metadata inside a Scope until its matching region is
    parsed.
    """

    __slots__ = (
        "operand_ids",
        "result_ids",
        "result_types",
        "result_bindings",
        "attributes",
        "regions",
        "tied_results",
        "implicit_values",
        "definition_scope_block_args",
        "func_arg_ids",
        "func_args_consumed",
        "operand_fields",
    )

    def __init__(self) -> None:
        self.operand_ids: list[int] = []
        self.result_ids: list[int | None] = []
        self.result_types: list[Type] = []
        self.result_bindings: list[dict[int, int]] = []
        self.attributes: dict[str, Any] = {}
        self.regions: list[Region] = []
        self.tied_results: list[IRTiedResult] = []
        self.implicit_values: dict[str, int] = {}
        self.definition_scope_block_args: dict[str, list[int]] = {}
        self.func_arg_ids: list[int] = []
        self.func_args_consumed = False
        self.operand_fields: dict[str, list[int]] = {}


def _func_args_field(op_decl: Op) -> str | None:
    """Return the explicit FuncArgs format field name, if present."""

    def walk(elements: Sequence[FormatElement]) -> str | None:
        for element in elements:
            match element:
                case FuncArgs(field=name):
                    return name
                case (
                    Clause(elements=inner)
                    | OptionalGroup(elements=inner)
                    | Scope(elements=inner)
                ):
                    nested = walk(inner)
                    if nested is not None:
                        return nested
                case _:
                    continue
        return None

    return walk(op_decl.format)


def _func_like_body_field(op_decl: Op) -> str | None:
    """Return the FuncLike body region field for op_decl, if any."""
    for interface in op_decl.interfaces:
        if isinstance(interface, FuncLikeInterface):
            return interface.body
    return None


def _loop_like_interface(op_decl: Op) -> LoopLikeInterface | None:
    """Return the LoopLike interface for op_decl, if any."""
    for interface in op_decl.interfaces:
        if isinstance(interface, LoopLikeInterface):
            return interface
    return None


def _region_def(op_decl: Op, name: str) -> Any | None:
    """Return the declared region matching name."""
    for region in op_decl.regions:
        if region.name == name:
            return region
    return None


class Parser:
    """Format-element-driven parser for loom IR text format.

    Usage:
        parser = Parser()
        parser.register_ops(ALL_TEST_OPS)
        parser.register_types(ALL_BUILTIN_TYPES)
        module = parser.parse(source_text)
    """

    def __init__(self) -> None:
        self._op_registry: dict[str, Op] = {}
        self._type_registry: dict[str, TypeDef] = {}
        self._parameterized_attr_registry: dict[str, ParameterizedAttrDef] = {}
        self._layouts: dict[str, FieldLayout] = {}
        self._scope: NameScope = NameScope()
        self._module: Module = Module()
        self._tokenizer: Tokenizer = Tokenizer("")
        self._implicit_source_id: int | None = None
        self._encoding_aliases: dict[str, EncodingInstance] = {}
        self._canonical_encoding_aliases: dict[str, tuple[str, EncodingAliasDef]] = {}
        self._known_encodings: set[str] = set()
        self._implicit_shaped_attachments: set[str] = set()
        self._reserved_result_names: list[str] = []
        self._reserved_result_ids: list[int] = []
        self._definition_scope_active: bool = False

    def register_ops(self, ops: Sequence[Op]) -> None:
        """Register op declarations."""
        for op in ops:
            self._op_registry[op.name] = op

    def register_types(self, types: Sequence[TypeDef]) -> None:
        """Register type declarations."""
        for td in types:
            self._type_registry[td.name] = td

    def register_parameterized_attrs(
        self, definitions: Sequence[ParameterizedAttrDef]
    ) -> None:
        """Register descriptor-backed parameterized attribute families."""
        for definition in definitions:
            self._parameterized_attr_registry[definition.name] = definition

    def _parse_type(
        self, tokenizer: Tokenizer, scope: NameScope, mode: TypeParseMode
    ) -> tuple[Type, dict[int, int]]:
        """Parses a type with every registry owned by this parser."""

        return parse_type_from_tokens(
            tokenizer,
            scope,
            self._module,
            self._type_registry,
            mode,
            parameterized_attr_registry=self._parameterized_attr_registry,
        )

    def register_encodings(self, names: Sequence[str]) -> None:
        """Register known encoding names for validation."""
        self._known_encodings.update(names)

    def register_encoding_families(self, families: Sequence[EncodingFamilyDef]) -> None:
        """Register encoding declarations and their generated type semantics."""
        for family in families:
            self._known_encodings.add(family.name)
            if family.implicit_shaped_attachment:
                self._implicit_shaped_attachments.add(family.name)
            for alias in family.aliases:
                self._known_encodings.add(alias.name)
                self._canonical_encoding_aliases[alias.name] = (
                    family.name,
                    alias,
                )

    # --- Top-level parsing ---

    def parse(
        self,
        source: str,
        filename: str = "<input>",
        *,
        verify: bool = False,
    ) -> Module:
        """Parse a complete .loom file into a Module.

        Handles:
          - Attribute aliases: #enc = #encoding.operand<element_format=i8,
            payload_elements=32, payload_packing=dense_lanes>
          - Function definitions and declarations
          - Top-level dispatch
        """
        global _CURRENT_ALIASES, _CURRENT_KNOWN_ENCODINGS
        global _CURRENT_CANONICAL_ENCODING_ALIASES
        global _CURRENT_IMPLICIT_SHAPED_ATTACHMENTS
        self._tokenizer = Tokenizer(source, filename)
        self._module = Module()
        self._implicit_source_id = (
            self._find_or_add_source(filename) if filename else None
        )
        self._scope = NameScope()
        self._encoding_aliases = {}
        _CURRENT_ALIASES = self._encoding_aliases
        _CURRENT_CANONICAL_ENCODING_ALIASES = self._canonical_encoding_aliases
        _CURRENT_KNOWN_ENCODINGS = (
            self._known_encodings if self._known_encodings else None
        )
        _CURRENT_IMPLICIT_SHAPED_ATTACHMENTS = (
            self._implicit_shaped_attachments
            if self._implicit_shaped_attachments
            else None
        )
        tok = self._tokenizer
        tok.peek()
        self._module.file_header = tuple(tok.take_file_header())

        while not tok.at(TokenKind.EOF):
            # Attribute alias: #name = ...
            if tok.at(TokenKind.HASH_ATTR):
                tok.take_pending_source_trivia()
                self._parse_attribute_alias()
                continue

            # Top-level symbol definition or module-scope operation.
            if tok.at(TokenKind.OP_NAME):
                op = self._parse_operation()
                self._register_top_level_operation(op)
                continue

            raise ParseError(
                f"expected attribute alias or top-level op, "
                f"got {tok.peek().kind.name} {tok.peek().text!r}",
                tok.peek().location,
                tok._filename,
            )

        _CURRENT_ALIASES = None
        _CURRENT_CANONICAL_ENCODING_ALIASES = None
        _CURRENT_KNOWN_ENCODINGS = None
        _CURRENT_IMPLICIT_SHAPED_ATTACHMENTS = None
        rebuild_value_metadata(self._module)
        if verify:
            from loom.verify import verify_module

            diagnostics = verify_module(
                self._module,
                ops=self._op_registry.values(),
            )
            diagnostics.raise_if_errors()
        return self._module

    def _parse_attribute_alias(self) -> None:
        """Parse #alias = #encoding<params> at file level."""
        tok = self._tokenizer
        alias_tok = tok.expect(TokenKind.HASH_ATTR)
        tok.expect(TokenKind.EQUALS)
        if (
            alias_tok.text in self._known_encodings
            or alias_tok.text in self._parameterized_attr_registry
        ):
            raise ParseError(
                "invalid encoding alias definition: "
                "alias name shadows a registered attribute family",
                alias_tok.location,
                tok._filename,
            )
        if alias_tok.text in self._encoding_aliases:
            raise ParseError(
                "invalid encoding alias definition: duplicate encoding alias name",
                alias_tok.location,
                tok._filename,
            )
        spec = _parse_static_encoding_from_tokens(
            tok,
            self._module,
            tok._filename,
            aliases=self._encoding_aliases,
            known_encodings=(self._known_encodings if self._known_encodings else None),
        )
        instance = EncodingInstance(
            name=spec.name,
            alias=alias_tok.text,
            params=spec.params,
        )
        self._encoding_aliases[alias_tok.text] = instance
        self._module.add_encoding(instance)

    def _register_top_level_operation(self, op: Operation) -> None:
        """Attach a parsed operation to the module body and applicable index."""
        op_decl = self._op_registry.get(op.name)
        if op_decl is not None and op_decl.has_trait("SymbolDefine"):
            self._module.add_symbol(symbol_from_operation(op, op_decl))
            return
        if op_decl is not None and op_decl.has_trait("ModuleScope"):
            self._module.add_top_level_operation(op)
            return

        location = SourceLocation(1, 1, 0)
        op_location = self._module.locations.get(op.location_id)
        if isinstance(op_location, FileLocation):
            location = SourceLocation(op_location.start_line, op_location.start_col, 0)
        raise ParseError(
            f"op '{op.name}' is not permitted at module scope",
            location,
            self._tokenizer._filename,
        )

    def _parse_func_arg(self) -> tuple[str, Type, int]:
        """Parse one function argument: %name: type.

        Returns (name, type, value_id).
        """
        tok = self._tokenizer
        name_tok = tok.expect(TokenKind.SSA_VALUE)
        tok.expect(TokenKind.COLON)
        arg_type, all_bindings = self._parse_type(
            tok, self._scope, TypeParseMode.SIGNATURE
        )
        # Extract bindings: dim_bindings are non-negative keys,
        # encoding_binding uses sentinel key -1.
        dim_bindings = {k: v for k, v in all_bindings.items() if k >= 0}
        encoding_binding = all_bindings.get(-1, -1)

        # If the name was already forward-referenced in another argument's
        # type, update the placeholder value.
        try:
            value_id = self._scope.lookup(name_tok.text)
            value = self._module.values[value_id]
            if not isinstance(value.type, PlaceholderType):
                raise ParseError(
                    f"SSA name '%{name_tok.text}' already defined",
                    name_tok.location,
                    tok._filename,
                )
            value.type = arg_type
            value.dim_bindings = dim_bindings
            value.encoding_binding = encoding_binding
        except KeyError:
            # First occurrence: define the argument value in scope.
            value_id = self._module.add_value(
                Value(
                    name=name_tok.text,
                    type=arg_type,
                    dim_bindings=dim_bindings,
                    encoding_binding=encoding_binding,
                )
            )
            self._scope.define(name_tok.text, value_id)
        return name_tok.text, arg_type, value_id

    def _layout(self, op_decl: Op) -> FieldLayout:
        """Get or compute the field layout for an op kind."""
        layout = self._layouts.get(op_decl.name)
        if layout is None:
            layout = compute_layout(op_decl)
            self._layouts[op_decl.name] = layout
        return layout

    def _format_element_covers_attr(
        self, element: FormatElement, attr_name: str, skip: FormatElement
    ) -> bool:
        """Returns whether a format element owns an attr's surface spelling."""
        if element is skip:
            return False
        match element:
            case Attr(field=name) | SymbolRef(field=name) | KeyRef(field=name):
                return name == attr_name
            case (
                AttrParams(field=name)
                | TemplateParam(field=name)
                | PredicateList(field=name)
            ):
                return name == attr_name
            case TemplateParamFlags(param=param_name, flags=flags_name):
                return param_name == attr_name or flags_name == attr_name
            case ScopedEnumRef(field=name):
                return name == attr_name
            case StableKeyRef(key=key, stable_id=stable_id):
                return key == attr_name or stable_id == attr_name
            case IndexList(static=name):
                return name == attr_name
            case AlignedRefs(alignments=name):
                return name == attr_name
            case FuncArgs(start_attr=start_attr, end_attr=end_attr):
                return start_attr == attr_name or end_attr == attr_name
            case (
                OperandDict(names=name) | AttrTable(keys=name) | RegionTable(keys=name)
            ):
                return name == attr_name
            case AttrDict(field=name):
                return bool(name) and name == attr_name
            case (
                Clause(elements=inner)
                | OptionalGroup(elements=inner)
                | Scope(elements=inner)
            ):
                return self._format_elements_cover_attr(inner, attr_name, skip)
            case _:
                return False

    def _format_elements_cover_attr(
        self,
        elements: Sequence[FormatElement],
        attr_name: str,
        skip: FormatElement,
    ) -> bool:
        return any(
            self._format_element_covers_attr(element, attr_name, skip)
            for element in elements
        )

    def _apply_elided_attr_defaults(
        self, op_decl: Op, inline_dict: AttrDict, parsed: ParsedFields
    ) -> None:
        """Restores required attrs omitted from an inline AttrDict."""
        for attr_def in op_decl.attrs:
            if not attr_def.elide_default or attr_def.name in parsed.attributes:
                continue
            if self._format_elements_cover_attr(
                op_decl.format, attr_def.name, inline_dict
            ):
                continue
            parsed.attributes[attr_def.name] = attr_def.default

    def _record_operand_id(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        field_name: str,
        value_id: int,
    ) -> None:
        parsed.operand_ids.append(value_id)
        self._record_operand_ids(parsed, op_decl, field_name, [value_id])

    def _record_operand_ids(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        field_name: str,
        value_ids: Sequence[int],
    ) -> None:
        field_desc = self._layout(op_decl).fields.get(field_name)
        if field_desc is not None and field_desc.kind == FieldKind.OPERAND:
            parsed.operand_fields.setdefault(field_name, []).extend(value_ids)

    def _canonical_operand_ids(self, op_decl: Op, parsed: ParsedFields) -> list[int]:
        if not parsed.operand_fields:
            return parsed.operand_ids
        operand_ids: list[int] = []
        for operand in op_decl.operands:
            field_values = parsed.operand_fields.get(operand.name)
            if not field_values:
                if operand.optional or operand.variadic:
                    continue
                return parsed.operand_ids
            if operand.variadic:
                operand_ids.extend(field_values)
            else:
                operand_ids.append(field_values[0])
        return operand_ids

    def _operand_segment_counts(
        self, op_decl: Op, parsed: ParsedFields
    ) -> tuple[int, ...]:
        layout = self._layout(op_decl)
        if not layout.segmented_operands:
            return ()
        return tuple(
            len(parsed.operand_fields.get(operand.name, ()))
            for operand in op_decl.operands
        )

    def _known_value_type(self, value_id: int) -> Type | None:
        value_type = self._module.values[value_id].type
        if value_type == NONE_TYPE or isinstance(value_type, PlaceholderType):
            return None
        return value_type

    def _check_operand_type_annotation(
        self,
        parsed: ParsedFields,
        field_name: str,
        annotated_type: Type,
        annotation_token: Token,
        *,
        element_index: int = 0,
        include_element_index: bool = False,
    ) -> None:
        value_ids = parsed.operand_fields.get(field_name, ())
        if element_index >= len(value_ids):
            raise ParseError(
                f"operand '{field_name}' has no value for type annotation",
                annotation_token.location,
                self._tokenizer._filename,
            )
        actual_type = self._module.values[value_ids[element_index]].type
        if actual_type == annotated_type:
            return
        display_name = (
            f"{field_name}[{element_index}]" if include_element_index else field_name
        )
        raise ParseError(
            f"operand '{display_name}' has type {actual_type}, "
            f"but type annotation is {annotated_type}",
            annotation_token.location,
            self._tokenizer._filename,
        )

    def _field_value_type(
        self,
        op_decl: Op,
        parsed: ParsedFields,
        field_name: str,
        reserved_result_ids: Sequence[int],
    ) -> Type | None:
        for operand in op_decl.operands:
            if operand.name != field_name:
                continue
            for value_id in parsed.operand_fields.get(field_name, ()):
                value_type = self._known_value_type(value_id)
                if value_type is not None:
                    return value_type
            return None

        for result_index, result in enumerate(op_decl.results):
            if result.name != field_name or result_index >= len(reserved_result_ids):
                continue
            return self._known_value_type(reserved_result_ids[result_index])
        return None

    def _infer_same_type_result(
        self,
        op_decl: Op,
        parsed: ParsedFields,
        result_index: int,
        reserved_result_ids: Sequence[int],
    ) -> Type | None:
        if result_index >= len(op_decl.results):
            return None
        result_name = op_decl.results[result_index].name
        for constraint in op_decl.constraints:
            if constraint.name != "SameType" or result_name not in constraint.args:
                continue
            for field_name in constraint.args:
                if field_name == result_name:
                    continue
                value_type = self._field_value_type(
                    op_decl, parsed, field_name, reserved_result_ids
                )
                if value_type is not None:
                    return value_type
        return None

    def _scope_allows_symbolic_type_values(
        self, op_decl: Op, inner_elements: tuple[FormatElement, ...]
    ) -> bool:
        """Returns true for declaration scopes that own symbolic type names."""
        return op_decl.has_trait("SymbolDefine") and not any(
            isinstance(element, FuncArgs) for element in inner_elements
        )

    def _assign_reserved_binding_types(self, bindings: Mapping[int, int]) -> None:
        reserved_result_ids = set(self._reserved_result_ids)
        for binding_position, value_id in bindings.items():
            if value_id not in reserved_result_ids:
                continue
            value = self._module.values[value_id]
            if binding_position == -1:
                value.type = ENCODING_TYPE
            elif binding_position >= 0:
                value.type = INDEX

    def _assign_symbolic_binding_types(self, bindings: Mapping[int, int]) -> None:
        for binding_position, value_id in bindings.items():
            value = self._module.values[value_id]
            if not isinstance(value.type, PlaceholderType):
                continue
            if binding_position == -1:
                value.type = ENCODING_TYPE
            elif binding_position >= 0:
                value.type = INDEX

    # --- Op parsing ---

    def parse_operation_from_text(
        self,
        text: str,
        module: Module | None = None,
        scope: NameScope | None = None,
    ) -> Operation:
        """Parse a single operation from text. Convenience for testing."""
        self._tokenizer = Tokenizer(text)
        self._module = module if module is not None else Module()
        self._implicit_source_id = self._find_or_add_source(self._tokenizer._filename)
        self._scope = scope if scope is not None else NameScope()
        op = self._parse_operation()
        op_decl = self._op_registry.get(op.name)
        operand_def_count = (
            len(op.operands)
            if (
                op_decl is not None
                and op_decl.has_trait("SymbolDefine")
                and not op.regions
            )
            else 0
        )
        record_operation_value_metadata(
            self._module,
            op,
            block_index=VALUE_DEF_BLOCK_NONE,
            op_index=0,
            operand_def_count=operand_def_count,
        )
        return op

    def _parse_operation(self) -> Operation:
        """Parse one operation from the token stream.

        Handles the result list, op name lookup, format walk,
        and Operation construction.
        """
        tok = self._tokenizer
        start_token = tok.peek()
        pending_comments, leading_blank_line = tok.take_pending_source_trivia()
        comments = tuple(pending_comments)
        start_loc = start_token.location

        # 1. Result list: %r = or %a, %b =
        result_names: list[str] = []
        if tok.at(TokenKind.SSA_VALUE):
            result_names.append(tok.next().text)
            while tok.try_consume(TokenKind.COMMA):
                result_names.append(tok.expect(TokenKind.SSA_VALUE).text)
            tok.expect(TokenKind.EQUALS)

        # 2. Op name.
        op_name_tok = tok.expect(TokenKind.OP_NAME)
        op_name = op_name_tok.text
        op_decl = self._op_registry.get(op_name)
        if op_decl is None:
            raise ParseError(
                f"unknown op '{op_name}'", op_name_tok.location, tok._filename
            )

        # 3. Pre-allocate result values so result type annotations can
        # reference them (e.g., tile<[%m]x[%k]xf32> where %m and %k
        # are co-results). The values start with NoneType and get
        # their real types assigned after the format walk. They are
        # NOT in the main scope — _parse_result_type pushes them
        # into a child scope only during result type parsing, so
        # operand/argument parsing cannot see them.
        parsed = ParsedFields()
        reserved_result_ids: list[int] = []
        if result_names:
            for name in result_names:
                value_id = self._module.add_value(Value(name=name, type=NONE_TYPE))
                reserved_result_ids.append(value_id)
        self._reserved_result_names = result_names
        self._reserved_result_ids = reserved_result_ids

        # 4. Walk format spec.
        self._walk_format(op_decl.format, op_decl, parsed)
        self._reserved_result_names = []
        self._reserved_result_ids = []
        parsed.operand_ids = self._canonical_operand_ids(op_decl, parsed)
        operand_segment_counts = self._operand_segment_counts(op_decl, parsed)

        # 5. Assign real types to pre-allocated result values and
        # define them in scope.
        if result_names:
            for i, value_id in enumerate(reserved_result_ids):
                value = self._module.values[value_id]
                if i < len(parsed.result_types):
                    bindings = (
                        parsed.result_bindings[i]
                        if i < len(parsed.result_bindings)
                        else {}
                    )
                    dim_bindings = {k: v for k, v in bindings.items() if k >= 0}
                    encoding_binding = bindings.get(-1, -1)
                    value.type = parsed.result_types[i]
                    value.dim_bindings = dim_bindings
                    value.encoding_binding = encoding_binding
                else:
                    result_decl = (
                        op_decl.results[i] if i < len(op_decl.results) else None
                    )
                    inferred_type = self._infer_same_type_result(
                        op_decl, parsed, i, reserved_result_ids
                    )
                    if inferred_type is None:
                        inferred_type = (
                            _concrete_type_for_constraint(result_decl.type_constraint)
                            if result_decl is not None and not result_decl.variadic
                            else None
                        )
                    if inferred_type is not None:
                        value.type = inferred_type
                    elif isinstance(value.type, PlaceholderType):
                        value.type = NONE_TYPE
                if i < len(parsed.result_ids):
                    parsed.result_ids[i] = value_id
                else:
                    parsed.result_ids.append(value_id)
                self._scope.define(result_names[i], value_id)
        elif parsed.result_types:
            # Symbol-defining op: no LHS names, create anonymous result values
            # for any that weren't already named in the signature.
            for i, result_type in enumerate(parsed.result_types):
                if i < len(parsed.result_ids) and parsed.result_ids[i] is not None:
                    continue

                bindings = (
                    parsed.result_bindings[i] if i < len(parsed.result_bindings) else {}
                )
                dim_bindings = {k: v for k, v in bindings.items() if k >= 0}
                encoding_binding = bindings.get(-1, -1)
                value_id = self._module.add_value(
                    Value(
                        name="",
                        type=result_type,
                        dim_bindings=dim_bindings,
                        encoding_binding=encoding_binding,
                    )
                )
                if i < len(parsed.result_ids):
                    parsed.result_ids[i] = value_id
                else:
                    parsed.result_ids.append(value_id)

        # 5. Build Operation.
        # Default location: implicit source position from tokenizer.
        end_loc = tok.current_location()
        location_id = self._add_implicit_location(start_loc, end_loc)

        # Explicit location annotation overrides the implicit one.
        if tok.at(TokenKind.BARE_IDENT) and tok.peek().text == "loc":
            location_id = self._parse_location_annotation()

        # All result slots must be resolved to concrete value IDs by now.
        result_ids: list[int] = []
        for rid in parsed.result_ids:
            if rid is None:
                raise ParseError(
                    "internal error: unresolved result slot",
                    end_loc,
                    tok._filename,
                )
            result_ids.append(rid)

        op = Operation(
            name=op_name,
            operands=parsed.operand_ids,
            operand_segment_counts=operand_segment_counts,
            results=result_ids,
            tied_results=parsed.tied_results,
            attributes=parsed.attributes,
            regions=parsed.regions,
            location_id=location_id,
            comments=comments,
            leading_blank_line=leading_blank_line,
        )
        self._validate_operation(op, start_loc)
        return op

    def _validate_operation(self, op: Operation, location: SourceLocation) -> None:
        """Apply parse-time checks that depend on multiple parsed fields."""
        if op.name != "encoding.define":
            return
        spec = op.attributes.get("spec")
        param_names = op.attributes.get("param_names")
        if not isinstance(spec, EncodingInstance) or not isinstance(
            param_names, Mapping
        ):
            return
        static_names = {name for name, _value in spec.params}
        for dynamic_name in param_names:
            if dynamic_name in static_names:
                raise ParseError(
                    "encoding.define parameter "
                    f"'{dynamic_name}' is both static and dynamic",
                    location,
                    self._tokenizer._filename,
                )

    # --- Location annotation parsing ---

    def _parse_location_annotation(self) -> int:
        """Parse loc(...) annotation and return the location ID.

        Syntax:
          loc("source":start_line:start_col to end_line:end_col)  — FILE
          loc(fused<child, child, ...>)                           — FUSED
          loc(opaque<"tag", "data">)                              — OPAQUE
        """
        tok = self._tokenizer
        tok.expect(TokenKind.BARE_IDENT, "loc")
        tok.expect(TokenKind.LPAREN)

        if tok.at(TokenKind.STRING):
            # FILE location: "source":start_line:start_col to end_line:end_col
            location_id = self._parse_file_location()
        elif tok.at(TokenKind.BARE_IDENT) and tok.peek().text == "fused":
            # FUSED location: fused<child, child, ...>
            location_id = self._parse_fused_location()
        elif tok.at(TokenKind.BARE_IDENT) and tok.peek().text == "opaque":
            # OPAQUE location: opaque<"tag", "data">
            location_id = self._parse_opaque_location()
        elif tok.at(TokenKind.BARE_IDENT) and tok.peek().text == "tagged":
            # TAGGED location: tagged<tag, "hex", child>
            location_id = self._parse_tagged_location()
        else:
            raise ParseError(
                f"expected location kind (string, 'fused', 'opaque', or 'tagged'), "
                f"got {tok.peek().kind.name} {tok.peek().text!r}",
                tok.peek().location,
                tok._filename,
            )

        tok.expect(TokenKind.RPAREN)
        return location_id

    def _find_or_add_source(self, name: str) -> int:
        """Find or add a source name in the module's source table."""
        for i, existing in enumerate(self._module.sources):
            if existing == name:
                return i
        source_id = len(self._module.sources)
        self._module.sources.append(name)
        return source_id

    def _add_implicit_location(self, start: SourceLocation, end: SourceLocation) -> int:
        """Adds a source-backed implicit location, or returns unknown."""
        if self._implicit_source_id is None:
            return 0
        return self._module.add_location(
            FileLocation(
                source_id=self._implicit_source_id,
                start_line=start.line,
                start_col=start.column,
                end_line=end.line,
                end_col=end.column,
            )
        )

    def _parse_file_location(self) -> int:
        """Parse "source":start_line:start_col to end_line:end_col.

        Called after loc( has been consumed; the closing ) is consumed
        by the caller.
        """
        tok = self._tokenizer
        source_name = tok.expect(TokenKind.STRING).text
        source_id = self._find_or_add_source(source_name)

        tok.expect(TokenKind.COLON)
        start_line = int(tok.expect(TokenKind.INTEGER).text)
        tok.expect(TokenKind.COLON)
        start_col = int(tok.expect(TokenKind.INTEGER).text)

        _expect_keyword(tok, "to")

        end_line = int(tok.expect(TokenKind.INTEGER).text)
        tok.expect(TokenKind.COLON)
        end_col = int(tok.expect(TokenKind.INTEGER).text)

        return self._module.add_location(
            FileLocation(
                source_id=source_id,
                start_line=start_line,
                start_col=start_col,
                end_line=end_line,
                end_col=end_col,
            )
        )

    def _parse_fused_location(self) -> int:
        """Parse fused<child, child, ...>.

        Each child is a FILE location: "source":line:col.
        Called after loc( has been consumed.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.BARE_IDENT, "fused")
        tok.expect(TokenKind.LANGLE)

        children: list[int] = []
        if not tok.at(TokenKind.RANGLE):
            children.append(self._parse_fused_child())
            while tok.try_consume(TokenKind.COMMA):
                children.append(self._parse_fused_child())

        tok.expect(TokenKind.RANGLE)
        return self._module.add_location(FusedLocation(children=tuple(children)))

    def _parse_fused_child(self) -> int:
        """Parse one fused location child: "source":line:col."""
        tok = self._tokenizer
        source_name = tok.expect(TokenKind.STRING).text
        source_id = self._find_or_add_source(source_name)

        tok.expect(TokenKind.COLON)
        line = int(tok.expect(TokenKind.INTEGER).text)
        tok.expect(TokenKind.COLON)
        col = int(tok.expect(TokenKind.INTEGER).text)

        # Fused children are stored as FILE locations (start = end).
        return self._module.add_location(
            FileLocation(
                source_id=source_id,
                start_line=line,
                start_col=col,
                end_line=line,
                end_col=col,
            )
        )

    def _parse_opaque_location(self) -> int:
        """Parse opaque<"tag", "data">.

        Called after loc( has been consumed.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.BARE_IDENT, "opaque")
        tok.expect(TokenKind.LANGLE)

        tag = tok.expect(TokenKind.STRING).text
        source_id = self._find_or_add_source(tag)

        tok.expect(TokenKind.COMMA)

        data = tok.expect(TokenKind.STRING).text.encode("utf-8")

        tok.expect(TokenKind.RANGLE)
        return self._module.add_location(OpaqueLocation(source_id=source_id, data=data))

    def _parse_tagged_location(self) -> int:
        """Parse tagged<tag, "hex", child>.

        The payload string is hexadecimal text that decodes to arbitrary bytes.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.BARE_IDENT, "tagged")
        tok.expect(TokenKind.LANGLE)

        tag_token = tok.peek()
        if tag_token.kind == TokenKind.BARE_IDENT:
            tag_name = tok.expect(TokenKind.BARE_IDENT).text
            tag = parse_builtin_location_tag(tag_name)
            if tag is None:
                raise ParseError(
                    f"unknown tagged location tag {tag_name!r}",
                    tag_token.location,
                    tok._filename,
                )
        else:
            tag = int(tok.expect(TokenKind.INTEGER).text)
            if tag <= 0 or tag > 0xFFFF:
                raise ParseError(
                    "tagged location tag must be in [1, 65535]",
                    tag_token.location,
                    tok._filename,
                )

        tok.expect(TokenKind.COMMA)
        payload_token = tok.expect(TokenKind.STRING)
        payload_hex = payload_token.text
        if len(payload_hex) % 2:
            raise ParseError(
                "tagged location payload hex length must be even",
                payload_token.location,
                tok._filename,
            )
        if any(c not in "0123456789abcdefABCDEF" for c in payload_hex):
            raise ParseError(
                "tagged location payload must contain only hex digits",
                payload_token.location,
                tok._filename,
            )
        data = bytes.fromhex(payload_hex)

        child = 0
        if tok.try_consume(TokenKind.COMMA):
            child = self._parse_fused_child()

        tok.expect(TokenKind.RANGLE)
        return self._module.add_location(
            TaggedLocation(tag=tag, child=child, data=data)
        )

    def _walk_format(
        self,
        elements: tuple[FormatElement, ...],
        op_decl: Op,
        parsed: ParsedFields,
    ) -> None:
        """Walk format elements, consuming tokens into ParsedFields."""
        tok = self._tokenizer
        for element in elements:
            match element:
                case Ref(field=name):
                    if name in ("iv",):
                        # Implicit region argument: create the value now, but
                        # defer its range-derived type and name until RegionFmt
                        # pushes the child scope.
                        ssa_tok = tok.expect(TokenKind.SSA_VALUE)
                        value_id = self._module.add_value(
                            Value(name=ssa_tok.text, type=NONE_TYPE)
                        )
                        parsed.implicit_values[ssa_tok.text] = value_id
                    else:
                        ssa_tok = tok.expect(TokenKind.SSA_VALUE)
                        try:
                            value_id = self._scope.lookup(ssa_tok.text)
                        except KeyError:
                            raise ParseError(
                                f"undefined SSA value '%{ssa_tok.text}'",
                                ssa_tok.location,
                                tok._filename,
                            ) from None
                        self._record_operand_id(parsed, op_decl, name, value_id)

                case Refs(field=name):
                    value_ids: list[int] = []
                    if tok.at(TokenKind.SSA_VALUE):
                        ssa_tok = tok.next()
                        try:
                            value_id = self._scope.lookup(ssa_tok.text)
                        except KeyError:
                            raise ParseError(
                                f"undefined SSA value '%{ssa_tok.text}'",
                                ssa_tok.location,
                                tok._filename,
                            ) from None
                        value_ids.append(value_id)
                        while tok.try_consume(TokenKind.COMMA):
                            if not tok.at(TokenKind.SSA_VALUE):
                                break
                            ssa_tok = tok.next()
                            try:
                                value_id = self._scope.lookup(ssa_tok.text)
                            except KeyError:
                                raise ParseError(
                                    f"undefined SSA value '%{ssa_tok.text}'",
                                    ssa_tok.location,
                                    tok._filename,
                                ) from None
                            value_ids.append(value_id)
                    if value_ids:
                        parsed.operand_ids.extend(value_ids)
                        self._record_operand_ids(parsed, op_decl, name, value_ids)

                case TypedRefs(field=name):
                    value_ids = []
                    while tok.at(TokenKind.SSA_VALUE):
                        ssa_tok = tok.next()
                        try:
                            value_id = self._scope.lookup(ssa_tok.text)
                        except KeyError:
                            raise ParseError(
                                f"undefined SSA value '%{ssa_tok.text}'",
                                ssa_tok.location,
                                tok._filename,
                            ) from None
                        tok.expect(TokenKind.COLON)
                        annotated_type, _bindings = self._parse_type(
                            tok, self._scope, TypeParseMode.BODY
                        )
                        actual_type = self._module.values[value_id].type
                        if actual_type != annotated_type:
                            raise ParseError(
                                "operand type annotation does not match value type",
                                ssa_tok.location,
                                tok._filename,
                            )
                        value_ids.append(value_id)
                        if not tok.try_consume(TokenKind.COMMA):
                            break
                    if value_ids:
                        parsed.operand_ids.extend(value_ids)
                        self._record_operand_ids(parsed, op_decl, name, value_ids)

                case Attr(field=name):
                    attr_def = op_decl.attr(name)
                    value = self._parse_attr_value(attr_def)
                    parsed.attributes[name] = value

                case SymbolRef(field=name):
                    sym_tok = tok.expect(TokenKind.SYMBOL)
                    parsed.attributes[name] = sym_tok.text

                case TypeOf(field=name):
                    field_desc = self._layout(op_decl).fields.get(name)
                    is_result = field_desc and field_desc.kind == FieldKind.RESULT
                    parse_mode = (
                        TypeParseMode.SIGNATURE
                        if self._definition_scope_active and is_result
                        else TypeParseMode.BODY
                    )
                    annotation_token = tok.peek()
                    parsed_type, bindings = self._parse_type(
                        tok, self._scope, parse_mode
                    )
                    # Check if this field is a result — store the type.
                    if is_result:
                        parsed.result_types.append(parsed_type)
                        parsed.result_bindings.append(bindings)
                        self._assign_reserved_binding_types(bindings)
                    elif field_desc and field_desc.kind == FieldKind.OPERAND:
                        self._check_operand_type_annotation(
                            parsed, name, parsed_type, annotation_token
                        )

                case TypesOf(field=name):
                    field_desc = self._layout(op_decl).fields.get(name)
                    is_result = field_desc and field_desc.kind == FieldKind.RESULT
                    parsed_types: list[Type] = []
                    parsed_bindings: list[Mapping[int, int]] = []
                    annotation_tokens: list[Token] = []
                    if _is_type_start(tok.peek(), self._type_registry):
                        annotation_tokens.append(tok.peek())
                        t, bindings = self._parse_type(
                            tok, self._scope, TypeParseMode.BODY
                        )
                        parsed_types.append(t)
                        parsed_bindings.append(bindings)
                        while tok.try_consume(TokenKind.COMMA):
                            if not _is_type_start(tok.peek(), self._type_registry):
                                break
                            annotation_tokens.append(tok.peek())
                            t, bindings = self._parse_type(
                                tok, self._scope, TypeParseMode.BODY
                            )
                            parsed_types.append(t)
                            parsed_bindings.append(bindings)
                    if is_result:
                        for t, bindings in zip(
                            parsed_types, parsed_bindings, strict=True
                        ):
                            parsed.result_types.append(t)
                            parsed.result_bindings.append(bindings)
                            self._assign_reserved_binding_types(bindings)
                    elif field_desc and field_desc.kind == FieldKind.OPERAND:
                        value_ids = parsed.operand_fields.get(name, ())
                        if len(parsed_types) != len(value_ids):
                            raise ParseError(
                                "operand type annotation count does not match "
                                "value count",
                                tok.peek().location,
                                tok._filename,
                            )
                        for element_index, (parsed_type, annotation_token) in enumerate(
                            zip(parsed_types, annotation_tokens, strict=True)
                        ):
                            self._check_operand_type_annotation(
                                parsed,
                                name,
                                parsed_type,
                                annotation_token,
                                element_index=element_index,
                                include_element_index=True,
                            )

                case ResultType(field=name):
                    self._parse_result_type(parsed)

                case ResultTypeList(field=name, parens=parens, uniform=uniform):
                    self._parse_result_type_list(
                        parsed,
                        op_decl,
                        name,
                        parens=parens,
                        uniform=uniform,
                    )

                case Keyword(text=text):
                    _expect_keyword(tok, text)

                case Clause(name=name, elements=inner):
                    _expect_keyword(tok, name)
                    tok.expect(TokenKind.LPAREN)
                    self._walk_format(inner, op_decl, parsed)
                    tok.expect(TokenKind.RPAREN)

                case AttrDict(field=dict_field):
                    if tok.at(TokenKind.LBRACE):
                        self._parse_attr_dict(parsed, dict_field, op_decl)
                    if not dict_field:
                        self._apply_elided_attr_defaults(op_decl, element, parsed)

                case AttrTable(keys=keys_field, values=values_field):
                    self._parse_attr_table(parsed, op_decl, keys_field, values_field)

                case AlignedRefs(refs=refs_field, alignments=alignments_field):
                    self._parse_aligned_refs(
                        parsed, op_decl, refs_field, alignments_field
                    )

                case RegionTable(
                    keys=keys_field,
                    case_regions=case_regions_field,
                    default_region=default_region_field,
                ):
                    self._parse_region_table(
                        parsed,
                        op_decl,
                        keys_field,
                        case_regions_field,
                        default_region_field,
                    )

                case OperandDict(operands=operand_field, names=names_field):
                    if tok.at(TokenKind.LBRACE):
                        self._parse_operand_dict(
                            parsed, op_decl, operand_field, names_field
                        )

                case RegionFmt(field=name, syntax=syntax):
                    implicit_terminator_decl = self._implicit_terminator_decl(op_decl)
                    loop_like = _loop_like_interface(op_decl)
                    if (
                        loop_like is not None
                        and loop_like.body == name
                        and loop_like.iv is not None
                    ):
                        lower_bound_ids = parsed.operand_fields.get(
                            loop_like.lower_bound or "", ()
                        )
                        if not lower_bound_ids or not parsed.implicit_values:
                            raise ParseError(
                                "counted loop is missing its lower bound or "
                                "induction variable",
                                tok.peek().location,
                                tok._filename,
                            )
                        iv_value_id = next(iter(parsed.implicit_values.values()))
                        self._module.values[iv_value_id].type = self._module.values[
                            lower_bound_ids[0]
                        ].type
                    implicit_arg_ids = (
                        parsed.implicit_values if parsed.implicit_values else None
                    )
                    # Get block arg info from binding list if available.
                    binding_names = parsed.attributes.pop("_binding_arg_names", None)
                    binding_types = parsed.attributes.pop("_binding_arg_types", None)
                    pre_arg_ids = None
                    region_def = _region_def(op_decl, name)
                    func_args_field = _func_args_field(op_decl)
                    definition_scope_arg_ids = parsed.definition_scope_block_args.pop(
                        name, None
                    )
                    if definition_scope_arg_ids is not None:
                        if binding_names or binding_types:
                            raise ParseError(
                                f"region '{name}' cannot combine definition-"
                                "scope block args with explicit binding args",
                                tok.peek().location,
                                tok._filename,
                            )
                        pre_arg_ids = definition_scope_arg_ids
                    elif (
                        parsed.func_arg_ids
                        and func_args_field is not None
                        and region_def is not None
                        and region_def.arg_source == func_args_field
                    ):
                        if binding_names or binding_types:
                            raise ParseError(
                                f"region '{name}' cannot combine projected "
                                "FuncArgs with explicit binding args",
                                tok.peek().location,
                                tok._filename,
                            )
                        binding_names = [
                            self._module.values[value_id].name
                            for value_id in parsed.func_arg_ids
                        ]
                        binding_types = [
                            self._module.values[value_id].type
                            for value_id in parsed.func_arg_ids
                        ]
                    elif (
                        parsed.func_arg_ids
                        and not parsed.func_args_consumed
                        and name == _func_like_body_field(op_decl)
                    ):
                        # Body regions receive the logical FuncArgs values.
                        pre_arg_ids = parsed.func_arg_ids
                        parsed.func_args_consumed = True
                    region = self._parse_region_with_syntax(
                        syntax,
                        implicit_terminator_decl=implicit_terminator_decl,
                        implicit_arg_ids=implicit_arg_ids,
                        block_arg_names=binding_names,
                        block_arg_types=binding_types,
                        pre_arg_ids=pre_arg_ids,
                    )
                    parsed.implicit_values = {}
                    parsed.regions.append(region)

                case IndexList(dynamic=dynamic_field, static=static_field):
                    self._parse_index_list(parsed, op_decl, dynamic_field, static_field)

                case BindingList(field=name, kind=binding_kind):
                    self._parse_binding_list(parsed, op_decl, name, binding_kind)

                case BlockArgs(region=name, definition_scope=definition_scope):
                    self._parse_block_args(parsed, name, definition_scope)

                case FuncArgs(field=name, end_attr=end_attr):
                    tok.expect(TokenKind.LPAREN)
                    arg_ids: list[int] = []
                    if not tok.at(TokenKind.RPAREN):
                        _, _, vid = self._parse_func_arg()
                        arg_ids.append(vid)
                        while tok.try_consume(TokenKind.COMMA):
                            _, _, vid = self._parse_func_arg()
                            arg_ids.append(vid)
                    tok.expect(TokenKind.RPAREN)
                    parsed.func_arg_ids.extend(arg_ids)
                    field_desc = self._layout(op_decl).fields.get(name)
                    if field_desc is not None:
                        if (
                            field_desc.kind != FieldKind.OPERAND
                            or not field_desc.variadic
                        ):
                            raise ParseError(
                                f"FuncArgs field '{name}' must be a variadic "
                                "operand field",
                                tok.peek().location,
                                tok._filename,
                            )
                        parsed.operand_fields.setdefault(name, []).extend(arg_ids)
                    if end_attr is not None:
                        parsed.attributes[end_attr] = (
                            len(parsed.operand_fields[name])
                            if field_desc is not None
                            else len(parsed.func_arg_ids)
                        )

                case PredicateList(field=name):
                    predicates = self._parse_predicate_list()
                    parsed.attributes[name] = predicates

                case OptionalGroup(elements=inner, anchor=_anchor):
                    if self._optional_group_present(inner, op_decl):
                        self._walk_format(inner, op_decl, parsed)

                case Scope(elements=inner):
                    if self._definition_scope_active:
                        raise RuntimeError(
                            "nested Scope(...) format elements are not supported"
                        )
                    allow_symbolic_type_values = (
                        self._scope_allows_symbolic_type_values(op_decl, inner)
                    )
                    parent_scope = self._scope
                    self._scope = self._scope.push()
                    self._definition_scope_active = True
                    try:
                        self._walk_format(inner, op_decl, parsed)
                        if allow_symbolic_type_values:
                            for bindings in parsed.result_bindings:
                                self._assign_symbolic_binding_types(bindings)
                        # Function-like signatures resolve placeholders through
                        # arguments. Global-like declaration scopes keep them
                        # as local symbolic type values referenced by metadata.
                        for name, value_id in self._scope._names.items():
                            value = self._module.values[value_id]
                            if (
                                isinstance(value.type, PlaceholderType)
                                and not allow_symbolic_type_values
                            ):
                                origin = (
                                    self._scope.placeholder_location(name)
                                    or tok.current_location()
                                )
                                raise ParseError(
                                    f"unresolved forward reference to "
                                    f"'%{name}' in signature",
                                    origin,
                                    tok._filename,
                                )
                    finally:
                        self._definition_scope_active = False
                        self._scope = parent_scope

                case Flags(field=name):
                    if tok.at(TokenKind.LANGLE):
                        tok.next()  # consume '<'
                        parts: list[str] = []
                        parts.append(tok.expect(TokenKind.BARE_IDENT).text)
                        while tok.at(TokenKind.PIPE):
                            tok.next()  # consume '|'
                            parts.append(tok.expect(TokenKind.BARE_IDENT).text)
                        tok.expect(TokenKind.RANGLE)
                        parsed.attributes[name] = "|".join(parts)

                case KeyRef(field=name):
                    tok.expect(TokenKind.LANGLE)
                    if tok.at(TokenKind.OP_NAME) or tok.at(TokenKind.BARE_IDENT):
                        key_tok = tok.next()
                    else:
                        key_tok = tok.expect(TokenKind.OP_NAME)
                    tok.expect(TokenKind.RANGLE)
                    parsed.attributes[name] = key_tok.text

                case ScopedEnumRef(field=name):
                    tok.expect(TokenKind.LANGLE)
                    if tok.at(TokenKind.OP_NAME) or tok.at(TokenKind.BARE_IDENT):
                        key_tok = tok.next()
                    else:
                        key_tok = tok.expect(TokenKind.OP_NAME)
                    tok.expect(TokenKind.RANGLE)
                    parsed.attributes[name] = key_tok.text

                case StableKeyRef(key=key, stable_id=stable_id):
                    tok.expect(TokenKind.LANGLE)
                    if tok.at(TokenKind.OP_NAME) or tok.at(TokenKind.BARE_IDENT):
                        key_tok = tok.next()
                    else:
                        key_tok = tok.expect(TokenKind.OP_NAME)
                    tok.expect(TokenKind.RANGLE)
                    parsed.attributes[key] = key_tok.text
                    parsed.attributes[stable_id] = stable_id_from_string(key_tok.text)

                case TemplateParam(field=name):
                    tok.expect(TokenKind.LANGLE)
                    attr_def = op_decl.attr(name)
                    parsed.attributes[name] = self._parse_attr_value(attr_def)
                    tok.expect(TokenKind.RANGLE)

                case AttrParams(field=name):
                    attr_def = cast(AttrDef, op_decl.attr(name))
                    definition = cast(ParameterizedAttrDef, attr_def.parameterized_attr)
                    parsed.attributes[name] = self._parse_parameterized_attr_parameters(
                        definition,
                        aggregate_nesting_depth=0,
                    )

                case TemplateParamFlags(param=param_name, flags=flags_name):
                    tok.expect(TokenKind.LANGLE)
                    attr_def = op_decl.attr(param_name)
                    parsed.attributes[param_name] = self._parse_attr_value(attr_def)
                    if tok.at(TokenKind.COMMA):
                        tok.next()
                        flag_parts: list[str] = []
                        flag_parts.append(tok.expect(TokenKind.BARE_IDENT).text)
                        while tok.at(TokenKind.PIPE):
                            tok.next()
                            flag_parts.append(tok.expect(TokenKind.BARE_IDENT).text)
                        parsed.attributes[flags_name] = "|".join(flag_parts)
                    tok.expect(TokenKind.RANGLE)

                case Glue():
                    pass

    def _optional_group_present(
        self,
        inner_elements: tuple[FormatElement, ...],
        op_decl: Op | None = None,
    ) -> bool:
        """Peek to decide if an OptionalGroup is present."""
        if not inner_elements:
            return False
        tok = self._tokenizer
        first = next(
            (element for element in inner_elements if not isinstance(element, Glue)),
            None,
        )
        match first:
            case Keyword(text=text):
                return _at_keyword(tok, text)
            case Clause(name=name):
                return (
                    tok.at(TokenKind.BARE_IDENT, name)
                    and tok.peek_n(1).kind == TokenKind.LPAREN
                )
            case RegionFmt():
                result = tok.at(TokenKind.LBRACE)
                return result
            case Attr(field=attr_name):
                if op_decl is not None:
                    attr_def = op_decl.attr(attr_name)
                    if attr_def is not None:
                        if attr_def.attr_type == "enum_array":
                            return tok.at(TokenKind.LBRACKET)
                        # Adjacent optional enum groups must only consume a
                        # declared keyword or an explicit raw open-enum value.
                        if (
                            attr_def.attr_type == "enum"
                            and attr_def.enum_def is not None
                        ):
                            if attr_def.open_enum and tok.at(TokenKind.LANGLE):
                                return True
                            return (
                                tok.at(TokenKind.BARE_IDENT)
                                and tok.peek().text in attr_def.enum_def.keywords
                            )
                return tok.at(TokenKind.BARE_IDENT)
            case SymbolRef():
                return tok.at(TokenKind.SYMBOL)
            case (
                KeyRef()
                | ScopedEnumRef()
                | StableKeyRef()
                | AttrParams()
                | TemplateParam()
                | TemplateParamFlags()
            ):
                return tok.at(TokenKind.LANGLE)
            case Ref() | Refs():
                return tok.at(TokenKind.SSA_VALUE)
            case BindingList() | BlockArgs():
                # BindingList and BlockArgs print/parse as parenthesized
                # clauses, so the trigger token is the opening paren.
                return tok.at(TokenKind.LPAREN)
            case _:
                return False

    # --- Attribute parsing ---

    def _parse_attr_value(
        self, attr_def: AttrDef | None, *, attr_dict_nesting_depth: int = 0
    ) -> Any:
        """Parse an attribute value based on its AttrDef type."""
        if attr_def is None:
            return self._parse_any_attr_value(
                attr_dict_nesting_depth=attr_dict_nesting_depth
            )
        return _parse_descriptor_attr_value_from_tokens(
            self._tokenizer,
            self._module,
            self._tokenizer._filename,
            attr_def,
            scope=self._scope,
            type_registry=self._type_registry,
            mode=TypeParseMode.BODY,
            parameterized_attr_registry=self._parameterized_attr_registry,
            aliases=self._encoding_aliases,
            known_encodings=(self._known_encodings if self._known_encodings else None),
            attr_dict_nesting_depth=attr_dict_nesting_depth,
        )

    def _parse_any_attr_value(self, attr_dict_nesting_depth: int = 0) -> Any:
        """Parse any attribute value (type-agnostic)."""
        return _parse_generic_attr_value_from_tokens(
            self._tokenizer,
            self._module,
            self._tokenizer._filename,
            attr_dict_nesting_depth=attr_dict_nesting_depth,
            scope=self._scope,
            type_registry=self._type_registry,
            mode=TypeParseMode.BODY,
            parameterized_attr_registry=self._parameterized_attr_registry,
            aliases=self._encoding_aliases,
            known_encodings=(self._known_encodings if self._known_encodings else None),
        )

    def _parse_parameterized_attr(
        self,
        expected_definition: ParameterizedAttrDef | None,
        *,
        aggregate_nesting_depth: int,
    ) -> ParameterizedAttr:
        """Parse one registered descriptor-backed parameterized attribute."""
        return _parse_parameterized_attr_from_tokens(
            self._tokenizer,
            self._module,
            self._tokenizer._filename,
            expected_definition,
            scope=self._scope,
            type_registry=self._type_registry,
            mode=TypeParseMode.BODY,
            parameterized_attr_registry=self._parameterized_attr_registry,
            aliases=self._encoding_aliases,
            known_encodings=(self._known_encodings if self._known_encodings else None),
            aggregate_nesting_depth=aggregate_nesting_depth,
        )

    def _parse_parameterized_attr_parameters(
        self,
        definition: ParameterizedAttrDef,
        *,
        aggregate_nesting_depth: int,
    ) -> ParameterizedAttr:
        """Parse one known-family parameter payload without a family prefix."""
        return _parse_parameterized_attr_parameters_from_tokens(
            self._tokenizer,
            self._module,
            self._tokenizer._filename,
            definition,
            scope=self._scope,
            type_registry=self._type_registry,
            mode=TypeParseMode.BODY,
            parameterized_attr_registry=self._parameterized_attr_registry,
            aliases=self._encoding_aliases,
            known_encodings=(self._known_encodings if self._known_encodings else None),
            aggregate_nesting_depth=aggregate_nesting_depth,
        )

    # --- Result type list ---

    def _parse_result_type(self, parsed: ParsedFields) -> None:
        """Parse a single bare result type (no parentheses).

        Pushes pre-allocated result names into a child scope so the
        type annotation can reference co-results by name (e.g.,
        tile<[%m]x[%k]xf32> where %m and %k are results of this op).
        """
        scope = self._scope
        if self._reserved_result_names:
            scope = scope.push()
            for name, value_id in zip(
                self._reserved_result_names, self._reserved_result_ids, strict=False
            ):
                value = self._module.values[value_id]
                if value.type == NONE_TYPE:
                    value.type = PlaceholderType()
                scope.define(name, value_id)
        result_type, bindings = self._parse_type(
            self._tokenizer,
            scope,
            TypeParseMode.SIGNATURE
            if self._definition_scope_active
            else TypeParseMode.BODY,
        )
        parsed.result_types.append(result_type)
        parsed.result_bindings.append(bindings)
        self._assign_reserved_binding_types(bindings)

    def _parse_result_type_list(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        field_name: str,
        *,
        parens: bool = True,
        uniform: bool = False,
    ) -> None:
        """Parse a parenthesized or bare result type list."""
        saved_scope = self._scope
        if self._reserved_result_names:
            self._scope = self._scope.push()
            for name, value_id in zip(
                self._reserved_result_names, self._reserved_result_ids, strict=False
            ):
                value = self._module.values[value_id]
                if value.type == NONE_TYPE:
                    value.type = PlaceholderType()
                self._scope.define(name, value_id)
        tok = self._tokenizer
        if parens:
            tok.expect(TokenKind.LPAREN)
        if parens:
            has_entry = not tok.at(TokenKind.RPAREN)
        else:
            has_entry = _is_type_start(tok.peek(), self._type_registry) or tok.at(
                TokenKind.SSA_VALUE
            )
        result_type_start = len(parsed.result_types)
        if has_entry:
            self._parse_one_result_type(parsed)
            if uniform:
                result_count = len(self._reserved_result_ids)
                if result_count > 1:
                    result_type = parsed.result_types[result_type_start]
                    result_bindings = parsed.result_bindings[result_type_start]
                    for _ in range(1, result_count):
                        parsed.result_types.append(result_type)
                        parsed.result_bindings.append(dict(result_bindings))
            else:
                while tok.try_consume(TokenKind.COMMA):
                    self._parse_one_result_type(parsed)
        if parens:
            tok.expect(TokenKind.RPAREN)
        self._scope = saved_scope

    def _parse_one_result_type(self, parsed: ParsedFields) -> None:
        """Parse one result type entry: type, %name: type, or %operand as type."""
        tok = self._tokenizer
        # Result types use SIGNATURE mode (creating placeholders for
        # unknown dims) only inside a Scope. Outside a Scope, unknown
        # dim names are errors — same as the C parser's
        # one-level declaration-scope state.
        result_mode = (
            TypeParseMode.SIGNATURE
            if self._definition_scope_active
            else TypeParseMode.BODY
        )
        if tok.at(TokenKind.SSA_VALUE):
            name_tok = tok.next()
            if tok.try_consume(TokenKind.COLON):
                # Named result: %name: type.
                result_type, all_bindings = self._parse_type(
                    tok, self._scope, result_mode
                )
                dim_bindings = {k: v for k, v in all_bindings.items() if k >= 0}
                encoding_binding = all_bindings.get(-1, -1)
                # Resolve placeholder or define new value.
                try:
                    value_id = self._scope.lookup(name_tok.text)
                    value = self._module.values[value_id]
                    if not isinstance(value.type, PlaceholderType):
                        raise ParseError(
                            f"SSA name '%{name_tok.text}' already defined",
                            name_tok.location,
                            tok._filename,
                        )
                    value.type = result_type
                    value.dim_bindings = dim_bindings
                    value.encoding_binding = encoding_binding
                except KeyError:
                    value_id = self._module.add_value(
                        Value(
                            name=name_tok.text,
                            type=result_type,
                            dim_bindings=dim_bindings,
                            encoding_binding=encoding_binding,
                        )
                    )
                    self._scope.define(name_tok.text, value_id)
                parsed.result_types.append(result_type)
                parsed.result_bindings.append(all_bindings)
                self._assign_reserved_binding_types(all_bindings)
                parsed.result_ids.append(value_id)
            elif tok.try_consume(TokenKind.BARE_IDENT, "as"):
                # Tied result: %operand as type.
                operand_name = name_tok.text
                result_type, bindings = self._parse_type(
                    tok, self._scope, TypeParseMode.BODY
                )
                parsed.result_types.append(result_type)
                parsed.result_bindings.append(bindings)
                self._assign_reserved_binding_types(bindings)
                parsed.result_ids.append(None)
                # Find the operand index.
                try:
                    operand_id = self._scope.lookup(operand_name)
                except KeyError as exc:
                    raise ParseError(
                        f"tied result {operand_name!r} not found in args or operands",
                        name_tok.location,
                        tok._filename,
                    ) from exc
                if operand_id in parsed.func_arg_ids:
                    operand_index = parsed.func_arg_ids.index(operand_id)
                elif operand_id in parsed.operand_ids:
                    operand_index = parsed.operand_ids.index(operand_id)
                else:
                    raise ParseError(
                        f"tied result {operand_name!r} not found in args or operands",
                        name_tok.location,
                        tok._filename,
                    )
                result_index = len(parsed.result_types) - 1
                parsed.tied_results.append(
                    IRTiedResult(result_index=result_index, operand_index=operand_index)
                )
            else:
                raise ParseError(
                    f"expected ':' or 'as' after result name {name_tok.text!r}",
                    tok.peek().location,
                    tok._filename,
                )
        else:
            # Fresh result: type.
            result_type, bindings = self._parse_type(tok, self._scope, result_mode)
            parsed.result_types.append(result_type)
            parsed.result_bindings.append(bindings)
            self._assign_reserved_binding_types(bindings)
            parsed.result_ids.append(None)

    # --- Index list ---

    def _parse_index_list(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        dynamic_field: str,
        static_field: str,
    ) -> None:
        """Parse [0, %x, 4] — mixed static/dynamic indices."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACKET)
        sentinel = -(2**63)
        static_values: list[int] = []
        dynamic_ids: list[int] = []

        if not tok.at(TokenKind.RBRACKET):
            self._parse_one_index_entry(static_values, dynamic_ids, sentinel)
            while tok.try_consume(TokenKind.COMMA):
                self._parse_one_index_entry(static_values, dynamic_ids, sentinel)

        tok.expect(TokenKind.RBRACKET)
        parsed.attributes[static_field] = static_values
        parsed.operand_ids.extend(dynamic_ids)
        self._record_operand_ids(parsed, op_decl, dynamic_field, dynamic_ids)

    def _parse_one_index_entry(
        self,
        static_values: list[int],
        dynamic_ids: list[int],
        sentinel: int,
    ) -> None:
        """Parse one entry in an index list: integer or %value."""
        tok = self._tokenizer
        if tok.at(TokenKind.INTEGER):
            static_values.append(int(tok.next().text))
        elif tok.at(TokenKind.SSA_VALUE):
            ssa_tok = tok.next()
            value_id = self._scope.lookup(ssa_tok.text)
            dynamic_ids.append(value_id)
            static_values.append(sentinel)
        else:
            raise ParseError(
                f"expected integer or SSA value in index list, "
                f"got {tok.peek().kind.name}",
                tok.peek().location,
                tok._filename,
            )

    # --- Binding list ---

    def _parse_binding_list(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        field_name: str,
        kind: str = "capture",
    ) -> None:
        """Parse (%block_arg = %operand : type, ...).

        kind determines how block arg types are derived:
          "capture" — block arg has same type as operand.
          "element" — block arg has element type of operand.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.LPAREN)
        block_arg_names: list[str] = []
        block_arg_types: list[Type] = []

        if not tok.at(TokenKind.RPAREN):
            name, arg_type = self._parse_one_binding(parsed, op_decl, field_name, kind)
            block_arg_names.append(name)
            block_arg_types.append(arg_type)
            while tok.try_consume(TokenKind.COMMA):
                name, arg_type = self._parse_one_binding(
                    parsed, op_decl, field_name, kind
                )
                block_arg_names.append(name)
                block_arg_types.append(arg_type)

        tok.expect(TokenKind.RPAREN)
        # Store block arg info for region parsing.
        parsed.attributes["_binding_arg_names"] = block_arg_names
        parsed.attributes["_binding_arg_types"] = block_arg_types

    def _parse_block_args(
        self,
        parsed: ParsedFields,
        region_name: str,
        definition_scope: bool,
    ) -> None:
        """Parse BlockArgs into pending entry block argument metadata."""
        tok = self._tokenizer
        tok.expect(TokenKind.LPAREN)
        if definition_scope:
            if not self._definition_scope_active:
                raise RuntimeError(
                    "definition-scope BlockArgs must be nested in Scope(...)"
                )
            block_arg_ids: list[int] = []
            if not tok.at(TokenKind.RPAREN):
                while True:
                    _name, _arg_type, value_id = self._parse_func_arg()
                    block_arg_ids.append(value_id)
                    if not tok.try_consume(TokenKind.COMMA):
                        break
            tok.expect(TokenKind.RPAREN)
            parsed.definition_scope_block_args[region_name] = block_arg_ids
            return

        block_arg_names: list[str] = []
        block_arg_types: list[Type] = []

        if not tok.at(TokenKind.RPAREN):
            while True:
                name_token = tok.expect(TokenKind.SSA_VALUE)
                tok.expect(TokenKind.COLON)
                arg_type, _bindings = self._parse_type(
                    tok, self._scope, TypeParseMode.BODY
                )
                block_arg_names.append(name_token.text)
                block_arg_types.append(arg_type)
                if not tok.try_consume(TokenKind.COMMA):
                    break

        tok.expect(TokenKind.RPAREN)
        parsed.attributes["_binding_arg_names"] = block_arg_names
        parsed.attributes["_binding_arg_types"] = block_arg_types

    def _parse_one_binding(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        field_name: str,
        kind: str,
    ) -> tuple[str, Type]:
        """Parse one binding: %block_arg = %operand : type.

        Returns (block_arg_name, block_arg_type) where block_arg_type
        is derived from the operand type according to the binding kind.
        """
        tok = self._tokenizer
        block_arg_name = tok.expect(TokenKind.SSA_VALUE).text
        tok.expect(TokenKind.EQUALS)
        operand_name = tok.expect(TokenKind.SSA_VALUE).text
        tok.expect(TokenKind.COLON)
        operand_type, _ = self._parse_type(tok, self._scope, TypeParseMode.BODY)
        operand_id = self._scope.lookup(operand_name)
        parsed.operand_ids.append(operand_id)
        self._record_operand_ids(parsed, op_decl, field_name, [operand_id])

        # Derive block arg type based on binding kind.
        if kind == "element":
            block_arg_type = binding_element_type(operand_type)
        else:
            block_arg_type = operand_type

        return block_arg_name, block_arg_type

    # --- Predicates ---

    def _parse_predicate_list(self) -> list[Predicate]:
        """Parse [pred(...), pred(...), ...].

        Called for both function where clauses and PredicateList format
        elements. Expects the opening '[' to be the next token.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACKET)
        predicates: list[Predicate] = []
        if not tok.at(TokenKind.RBRACKET):
            predicates.append(self._parse_one_predicate())
            while tok.try_consume(TokenKind.COMMA):
                predicates.append(self._parse_one_predicate())
        tok.expect(TokenKind.RBRACKET)
        return predicates

    def _parse_one_predicate(self) -> Predicate:
        """Parse one predicate: kind(arg, arg, ...).

        Predicate kind is a bare identifier from PREDICATE_KINDS.
        """
        tok = self._tokenizer
        kind_tok = tok.expect(TokenKind.BARE_IDENT)
        kind = kind_tok.text
        if kind not in PREDICATE_KINDS:
            raise ParseError(
                f"unknown predicate kind '{kind}', "
                f"expected one of: {', '.join(sorted(PREDICATE_KINDS))}",
                kind_tok.location,
            )
        tok.expect(TokenKind.LPAREN)
        args: list[PredicateArg] = []
        if not tok.at(TokenKind.RPAREN):
            while True:
                args.append(self._parse_predicate_arg())
                if not tok.try_consume(TokenKind.COMMA):
                    break
        tok.expect(TokenKind.RPAREN)
        expected_argument_count = PREDICATE_KINDS[kind]
        actual_argument_count = len(args)
        if actual_argument_count != expected_argument_count:
            raise ParseError(
                f"predicate '{kind}' expects {expected_argument_count} "
                f"arguments, got {actual_argument_count}",
                kind_tok.location,
                tok._filename,
            )
        return Predicate(kind=kind, args=tuple(args))

    def _parse_predicate_arg(self) -> PredicateArg:
        """Parse a single predicate argument: %name or integer."""
        tok = self._tokenizer
        if tok.at(TokenKind.SSA_VALUE):
            name_tok = tok.next()
            return PredicateArg(tag="value", value=name_tok.text)
        if tok.at(TokenKind.INTEGER):
            int_tok = tok.next()
            return PredicateArg(tag="const", value=int(int_tok.text))
        raise ParseError(
            f"expected predicate argument: %name or integer, "
            f"got {tok.peek().kind.name} '{tok.peek().text}'",
            tok.peek().location,
        )

    # --- Region ---

    def _parse_region_with_syntax(
        self,
        syntax: str,
        *,
        implicit_terminator_decl: Op | None = None,
        implicit_arg_ids: dict[str, int] | None = None,
        block_arg_names: list[str] | None = None,
        block_arg_types: list[Type] | None = None,
        pre_arg_ids: list[int] | None = None,
    ) -> Region:
        """Parse a region using the selected declarative surface syntax."""
        tok = self._tokenizer
        if syntax == "test.do":
            _expect_keyword(tok, "do")
            return self._parse_region(
                implicit_terminator_decl=implicit_terminator_decl,
                implicit_arg_ids=implicit_arg_ids,
                block_arg_names=block_arg_names,
                block_arg_types=block_arg_types,
                pre_arg_ids=pre_arg_ids,
            )
        if syntax == "pipeline" and tok.at(TokenKind.BARE_IDENT, "pipeline"):
            if implicit_arg_ids or block_arg_names or block_arg_types or pre_arg_ids:
                raise ParseError(
                    "pipeline region syntax does not support entry block arguments",
                    tok.peek().location,
                    tok._filename,
                )
            tok.next()
            return self._parse_pipeline_region(
                implicit_terminator_decl=implicit_terminator_decl
            )
        return self._parse_region(
            implicit_terminator_decl=implicit_terminator_decl,
            implicit_arg_ids=implicit_arg_ids,
            block_arg_names=block_arg_names,
            block_arg_types=block_arg_types,
            pre_arg_ids=pre_arg_ids,
        )

    def _implicit_terminator_decl(self, op_decl: Op) -> Op | None:
        """Returns the validated implicit terminator declaration for op_decl."""
        terminator_name = _implicit_terminator_name(op_decl)
        if terminator_name is None:
            return None

        terminator_decl = self._op_registry.get(terminator_name)
        if terminator_decl is None:
            raise ValueError(
                f"Op '{op_decl.name}' references unknown implicit terminator "
                f"'{terminator_name}'"
            )
        if not terminator_decl.is_terminator:
            raise ValueError(
                f"Op '{op_decl.name}' references non-terminator "
                f"'{terminator_name}' in ImplicitTerminator"
            )

        terminator_layout = self._layout(terminator_decl)
        if (
            terminator_layout.fixed_operand_count != 0
            or terminator_layout.fixed_result_count != 0
            or terminator_decl.attrs
            or terminator_decl.regions
        ):
            raise ValueError(
                f"Op '{op_decl.name}' implicit terminator '{terminator_name}' "
                "must be instantiable with zero operands, results, attrs, "
                "and regions"
            )
        return terminator_decl

    def _parse_region(
        self,
        implicit_terminator_decl: Op | None = None,
        implicit_arg_ids: dict[str, int] | None = None,
        block_arg_names: list[str] | None = None,
        block_arg_types: list[Type] | None = None,
        pre_arg_ids: list[int] | None = None,
    ) -> Region:
        """Parse { block+ }.

        block_arg_names/types: pre-defined block args from a BindingList.
          These are NEW values defined in the region's scope.
        pre_arg_ids: already-defined value IDs (from function args).
          These are EXISTING values already in scope — just add to
          the entry block's arg list without re-defining.
        implicit_arg_ids: parser-created values from implicit region
          operands such as loop IVs. These are NEW names defined in the
          region's child scope.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACE)
        parent_scope = self._scope
        self._scope = parent_scope.push()

        # For function args: they're already in the parent scope.
        # Copy them into the child scope so the body can see them.
        entry_arg_ids: list[int] = []
        if pre_arg_ids:
            for vid in pre_arg_ids:
                value = self._module.values[vid]
                # Don't re-define — just make visible in child scope.
                self._scope._names[value.name] = vid
                entry_arg_ids.append(vid)

        if implicit_arg_ids:
            for name, value_id in implicit_arg_ids.items():
                self._scope.define(name, value_id)
                entry_arg_ids.append(value_id)

        # For binding list args: define new values in the child scope.
        if block_arg_names and block_arg_types:
            for name, arg_type in zip(block_arg_names, block_arg_types, strict=False):
                value_id = self._module.add_value(Value(name=name, type=arg_type))
                self._scope.define(name, value_id)
                entry_arg_ids.append(value_id)

        blocks: list[Block] = []
        is_first = True
        while not tok.at(TokenKind.RBRACE):
            block = self._parse_block(implicit_terminator_decl=implicit_terminator_decl)
            if is_first and entry_arg_ids:
                block.arg_ids = entry_arg_ids + block.arg_ids
                is_first = False
            blocks.append(block)

        if not blocks and implicit_terminator_decl is not None:
            blocks.append(
                Block(
                    arg_ids=entry_arg_ids,
                    ops=[Operation(name=implicit_terminator_decl.name)],
                )
            )

        tok.take_pending_source_trivia()
        tok.expect(TokenKind.RBRACE)
        self._scope = parent_scope
        return Region(blocks=blocks)

    def _parse_pipeline_region(self, *, implicit_terminator_decl: Op | None) -> Region:
        """Parse `pipeline { ... }` sugar into canonical pass.* ops."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACE)
        parent_scope = self._scope
        self._scope = parent_scope.push()

        ops: list[Operation] = []
        while not tok.at(TokenKind.RBRACE):
            if tok.at(TokenKind.EOF):
                tok.expect(TokenKind.RBRACE)
            if tok.at(TokenKind.BLOCK_LABEL):
                label = tok.peek()
                raise ParseError(
                    "pipeline syntax does not support block labels",
                    label.location,
                    tok._filename,
                )
            ops.append(self._parse_pipeline_statement(implicit_terminator_decl))

        if implicit_terminator_decl is not None:
            has_terminator = False
            if ops:
                final_op_decl = self._op_registry.get(ops[-1].name)
                has_terminator = (
                    final_op_decl is not None and final_op_decl.is_terminator
                )
            if not has_terminator:
                ops.append(Operation(name=implicit_terminator_decl.name))

        tok.take_pending_source_trivia()
        tok.expect(TokenKind.RBRACE)
        self._scope = parent_scope
        return Region(blocks=[Block(ops=ops)])

    def _parse_pipeline_statement(
        self, implicit_terminator_decl: Op | None
    ) -> Operation:
        """Parse one friendly pipeline statement into a canonical pass op."""
        tok = self._tokenizer
        start_token = tok.peek()
        pending_comments, leading_blank_line = tok.take_pending_source_trivia()
        comments = tuple(pending_comments)
        start_loc = start_token.location

        if tok.at(TokenKind.BARE_IDENT, "for"):
            tok.next()
            anchor = self._parse_pipeline_name("pass anchor").text
            body = self._parse_pipeline_nested_region(implicit_terminator_decl)
            return self._pipeline_operation(
                "pass.for",
                {"anchor": anchor},
                [body],
                comments,
                leading_blank_line,
                start_loc,
            )

        if tok.at(TokenKind.BARE_IDENT, "where"):
            tok.next()
            predicate = self._parse_pipeline_name("pass predicate").text
            attrs = self._parse_pipeline_attr_parens()
            body = self._parse_pipeline_nested_region(implicit_terminator_decl)
            where_attributes: dict[str, Any] = {"predicate": predicate}
            if attrs:
                where_attributes["attrs"] = attrs
            return self._pipeline_operation(
                "pass.where",
                where_attributes,
                [body],
                comments,
                leading_blank_line,
                start_loc,
            )

        if tok.at(TokenKind.BARE_IDENT, "repeat"):
            tok.next()
            mode_token = self._parse_pipeline_name("repeat mode")
            mode = mode_token.text
            if mode not in ("fixed", "until_converged"):
                raise ParseError(
                    f"invalid repeat mode '{mode}', expected fixed or until_converged",
                    mode_token.location,
                    tok._filename,
                )
            attrs = self._parse_pipeline_attr_parens()
            repeat_attributes: dict[str, Any] = {"mode": mode}
            for key, value in attrs.items():
                if key not in ("count", "max_iterations"):
                    raise ParseError(
                        f"unknown repeat option '{key}'",
                        start_loc,
                        tok._filename,
                    )
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ParseError(
                        f"repeat option '{key}' must be an integer",
                        start_loc,
                        tok._filename,
                    )
                repeat_attributes[key] = value
            body = self._parse_pipeline_nested_region(implicit_terminator_decl)
            return self._pipeline_operation(
                "pass.repeat",
                repeat_attributes,
                [body],
                comments,
                leading_blank_line,
                start_loc,
            )

        if tok.at(TokenKind.BARE_IDENT, "if"):
            tok.next()
            condition_token = self._parse_pipeline_name("pipeline condition")
            if condition_token.text != "changed":
                raise ParseError(
                    "invalid pipeline condition "
                    f"'{condition_token.text}', expected changed",
                    condition_token.location,
                    tok._filename,
                )
            body = self._parse_pipeline_nested_region(implicit_terminator_decl)
            return self._pipeline_operation(
                "pass.if_changed",
                {},
                [body],
                comments,
                leading_blank_line,
                start_loc,
            )

        if tok.at(TokenKind.BARE_IDENT, "call"):
            tok.next()
            callee = tok.expect(TokenKind.SYMBOL).text
            return self._pipeline_operation(
                "pass.call",
                {"callee": callee},
                [],
                comments,
                leading_blank_line,
                start_loc,
            )

        if tok.at(TokenKind.BARE_IDENT, "fail"):
            tok.next()
            message = tok.expect(TokenKind.STRING).text
            return self._pipeline_operation(
                "pass.fail",
                {"message": message},
                [],
                comments,
                leading_blank_line,
                start_loc,
            )

        if tok.at(TokenKind.BARE_IDENT, "halt"):
            tok.next()
            message = tok.expect(TokenKind.STRING).text
            return self._pipeline_operation(
                "pass.halt",
                {"message": message},
                [],
                comments,
                leading_blank_line,
                start_loc,
            )

        key = self._parse_pipeline_name("pass name").text
        options = self._parse_pipeline_attr_parens()
        run_attributes: dict[str, Any] = {"key": key}
        if options:
            run_attributes["options"] = options
        return self._pipeline_operation(
            "pass.run",
            run_attributes,
            [],
            comments,
            leading_blank_line,
            start_loc,
        )

    def _parse_pipeline_nested_region(
        self, implicit_terminator_decl: Op | None
    ) -> Region:
        return self._parse_pipeline_region(
            implicit_terminator_decl=implicit_terminator_decl
        )

    def _parse_pipeline_name(self, expected: str) -> Token:
        tok = self._tokenizer
        if tok.at(TokenKind.BARE_IDENT) or tok.at(TokenKind.OP_NAME):
            return tok.next()
        peek = tok.peek()
        raise ParseError(
            f"expected {expected}, got {peek.kind.name} {peek.text!r}",
            peek.location,
            tok._filename,
        )

    def _parse_pipeline_attr_parens(self) -> CanonicalAttrDict:
        tok = self._tokenizer
        if not tok.at(TokenKind.LPAREN):
            return CanonicalAttrDict()
        tok.expect(TokenKind.LPAREN)
        entries: list[tuple[str, Any]] = []
        seen_keys: set[str] = set()
        while not tok.at(TokenKind.RPAREN):
            if entries:
                tok.expect(TokenKind.COMMA)
            key_tok = tok.expect(TokenKind.BARE_IDENT)
            if key_tok.text in seen_keys:
                raise ParseError(
                    f"duplicate pipeline option '{key_tok.text}'",
                    key_tok.location,
                    tok._filename,
                )
            seen_keys.add(key_tok.text)
            tok.expect(TokenKind.EQUALS)
            value = self._parse_attr_value(None, attr_dict_nesting_depth=1)
            entries.append((key_tok.text, value))
        tok.expect(TokenKind.RPAREN)
        return CanonicalAttrDict(entries)

    def _pipeline_operation(
        self,
        name: str,
        attributes: Mapping[str, Any],
        regions: list[Region],
        comments: tuple[str, ...],
        leading_blank_line: bool,
        start_loc: SourceLocation,
    ) -> Operation:
        end_loc = self._tokenizer.current_location()
        location_id = self._add_implicit_location(start_loc, end_loc)
        op = Operation(
            name=name,
            attributes=attributes,
            regions=regions,
            location_id=location_id,
            comments=comments,
            leading_blank_line=leading_blank_line,
        )
        self._validate_operation(op, start_loc)
        return op

    def _parse_block(
        self,
        implicit_terminator_decl: Op | None = None,
    ) -> Block:
        """Parse a block (optional label, then operations)."""
        tok = self._tokenizer
        label = ""
        arg_ids: list[int] = []
        comments: tuple[str, ...] = ()
        leading_blank_line = False

        # Block label: ^name(args):
        if tok.peek().kind == TokenKind.BLOCK_LABEL:
            pending_comments, leading_blank_line = tok.take_pending_source_trivia()
            comments = tuple(pending_comments)
            label = tok.next().text
            if tok.at(TokenKind.LPAREN):
                tok.expect(TokenKind.LPAREN)
                while not tok.at(TokenKind.RPAREN):
                    arg_name = tok.expect(TokenKind.SSA_VALUE).text
                    tok.expect(TokenKind.COLON)
                    arg_type, _ = self._parse_type(tok, self._scope, TypeParseMode.BODY)
                    value_id = self._module.add_value(
                        Value(name=arg_name, type=arg_type)
                    )
                    self._scope.define(arg_name, value_id)
                    arg_ids.append(value_id)
                    tok.try_consume(TokenKind.COMMA)
                tok.expect(TokenKind.RPAREN)
            tok.expect(TokenKind.COLON)

        # Operations.
        ops: list[Operation] = []
        while (
            not tok.at(TokenKind.RBRACE)
            and not tok.at(TokenKind.BLOCK_LABEL)
            and not tok.at(TokenKind.EOF)
            and not tok.at(TokenKind.BARE_IDENT, "else")
        ):
            op = self._parse_operation()
            ops.append(op)

        if implicit_terminator_decl is not None:
            has_terminator = False
            if ops:
                final_op_decl = self._op_registry.get(ops[-1].name)
                has_terminator = (
                    final_op_decl is not None and final_op_decl.is_terminator
                )
            if not has_terminator:
                ops.append(Operation(name=implicit_terminator_decl.name))

        return Block(
            label=label,
            arg_ids=arg_ids,
            ops=ops,
            comments=comments,
            leading_blank_line=leading_blank_line,
        )

    # --- Attr dict ---

    def _parse_attr_dict(
        self, parsed: ParsedFields, field: str, op_decl: Op | None
    ) -> None:
        """Parse {key = value, ...} into a named dict attribute."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACE)
        entries: list[tuple[str, Any]] = []
        seen_keys: set[str] = set()
        while not tok.at(TokenKind.RBRACE):
            key_tok = tok.expect(TokenKind.BARE_IDENT)
            key = key_tok.text
            if key in seen_keys:
                raise ParseError(
                    f"duplicate attribute dict key '{key}'",
                    key_tok.location,
                    tok._filename,
                )
            if not field and key in parsed.attributes:
                raise ParseError(
                    f"duplicate attribute '{key}'",
                    key_tok.location,
                    tok._filename,
                )
            seen_keys.add(key)
            tok.expect(TokenKind.EQUALS)
            attr_def = None if field or op_decl is None else op_decl.attr(key)
            value = self._parse_attr_value(attr_def, attr_dict_nesting_depth=1)
            entries.append((key, value))
            tok.try_consume(TokenKind.COMMA)
        tok.expect(TokenKind.RBRACE)
        if field:
            parsed.attributes[field] = CanonicalAttrDict(entries)
        else:
            parsed.attributes.update(CanonicalAttrDict(entries))

    # --- Operand dict ---

    def _parse_operand_dict(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        operand_field: str,
        names_field: str,
    ) -> None:
        """Parse {key = %value : type, ...} into keyed variadic operands."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACE)
        entries: list[tuple[str, int]] = []
        seen_keys: set[str] = set()
        while not tok.at(TokenKind.RBRACE):
            key_tok = tok.expect(TokenKind.BARE_IDENT)
            key = key_tok.text
            if key in seen_keys:
                raise ParseError(
                    f"duplicate operand dictionary key '{key}'",
                    key_tok.location,
                    tok._filename,
                )
            seen_keys.add(key)
            tok.expect(TokenKind.EQUALS)
            value_tok = tok.expect(TokenKind.SSA_VALUE)
            try:
                value_id = self._scope.lookup(value_tok.text)
            except KeyError:
                raise ParseError(
                    f"undefined SSA value '%{value_tok.text}'",
                    value_tok.location,
                    tok._filename,
                ) from None
            tok.expect(TokenKind.COLON)
            annotated_type, _ = self._parse_type(tok, self._scope, TypeParseMode.BODY)
            actual_type = self._module.values[value_id].type
            if actual_type != annotated_type:
                raise ParseError(
                    f"operand dictionary entry '{key}' has type "
                    f"{actual_type}, but annotation is {annotated_type}",
                    value_tok.location,
                    tok._filename,
                )
            entries.append((key, value_id))
            if not tok.try_consume(TokenKind.COMMA):
                break
        tok.expect(TokenKind.RBRACE)
        sorted_entries = sorted(entries, key=lambda item: item[0])
        name_entries: list[tuple[str, int]] = []
        for ordinal, (key, value_id) in enumerate(sorted_entries):
            name_entries.append((key, ordinal))
            parsed.operand_ids.append(value_id)
            self._record_operand_ids(parsed, op_decl, operand_field, [value_id])
        if name_entries:
            parsed.attributes[names_field] = CanonicalAttrDict(name_entries)

    def _parse_aligned_refs(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        refs_field: str,
        alignments_field: str,
    ) -> None:
        """Parse [align(N) %value, ...] into paired attrs and operands."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACKET)
        alignments: list[int] = []
        value_ids: list[int] = []
        while not tok.at(TokenKind.RBRACKET):
            _expect_keyword(tok, "align")
            tok.expect(TokenKind.LPAREN)
            alignments.append(int(tok.expect(TokenKind.INTEGER).text))
            tok.expect(TokenKind.RPAREN)
            value_tok = tok.expect(TokenKind.SSA_VALUE)
            try:
                value_ids.append(self._scope.lookup(value_tok.text))
            except KeyError:
                raise ParseError(
                    f"undefined SSA value '%{value_tok.text}'",
                    value_tok.location,
                    tok._filename,
                ) from None
            if not tok.try_consume(TokenKind.COMMA):
                break
        tok.expect(TokenKind.RBRACKET)
        parsed.attributes[alignments_field] = alignments
        parsed.operand_ids.extend(value_ids)
        self._record_operand_ids(parsed, op_decl, refs_field, value_ids)

    def _parse_attr_table_row(self) -> list[int]:
        tok = self._tokenizer
        tok.expect(TokenKind.LPAREN)
        row: list[int] = []
        if not tok.at(TokenKind.RPAREN):
            value_tok = tok.expect(TokenKind.SSA_VALUE)
            try:
                row.append(self._scope.lookup(value_tok.text))
            except KeyError:
                raise ParseError(
                    f"undefined SSA value '%{value_tok.text}'",
                    value_tok.location,
                    tok._filename,
                ) from None
            while tok.try_consume(TokenKind.COMMA):
                value_tok = tok.expect(TokenKind.SSA_VALUE)
                try:
                    row.append(self._scope.lookup(value_tok.text))
                except KeyError:
                    raise ParseError(
                        f"undefined SSA value '%{value_tok.text}'",
                        value_tok.location,
                        tok._filename,
                    ) from None
        tok.expect(TokenKind.RPAREN)
        return row

    def _parse_attr_table(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        keys_field: str,
        values_field: str,
    ) -> None:
        """Parse {key = (%row), ...} default(%row) into flattened operands."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACE)
        keys: list[int] = []
        values: list[int] = []
        row_width: int | None = None
        while not tok.at(TokenKind.RBRACE):
            key_tok = tok.expect(TokenKind.INTEGER)
            keys.append(int(key_tok.text))
            tok.expect(TokenKind.EQUALS)
            row = self._parse_attr_table_row()
            if row_width is None:
                row_width = len(row)
            elif len(row) != row_width:
                raise ParseError(
                    "attribute table rows must all have the same width",
                    key_tok.location,
                    tok._filename,
                )
            values.extend(row)
            if not tok.try_consume(TokenKind.COMMA):
                break
        tok.expect(TokenKind.RBRACE)
        _expect_keyword(tok, "default")
        default_row = self._parse_attr_table_row()
        if row_width is None:
            row_width = len(default_row)
        elif len(default_row) != row_width:
            raise ParseError(
                "attribute table default row must match case row width",
                tok.peek().location,
                tok._filename,
            )
        values.extend(default_row)
        parsed.attributes[keys_field] = keys
        parsed.operand_ids.extend(values)
        self._record_operand_ids(parsed, op_decl, values_field, values)

    def _parse_region_table(
        self,
        parsed: ParsedFields,
        op_decl: Op,
        keys_field: str,
        case_regions_field: str,
        default_region_field: str,
    ) -> None:
        """Parse {case key region... default region} into keyed regions."""
        tok = self._tokenizer
        layout = self._layout(op_decl)
        case_desc = layout.fields[case_regions_field]
        default_desc = layout.fields[default_region_field]
        implicit_terminator_decl = self._implicit_terminator_decl(op_decl)

        tok.expect(TokenKind.LBRACE)
        keys: list[int] = []
        case_regions: list[Region] = []
        while tok.at(TokenKind.BARE_IDENT, "case"):
            tok.next()
            key_tok = tok.expect(TokenKind.INTEGER)
            keys.append(int(key_tok.text))
            case_regions.append(
                self._parse_region(implicit_terminator_decl=implicit_terminator_decl)
            )

        _expect_keyword(tok, "default")
        default_region = self._parse_region(
            implicit_terminator_decl=implicit_terminator_decl
        )
        tok.expect(TokenKind.RBRACE)

        while len(parsed.regions) <= default_desc.index:
            parsed.regions.append(Region(blocks=[]))
        parsed.regions[default_desc.index] = default_region
        while len(parsed.regions) < case_desc.index:
            parsed.regions.append(Region(blocks=[]))
        parsed.regions.extend(case_regions)
        parsed.attributes[keys_field] = keys
