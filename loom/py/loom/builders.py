# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dynamic typed builder facade for Loom IR."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, cast

from loom.builder import IRBuilder, TiedResultSpec, ValueRef
from loom.builder_model import (
    BuilderParam,
    BuilderParamKind,
    BuilderSignature,
    dialect_python_name,
    fixed_result_type_constraints,
    signatures_for_ops,
)
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dsl import (
    ATTR_TYPE_ENUM_ARRAY,
    ATTR_TYPE_PARAMETERIZED_ARRAY,
    ATTR_TYPE_SIGNED_ENUM_SET,
    ATTR_TYPE_SYMBOL_ARRAY,
    ATTR_TYPE_SYMBOL_SET,
    AttrDef,
    Op,
    TypeConstraint,
    TypeDef,
)
from loom.fields import compute_layout
from loom.ir import (
    I1,
    I32,
    INDEX,
    OFFSET,
    Block,
    EnumArrayAttr,
    Module,
    ParameterizedAttrArray,
    Region,
    SignedEnumSetAttr,
    SymbolName,
    SymbolNameArray,
    SymbolNameSet,
    Type,
)
from loom.stable_id import stable_id_from_string

__all__ = [
    "DialectBuilder",
    "LoomBuilder",
    "OpCallable",
    "module_builder",
]


_STATIC_INDEX_SENTINEL = -(2**63)

_FIXED_RESULT_TYPES: dict[TypeConstraint, Type] = {
    TypeConstraint.I1: I1,
    TypeConstraint.I32: I32,
    TypeConstraint.INDEX: INDEX,
    TypeConstraint.OFFSET: OFFSET,
}


@dataclass(frozen=True, slots=True)
class _DialectRegistry:
    attr_name: str
    ir_name: str
    signatures: Mapping[str, BuilderSignature]


def module_builder(
    *,
    module: Module | None = None,
    insertion_block: Block | None = None,
    ops: Sequence[Op] | None = None,
    types: Sequence[TypeDef] | None = None,
) -> tuple[Module, LoomBuilder]:
    """Create a module and dynamic Loom builder facade."""
    builder = LoomBuilder(
        module=module,
        insertion_block=insertion_block,
        ops=ops,
        types=types,
    )
    return builder.module, builder


class LoomBuilder:
    """Facade exposing dialect namespaces as `b.<dialect>.<op>()`."""

    def __init__(
        self,
        *,
        module: Module | None = None,
        insertion_block: Block | None = None,
        ops: Sequence[Op] | None = None,
        types: Sequence[TypeDef] | None = None,
    ) -> None:
        self._ir = IRBuilder(module=module, insertion_block=insertion_block)
        registered_ops = tuple(ops) if ops is not None else default_ops()
        registered_types = tuple(types) if types is not None else default_types()
        self._ir.register_ops(registered_ops)
        self._ir.register_types(registered_types)
        self._dialects = _build_dialect_registries(registered_ops)
        self._namespace_cache: dict[str, DialectBuilder] = {}

    @property
    def module(self) -> Module:
        """The module being built."""
        return self._ir.module

    @property
    def ir(self) -> IRBuilder:
        """Lower-level generic IR builder escape hatch."""
        return self._ir

    def value(self, name: str, value_type: Type, **kwargs: Any) -> ValueRef:
        """Create a named value in the underlying module."""
        return self._ir.value(name, value_type, **kwargs)

    def region(self, args: Sequence[tuple[str, Type]] = ()) -> Region:
        """Create a single-block region with named block arguments."""
        return self._ir.region(args)

    @contextmanager
    def insertion_block(self, block: Block | None) -> Iterator[None]:
        """Temporarily set the insertion block."""
        old_block = self._ir.insertion_block
        self._ir.set_insertion_block(block)
        try:
            yield
        finally:
            self._ir.set_insertion_block(old_block)

    @contextmanager
    def location(self, location_id: int) -> Iterator[None]:
        """Temporarily set the source location for subsequently built ops."""
        with self._ir.location(location_id):
            yield

    def __getattr__(self, name: str) -> DialectBuilder:
        registry = self._dialects.get(name)
        if registry is None:
            raise _attribute_error(name, self._dialects)
        namespace = self._namespace_cache.get(name)
        if namespace is None:
            namespace = DialectBuilder(self, registry)
            self._namespace_cache[name] = namespace
        return namespace

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._dialects))


class DialectBuilder:
    """One dialect namespace under `LoomBuilder`."""

    def __init__(self, owner: LoomBuilder, registry: _DialectRegistry) -> None:
        self._owner = owner
        self._registry = registry
        self._callable_cache: dict[str, OpCallable] = {}

    @property
    def name(self) -> str:
        """The underlying Loom dialect name."""
        return self._registry.ir_name

    def __getattr__(self, name: str) -> OpCallable:
        signature = self._registry.signatures.get(name)
        if signature is None:
            raise _attribute_error(name, self._registry.signatures)
        callable_op = self._callable_cache.get(name)
        if callable_op is None:
            callable_op = OpCallable(self._owner.ir, signature)
            self._callable_cache[name] = callable_op
        return callable_op

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._registry.signatures))


class OpCallable:
    """Callable dynamic builder for one Loom op declaration."""

    def __init__(self, ir_builder: IRBuilder, signature: BuilderSignature) -> None:
        self._ir = ir_builder
        self._signature = signature

    @property
    def op_name(self) -> str:
        """The underlying Loom op name."""
        return self._signature.op.name

    def __call__(self, **kwargs: Any) -> ValueRef | list[ValueRef] | None:
        op = self._signature.op
        values = _validate_and_normalize_kwargs(self._signature, kwargs)
        attributes: dict[str, Any] = {}
        operands: list[ValueRef | int] = []
        successors: list[Block] = []
        func_args: list[ValueRef | int] = []
        block_args_by_region: dict[str, Sequence[tuple[str, Type]]] = {}
        regions: list[Region] = []

        for param in self._signature.params:
            if param.kind in _OPERAND_PARAM_KINDS or (
                param.kind == BuilderParamKind.FUNC_ARGS
                and param.operand_field is not None
            ):
                continue
            if param.kind == BuilderParamKind.RESULT_TYPES:
                continue
            value = values.get(param.name)
            match param.kind:
                case BuilderParamKind.ATTR:
                    if value is not None:
                        if (
                            param.attr_def is not None
                            and param.attr_def.attr_type == ATTR_TYPE_ENUM_ARRAY
                        ):
                            value = _normalize_enum_array_attr(
                                op.name, param.attr_def, value
                            )
                        elif (
                            param.attr_def is not None
                            and param.attr_def.attr_type == ATTR_TYPE_SIGNED_ENUM_SET
                        ):
                            value = _normalize_signed_enum_set_attr(
                                op.name, param.attr_def, value
                            )
                        elif (
                            param.attr_def is not None
                            and param.attr_def.attr_type
                            == ATTR_TYPE_PARAMETERIZED_ARRAY
                        ):
                            value = _normalize_parameterized_attr_array(
                                op.name, param.attr_def, value
                            )
                        elif (
                            param.attr_def is not None
                            and param.attr_def.attr_type
                            in (ATTR_TYPE_SYMBOL_ARRAY, ATTR_TYPE_SYMBOL_SET)
                        ):
                            value = _normalize_symbol_collection_attr(
                                op.name, param.attr_def, value
                            )
                        attributes[param.name] = value
                case BuilderParamKind.STABLE_KEY_REF:
                    if value is not None:
                        attributes[param.name] = value
                        if param.stable_id_field is None:
                            raise ValueError(
                                f"stable key builder parameter '{param.name}' "
                                "has no stable ID field"
                            )
                        attributes[param.stable_id_field] = stable_id_from_string(value)
                case BuilderParamKind.FLAGS:
                    if value is not None:
                        attributes[param.name] = value
                case BuilderParamKind.BLOCK_ARGS:
                    region_field = param.region_field
                    if region_field is None:
                        raise ValueError(
                            f"{op.name}: block-argument parameter "
                            f"'{param.name}' has no region field"
                        )
                    block_args_by_region[region_field] = value or []
                case BuilderParamKind.FUNC_ARGS:
                    func_args.extend(value or [])
                    if param.end_attr_field is not None:
                        attributes[param.end_attr_field] = len(func_args)
                case BuilderParamKind.PREDICATE_LIST:
                    if value:
                        attributes[param.name] = value
                case BuilderParamKind.REGION:
                    region = cast(Region | None, value)
                    entry_args = block_args_by_region.pop(param.name, ())
                    if region is None and not param.region_optional:
                        region = self._ir.region(entry_args)
                    elif region is not None and entry_args:
                        _ensure_region_entry_args(
                            self._ir, op.name, param.name, region, entry_args
                        )
                    if region is not None:
                        regions.append(region)
                case BuilderParamKind.REGION_TABLE_DEFAULT:
                    regions.append(cast(Region, value))
                case BuilderParamKind.REGION_TABLE_CASES:
                    regions.extend(cast(Sequence[Region], value))
                case BuilderParamKind.SUCCESSOR:
                    successors.append(cast(Block, value))
                case _:
                    raise ValueError(
                        f"{op.name}: unsupported builder parameter kind "
                        f"{param.kind.name}"
                    )

        operand_segment_counts = self._append_operands(op, values, operands, attributes)

        results = values.get("results")
        if results is None:
            fixed_constraints = fixed_result_type_constraints(op)
            if fixed_constraints is not None:
                results = [
                    _FIXED_RESULT_TYPES[constraint] for constraint in fixed_constraints
                ]
        result_names = _normalize_result_names(
            op,
            results or [],
            name=values.get("name"),
            names=values.get("names"),
            result_names=values.get("result_names"),
        )
        return self._ir.build(
            op.name,
            operands,
            operand_segment_counts=operand_segment_counts,
            successors=successors,
            func_args=func_args,
            results=results,
            result_names=result_names,
            attributes=attributes,
            regions=regions,
            location_id=values.get("location_id"),
        )

    def _append_operands(
        self,
        op: Op,
        values: dict[str, Any],
        operands: list[ValueRef | int],
        attributes: dict[str, Any],
    ) -> list[int]:
        """Appends operand fields in storage order and returns segment counts."""
        params_by_field: dict[str, list[BuilderParam]] = {}
        for param in self._signature.params:
            field = param.operand_field or param.name
            params_by_field.setdefault(field, []).append(param)
        operand_segment_counts: list[int] = []
        layout = compute_layout(op)
        if layout.segmented_operands:
            for operand in op.operands:
                operand_params = params_by_field.get(operand.name, [])
                if not operand_params:
                    operand_segment_counts.append(0)
                    continue
                before_count = len(operands)
                for operand_param in operand_params:
                    value = values.get(operand_param.name)
                    self._append_operand_param(
                        op, operand_param, value, operands, attributes
                    )
                    if operand_param.end_attr_field is not None:
                        attributes[operand_param.end_attr_field] = (
                            len(operands) - before_count
                        )
                operand_segment_counts.append(len(operands) - before_count)
        else:
            optional_operand_tail_closed = False
            for operand in op.operands:
                operand_params = params_by_field.get(operand.name, [])
                if not operand_params:
                    continue
                before_count = len(operands)
                for operand_param in operand_params:
                    value = values.get(operand_param.name)
                    if (
                        operand_param.kind == BuilderParamKind.OPERAND
                        and value is None
                        and not operand_param.required
                    ):
                        optional_operand_tail_closed = True
                        continue
                    if (
                        operand_param.kind == BuilderParamKind.OPERAND
                        and value is not None
                        and optional_operand_tail_closed
                    ):
                        raise ValueError(
                            f"{op.name}: optional operand "
                            f"'{operand_param.name}' cannot be present after an "
                            "earlier optional operand was omitted"
                        )
                    self._append_operand_param(
                        op, operand_param, value, operands, attributes
                    )
                    if operand_param.end_attr_field is not None:
                        attributes[operand_param.end_attr_field] = (
                            len(operands) - before_count
                        )

        return operand_segment_counts

    def _append_operand_param(
        self,
        op: Op,
        operand_param: BuilderParam,
        value: Any,
        operands: list[ValueRef | int],
        attributes: dict[str, Any],
    ) -> None:
        match operand_param.kind:
            case BuilderParamKind.OPERAND:
                if value is not None:
                    operands.append(cast(ValueRef | int, value))
            case BuilderParamKind.OPERAND_VARIADIC:
                operands.extend(value or [])
            case BuilderParamKind.FUNC_ARGS:
                operands.extend(value or [])
            case BuilderParamKind.INDEX_LIST:
                static_offsets: list[int] = []
                for index in cast(Sequence[int | ValueRef], value):
                    if isinstance(index, ValueRef):
                        static_offsets.append(_STATIC_INDEX_SENTINEL)
                        operands.append(index)
                    elif isinstance(index, int):
                        static_offsets.append(index)
                    else:
                        raise TypeError(
                            f"{op.name}: index list '{operand_param.name}' entries "
                            "must be int or ValueRef, "
                            f"got {type(index).__name__}"
                        )
                static_field = _require_param_field(operand_param, "static_field")
                attributes[static_field] = static_offsets
            case BuilderParamKind.OPERAND_DICT:
                operand_dict = value or {}
                if operand_dict:
                    operand_names: dict[str, int] = {}
                    for name in sorted(operand_dict):
                        operand_names[name] = len(operand_names)
                        operands.append(operand_dict[name])
                    names_field = _require_param_field(operand_param, "names_field")
                    attributes[names_field] = operand_names
            case _:
                raise ValueError(
                    f"{op.name}: operand field '{operand_param.name}' is bound to "
                    f"unsupported builder parameter kind {operand_param.kind.name}"
                )


_OPERAND_PARAM_KINDS = frozenset(
    {
        BuilderParamKind.INDEX_LIST,
        BuilderParamKind.OPERAND,
        BuilderParamKind.OPERAND_DICT,
        BuilderParamKind.OPERAND_VARIADIC,
    }
)


def _validate_and_normalize_kwargs(
    signature: BuilderSignature,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    valid_names = {param.py_name for param in signature.params}
    valid_names.add("location_id")
    if signature.op.results:
        valid_names.update(("name", "names", "result_names"))
    unexpected = sorted(set(kwargs) - valid_names)
    if unexpected:
        raise TypeError(
            f"{signature.op.name}: unexpected parameter "
            f"{unexpected[0]!r}{_did_you_mean(unexpected[0], valid_names)}"
        )

    values: dict[str, Any] = {}
    for param in signature.params:
        if param.py_name in kwargs:
            values[param.name] = kwargs[param.py_name]
            continue
        if param.required:
            raise TypeError(
                f"{signature.op.name}: missing required parameter {param.py_name!r}"
            )
        values[param.name] = _default_value(param)
    if signature.op.results:
        values["name"] = kwargs.get("name")
        values["names"] = kwargs.get("names")
        values["result_names"] = kwargs.get("result_names")
    values["location_id"] = kwargs.get("location_id")
    return values


def _default_value(param: BuilderParam) -> Any:
    match param.kind:
        case BuilderParamKind.BLOCK_ARGS:
            return []
        case BuilderParamKind.FUNC_ARGS:
            return []
        case BuilderParamKind.OPERAND_DICT:
            return {}
        case BuilderParamKind.OPERAND_VARIADIC:
            return []
        case BuilderParamKind.REGION:
            return None
        case BuilderParamKind.FLAGS:
            return None
        case BuilderParamKind.PREDICATE_LIST if not param.required:
            return []
        case _:
            return None


def _normalize_enum_array_attr(
    op_name: str, attr_def: AttrDef, value: Any
) -> EnumArrayAttr:
    """Resolves symbolic enum-array builder values to stable byte values."""
    if isinstance(value, EnumArrayAttr):
        values = value.values
    else:
        if isinstance(value, str) or not isinstance(value, Iterable):
            raise TypeError(
                f"{op_name}: enum array '{attr_def.name}' must be an iterable "
                "of enum keywords or byte values"
            )
        values = tuple(value)

    assert attr_def.enum_def is not None
    value_by_keyword = {case.keyword: case.value for case in attr_def.enum_def.cases}
    declared_values = frozenset(value_by_keyword.values())
    normalized_values: list[int] = []
    for index, element in enumerate(values):
        if isinstance(element, str):
            stable_value = value_by_keyword.get(element)
            if stable_value is None:
                raise ValueError(
                    f"{op_name}: enum array '{attr_def.name}' element {index} "
                    f"has unknown keyword {element!r}"
                )
        elif type(element) is int and 0 <= element <= 0xFF:
            stable_value = element
            if not attr_def.open_enum and stable_value not in declared_values:
                raise ValueError(
                    f"{op_name}: enum array '{attr_def.name}' element {index} "
                    f"has undeclared value {stable_value}"
                )
        else:
            raise TypeError(
                f"{op_name}: enum array '{attr_def.name}' element {index} "
                f"must be an enum keyword or byte value, got {element!r}"
            )
        normalized_values.append(stable_value)
    return EnumArrayAttr(normalized_values)


def _normalize_signed_enum_set_attr(
    op_name: str, attr_def: AttrDef, value: Any
) -> SignedEnumSetAttr:
    """Resolves signed enum-set assertions to stable byte values."""
    if isinstance(value, SignedEnumSetAttr):
        assertions: Mapping[str | int, bool] = {
            **{ordinal: True for ordinal in value.positive_values},
            **{ordinal: False for ordinal in value.negative_values},
        }
    elif isinstance(value, Mapping):
        assertions = value
    else:
        raise TypeError(
            f"{op_name}: signed enum set '{attr_def.name}' must be a mapping "
            "of enum keywords or byte values to Booleans"
        )

    assert attr_def.enum_def is not None
    value_by_keyword = {case.keyword: case.value for case in attr_def.enum_def.cases}
    declared_values = frozenset(value_by_keyword.values())
    positive_values: list[int] = []
    negative_values: list[int] = []
    resolved_values: set[int] = set()
    for enum_value, assertion in assertions.items():
        if type(assertion) is not bool:
            raise TypeError(
                f"{op_name}: signed enum set '{attr_def.name}' assertion "
                f"for {enum_value!r} must be a Boolean, got {assertion!r}"
            )
        if isinstance(enum_value, str):
            stable_value = value_by_keyword.get(enum_value)
            if stable_value is None:
                raise ValueError(
                    f"{op_name}: signed enum set '{attr_def.name}' has "
                    f"unknown keyword {enum_value!r}"
                )
        elif type(enum_value) is int and 0 <= enum_value <= 0xFF:
            stable_value = enum_value
            if stable_value not in declared_values:
                raise ValueError(
                    f"{op_name}: signed enum set '{attr_def.name}' has "
                    f"undeclared value {stable_value}"
                )
        else:
            raise TypeError(
                f"{op_name}: signed enum set '{attr_def.name}' key must be "
                f"an enum keyword or byte value, got {enum_value!r}"
            )
        if stable_value in resolved_values:
            raise ValueError(
                f"{op_name}: signed enum set '{attr_def.name}' names stable "
                f"value {stable_value} more than once"
            )
        resolved_values.add(stable_value)
        (positive_values if assertion else negative_values).append(stable_value)
    return SignedEnumSetAttr(positive_values, negative_values)


def _normalize_parameterized_attr_array(
    op_name: str, attr_def: AttrDef, value: Any
) -> ParameterizedAttrArray:
    """Canonicalizes and validates a parameterized-attribute array field."""
    if isinstance(value, ParameterizedAttrArray):
        array = value
    else:
        if isinstance(value, str | bytes | bytearray) or not isinstance(
            value, Iterable
        ):
            raise TypeError(
                f"{op_name}: parameterized attribute array '{attr_def.name}' "
                "must be an iterable of ParameterizedAttr values"
            )
        array = ParameterizedAttrArray(value)

    expected_family = attr_def.parameterized_attr
    if expected_family is not None:
        for index, element in enumerate(array):
            if element.family_name != expected_family.name:
                raise ValueError(
                    f"{op_name}: parameterized attribute array "
                    f"'{attr_def.name}' element {index} has family "
                    f"'{element.family_name}', expected "
                    f"'{expected_family.name}'"
                )
    return array


def _normalize_symbol_collection_attr(
    op_name: str, attr_def: AttrDef, value: Any
) -> SymbolNameArray | SymbolNameSet:
    """Normalizes an ordered symbol array or symbol set."""
    values = (
        value.values if isinstance(value, SymbolNameArray | SymbolNameSet) else value
    )
    if isinstance(values, str | bytes | bytearray) or not isinstance(values, Iterable):
        raise TypeError(
            f"{op_name}: {attr_def.attr_type.replace('_', ' ')} "
            f"'{attr_def.name}' must be an iterable "
            "of symbol names"
        )
    names: list[SymbolName] = []
    for index, element in enumerate(values):
        if not isinstance(element, str):
            raise TypeError(
                f"{op_name}: {attr_def.attr_type.replace('_', ' ')} "
                f"'{attr_def.name}' element {index} "
                f"must be a symbol name, got {element!r}"
            )
        if element.startswith("@"):
            raise ValueError(
                f"{op_name}: {attr_def.attr_type.replace('_', ' ')} "
                f"'{attr_def.name}' element {index} "
                "must not include '@'"
            )
        names.append(SymbolName(element))
    if attr_def.attr_type == ATTR_TYPE_SYMBOL_ARRAY:
        return SymbolNameArray(names)
    try:
        return SymbolNameSet(names)
    except ValueError as exc:
        raise ValueError(f"{op_name}: symbol set '{attr_def.name}' {exc}") from exc


def _normalize_result_names(
    op: Op,
    results: Sequence[Type | TiedResultSpec],
    *,
    name: str | None,
    names: Sequence[str] | None,
    result_names: Sequence[str] | None,
) -> list[str] | None:
    supplied = sum(value is not None for value in (name, names, result_names))
    if supplied > 1:
        raise TypeError(
            f"{op.name}: pass only one of 'name', 'names', or 'result_names'"
        )
    if not op.results:
        if supplied:
            raise TypeError(f"{op.name}: result names were supplied for a void op")
        return None
    if name is not None:
        if len(op.results) != 1 or op.results[0].variadic:
            raise TypeError(f"{op.name}: 'name' is only valid for one fixed result")
        return [name]
    selected_names = names if names is not None else result_names
    if selected_names is None:
        return None
    result_name_list = list(selected_names)
    if any(result.variadic for result in op.results):
        if len(result_name_list) != len(results):
            raise TypeError(
                f"{op.name}: result name count {len(result_name_list)} does not "
                f"match result spec count {len(results)}"
            )
        return result_name_list
    if len(result_name_list) != len(op.results):
        raise TypeError(
            f"{op.name}: result name count {len(result_name_list)} does not "
            f"match fixed result count {len(op.results)}"
        )
    return result_name_list


def _ensure_region_entry_args(
    ir_builder: IRBuilder,
    op_name: str,
    region_name: str,
    region: Region,
    args: Sequence[tuple[str, Type]],
) -> None:
    if not region.blocks:
        region.blocks.append(ir_builder.region(args).blocks[0])
        return

    entry_block = region.blocks[0]
    if not entry_block.arg_ids:
        entry_block.arg_ids.extend(ir_builder.region(args).blocks[0].arg_ids)
        return

    if len(entry_block.arg_ids) != len(args):
        raise ValueError(
            f"{op_name}: region '{region_name}' entry block has "
            f"{len(entry_block.arg_ids)} args but builder received "
            f"{len(args)} block args."
        )
    for arg_index, (value_id, (expected_name, expected_type)) in enumerate(
        zip(entry_block.arg_ids, args, strict=True)
    ):
        value = ir_builder.module.values[value_id]
        if value.type != expected_type:
            raise ValueError(
                f"{op_name}: region '{region_name}' arg {arg_index} "
                f"has type {value.type!r} but builder received "
                f"{expected_type!r}."
            )
        if expected_name and value.name != expected_name:
            raise ValueError(
                f"{op_name}: region '{region_name}' arg {arg_index} "
                f"is named {value.name!r} but builder received "
                f"{expected_name!r}."
            )


def _require_param_field(param: BuilderParam, field_name: str) -> str:
    value = getattr(param, field_name)
    if not isinstance(value, str):
        raise ValueError(f"{param.name}: missing builder model field {field_name}")
    return value


def _attribute_error(name: str, available: Mapping[str, Any]) -> AttributeError:
    return AttributeError(
        f"unknown builder name {name!r}{_did_you_mean(name, available)}"
    )


def _did_you_mean(name: str, available: Iterable[object]) -> str:
    candidates = [str(candidate) for candidate in available]
    matches = get_close_matches(name, candidates, n=1)
    if not matches:
        return ""
    return f"; did you mean {matches[0]!r}?"


def _build_dialect_registries(ops: Sequence[Op]) -> dict[str, _DialectRegistry]:
    grouped_ops: dict[str, list[Op]] = {}
    for op in ops:
        dialect_name = op.name.split(".", 1)[0]
        grouped_ops.setdefault(dialect_name, []).append(op)
    registries: dict[str, _DialectRegistry] = {}
    for dialect_name, dialect_ops in sorted(grouped_ops.items()):
        attr_name = dialect_python_name(dialect_name)
        registries[attr_name] = _DialectRegistry(
            attr_name=attr_name,
            ir_name=dialect_name,
            signatures=signatures_for_ops(dialect_ops),
        )
    return registries


def default_ops() -> tuple[Op, ...]:
    """Return every op in the default Loom Python dialect registry."""
    from loom.dialect.buffer import ALL_BUFFER_OPS
    from loom.dialect.cfg import ALL_CFG_OPS
    from loom.dialect.check import ALL_CHECK_OPS
    from loom.dialect.command import ALL_COMMAND_OPS
    from loom.dialect.encoding import ALL_ENCODING_OPS
    from loom.dialect.func import ALL_FUNC_OPS
    from loom.dialect.globals import ALL_GLOBAL_OPS
    from loom.dialect.index import ALL_INDEX_OPS
    from loom.dialect.kernel import ALL_KERNEL_OPS
    from loom.dialect.llvmir import ALL_LLVMIR_OPS
    from loom.dialect.low import ALL_LOW_OPS
    from loom.dialect.pass_ import ALL_PASS_OPS
    from loom.dialect.pool import ALL_POOL_OPS
    from loom.dialect.sanitizer import ALL_SANITIZER_OPS
    from loom.dialect.scalar import ALL_SCALAR_OPS
    from loom.dialect.scf import ALL_SCF_OPS
    from loom.dialect.target import ALL_TARGET_OPS
    from loom.dialect.test import ALL_TEST_OPS
    from loom.dialect.vector import ALL_VECTOR_OPS
    from loom.dialect.view import ALL_VIEW_OPS

    return (
        *ALL_BUFFER_OPS,
        *ALL_CFG_OPS,
        *ALL_CHECK_OPS,
        *ALL_COMMAND_OPS,
        *ALL_ENCODING_OPS,
        *ALL_FUNC_OPS,
        *ALL_GLOBAL_OPS,
        *ALL_INDEX_OPS,
        *ALL_KERNEL_OPS,
        *ALL_LLVMIR_OPS,
        *ALL_LOW_OPS,
        *ALL_PASS_OPS,
        *ALL_POOL_OPS,
        *ALL_SANITIZER_OPS,
        *ALL_SCALAR_OPS,
        *ALL_SCF_OPS,
        *ALL_TARGET_OPS,
        *ALL_TEST_OPS,
        *ALL_VECTOR_OPS,
        *ALL_VIEW_OPS,
    )


def default_types() -> tuple[TypeDef, ...]:
    """Return every type in the default Loom Python type registry."""
    from loom.dialect.hal import ALL_HAL_TYPES
    from loom.dialect.kernel import ALL_KERNEL_TYPES

    return (
        *ALL_BUILTIN_TYPES,
        *ALL_HAL_TYPES,
        *ALL_KERNEL_TYPES,
    )
