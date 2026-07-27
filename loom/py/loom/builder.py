# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR builder: construct loom Operations from Op declarations.

The builder provides a typed, validated API for constructing IR.
Parameter order follows the format spec: the order you see in the
textual assembly is the order you pass to the builder.

Two layers:
  1. Generic builder (this module): creates Operations from any Op
     declaration. Called by generated stubs and directly for one-offs.
  2. Generated typed stubs (gen/builders_*.py): per-op methods with
     typed parameters and docstrings. Checked in, greppable, IDE-friendly.

Value references:
  The builder returns ValueRef objects (not bare ints). ValueRef
  supports indexing (tensor[off]) and tied result marking (a.as_type(t)).

    b = IRBuilder()
    ...
    tensor = b.value("tensor", tensor_t)
    tile = b.value("tile", tile_t)
    off = b.value("off", INDEX)

    # Natural indexing: tensor[off] captures value + offsets
    result = b.tensor.update(source=tile, target=tensor[off])

    # Tied results: a.as_type(t) marks 'a' as consumed, result has type t
    out, count = b.func.call("@f", a, b_val,
                              results=[tied(a, tensor_t), INDEX])
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from loom.assembly import Clause, FuncArgs, OptionalGroup, Scope
from loom.dsl import FuncLikeInterface, Op, TypeDef
from loom.fields import FieldLayout
from loom.ir import (
    LOCATION_UNKNOWN,
    VALUE_DEF_BLOCK_NONE,
    VALUE_FLAG_BLOCK_ARG,
    Block,
    Module,
    Operation,
    Region,
    ScalarType,
    ShapedType,
    Type,
    Value,
    canonicalize_attr_dict,
    record_operation_value_metadata,
    symbol_from_operation,
)
from loom.ir import (
    TiedResult as IRTiedResult,
)

__all__ = [
    "IRBuilder",
    "ValueRef",
    "IndexedValue",
    "TiedResultSpec",
    "tied",
]


# ============================================================================
# Value reference types
# ============================================================================


class ValueRef:
    """A reference to an SSA value in the builder.

    Returned by IRBuilder.value() and build(). Supports:
      - Indexing: tensor[off] or tensor[i, j] → IndexedValue
      - Tied marking: val.as_type(new_t) → TiedResultSpec
      - Use as operand: pass directly to build() or generated stubs
    """

    __slots__ = ("_id", "_builder")

    def __init__(self, value_id: int, builder: IRBuilder) -> None:
        self._id = value_id
        self._builder = builder

    @property
    def id(self) -> int:
        """The underlying value ID in the module."""
        return self._id

    @property
    def type(self) -> Type:
        """The type of this value."""
        return self._builder.module.values[self._id].type

    @property
    def name(self) -> str:
        """The SSA name of this value."""
        return self._builder.module.values[self._id].name

    def __getitem__(self, indices: Any) -> IndexedValue:
        """tensor[off] or tensor[i, j] → IndexedValue.

        Each index is either an int (static) or a ValueRef (dynamic).
        """
        if not isinstance(indices, tuple):
            indices = (indices,)
        return IndexedValue(self._id, list(indices), self._builder)

    def as_type(self, new_type: Type) -> TiedResultSpec:
        """Mark this value as tied to a result with the given type.

        Used in result specs: results=[a.as_type(tensor_t), INDEX]
        Mirrors the textual: -> (%a as tensor<...>, index)
        """
        return TiedResultSpec(self._id, new_type)

    def __repr__(self) -> str:
        name = self._builder.module.values[self._id].name
        return f"ValueRef({name or self._id})"

    def __int__(self) -> int:
        """Allow using ValueRef where an int value ID is expected."""
        return self._id


class IndexedValue:
    """A value with index offsets: tensor[%i, %j] or tensor[0, %off].

    Created by ValueRef.__getitem__. Passed to builder methods that
    expect an indexed operand (ops with IndexList in their format).
    The builder extracts the base value and offsets to build the
    static/dynamic offset arrays.
    """

    __slots__ = ("_value_id", "_offsets", "_builder")

    def __init__(self, value_id: int, offsets: list[Any], builder: IRBuilder) -> None:
        self._value_id = value_id
        self._offsets = offsets
        self._builder = builder

    @property
    def value_id(self) -> int:
        return self._value_id

    @property
    def offsets(self) -> list[Any]:
        """List of int (static) or ValueRef (dynamic) offsets."""
        return self._offsets

    def decompose(self) -> tuple[int, list[int], list[int]]:
        """Decompose into (base_id, static_offsets, dynamic_ids).

        Static offsets are integers. Dynamic offsets are replaced with
        a sentinel in the static array, and the dynamic value IDs are
        collected separately.
        """
        sentinel = -(2**63)
        static: list[int] = []
        dynamic: list[int] = []
        for offset in self._offsets:
            if isinstance(offset, ValueRef):
                static.append(sentinel)
                dynamic.append(offset._id)
            elif isinstance(offset, int):
                static.append(offset)
            else:
                raise TypeError(
                    f"Index must be int (static) or ValueRef (dynamic), "
                    f"got {type(offset).__name__}"
                )
        return self._value_id, static, dynamic

    def __repr__(self) -> str:
        return f"IndexedValue({self._value_id}, {self._offsets})"


@dataclass(frozen=True, slots=True)
class TiedResultSpec:
    """Specifies that a result is tied to an operand.

    Created by ValueRef.as_type() or the tied() function.
    Used in result specs: results=[tied(a, tensor_t), INDEX]
    """

    operand_id: int
    result_type: Type

    def __repr__(self) -> str:
        return f"tied({self.operand_id}, {self.result_type})"


def tied(operand: ValueRef | int, result_type: Type) -> TiedResultSpec:
    """Mark a result as tied to an operand with a (possibly different) type.

    Usage: results=[tied(tensor, new_tensor_t), INDEX]
    Mirrors: -> (%tensor as new_tensor_t, index)
    """
    operand_id = operand._id if isinstance(operand, ValueRef) else operand
    return TiedResultSpec(operand_id, result_type)


def _find_func_like_interface(op_decl: Op) -> FuncLikeInterface | None:
    """Return op_decl's FuncLikeInterface implementation, if any."""
    for interface in op_decl.interfaces:
        if isinstance(interface, FuncLikeInterface):
            return interface
    return None


def _find_func_args_field(op_decl: Op) -> str:
    """Return op_decl's FuncArgs format field name, defaulting to args."""

    def walk(elements: Sequence[Any]) -> str | None:
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

    return walk(op_decl.format) or "args"


class IRBuilder:
    """Generic IR builder: constructs Operations from Op declarations.

    The builder manages a Module and provides methods to create values
    and operations. It validates types and constraints at construction
    time (fail-fast, no silent errors).

    Symbol-defining ops are appended to module.symbols. Non-symbol ops are
    appended to insertion_block, which must be set before building body ops.
    """

    def __init__(
        self, module: Module | None = None, insertion_block: Block | None = None
    ) -> None:
        self._module = module if module is not None else Module()
        self._insertion_block = insertion_block
        self._op_registry: dict[str, Op] = {}
        self._type_registry: dict[str, TypeDef] = {}
        self._layouts: dict[str, FieldLayout] = {}
        self._location_id = LOCATION_UNKNOWN

    @property
    def module(self) -> Module:
        """The module being built."""
        return self._module

    @property
    def insertion_block(self) -> Block | None:
        """The block where non-symbol ops are appended."""
        return self._insertion_block

    @property
    def location_id(self) -> int:
        """Location assigned to subsequently built operations."""
        return self._location_id

    def set_insertion_block(self, block: Block | None) -> None:
        """Set the insertion block for subsequent non-symbol ops."""
        self._insertion_block = block

    @contextmanager
    def location(self, location_id: int) -> Iterator[None]:
        """Temporarily set the source location for subsequently built ops."""
        old_location_id = self._location_id
        self._location_id = location_id
        try:
            yield
        finally:
            self._location_id = old_location_id

    def register_ops(self, ops: Sequence[Op]) -> None:
        """Register op declarations."""
        for op in ops:
            self._op_registry[op.name] = op

    def register_types(self, types: Sequence[TypeDef]) -> None:
        """Register type declarations."""
        for td in types:
            self._type_registry[td.name] = td

    # --- Value creation ---

    def value(self, name: str, value_type: Type, **kwargs: Any) -> ValueRef:
        """Create a named value and return a ValueRef.

        Used for function arguments, constants, and any value that
        needs to exist before ops that reference it.
        """
        value = Value(name=name, type=value_type, **kwargs)
        value_id = self._module.add_value(value)
        return ValueRef(value_id, self)

    def region(self, args: Sequence[tuple[str, Type]] = ()) -> Region:
        """Create a single-block region with named block arguments."""
        block = Block()
        for arg_index, (name, arg_type) in enumerate(args):
            value_id = self._module.add_value(
                Value(
                    name=name,
                    type=arg_type,
                    flags=VALUE_FLAG_BLOCK_ARG,
                    def_result_index=arg_index,
                )
            )
            block.arg_ids.append(value_id)
        return Region(blocks=[block])

    # --- Operand resolution ---

    def _resolve_operand(self, operand: ValueRef | IndexedValue | int) -> int:
        """Resolve an operand to a value ID."""
        if isinstance(operand, ValueRef):
            return operand._id
        if isinstance(operand, IndexedValue):
            return operand._value_id
        if isinstance(operand, int):
            return operand
        raise TypeError(
            f"Operand must be ValueRef, IndexedValue, or int, "
            f"got {type(operand).__name__}"
        )

    def _resolve_operands(
        self,
        operands: Sequence[ValueRef | IndexedValue | int],
    ) -> list[int]:
        """Resolve a list of operands to value IDs."""
        return [self._resolve_operand(op) for op in operands]

    def _prepare_func_like_args(
        self,
        op_decl: Op,
        operand_ids: list[int],
        region_list: list[Region],
        func_args: Sequence[ValueRef | int] | None,
    ) -> list[int]:
        """Lower FuncArgs values into the op's abstract signature arg domain.

        For declaration-style ops, signature args are op operands. For bodyful
        funcs, signature args are the entry block arguments of the body region.

        Returns the value IDs that tied result specs should resolve against.
        """
        func_like = _find_func_like_interface(op_decl)
        func_arg_ids = self._resolve_operands(func_args or [])
        if func_like is None:
            if func_arg_ids:
                raise ValueError(
                    f"Op '{op_decl.name}' does not implement FuncLikeInterface "
                    "and cannot accept func_args."
                )
            return operand_ids

        if func_like.args_as_operands:
            operand_ids.extend(func_arg_ids)
            return operand_ids

        body_field = func_like.body
        if body_field is None:
            if func_arg_ids:
                raise ValueError(
                    f"Op '{op_decl.name}' stores signature args in a body region "
                    "but FuncLikeInterface.body is not set."
                )
            return operand_ids

        body_region_index = next(
            (
                i
                for i, region_def in enumerate(op_decl.regions)
                if region_def.name == body_field
            ),
            None,
        )
        if body_region_index is None:
            raise ValueError(
                f"Op '{op_decl.name}' FuncLikeInterface.body='{body_field}' does not "
                "name a declared region."
            )

        while len(region_list) <= body_region_index:
            region_list.append(Region())
        body_region = region_list[body_region_index]
        if not body_region.blocks:
            body_region.blocks.append(Block(arg_ids=list(func_arg_ids)))
        else:
            entry_block = body_region.blocks[0]
            if func_arg_ids:
                if entry_block.arg_ids and entry_block.arg_ids != list(func_arg_ids):
                    raise ValueError(
                        f"Op '{op_decl.name}' body entry block arg_ids "
                        f"{entry_block.arg_ids} do not match func_args "
                        f"{list(func_arg_ids)}."
                    )
                if not entry_block.arg_ids:
                    entry_block.arg_ids.extend(func_arg_ids)

        signature_arg_ids = list(body_region.blocks[0].arg_ids)
        func_args_field = _find_func_args_field(op_decl)
        for region_index, region_def in enumerate(op_decl.regions):
            if region_index == body_region_index:
                continue
            if region_def.arg_source != func_args_field:
                continue
            while len(region_list) <= region_index:
                region_list.append(Region())
            projected_region = region_list[region_index]
            if not projected_region.blocks:
                projected_region.blocks.append(
                    Block(arg_ids=self._clone_func_signature_args(signature_arg_ids))
                )
                continue
            projected_entry = projected_region.blocks[0]
            if not projected_entry.arg_ids:
                projected_entry.arg_ids.extend(
                    self._clone_func_signature_args(signature_arg_ids)
                )
                continue
            self._validate_projected_region_args(
                op_decl.name,
                region_def.name,
                signature_arg_ids,
                projected_entry.arg_ids,
            )
        return signature_arg_ids

    def _clone_func_signature_args(self, arg_ids: Sequence[int]) -> list[int]:
        """Clone function signature values into another region's entry block."""
        cloned_ids: list[int] = []
        for arg_index, arg_id in enumerate(arg_ids):
            source = self._module.values[arg_id]
            cloned_ids.append(
                self._module.add_value(
                    Value(
                        name=source.name,
                        type=source.type,
                        flags=source.flags,
                        def_result_index=arg_index,
                    )
                )
            )
        return cloned_ids

    def _validate_projected_region_args(
        self,
        op_name: str,
        region_name: str,
        signature_arg_ids: Sequence[int],
        projected_arg_ids: Sequence[int],
    ) -> None:
        """Validate explicit projected region args against the signature."""
        if len(projected_arg_ids) != len(signature_arg_ids):
            raise ValueError(
                f"Op '{op_name}' region '{region_name}' entry block has "
                f"{len(projected_arg_ids)} args but function signature has "
                f"{len(signature_arg_ids)}."
            )
        for arg_index, (signature_id, projected_id) in enumerate(
            zip(signature_arg_ids, projected_arg_ids, strict=True)
        ):
            signature_type = self._module.values[signature_id].type
            projected_type = self._module.values[projected_id].type
            if projected_type != signature_type:
                raise ValueError(
                    f"Op '{op_name}' region '{region_name}' arg {arg_index} "
                    f"has type {projected_type!r} but function signature arg "
                    f"has type {signature_type!r}."
                )

    def _insert_operation(self, op_decl: Op, operation: Operation) -> None:
        """Insert operation into module symbol state or the current block."""
        if op_decl.has_trait("SymbolDefine"):
            symbol_index = self._module.add_symbol(
                symbol_from_operation(operation, op_decl)
            )
            record_operation_value_metadata(
                self._module,
                operation,
                block_index=VALUE_DEF_BLOCK_NONE,
                op_index=symbol_index,
                operand_def_count=len(operation.operands)
                if not operation.regions
                else 0,
            )
            return

        if self._insertion_block is None:
            raise ValueError(
                f"Op '{op_decl.name}' is not a module symbol and no insertion "
                "block is set."
            )
        self._insertion_block.ops.append(operation)
        block_index = self._find_block_index(self._insertion_block)
        if block_index != VALUE_DEF_BLOCK_NONE:
            record_operation_value_metadata(
                self._module,
                operation,
                block_index=block_index,
                op_index=len(self._insertion_block.ops) - 1,
            )

    def _find_block_index(self, target_block: Block) -> int:
        """Returns target_block's index in an attached region, if any."""
        for symbol in self._module.symbols:
            if symbol.op is None:
                continue
            block_index = self._find_block_index_in_operation(symbol.op, target_block)
            if block_index is not None:
                return block_index
        return VALUE_DEF_BLOCK_NONE

    def _find_block_index_in_operation(
        self,
        operation: Operation,
        target_block: Block,
    ) -> int | None:
        """Returns target_block's index in one of operation's nested regions."""
        for region in operation.regions:
            for block_index, block in enumerate(region.blocks):
                if block is target_block:
                    return block_index
                for nested_operation in block.ops:
                    nested_block_index = self._find_block_index_in_operation(
                        nested_operation,
                        target_block,
                    )
                    if nested_block_index is not None:
                        return nested_block_index
        return None

    # --- Generic op building ---

    def build(
        self,
        op_name: str,
        operands: Sequence[ValueRef | int] | None = None,
        *,
        operand_segment_counts: Sequence[int] | None = None,
        successors: Sequence[Block] | None = None,
        func_args: Sequence[ValueRef | int] | None = None,
        results: Sequence[Type | TiedResultSpec] | None = None,
        result_names: Sequence[str] | None = None,
        attributes: Mapping[str, Any] | None = None,
        regions: Sequence[Region] | None = None,
        location_id: int | None = None,
    ) -> ValueRef | list[ValueRef] | None:
        """Build an operation and return the result ValueRef(s).

        Parameters:
          op_name: Full op name (e.g., "test.addi").
          operands: ValueRefs or ints for operands.
          operand_segment_counts: Structural segment counts for ops with
            multiple named operand spans.
          successors: Target blocks for CFG terminator edges.
          func_args: Func-like signature argument ValueRefs. Declaration-style
            funcs store these as op operands; bodyful funcs store them as body
            entry block args.
          results: Result specs. Each is either:
            - A Type (fresh result): INDEX, F32, tensor_t
            - A TiedResultSpec (tied): tied(operand, type) or operand.as_type(type)
          result_names: SSA names for results (auto-generated if None).
          attributes: Op attributes.
          regions: Nested regions.
          location_id: Optional explicit operation location. When omitted,
            the builder's current location context is used.

        Returns:
          Single fixed result: ValueRef.
          Variadic results: list[ValueRef], even if the concrete count is 0 or 1.
          Multiple results: list[ValueRef].
          No results: None.
        """
        op_decl = self._op_registry.get(op_name)
        if op_decl is None:
            raise ValueError(
                f"Unknown op '{op_name}'. Register it with register_ops()."
            )

        operand_ids = self._resolve_operands(operands or [])
        result_specs = results or []
        attr_dict = canonicalize_attr_dict(attributes)
        region_list = list(regions) if regions else []
        tied_operand_ids = self._prepare_func_like_args(
            op_decl,
            operand_ids,
            region_list,
            func_args,
        )

        # Process result specs: extract types and tied bindings.
        result_ids: list[int] = []
        tied_results: list[IRTiedResult] = []
        for i, spec in enumerate(result_specs):
            if isinstance(spec, TiedResultSpec):
                # Tied result: find the operand index.
                operand_index = tied_operand_ids.index(spec.operand_id)
                tied_results.append(
                    IRTiedResult(result_index=i, operand_index=operand_index)
                )
                result_type = spec.result_type
            elif isinstance(spec, ScalarType | ShapedType):
                result_type = spec
            else:
                result_type = spec  # Any Type

            name = ""
            if result_names and i < len(result_names):
                name = result_names[i]
            value_id = self._module.add_value(Value(name=name, type=result_type))
            result_ids.append(value_id)

        operation = Operation(
            name=op_name,
            operands=operand_ids,
            operand_segment_counts=tuple(operand_segment_counts or ()),
            results=result_ids,
            tied_results=tied_results,
            successors=list(successors) if successors else [],
            attributes=attr_dict,
            regions=region_list,
            location_id=self._location_id if location_id is None else location_id,
        )
        self._insert_operation(op_decl, operation)

        result_refs = [ValueRef(result_id, self) for result_id in result_ids]
        if any(result.variadic for result in op_decl.results):
            return result_refs
        if len(result_refs) == 0:
            return None
        if len(result_refs) == 1:
            return result_refs[0]
        return result_refs
