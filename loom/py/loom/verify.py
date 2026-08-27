# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Table-driven verification for Python-built Loom modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from loom.diagnostics import DiagnosticEngine
from loom.dsl import (
    FuncLikeInterface,
    InlinePolicy,
    Op,
    SymbolReferenceRole,
    TypeConstraint,
    type_constraint_name,
)
from loom.fields import FieldKind, FieldLayout, compute_layout, resolve_fields
from loom.ir import (
    Block,
    BufferType,
    DynamicDim,
    DynamicEncoding,
    EncodingRole,
    EncodingType,
    Module,
    Operation,
    PoolType,
    Region,
    RegisterType,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    StorageType,
    Symbol,
    SymbolNameArray,
    SymbolNameSet,
    Type,
    TypeKind,
    Value,
)

__all__ = [
    "ModuleVerifier",
    "VerifierRegistry",
    "type_satisfies_constraint",
    "verify_module",
]


@dataclass(frozen=True, slots=True)
class VerifierRegistry:
    """Resolved op declarations used by the Python verifier."""

    ops: tuple[Op, ...]
    op_by_name: dict[str, Op] = field(init=False)
    layout_by_name: dict[str, FieldLayout] = field(init=False)

    def __post_init__(self) -> None:
        op_by_name = {op.name: op for op in self.ops}
        object.__setattr__(self, "op_by_name", op_by_name)
        object.__setattr__(
            self,
            "layout_by_name",
            {op.name: compute_layout(op) for op in self.ops},
        )

    @classmethod
    def from_ops(cls, ops: Iterable[Op] | None = None) -> VerifierRegistry:
        """Build a verifier registry from op declarations."""
        if ops is None:
            from loom.builders import default_ops

            ops = default_ops()
        return cls(tuple(ops))

    def op(self, name: str) -> Op | None:
        """Return the declaration for an operation name, if registered."""
        return self.op_by_name.get(name)

    def layout(self, op_decl: Op) -> FieldLayout:
        """Return the positional field layout for an op declaration."""
        return self.layout_by_name[op_decl.name]


@dataclass(frozen=True, slots=True)
class _ConstraintRegionValue:
    """Resolved region data consumed by declarative constraint predicates."""

    entry_args: tuple[Value | None, ...] | None
    terminator_operands: tuple[Value | None, ...] | None


@dataclass(slots=True)
class ModuleVerifier:
    """Verifies one in-memory Loom module against op declarations."""

    module: Module
    diagnostics: DiagnosticEngine
    registry: VerifierRegistry = field(default_factory=VerifierRegistry.from_ops)
    _symbols_by_name: dict[str, Symbol] = field(default_factory=dict, init=False)

    def verify(self) -> None:
        """Verify the module and append diagnostics for every detected error."""
        self._verify_symbol_table()
        self._verify_top_level_operation_ownership()
        for operation_index, operation in enumerate(self.module.body.ops):
            self._verify_operation(
                operation,
                f"module operation[{operation_index}]",
                parent_stack=(),
            )
        if not self.diagnostics.has_errors:
            self._verify_keyed_module_records()

    def _verify_symbol_table(self) -> None:
        seen: dict[str, int] = {}
        for symbol_index, symbol in enumerate(self.module.symbols):
            path = f"symbol[{symbol_index}]"
            if not symbol.name:
                self.diagnostics.error("symbol has empty name", source=path)
                continue
            previous = seen.get(symbol.name)
            if previous is not None:
                self.diagnostics.error(
                    "duplicate symbol name",
                    source=f"{path} @{symbol.name}",
                    details=(f"previous definition is symbol[{previous}]",),
                )
                continue
            seen[symbol.name] = symbol_index
            self._symbols_by_name[symbol.name] = symbol

    def _verify_top_level_operation_ownership(self) -> None:
        body_operation_ids = {id(operation) for operation in self.module.body.ops}
        indexed_operation_ids: dict[int, int] = {}
        for symbol_index, symbol in enumerate(self.module.symbols):
            path = f"symbol[{symbol_index}] @{symbol.name}"
            if symbol.op is None:
                self.diagnostics.error(
                    "symbol has no defining operation",
                    source=path,
                )
                continue
            operation_id = id(symbol.op)
            if operation_id not in body_operation_ids:
                self.diagnostics.error(
                    "symbol defining operation is not owned by the module body",
                    source=path,
                )
            previous_symbol_index = indexed_operation_ids.get(operation_id)
            if previous_symbol_index is not None:
                self.diagnostics.error(
                    "symbol defining operation has multiple symbol-table entries",
                    source=path,
                    details=(
                        f"operation is also indexed by symbol[{previous_symbol_index}]",
                    ),
                )
                continue
            indexed_operation_ids[operation_id] = symbol_index

        for operation_index, operation in enumerate(self.module.body.ops):
            op_decl = self.registry.op(operation.name)
            if op_decl is None:
                continue
            if not (
                op_decl.has_trait("SymbolDefine") or op_decl.has_trait("ModuleScope")
            ):
                self.diagnostics.error(
                    "op is not permitted at module scope",
                    source=f"module operation[{operation_index}] {operation.name}",
                )
                continue
            if not op_decl.has_trait("SymbolDefine"):
                continue
            if id(operation) not in indexed_operation_ids:
                self.diagnostics.error(
                    "module body symbol definition is missing from the symbol table",
                    source=f"module operation[{operation_index}] {operation.name}",
                )

    def _verify_keyed_module_records(self) -> None:
        seen: dict[tuple[str, str], int] = {}
        for operation_index, operation in enumerate(self.module.body.ops):
            if operation.is_dead:
                continue
            declaration = self.registry.op(operation.name)
            if declaration is None or declaration.keyed_module_record_attr is None:
                continue
            key = operation.attributes[declaration.keyed_module_record_attr]
            identity = (operation.name, key)
            previous_index = seen.get(identity)
            if previous_index is not None:
                self.diagnostics.error(
                    "duplicate keyed module record",
                    source=f"module operation[{operation_index}] {operation.name}",
                    details=(
                        f"key is {key!r}",
                        f"previous record is module operation[{previous_index}]",
                    ),
                )
                continue
            seen[identity] = operation_index

    def _verify_operation(
        self,
        operation: Operation,
        path: str,
        *,
        parent_stack: tuple[Operation, ...],
    ) -> None:
        op_path = f"{path} {operation.name}"
        values_ok = self._verify_value_ids(operation.operands, f"{op_path} operands")
        values_ok &= self._verify_value_ids(operation.results, f"{op_path} results")
        self._verify_tied_results(operation, op_path)

        op_decl = self.registry.op(operation.name)
        if op_decl is None:
            self.diagnostics.error("unknown op", source=op_path)
            self._verify_regions(
                operation,
                None,
                op_path,
                parent_stack=(*parent_stack, operation),
            )
            return

        self._verify_traits(op_decl, op_path, parent_stack)
        shape_ok = self._verify_field_counts(op_decl, operation, op_path)
        self._verify_attrs(op_decl, operation, op_path)
        if values_ok and shape_ok:
            self._verify_type_constraints(op_decl, operation, op_path)
            self._verify_declarative_constraints(op_decl, operation, op_path)
        self._verify_symbol_refs(op_decl, operation, op_path)
        self._verify_regions(
            operation,
            op_decl,
            op_path,
            parent_stack=(*parent_stack, operation),
        )

    def _verify_field_counts(
        self,
        op_decl: Op,
        operation: Operation,
        path: str,
    ) -> bool:
        layout = self.registry.layout(op_decl)
        ok = True
        if layout.segmented_operands:
            ok &= self._verify_operand_segments(op_decl, operation, path)
        else:
            ok &= self._verify_operand_count(
                len(operation.operands),
                layout,
                path,
            )
            if operation.operand_segment_counts:
                self.diagnostics.error(
                    "unexpected operand segment counts",
                    source=path,
                    details=("op does not use segmented operands",),
                )
                ok = False
        ok &= self._verify_count(
            "result",
            len(operation.results),
            layout.fixed_result_count,
            layout.variadic_result,
            path,
        )
        ok &= self._verify_count(
            "successor",
            len(operation.successors),
            layout.fixed_successor_count,
            layout.variadic_successor,
            path,
        )
        ok &= self._verify_region_count(
            len(operation.regions),
            layout,
            path,
        )
        if (
            op_decl.name == "scf.if"
            and operation.results
            and len(operation.regions) < 2
        ):
            self.diagnostics.error(
                "missing required else region",
                source=path,
                details=("scf.if with results requires an else region",),
            )
            ok = False
        return ok

    def _verify_operand_segments(
        self,
        op_decl: Op,
        operation: Operation,
        path: str,
    ) -> bool:
        counts = operation.operand_segment_counts
        expected_count = len(op_decl.operands)
        if len(counts) != expected_count:
            self.diagnostics.error(
                "wrong operand segment count",
                source=path,
                details=(f"expected {expected_count}, found {len(counts)}",),
            )
            return False

        ok = True
        total_count = 0
        for operand_decl, segment_count in zip(op_decl.operands, counts, strict=False):
            if segment_count < 0:
                self.diagnostics.error(
                    "negative operand segment count",
                    source=path,
                    details=(
                        f"operand field '{operand_decl.name}' has count "
                        f"{segment_count}",
                    ),
                )
                ok = False
                continue
            total_count += segment_count
            if operand_decl.variadic:
                continue
            if operand_decl.optional:
                if segment_count > 1:
                    self.diagnostics.error(
                        "wrong optional operand segment count",
                        source=path,
                        details=(
                            f"operand field '{operand_decl.name}' expected "
                            f"0..1 values, found {segment_count}",
                        ),
                    )
                    ok = False
                continue
            if segment_count != 1:
                self.diagnostics.error(
                    "wrong required operand segment count",
                    source=path,
                    details=(
                        f"operand field '{operand_decl.name}' expected "
                        f"1 value, found {segment_count}",
                    ),
                )
                ok = False

        if total_count != len(operation.operands):
            self.diagnostics.error(
                "operand segment counts do not sum to operand count",
                source=path,
                details=(
                    f"segment counts sum to {total_count}",
                    f"flat operand count is {len(operation.operands)}",
                ),
            )
            ok = False
        return ok

    def _verify_operand_count(
        self,
        actual_count: int,
        layout: FieldLayout,
        path: str,
    ) -> bool:
        if layout.variadic_operand is not None:
            if actual_count >= layout.fixed_operand_count:
                return True
            self.diagnostics.error(
                "wrong operand count",
                source=path,
                details=(
                    f"expected at least {layout.fixed_operand_count}, "
                    f"found {actual_count}",
                    f"trailing variadic field is '{layout.variadic_operand}'",
                ),
            )
            return False

        max_count = sum(
            1 for desc in layout.fields.values() if desc.kind == FieldKind.OPERAND
        )
        if layout.fixed_operand_count <= actual_count <= max_count:
            return True
        expected = (
            str(max_count)
            if layout.fixed_operand_count == max_count
            else f"{layout.fixed_operand_count}..{max_count}"
        )
        self.diagnostics.error(
            "wrong operand count",
            source=path,
            details=(f"expected {expected}, found {actual_count}",),
        )
        return False

    def _verify_count(
        self,
        noun: str,
        actual_count: int,
        fixed_count: int,
        variadic_field: str | None,
        path: str,
    ) -> bool:
        if variadic_field is None:
            if actual_count == fixed_count:
                return True
            self.diagnostics.error(
                f"wrong {noun} count",
                source=path,
                details=(f"expected {fixed_count}, found {actual_count}",),
            )
            return False

        if actual_count >= fixed_count:
            return True
        self.diagnostics.error(
            f"wrong {noun} count",
            source=path,
            details=(
                f"expected at least {fixed_count}, found {actual_count}",
                f"trailing variadic field is '{variadic_field}'",
            ),
        )
        return False

    def _verify_region_count(
        self,
        actual_count: int,
        layout: FieldLayout,
        path: str,
    ) -> bool:
        if layout.variadic_region is not None:
            if actual_count >= layout.fixed_region_count:
                return True
            self.diagnostics.error(
                "wrong region count",
                source=path,
                details=(
                    f"expected at least {layout.fixed_region_count}, "
                    f"found {actual_count}",
                    f"trailing variadic field is '{layout.variadic_region}'",
                ),
            )
            return False

        if layout.required_region_count <= actual_count <= layout.fixed_region_count:
            return True
        expected = (
            str(layout.fixed_region_count)
            if layout.required_region_count == layout.fixed_region_count
            else (f"{layout.required_region_count}..{layout.fixed_region_count}")
        )
        self.diagnostics.error(
            "wrong region count",
            source=path,
            details=(f"expected {expected}, found {actual_count}",),
        )
        return False

    def _verify_attrs(self, op_decl: Op, operation: Operation, path: str) -> None:
        for attr_def in op_decl.attrs:
            if (
                attr_def.name not in operation.attributes
                and not attr_def.optional
                and attr_def.default is None
            ):
                self.diagnostics.error(
                    "missing required attribute",
                    source=path,
                    details=(f"attribute '{attr_def.name}' is required",),
                )

    def _verify_type_constraints(
        self,
        op_decl: Op,
        operation: Operation,
        path: str,
    ) -> None:
        layout = self.registry.layout(op_decl)
        if layout.segmented_operands:
            self._verify_segmented_operand_type_constraints(
                op_decl,
                operation,
                path,
            )
        else:
            self._verify_value_type_constraints(
                operation.operands,
                op_decl.operands,
                path,
                field_kind="operand",
            )
        self._verify_value_type_constraints(
            operation.results,
            op_decl.results,
            path,
            field_kind="result",
        )

    def _verify_segmented_operand_type_constraints(
        self,
        op_decl: Op,
        operation: Operation,
        path: str,
    ) -> None:
        value_index = 0
        for field_decl, segment_count in zip(
            op_decl.operands, operation.operand_segment_counts, strict=False
        ):
            field_value_ids = operation.operands[
                value_index : value_index + segment_count
            ]
            value_index += segment_count
            for relative_index, value_id in enumerate(field_value_ids):
                value = self.module.values[value_id]
                if type_satisfies_constraint(value.type, field_decl.type_constraint):
                    continue
                display_name = field_decl.name
                if field_decl.variadic:
                    display_name = f"{display_name}[{relative_index}]"
                self.diagnostics.error(
                    "operand type constraint violated",
                    source=path,
                    details=(
                        f"operand '{display_name}' has type {value.type}",
                        f"expected {type_constraint_name(field_decl.type_constraint)}",
                    ),
                )

    def _verify_value_type_constraints(
        self,
        value_ids: Sequence[int],
        field_decls: Sequence[Any],
        path: str,
        *,
        field_kind: str,
    ) -> None:
        value_index = 0
        for field_decl in field_decls:
            field_value_ids = (
                value_ids[value_index:]
                if field_decl.variadic
                else value_ids[value_index : value_index + 1]
            )
            for relative_index, value_id in enumerate(field_value_ids):
                value = self.module.values[value_id]
                if type_satisfies_constraint(value.type, field_decl.type_constraint):
                    continue
                display_name = field_decl.name
                if field_decl.variadic:
                    display_name = f"{display_name}[{relative_index}]"
                self.diagnostics.error(
                    f"{field_kind} type constraint violated",
                    source=path,
                    details=(
                        f"{field_kind} '{display_name}' has type {value.type}",
                        f"expected {type_constraint_name(field_decl.type_constraint)}",
                    ),
                )
            if not field_decl.variadic:
                value_index += 1

    def _verify_declarative_constraints(
        self,
        op_decl: Op,
        operation: Operation,
        path: str,
    ) -> None:
        if not op_decl.constraints:
            return
        field_values = self._constraint_field_values(op_decl, operation, path)
        if field_values is None:
            return
        for constraint in op_decl.constraints:
            ok, message = constraint.check(field_values)
            if ok:
                continue
            self.diagnostics.error(
                f"{constraint.name} constraint violated",
                source=path,
                details=(message,) if message else (),
                error_def=constraint.error,
            )

    def _constraint_field_values(
        self,
        op_decl: Op,
        operation: Operation,
        path: str,
    ) -> dict[str, Any] | None:
        layout = self.registry.layout(op_decl)
        resolved = resolve_fields(layout, operation, self.module)
        values: dict[str, Any] = {}
        try:
            for field_name, field_desc in layout.fields.items():
                match field_desc.kind:
                    case FieldKind.OPERAND | FieldKind.RESULT:
                        if field_desc.variadic:
                            values[field_name] = resolved.values(field_name)
                        elif field_desc.optional and not resolved.is_present(
                            field_name
                        ):
                            values[field_name] = None
                        else:
                            values[field_name] = resolved.value(field_name)
                    case FieldKind.ATTR:
                        values[field_name] = operation.attributes.get(field_name)
                    case FieldKind.REGION:
                        region_decl = op_decl.regions[field_desc.index]
                        if field_desc.variadic:
                            values[field_name] = [
                                self._constraint_region_value(
                                    region, region_decl.terminator
                                )
                                for region in resolved.regions(field_name)
                            ]
                        else:
                            values[field_name] = self._constraint_region_value(
                                resolved.region(field_name), region_decl.terminator
                            )
                    case FieldKind.SUCCESSOR:
                        values[field_name] = (
                            resolved.successors(field_name)
                            if field_desc.variadic
                            else resolved.successor(field_name)
                        )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            self.diagnostics.error(
                "failed to resolve fields for declarative constraints",
                source=path,
                details=(str(exc),),
            )
            return None
        return values

    def _constraint_region_value(
        self,
        region: Region | None,
        expected_terminator: str | None,
    ) -> _ConstraintRegionValue:
        """Resolves a region without hiding structural failures from its owner."""
        if region is None or not region.blocks:
            return _ConstraintRegionValue(None, None)
        entry_block = region.blocks[0]
        entry_args = tuple(
            self.module.values[value_id]
            if 0 <= value_id < len(self.module.values)
            else None
            for value_id in entry_block.arg_ids
        )
        if not entry_block.ops:
            return _ConstraintRegionValue(entry_args, None)
        terminator = entry_block.ops[-1]
        terminator_decl = self.registry.op(terminator.name)
        if (
            terminator_decl is None
            or not terminator_decl.is_terminator
            or (
                expected_terminator is not None
                and terminator.name != expected_terminator
            )
        ):
            return _ConstraintRegionValue(entry_args, None)
        terminator_operands = tuple(
            self.module.values[value_id]
            if 0 <= value_id < len(self.module.values)
            else None
            for value_id in terminator.operands
        )
        return _ConstraintRegionValue(entry_args, terminator_operands)

    def _verify_symbol_refs(
        self,
        op_decl: Op,
        operation: Operation,
        path: str,
    ) -> None:
        for attr_def in op_decl.attrs:
            symbol_ref = attr_def.symbol_ref
            if symbol_ref is None or attr_def.name not in operation.attributes:
                continue
            value = operation.attributes[attr_def.name]
            if attr_def.attr_type == "symbol_array":
                if not isinstance(value, SymbolNameArray):
                    self.diagnostics.error(
                        "symbol reference array attribute must be a SymbolNameArray",
                        source=path,
                        details=(f"attribute '{attr_def.name}' has value {value!r}",),
                    )
                    continue
                target_names = tuple(value)
            elif attr_def.attr_type == "symbol_set":
                if not isinstance(value, SymbolNameSet):
                    self.diagnostics.error(
                        "symbol reference set attribute must be a SymbolNameSet",
                        source=path,
                        details=(f"attribute '{attr_def.name}' has value {value!r}",),
                    )
                    continue
                target_names = tuple(value)
            elif isinstance(value, str):
                target_names = (value,)
            else:
                self.diagnostics.error(
                    "symbol reference attribute must be a string",
                    source=path,
                    details=(f"attribute '{attr_def.name}' has value {value!r}",),
                )
                continue
            for index, target_name in enumerate(target_names):
                field = f"attribute '{attr_def.name}'"
                if attr_def.attr_type in ("symbol_array", "symbol_set"):
                    field += f" element {index}"
                target_symbol = self._symbols_by_name.get(target_name)
                if target_symbol is None:
                    if symbol_ref.role is SymbolReferenceRole.AVAILABILITY:
                        continue
                    self.diagnostics.error(
                        "unresolved symbol reference",
                        source=path,
                        details=(f"{field} references @{target_name}",),
                    )
                    continue
                target_decl = (
                    self.registry.op(target_symbol.op.name)
                    if target_symbol.op is not None
                    else None
                )
                target_symbol_def = target_decl.symbol_def if target_decl else None
                if target_symbol_def is None:
                    continue
                if not symbol_ref.interfaces:
                    continue
                if any(
                    interface in target_symbol_def.interfaces
                    for interface in symbol_ref.interfaces
                ):
                    continue
                self.diagnostics.error(
                    "symbol reference target has wrong interface",
                    source=path,
                    details=(
                        f"{field} references @{target_name}",
                        "expected one of "
                        f"{', '.join(symbol_ref.interfaces)} for {symbol_ref.name}",
                        f"target provides {', '.join(target_symbol_def.interfaces)}",
                    ),
                )

    def _verify_traits(
        self,
        op_decl: Op,
        path: str,
        parent_stack: tuple[Operation, ...],
    ) -> None:
        for trait in op_decl.traits:
            match trait.name:
                case "ModuleScope":
                    if parent_stack:
                        self.diagnostics.error(
                            "module-scope op is nested",
                            source=path,
                            details=(
                                "operation must be a direct child of the module body",
                            ),
                        )
                case "HasParent":
                    parent_name = parent_stack[-1].name if parent_stack else None
                    if not trait.args or parent_name == trait.args[0]:
                        continue
                    self.diagnostics.error(
                        "op has wrong parent",
                        source=path,
                        details=(f"expected parent op '{trait.args[0]}'",),
                    )
                case "HasAncestor":
                    if trait.args and any(
                        operation.name == trait.args[0] for operation in parent_stack
                    ):
                        continue
                    # Templates and required-inline functions are verified
                    # before their final placement context is known.
                    if self._has_deferred_required_ancestor(parent_stack):
                        continue
                    expected = trait.args[0] if trait.args else "<missing>"
                    self.diagnostics.error(
                        "op is missing required ancestor",
                        source=path,
                        details=(f"expected ancestor op '{expected}'",),
                    )
                case "NoAncestor":
                    if not trait.args:
                        continue
                    if any(
                        operation.name == trait.args[0] for operation in parent_stack
                    ):
                        self.diagnostics.error(
                            "op has forbidden ancestor",
                            source=path,
                            details=(f"forbidden ancestor op '{trait.args[0]}'",),
                        )

    def _has_deferred_required_ancestor(
        self, parent_stack: tuple[Operation, ...]
    ) -> bool:
        for operation in reversed(parent_stack):
            declaration = self.registry.op(operation.name)
            if operation.name == "template.def":
                return True
            if declaration:
                for interface in declaration.interfaces:
                    if not isinstance(interface, FuncLikeInterface):
                        continue
                    inline_policy = interface.inline_policy
                    if (
                        inline_policy
                        and operation.attributes.get(inline_policy)
                        == InlinePolicy.INLINE.value
                    ):
                        return True
            if declaration and any(
                trait.name == "IsolatedFromAbove" for trait in declaration.traits
            ):
                return False
        return False

    def _verify_regions(
        self,
        operation: Operation,
        op_decl: Op | None,
        path: str,
        *,
        parent_stack: tuple[Operation, ...],
    ) -> None:
        for region_index, region in enumerate(operation.regions):
            region_path = f"{path}.regions[{region_index}]"
            region_decl = (
                op_decl.regions[region_index]
                if op_decl is not None and region_index < len(op_decl.regions)
                else None
            )
            if (
                region_decl is not None
                and region_decl.single_block
                and len(region.blocks) != 1
            ):
                self.diagnostics.error(
                    "single-block region has wrong block count",
                    source=region_path,
                    details=(f"expected 1, found {len(region.blocks)}",),
                )
            for block_index, block in enumerate(region.blocks):
                self._verify_block(
                    block,
                    f"{region_path}.blocks[{block_index}]",
                    region_terminator=region_decl.terminator
                    if region_decl is not None
                    else None,
                    parent_stack=parent_stack,
                )

    def _verify_block(
        self,
        block: Block,
        path: str,
        *,
        region_terminator: str | None,
        parent_stack: tuple[Operation, ...],
    ) -> None:
        self._verify_value_ids(block.arg_ids, f"{path} args")
        for operation_index, operation in enumerate(block.ops):
            op_decl = self.registry.op(operation.name)
            op_path = f"{path}.ops[{operation_index}]"
            if (
                op_decl is not None
                and op_decl.is_terminator
                and operation_index != len(block.ops) - 1
            ):
                self.diagnostics.error(
                    "terminator is not last in block",
                    source=op_path,
                )
            self._verify_operation(operation, op_path, parent_stack=parent_stack)
        if region_terminator is None:
            return
        if not block.ops:
            self.diagnostics.error(
                "block is missing required terminator",
                source=path,
                details=(f"expected terminator '{region_terminator}'",),
            )
            return
        terminator = block.ops[-1]
        terminator_decl = self.registry.op(terminator.name)
        if terminator.name == region_terminator:
            return
        if terminator_decl is not None and terminator_decl.is_terminator:
            self.diagnostics.error(
                "region terminated by wrong op",
                source=path,
                details=(f"expected '{region_terminator}', found '{terminator.name}'",),
            )
            return
        self.diagnostics.error(
            "block is missing required terminator",
            source=path,
            details=(f"expected terminator '{region_terminator}'",),
        )

    def _verify_tied_results(self, operation: Operation, path: str) -> None:
        entry_arg_count = 0
        if operation.regions and operation.regions[0].blocks:
            entry_arg_count = len(operation.regions[0].blocks[0].arg_ids)
        max_tie_operand_count = len(operation.operands) + entry_arg_count
        for tied_result in operation.tied_results:
            if tied_result.result_index < 0 or tied_result.result_index >= len(
                operation.results
            ):
                self.diagnostics.error(
                    "tied result references missing result",
                    source=path,
                    details=(
                        f"result index {tied_result.result_index} is outside "
                        f"[0, {len(operation.results)})",
                    ),
                )
            if (
                tied_result.operand_index < 0
                or tied_result.operand_index >= max_tie_operand_count
            ):
                self.diagnostics.error(
                    "tied result references missing operand",
                    source=path,
                    details=(
                        f"operand index {tied_result.operand_index} is outside "
                        f"[0, {max_tie_operand_count})",
                    ),
                )

    def _verify_value_ids(self, value_ids: Sequence[int], source: str) -> bool:
        value_count = len(self.module.values)
        ok = True
        for value_id in value_ids:
            if value_id < 0 or value_id >= value_count:
                self.diagnostics.error(
                    "operation references missing value",
                    source=source,
                    details=(f"value id {value_id} is outside [0, {value_count})",),
                )
                ok = False
                continue
            value = self.module.values[value_id]
            ok &= self._verify_value_bindings(value, value_id, source)
        return ok

    def _verify_value_bindings(self, value: Any, value_id: int, source: str) -> bool:
        ok = True
        required_dim_positions = set(_dynamic_dim_positions(value.type))
        provided_dim_positions = set(value.dim_bindings)
        missing_dim_positions = sorted(required_dim_positions - provided_dim_positions)
        if missing_dim_positions:
            positions = ", ".join(str(position) for position in missing_dim_positions)
            self.diagnostics.error(
                "dynamic dimension has no SSA binding",
                source=source,
                details=(
                    f"value {value_id} has dynamic dimension position(s) "
                    f"{positions} without dim_bindings entries",
                ),
            )
            ok = False
        unexpected_dim_positions = sorted(
            provided_dim_positions - required_dim_positions
        )
        if unexpected_dim_positions:
            positions = ", ".join(
                str(position) for position in unexpected_dim_positions
            )
            self.diagnostics.error(
                "static dimension has unexpected SSA binding",
                source=source,
                details=(
                    f"value {value_id} has dim_bindings entries for "
                    f"non-dynamic dimension position(s) {positions}",
                ),
            )
            ok = False
        for dim_position, binding_id in value.dim_bindings.items():
            if binding_id < 0 or binding_id >= len(self.module.values):
                self.diagnostics.error(
                    "dynamic dimension references missing value",
                    source=source,
                    details=(
                        f"value {value_id} dim binding {dim_position} "
                        f"references value id {binding_id}",
                    ),
                )
                ok = False
        if _has_dynamic_encoding(value.type) and value.encoding_binding < 0:
            self.diagnostics.error(
                "dynamic encoding has no SSA binding",
                source=source,
                details=(f"value {value_id} has dynamic encoding without binding",),
            )
            ok = False
        if value.encoding_binding >= 0 and value.encoding_binding >= len(
            self.module.values
        ):
            self.diagnostics.error(
                "dynamic encoding references missing value",
                source=source,
                details=(
                    f"value {value_id} references encoding value id "
                    f"{value.encoding_binding}",
                ),
            )
            ok = False
        return ok


def _dynamic_dim_positions(value_type: Type) -> tuple[int, ...]:
    if isinstance(value_type, ShapedType):
        return tuple(
            position
            for position, dim in enumerate(value_type.dims)
            if isinstance(dim, DynamicDim)
        )
    if isinstance(value_type, PoolType) and isinstance(
        value_type.block_size, DynamicDim
    ):
        return (0,)
    return ()


def _has_dynamic_encoding(value_type: Type) -> bool:
    return isinstance(value_type, ShapedType) and isinstance(
        value_type.encoding, DynamicEncoding
    )


def verify_module(
    module: Module,
    *,
    ops: Iterable[Op] | VerifierRegistry | None = None,
    diagnostics: DiagnosticEngine | None = None,
) -> DiagnosticEngine:
    """Verify a module and return the diagnostic engine used."""
    diagnostic_engine = diagnostics if diagnostics is not None else DiagnosticEngine()
    registry = (
        ops if isinstance(ops, VerifierRegistry) else VerifierRegistry.from_ops(ops)
    )
    ModuleVerifier(module, diagnostic_engine, registry).verify()
    return diagnostic_engine


def type_satisfies_constraint(value_type: Type, constraint: TypeConstraint) -> bool:
    """Return true if an IR type satisfies a declarative type constraint."""
    if constraint == TypeConstraint.ANY:
        return True
    if constraint == TypeConstraint.BUFFER:
        return isinstance(value_type, BufferType)
    if constraint == TypeConstraint.POOL:
        return isinstance(value_type, PoolType)
    if constraint == TypeConstraint.REGISTER:
        return isinstance(value_type, RegisterType)
    if constraint == TypeConstraint.STORAGE:
        return isinstance(value_type, StorageType)
    if constraint == TypeConstraint.ANY_ENCODING:
        return isinstance(value_type, EncodingType)
    if constraint == TypeConstraint.ENCODING_LAYOUT:
        return _encoding_role_is(value_type, EncodingRole.LAYOUT)
    if constraint == TypeConstraint.ENCODING_SCHEMA:
        return _encoding_role_is(value_type, EncodingRole.SCHEMA)
    if constraint == TypeConstraint.ENCODING_STORAGE:
        return _encoding_role_is(value_type, EncodingRole.STORAGE)
    if constraint == TypeConstraint.ENCODING_TRANSFORM:
        return _encoding_role_is(value_type, EncodingRole.TRANSFORM)
    if constraint == TypeConstraint.I1:
        return _scalar_kind_is(value_type, ScalarTypeKind.I1)
    if constraint == TypeConstraint.I32:
        return _scalar_kind_is(value_type, ScalarTypeKind.I32)

    if isinstance(value_type, ScalarType):
        return _scalar_satisfies_constraint(value_type.kind, constraint)
    if isinstance(value_type, ShapedType):
        return _shaped_satisfies_constraint(value_type, constraint)
    return False


def _encoding_role_is(value_type: Type, role: EncodingRole) -> bool:
    return isinstance(value_type, EncodingType) and value_type.role == role


def _scalar_kind_is(value_type: Type, kind: ScalarTypeKind) -> bool:
    return isinstance(value_type, ScalarType) and value_type.kind == kind


def _scalar_satisfies_constraint(
    scalar_kind: ScalarTypeKind,
    constraint: TypeConstraint,
) -> bool:
    if constraint == TypeConstraint.SCALAR:
        return True
    if constraint == TypeConstraint.INDEX:
        return scalar_kind == ScalarTypeKind.INDEX
    if constraint == TypeConstraint.OFFSET:
        return scalar_kind == ScalarTypeKind.OFFSET
    if constraint == TypeConstraint.ADDRESS:
        return scalar_kind in {ScalarTypeKind.INDEX, ScalarTypeKind.OFFSET}
    if constraint == TypeConstraint.INTEGER:
        return scalar_kind in _INTEGER_SCALAR_KINDS
    if constraint == TypeConstraint.FLOAT:
        return scalar_kind in _FLOAT_SCALAR_KINDS
    if constraint == TypeConstraint.BITWISE_SCALAR:
        return scalar_kind in _BITWISE_SCALAR_KINDS
    if constraint == TypeConstraint.BYTE_PATTERN_SCALAR:
        return scalar_kind in _BYTE_PATTERN_SCALAR_KINDS
    if constraint == TypeConstraint.INDEX_OR_NON_I1_INTEGER_SCALAR:
        return scalar_kind in _INDEX_OR_NON_I1_INTEGER_SCALAR_KINDS
    return False


def _shaped_satisfies_constraint(
    shaped_type: ShapedType,
    constraint: TypeConstraint,
) -> bool:
    if constraint == TypeConstraint.TILE:
        return shaped_type.type_kind == TypeKind.TILE
    if constraint == TypeConstraint.TENSOR:
        return shaped_type.type_kind == TypeKind.TENSOR
    if constraint == TypeConstraint.VECTOR:
        return shaped_type.type_kind == TypeKind.VECTOR
    if constraint == TypeConstraint.VIEW:
        return shaped_type.type_kind == TypeKind.VIEW
    if constraint == TypeConstraint.RANK_ONE_VECTOR:
        return shaped_type.type_kind == TypeKind.VECTOR and shaped_type.rank == 1
    if constraint == TypeConstraint.ALL_STATIC_VECTOR:
        return shaped_type.type_kind == TypeKind.VECTOR and shaped_type.is_all_static
    if constraint == TypeConstraint.ALL_STATIC_RANK_ONE_VECTOR:
        return (
            shaped_type.type_kind == TypeKind.VECTOR
            and shaped_type.rank == 1
            and shaped_type.is_all_static
        )
    if constraint == TypeConstraint.INTEGER_ELEMENT:
        return shaped_type.element_type.kind in _INTEGER_SCALAR_KINDS
    if constraint == TypeConstraint.FLOAT_ELEMENT:
        return shaped_type.element_type.kind in _FLOAT_SCALAR_KINDS
    if constraint == TypeConstraint.BITWISE_ELEMENT:
        return shaped_type.element_type.kind in _BITWISE_SCALAR_KINDS
    if constraint == TypeConstraint.INDEX_OR_NON_I1_INTEGER_ELEMENT:
        return shaped_type.element_type.kind in _INDEX_OR_NON_I1_INTEGER_SCALAR_KINDS
    if constraint == TypeConstraint.I1_ELEMENT:
        return shaped_type.element_type.kind == ScalarTypeKind.I1
    if constraint == TypeConstraint.I8_ELEMENT:
        return shaped_type.element_type.kind == ScalarTypeKind.I8
    if constraint == TypeConstraint.I32_ELEMENT:
        return shaped_type.element_type.kind == ScalarTypeKind.I32
    if constraint == TypeConstraint.F16_OR_BF16_ELEMENT:
        return shaped_type.element_type.kind in {
            ScalarTypeKind.F16,
            ScalarTypeKind.BF16,
        }
    if constraint == TypeConstraint.F32_ELEMENT:
        return shaped_type.element_type.kind == ScalarTypeKind.F32
    return False


_INTEGER_SCALAR_KINDS = frozenset(
    {
        ScalarTypeKind.I1,
        ScalarTypeKind.I8,
        ScalarTypeKind.I16,
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
    }
)

_FLOAT_SCALAR_KINDS = frozenset(
    {
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        ScalarTypeKind.BF16,
        ScalarTypeKind.F32,
        ScalarTypeKind.F64,
    }
)

_BYTE_PATTERN_SCALAR_KINDS = frozenset(
    {
        ScalarTypeKind.I8,
        ScalarTypeKind.I16,
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        ScalarTypeKind.BF16,
        ScalarTypeKind.F32,
        ScalarTypeKind.F64,
    }
)

_BITWISE_SCALAR_KINDS = frozenset({ScalarTypeKind.INDEX, *_BYTE_PATTERN_SCALAR_KINDS})

_INDEX_OR_NON_I1_INTEGER_SCALAR_KINDS = frozenset(
    {
        ScalarTypeKind.INDEX,
        ScalarTypeKind.I8,
        ScalarTypeKind.I16,
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
    }
)
