# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bytecode symbol dependency projection over shared type and attribute graphs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from loom.dsl import SymbolReferenceRole
from loom.format.bytecode.op_decls import attr_def_for_op
from loom.ir import (
    DialectType,
    EncodingInstance,
    FunctionType,
    Module,
    Operation,
    ParameterizedAttr,
    ParameterizedAttrArray,
    ParameterizedType,
    ReferenceType,
    Region,
    RegisterType,
    ScalarType,
    ShapedType,
    SymbolName,
    SymbolNameArray,
    SymbolNameSet,
)

SYMBOL_INTERFACE_BITS = {
    "func_like": 1 << 0,
    "global": 1 << 1,
    "executable": 1 << 2,
    "record": 1 << 3,
    "target": 1 << 4,
    "config": 1 << 5,
    "rodata": 1 << 6,
    "kernel": 1 << 7,
    "callable": 1 << 8,
    "command_program": 1 << 9,
    "template_family": 1 << 10,
    "template_provider": 1 << 11,
    "kernel_entry": 1 << 12,
    "pipeline": 1 << 13,
}

SYMBOL_REFERENCE_CONTRACT_ROLE_SHIFT = 14
SYMBOL_REFERENCE_CONTRACT_ROLE_MASK = 0x3 << SYMBOL_REFERENCE_CONTRACT_ROLE_SHIFT

_SYMBOL_REFERENCE_ROLE_BITS = {
    SymbolReferenceRole.DEPENDENCY: 0,
    SymbolReferenceRole.TEST_SUPPORT_DEPENDENCY: 2,
}


def symbol_reference_dependency_contract(
    target_interfaces: int, role: SymbolReferenceRole
) -> int:
    """Packs one durable dependency contract."""
    try:
        role_bits = _SYMBOL_REFERENCE_ROLE_BITS[role]
    except KeyError as exc:
        raise ValueError(f"{role.value} references are not dependencies") from exc
    return target_interfaces | (role_bits << SYMBOL_REFERENCE_CONTRACT_ROLE_SHIFT)


def symbol_reference_dependency_contract_parts(contract: int) -> tuple[int, int]:
    """Validates and unpacks one durable dependency contract."""
    contract_mask = (
        (1 << SYMBOL_REFERENCE_CONTRACT_ROLE_SHIFT) - 1
    ) | SYMBOL_REFERENCE_CONTRACT_ROLE_MASK
    if contract & ~contract_mask:
        raise ValueError(f"dependency contract has unknown bits {contract:#x}")
    role_bits = (
        contract & SYMBOL_REFERENCE_CONTRACT_ROLE_MASK
    ) >> SYMBOL_REFERENCE_CONTRACT_ROLE_SHIFT
    if role_bits not in _SYMBOL_REFERENCE_ROLE_BITS.values():
        raise ValueError(f"unknown dependency role {role_bits}")
    return contract & ((1 << SYMBOL_REFERENCE_CONTRACT_ROLE_SHIFT) - 1), role_bits


@dataclass(frozen=True, slots=True)
class _SymbolReferenceSourceScope:
    """Symbol and independently serializable root that own a reference."""

    # Owning wire symbol, or module scope when absent.
    symbol_index: int | None = None
    # Independently decoded root ordinal, or zero for the symbol contract.
    root_region_index_plus_one: int = 0


@dataclass(frozen=True, slots=True)
class _SymbolReferenceRecord:
    """Wire symbol reference with its source contract or root region."""

    # Owning root ordinal, or zero for the symbol contract.
    source_root_region_index_plus_one: int
    # Target ordinal in the wire symbol table.
    target_symbol_index: int
    # Required interface flags; zero carries no interface restriction.
    target_interfaces: int = 0
    # Reachability and test-support role of this dependency.
    role: SymbolReferenceRole = SymbolReferenceRole.DEPENDENCY

    @property
    def contract(self) -> int:
        return symbol_reference_dependency_contract(self.target_interfaces, self.role)


@dataclass(frozen=True, slots=True)
class _SymbolReferenceTarget:
    """Source-independent leaf of a structural occurrence summary."""

    # Resolved wire symbol ordinal.
    symbol_index: int
    # Interfaces required by the owning descriptor.
    interfaces: int
    # Reachability and test-support role of this dependency.
    role: SymbolReferenceRole


type _SymbolReferenceSummary = (
    _SymbolReferenceTarget | tuple[_SymbolReferenceSummary, ...]
)


class SymbolReferenceProjectionBuilder:
    """Builds wire-symbol dependency and abstract-provider rows."""

    def __init__(
        self,
        module: Module,
        encodings: Sequence[EncodingInstance],
        wire_symbol_indices: dict[str, int],
        op_decls_by_name: Mapping[str, Any],
    ) -> None:
        self._module = module
        # The writer owns the completed static catalog, including reached entries
        # that were not explicitly listed on the source module.
        self._encodings = encodings
        self._wire_symbol_indices = wire_symbol_indices
        self._op_decls_by_name = op_decls_by_name
        self._module_dependencies: list[_SymbolReferenceRecord] = []
        self._symbol_dependencies: list[list[_SymbolReferenceRecord]] = [
            [] for _ in wire_symbol_indices
        ]
        self._symbol_template_demands: list[list[_SymbolReferenceRecord]] = [
            [] for _ in wire_symbol_indices
        ]
        # Source and descriptor references keep storage identity keys valid.
        # Summaries expire after projection; only wire occurrence rows escape.
        self._summaries: dict[
            tuple[int, int], tuple[Any, Any, _SymbolReferenceSummary]
        ] = {}

    def build(
        self,
    ) -> tuple[
        tuple[_SymbolReferenceRecord, ...],
        tuple[tuple[_SymbolReferenceRecord, ...], ...],
        tuple[tuple[_SymbolReferenceRecord, ...], ...],
    ]:
        """Builds rows in the linked-list order used by the C analysis."""
        module_scope = _SymbolReferenceSourceScope()
        for operation in self._module.body.ops:
            self._visit_operation(module_scope, operation)
        for encoding in self._encodings:
            self._visit_attr(module_scope, encoding)
        return (
            tuple(reversed(self._module_dependencies)),
            tuple(tuple(reversed(row)) for row in self._symbol_dependencies),
            tuple(tuple(reversed(row)) for row in self._symbol_template_demands),
        )

    def _summary_parts(
        self, value: Any, attr_def: Any | None
    ) -> tuple[_SymbolReferenceSummary, tuple[tuple[Any, Any], ...]]:
        """Resolve direct leaves or immediate edges without flattening types."""
        attr_type = getattr(attr_def, "attr_type", None)
        symbol_ref = getattr(attr_def, "symbol_ref", None)
        names = None
        if isinstance(value, ReferenceType):
            names = (str(value.declaration),)
        elif attr_type == "symbol" or isinstance(value, SymbolName):
            names = (str(value),)
        elif attr_type in ("symbol_array", "symbol_set") or isinstance(
            value, SymbolNameArray | SymbolNameSet
        ):
            names = value
        if names is not None:
            if (
                symbol_ref is not None
                and symbol_ref.role is SymbolReferenceRole.AVAILABILITY
            ):
                return (), ()
            interfaces = (
                SYMBOL_INTERFACE_BITS["record"]
                if isinstance(value, ReferenceType)
                else 0
            )
            role = SymbolReferenceRole.DEPENDENCY
            if symbol_ref is not None:
                role = symbol_ref.role
                for interface in symbol_ref.interfaces:
                    interfaces |= SYMBOL_INTERFACE_BITS[interface]
            targets = []
            for name in names:
                try:
                    target = self._wire_symbol_indices[str(name)]
                except KeyError as exc:
                    raise ValueError(f"unresolved symbol dependency {name!r}") from exc
                targets.append(_SymbolReferenceTarget(target, interfaces, role))
            return self._compact(tuple(targets)), ()
        children = ()
        match value:
            case ShapedType(element_type=element, encoding=encoding):
                children = (element, encoding)
            case FunctionType(arg_types=args, result_types=results):
                children = (*args, *results)
            case DialectType(params=parameters):
                children = parameters
            case ParameterizedType() | ParameterizedAttr():
                parameters = (
                    value.definition.params
                    if isinstance(value, ParameterizedType)
                    else value.definition.parameters
                )
                return (), tuple(
                    (slot, parameter)
                    for parameter, slot in zip(parameters, value.slots, strict=True)
                    if slot is not None
                )
            case RegisterType(value_type=child) if child is not None:
                children = (child,)
            case EncodingInstance(params=parameters):
                children = tuple(child for _, child in parameters)
            case ParameterizedAttrArray(values=values):
                children = values
            case Mapping():
                children = tuple(value.values())
            case list() | tuple():
                children = value
            case _:
                pass
        return (), tuple((child, None) for child in children)

    @staticmethod
    def _compact(
        children: tuple[_SymbolReferenceSummary, ...],
    ) -> _SymbolReferenceSummary:
        """Prune empty edges and contract unary paths, retaining multiplicity."""
        children = tuple(child for child in children if child)
        return children[0] if len(children) == 1 else children

    def _visit_attr(
        self,
        source_scope: _SymbolReferenceSourceScope,
        value: Any,
        attr_def: Any | None = None,
    ) -> None:
        # Scalars and SSA-bound shapes cannot carry symbol references. Keep this
        # ordinary path out of the graph index, as in the native classification.
        if isinstance(value, (int, float, bytes, bytearray, ScalarType)):
            return
        if type(value) is str and getattr(attr_def, "attr_type", None) != "symbol":
            return
        if isinstance(value, ShapedType) and not isinstance(
            value.encoding, EncodingInstance
        ):
            return
        root_key = (id(value), id(attr_def))
        pending = [(value, attr_def, False)]
        while pending:
            current, descriptor, expanded = pending.pop()
            key = (id(current), id(descriptor))
            if key in self._summaries:
                continue
            leaf, children = self._summary_parts(current, descriptor)
            if children and not expanded:
                pending.append((current, descriptor, True))
                pending.extend((child, desc, False) for child, desc in children)
                continue
            summary = (
                self._compact(
                    tuple(
                        self._summaries[(id(child), id(desc))][2]
                        for child, desc in children
                    )
                )
                if children
                else leaf
            )
            self._summaries[key] = (current, descriptor, summary)
        # Compact summaries expand only actual output occurrences. Repeated
        # edges retain multiplicity; source/root ownership is applied here.
        summaries = [self._summaries[root_key][2]]
        while summaries:
            summary = summaries.pop()
            if isinstance(summary, _SymbolReferenceTarget):
                record = _SymbolReferenceRecord(
                    source_root_region_index_plus_one=source_scope.root_region_index_plus_one,
                    target_symbol_index=summary.symbol_index,
                    target_interfaces=summary.interfaces,
                    role=summary.role,
                )
                if source_scope.symbol_index is None:
                    self._module_dependencies.append(record)
                else:
                    self._symbol_dependencies[source_scope.symbol_index].append(record)
            else:
                summaries.extend(reversed(summary))

    def _visit_value(
        self, source_scope: _SymbolReferenceSourceScope, value_id: int
    ) -> None:
        if 0 <= value_id < len(self._module.values):
            self._visit_attr(source_scope, self._module.values[value_id].type)

    def _visit_region(
        self, source_scope: _SymbolReferenceSourceScope, region: Region
    ) -> None:
        for block in region.blocks:
            for argument_id in block.arg_ids:
                self._visit_value(source_scope, argument_id)
            for operation in block.ops:
                self._visit_operation(source_scope, operation)

    def _visit_operation(
        self, source_scope: _SymbolReferenceSourceScope, operation: Operation
    ) -> None:
        op_decl = self._op_decls_by_name.get(operation.name)
        symbol_def = getattr(op_decl, "symbol_def", None)
        nested_source_scope = source_scope
        defines_symbol = False
        if symbol_def is not None:
            symbol_name = operation.attributes.get(symbol_def.field)
            if isinstance(symbol_name, str):
                try:
                    nested_source_scope = _SymbolReferenceSourceScope(
                        symbol_index=self._wire_symbol_indices[symbol_name]
                    )
                    defines_symbol = True
                except KeyError as exc:
                    raise ValueError(
                        f"symbol-defining operation {operation.name!r} names "
                        f"unindexed symbol {symbol_name!r}"
                    ) from exc

        if operation.name == "template.apply":
            family = operation.attributes.get("family")
            if nested_source_scope.symbol_index is None:
                raise ValueError("template.apply is not owned by a module symbol")
            if not isinstance(family, str):
                raise ValueError("template.apply family must be a symbol")
            try:
                family_symbol_ordinal = self._wire_symbol_indices[family]
            except KeyError as exc:
                raise ValueError(
                    f"template.apply references unknown family {family!r}"
                ) from exc
            self._symbol_template_demands[nested_source_scope.symbol_index].append(
                _SymbolReferenceRecord(
                    source_root_region_index_plus_one=(
                        nested_source_scope.root_region_index_plus_one
                    ),
                    target_symbol_index=family_symbol_ordinal,
                )
            )

        for value_id in (*operation.operands, *operation.results):
            self._visit_value(nested_source_scope, value_id)
        for key, value in operation.attributes.items():
            if symbol_def is not None and key == symbol_def.field:
                continue
            self._visit_attr(
                nested_source_scope,
                value,
                attr_def_for_op(self._op_decls_by_name, operation.name, key),
            )
        for region_index, region in enumerate(operation.regions):
            child_source_scope = nested_source_scope
            if defines_symbol:
                child_source_scope = _SymbolReferenceSourceScope(
                    symbol_index=nested_source_scope.symbol_index,
                    root_region_index_plus_one=region_index + 1,
                )
            self._visit_region(child_source_scope, region)
