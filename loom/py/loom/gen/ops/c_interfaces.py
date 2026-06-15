# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Interface vtable emission for generated C op metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loom.dsl import (
    ATTR_TYPE_FLAGS,
    CallLikeInterface,
    CallLikeKind,
    FuncLikeInterface,
    LoopLikeInterface,
    MemoryAccessInterface,
    Op,
    RegionBranchInterface,
    TargetLikeInterface,
)
from loom.fields import compute_layout
from loom.gen.ops import c_queries, c_symbols
from loom.gen.ops.c_enums import CALL_LIKE_KIND_MAP
from loom.gen.ops.c_names import c_prefix

# Interfaces declared in the Python DSL are emitted as per-op .rodata vtable
# structs on the C side. Each interface also has a pointer slot on cache line 3
# of loom_op_vtable_t.
#
# Adding a new interface is four steps:
#   1. Declare the Python NamedTuple in dsl.py.
#   2. Add the C struct, fat reference, cast function, and inline accessors to
#      ir.h / op_defs.{h,c}.
#   3. Add a pointer slot on cache line 3 of loom_op_vtable_t in ir.h.
#   4. Add an InterfaceSpec entry to INTERFACES below.
#
# The generator code is entirely table-driven off INTERFACES. Adding an
# interface does not require any changes to the emission loops.


@dataclass(frozen=True, slots=True)
class InterfaceFieldSpec:
    """Describes one field in an interface's C vtable struct.

    py_field: Attribute name on the Python NamedTuple, such as "body",
        "callee", or "iv". The generator reads this via getattr.
    c_field: Field name in the C struct, such as "body_region_index" or
        "callee_attr_index". Emitted as the designated initializer key.
    kind: How to resolve the Python value to a C integer index or literal.
        One of "attr", "region", "operand", "result", "block_arg", "bool",
        "call_kind", or "c_ptr".
    region_field: For kind="block_arg", the name of the interface field that
        names the region the block arg belongs to. Unused for other kinds.
    required: True when a soft default is still semantically required and must
        resolve to a declared op field.
    c_pointee_type: For kind="c_ptr", the const pointee type used to emit an
        extern declaration for the referenced C symbol.
    """

    py_field: str
    c_field: str
    kind: str
    region_field: str = ""
    required: bool = False
    c_pointee_type: str = ""


@dataclass(frozen=True, slots=True)
class InterfaceSpec:
    """Generator metadata for one interface type.

    python_class: The Python NamedTuple class declared in dsl.py.
    name: Human-readable interface name used in error messages.
    c_struct: Name of the C vtable struct type.
    vtable_field: Field name on loom_op_vtable_t and the per-op .rodata
        constant suffix. These share a name by convention so the generator
        only needs one identifier per interface.
    fields: Ordered tuple of InterfaceFieldSpec for every field on the C
        struct. Order matches the desired .rodata emission order.
    """

    python_class: type
    name: str
    c_struct: str
    vtable_field: str
    fields: tuple[InterfaceFieldSpec, ...]


# Registry of all interfaces known to the generator. Adding a new interface is
# a single entry here plus the C-side struct/cast code.
INTERFACES: tuple[InterfaceSpec, ...] = (
    InterfaceSpec(
        python_class=CallLikeInterface,
        name="CallLikeInterface",
        c_struct="loom_call_like_vtable_t",
        vtable_field="call_like",
        fields=(
            InterfaceFieldSpec("callee", "callee_attr_index", "attr"),
            InterfaceFieldSpec("purity", "purity_attr_index", "attr"),
            InterfaceFieldSpec("temperature", "temperature_attr_index", "attr"),
            InterfaceFieldSpec("inline_policy", "inline_policy_attr_index", "attr"),
            InterfaceFieldSpec("operands", "operand_offset", "operand"),
            InterfaceFieldSpec("results", "result_offset", "result"),
            InterfaceFieldSpec("kind", "kind", "call_kind"),
        ),
    ),
    InterfaceSpec(
        python_class=FuncLikeInterface,
        name="FuncLikeInterface",
        c_struct="loom_func_like_vtable_t",
        vtable_field="func_like",
        fields=(
            InterfaceFieldSpec("callee", "callee_attr_index", "attr"),
            InterfaceFieldSpec("import_module", "import_module_attr_index", "attr"),
            InterfaceFieldSpec("import_symbol", "import_symbol_attr_index", "attr"),
            InterfaceFieldSpec("target", "target_attr_index", "attr"),
            InterfaceFieldSpec("abi", "abi_attr_index", "attr"),
            InterfaceFieldSpec("abi_attrs", "abi_attrs_attr_index", "attr"),
            InterfaceFieldSpec("export_symbol", "export_symbol_attr_index", "attr"),
            InterfaceFieldSpec("export_attrs", "export_attrs_attr_index", "attr"),
            InterfaceFieldSpec("export_linkage", "export_linkage_attr_index", "attr"),
            InterfaceFieldSpec("visibility", "visibility_attr_index", "attr"),
            InterfaceFieldSpec("cc", "cc_attr_index", "attr"),
            InterfaceFieldSpec("purity", "purity_attr_index", "attr"),
            InterfaceFieldSpec("temperature", "temperature_attr_index", "attr"),
            InterfaceFieldSpec("inline_policy", "inline_policy_attr_index", "attr"),
            InterfaceFieldSpec("predicates", "predicates_attr_index", "attr"),
            InterfaceFieldSpec("body", "body_region_index", "region"),
            InterfaceFieldSpec("implements", "implements_attr_index", "attr"),
            InterfaceFieldSpec("priority", "priority_attr_index", "attr"),
            InterfaceFieldSpec("args_as_operands", "args_as_operands", "bool"),
        ),
    ),
    InterfaceSpec(
        python_class=TargetLikeInterface,
        name="TargetLikeInterface",
        c_struct="loom_target_like_vtable_t",
        vtable_field="target_like",
        fields=(
            InterfaceFieldSpec("symbol", "symbol_attr_index", "attr"),
            InterfaceFieldSpec("selector", "selector_attr_index", "attr"),
            InterfaceFieldSpec("extensions", "extension_attrs_attr_index", "attr"),
            InterfaceFieldSpec(
                "descriptor",
                "descriptor",
                "c_ptr",
                c_pointee_type="loom_target_like_descriptor_t",
            ),
        ),
    ),
    InterfaceSpec(
        python_class=LoopLikeInterface,
        name="LoopLikeInterface",
        c_struct="loom_loop_like_vtable_t",
        vtable_field="loop_like",
        fields=(
            InterfaceFieldSpec("body", "body_region_index", "region"),
            InterfaceFieldSpec("condition_region", "condition_region_index", "region"),
            InterfaceFieldSpec("iv", "iv_block_arg_index", "block_arg", region_field="body"),
            InterfaceFieldSpec("iter_args", "iter_args_operand_field_index", "operand"),
            InterfaceFieldSpec("lower_bound", "lower_bound_operand_index", "operand"),
            InterfaceFieldSpec("upper_bound", "upper_bound_operand_index", "operand"),
            InterfaceFieldSpec("step", "step_operand_index", "operand"),
        ),
    ),
    InterfaceSpec(
        python_class=RegionBranchInterface,
        name="RegionBranchInterface",
        c_struct="loom_region_branch_vtable_t",
        vtable_field="region_branch",
        fields=(InterfaceFieldSpec("selector", "selector_operand_index", "operand"),),
    ),
    InterfaceSpec(
        python_class=MemoryAccessInterface,
        name="MemoryAccessInterface",
        c_struct="loom_memory_access_vtable_t",
        vtable_field="memory_access",
        fields=(
            InterfaceFieldSpec("view", "view_operand_index", "operand", required=True),
            InterfaceFieldSpec("value", "value_operand_index", "operand"),
            InterfaceFieldSpec("expected", "expected_operand_index", "operand"),
            InterfaceFieldSpec("replacement", "replacement_operand_index", "operand"),
            InterfaceFieldSpec("mask", "mask_operand_index", "operand"),
            InterfaceFieldSpec("passthrough", "passthrough_operand_index", "operand"),
            InterfaceFieldSpec("offsets", "offsets_operand_index", "operand"),
            InterfaceFieldSpec("indices", "indices_operand_offset", "operand"),
            InterfaceFieldSpec("static_indices", "static_indices_attr_index", "attr"),
            InterfaceFieldSpec("cache_scope", "cache_scope_attr_index", "attr"),
            InterfaceFieldSpec("cache_temporal", "cache_temporal_attr_index", "attr"),
            InterfaceFieldSpec("atomic_kind", "atomic_kind_attr_index", "attr"),
            InterfaceFieldSpec("atomic_ordering", "atomic_ordering_attr_index", "attr"),
            InterfaceFieldSpec("atomic_success_ordering", "atomic_success_ordering_attr_index", "attr"),
            InterfaceFieldSpec("atomic_failure_ordering", "atomic_failure_ordering_attr_index", "attr"),
            InterfaceFieldSpec("atomic_scope", "atomic_scope_attr_index", "attr"),
        ),
    ),
)


_TARGET_PROJECTION_FIELDS: dict[str, tuple[str, str]] = {
    "codegen_format": ("LOOM_TARGET_PROJECTION_VALUE_ENUM_U8", "snapshot.codegen_format"),
    "artifact_format": ("LOOM_TARGET_PROJECTION_VALUE_ENUM_U8", "snapshot.artifact_format"),
    "default_pointer_bitwidth": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.default_pointer_bitwidth"),
    "index_bitwidth": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.index_bitwidth"),
    "offset_bitwidth": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.offset_bitwidth"),
    "max_workgroup_size_x": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_size.x"),
    "max_workgroup_size_y": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_size.y"),
    "max_workgroup_size_z": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_size.z"),
    "max_flat_workgroup_size": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_flat_workgroup_size"),
    "subgroup_size": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.subgroup_size"),
    "max_grid_size_x": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_grid_size.x"),
    "max_grid_size_y": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_grid_size.y"),
    "max_grid_size_z": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_grid_size.z"),
    "max_flat_grid_size": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64", "snapshot.max_flat_grid_size"),
    "max_workgroup_count_x": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_count.x"),
    "max_workgroup_count_y": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_count.y"),
    "max_workgroup_count_z": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.max_workgroup_count.z"),
    "memory_space_generic": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.generic"),
    "memory_space_global": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.global"),
    "memory_space_workgroup": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.workgroup"),
    "memory_space_constant": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.constant"),
    "memory_space_private": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.private_memory"),
    "memory_space_host": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.host"),
    "memory_space_descriptor": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "snapshot.memory_spaces.descriptor"),
    "abi": ("LOOM_TARGET_PROJECTION_VALUE_ENUM_U8", "export_plan.abi_kind"),
    "export_symbol": ("LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW", "export_plan.export_symbol"),
    "linkage": ("LOOM_TARGET_PROJECTION_VALUE_ENUM_U8", "export_plan.linkage"),
    "hal_buffer_resource_flags": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32", "export_plan.hal_kernel.buffer_resource_flags"),
    "contract_set_key": ("LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW", "config.contract_set_key"),
    "contract_feature_bits": ("LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64", "config.contract_feature_bits"),
}


def _interface_field_is_explicit(iface: Any, field_name: str) -> bool:
    """Returns true when the declaration explicitly supplied an interface field."""
    explicit_fields = getattr(iface, "_explicit_fields", None)
    return explicit_fields is None or field_name in explicit_fields


def _interface_reference_exists(
    op: Op,
    iface: Any,
    field_spec: InterfaceFieldSpec,
    py_value: Any,
) -> bool:
    """Returns true if an implicitly defaulted interface reference exists."""
    if py_value is None:
        return True
    if field_spec.kind == "attr":
        return any(attr_def.name == py_value and attr_def.attr_type != ATTR_TYPE_FLAGS for attr_def in op.attrs)
    if field_spec.kind == "region":
        return any(region_def.name == py_value for region_def in op.regions)
    if field_spec.kind == "operand":
        return any(operand.name == py_value for operand in op.operands)
    if field_spec.kind == "result":
        return any(result.name == py_value for result in op.results)
    if field_spec.kind == "block_arg":
        region_value = getattr(iface, field_spec.region_field)
        region_def = next(
            (region for region in op.regions if region.name == region_value),
            None,
        )
        if region_def is None:
            return False
        return any(name == py_value for name, _type in region_def.implicit_args)
    return True


def _interface_soft_default_is_absent(
    op: Op,
    iface: Any,
    field_spec: InterfaceFieldSpec,
    py_value: Any,
) -> bool:
    """Returns true when a non-required soft default should emit none."""
    if field_spec.required:
        return False
    if _interface_field_is_explicit(iface, field_spec.py_field):
        return False
    return not _interface_reference_exists(op, iface, field_spec, py_value)


def _resolve_interface_field(
    op: Op,
    iface: Any,
    field_spec: InterfaceFieldSpec,
    interface_name: str,
) -> str:
    """Resolves one interface field to its emitted C initializer value."""
    py_value = getattr(iface, field_spec.py_field)
    if _interface_soft_default_is_absent(op, iface, field_spec, py_value):
        return "255"
    if field_spec.kind == "attr":
        return str(c_queries.resolve_attr_index(op, py_value, interface_name))
    if field_spec.kind == "region":
        return str(c_queries.resolve_region_index(op, py_value, interface_name))
    if field_spec.kind == "operand":
        return str(c_queries.resolve_operand_index(op, py_value, interface_name))
    if field_spec.kind == "result":
        return str(c_queries.resolve_result_index(op, py_value, interface_name))
    if field_spec.kind == "block_arg":
        if not field_spec.region_field:
            raise ValueError(f"{interface_name} field {field_spec.py_field!r}: kind='block_arg' requires region_field to name the region this block arg belongs to")
        region_value = getattr(iface, field_spec.region_field)
        return str(c_queries.resolve_block_arg_index(op, region_value, py_value, interface_name))
    if field_spec.kind == "bool":
        return "true" if py_value else "false"
    if field_spec.kind == "call_kind":
        if not isinstance(py_value, CallLikeKind):
            raise ValueError(f"{interface_name} field {field_spec.py_field!r}: expected CallLikeKind, got {py_value!r}")
        return CALL_LIKE_KIND_MAP[py_value]
    if field_spec.kind == "c_ptr":
        if isinstance(iface, TargetLikeInterface) and field_spec.py_field == "descriptor" and iface.bundle_table is not None:
            descriptor = iface.descriptor or f"{c_prefix(op)}_target_like_descriptor"
            return f"&{c_symbols.normalize_c_symbol_reference(descriptor)}"
        if py_value is None:
            return "NULL"
        if not isinstance(py_value, str) or not py_value:
            raise ValueError(f"{interface_name} field {field_spec.py_field!r}: expected C symbol name or None, got {py_value!r}")
        return f"&{c_symbols.normalize_c_symbol_reference(py_value)}"
    raise ValueError(f"{interface_name} field {field_spec.py_field!r}: unknown kind {field_spec.kind!r}")


def _resolve_soft_memory_field(
    op: Op,
    iface: MemoryAccessInterface,
    field_name: str,
    field_kind: str,
    interface_name: str,
) -> int | None:
    """Resolves a MemoryAccessInterface field, returning None for absent defaults."""
    field_spec = InterfaceFieldSpec(field_name, "", field_kind)
    py_value = getattr(iface, field_name)
    if _interface_soft_default_is_absent(op, iface, field_spec, py_value):
        return None
    if field_kind == "attr":
        index = c_queries.resolve_attr_index(op, py_value, interface_name)
    elif field_kind == "operand":
        index = c_queries.resolve_operand_index(op, py_value, interface_name)
    else:
        raise ValueError(f"unsupported MemoryAccessInterface field kind {field_kind!r}")
    return None if index == 0xFF else index


def _validate_call_like_interface(op: Op, iface: CallLikeInterface, interface_name: str) -> None:
    """Validates CallLikeInterface's trailing variadic slice contract."""
    operand_index = c_queries.resolve_operand_index(op, iface.operands, interface_name)
    operand = op.operands[operand_index]
    if not operand.variadic:
        raise ValueError(f"{interface_name} on {op.name!r}: operand {iface.operands!r} must be variadic")
    if operand_index + 1 != len(op.operands):
        raise ValueError(f"{interface_name} on {op.name!r}: operand {iface.operands!r} must be the trailing operand field")

    result_index = c_queries.resolve_result_index(op, iface.results, interface_name)
    result = op.results[result_index]
    if not result.variadic:
        raise ValueError(f"{interface_name} on {op.name!r}: result {iface.results!r} must be variadic")
    if result_index + 1 != len(op.results):
        raise ValueError(f"{interface_name} on {op.name!r}: result {iface.results!r} must be the trailing result field")


def _validate_memory_access_interface(op: Op, iface: MemoryAccessInterface, interface_name: str) -> None:
    """Validates MemoryAccessInterface's optional role coherence."""
    if iface.view is None:
        raise ValueError(f"{interface_name} on {op.name!r}: view operand is required")
    c_queries.resolve_operand_index(op, iface.view, interface_name)
    indices_index = _resolve_soft_memory_field(op, iface, "indices", "operand", interface_name)
    if indices_index is not None:
        indices_operand = op.operands[indices_index]
        if not indices_operand.variadic:
            raise ValueError(f"{interface_name} on {op.name!r}: operand {iface.indices!r} must be variadic")
        if indices_index + 1 != len(op.operands):
            raise ValueError(f"{interface_name} on {op.name!r}: operand {iface.indices!r} must be the trailing operand field")

    cache_scope_index = _resolve_soft_memory_field(op, iface, "cache_scope", "attr", interface_name)
    cache_temporal_index = _resolve_soft_memory_field(op, iface, "cache_temporal", "attr", interface_name)
    if (cache_scope_index is None) != (cache_temporal_index is None):
        raise ValueError(f"{interface_name} on {op.name!r}: cache_scope and cache_temporal must be declared together")

    atomic_ordering_index = _resolve_soft_memory_field(op, iface, "atomic_ordering", "attr", interface_name)
    atomic_success_ordering_index = _resolve_soft_memory_field(op, iface, "atomic_success_ordering", "attr", interface_name)
    atomic_failure_ordering_index = _resolve_soft_memory_field(op, iface, "atomic_failure_ordering", "attr", interface_name)
    if (atomic_success_ordering_index is None) != (atomic_failure_ordering_index is None):
        raise ValueError(f"{interface_name} on {op.name!r}: atomic_success_ordering and atomic_failure_ordering must be declared together")
    if atomic_ordering_index is not None and atomic_success_ordering_index is not None:
        raise ValueError(f"{interface_name} on {op.name!r}: use either atomic_ordering or success/failure orderings, not both")


def _target_like_projection_entries(op: Op) -> list[tuple[int, str, str]]:
    entries: list[tuple[int, str, str]] = []
    for attr_def in op.attrs:
        projection = _TARGET_PROJECTION_FIELDS.get(attr_def.name)
        if projection is None:
            continue
        value_kind, storage_field = projection
        attr_index = c_queries.resolve_attr_index(op, attr_def.name, "TargetLikeInterface")
        entries.append((attr_index, value_kind, storage_field))
    return entries


def emit_target_like_descriptor(op: Op, iface: TargetLikeInterface, lines: list[str]) -> None:
    if iface.bundle_table is None:
        return
    descriptor = c_symbols.normalize_c_symbol_reference(iface.descriptor or f"{c_prefix(op)}_target_like_descriptor")
    bundle_table = c_symbols.normalize_c_symbol_reference(iface.bundle_table)
    prefix = c_prefix(op)
    projections = _target_like_projection_entries(op)
    projection_array = "NULL"
    if projections:
        projection_array = f"{prefix}_target_projections"
        lines.append(f"static const loom_target_projection_t {projection_array}[] = {{")
        for attr_index, value_kind, storage_field in projections:
            lines.append(f"    {{offsetof(loom_target_bundle_storage_t, {storage_field}), {attr_index}, {value_kind}}},")
        lines.append("};")
    lines.append(f"static const loom_target_like_descriptor_t {descriptor} = {{")
    lines.append(f"    .bundle_table = &{bundle_table},")
    if projection_array != "NULL":
        lines.append(f"    .projections = {projection_array},")
        lines.append(f"    .projection_count = IREE_ARRAYSIZE({projection_array}),")
    lines.append("};")
    lines.append("")


def emit_interface_vtable(op: Op, spec: InterfaceSpec, lines: list[str]) -> None:
    """Appends the .rodata struct declaration for |op|'s |spec| interface."""
    iface = c_queries.find_interface(op, spec.python_class)
    if iface is None:
        return
    if isinstance(iface, CallLikeInterface):
        _validate_call_like_interface(op, iface, spec.name)
    if isinstance(iface, MemoryAccessInterface):
        _validate_memory_access_interface(op, iface, spec.name)
    prefix = c_prefix(op)
    lines.append(f"static const {spec.c_struct} {prefix}_{spec.vtable_field} = {{")
    for field_spec in spec.fields:
        value_str = _resolve_interface_field(op, iface, field_spec, spec.name)
        lines.append(f"    .{field_spec.c_field} = {value_str},")
    if isinstance(iface, LoopLikeInterface):
        layout = compute_layout(op)
        lines.append(f"    .operand_field_count = {len(op.operands)},")
        lines.append(f"    .segmented_operands = {'true' if layout.segmented_operands else 'false'},")
    lines.append("};")
    lines.append("")


def interface_vtable_ptr(op: Op, spec: InterfaceSpec) -> str:
    """Returns the C expression for the interface pointer on the main vtable."""
    if c_queries.find_interface(op, spec.python_class) is None:
        return "NULL"
    return f"&{c_prefix(op)}_{spec.vtable_field}"


def interface_c_ptr_symbols(ops: Sequence[Op]) -> list[tuple[str, str]]:
    """Returns sorted (type, symbol) extern declarations for interface C pointers."""
    declarations: set[tuple[str, str]] = set()
    for op in ops:
        for spec in INTERFACES:
            iface = c_queries.find_interface(op, spec.python_class)
            if iface is None:
                continue
            for field_spec in spec.fields:
                if field_spec.kind != "c_ptr":
                    continue
                if isinstance(iface, TargetLikeInterface) and field_spec.py_field == "descriptor" and iface.bundle_table is not None:
                    continue
                if not field_spec.c_pointee_type:
                    raise ValueError(f"{spec.name} field {field_spec.py_field!r}: kind='c_ptr' requires c_pointee_type")
                py_value = getattr(iface, field_spec.py_field)
                if py_value is None:
                    continue
                if not isinstance(py_value, str) or not py_value:
                    raise ValueError(f"{spec.name} field {field_spec.py_field!r}: expected C symbol name or None, got {py_value!r}")
                declarations.add((field_spec.c_pointee_type, c_symbols.normalize_c_symbol_reference(py_value)))
    return sorted(declarations)


def target_like_bundle_table_symbols(ops: Sequence[Op]) -> list[str]:
    symbols: set[str] = set()
    for op in ops:
        iface = c_queries.find_interface(op, TargetLikeInterface)
        if iface is None or iface.bundle_table is None:
            continue
        symbols.add(c_symbols.normalize_c_symbol_reference(iface.bundle_table))
    return sorted(symbols)
