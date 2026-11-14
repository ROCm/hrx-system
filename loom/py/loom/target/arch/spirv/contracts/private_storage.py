# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Function-private cells addressed through retained logical positions."""

from dataclasses import replace

from loom.dialect.buffer import defs as buffer
from loom.dialect.scf import defs as scf
from loom.dialect.view import defs as view
from loom.target.arch.spirv.contracts.descriptor_rule import (
    IntegerRecipe,
    descriptor_feature_guards,
    logical_core_descriptor,
)
from loom.target.arch.spirv.contracts.memory import (
    STORAGE_BUFFER_MEMORY_SPACES,
    storage_buffer_address_materializer,
    storage_buffer_alignment_diagnostic,
    storage_buffer_source_memory,
)
from loom.target.arch.spirv.scalar_memory import (
    SOURCE_STORAGE_BUFFER_SCALARS,
    StorageBufferScalar,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    SourceMemoryOperation,
    SourceValueKind,
    ValueMaterializer,
    ValueRef,
    View,
)

_PRIVATE_REPRESENTATION_BASE = 16  # lower/private_storage.h
_HEADER = "loom/target/arch/spirv/lower/private_storage.h"
PRIVATE_STORAGE_MATERIALIZER = ValueMaterializer(
    name="spirv.private_storage",
    can_materialize="loom_spirv_can_materialize_private_storage",
    prepare="loom_spirv_prepare_private_storage",
    result_type="loom_spirv_private_storage_type",
    materialize="loom_spirv_materialize_private_storage",
    header=_HEADER,
)
PRIVATE_ORIGIN_MATERIALIZER = replace(
    PRIVATE_STORAGE_MATERIALIZER,
    name="spirv.private_origin",
    result_type="loom_spirv_private_origin_type",
    materialize="loom_spirv_materialize_private_origin",
)
PRIVATE_POSITION_MATERIALIZER = ValueMaterializer(
    name="spirv.private_position",
    can_materialize="loom_spirv_can_materialize_private_position",
    prepare="loom_spirv_prepare_private_position",
    result_type="loom_spirv_private_position_type",
    materialize="loom_spirv_materialize_private_position",
    header=_HEADER,
)
PRIVATE_STORAGE_MATERIALIZERS = (
    PRIVATE_STORAGE_MATERIALIZER,
    PRIVATE_ORIGIN_MATERIALIZER,
    PRIVATE_POSITION_MATERIALIZER,
)

_ADDRESS = storage_buffer_address_materializer(SOURCE_STORAGE_BUFFER_SCALARS[0])
_PRIVATE_ADDRESS = replace(
    _ADDRESS, address=logical_core_descriptor("spirv.op_iadd.offset64")
)


def _emit(
    key: str,
    operands: dict[str, ValueRef],
    result: ValueRef | None = None,
    *,
    immediates: dict[str, int] | None = None,
    form: DescriptorEmitForm = DescriptorEmitForm.OP,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=logical_core_descriptor(f"spirv.op_{key}"),
        operands=operands,
        results={"dst": result} if result else {},
        result_types=(
            {"dst": DescriptorResultType()}
            if result and result.kind is SourceValueKind.TEMPORARY
            else None
        ),
        immediates=immediates or {},
        form=form,
    )


def _constant(value: int, name: str) -> tuple[ValueRef, EmitDescriptorOp]:
    result = ValueRef.temporary(name)
    return result, _emit(
        "constant.offset64",
        {},
        result,
        immediates={"offset64_value": value},
        form=DescriptorEmitForm.CONST,
    )


def _memory_rule(
    scalar: StorageBufferScalar,
    cell_shift: int,
    *,
    is_load: bool,
    raw: bool = False,
    bfloat_carrier: bool = False,
    dynamic: bool = False,
) -> DescriptorRule:
    cell_width = 1 << cell_shift
    cell_suffix = f"i{cell_width * 8}"
    source_types = (
        ("i8", "f8E4M3", "f8E5M2") if scalar.source_type == "i8" else scalar.source_type
    )
    source_field = "source" if raw and is_load else "target" if raw else "view"
    source_op = (
        (buffer.buffer_load_i8_u if is_load else buffer.buffer_store_i8)
        if raw
        else (view.view_load if is_load else view.view_store)
    )
    operation = SourceMemoryOperation.LOAD if is_load else SourceMemoryOperation.STORE
    constraint = replace(
        storage_buffer_source_memory(operation, scalar),
        memory_spaces=("private",),
        minimum_alignment=cell_width,
        dynamic_term_count=None if dynamic else 0,
        dynamic_term_count_minimum=1 if dynamic else 0,
        allow_dynamic_stride_values=dynamic,
        alignment_diagnostic=None
        if raw
        else storage_buffer_alignment_diagnostic(cell_width),
    )
    origin = ValueRef.operand(
        source_field, materializer=PRIVATE_ORIGIN_MATERIALIZER.name
    )
    storage = ValueRef.operand(
        source_field, materializer=PRIVATE_STORAGE_MATERIALIZER.name
    )
    address = ValueRef.temporary("private_address")
    emits = [
        replace(
            _emit(
                "isub.offset64",
                {"lhs": ValueRef.source_memory_address(), "rhs": origin},
                address,
            ),
            source_memory=constraint,
            source_memory_address_materializer=_PRIVATE_ADDRESS,
        )
    ]
    source_value = ValueRef.operand("value")
    final_result = ValueRef.result("result")
    payload_width = 32 if raw or bfloat_carrier else max(32, scalar.byte_width * 8)
    recipe = IntegerRecipe(payload_width, 1)
    integer_suffix = f"i{payload_width}"
    is_float = scalar.source_type.startswith("f") or scalar.source_type == "bf16"
    direct = scalar.byte_width == cell_width and not raw and not bfloat_carrier
    payload = source_value
    if not is_load and not bfloat_carrier and not raw:
        if is_float:
            payload = recipe.import_float_bits(
                source_value, scalar.suffix, scalar.byte_width * 8
            )
        elif not direct and scalar.byte_width * 8 != payload_width:
            payload = recipe.emit(
                f"s_convert.{scalar.suffix}.{integer_suffix}", {"input": source_value}
            )
    recipe_cursor = 0
    # Keep payload transformations before any store. Load transformations are
    # appended as cells arrive, preserving their source access semantics.
    emits.extend(recipe.emits[recipe_cursor:])
    recipe_cursor = len(recipe.emits)
    loaded = None
    for part in range(scalar.byte_width // cell_width):
        offset = address
        if part:
            displacement, constant = _constant(part * cell_width, f"cell_offset_{part}")
            offset = ValueRef.temporary(f"cell_address_{part}")
            emits.extend(
                (
                    constant,
                    _emit(
                        "iadd.offset64", {"lhs": address, "rhs": displacement}, offset
                    ),
                )
            )
        pointer = ValueRef.temporary(f"cell_pointer_{part}")
        emits.append(
            EmitDescriptorOp(
                descriptor=logical_core_descriptor(
                    f"spirv.op_access_chain.function.{cell_suffix}.byte_offset"
                ),
                operands={"base": storage, "byte_offset": offset},
                results={"ptr": pointer},
                result_types={"ptr": DescriptorResultType()},
            )
        )
        cell = ValueRef.temporary(f"cell_value_{part}")
        if not is_load:
            cell_payload = payload
            if part:
                cell_payload = recipe.binary(
                    "shift_right_logical",
                    cell_payload,
                    recipe.constant(part * cell_width * 8),
                )
            if direct and not is_float:
                cell = source_value
            elif cell_width * 8 == payload_width:
                cell = cell_payload
            else:
                cell = recipe.emit(
                    f"s_convert.{integer_suffix}.{cell_suffix}", {"input": cell_payload}
                )
            emits.extend(recipe.emits[recipe_cursor:])
            recipe_cursor = len(recipe.emits)
        transfer = _emit(
            f"{'load' if is_load else 'store'}.function.{cell_suffix}",
            {"ptr": pointer} | ({} if is_load else {"value": cell}),
            cell if is_load else None,
        )
        emits.append(replace(transfer, source_memory=constraint))
        if is_load:
            value = cell
            if direct and not is_float:
                loaded = value
                continue
            if cell_width * 8 != payload_width:
                value = recipe.emit(
                    f"s_convert.{cell_suffix}.{integer_suffix}", {"input": value}
                )
                value = recipe.binary(
                    "bitwise_and", value, recipe.constant((1 << (cell_width * 8)) - 1)
                )
            if part:
                value = recipe.binary(
                    "shift_left_logical", value, recipe.constant(part * cell_width * 8)
                )
            loaded = (
                value if loaded is None else recipe.binary("bitwise_or", loaded, value)
            )
            emits.extend(recipe.emits[recipe_cursor:])
            recipe_cursor = len(recipe.emits)
    if is_load:
        if is_float and not bfloat_carrier:
            recipe.export_float_bits(loaded, scalar.suffix, scalar.byte_width * 8)
        else:
            result_suffix = "i32" if raw or bfloat_carrier else scalar.suffix
            actual_suffix = scalar.suffix if direct else integer_suffix
            key = (
                f"copy_object.{result_suffix}"
                if actual_suffix == result_suffix
                else f"s_convert.{actual_suffix}.{result_suffix}"
            )
            recipe.emit(key, {"input": loaded}, result=final_result)
        emits.extend(recipe.emits[recipe_cursor:])
    guards = (
        Guard.value_memory_space(source_field, ("private",)),
        Guard.low_value_representation(
            source_field, _PRIVATE_REPRESENTATION_BASE + cell_shift
        ),
        Guard.value_type(
            "result" if is_load else "value", Scalar("i32" if raw else source_types)
        ),
        *((Guard.value_type("view", View(source_types)),) if not raw else ()),
        *descriptor_feature_guards(*(step.descriptor for step in emits)),
    )
    if bfloat_carrier:
        guards += (
            Guard.low_value_register_class(
                "result" if is_load else "value", "spirv.id"
            ),
        )
    return DescriptorRule(source_op=source_op, guards=guards, emit=tuple(emits))


def _private_subview_rule(dynamic: bool) -> DescriptorRule:
    materializer = replace(
        _ADDRESS, address=logical_core_descriptor("spirv.op_iadd.offset64")
    )
    constraint = replace(
        storage_buffer_source_memory(
            SourceMemoryOperation.VIEW_CARRIER, SOURCE_STORAGE_BUFFER_SCALARS[0]
        ),
        memory_spaces=("private",),
        dynamic_term_count=None if dynamic else 0,
        dynamic_term_count_minimum=1 if dynamic else 0,
        allow_dynamic_stride_values=dynamic,
    )
    return DescriptorRule(
        source_op=view.view_subview,
        guards=(Guard.value_memory_space("result", ("private",)),),
        emit=(
            replace(
                _emit(
                    "copy_object.offset64",
                    {"input": ValueRef.source_memory_address()},
                    ValueRef.result("result"),
                ),
                source_memory=constraint,
                source_memory_address_materializer=materializer,
            ),
        ),
    )


def private_storage_rules() -> tuple[DescriptorRule, ...]:
    rules = [_private_subview_rule(dynamic) for dynamic in (False, True)]

    def private_operand(field: str) -> ValueRef:
        return ValueRef.operand(field, materializer=PRIVATE_POSITION_MATERIALIZER.name)

    rules.append(
        DescriptorRule(
            source_op=scf.scf_select,
            guards=(Guard.value_memory_space("result", ("private",)),),
            emit=(
                _emit(
                    "select.offset64",
                    {
                        "condition": ValueRef.operand("condition"),
                        "true_value": private_operand("true_value"),
                        "false_value": private_operand("false_value"),
                    },
                    ValueRef.result("result"),
                ),
            ),
        )
    )
    zero, constant = _constant(0, "private_null")
    rules.append(
        DescriptorRule(
            source_op=buffer.buffer_compare_null,
            guards=(Guard.value_memory_space("buffer", ("private",)),),
            emit=(
                constant,
                _emit(
                    "i_equal.offset64",
                    {"lhs": ValueRef.operand("buffer"), "rhs": zero},
                    ValueRef.result("result"),
                ),
            ),
        )
    )
    for predicate, suffix in (("eq", "i_equal"), ("ne", "i_not_equal")):
        addresses = tuple(
            _emit(
                "iadd.offset64",
                {
                    "lhs": private_operand(side),
                    "rhs": ValueRef.operand(f"{side}_offset"),
                },
                ValueRef.temporary(f"{side}_address"),
            )
            for side in ("lhs", "rhs")
        )
        rules.append(
            DescriptorRule(
                source_op=buffer.buffer_compare_address,
                guards=(
                    Guard.enum_attr_equals("predicate", predicate),
                    Guard.value_memory_space("lhs", ("private",)),
                    Guard.value_memory_space("rhs", ("private",)),
                ),
                emit=(
                    *addresses,
                    _emit(
                        f"{suffix}.offset64",
                        {
                            "lhs": ValueRef.temporary("lhs_address"),
                            "rhs": ValueRef.temporary("rhs_address"),
                        },
                        ValueRef.result("result"),
                    ),
                ),
            )
        )
        # Function storage and physical device storage cannot alias. Nullable
        # references from the two domains compare equal only when both are null.
        external_suffix = "iequal" if predicate == "eq" else "inotequal"
        for private, external in (("lhs", "rhs"), ("rhs", "lhs")):
            external_zero = ValueRef.temporary("external_null")
            private_test = ValueRef.temporary("private_test")
            external_test = ValueRef.temporary("external_test")
            rules.append(
                DescriptorRule(
                    source_op=buffer.buffer_compare_address,
                    guards=(
                        Guard.enum_attr_equals("predicate", predicate),
                        Guard.value_memory_space(private, ("private",)),
                        Guard.value_memory_space(
                            external, STORAGE_BUFFER_MEMORY_SPACES
                        ),
                    ),
                    emit=(
                        constant,
                        _emit(
                            "constant.storage_buffer",
                            {},
                            external_zero,
                            immediates={"offset64_value": 0},
                            form=DescriptorEmitForm.CONST,
                        ),
                        _emit(
                            f"{suffix}.offset64",
                            {"lhs": private_operand(private), "rhs": zero},
                            private_test,
                        ),
                        _emit(
                            f"{external_suffix}.storage_buffer",
                            {"lhs": ValueRef.operand(external), "rhs": external_zero},
                            external_test,
                        ),
                        _emit(
                            f"logical_{'and' if predicate == 'eq' else 'or'}.bool",
                            {"lhs": private_test, "rhs": external_test},
                            ValueRef.result("result"),
                        ),
                    ),
                )
            )
    for scalar in SOURCE_STORAGE_BUFFER_SCALARS:
        for shift in range(scalar.byte_width.bit_length()):
            for is_load in (False, True):
                for dynamic in (False, True):
                    rules.append(
                        _memory_rule(scalar, shift, is_load=is_load, dynamic=dynamic)
                    )
                    if scalar.source_type == "bf16":
                        rules.append(
                            _memory_rule(
                                scalar,
                                shift,
                                is_load=is_load,
                                bfloat_carrier=True,
                                dynamic=dynamic,
                            )
                        )
    rules.extend(
        _memory_rule(
            SOURCE_STORAGE_BUFFER_SCALARS[0],
            0,
            is_load=is_load,
            raw=True,
            dynamic=dynamic,
        )
        for is_load in (False, True)
        for dynamic in (False, True)
    )
    return tuple(rules)
