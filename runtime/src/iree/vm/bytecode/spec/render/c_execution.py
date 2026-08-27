# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders compact interpreter verification and dispatch rows."""

from __future__ import annotations

from collections.abc import Sequence

from execution import ExecutableInstruction
from model.isa import Instruction, InstructionFieldRole
from model.isa.validation import (
    ABI_SLOT,
    ALLOWED_RANGE,
    ALLOWED_VALUES,
    ANY_BITS,
    CONSTANT_POOL_ORDINAL,
    CONSTRAINT_MEMBER,
    CONTROL_SWITCH_TARGETS,
    CONTROL_TARGET_RELATIVE_S16,
    CONTROL_TARGET_RELATIVE_S32,
    FIELDS_DISTINCT,
    FUNCTION_ADDRESS,
    FUNCTION_LOCAL_ORDINAL,
    GLOBAL_ORDINAL,
    IMPORT_ORDINAL_OPTIONAL,
    LOCAL_BYTES_FIXED_BASE,
    LOCAL_BYTES_RANGE_BASE,
    LOCAL_BYTES_RANGE_LENGTH,
    LOCAL_BYTES_RANGE_MEMORY_FORMAT,
    LOCAL_BYTES_REPEATED_BASE,
    LOCAL_BYTES_REPEATED_COUNT,
    PACKED_SELECTOR_ALLOWED_PAIRS,
    PACKED_SELECTOR_TARGET_SUPPORTED,
    PACKED_SELECTORS,
    REF_SLOT,
    REGISTER_FUNCTION,
    REGISTER_REF,
    REGISTER_VALUE,
    RODATA_OFFSET,
    RODATA_ORDINAL,
    RODATA_STATIC_OFFSET,
    SELECTOR,
    VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT,
    ZERO,
)
from model.isa.validation import (
    CONTROL_CALL as CONTROL_CALL_RULE,
)
from model.isa.validation import (
    CONTROL_CALL_INDIRECT as CONTROL_CALL_INDIRECT_RULE,
)
from model.schema import EntityReference, FieldReference, RuleUse, ScalarEncoding
from model.specification import Projection


def _opcode_token(mnemonic: str) -> str:
    return mnemonic.replace(".", "_").upper()


def _handler_label(mnemonic: str) -> str:
    return mnemonic.replace(".", "_")


def _require_field(
    instruction: Instruction,
    entities_by_id: dict[str, object],
    *,
    offset: int,
    byte_length: int,
    rule_id: str,
    roles: tuple[InstructionFieldRole, ...],
    array_length: int = 1,
    rule_arguments: tuple[object, ...] | None = None,
) -> None:
    fields = tuple(field for field in instruction.fields if field.offset == offset)
    if len(fields) != 1:
        raise ValueError(
            f"{instruction.mnemonic}: verification form requires one field "
            f"at byte {offset}"
        )
    field = fields[0]
    encoding = entities_by_id.get(field.encoding_id)
    if (
        not isinstance(encoding, ScalarEncoding)
        or encoding.byte_length != byte_length
        or field.array_length != array_length
        or field.role not in roles
        or tuple(rule.rule_id for rule in field.validation) != (rule_id,)
        or (
            rule_arguments is not None
            and tuple(field.validation[0].arguments) != rule_arguments
        )
    ):
        raise ValueError(
            f"{instruction.mnemonic}.{field.name}: field does not match its "
            "runtime verification form"
        )


def _validate_verification_form(
    instruction: Instruction,
    verification_form: str,
    entities_by_id: dict[str, object],
) -> None:
    verified_field_offsets: set[int] = set()
    result_or_operand = (
        InstructionFieldRole.RESULT,
        InstructionFieldRole.OPERAND,
    )

    def require_field(
        offset: int,
        byte_length: int,
        rule_id: str,
        roles: tuple[InstructionFieldRole, ...],
        *,
        array_length: int = 1,
        rule_arguments: tuple[object, ...] | None = None,
    ) -> None:
        _require_field(
            instruction,
            entities_by_id,
            offset=offset,
            byte_length=byte_length,
            rule_id=rule_id,
            roles=roles,
            array_length=array_length,
            rule_arguments=rule_arguments,
        )
        verified_field_offsets.add(offset)

    def require_value(offset: int) -> None:
        require_field(
            offset,
            1,
            REGISTER_VALUE.entity_id,
            result_or_operand,
        )

    def require_ref(offset: int) -> None:
        require_field(
            offset,
            1,
            REGISTER_REF.entity_id,
            result_or_operand,
        )

    def require_packed_selector(
        offset: int,
        zero_mask: int,
        components: tuple[tuple[str, int, int, str, tuple[int, ...]], ...],
    ) -> None:
        normalized_components = tuple(
            (
                name,
                bit_offset,
                bit_length,
                EntityReference(selector_table_id),
                allowed_values,
            )
            for name, bit_offset, bit_length, selector_table_id, allowed_values in components
        )
        require_field(
            offset,
            1,
            PACKED_SELECTORS.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(zero_mask, normalized_components),
        )

    def require_function(offset: int) -> None:
        require_field(
            offset,
            1,
            REGISTER_FUNCTION.entity_id,
            result_or_operand,
        )

    def require_ref_slot(offset: int) -> None:
        require_field(
            offset,
            2,
            REF_SLOT.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )

    def require_zero(offset: int, byte_length: int, *, array_length: int = 1) -> None:
        require_field(
            offset,
            byte_length,
            ZERO.entity_id,
            (InstructionFieldRole.PADDING,),
            array_length=array_length,
        )

    def require_lane_range(base_field: str, format_field: str) -> None:
        if len(instruction.constraints) != 1:
            raise ValueError(
                f"{instruction.mnemonic}: lane transfer requires exactly one "
                "record constraint"
            )
        constraint = instruction.constraints[0]
        expected_arguments = (
            FieldReference(base_field),
            FieldReference(format_field),
        )
        if (
            constraint.rule_id != VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT.entity_id
            or tuple(constraint.arguments) != expected_arguments
        ):
            raise ValueError(
                f"{instruction.mnemonic}: lane-range constraint does not match "
                "its runtime verification form"
            )

    if verification_form in ("CONTROL_BLOCK", "CONTROL_RETURN"):
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: control record is not 4 bytes")
        require_zero(1, 1, array_length=3)
    elif verification_form == "CONTROL_BRANCH_S16":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: short branch is not 4 bytes")
        require_zero(1, 1)
        require_field(
            2,
            2,
            CONTROL_TARGET_RELATIVE_S16.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form in ("CONTROL_BRANCH_S32", "CONTROL_YIELD_S32"):
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: wide branch is not 8 bytes")
        require_zero(1, 1, array_length=3)
        require_field(
            4,
            4,
            CONTROL_TARGET_RELATIVE_S32.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "CONTROL_BRANCH_CONDITIONAL_S16":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: short conditional branch is not 4 bytes"
            )
        require_value(1)
        require_field(
            2,
            2,
            CONTROL_TARGET_RELATIVE_S16.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "CONTROL_BRANCH_CONDITIONAL_S32":
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: wide conditional branch is not 8 bytes"
            )
        require_value(1)
        require_zero(2, 2)
        require_field(
            4,
            4,
            CONTROL_TARGET_RELATIVE_S32.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "CONTROL_SWITCH":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: switch is not 8 bytes")
        require_value(1)
        require_field(
            2,
            2,
            CONSTRAINT_MEMBER.entity_id,
            (InstructionFieldRole.CONSTRAINT_MEMBER,),
            rule_arguments=("control.switch.targets",),
        )
        require_field(
            4,
            4,
            CONSTRAINT_MEMBER.entity_id,
            (InstructionFieldRole.CONSTRAINT_MEMBER,),
            rule_arguments=("control.switch.targets",),
        )
        expected_arguments = (
            FieldReference("target_count_u16"),
            FieldReference("target_base_u32"),
        )
        if (
            len(instruction.constraints) != 1
            or instruction.constraints[0].rule_id != CONTROL_SWITCH_TARGETS.entity_id
            or tuple(instruction.constraints[0].arguments) != expected_arguments
        ):
            raise ValueError(
                f"{instruction.mnemonic}: switch-target constraint does not "
                "match its runtime verification form"
            )
    elif verification_form == "CONTROL_CALL":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: direct call is not 8 bytes")
        require_field(
            1,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.control.call.target"),),
        )
        for offset in (2, 4):
            require_field(
                offset,
                2,
                CONSTRAINT_MEMBER.entity_id,
                (InstructionFieldRole.CONSTRAINT_MEMBER,),
                rule_arguments=("control.call",),
            )
        require_zero(6, 2)
        expected_constraint = RuleUse(
            CONTROL_CALL_RULE.entity_id,
            (
                FieldReference("target_kind_u8"),
                FieldReference("target_ordinal_u16"),
                FieldReference("direct_ref_move_mask_u16"),
            ),
        )
        if instruction.constraints != (expected_constraint,):
            raise ValueError(
                f"{instruction.mnemonic}: call constraint does not match its "
                "runtime verification form"
            )
    elif verification_form == "CONTROL_CALL_INDIRECT":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: indirect call is not 8 bytes")
        require_function(1)
        for offset in (2, 4):
            require_field(
                offset,
                2,
                CONSTRAINT_MEMBER.entity_id,
                (InstructionFieldRole.CONSTRAINT_MEMBER,),
                rule_arguments=("control.call.indirect",),
            )
        require_zero(6, 2)
        expected_constraint = RuleUse(
            CONTROL_CALL_INDIRECT_RULE.entity_id,
            (
                FieldReference("target_f8"),
                FieldReference("callable_type_ordinal_u16"),
                FieldReference("direct_ref_move_mask_u16"),
            ),
        )
        if instruction.constraints != (expected_constraint,):
            raise ValueError(
                f"{instruction.mnemonic}: indirect-call constraint does not "
                "match its runtime verification form"
            )
    elif verification_form == "CONTROL_ASSERT":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: assertion record is not 4 bytes")
        require_value(1)
        require_zero(2, 1, array_length=2)
    elif verification_form == "CONTROL_FAIL":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: failure record is not 4 bytes")
        require_field(
            1,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.control.status"),),
        )
        require_ref(2)
        require_zero(3, 1)
    elif verification_form in (
        "VALUE_ABI_ARGUMENT_LOAD",
        "VALUE_ABI_RESULT_STORE",
        "REF_ABI_ARGUMENT_LOAD",
        "REF_ABI_RESULT_STORE",
        "FUNC_ABI_ARGUMENT_LOAD",
        "FUNC_ABI_RESULT_STORE",
    ):
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: ABI record is not 4 bytes")
        is_argument = verification_form in (
            "VALUE_ABI_ARGUMENT_LOAD",
            "REF_ABI_ARGUMENT_LOAD",
            "FUNC_ABI_ARGUMENT_LOAD",
        )
        is_ref = verification_form in (
            "REF_ABI_ARGUMENT_LOAD",
            "REF_ABI_RESULT_STORE",
        )
        is_function = verification_form in (
            "FUNC_ABI_ARGUMENT_LOAD",
            "FUNC_ABI_RESULT_STORE",
        )
        if is_function:
            require_function(1)
        elif is_ref:
            require_ref(1)
        else:
            require_value(1)
        packet_contract = (
            f"{'argument' if is_argument else 'result'}."
            f"{'function' if is_function else 'ref' if is_ref else 'value'}"
        )
        require_field(
            2,
            2,
            ABI_SLOT.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(packet_contract,),
        )
    elif verification_form in ("CONSTANT_ZERO", "CONSTANT_I32", "CONSTANT_I64"):
        require_value(1)
        require_zero(2, 2)
    elif verification_form == "CONSTANT_S16":
        require_value(1)
    elif verification_form in ("CONSTANT_POOL_LOAD_I32", "CONSTANT_POOL_LOAD_I64"):
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: constant-pool load is not 4 bytes"
            )
        require_value(1)
        require_field(
            2,
            2,
            CONSTANT_POOL_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "VALUE_UNARY_4":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: unary record is not 4 bytes")
        require_value(1)
        require_value(2)
        require_zero(3, 1)
    elif verification_form == "VALUE_SELECT":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: select record is not 8 bytes")
        for offset in range(1, 5):
            require_value(offset)
        require_zero(5, 1, array_length=3)
    elif verification_form == "FUNC_NULL":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: function null is not 4 bytes")
        require_function(1)
        require_zero(2, 2)
    elif verification_form == "FUNC_COMPARE_NULL":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: function null comparison is not 4 bytes"
            )
        require_value(1)
        require_function(2)
        require_zero(3, 1)
    elif verification_form == "FUNC_COPY":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: function copy is not 4 bytes")
        require_function(1)
        require_function(2)
        require_zero(3, 1)
    elif verification_form == "FUNC_ADDRESS":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: function address is not 8 bytes")
        require_function(1)
        require_field(
            2,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.control.call.target"),),
        )
        require_zero(3, 1)
        for offset in (4, 6):
            require_field(
                offset,
                2,
                CONSTRAINT_MEMBER.entity_id,
                (InstructionFieldRole.CONSTRAINT_MEMBER,),
                rule_arguments=("func.address",),
            )
        expected_constraint = RuleUse(
            FUNCTION_ADDRESS.entity_id,
            (
                FieldReference("target_kind_u8"),
                FieldReference("target_ordinal_u16"),
                FieldReference("callable_type_ordinal_u16"),
            ),
        )
        if instruction.constraints != (expected_constraint,):
            raise ValueError(
                f"{instruction.mnemonic}: address constraint does not match its "
                "runtime verification form"
            )
    elif verification_form == "FUNC_IMPORT_RESOLVED":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: import resolution test is not 4 bytes"
            )
        require_value(1)
        require_field(
            2,
            2,
            IMPORT_ORDINAL_OPTIONAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "FUNC_STACK_TRANSFER":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: function stack transfer is not 4 bytes"
            )
        require_function(1)
        require_field(
            2,
            2,
            FUNCTION_LOCAL_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form in (
        "GLOBAL_VALUE_IMMUTABLE_LOAD",
        "GLOBAL_VALUE_IMMUTABLE_STORE",
        "GLOBAL_VALUE_MUTABLE_LOAD",
        "GLOBAL_VALUE_MUTABLE_STORE",
        "GLOBAL_REF_IMMUTABLE_LOAD",
        "GLOBAL_REF_IMMUTABLE_STORE",
        "GLOBAL_REF_MUTABLE_LOAD",
        "GLOBAL_REF_MUTABLE_STORE",
        "GLOBAL_FUNC_IMMUTABLE_LOAD",
        "GLOBAL_FUNC_IMMUTABLE_STORE",
        "GLOBAL_FUNC_MUTABLE_LOAD",
        "GLOBAL_FUNC_MUTABLE_STORE",
    ):
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: global record is not 4 bytes")
        if verification_form.startswith("GLOBAL_FUNC_"):
            require_function(1)
        elif verification_form.startswith("GLOBAL_REF_"):
            require_ref(1)
        else:
            require_value(1)
        require_field(
            2,
            2,
            GLOBAL_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "VALUE_BINARY_4":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: binary record is not 4 bytes")
        for offset in range(1, 4):
            require_value(offset)
    elif verification_form == "INTEGER_COMPARE":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: compare record is not 8 bytes")
        for offset in range(1, 4):
            require_value(offset)
        require_field(
            4,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.integer.compare"),),
        )
        require_zero(5, 1, array_length=3)
    elif verification_form in ("FLOAT_MINMAX", "FLOAT_COMPARE"):
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: floating selector record is not 8 bytes"
            )
        for offset in range(1, 4):
            require_value(offset)
        selector_table_id = {
            "FLOAT_MINMAX": "core.selector.float.minmax",
            "FLOAT_COMPARE": "core.selector.float.compare",
        }[verification_form]
        require_field(
            4,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference(selector_table_id),),
        )
        require_zero(5, 1, array_length=3)
    elif verification_form == "FLOAT_CLASSIFY":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: float classify record is not 4 bytes"
            )
        require_value(1)
        require_value(2)
        require_field(
            3,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.float.classify"),),
        )
    elif verification_form == "FLOAT_CLAMP":
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: float clamp record is not 8 bytes"
            )
        for offset in range(1, 5):
            require_value(offset)
        require_field(
            5,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.float.clamp"),),
        )
        require_zero(6, 2)
    elif verification_form == "FLOAT_MATH_UNARY":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: float math unary record is not 4 bytes"
            )
        require_value(1)
        require_value(2)
        require_field(
            3,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.float.math.unary"),),
        )
    elif verification_form == "FLOAT_MATH_BINARY":
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: float math binary record is not 8 bytes"
            )
        for offset in range(1, 4):
            require_value(offset)
        require_field(
            4,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.float.math.binary"),),
        )
        require_zero(5, 1, array_length=3)
    elif verification_form == "FLOAT_MATH_TERNARY":
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: float math ternary record is not 8 bytes"
            )
        for offset in range(1, 5):
            require_value(offset)
        require_field(
            5,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.float.math.ternary"),),
        )
        require_zero(6, 2)
    elif verification_form == "INTEGER_LEA":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: LEA record is not 8 bytes")
        for offset in range(1, 4):
            require_value(offset)
        require_zero(5, 1)
    elif verification_form in (
        "INTEGER_CEILDIV_POW2_U32",
        "INTEGER_CEILDIV_POW2_U64",
    ):
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: power-of-two ceildiv record is not 4 bytes"
            )
        require_value(1)
        require_value(2)
        maximum_log2 = 31 if verification_form.endswith("U32") else 63
        require_field(
            3,
            1,
            ALLOWED_RANGE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(0, maximum_log2),
        )
    elif verification_form in (
        "INTEGER_BITSTREAM_PACK",
        "INTEGER_BITSTREAM_UNPACK",
    ):
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: integer bitstream record is not 8 bytes"
            )
        require_value(1)
        require_value(2)
        require_field(
            3,
            1,
            ALLOWED_RANGE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(1, 64),
        )
        require_field(
            4,
            1,
            ALLOWED_RANGE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(1, 255),
        )
        require_field(
            5,
            1,
            ALLOWED_RANGE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(1, 255),
        )
        carrier_values = (8, 16, 32, 64)
        require_field(
            6,
            1,
            ALLOWED_VALUES.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(carrier_values,),
        )
        require_field(
            7,
            1,
            ALLOWED_VALUES.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(carrier_values,),
        )
    elif verification_form == "REF_CLEAR":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: ref clear record is not 4 bytes")
        require_ref(1)
        require_zero(2, 2)
    elif verification_form == "REF_COMPARE_NULL":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: ref null comparison is not 4 bytes"
            )
        require_value(1)
        require_ref(2)
        require_zero(3, 1)
    elif verification_form == "REF_COMPARE_EQ":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: ref equality comparison is not 4 bytes"
            )
        require_value(1)
        require_ref(2)
        require_ref(3)
    elif verification_form in ("REF_RETAIN", "REF_MOVE"):
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: ref transfer record is not 4 bytes"
            )
        require_ref(1)
        require_ref(2)
        require_zero(3, 1)
        if verification_form == "REF_MOVE":
            expected_constraint = RuleUse(
                FIELDS_DISTINCT.entity_id,
                (FieldReference("dst_r8"), FieldReference("src_r8")),
            )
            if instruction.constraints != (expected_constraint,):
                raise ValueError(
                    f"{instruction.mnemonic}: move constraint does not match its "
                    "runtime verification form"
                )
    elif verification_form == "REF_STACK_TRANSFER":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: ref stack transfer is not 4 bytes"
            )
        require_ref(1)
        require_ref_slot(2)
    elif verification_form == "REF_STACK_DISCARD":
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: ref stack discard is not 4 bytes"
            )
        require_zero(1, 1)
        require_ref_slot(2)
    elif verification_form == "BUFFER_ALLOCATE":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: buffer allocate is not 4 bytes")
        require_ref(1)
        require_value(2)
        require_zero(3, 1)
    elif verification_form == "BUFFER_LENGTH":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: buffer length is not 4 bytes")
        require_value(1)
        require_ref(2)
        require_zero(3, 1)
    elif verification_form == "BUFFER_SUBSPAN":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: buffer subspan is not 8 bytes")
        require_ref(1)
        require_ref(2)
        require_value(3)
        require_value(4)
        require_zero(5, 1, array_length=3)
    elif verification_form in ("BUFFER_LOAD", "BUFFER_STORE"):
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: buffer lane access is not 8 bytes"
            )
        is_load = verification_form == "BUFFER_LOAD"
        if is_load:
            require_value(1)
            require_ref(2)
            require_value(3)
            require_value(4)
            scale_offset = 5
            register_field = "dst_v8"
        else:
            require_ref(1)
            require_value(2)
            require_value(3)
            scale_offset = 4
            require_value(5)
            register_field = "src_v8"
        require_field(
            scale_offset,
            1,
            ANY_BITS.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
        require_field(
            6,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.memory.format"),),
        )
        require_zero(7, 1)
        require_lane_range(register_field, "format_u8")
    elif verification_form in ("BUFFER_ATOMIC_REDUCE", "BUFFER_ATOMIC_RMW"):
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: buffer atomic apply is not 8 bytes"
            )
        is_reduce = verification_form == "BUFFER_ATOMIC_REDUCE"
        register_offset = 1
        if not is_reduce:
            require_value(register_offset)
            register_offset += 1
        require_ref(register_offset)
        require_value(register_offset + 1)
        require_value(register_offset + 2)
        require_packed_selector(
            register_offset + 3,
            0x70,
            (
                (
                    "kind",
                    0,
                    4,
                    "core.selector.buffer.atomic.kind",
                    tuple(range(2, 16)) if is_reduce else (),
                ),
                (
                    "carrier",
                    7,
                    1,
                    "core.selector.buffer.atomic.carrier",
                    (),
                ),
            ),
        )
        require_packed_selector(
            register_offset + 4,
            0xC0,
            (
                (
                    "ordering",
                    0,
                    3,
                    "core.selector.buffer.atomic.ordering",
                    (),
                ),
                ("scope", 3, 3, "core.selector.buffer.atomic.scope", ()),
            ),
        )
        require_zero(register_offset + 5, 2 if is_reduce else 1)
        expected_constraint = RuleUse(
            PACKED_SELECTOR_TARGET_SUPPORTED.entity_id,
            (FieldReference("selector0_u8"), "carrier"),
        )
        if instruction.constraints != (expected_constraint,):
            raise ValueError(
                f"{instruction.mnemonic}: atomic carrier constraint does not "
                "match its runtime verification form"
            )
    elif verification_form == "BUFFER_ATOMIC_CMPXCHG":
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: buffer atomic cmpxchg is not 8 bytes"
            )
        require_value(1)
        require_ref(2)
        require_value(3)
        require_value(4)
        require_value(5)
        require_packed_selector(
            6,
            0x40,
            (
                (
                    "success_ordering",
                    0,
                    3,
                    "core.selector.buffer.atomic.ordering",
                    (),
                ),
                (
                    "failure_ordering",
                    3,
                    3,
                    "core.selector.buffer.atomic.ordering",
                    (),
                ),
                (
                    "carrier",
                    7,
                    1,
                    "core.selector.buffer.atomic.carrier",
                    (),
                ),
            ),
        )
        require_packed_selector(
            7,
            0xF8,
            (("scope", 0, 3, "core.selector.buffer.atomic.scope", ()),),
        )
        expected_constraints = (
            RuleUse(
                PACKED_SELECTOR_TARGET_SUPPORTED.entity_id,
                (FieldReference("selector0_u8"), "carrier"),
            ),
            RuleUse(
                PACKED_SELECTOR_ALLOWED_PAIRS.entity_id,
                (
                    FieldReference("selector0_u8"),
                    "success_ordering",
                    "failure_ordering",
                    (
                        (0, 0),
                        (1, 0),
                        (1, 1),
                        (2, 0),
                        (3, 0),
                        (3, 1),
                        (4, 0),
                        (4, 1),
                        (4, 4),
                    ),
                ),
            ),
        )
        if instruction.constraints != expected_constraints:
            raise ValueError(
                f"{instruction.mnemonic}: atomic constraints do not match "
                "its runtime verification form"
            )
    elif verification_form == "BUFFER_FILL":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: buffer fill is not 8 bytes")
        require_ref(1)
        require_value(2)
        require_value(3)
        require_value(4)
        require_field(
            5,
            1,
            ALLOWED_VALUES.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=((1, 2, 4, 8),),
        )
        require_zero(6, 2)
    elif verification_form == "BUFFER_COPY":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: buffer copy is not 8 bytes")
        require_ref(1)
        require_value(2)
        require_ref(3)
        require_value(4)
        require_value(5)
        require_zero(6, 2)
    elif verification_form == "BUFFER_COMPARE":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: buffer compare is not 8 bytes")
        require_value(1)
        require_ref(2)
        require_value(3)
        require_ref(4)
        require_value(5)
        require_value(6)
        require_zero(7, 1)
    elif verification_form == "BUFFER_COPY_RODATA":
        if instruction.byte_length != 12:
            raise ValueError(
                f"{instruction.mnemonic}: buffer rodata copy is not 12 bytes"
            )
        require_ref(1)
        require_value(2)
        require_zero(3, 1)
        require_field(
            4,
            2,
            RODATA_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
        require_value(6)
        require_zero(7, 1)
        require_field(
            8,
            4,
            RODATA_OFFSET.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(FieldReference("rodata_u16"),),
        )
    elif verification_form == "BUFFER_RODATA_LOAD":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: rodata load is not 4 bytes")
        require_field(
            1,
            1,
            REGISTER_REF.entity_id,
            (InstructionFieldRole.RESULT,),
        )
        require_field(
            2,
            2,
            RODATA_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form in (
        "CONVERSION_INTEGER",
        "CONVERSION_FLOAT_EXTEND",
        "CONVERSION_FLOAT_TRUNCATE",
        "CONVERSION_FLOAT_WIDTH",
        "CONVERSION_INTEGER_TO_FLOAT",
        "CONVERSION_FLOAT_TO_INTEGER",
    ):
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: conversion record is not 4 bytes"
            )
        require_value(1)
        require_value(2)
        selector_table_id = {
            "CONVERSION_INTEGER": "core.selector.integer.convert",
            "CONVERSION_FLOAT_EXTEND": "core.selector.float.extend",
            "CONVERSION_FLOAT_TRUNCATE": "core.selector.float.truncate",
            "CONVERSION_FLOAT_WIDTH": "core.selector.float.width",
            "CONVERSION_INTEGER_TO_FLOAT": "core.selector.integer.to.float",
            "CONVERSION_FLOAT_TO_INTEGER": "core.selector.float.to.integer",
        }[verification_form]
        require_field(
            3,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference(selector_table_id),),
        )
    elif verification_form == "STACK_LOAD":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: stack load is not 8 bytes")
        require_value(1)
        require_field(
            2,
            2,
            LOCAL_BYTES_RANGE_MEMORY_FORMAT.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(FieldReference("format_u8"),),
        )
        require_field(
            4,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.memory.format"),),
        )
        require_zero(5, 1, array_length=3)
        require_lane_range("dst_v8", "format_u8")
    elif verification_form == "STACK_STORE":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: stack store is not 8 bytes")
        require_zero(1, 1)
        require_field(
            2,
            2,
            LOCAL_BYTES_RANGE_MEMORY_FORMAT.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(FieldReference("format_u8"),),
        )
        require_value(4)
        require_field(
            5,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.memory.format"),),
        )
        require_zero(6, 2)
        require_lane_range("src_v8", "format_u8")
    elif verification_form in ("STACK_LOAD_INDEXED", "STACK_STORE_INDEXED"):
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: indexed stack access is not 8 bytes"
            )
        is_load = verification_form == "STACK_LOAD_INDEXED"
        if is_load:
            require_value(1)
        else:
            require_zero(1, 1)
        require_field(
            2,
            2,
            LOCAL_BYTES_RANGE_MEMORY_FORMAT.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(FieldReference("format_u8"),),
        )
        require_value(4)
        require_field(
            5,
            1,
            ALLOWED_RANGE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(1, 255),
        )
        register_offset = 1 if is_load else 6
        format_offset = 6 if is_load else 7
        if not is_load:
            require_value(register_offset)
        require_field(
            format_offset,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.memory.format"),),
        )
        if is_load:
            require_zero(7, 1)
        require_lane_range("dst_v8" if is_load else "src_v8", "format_u8")
    elif verification_form == "STACK_FILL":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: stack fill is not 8 bytes")
        require_zero(1, 1)
        require_field(
            2,
            2,
            LOCAL_BYTES_RANGE_BASE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(FieldReference("length_u16"),),
        )
        require_field(
            4,
            2,
            LOCAL_BYTES_RANGE_LENGTH.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
        require_value(6)
        require_field(
            7,
            1,
            ALLOWED_VALUES.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=((1, 2, 4, 8),),
        )
    elif verification_form in ("STACK_COPY", "STACK_COMPARE"):
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: stack range operation is not 8 bytes"
            )
        if verification_form == "STACK_COPY":
            require_zero(1, 1)
        else:
            require_value(1)
        for offset in (2, 4):
            require_field(
                offset,
                2,
                LOCAL_BYTES_RANGE_BASE.entity_id,
                (InstructionFieldRole.IMMEDIATE,),
                rule_arguments=(FieldReference("length_u16"),),
            )
        require_field(
            6,
            2,
            LOCAL_BYTES_RANGE_LENGTH.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "STACK_COPY_RODATA":
        if instruction.byte_length != 12:
            raise ValueError(
                f"{instruction.mnemonic}: stack rodata copy is not 12 bytes"
            )
        require_zero(1, 1)
        require_field(
            2,
            2,
            LOCAL_BYTES_RANGE_BASE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(FieldReference("length_u16"),),
        )
        require_field(
            4,
            2,
            RODATA_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
        require_field(
            6,
            2,
            LOCAL_BYTES_RANGE_LENGTH.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
        require_field(
            8,
            4,
            RODATA_STATIC_OFFSET.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(
                FieldReference("rodata_u16"),
                FieldReference("length_u16"),
            ),
        )
    elif verification_form == "STACK_COPY_FROM_BUFFER":
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: stack-buffer copy is not 8 bytes"
            )
        require_zero(1, 1)
        require_field(
            2,
            2,
            LOCAL_BYTES_RANGE_BASE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(FieldReference("length_u16"),),
        )
        require_ref(4)
        require_value(5)
        require_field(
            6,
            2,
            LOCAL_BYTES_RANGE_LENGTH.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "STACK_COPY_TO_BUFFER":
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: stack-buffer copy is not 8 bytes"
            )
        require_ref(1)
        require_value(2)
        require_zero(3, 1)
        require_field(
            4,
            2,
            LOCAL_BYTES_RANGE_BASE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(FieldReference("length_u16"),),
        )
        require_field(
            6,
            2,
            LOCAL_BYTES_RANGE_LENGTH.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form in ("STACK_CONST_S16_I32", "STACK_CONST_S16_I64"):
        if instruction.byte_length != 8:
            raise ValueError(
                f"{instruction.mnemonic}: repeated stack constant is not 8 bytes"
            )
        element_byte_length = 4 if verification_form.endswith("I32") else 8
        require_zero(1, 1)
        require_field(
            2,
            2,
            LOCAL_BYTES_REPEATED_BASE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(
                FieldReference("count_u16"),
                element_byte_length,
                element_byte_length,
            ),
        )
        require_field(
            4,
            2,
            LOCAL_BYTES_REPEATED_COUNT.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
        require_field(
            6,
            2,
            ANY_BITS.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form in ("STACK_PACK_I32", "STACK_PACK_I64"):
        immediate_byte_length = 2 if verification_form.endswith("I32") else 4
        destination_byte_length = 4 if verification_form.endswith("I32") else 8
        payload_byte_length = instruction.byte_length - 4
        if payload_byte_length <= 0 or payload_byte_length % immediate_byte_length:
            raise ValueError(f"{instruction.mnemonic}: stack pack payload is malformed")
        lane_count = payload_byte_length // immediate_byte_length
        if lane_count not in (2, 4, 8):
            raise ValueError(
                f"{instruction.mnemonic}: stack pack lane count is unsupported"
            )
        require_zero(1, 1)
        require_field(
            2,
            2,
            LOCAL_BYTES_FIXED_BASE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(
                destination_byte_length * lane_count,
                destination_byte_length,
            ),
        )
        require_field(
            4,
            immediate_byte_length,
            ANY_BITS.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            array_length=lane_count,
        )
    else:
        raise ValueError(
            f"{instruction.mnemonic}: unknown runtime verification form "
            f"{verification_form!r}"
        )
    unchecked_validated_fields = tuple(
        field.name
        for field in instruction.fields
        if any(rule.rule_id != ANY_BITS.entity_id for rule in field.validation)
        and field.offset not in verified_field_offsets
    )
    if unchecked_validated_fields:
        raise ValueError(
            f"{instruction.mnemonic}: runtime verification form leaves "
            f"validated fields unchecked: {unchecked_validated_fields!r}"
        )


def render_execution_tables(
    isa_projection: Projection,
    executable_instructions: Sequence[ExecutableInstruction],
) -> str:
    """Renders one multi-include table for verifier and interpreter use."""

    entities_by_id = isa_projection.entity_map()
    instructions_by_mnemonic = {
        entity.mnemonic: entity
        for entity in isa_projection.entities
        if isinstance(entity, Instruction)
    }
    resolved: list[tuple[Instruction, ExecutableInstruction]] = []
    seen_mnemonics: set[str] = set()
    seen_opcodes: set[int] = set()
    for executable in executable_instructions:
        if executable.mnemonic in seen_mnemonics:
            raise ValueError(
                f"duplicate executable instruction {executable.mnemonic!r}"
            )
        instruction = instructions_by_mnemonic.get(executable.mnemonic)
        if instruction is None:
            raise ValueError(f"unknown executable instruction {executable.mnemonic!r}")
        if instruction.since.domain != "core":
            raise ValueError(
                f"executable instruction {executable.mnemonic!r} is not Core"
            )
        if instruction.opcode in seen_opcodes:
            raise ValueError(f"duplicate executable opcode 0x{instruction.opcode:02X}")
        if instruction.byte_length > 0xFF:
            raise ValueError(f"{executable.mnemonic}: record length does not fit in u8")
        _validate_verification_form(
            instruction, executable.verification_form, entities_by_id
        )
        seen_mnemonics.add(executable.mnemonic)
        seen_opcodes.add(instruction.opcode)
        resolved.append((instruction, executable))
    resolved.sort(key=lambda pair: pair[0].opcode)

    by_opcode = {
        instruction.opcode: (instruction, executable)
        for instruction, executable in resolved
    }
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED FILE: DO NOT EDIT.",
        "// Private build-tree projection: executable Core instruction rows.",
        "// clang-format off",
        "",
        "#if defined(IREE_VM_BYTECODE_DEFINE_EXECUTION_INFO_ROWS)",
    ]
    for opcode in range(256):
        pair = by_opcode.get(opcode)
        if pair is None:
            lines.append("IREE_VM_BYTECODE_EXECUTION_INFO_ROW(0, NONE)")
        else:
            instruction, executable = pair
            lines.append(
                "IREE_VM_BYTECODE_EXECUTION_INFO_ROW("
                f"{instruction.byte_length}, {executable.verification_form})"
            )
    lines.extend(
        (
            "#elif defined(IREE_VM_BYTECODE_DEFINE_EXECUTABLE_OPCODE_LIST)",
            "#define IREE_VM_BYTECODE_EXECUTABLE_OPCODE_LIST(OP) \\",
        )
    )
    for index, (instruction, _) in enumerate(resolved):
        continuation = " \\" if index + 1 != len(resolved) else ""
        lines.append(
            f"  OP({_opcode_token(instruction.mnemonic)}, "
            f"{_handler_label(instruction.mnemonic)}){continuation}"
        )
    lines.extend(
        (
            "#else",
            '#error "define one VM bytecode execution-table projection"',
            "#endif",
            "",
            "// clang-format on",
            "",
        )
    )
    return "\n".join(lines)
