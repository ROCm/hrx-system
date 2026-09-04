# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core instruction verification rule declarations."""

import enum

from iree.vm.bytecode.spec.isa import ControlFlow, Suspension
from iree.vm.bytecode.spec.schema import (
    I16,
    I32,
    U8,
    U16,
    U32,
    NumericTable,
    RuleKind,
)


class StateAccess(enum.Enum):
    UNKNOWN = "unknown"
    READ = "read"
    WRITE = "write"
    ALLOCATE = "allocate"


class StateResource(enum.Enum):
    ANY = "any"
    INVOCATION_ARGUMENTS = "invocation.arguments"
    INVOCATION_RESULTS = "invocation.results"
    FRAME_LOCALS = "frame.locals"
    PROCESS_GLOBALS = "process.globals"
    BUFFER = "buffer"


class RefNullPolicy(enum.Enum):
    DESCRIPTOR = "descriptor"
    NULLABLE = "nullable"
    REQUIRED = "required"
    RESULT_NONNULL = "result_nonnull"


class RefOwnership(enum.Enum):
    BORROW = "borrow"
    CLEAR = "clear"
    DIAGNOSTIC_BORROW = "diagnostic_borrow"
    INSPECT = "inspect"
    MOVE = "move"
    PUBLISH_MOVE = "publish_move"
    REPLACE_BORROW = "replace_borrow"
    REPLACE_MOVE = "replace_move"
    REPLACE_OWNER = "replace_owner"
    REPLACE_RETAIN = "replace_retain"
    RETAIN = "retain"


class FieldRule:
    ANY_BITS = RuleKind("any_bits", summary="Any bit pattern.")
    ZERO = RuleKind("zero", summary="Must be zero.")
    ALLOWED_RANGE = RuleKind(
        "allowed_range",
        value_count=2,
        summary="Must be in the inclusive range [{0}, {1}].",
    )
    ALLOWED_VALUES = RuleKind(
        "allowed_values",
        value_count=-1,
        summary="Must equal one of {values}.",
    )
    REGISTER_VALUE = RuleKind(
        "register_value", U8, summary="Must name an in-range value register."
    )
    REGISTER_REF = RuleKind(
        "register_ref", U8, summary="Must name an in-range ref register."
    )
    SELECTOR = RuleKind("selector", data_type=NumericTable)
    REGISTER_FUNCTION = RuleKind(
        "register_function", U8, summary="Must name an in-range function register."
    )
    GLOBAL_ORDINAL = RuleKind(
        "global_ordinal",
        U16,
        value_count=1,
        summary="Must name an in-range global in the encoded storage partition.",
    )
    CONSTRAINT_MEMBER = RuleKind(
        "constraint_member",
        summary="Validated by the instruction's cross-field rule.",
    )
    LOCAL_BYTES_RANGE_BASE = RuleKind(
        "local_bytes_range_base",
        U16,
        field_count=1,
        summary="Must form an in-bounds local-byte range with the related u16 length.",
    )
    ABI_SLOT = RuleKind(
        "abi_slot",
        U16,
        value_count=1,
        summary="Must name an in-range overflow slot in the encoded packet region.",
    )
    LOCAL_BYTES_FIXED_BASE = RuleKind(
        "local_bytes_fixed_base",
        U16,
        value_count=2,
        summary="Must begin an in-bounds {0}-byte local range aligned to {1} bytes.",
    )
    LOCAL_BYTES_RANGE_LENGTH = RuleKind(
        "local_bytes_range_length",
        U16,
        summary="Validated with each local-byte range base that names this length.",
    )
    PACKED_SELECTORS = RuleKind("packed_selectors", value_count=1, data_type=tuple)
    REF_SLOT = RuleKind(
        "ref_slot", U16, summary="Must name an in-range function-local ref slot."
    )
    CONTROL_TARGET_S16 = RuleKind(
        "control_target_s16",
        I16,
        summary="Must resolve to an in-function control.block using widened signed arithmetic.",
    )
    CONTROL_TARGET_S32 = RuleKind(
        "control_target_s32",
        I32,
        summary="Must resolve to an in-function control.block using widened signed arithmetic.",
    )
    LOCAL_BYTES_RANGE_MEMORY_FORMAT = RuleKind(
        "local_bytes_range_memory_format",
        U16,
        field_count=1,
        summary="Must contain the complete lane group selected by the related memory format.",
    )
    RODATA_ORDINAL = RuleKind(
        "rodata_ordinal",
        U16,
        summary="Must name an in-range module rodata block.",
    )
    CONSTANT_POOL_ORDINAL = RuleKind(
        "constant_pool_ordinal",
        U16,
        summary="Must name an in-range module constant-pool cell.",
    )
    FUNCTION_LOCAL_ORDINAL = RuleKind(
        "function_local_ordinal",
        U16,
        summary="Must name an in-range function-local cell.",
    )
    LOCAL_BYTES_REPEATED_BASE = RuleKind(
        "local_bytes_repeated_base",
        U16,
        field_count=1,
        value_count=2,
        summary="Must begin an aligned local range of related count * {0} bytes with {1}-byte alignment.",
    )
    LOCAL_BYTES_REPEATED_COUNT = RuleKind(
        "local_bytes_repeated_count",
        U16,
        summary="Validated with the repeated local-byte range base that names this count.",
    )
    IMPORT_ORDINAL_OPTIONAL = RuleKind(
        "import_ordinal_optional",
        U16,
        summary="Must name an optional import declaration.",
    )
    RODATA_OFFSET = RuleKind(
        "rodata_offset",
        U32,
        field_count=1,
        summary="Must not exceed the length of the rodata block named by the related field.",
    )
    RODATA_STATIC_OFFSET = RuleKind(
        "rodata_static_offset",
        U32,
        field_count=2,
        summary="Must form an in-bounds range in the related rodata block with the related length.",
    )


class RecordRuleKind:
    CALL = RuleKind("call", field_count=3)
    CALL_INDIRECT = RuleKind("call_indirect", field_count=3)
    SWITCH_TARGETS = RuleKind("switch_targets", field_count=2)
    FIELDS_DISTINCT = RuleKind("fields_distinct", field_count=2)
    FUNCTION_ADDRESS = RuleKind("function_address", field_count=3)
    INTEGER_BITSTREAM_SHAPE = RuleKind(
        "integer_bitstream_shape", field_count=5, value_count=2
    )
    PACKED_SELECTOR_PAIRS = RuleKind(
        "packed_selector_pairs",
        field_count=1,
        value_count=-1,
        data_count=2,
        data_type=tuple,
    )
    ATOMIC_CARRIER_REQUIREMENT = RuleKind(
        "atomic_carrier_requirement", field_count=1, data_count=1, data_type=tuple
    )
    VALUE_REGISTER_RANGE = RuleKind("value_register_range", field_count=2)
    VALUE_REGISTER_FORMAT_RANGE = RuleKind(
        "value_register_format_range", field_count=2, value_count=1
    )


FIELD_RULES = tuple(value for name, value in vars(FieldRule).items() if name.isupper())
RECORD_RULES = tuple(
    value for name, value in vars(RecordRuleKind).items() if name.isupper()
)
DIRECT_TARGET_RULES = (FieldRule.CONTROL_TARGET_S16, FieldRule.CONTROL_TARGET_S32)
CONTROL_CONTRACTS = {
    ControlFlow.BRANCH: (1, Suspension.NEVER, ()),
    ControlFlow.CONDITIONAL_BRANCH: (1, Suspension.NEVER, ()),
    ControlFlow.YIELD: (1, Suspension.ALWAYS, ()),
    ControlFlow.RETURN: (0, Suspension.NEVER, ()),
    ControlFlow.SWITCH: (0, Suspension.NEVER, (RecordRuleKind.SWITCH_TARGETS,)),
    ControlFlow.CALL: (
        0,
        Suspension.TARGET_DEPENDENT,
        (RecordRuleKind.CALL, RecordRuleKind.CALL_INDIRECT),
    ),
}
CONTROL_RULES = frozenset(
    kind for _, _, kinds in CONTROL_CONTRACTS.values() for kind in kinds
)
