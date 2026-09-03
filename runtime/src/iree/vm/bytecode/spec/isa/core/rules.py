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
    ANY_BITS = RuleKind("any_bits")
    ZERO = RuleKind("zero")
    ALLOWED_RANGE = RuleKind("allowed_range", value_count=2)
    ALLOWED_VALUES = RuleKind("allowed_values", value_count=-1)
    REGISTER_VALUE = RuleKind("register_value", U8)
    REGISTER_REF = RuleKind("register_ref", U8)
    SELECTOR = RuleKind("selector", data_type=NumericTable)
    REGISTER_FUNCTION = RuleKind("register_function", U8)
    GLOBAL_ORDINAL = RuleKind("global_ordinal", U16, value_count=1)
    CONSTRAINT_MEMBER = RuleKind("constraint_member")
    LOCAL_BYTES_RANGE_BASE = RuleKind("local_bytes_range_base", U16, field_count=1)
    ABI_SLOT = RuleKind("abi_slot", U16, value_count=1)
    LOCAL_BYTES_FIXED_BASE = RuleKind("local_bytes_fixed_base", U16, value_count=2)
    LOCAL_BYTES_RANGE_LENGTH = RuleKind("local_bytes_range_length", U16)
    PACKED_SELECTORS = RuleKind("packed_selectors", value_count=1, data_type=tuple)
    REF_SLOT = RuleKind("ref_slot", U16)
    CONTROL_TARGET_S16 = RuleKind("control_target_s16", I16)
    CONTROL_TARGET_S32 = RuleKind("control_target_s32", I32)
    LOCAL_BYTES_RANGE_MEMORY_FORMAT = RuleKind(
        "local_bytes_range_memory_format", U16, field_count=1
    )
    RODATA_ORDINAL = RuleKind("rodata_ordinal", U16)
    CONSTANT_POOL_ORDINAL = RuleKind("constant_pool_ordinal", U16)
    FUNCTION_LOCAL_ORDINAL = RuleKind("function_local_ordinal", U16)
    LOCAL_BYTES_REPEATED_BASE = RuleKind(
        "local_bytes_repeated_base", U16, field_count=1, value_count=2
    )
    LOCAL_BYTES_REPEATED_COUNT = RuleKind("local_bytes_repeated_count", U16)
    IMPORT_ORDINAL_OPTIONAL = RuleKind("import_ordinal_optional", U16)
    RODATA_OFFSET = RuleKind("rodata_offset", U32, field_count=1)
    RODATA_STATIC_OFFSET = RuleKind("rodata_static_offset", U32, field_count=2)


class RecordRuleKind:
    CALL = RuleKind("call", field_count=3)
    CALL_INDIRECT = RuleKind("call_indirect", field_count=3)
    RETURN_SIGNATURE = RuleKind("return_signature")
    SWITCH_TARGETS = RuleKind("switch_targets", field_count=2)
    FIELDS_DISTINCT = RuleKind("fields_distinct", field_count=2)
    FUNCTION_ADDRESS = RuleKind("function_address", field_count=3)
    INTEGER_BITSTREAM = RuleKind(
        "integer_bitstream", field_count=5, value_count=1, name_count=1
    )
    PACKED_SELECTOR_PAIRS = RuleKind(
        "packed_selector_pairs", field_count=1, value_count=-1, name_count=2
    )
    PACKED_SELECTOR_TARGET = RuleKind(
        "packed_selector_target", field_count=1, name_count=1
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
    ControlFlow.RETURN: (0, Suspension.NEVER, (RecordRuleKind.RETURN_SIGNATURE,)),
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
