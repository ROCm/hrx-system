# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Versioned instruction-field and instruction-record verification rules."""

from __future__ import annotations

from model.schema import (
    ALLOWED_BITS,
    ALLOWED_BITS_EXACTLY_ONE,
    ALLOWED_RANGE,
    ALLOWED_VALUES,
    ANY_BITS,
    BASIC_FIELD_RULES,
    ENTITY_ARGUMENT,
    FIELD_ARGUMENT,
    INTEGER_ARGUMENT,
    TEXT_ARGUMENT,
    ZERO,
    ArgumentShape,
    RuleParameter,
    ValidationRule,
    ValidationScope,
)
from model.specification import CORE_0

INTEGER_SEQUENCE = ArgumentShape.sequence(INTEGER_ARGUMENT)
NONEMPTY_INTEGER_SEQUENCE = ArgumentShape.sequence(
    INTEGER_ARGUMENT,
    minimum_count=1,
)
INTEGER_PAIR = ArgumentShape.tuple_of(INTEGER_ARGUMENT, INTEGER_ARGUMENT)
INTEGER_PAIR_SEQUENCE = ArgumentShape.sequence(INTEGER_PAIR, minimum_count=1)
PACKED_SELECTOR_COMPONENT = ArgumentShape.tuple_of(
    TEXT_ARGUMENT,
    INTEGER_ARGUMENT,
    INTEGER_ARGUMENT,
    ENTITY_ARGUMENT,
    INTEGER_SEQUENCE,
)
PACKED_SELECTOR_COMPONENT_SEQUENCE = ArgumentShape.sequence(
    PACKED_SELECTOR_COMPONENT,
    minimum_count=1,
)


def _field_rule(
    name: str,
    parameters: tuple[RuleParameter, ...],
    summary: str,
    normative_text: str,
) -> ValidationRule:
    return ValidationRule(
        entity_id=f"core.validation.instruction.field.{name}",
        since=CORE_0,
        summary=summary,
        scope=ValidationScope.FIELD,
        parameters=parameters,
        normative_text=normative_text,
    )


def _record_rule(
    name: str,
    parameters: tuple[RuleParameter, ...],
    summary: str,
    normative_text: str,
) -> ValidationRule:
    return ValidationRule(
        entity_id=f"core.validation.instruction.record.{name}",
        since=CORE_0,
        summary=summary,
        scope=ValidationScope.RECORD,
        parameters=parameters,
        normative_text=normative_text,
    )


ABI_SLOT = _field_rule(
    "abi_slot",
    (RuleParameter("packet_contract", TEXT_ARGUMENT),),
    "Requires a valid logical slot in one function ABI packet region.",
    "The u16 value must name an existing logical slot in the argument or "
    "result value, ref, or function region selected by packet_contract.",
)
CONSTANT_POOL_ORDINAL = _field_rule(
    "constant_pool_ordinal",
    (),
    "Requires a valid module constant-pool ordinal.",
    "The u16 value must be less than the module constant count.",
)
CONSTRAINT_MEMBER = _field_rule(
    "constraint_member",
    (RuleParameter("constraint_kind", TEXT_ARGUMENT),),
    "Delegates field validation to one owning record constraint.",
    "Exactly one record constraint of constraint_kind must reference this "
    "field and must validate its value in relation to the other named fields.",
)
CONTROL_TARGET_RELATIVE_S16 = _field_rule(
    "control_target_relative_s16",
    (),
    "Requires a valid signed-16 relative control target.",
    "The signed word displacement is relative to the end of this record and "
    "must resolve without overflow to a decoded control.block in the same "
    "function.",
)
CONTROL_TARGET_RELATIVE_S32 = _field_rule(
    "control_target_relative_s32",
    (),
    "Requires a valid signed-32 relative control target.",
    "The signed word displacement is relative to the end of this record and "
    "must resolve without overflow to a decoded control.block in the same "
    "function.",
)
FUNCTION_LOCAL_ORDINAL = _field_rule(
    "function_local_ordinal",
    (),
    "Requires a valid current-function local ordinal.",
    "The u16 value must be less than the current function's declared local "
    "extent for the resource addressed by the instruction.",
)
GLOBAL_ORDINAL = _field_rule(
    "global_ordinal",
    (RuleParameter("global_contract", TEXT_ARGUMENT),),
    "Requires a valid global ordinal in one storage partition.",
    "The u16 value must name an existing mutable or immutable value, ref, or "
    "function global selected by global_contract.",
)
IMPORT_ORDINAL_OPTIONAL = _field_rule(
    "import_ordinal_optional",
    (),
    "Requires a valid optional-import ordinal.",
    "The u16 value must name an import declaration carrying the OPTIONAL flag.",
)
LOCAL_BYTES_FIXED_BASE = _field_rule(
    "local_bytes_fixed_base",
    (
        RuleParameter("byte_length", INTEGER_ARGUMENT),
        RuleParameter("alignment", INTEGER_ARGUMENT),
    ),
    "Requires an aligned fixed-size local-byte range.",
    "The u16 base and declared byte_length must form an in-bounds range in "
    "the current frame's local bytes, and base must satisfy alignment.",
)
LOCAL_BYTES_RANGE_BASE = _field_rule(
    "local_bytes_range_base",
    (RuleParameter("length_field", FIELD_ARGUMENT),),
    "Requires the base of a variable local-byte range.",
    "The u16 base and the u16 byte length in length_field must form an "
    "in-bounds range in the current frame's local bytes.",
)
LOCAL_BYTES_RANGE_LENGTH = _field_rule(
    "local_bytes_range_length",
    (),
    "Identifies the u16 length of a variable local-byte range.",
    "The value is validated together with its owning local-byte base field.",
)
LOCAL_BYTES_RANGE_MEMORY_FORMAT = _field_rule(
    "local_bytes_range_memory_format",
    (RuleParameter("format_field", FIELD_ARGUMENT),),
    "Requires a local-byte range sized by a memory-format selector.",
    "The local range must contain exactly one value of the memory format "
    "selected by format_field. The range is byte-addressed and has no natural-"
    "alignment requirement.",
)
LOCAL_BYTES_REPEATED_BASE = _field_rule(
    "local_bytes_repeated_base",
    (
        RuleParameter("count_field", FIELD_ARGUMENT),
        RuleParameter("element_byte_length", INTEGER_ARGUMENT),
        RuleParameter("element_alignment", INTEGER_ARGUMENT),
    ),
    "Requires an aligned repeated local-byte range.",
    "The u16 base and count_field must describe count contiguous elements of "
    "element_byte_length bytes, remain in bounds, and satisfy "
    "element_alignment.",
)
LOCAL_BYTES_REPEATED_COUNT = _field_rule(
    "local_bytes_repeated_count",
    (),
    "Identifies the u16 count of a repeated local-byte range.",
    "The value is validated together with its owning repeated-range base field.",
)
PACKED_SELECTORS = _field_rule(
    "packed_selectors",
    (
        RuleParameter("zero_mask", INTEGER_ARGUMENT),
        RuleParameter(
            "components",
            PACKED_SELECTOR_COMPONENT_SEQUENCE,
        ),
    ),
    "Partitions a field into selector slices and required-zero bits.",
    "zero_mask and the nonoverlapping component bit ranges must classify "
    "every encoded bit exactly once. Zero-mask bits must be zero and each "
    "component must name an available value in its selector table and optional "
    "allowed-value subset.",
)
RANGE_BASE = _field_rule(
    "range_base",
    (RuleParameter("range_name", TEXT_ARGUMENT),),
    "Identifies a base field owned by one counted range group.",
    "The field must be the unique base member named by range_name; the range "
    "group performs its bounds, alignment, and element-policy validation.",
)
RANGE_COUNT = _field_rule(
    "range_count",
    (RuleParameter("range_name", TEXT_ARGUMENT),),
    "Identifies the count field owned by one range group.",
    "The field must be the unique count of range_name and is interpreted by "
    "that group's declared members.",
)
REF_SLOT = _field_rule(
    "ref_slot",
    (),
    "Requires a valid ref slot in the current instruction contract.",
    "The u16 value must name an available ref slot in the enclosing operation's "
    "declared local storage contract.",
)
REGISTER_FUNCTION = _field_rule(
    "register_function",
    (),
    "Requires a valid function-register index.",
    "The u8 value must be less than the current frame's function-register count.",
)
REGISTER_REF = _field_rule(
    "register_ref",
    (),
    "Requires a valid ref-register index.",
    "The u8 value must be less than the current frame's ref-register count; "
    "the instruction's separate runtime ref policy defines type, null, and "
    "ownership behavior.",
)
REGISTER_VALUE = _field_rule(
    "register_value",
    (),
    "Requires a valid value-register index.",
    "The u8 value must be less than the current frame's value-register count.",
)
RODATA_EXECUTABLE_NAME_TABLE = _field_rule(
    "rodata_executable_name_table",
    (),
    "Requires rodata containing a canonical executable-name table.",
    "The u16 value must name a rodata block whose complete bytes decode as the "
    "required executable-name table grammar.",
)
RODATA_OFFSET = _field_rule(
    "rodata_offset",
    (RuleParameter("rodata_field", FIELD_ARGUMENT),),
    "Requires an in-bounds offset into a selected rodata block.",
    "The u32 byte offset must not exceed the rodata block named by "
    "rodata_field; the operation contract determines whether the terminal "
    "offset is usable.",
)
RODATA_ORDINAL = _field_rule(
    "rodata_ordinal",
    (),
    "Requires a valid module rodata-block ordinal.",
    "The u16 value must be less than the module rodata block count.",
)
RODATA_STATIC_OFFSET = _field_rule(
    "rodata_static_offset",
    (
        RuleParameter("rodata_field", FIELD_ARGUMENT),
        RuleParameter("length_field", FIELD_ARGUMENT),
    ),
    "Requires a complete static range in one rodata block.",
    "The u32 byte offset and u16 byte length must form an in-bounds range in "
    "the rodata block named by rodata_field.",
)
SELECTOR = _field_rule(
    "selector",
    (RuleParameter("table", ENTITY_ARGUMENT),),
    "Requires one value from a closed selector table.",
    "The decoded unsigned field value must equal an available value declared "
    "by the referenced selector table.",
)
STRING_ORDINAL = _field_rule(
    "string_ordinal",
    (),
    "Requires a valid module string ordinal.",
    "The u16 value must be less than the module string count.",
)
STRING_ORDINAL_NONEMPTY = _field_rule(
    "string_ordinal_nonempty",
    (),
    "Requires a valid nonempty module string ordinal.",
    "The u16 value must name a module string whose byte length is nonzero.",
)

CONTROL_CALL = _record_rule(
    "control_call",
    (
        RuleParameter("target_kind_field", FIELD_ARGUMENT),
        RuleParameter("target_ordinal_field", FIELD_ARGUMENT),
        RuleParameter("direct_ref_move_mask_field", FIELD_ARGUMENT),
    ),
    "Validates a direct call target and move mask against its signature.",
    "The selected local function or import declaration must exist with "
    "matching import optionality. A local target uses its function signature; "
    "an import uses its callable-type signature. The caller's three direct "
    "register prefixes and canonical overflow packet must cover that signature, "
    "and direct_ref_move_mask must not name a direct ref argument outside it. "
    "A local call does not require or infer a callable-type declaration.",
)
CONTROL_CALL_INDIRECT = _record_rule(
    "control_call_indirect",
    (
        RuleParameter("target_field", FIELD_ARGUMENT),
        RuleParameter("callable_type_field", FIELD_ARGUMENT),
        RuleParameter("direct_ref_move_mask_field", FIELD_ARGUMENT),
    ),
    "Validates an indirect call's declared callable type and move mask.",
    "callable_type_field must name an available callable type, the call packet "
    "must match it, and direct_ref_move_mask must not name a direct ref argument "
    "outside that callable type. The dynamic target is checked at execution.",
)
CONTROL_RETURN_SIGNATURE = _record_rule(
    "control_return_signature",
    (),
    "Validates return storage against the enclosing function signature.",
    "The function's direct and overflow result regions must fit its declared "
    "frame storage and be addressable by control.return.",
)
CONTROL_SWITCH_TARGETS = _record_rule(
    "control_switch_targets",
    (
        RuleParameter("target_count_field", FIELD_ARGUMENT),
        RuleParameter("target_base_field", FIELD_ARGUMENT),
    ),
    "Validates a function-local switch-target table slice.",
    "target_base plus target_count must form an in-bounds slice of the owning "
    "function's switch-target entries; every entry must resolve to a decoded "
    "control.block in that function.",
)
COUNT_NONZERO_WHEN_SELECTOR = _record_rule(
    "count_nonzero_when_selector",
    (
        RuleParameter("selector_field", FIELD_ARGUMENT),
        RuleParameter("selector_values", NONEMPTY_INTEGER_SEQUENCE),
        RuleParameter("count_field", FIELD_ARGUMENT),
    ),
    "Requires a nonzero range count for selected operation modes.",
    "count_field must be nonzero whenever selector_field equals any member of "
    "selector_values.",
)
FIELDS_DISTINCT = _record_rule(
    "fields_distinct",
    (
        RuleParameter("first_field", FIELD_ARGUMENT),
        RuleParameter("second_field", FIELD_ARGUMENT),
    ),
    "Requires two encoded register fields to name different slots.",
    "The decoded values of first_field and second_field must differ.",
)
FUNCTION_ADDRESS = _record_rule(
    "function_address",
    (
        RuleParameter("target_kind_field", FIELD_ARGUMENT),
        RuleParameter("target_ordinal_field", FIELD_ARGUMENT),
        RuleParameter("callable_type_field", FIELD_ARGUMENT),
    ),
    "Validates a first-class function address declaration.",
    "The selected local or import target must exist and its resolved callable "
    "type must equal callable_type_field.",
)
INTEGER_BITSTREAM_SHAPE = _record_rule(
    "integer_bitstream_shape",
    (
        RuleParameter("mode", TEXT_ARGUMENT),
        RuleParameter("field_width_field", FIELD_ARGUMENT),
        RuleParameter("source_count_field", FIELD_ARGUMENT),
        RuleParameter("result_count_field", FIELD_ARGUMENT),
        RuleParameter("source_width_field", FIELD_ARGUMENT),
        RuleParameter("result_width_field", FIELD_ARGUMENT),
    ),
    "Validates exact source/result capacity for integer bitstream packing.",
    "mode must be pack or unpack. The declared carrier widths and counts must "
    "provide exactly the bit capacity required by field_width without reading "
    "or producing an incomplete logical element.",
)
PACKED_SELECTOR_ALLOWED_PAIRS = _record_rule(
    "packed_selector_allowed_pairs",
    (
        RuleParameter("selector_field", FIELD_ARGUMENT),
        RuleParameter("first_component", TEXT_ARGUMENT),
        RuleParameter("second_component", TEXT_ARGUMENT),
        RuleParameter("allowed_pairs", INTEGER_PAIR_SEQUENCE),
    ),
    "Restricts two packed-selector components to explicit value pairs.",
    "The pair decoded from first_component and second_component in "
    "selector_field must equal one member of allowed_pairs.",
)
PACKED_SELECTOR_INDEXED_ALLOWED_MASK = _record_rule(
    "packed_selector_indexed_allowed_mask",
    (
        RuleParameter("selector_field", FIELD_ARGUMENT),
        RuleParameter("index_component", TEXT_ARGUMENT),
        RuleParameter("value_component", TEXT_ARGUMENT),
        RuleParameter("allowed_masks", NONEMPTY_INTEGER_SEQUENCE),
    ),
    "Restricts one packed selector component by another component's value.",
    "The decoded index_component selects one mask in allowed_masks; the bit "
    "for decoded value_component must be set in that mask.",
)
PACKED_SELECTOR_TARGET_SUPPORTED = _record_rule(
    "packed_selector_target_supported",
    (
        RuleParameter("selector_field", FIELD_ARGUMENT),
        RuleParameter("component", TEXT_ARGUMENT),
    ),
    "Requires a packed-selector choice supported by the target contract.",
    "The decoded component of selector_field must be enabled by the selected "
    "target's declared capabilities.",
)
VALUE_REGISTER_RANGE = _record_rule(
    "value_register_range",
    (
        RuleParameter("base_field", FIELD_ARGUMENT),
        RuleParameter("count_field", FIELD_ARGUMENT),
    ),
    "Requires a contiguous nonempty value-register range.",
    "The u8 base and nonzero u8 count must describe an in-bounds contiguous "
    "range in the current frame's value registers.",
)
VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT = _record_rule(
    "value_register_range_from_memory_format",
    (
        RuleParameter("base_field", FIELD_ARGUMENT),
        RuleParameter("format_field", FIELD_ARGUMENT),
    ),
    "Requires the value-register range implied by a memory format.",
    "The u8 base must leave enough consecutive value registers for the lane "
    "count selected by format_field.",
)

INSTRUCTION_FIELD_RULES = (
    ABI_SLOT,
    CONSTANT_POOL_ORDINAL,
    CONSTRAINT_MEMBER,
    CONTROL_TARGET_RELATIVE_S16,
    CONTROL_TARGET_RELATIVE_S32,
    FUNCTION_LOCAL_ORDINAL,
    GLOBAL_ORDINAL,
    IMPORT_ORDINAL_OPTIONAL,
    LOCAL_BYTES_FIXED_BASE,
    LOCAL_BYTES_RANGE_BASE,
    LOCAL_BYTES_RANGE_LENGTH,
    LOCAL_BYTES_RANGE_MEMORY_FORMAT,
    LOCAL_BYTES_REPEATED_BASE,
    LOCAL_BYTES_REPEATED_COUNT,
    PACKED_SELECTORS,
    RANGE_BASE,
    RANGE_COUNT,
    REF_SLOT,
    REGISTER_FUNCTION,
    REGISTER_REF,
    REGISTER_VALUE,
    RODATA_EXECUTABLE_NAME_TABLE,
    RODATA_OFFSET,
    RODATA_ORDINAL,
    RODATA_STATIC_OFFSET,
    SELECTOR,
    STRING_ORDINAL,
    STRING_ORDINAL_NONEMPTY,
)
FIELD_RULES = (*BASIC_FIELD_RULES, *INSTRUCTION_FIELD_RULES)
RECORD_RULES = (
    CONTROL_CALL,
    CONTROL_CALL_INDIRECT,
    CONTROL_RETURN_SIGNATURE,
    CONTROL_SWITCH_TARGETS,
    COUNT_NONZERO_WHEN_SELECTOR,
    FIELDS_DISTINCT,
    FUNCTION_ADDRESS,
    INTEGER_BITSTREAM_SHAPE,
    PACKED_SELECTOR_ALLOWED_PAIRS,
    PACKED_SELECTOR_INDEXED_ALLOWED_MASK,
    PACKED_SELECTOR_TARGET_SUPPORTED,
    VALUE_REGISTER_RANGE,
    VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT,
)

FIELD_RULES_BY_KIND = {
    "abi.slot": ABI_SLOT,
    "allowed.bits": ALLOWED_BITS,
    "allowed.bits.exactly_one": ALLOWED_BITS_EXACTLY_ONE,
    "allowed.range": ALLOWED_RANGE,
    "allowed.values": ALLOWED_VALUES,
    "constant.pool.ordinal": CONSTANT_POOL_ORDINAL,
    "constraint.member": CONSTRAINT_MEMBER,
    "control.target.relative.s16": CONTROL_TARGET_RELATIVE_S16,
    "control.target.relative.s32": CONTROL_TARGET_RELATIVE_S32,
    "function.local.ordinal": FUNCTION_LOCAL_ORDINAL,
    "global.ordinal": GLOBAL_ORDINAL,
    "immediate.bits": ANY_BITS,
    "import.ordinal.optional": IMPORT_ORDINAL_OPTIONAL,
    "local.bytes.fixed.base": LOCAL_BYTES_FIXED_BASE,
    "local.bytes.range.base": LOCAL_BYTES_RANGE_BASE,
    "local.bytes.range.length": LOCAL_BYTES_RANGE_LENGTH,
    "local.bytes.range.memory_format": LOCAL_BYTES_RANGE_MEMORY_FORMAT,
    "local.bytes.repeated.base": LOCAL_BYTES_REPEATED_BASE,
    "local.bytes.repeated.count": LOCAL_BYTES_REPEATED_COUNT,
    "packed.selectors": PACKED_SELECTORS,
    "range.base": RANGE_BASE,
    "range.count": RANGE_COUNT,
    "ref.slot": REF_SLOT,
    "register.function": REGISTER_FUNCTION,
    "register.ref": REGISTER_REF,
    "register.value": REGISTER_VALUE,
    "reserved.zero": ZERO,
    "rodata.executable_name_table": RODATA_EXECUTABLE_NAME_TABLE,
    "rodata.offset": RODATA_OFFSET,
    "rodata.ordinal": RODATA_ORDINAL,
    "rodata.static.offset": RODATA_STATIC_OFFSET,
    "selector": SELECTOR,
    "string.ordinal": STRING_ORDINAL,
    "string.ordinal.nonempty": STRING_ORDINAL_NONEMPTY,
}
RECORD_RULES_BY_KIND = {
    "control.call": CONTROL_CALL,
    "control.call.indirect": CONTROL_CALL_INDIRECT,
    "control.return.signature": CONTROL_RETURN_SIGNATURE,
    "control.switch.targets": CONTROL_SWITCH_TARGETS,
    "count.nonzero.when.selector": COUNT_NONZERO_WHEN_SELECTOR,
    "fields.distinct": FIELDS_DISTINCT,
    "func.address": FUNCTION_ADDRESS,
    "integer.bitstream.shape": INTEGER_BITSTREAM_SHAPE,
    "packed.selector.allowed_pairs": PACKED_SELECTOR_ALLOWED_PAIRS,
    "packed.selector.indexed_allowed_mask": (PACKED_SELECTOR_INDEXED_ALLOWED_MASK),
    "packed.selector.target_supported": PACKED_SELECTOR_TARGET_SUPPORTED,
    "value.register.range": VALUE_REGISTER_RANGE,
    "value.register.range.from_memory_format": (
        VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT
    ),
}
ENTITIES = (*FIELD_RULES, *RECORD_RULES)
