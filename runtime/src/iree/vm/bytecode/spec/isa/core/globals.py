# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bytecode-private per-process global-state instructions."""

from iree.vm.bytecode.spec.isa import (
    FailureCase,
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
    RuntimeRefPolicy,
    StateEffect,
)
from iree.vm.bytecode.spec.isa.core.rules import (
    FieldRule,
    RefNullPolicy,
    RefOwnership,
    StateAccess,
    StateResource,
)
from iree.vm.bytecode.spec.module.records import GLOBALS_HEADER
from iree.vm.bytecode.spec.schema import U8, U16, Field
from iree.vm.bytecode.spec.version import CORE_0

GLOBAL_FAMILY = InstructionFamily(
    name="global",
    since=CORE_0,
    summary="Bytecode-private per-process global-state operations.",
    contract=(
        "Global instructions access the calling module's opaque process-state slice "
        "through direct module-local ordinals. Value, ref, and function globals have "
        "separate spaces, each partitioned into a dense immutable prefix and mutable "
        "suffix. Verification proves the opcode-selected range, so execution performs "
        "no symbol lookup or mutability test. Immutable globals are set-once state of "
        "an unpublished process. Each has a transient initialization bit: an immutable "
        "store requires active construction authority and a clear bit, writes the value, "
        "then sets the bit; a construction-time load requires a set bit. Process sealing "
        "validates every required immutable and non-null global and makes the bits "
        "unreachable from published execution. Value globals are complete 64-bit cells. "
        "Every non-null ref global is an owning root whose descriptor declares exact "
        "type and nullability. Function globals contain complete non-owning 16-byte "
        "carriers and validate current-program identity, target, callable token, and "
        "MAY_YIELD implication before a store. All fallible checks precede mutation. "
        "These instructions add no synchronization; the host supplies data-race "
        "discipline for invocations sharing a process. No global operation suspends."
    ),
)

_GLOBALS_FIELD_OFFSETS = {
    wire_field.field.name: offset
    for wire_field, offset in zip(
        GLOBALS_HEADER.fields, GLOBALS_HEADER.field_offsets, strict=True
    )
}


def _global_extent_parameter(
    total_count_field: str, immutable_count_field: str, mutable: bool
) -> int:
    """Packs lower-count offset-plus-one:u16 and upper-count offset:u16."""
    upper_field = total_count_field if mutable else immutable_count_field
    lower_offset_plus_one = (
        _GLOBALS_FIELD_OFFSETS[immutable_count_field] + 1 if mutable else 0
    )
    return (lower_offset_plus_one << 16) | _GLOBALS_FIELD_OFFSETS[upper_field]


def _field(
    name: str, encoding, summary: str, role: FieldRole, rule
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, encoding, summary), role, rule)


def _value_register(name: str, role: FieldRole) -> InstructionField:
    return _field(
        name,
        U8,
        "Value-register side of the global transfer.",
        role,
        FieldRule.REGISTER_VALUE,
    )


def _function_register(name: str, role: FieldRole) -> InstructionField:
    return _field(
        name,
        U8,
        "Function-register side of the global transfer.",
        role,
        FieldRule.REGISTER_FUNCTION,
    )


def _ref_register(
    name: str,
    role: FieldRole,
    global_contract: str,
    ownership: RefOwnership,
) -> InstructionField:
    return InstructionField(
        Field(name, U8, "Ref-register side of the global transfer."),
        role,
        FieldRuleUse(FieldRule.REGISTER_REF),
        RuntimeRefPolicy(
            f"global.{global_contract}", RefNullPolicy.DESCRIPTOR, ownership
        ),
    )


def _global(
    global_contract: str,
    total_count_field: str,
    immutable_count_field: str,
    mutable: bool,
) -> InstructionField:
    return _field(
        "global_u16",
        U16,
        f"Direct module-local {global_contract} global ordinal.",
        FieldRole.IMMEDIATE,
        FieldRuleUse(
            FieldRule.GLOBAL_ORDINAL,
            values=(
                _global_extent_parameter(
                    total_count_field, immutable_count_field, mutable
                ),
            ),
        ),
    )


def _instruction(
    opcode: int,
    mnemonic: str,
    summary: str,
    register: InstructionField,
    global_field: InstructionField,
    behavior: str,
    success: tuple[str, ...],
    state_access: StateAccess,
    assembly: str,
    pseudocode: str,
    *,
    preconditions: tuple[str, ...] = (),
    failures: tuple[FailureCase, ...] = (),
    ownership: tuple[str, ...] = (),
) -> Instruction:
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=GLOBAL_FAMILY,
        summary=summary,
        fields=(register, global_field),
        semantics=None,
        behavior=behavior,
        success=success,
        assembly=assembly,
        pseudocode=pseudocode,
        state_effects=(
            StateEffect(state_access, StateResource.PROCESS_GLOBALS, ("global_u16",)),
        ),
        preconditions=preconditions,
        failures=failures,
        ownership=ownership,
    )


def _value_global(mutable: bool) -> InstructionField:
    return _global(
        f"value.{'mutable' if mutable else 'immutable'}",
        "value_count_u32",
        "immutable_value_count_u32",
        mutable,
    )


GLOBAL_VALUE_IMMUTABLE_LOAD = _instruction(
    0x30,
    "global.value.immutable.load",
    "Loads a set-once value global.",
    _value_register("destination_v8", FieldRole.RESULT),
    _value_global(False),
    "Copies all 64 bits from a set-once value global into destination_v8.",
    ("destination_v8 receives all 64 bits of the selected immutable global.",),
    StateAccess.READ,
    "%v<destination> = global.value.immutable.load @gv<global>",
    "require(construction == NULL || immutable_value_bits[global_u16]);\n"
    "values[destination_v8] = value_globals[global_u16];\n"
    "pc = pc + 4;",
    preconditions=(
        "During process construction, the selected initialization bit is set.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "Construction is active and the initialization bit is clear.",
            "destination_v8 and all process state remain unchanged.",
        ),
    ),
)

GLOBAL_VALUE_IMMUTABLE_STORE = _instruction(
    0x31,
    "global.value.immutable.store",
    "Initializes one set-once value global.",
    _value_register("source_v8", FieldRole.OPERAND),
    _value_global(False),
    "Copies all 64 bits of source_v8 into an open set-once value global.",
    (
        "The global receives all 64 bits of source_v8, then its initialization bit is "
        "set; source_v8 is unchanged.",
    ),
    StateAccess.WRITE,
    "global.value.immutable.store %v<source>, @gv<global>",
    "require(construction != NULL && !immutable_value_bits[global_u16]);\n"
    "value_globals[global_u16] = values[source_v8];\n"
    "immutable_value_bits[global_u16] = true;\n"
    "pc = pc + 4;",
    preconditions=(
        "Process-construction authority is active and the initialization bit is clear.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "Construction authority is absent or the slot was already set.",
            "The global value, initialization bit, and source remain unchanged.",
        ),
    ),
)

GLOBAL_VALUE_MUTABLE_LOAD = _instruction(
    0x32,
    "global.value.mutable.load",
    "Loads a mutable value global.",
    _value_register("destination_v8", FieldRole.RESULT),
    _value_global(True),
    "Copies all 64 bits from a mutable value global into destination_v8.",
    ("destination_v8 receives all 64 bits of the selected mutable global.",),
    StateAccess.READ,
    "%v<destination> = global.value.mutable.load @gv<global>",
    "values[destination_v8] = value_globals[global_u16];\npc = pc + 4;",
)

GLOBAL_VALUE_MUTABLE_STORE = _instruction(
    0x33,
    "global.value.mutable.store",
    "Stores a mutable value global.",
    _value_register("source_v8", FieldRole.OPERAND),
    _value_global(True),
    "Copies all 64 bits of source_v8 into a mutable value global.",
    (
        "The selected global receives all 64 bits of source_v8; the source is unchanged.",
    ),
    StateAccess.WRITE,
    "global.value.mutable.store %v<source>, @gv<global>",
    "value_globals[global_u16] = values[source_v8];\npc = pc + 4;",
)


def _ref_global(mutable: bool) -> InstructionField:
    return _global(
        f"ref.{'mutable' if mutable else 'immutable'}",
        "ref_count_u32",
        "immutable_ref_count_u32",
        mutable,
    )


GLOBAL_REF_IMMUTABLE_LOAD_BORROW = _instruction(
    0x34,
    "global.ref.immutable.load.borrow",
    "Borrows a set-once ref global.",
    _ref_register(
        "destination_r8",
        FieldRole.RESULT,
        "ref.immutable",
        RefOwnership.BORROW,
    ),
    _ref_global(False),
    "Publishes an internal borrow from a set-once ref-global owner into "
    "destination_r8 without retaining it.",
    (
        "A nullable null global publishes canonical null; a non-null global publishes "
        "its exact object and descriptor tagged borrowed.",
    ),
    StateAccess.READ,
    "%r<destination> = global.ref.immutable.load.borrow @gr<global>",
    "require(construction == NULL || immutable_ref_bits[global_u16]);\n"
    "new_ref = borrow_ref(ref_globals[global_u16]);\n"
    "replace_ref(&refs[destination_r8], new_ref);\n"
    "pc = pc + 4;",
    preconditions=(
        "During process construction, the selected initialization bit is set.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "Construction is active and the initialization bit is clear.",
            "destination_r8 and all process state remain unchanged.",
        ),
    ),
    ownership=(
        "The immutable global retains the sole owner and dominates every frame and "
        "suspension using the borrow. A previous destination owner is released only "
        "after the borrowed state is installed.",
    ),
)

GLOBAL_REF_IMMUTABLE_STORE_MOVE = _instruction(
    0x35,
    "global.ref.immutable.store.move",
    "Publishes a ref into a set-once global.",
    _ref_register(
        "source_r8",
        FieldRole.OPERAND,
        "ref.immutable",
        RefOwnership.PUBLISH_MOVE,
    ),
    _ref_global(False),
    "Promotes a borrow when necessary, moves the owner or accepted null into an open "
    "set-once ref global, clears source_r8, and sets the initialization bit.",
    (
        "The global receives canonical null or an owner-backed exact ref, source_r8 "
        "becomes null, and the initialization bit becomes set.",
    ),
    StateAccess.WRITE,
    "global.ref.immutable.store.move %r<source>, @gr<global>",
    "require(construction != NULL && !immutable_ref_bits[global_u16]);\n"
    "new_owner = validate_and_promote_ref(refs[source_r8], descriptor);\n"
    "ref_globals[global_u16] = new_owner;\n"
    "refs[source_r8] = canonical_null_ref;\n"
    "immutable_ref_bits[global_u16] = true;\n"
    "pc = pc + 4;",
    preconditions=(
        "Construction authority is active and the initialization bit is clear.",
        "source_r8 satisfies the global descriptor's exact type and nullability.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "Construction authority is absent or the slot was already set.",
            "The source, global, and initialization bit remain unchanged.",
        ),
        FailureCase(
            "invalid_argument",
            "The source violates exact type or nullability.",
            "The source, global, and initialization bit remain unchanged.",
        ),
    ),
    ownership=(
        "A borrowed source is retained into an owner before mutation; an owned source "
        "transfers its obligation. The clear initialization bit guarantees that the "
        "physical destination owns nothing before the store.",
    ),
)

GLOBAL_REF_MUTABLE_LOAD_RETAIN = _instruction(
    0x36,
    "global.ref.mutable.load.retain",
    "Retains a snapshot of a mutable ref global.",
    _ref_register(
        "destination_r8",
        FieldRole.RESULT,
        "ref.mutable",
        RefOwnership.RETAIN,
    ),
    _ref_global(True),
    "Creates an owned snapshot of a mutable ref global in destination_r8.",
    (
        "A nullable null global publishes null; a non-null global publishes one "
        "retained owner with its exact object and descriptor.",
    ),
    StateAccess.READ,
    "%r<destination> = global.ref.mutable.load.retain @gr<global>",
    "snapshot = ref_globals[global_u16];\n"
    "require(construction == NULL || snapshot.object != NULL || descriptor.nullable);\n"
    "new_owner = retain_ref(snapshot);\n"
    "replace_ref(&refs[destination_r8], new_owner);\n"
    "pc = pc + 4;",
    preconditions=(
        "During process construction, a nonnullable global is not still temporary null.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "Construction is active and a nonnullable global is still null.",
            "destination_r8 and the global remain unchanged.",
        ),
    ),
    ownership=(
        "A non-null snapshot is retained before destination_r8 is replaced and a "
        "previous destination owner is released.",
    ),
)

GLOBAL_REF_MUTABLE_STORE_MOVE = _instruction(
    0x37,
    "global.ref.mutable.store.move",
    "Moves a ref into a mutable global.",
    _ref_register(
        "source_r8",
        FieldRole.OPERAND,
        "ref.mutable",
        RefOwnership.PUBLISH_MOVE,
    ),
    _ref_global(True),
    "Promotes a borrowed source when necessary, moves the owner or accepted null into "
    "a mutable ref global, clears source_r8, and releases the previous global owner.",
    (
        "The global receives canonical null or an owner-backed exact ref and source_r8 "
        "becomes canonical null.",
    ),
    StateAccess.WRITE,
    "global.ref.mutable.store.move %r<source>, @gr<global>",
    "new_owner = validate_and_promote_ref(refs[source_r8], descriptor);\n"
    "old_owner = ref_globals[global_u16];\n"
    "ref_globals[global_u16] = new_owner;\n"
    "refs[source_r8] = canonical_null_ref;\n"
    "release_if_owned(old_owner);\n"
    "pc = pc + 4;",
    preconditions=(
        "source_r8 satisfies the global's exact type and nullability contract.",
    ),
    failures=(
        FailureCase(
            "invalid_argument",
            "The source violates exact type or nullability.",
            "The source and global remain unchanged.",
        ),
    ),
    ownership=(
        "A borrow is retained before mutation; an owner transfers its obligation. The "
        "new state is installed and source_r8 cleared before the previous global owner "
        "is released, making self-aliasing safe.",
    ),
)


def _function_global(mutable: bool) -> InstructionField:
    return _global(
        f"func.{'mutable' if mutable else 'immutable'}",
        "function_count_u32",
        "immutable_function_count_u32",
        mutable,
    )


GLOBAL_FUNC_IMMUTABLE_LOAD = _instruction(
    0x38,
    "global.func.immutable.load",
    "Loads a set-once function global.",
    _function_register("destination_f8", FieldRole.RESULT),
    _function_global(False),
    "Copies one complete set-once function global into destination_f8.",
    ("destination_f8 receives an exact 16-byte copy of the function global.",),
    StateAccess.READ,
    "%f<destination> = global.func.immutable.load @gf<global>",
    "require(construction == NULL || immutable_function_bits[global_u16]);\n"
    "functions[destination_f8] = function_globals[global_u16];\n"
    "pc = pc + 4;",
    preconditions=(
        "During process construction, the selected initialization bit is set.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "Construction is active and the initialization bit is clear.",
            "destination_f8 and process state remain unchanged.",
        ),
    ),
    ownership=(
        "Function values own nothing and the copy retains no program or process.",
    ),
)

GLOBAL_FUNC_IMMUTABLE_STORE = _instruction(
    0x39,
    "global.func.immutable.store",
    "Initializes one set-once function global.",
    _function_register("source_f8", FieldRole.OPERAND),
    _function_global(False),
    "Validates and copies source_f8 into an open set-once function global.",
    (
        "The global receives an exact copy of source_f8, its initialization bit is set, "
        "and source_f8 remains unchanged.",
    ),
    StateAccess.WRITE,
    "global.func.immutable.store %f<source>, @gf<global>",
    "require(construction != NULL && !immutable_function_bits[global_u16]);\n"
    "validate_function_ref(functions[source_f8], descriptor, program);\n"
    "function_globals[global_u16] = functions[source_f8];\n"
    "immutable_function_bits[global_u16] = true;\n"
    "pc = pc + 4;",
    preconditions=(
        "Construction authority is active and the initialization bit is clear.",
        "source_f8 satisfies the global's nullability, current-program, target, "
        "callable-token, and MAY_YIELD contract.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "Construction authority is absent or the slot was already set.",
            "The source, global, and initialization bit remain unchanged.",
        ),
        FailureCase(
            "invalid_argument",
            "The function value violates the global descriptor.",
            "The source, global, and initialization bit remain unchanged.",
        ),
    ),
    ownership=(
        "Function values own nothing and the copy retains no program or process.",
    ),
)

GLOBAL_FUNC_MUTABLE_LOAD = _instruction(
    0x3A,
    "global.func.mutable.load",
    "Loads a mutable function global.",
    _function_register("destination_f8", FieldRole.RESULT),
    _function_global(True),
    "Copies one complete mutable function global into destination_f8.",
    ("destination_f8 receives an exact 16-byte copy of the function global.",),
    StateAccess.READ,
    "%f<destination> = global.func.mutable.load @gf<global>",
    "value = function_globals[global_u16];\n"
    "require(construction == NULL || !value.is_null || descriptor.nullable);\n"
    "functions[destination_f8] = value;\n"
    "pc = pc + 4;",
    preconditions=(
        "During process construction, a nonnullable global is not still temporary null.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "Construction is active and a nonnullable global is still null.",
            "destination_f8 and the global remain unchanged.",
        ),
    ),
    ownership=(
        "Function values own nothing and the copy retains no program or process.",
    ),
)

GLOBAL_FUNC_MUTABLE_STORE = _instruction(
    0x3B,
    "global.func.mutable.store",
    "Stores a mutable function global.",
    _function_register("source_f8", FieldRole.OPERAND),
    _function_global(True),
    "Validates and copies source_f8 into a mutable function global.",
    (
        "The global receives an exact 16-byte copy of source_f8 and the source remains "
        "unchanged.",
    ),
    StateAccess.WRITE,
    "global.func.mutable.store %f<source>, @gf<global>",
    "validate_function_ref(functions[source_f8], descriptor, program);\n"
    "function_globals[global_u16] = functions[source_f8];\n"
    "pc = pc + 4;",
    preconditions=(
        "source_f8 satisfies the global's nullability, current-program, target, "
        "callable-token, and MAY_YIELD contract.",
    ),
    failures=(
        FailureCase(
            "invalid_argument",
            "The function value violates the global descriptor.",
            "The source and global remain unchanged.",
        ),
    ),
    ownership=("Function values own nothing; replacement requires no cleanup.",),
)

GLOBAL_INSTRUCTIONS = (
    GLOBAL_VALUE_IMMUTABLE_LOAD,
    GLOBAL_VALUE_IMMUTABLE_STORE,
    GLOBAL_VALUE_MUTABLE_LOAD,
    GLOBAL_VALUE_MUTABLE_STORE,
    GLOBAL_REF_IMMUTABLE_LOAD_BORROW,
    GLOBAL_REF_IMMUTABLE_STORE_MOVE,
    GLOBAL_REF_MUTABLE_LOAD_RETAIN,
    GLOBAL_REF_MUTABLE_STORE_MOVE,
    GLOBAL_FUNC_IMMUTABLE_LOAD,
    GLOBAL_FUNC_IMMUTABLE_STORE,
    GLOBAL_FUNC_MUTABLE_LOAD,
    GLOBAL_FUNC_MUTABLE_STORE,
)
