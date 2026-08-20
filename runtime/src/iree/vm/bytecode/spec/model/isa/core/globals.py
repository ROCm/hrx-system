# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 process-global state instructions."""

from __future__ import annotations

from model.isa import (
    FailureCase,
    Instruction,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
    RefNullPolicy,
    RefOwnership,
    RuntimeRefPolicy,
)
from model.isa.declarations import (
    core_instruction,
    function_register,
    instruction_field,
    ref_register,
    value_register,
)
from model.isa.validation import GLOBAL_ORDINAL
from model.schema import U16, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.global",
    since=CORE_0,
    summary="Bytecode-private per-process global-state operations.",
    dependencies=("core.contract.machine",),
    document_order=7,
    normative_text=(
        "Global instructions access the calling module's opaque process-state "
        "slice through direct module-local ordinals. Value, ref, and function "
        "globals have separate spaces, each partitioned into a dense immutable "
        "prefix and mutable suffix. Verification proves the opcode-selected "
        "range, so execution performs no symbol lookup or mutability test. "
        "Immutable globals are set-once state of an unpublished process. Each "
        "has a transient initialization bit: an immutable store requires active "
        "construction authority and a clear bit, then writes the value before "
        "setting the bit; a construction-time load requires a set bit. Sealing "
        "validates all required immutable and non-null globals and makes those "
        "bits unreachable from published execution. Value globals are complete "
        "64-bit cells. Every non-null ref global is an owning root and its "
        "descriptor declares exact type and nullability. Function globals store "
        "complete non-owning 16-byte carriers and validate current-program "
        "identity, target, callable token, and MAY_YIELD implication before a "
        "store. All fallible checks precede mutation. These instructions add no "
        "synchronization; hosts provide the data-race discipline for invocations "
        "sharing a process. No global operation suspends."
    ),
)


def _global(global_contract: str):
    return instruction_field(
        "global_u16",
        2,
        U16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        f"Direct module-local {global_contract} global ordinal.",
        (RuleUse(GLOBAL_ORDINAL.entity_id, (global_contract,)),),
    )


def _semantics(
    description: str,
    verification: tuple[str, ...],
    preconditions: tuple[str, ...],
    success: tuple[str, ...],
    failures: tuple[FailureCase, ...],
    ownership: tuple[str, ...],
    assembly: str,
    pseudocode: str,
) -> InstructionSemantics:
    return InstructionSemantics(
        description=description,
        verification=verification,
        preconditions=preconditions,
        success=(*success, "The program counter advances by four bytes."),
        failures=failures,
        ownership=ownership,
        assembly=(assembly,),
        pseudocode=pseudocode,
    )


def _value_global(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    register_name: str,
    register_role: InstructionFieldRole,
    global_contract: str,
    semantics: InstructionSemantics,
) -> Instruction:
    return core_instruction(
        entity_id=entity_id,
        since=CORE_0,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            value_register(
                register_name,
                1,
                register_role,
                "Value-register side of the global transfer.",
            ),
            _global(global_contract),
        ),
        semantics=semantics,
    )


GLOBAL_VALUE_IMMUTABLE_LOAD = _value_global(
    entity_id="core.instruction.global.value.immutable.load",
    summary="Loads a set-once value global.",
    opcode=0x30,
    mnemonic="global.value.immutable.load",
    register_name="dst_v8",
    register_role=InstructionFieldRole.RESULT,
    global_contract="value.immutable",
    semantics=_semantics(
        "Copies all 64 bits from a set-once value global into dst_v8.",
        (
            "dst_v8 must be a valid value-register ordinal.",
            "global_u16 must be less than immutable_value_count.",
        ),
        (
            "During process construction, the selected initialization bit "
            "must already be set.",
        ),
        ("dst_v8 receives all 64 bits of the selected immutable global.",),
        (
            FailureCase(
                "failed_precondition",
                "Construction is active and the initialization bit is clear.",
                "dst_v8 and all process state remain unchanged.",
            ),
        ),
        (),
        "%v<dst> = global.value.immutable.load @gv<global>",
        (
            "if (construction != NULL &&\n"
            "    !state->immutable_value_bits[global_u16]) {\n"
            "  fail(failed_precondition);\n"
            "}\n"
            "values[dst_v8] = state->value_globals[global_u16];\n"
            "pc = pc + 4;"
        ),
    ),
)

GLOBAL_VALUE_IMMUTABLE_STORE = _value_global(
    entity_id="core.instruction.global.value.immutable.store",
    summary="Initializes one set-once value global.",
    opcode=0x31,
    mnemonic="global.value.immutable.store",
    register_name="src_v8",
    register_role=InstructionFieldRole.OPERAND,
    global_contract="value.immutable",
    semantics=_semantics(
        "Copies all 64 bits of src_v8 into an open set-once value global.",
        (
            "src_v8 must be a valid value-register ordinal.",
            "global_u16 must be less than immutable_value_count.",
        ),
        (
            "Process-construction authority must be active and the selected "
            "initialization bit must be clear.",
        ),
        (
            "The global receives all 64 bits of src_v8, then its initialization "
            "bit is set; src_v8 is unchanged.",
        ),
        (
            FailureCase(
                "failed_precondition",
                "Construction authority is absent or the slot was already set.",
                "The global value, initialization bit, and source remain unchanged.",
            ),
        ),
        (),
        "global.value.immutable.store %v<src>, @gv<global>",
        (
            "if (construction == NULL ||\n"
            "    state->immutable_value_bits[global_u16]) {\n"
            "  fail(failed_precondition);\n"
            "}\n"
            "state->value_globals[global_u16] = values[src_v8];\n"
            "state->immutable_value_bits[global_u16] = true;\n"
            "pc = pc + 4;"
        ),
    ),
)

GLOBAL_VALUE_MUTABLE_LOAD = _value_global(
    entity_id="core.instruction.global.value.mutable.load",
    summary="Loads a mutable value global.",
    opcode=0x32,
    mnemonic="global.value.mutable.load",
    register_name="dst_v8",
    register_role=InstructionFieldRole.RESULT,
    global_contract="value.mutable",
    semantics=_semantics(
        "Copies all 64 bits from a mutable value global into dst_v8.",
        (
            "dst_v8 must be a valid value-register ordinal.",
            "global_u16 must be in the mutable value-global suffix.",
        ),
        (),
        ("dst_v8 receives all 64 bits of the selected mutable global.",),
        (),
        (),
        "%v<dst> = global.value.mutable.load @gv<global>",
        "values[dst_v8] = state->value_globals[global_u16];\npc = pc + 4;",
    ),
)

GLOBAL_VALUE_MUTABLE_STORE = _value_global(
    entity_id="core.instruction.global.value.mutable.store",
    summary="Stores a mutable value global.",
    opcode=0x33,
    mnemonic="global.value.mutable.store",
    register_name="src_v8",
    register_role=InstructionFieldRole.OPERAND,
    global_contract="value.mutable",
    semantics=_semantics(
        "Copies all 64 bits of src_v8 into a mutable value global.",
        (
            "src_v8 must be a valid value-register ordinal.",
            "global_u16 must be in the mutable value-global suffix.",
        ),
        (),
        (
            "The selected global receives all 64 bits of src_v8; the source is "
            "unchanged.",
        ),
        (),
        (),
        "global.value.mutable.store %v<src>, @gv<global>",
        "state->value_globals[global_u16] = values[src_v8];\npc = pc + 4;",
    ),
)


def _ref_global(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    register_name: str,
    register_role: InstructionFieldRole,
    global_contract: str,
    ownership_policy: RefOwnership,
    semantics: InstructionSemantics,
) -> Instruction:
    return core_instruction(
        entity_id=entity_id,
        since=CORE_0,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            ref_register(
                register_name,
                1,
                register_role,
                "Ref-register side of the global transfer.",
                RuntimeRefPolicy(
                    f"global.{global_contract}",
                    RefNullPolicy.DESCRIPTOR,
                    ownership_policy,
                ),
            ),
            _global(global_contract),
        ),
        semantics=semantics,
    )


GLOBAL_REF_IMMUTABLE_LOAD_BORROW = _ref_global(
    entity_id="core.instruction.global.ref.immutable.load.borrow",
    summary="Borrows a set-once ref global.",
    opcode=0x34,
    mnemonic="global.ref.immutable.load.borrow",
    register_name="dst_r8",
    register_role=InstructionFieldRole.RESULT,
    global_contract="ref.immutable",
    ownership_policy=RefOwnership.BORROW,
    semantics=_semantics(
        (
            "Publishes an internal borrow from a set-once ref-global owner into "
            "dst_r8 without retaining it."
        ),
        (
            "dst_r8 must be a valid ref-register ordinal.",
            "global_u16 must be less than immutable_ref_count.",
        ),
        (
            "During process construction, the selected initialization bit "
            "must already be set.",
        ),
        (
            "A nullable null global publishes canonical null; a non-null global "
            "publishes its exact object and descriptor tagged borrowed.",
        ),
        (
            FailureCase(
                "failed_precondition",
                "Construction is active and the initialization bit is clear.",
                "dst_r8 and all process state remain unchanged.",
            ),
        ),
        (
            "The immutable global retains the sole owner and dominates every "
            "frame and suspension using the borrow. A previous owned dst_r8 is "
            "released only after the new state is installed.",
        ),
        "%r<dst> = global.ref.immutable.load.borrow @gr<global>",
        (
            "if (construction != NULL &&\n"
            "    !state->immutable_ref_bits[global_u16]) {\n"
            "  fail(failed_precondition);\n"
            "}\n"
            "new_ref = borrow_ref(state->ref_globals[global_u16]);\n"
            "replace_ref(&refs[dst_r8], new_ref);\n"
            "pc = pc + 4;"
        ),
    ),
)

GLOBAL_REF_IMMUTABLE_STORE_MOVE = _ref_global(
    entity_id="core.instruction.global.ref.immutable.store.move",
    summary="Publishes a ref into a set-once global.",
    opcode=0x35,
    mnemonic="global.ref.immutable.store.move",
    register_name="src_r8",
    register_role=InstructionFieldRole.OPERAND,
    global_contract="ref.immutable",
    ownership_policy=RefOwnership.PUBLISH_MOVE,
    semantics=_semantics(
        (
            "Promotes a borrow when necessary, moves the owner or accepted null "
            "from src_r8 into an open set-once ref global, clears src_r8, and "
            "sets the initialization bit."
        ),
        (
            "src_r8 must be a valid ref-register ordinal.",
            "global_u16 must be less than immutable_ref_count.",
        ),
        (
            "Construction authority must be active and the initialization bit clear.",
            "src_r8 must satisfy the global descriptor's exact type and nullability.",
        ),
        (
            "The global receives canonical null or an owner-backed exact ref, "
            "src_r8 becomes null, and the initialization bit becomes set.",
        ),
        (
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
        (
            "A borrowed source is retained into an owner before mutation; an "
            "owned source transfers its obligation. The clear initialization "
            "bit guarantees the physical destination owns nothing before store.",
        ),
        "global.ref.immutable.store.move %r<src>, @gr<global>",
        (
            "require(construction != NULL &&\n"
            "        !state->immutable_ref_bits[global_u16]);\n"
            "new_owner = validate_and_promote_ref(refs[src_r8], descriptor);\n"
            "state->ref_globals[global_u16] = new_owner;\n"
            "refs[src_r8] = canonical_null_ref;\n"
            "state->immutable_ref_bits[global_u16] = true;\n"
            "pc = pc + 4;"
        ),
    ),
)

GLOBAL_REF_MUTABLE_LOAD_RETAIN = _ref_global(
    entity_id="core.instruction.global.ref.mutable.load.retain",
    summary="Retains a snapshot of a mutable ref global.",
    opcode=0x36,
    mnemonic="global.ref.mutable.load.retain",
    register_name="dst_r8",
    register_role=InstructionFieldRole.RESULT,
    global_contract="ref.mutable",
    ownership_policy=RefOwnership.RETAIN,
    semantics=_semantics(
        "Creates an owned snapshot of a mutable ref global in dst_r8.",
        (
            "dst_r8 must be a valid ref-register ordinal.",
            "global_u16 must be in the mutable ref-global suffix.",
        ),
        (
            "A nonnullable global must not still contain its temporary "
            "construction-time null state.",
        ),
        (
            "A nullable null global publishes null; a non-null global publishes "
            "one retained owner with its exact object and descriptor.",
        ),
        (
            FailureCase(
                "failed_precondition",
                "A nonnullable global still contains temporary null.",
                "dst_r8 and the global remain unchanged.",
            ),
        ),
        (
            "A non-null snapshot is retained before dst_r8 is replaced and a "
            "previous destination owner is released.",
        ),
        "%r<dst> = global.ref.mutable.load.retain @gr<global>",
        (
            "snapshot = state->ref_globals[global_u16];\n"
            "require(snapshot.object != NULL || descriptor.nullable);\n"
            "new_owner = retain_ref(snapshot);\n"
            "replace_ref(&refs[dst_r8], new_owner);\n"
            "pc = pc + 4;"
        ),
    ),
)

GLOBAL_REF_MUTABLE_STORE_MOVE = _ref_global(
    entity_id="core.instruction.global.ref.mutable.store.move",
    summary="Moves a ref into a mutable global.",
    opcode=0x37,
    mnemonic="global.ref.mutable.store.move",
    register_name="src_r8",
    register_role=InstructionFieldRole.OPERAND,
    global_contract="ref.mutable",
    ownership_policy=RefOwnership.PUBLISH_MOVE,
    semantics=_semantics(
        (
            "Promotes a borrowed source when necessary, moves the resulting "
            "owner or accepted null into a mutable ref global, clears src_r8, "
            "and releases the previous global owner."
        ),
        (
            "src_r8 must be a valid ref-register ordinal.",
            "global_u16 must be in the mutable ref-global suffix.",
        ),
        ("src_r8 must satisfy the global's exact type and nullability contract.",),
        (
            "The global receives canonical null or an owner-backed exact ref "
            "and src_r8 becomes canonical null.",
        ),
        (
            FailureCase(
                "invalid_argument",
                "The source violates exact type or nullability.",
                "The source and global remain unchanged.",
            ),
        ),
        (
            "A borrow is retained before mutation; an owner transfers its "
            "obligation. The new state is installed and src_r8 cleared before "
            "the previous global owner is released, making self-aliasing safe.",
        ),
        "global.ref.mutable.store.move %r<src>, @gr<global>",
        (
            "new_owner = validate_and_promote_ref(refs[src_r8], descriptor);\n"
            "old_owner = state->ref_globals[global_u16];\n"
            "state->ref_globals[global_u16] = new_owner;\n"
            "refs[src_r8] = canonical_null_ref;\n"
            "release_if_owned(old_owner);\n"
            "pc = pc + 4;"
        ),
    ),
)


def _function_global(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    register_name: str,
    register_role: InstructionFieldRole,
    global_contract: str,
    semantics: InstructionSemantics,
) -> Instruction:
    return core_instruction(
        entity_id=entity_id,
        since=CORE_0,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            function_register(
                register_name,
                1,
                register_role,
                "Function-register side of the global transfer.",
            ),
            _global(global_contract),
        ),
        semantics=semantics,
    )


GLOBAL_FUNC_IMMUTABLE_LOAD = _function_global(
    entity_id="core.instruction.global.func.immutable.load",
    summary="Loads a set-once function global.",
    opcode=0x38,
    mnemonic="global.func.immutable.load",
    register_name="dst_f8",
    register_role=InstructionFieldRole.RESULT,
    global_contract="func.immutable",
    semantics=_semantics(
        "Copies one complete set-once function global into dst_f8.",
        (
            "dst_f8 must be a valid function-register ordinal.",
            "global_u16 must be less than immutable_function_count.",
        ),
        (
            "During process construction, the selected initialization bit "
            "must already be set.",
        ),
        ("dst_f8 receives an exact 16-byte copy of the function global.",),
        (
            FailureCase(
                "failed_precondition",
                "Construction is active and the initialization bit is clear.",
                "dst_f8 and process state remain unchanged.",
            ),
        ),
        (),
        "%f<dst> = global.func.immutable.load @gf<global>",
        (
            "require(construction == NULL ||\n"
            "        state->immutable_function_bits[global_u16]);\n"
            "functions[dst_f8] = state->function_globals[global_u16];\n"
            "pc = pc + 4;"
        ),
    ),
)

GLOBAL_FUNC_IMMUTABLE_STORE = _function_global(
    entity_id="core.instruction.global.func.immutable.store",
    summary="Initializes one set-once function global.",
    opcode=0x39,
    mnemonic="global.func.immutable.store",
    register_name="src_f8",
    register_role=InstructionFieldRole.OPERAND,
    global_contract="func.immutable",
    semantics=_semantics(
        "Validates and copies src_f8 into an open set-once function global.",
        (
            "src_f8 must be a valid function-register ordinal.",
            "global_u16 must be less than immutable_function_count.",
        ),
        (
            "Construction authority must be active and the initialization bit clear.",
            "src_f8 must be accepted by the global's nullability, current-program, "
            "target, callable-token, and MAY_YIELD contract.",
        ),
        (
            "The global receives an exact copy of src_f8, its initialization "
            "bit is set, and src_f8 remains unchanged.",
        ),
        (
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
        ("Function values own nothing and the copy retains no program or process.",),
        "global.func.immutable.store %f<src>, @gf<global>",
        (
            "require(construction != NULL &&\n"
            "        !state->immutable_function_bits[global_u16]);\n"
            "validate_function_ref(functions[src_f8], descriptor, program);\n"
            "state->function_globals[global_u16] = functions[src_f8];\n"
            "state->immutable_function_bits[global_u16] = true;\n"
            "pc = pc + 4;"
        ),
    ),
)

GLOBAL_FUNC_MUTABLE_LOAD = _function_global(
    entity_id="core.instruction.global.func.mutable.load",
    summary="Loads a mutable function global.",
    opcode=0x3A,
    mnemonic="global.func.mutable.load",
    register_name="dst_f8",
    register_role=InstructionFieldRole.RESULT,
    global_contract="func.mutable",
    semantics=_semantics(
        "Copies one complete mutable function global into dst_f8.",
        (
            "dst_f8 must be a valid function-register ordinal.",
            "global_u16 must be in the mutable function-global suffix.",
        ),
        (
            "While construction is open, a nonnullable global must not still "
            "contain temporary canonical null.",
        ),
        ("dst_f8 receives an exact 16-byte copy of the function global.",),
        (
            FailureCase(
                "failed_precondition",
                "Construction is active and a nonnullable global is still null.",
                "dst_f8 and the global remain unchanged.",
            ),
        ),
        (),
        "%f<dst> = global.func.mutable.load @gf<global>",
        (
            "value = state->function_globals[global_u16];\n"
            "require(construction == NULL || !value.is_null || descriptor.nullable);\n"
            "functions[dst_f8] = value;\n"
            "pc = pc + 4;"
        ),
    ),
)

GLOBAL_FUNC_MUTABLE_STORE = _function_global(
    entity_id="core.instruction.global.func.mutable.store",
    summary="Stores a mutable function global.",
    opcode=0x3B,
    mnemonic="global.func.mutable.store",
    register_name="src_f8",
    register_role=InstructionFieldRole.OPERAND,
    global_contract="func.mutable",
    semantics=_semantics(
        "Validates and copies src_f8 into a mutable function global.",
        (
            "src_f8 must be a valid function-register ordinal.",
            "global_u16 must be in the mutable function-global suffix.",
        ),
        (
            "src_f8 must be accepted by the global's nullability, current-program, "
            "target, callable-token, and MAY_YIELD contract.",
        ),
        (
            "The global receives an exact 16-byte copy of src_f8 and the source "
            "remains unchanged.",
        ),
        (
            FailureCase(
                "invalid_argument",
                "The function value violates the global descriptor.",
                "The source and global remain unchanged.",
            ),
        ),
        ("Function values own nothing; replacement requires no cleanup.",),
        "global.func.mutable.store %f<src>, @gf<global>",
        (
            "validate_function_ref(functions[src_f8], descriptor, program);\n"
            "state->function_globals[global_u16] = functions[src_f8];\n"
            "pc = pc + 4;"
        ),
    ),
)

INSTRUCTIONS = (
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
ENTITIES = (FAMILY, *INSTRUCTIONS)
