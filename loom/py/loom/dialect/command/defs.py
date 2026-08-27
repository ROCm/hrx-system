# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-program source operations."""

from loom.assembly import (
    COLON,
    COMMA,
    GLUE,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    Attr,
    FuncArgs,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    Region,
    ResultType,
    Scope,
    SymbolRef,
    TypesOf,
    kw,
)
from loom.dialect.func.defs import Retain, Visibility
from loom.dsl import (
    ANY,
    ATTR_TYPE_I64,
    ATTR_TYPE_STRING,
    BUFFER,
    COMMAND_EFFECT,
    INDEX,
    ISOLATED_FROM_ABOVE,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    VIEW,
    AttrDef,
    CallLikeInterface,
    CallLikeKind,
    Dialect,
    FuncLikeInterface,
    ImplicitTerminator,
    Op,
    Operand,
    OpPhase,
    RegionDef,
    Result,
    SymbolDefinition,
    SymbolDefinitionFlag,
    SymbolReference,
)

command_ops = Dialect(
    "command",
    dialect_id=0x1E,
    doc="Reusable command-program construction operations.",
)


_PROGRAM_MODIFIER_FORMAT = [
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    OptionalGroup([Attr("retain")], anchor="retain"),
    OptionalGroup(
        [kw("target"), GLUE, LPAREN, SymbolRef("target"), GLUE, RPAREN],
        anchor="target",
    ),
]

_PROGRAM_SIGNATURE_FORMAT = [
    SymbolRef("callee"),
    Scope(
        [
            FuncArgs(
                "args",
                group="specializations",
                end_attr="specialization_count",
            ),
            kw("launch"),
            FuncArgs(
                "args",
                group="bindings",
                start_attr="specialization_count",
            ),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")],
                anchor="predicates",
            ),
        ]
    ),
]

_PROGRAM_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef(
        "target",
        "symbol",
        optional=True,
        symbol_ref=SymbolReference("target", ["target"]),
    ),
    AttrDef("predicates", "predicate_list", optional=True),
    AttrDef(
        "specialization_count",
        ATTR_TYPE_I64,
        doc="Number of leading staged program arguments.",
    ),
    AttrDef("retain", "enum", enum_def=Retain, optional=True),
]

_PROGRAM_SYMBOL = dict(
    field="callee",
    name="command program",
    interfaces=["func_like", "command_program"],
    fact_domain="loom_func_symbol_fact_domain",
    retain="retain",
)

_PROGRAM_FUNC_LIKE = dict(
    callee="callee",
    visibility="visibility",
    target="target",
    predicates="predicates",
    specialization_count="specialization_count",
)


command_program_def = Op(
    "command.program.def",
    group=command_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Reusable command-program definition. Leading specialization arguments "
        "participate in staged specialization and launch-count evaluation; buffer "
        "bindings are provided when the materialized program is issued. Observable "
        "effects in the body must be explicit command operations; ordinary SSA "
        "preparation is effect-free."
    ),
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=list(_PROGRAM_ATTRS),
    symbol_def=SymbolDefinition(
        **_PROGRAM_SYMBOL,
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
    ),
    regions=[
        RegionDef(
            "body",
            doc="Commands and source-level control flow forming the program.",
            terminator="command.return",
            command_effects_only=True,
        ),
    ],
    interfaces=[FuncLikeInterface(**_PROGRAM_FUNC_LIKE, body="body")],
    verify="loom_command_program_def_verify",
    format=[
        *_PROGRAM_MODIFIER_FORMAT,
        *_PROGRAM_SIGNATURE_FORMAT,
        Region("body"),
    ],
    examples=[
        "command.program.def @decode(%token_count: index) launch(%parameters: buffer, %transient: buffer) {\n  command.return\n}",
        "command.program.def public target(@gfx1100) @prefill(%token_count: index) launch(%parameters: buffer) {\n  command.return\n}",
    ],
)


command_program_decl = Op(
    "command.program.decl",
    group=command_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Bodyless declaration of a reusable command program.",
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", ANY, variadic=True)],
    attrs=list(_PROGRAM_ATTRS),
    symbol_def=SymbolDefinition(
        **_PROGRAM_SYMBOL,
        bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
        flags=[SymbolDefinitionFlag.DECLARATION],
    ),
    interfaces=[FuncLikeInterface(**_PROGRAM_FUNC_LIKE, args="args")],
    verify="loom_command_program_decl_verify",
    format=[
        *_PROGRAM_MODIFIER_FORMAT,
        *_PROGRAM_SIGNATURE_FORMAT,
    ],
    examples=[
        "command.program.decl @decode(%token_count: index) launch(%parameters: buffer, %transient: buffer)",
        "command.program.decl @static_program() launch(%parameters: buffer)",
    ],
)


command_program_launch = Op(
    "command.program.launch",
    group=command_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Invoke a command program with explicit specialization values and issue-time buffer bindings."),
    operands=[
        Operand("specializations", ANY, variadic=True),
        Operand("bindings", BUFFER, variadic=True),
    ],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("command program", ["command_program"]),
        ),
    ],
    traits=[UNKNOWN_EFFECTS, COMMAND_EFFECT],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="specializations",
            results=None,
            kind=CallLikeKind.COMMAND_PROGRAM,
        ),
    ],
    verify="loom_command_program_launch_verify",
    format=[
        SymbolRef("callee"),
        OptionalGroup(
            [GLUE, LBRACKET, Refs("specializations"), RBRACKET],
            anchor="specializations",
        ),
        GLUE,
        LPAREN,
        Refs("bindings"),
        RPAREN,
        COLON,
        OptionalGroup(
            [LBRACKET, TypesOf("specializations"), RBRACKET, GLUE],
            anchor="specializations",
        ),
        LPAREN,
        TypesOf("bindings"),
        RPAREN,
    ],
    examples=[
        "command.program.launch @decode[%token_count](%parameters, %transient) : [index](buffer, buffer)",
        "command.program.launch @static_program(%parameters) : (buffer)",
    ],
)


command_parameter = Op(
    "command.parameter",
    group=command_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Associate immutable named parameter content with an explicit command-program "
        "buffer root. The pattern contains one canonical decimal placeholder for each "
        "index substitution. The result is a logical typed view; this operation performs "
        "no allocation, lookup, transfer, or synchronization."
    ),
    operands=[
        Operand("source", BUFFER, doc="Command-program buffer containing the parameter."),
        Operand(
            "substitutions",
            INDEX,
            variadic=True,
            doc="Canonical decimal values substituted into the parameter pattern.",
        ),
    ],
    results=[Result("result", VIEW, doc="Typed logical view of the named parameter.")],
    attrs=[
        AttrDef(
            "pattern",
            ATTR_TYPE_STRING,
            doc="Parameter key with one '{}' placeholder per substitution operand.",
        ),
    ],
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    verify="loom_command_parameter_verify",
    facts="loom_command_parameter_facts",
    format=[
        Ref("source"),
        COMMA,
        Attr("pattern"),
        OptionalGroup(
            [GLUE, LBRACKET, Refs("substitutions"), RBRACKET],
            anchor="substitutions",
        ),
        COLON,
        ResultType("result"),
    ],
    examples=[
        '%embedding = command.parameter %parameters, "token_embd.weight" : view<175030272xi8>',
        '%query = command.parameter %parameters, "blk.{}.attn_q.weight"[%layer] : view<4718592xi8>',
    ],
)


command_return = Op(
    "command.return",
    group=command_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Terminate a command-program definition.",
    traits=[TERMINATOR],
    examples=["command.return"],
)


command_yield = Op(
    "command.yield",
    group=command_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Terminate a structured command schedule region.",
    traits=[TERMINATOR],
    examples=["command.yield"],
)


def _schedule(name: str, doc: str) -> Op:
    return Op(
        name,
        group=command_ops,
        phase=OpPhase.EXECUTABLE,
        doc=doc,
        regions=[
            RegionDef(
                "body",
                doc="Nested command operations and source-level control flow.",
                single_block=True,
                terminator="command.yield",
            ),
        ],
        traits=[
            UNKNOWN_EFFECTS,
            COMMAND_EFFECT,
            ImplicitTerminator("command.yield"),
        ],
        format=[Region("body")],
        examples=[f"{name} {{\n  command.yield\n}}"],
    )


command_serial = _schedule(
    "command.serial",
    "Order each child command after the preceding child completes.",
)


command_concurrent = _schedule(
    "command.concurrent",
    "Permit child commands to execute without dependency edges between siblings and join them on exit.",
)


ALL_COMMAND_OPS: tuple[Op, ...] = (
    command_program_def,
    command_program_decl,
    command_program_launch,
    command_return,
    command_yield,
    command_serial,
    command_concurrent,
    command_parameter,
)
