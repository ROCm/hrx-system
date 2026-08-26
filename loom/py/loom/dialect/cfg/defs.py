# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CFG dialect op definitions.

The dialect is intentionally small: branches carry semantic successor edges,
and block arguments represent edge payloads. Conditional branches do not carry
per-edge payloads yet; lowering can split critical edges through `cfg.br`
blocks when a selected arm needs values.
"""

from loom.assembly import (
    ARROW,
    COMMA,
    GLUE,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    Attr,
    BlockRef,
    BlockRefs,
    OptionalGroup,
    Ref,
    TypedRefs,
    kw,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_I64_ARRAY,
    I1,
    INDEX,
    TERMINATOR,
    AttrDef,
    Dialect,
    Op,
    Operand,
    OperandRole,
    OpPhase,
    Successor,
)

# ============================================================================
# Dialect
# ============================================================================

cfg_ops = Dialect(
    "cfg",
    dialect_id=0x12,
    doc="Unstructured control-flow operations.",
    default_phase=OpPhase.EXECUTABLE,
)

# ============================================================================
# cfg.br — unconditional branch
# ============================================================================

cfg_br = Op(
    "cfg.br",
    group=cfg_ops,
    doc="Unconditional branch to a successor block, forwarding zero or more block argument values.",
    operands=[Operand("args", ANY, variadic=True, doc="Values forwarded to the destination block arguments.")],
    successors=[Successor("dest", doc="Destination block.")],
    traits=[TERMINATOR],
    verify="loom_cfg_br_verify",
    format=[
        BlockRef("dest"),
        OptionalGroup(
            [GLUE, LPAREN, TypedRefs("args"), RPAREN],
            anchor="args",
        ),
    ],
    examples=[
        "cfg.br ^done",
        "cfg.br ^join(%value: i32)",
    ],
)

# ============================================================================
# cfg.cond_br — two-way conditional branch
# ============================================================================

cfg_cond_br = Op(
    "cfg.cond_br",
    group=cfg_ops,
    doc="Conditional branch to one of two successor blocks based on an i1 condition.",
    operands=[
        Operand(
            "condition",
            I1,
            doc="Scalar i1 branch condition.",
            role=OperandRole.CONTROL_CONDITION,
        ),
    ],
    successors=[
        Successor("true_dest", doc="Destination block when the condition is true."),
        Successor("false_dest", doc="Destination block when the condition is false."),
    ],
    successor_selector="condition",
    traits=[TERMINATOR],
    verify="loom_cfg_cond_br_verify",
    format=[
        Ref("condition"),
        COMMA,
        BlockRef("true_dest"),
        COMMA,
        BlockRef("false_dest"),
    ],
    examples=["cfg.cond_br %condition, ^then, ^else"],
)

# ============================================================================
# cfg.switch — multiway branch
# ============================================================================

cfg_switch = Op(
    "cfg.switch",
    group=cfg_ops,
    doc="Branch to one of a sorted set of case destinations or a default destination.",
    operands=[
        Operand(
            "selector",
            INDEX,
            doc="Index value selecting an explicit case or the default destination.",
            role=OperandRole.CONTROL_CONDITION,
        ),
    ],
    attrs=[
        AttrDef(
            "case_keys",
            ATTR_TYPE_I64_ARRAY,
            doc="Strictly increasing case keys paired with case destinations.",
        ),
    ],
    successors=[
        Successor("default_dest", doc="Destination when no case key matches."),
        Successor(
            "case_dests",
            doc="Destinations paired by ordinal with case keys.",
            variadic=True,
        ),
    ],
    successor_selector="selector",
    traits=[TERMINATOR],
    verify="loom_cfg_switch_verify",
    format=[
        Ref("selector"),
        kw("cases"),
        Attr("case_keys"),
        ARROW,
        LBRACKET,
        BlockRefs("case_dests"),
        RBRACKET,
        kw("default"),
        BlockRef("default_dest"),
    ],
    examples=[
        "cfg.switch %selector cases [0, 2] -> [^case0, ^case2] default ^fallback",
    ],
)

ALL_CFG_OPS: tuple[Op, ...] = (
    cfg_br,
    cfg_cond_br,
    cfg_switch,
)
