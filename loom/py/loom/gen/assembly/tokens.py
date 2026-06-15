# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared Loom assembly token declarations used by generators."""

from __future__ import annotations

# Maps Python Keyword text to C keyword enum name. This is the single source of
# truth for all format keywords. The C table generator produces keyword_enum.inc
# and keyword_table.inc from this dict. Ordinals are assigned by position; append
# new keywords at the end.
KEYWORD_MAP: dict[str, str] = {
    ",": "LOOM_KW_COMMA",
    ":": "LOOM_KW_COLON",
    "->": "LOOM_KW_ARROW",
    "=": "LOOM_KW_EQUALS",
    "(": "LOOM_KW_LPAREN",
    ")": "LOOM_KW_RPAREN",
    "[": "LOOM_KW_LBRACKET",
    "]": "LOOM_KW_RBRACKET",
    "{": "LOOM_KW_LBRACE",
    "}": "LOOM_KW_RBRACE",
    "to": "LOOM_KW_TO",
    "step": "LOOM_KW_STEP",
    "else": "LOOM_KW_ELSE",
    "iter_args": "LOOM_KW_ITER_ARGS",
    "where": "LOOM_KW_WHERE",
    "as": "LOOM_KW_AS",
    "public": "LOOM_KW_PUBLIC",
    "host": "LOOM_KW_HOST",
    "device": "LOOM_KW_DEVICE",
    "priority": "LOOM_KW_PRIORITY",
    "x": "LOOM_KW_X",
    "import": "LOOM_KW_IMPORT",
    "layout": "LOOM_KW_LAYOUT",
    "into": "LOOM_KW_INTO",
    "default": "LOOM_KW_DEFAULT",
    "case": "LOOM_KW_CASE",
    "do": "LOOM_KW_DO",
    "using": "LOOM_KW_USING",
    "dgroups": "LOOM_KW_DGROUPS",
    "target": "LOOM_KW_TARGET",
    "allocation": "LOOM_KW_ALLOCATION",
    "schedule": "LOOM_KW_SCHEDULE",
    "source": "LOOM_KW_SOURCE",
    "abi": "LOOM_KW_ABI",
    "export": "LOOM_KW_EXPORT",
    "for": "LOOM_KW_FOR",
    "value": "LOOM_KW_VALUE",
    "seed": "LOOM_KW_SEED",
    "range": "LOOM_KW_RANGE",
    "path": "LOOM_KW_PATH",
    "actual": "LOOM_KW_ACTUAL",
    "expected": "LOOM_KW_EXPECTED",
    "atol": "LOOM_KW_ATOL",
    "attrs": "LOOM_KW_ATTRS",
    "base": "LOOM_KW_BASE",
    "bounds": "LOOM_KW_BOUNDS",
    "callee": "LOOM_KW_CALLEE",
    "count": "LOOM_KW_COUNT",
    "inputs": "LOOM_KW_INPUTS",
    "mode": "LOOM_KW_MODE",
    "nan": "LOOM_KW_NAN",
    "offset": "LOOM_KW_OFFSET",
    "reason": "LOOM_KW_REASON",
    "rtol": "LOOM_KW_RTOL",
    "shape": "LOOM_KW_SHAPE",
    "values": "LOOM_KW_VALUES",
    "artifact": "LOOM_KW_ARTIFACT",
    "ordinal": "LOOM_KW_ORDINAL",
    "linkage": "LOOM_KW_LINKAGE",
    "workgroup_size": "LOOM_KW_WORKGROUP_SIZE",
    "from": "LOOM_KW_FROM",
    "axes": "LOOM_KW_AXES",
    "config": "LOOM_KW_CONFIG",
    "launch": "LOOM_KW_LAUNCH",
    "workgroups": "LOOM_KW_WORKGROUPS",
    "abi_layout": "LOOM_KW_ABI_LAYOUT",
    "extent": "LOOM_KW_EXTENT",
    "module": "LOOM_KW_MODULE",
    "symbol": "LOOM_KW_SYMBOL",
    "unroll": "LOOM_KW_UNROLL",
    "name": "LOOM_KW_NAME",
}

# Maps Region(..., syntax=...) names to C parser/printer selector IDs. The
# empty name is the canonical braced region form.
REGION_SYNTAX_MAP: dict[str, str] = {
    "": "LOOM_REGION_SYNTAX_DEFAULT",
    "test.do": "LOOM_REGION_SYNTAX_TEST_DO",
    "low.asm": "LOOM_REGION_SYNTAX_LOW_ASM",
    "low.asm.optional": "LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL",
    "pipeline": "LOOM_REGION_SYNTAX_PIPELINE",
}
