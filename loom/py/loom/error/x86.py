# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""X86 domain — X86-owned legality and lowering diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_X86_001: X86 SysV ABI layout is invalid.
ERR_X86_001 = ErrorDef(
    domain=ErrorDomain.X86,
    code=1,
    severity=Severity.ERROR,
    summary="X86 SysV ABI layout is invalid.",
    message="X86 function '@{function_name}' has an invalid SysV ABI layout",
    params=(ErrorParam("function_name", ParamKind.STRING),),
    fix_hint=(
        "Regenerate the function through source-to-Low lowering or use the "
        "canonical SysV argument and result locations for its signature"
    ),
)

# ERR_X86_002: X86 SysV argument placement is inconsistent.
ERR_X86_002 = ErrorDef(
    domain=ErrorDomain.X86,
    code=2,
    severity=Severity.ERROR,
    summary="X86 SysV argument placement is inconsistent.",
    message=(
        "X86 function '@{function_name}' operation '{op_name}' places argument "
        "{argument_ordinal} in {actual_location}, but its canonical SysV "
        "location is {expected_location}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("argument_ordinal", ParamKind.U32),
        ErrorParam("actual_location", ParamKind.STRING),
        ErrorParam("expected_location", ParamKind.STRING),
    ),
    fix_hint="Run x86 SysV ABI materialization after call inlining and cleanup",
)

# ERR_X86_003: X86 SysV stack argument offset is inconsistent.
ERR_X86_003 = ErrorDef(
    domain=ErrorDomain.X86,
    code=3,
    severity=Severity.ERROR,
    summary="X86 SysV stack argument offset is inconsistent.",
    message=(
        "X86 function '@{function_name}' operation '{op_name}' gives argument "
        "{argument_ordinal} stack byte offset {actual_offset}, but its canonical "
        "SysV offset is {expected_offset}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("argument_ordinal", ParamKind.U32),
        ErrorParam("actual_offset", ParamKind.I64),
        ErrorParam("expected_offset", ParamKind.U32),
    ),
    fix_hint="Run x86 SysV ABI materialization after call inlining and cleanup",
)

ALL_X86_ERRORS: tuple[ErrorDef, ...] = (
    ERR_X86_001,
    ERR_X86_002,
    ERR_X86_003,
)
