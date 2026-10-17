# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""X86 domain — X86-owned legality and lowering diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

ERR_X86_004 = ErrorDef(
    domain=ErrorDomain.X86,
    code=4,
    severity=Severity.ERROR,
    summary="X86 native emission contract is not satisfied.",
    message="x86 native emission requires {constraint}",
    params=(ErrorParam("constraint", ParamKind.STRING),),
)

ERR_X86_005 = ErrorDef(
    domain=ErrorDomain.X86,
    code=5,
    severity=Severity.ERROR,
    summary="An instruction has no native X86 encoding.",
    message="x86 native encoding is unavailable for '{operation_name}'",
    params=(ErrorParam("operation_name", ParamKind.STRING),),
)

ERR_X86_006 = ErrorDef(
    domain=ErrorDomain.X86,
    code=6,
    severity=Severity.ERROR,
    summary="A private x86 contract field was authored directly.",
    message=(
        "x86 target attribute '{attribute_name}' is compiler-private; use 'features'"
    ),
    params=(ErrorParam("attribute_name", ParamKind.STRING),),
)

ERR_X86_007 = ErrorDef(
    domain=ErrorDomain.X86,
    code=7,
    severity=Severity.ERROR,
    summary="An x86 target selector cannot enable the requested feature.",
    message=(
        "x86 target feature '{feature_name}' cannot be enabled for "
        "selector '{selector_name}'"
    ),
    params=(
        ErrorParam("feature_name", ParamKind.STRING),
        ErrorParam("selector_name", ParamKind.STRING),
    ),
)

ALL_X86_ERRORS = (
    ERR_X86_004,
    ERR_X86_005,
    ERR_X86_006,
    ERR_X86_007,
)
