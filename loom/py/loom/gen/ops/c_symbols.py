# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C symbol reference validation for generated op metadata."""

from __future__ import annotations

import re
from typing import Any

_C_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def symbol_fact_domain_symbol(op: Any) -> str | None:
    """Returns the validated C symbol fact-domain symbol for an Op, if any."""
    if op.symbol_def is None:
        return None
    fact_domain = getattr(op.symbol_def, "fact_domain", None)
    if fact_domain is None:
        return None
    if not isinstance(fact_domain, str) or not _C_SYMBOL_RE.fullmatch(fact_domain):
        raise ValueError(f"Op {op.name!r}: symbol_def.fact_domain must be a C symbol name, got {fact_domain!r}")
    return fact_domain


def normalize_c_symbol_reference(symbol: str) -> str:
    """Returns a bare C symbol name from an optional address-of reference."""
    symbol = symbol[1:] if symbol.startswith("&") else symbol
    if not _C_SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"C pointer interface field must be a C symbol name, got {symbol!r}")
    return symbol
