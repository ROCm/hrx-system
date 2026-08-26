# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders the complete module-verification obligation inventory."""

from __future__ import annotations

import re

from model.module import ValidationObligation
from model.specification import Projection

_TOKEN_PATTERN = re.compile(r"[^A-Z0-9]+")
_ENTITY_PREFIX = "core.validation.module."


def _obligation_token(entity_id: str) -> str:
    if not entity_id.startswith(_ENTITY_PREFIX):
        raise ValueError(f"module obligation has unexpected ID {entity_id!r}")
    token = _TOKEN_PATTERN.sub("_", entity_id.removeprefix(_ENTITY_PREFIX).upper())
    if not token or token[0].isdigit():
        raise ValueError(f"module obligation has invalid token {token!r}")
    return token


def render_module_validation_obligations(module_projection: Projection) -> str:
    """Renders an X-macro row for every authoritative loader obligation."""

    obligations = tuple(
        entity
        for entity in module_projection.entities
        if isinstance(entity, ValidationObligation)
    )
    tokens = tuple(_obligation_token(entity.entity_id) for entity in obligations)
    if len(tokens) != len(set(tokens)):
        raise ValueError("module verification obligation tokens are not unique")

    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED FILE: DO NOT EDIT.",
        "// Private build-tree projection: module verification obligations.",
        "// clang-format off",
        "",
        "#if !defined(IREE_VM_BYTECODE_MODULE_VALIDATION_OBLIGATION)",
        '#error "define IREE_VM_BYTECODE_MODULE_VALIDATION_OBLIGATION"',
        "#endif",
    ]
    for obligation, token in zip(obligations, tokens, strict=True):
        lines.append(
            "IREE_VM_BYTECODE_MODULE_VALIDATION_OBLIGATION("
            f'{token}, "{obligation.entity_id}")'
        )
    lines.extend(("", "// clang-format on", ""))
    return "\n".join(lines)
