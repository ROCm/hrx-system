# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU target-low aliases that are intentionally blocked."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

_AMDGPU_DESCRIPTOR_PREFIX = "amdgpu."


def _validate_token(kind: str, value: str) -> None:
    if not value:
        raise ValueError(f"{kind} must not be empty")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    if any(char not in allowed for char in value):
        raise ValueError(
            f"{kind} {value!r} must contain only lowercase letters, digits, "
            "'.', '_', or '-'"
        )


@dataclass(frozen=True, slots=True)
class AmdgpuBlockedLowAlias:
    """Compatibility-only ISA alias that must stay out of authored low IR."""

    descriptor_key: str
    asm_mnemonic: str
    alias_semantics: str
    replacement_descriptor_key: str
    replacement_mnemonic: str
    decision_key: str = "rejected"
    reason_key: str = "compatibility_zero_semantics"

    @property
    def lookup_names(self) -> tuple[str, str]:
        return (self.descriptor_key, self.asm_mnemonic)

    def __post_init__(self) -> None:
        _validate_token("alias descriptor key", self.descriptor_key)
        _validate_token("alias asm mnemonic", self.asm_mnemonic)
        _validate_token("alias semantics", self.alias_semantics)
        _validate_token("replacement descriptor key", self.replacement_descriptor_key)
        _validate_token("replacement mnemonic", self.replacement_mnemonic)
        _validate_token("decision key", self.decision_key)
        _validate_token("reason key", self.reason_key)
        if not self.descriptor_key.startswith(_AMDGPU_DESCRIPTOR_PREFIX):
            raise ValueError(
                f"alias descriptor key {self.descriptor_key!r} must be qualified"
            )
        if self.asm_mnemonic.startswith(_AMDGPU_DESCRIPTOR_PREFIX):
            raise ValueError(
                f"alias asm mnemonic {self.asm_mnemonic!r} must be unqualified"
            )
        if not self.replacement_descriptor_key.startswith(_AMDGPU_DESCRIPTOR_PREFIX):
            raise ValueError(
                "replacement descriptor key "
                f"{self.replacement_descriptor_key!r} must be qualified"
            )
        if self.replacement_mnemonic.startswith(_AMDGPU_DESCRIPTOR_PREFIX):
            raise ValueError(
                "replacement mnemonic "
                f"{self.replacement_mnemonic!r} must be unqualified"
            )
        if self.descriptor_key.removeprefix(_AMDGPU_DESCRIPTOR_PREFIX) != (
            self.asm_mnemonic
        ):
            raise ValueError(
                f"alias descriptor key {self.descriptor_key!r} must match "
                f"alias mnemonic {self.asm_mnemonic!r}"
            )


AMDGPU_BLOCKED_LOW_ALIASES = (
    AmdgpuBlockedLowAlias(
        descriptor_key="amdgpu.v_fma_dx9_zero_f32",
        asm_mnemonic="v_fma_dx9_zero_f32",
        alias_semantics="dx9_zero",
        replacement_descriptor_key="amdgpu.v_fma_f32",
        replacement_mnemonic="v_fma_f32",
    ),
    AmdgpuBlockedLowAlias(
        descriptor_key="amdgpu.v_fmac_dx9_zero_f32",
        asm_mnemonic="v_fmac_dx9_zero_f32",
        alias_semantics="dx9_zero",
        replacement_descriptor_key="amdgpu.v_fmac_f32",
        replacement_mnemonic="v_fmac_f32",
    ),
)


def validate_amdgpu_blocked_low_aliases(
    aliases: Sequence[AmdgpuBlockedLowAlias],
) -> None:
    descriptor_keys = [alias.descriptor_key for alias in aliases]
    if descriptor_keys != sorted(descriptor_keys):
        raise ValueError("AMDGPU blocked low aliases must be sorted by descriptor key")
    if len(descriptor_keys) != len(set(descriptor_keys)):
        raise ValueError("AMDGPU blocked low alias descriptor keys must be unique")
    lookup_names = [
        lookup_name for alias in aliases for lookup_name in alias.lookup_names
    ]
    if len(lookup_names) != len(set(lookup_names)):
        raise ValueError("AMDGPU blocked low alias lookup names must be unique")


def sorted_amdgpu_blocked_low_aliases() -> tuple[AmdgpuBlockedLowAlias, ...]:
    validate_amdgpu_blocked_low_aliases(AMDGPU_BLOCKED_LOW_ALIASES)
    return AMDGPU_BLOCKED_LOW_ALIASES
