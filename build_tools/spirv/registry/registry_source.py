# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source identities and diagnostics shared by Khronos registry importers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegistryOrigin:
    """A stable semantic location within one canonical registry source."""

    # Human-readable source name or path supplied at the importer boundary.
    source: str
    # Stable semantic location within the source document.
    locator: str

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("registry origin source must not be empty")
        if not self.locator:
            raise ValueError("registry origin locator must not be empty")

    def child(self, locator: str) -> RegistryOrigin:
        """Returns an origin beneath this source location."""

        if not locator:
            raise ValueError("registry child locator must not be empty")
        separator = "" if locator.startswith("[") else "."
        return RegistryOrigin(self.source, f"{self.locator}{separator}{locator}")

    def __str__(self) -> str:
        return f"{self.source}:{self.locator}"


class RegistrySourceError(ValueError):
    """A malformed or contradictory canonical-registry source fact."""

    def __init__(self, origin: RegistryOrigin, reason: str) -> None:
        if not reason:
            raise ValueError("registry source error reason must not be empty")
        # Exact source location whose fact violated the importer contract.
        self.origin = origin
        # Human-readable description of the violated contract.
        self.reason = reason
        super().__init__(f"{origin}: {reason}")
