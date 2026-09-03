# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compatibility versions for VM specification authorities."""

import re
from typing import NamedTuple

_AUTHORITY_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


class Version(NamedTuple):
    authority: str
    major: int
    minor: int

    def is_valid(self) -> bool:
        return bool(_AUTHORITY_PATTERN.fullmatch(self.authority)) and all(
            isinstance(component, int) and 0 <= component <= 0xFFFF
            for component in (self.major, self.minor)
        )

    def is_available_in(self, version: "Version") -> bool:
        """Returns whether a declaration introduced here is visible in version."""
        valid = self.is_valid() and version.is_valid()
        valid &= (self.authority, self.major) == (version.authority, version.major)
        return valid and self.minor <= version.minor

    def select(self, items):
        return tuple(item for item in items if item.since.is_available_in(self))


CORE_0 = Version("core", 0, 0)
