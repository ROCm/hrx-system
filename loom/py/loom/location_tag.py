# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in location tag values and assembly spellings."""

from __future__ import annotations

from enum import IntEnum, unique

__all__ = [
    "BuiltinLocationTag",
    "LOCATION_TAG_SANITIZER_SITE",
    "LOCATION_TAG_TEMPLATE_INSTANTIATION",
    "LOCATION_TAG_TILE_LOWERING",
    "LOCATION_TAG_UKERNEL_SELECTION",
    "LOCATION_TAG_USER_BASE",
    "builtin_location_tag_name",
    "parse_builtin_location_tag",
]


@unique
class BuiltinLocationTag(IntEnum):
    """Built-in location payload tag with its canonical assembly spelling."""

    spelling: str

    def __new__(cls, value: int, spelling: str) -> BuiltinLocationTag:
        member = int.__new__(cls, value)
        member._value_ = value
        member.spelling = spelling
        return member

    SANITIZER_SITE = (0x0001, "sanitizer_site")
    TEMPLATE_INSTANTIATION = (0x0002, "template_instantiation")
    TILE_LOWERING = (0x0003, "tile_lowering")
    UKERNEL_SELECTION = (0x0004, "ukernel_selection")


# First tag value available to embedders and user-defined tooling.
LOCATION_TAG_USER_BASE = 0x8000

if any(
    tag.value <= 0 or tag.value >= LOCATION_TAG_USER_BASE for tag in BuiltinLocationTag
):
    raise ValueError("built-in location tags must precede the user tag namespace")
if any(
    not tag.spelling.replace("_", "").isalnum()
    or not tag.spelling.isascii()
    or tag.spelling != tag.spelling.lower()
    for tag in BuiltinLocationTag
):
    raise ValueError(
        "built-in location tag spellings must be lowercase ASCII identifier text"
    )
if len({tag.spelling for tag in BuiltinLocationTag}) != len(BuiltinLocationTag):
    raise ValueError("built-in location tag spellings must be unique")

_LOCATION_TAG_NAME_BY_VALUE = {tag.value: tag.spelling for tag in BuiltinLocationTag}
_LOCATION_TAG_VALUE_BY_NAME = {tag.spelling: tag.value for tag in BuiltinLocationTag}

LOCATION_TAG_SANITIZER_SITE = BuiltinLocationTag.SANITIZER_SITE.value
LOCATION_TAG_TEMPLATE_INSTANTIATION = BuiltinLocationTag.TEMPLATE_INSTANTIATION.value
LOCATION_TAG_TILE_LOWERING = BuiltinLocationTag.TILE_LOWERING.value
LOCATION_TAG_UKERNEL_SELECTION = BuiltinLocationTag.UKERNEL_SELECTION.value


def builtin_location_tag_name(tag: int) -> str | None:
    """Returns the canonical name of a built-in tag, if one is declared."""
    return _LOCATION_TAG_NAME_BY_VALUE.get(tag)


def parse_builtin_location_tag(name: str) -> int | None:
    """Returns the built-in tag matching ``name``, if one is declared."""
    return _LOCATION_TAG_VALUE_BY_NAME.get(name)
