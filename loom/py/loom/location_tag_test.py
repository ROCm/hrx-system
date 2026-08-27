# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.location_tag import (
    LOCATION_TAG_USER_BASE,
    BuiltinLocationTag,
    builtin_location_tag_name,
    parse_builtin_location_tag,
)


def test_builtin_location_tags_round_trip() -> None:
    for tag in BuiltinLocationTag:
        assert builtin_location_tag_name(tag.value) == tag.spelling
        assert parse_builtin_location_tag(tag.spelling) == tag.value


def test_unknown_location_tags_have_no_builtin_name() -> None:
    assert builtin_location_tag_name(0) is None
    assert builtin_location_tag_name(LOCATION_TAG_USER_BASE) is None
    assert parse_builtin_location_tag("not_a_location_tag") is None
