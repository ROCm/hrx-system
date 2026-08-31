# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.arch.amd.xdna.hardware_tables import (
    emit_array_facts,
    emit_device_profiles,
    emit_register_facts,
)


def test_hardware_table_family_is_emitted_from_one_materialized_source() -> None:
    array_contents = emit_array_facts()
    register_contents = emit_register_facts()
    profile_contents = emit_device_profiles()

    assert "kLoomXdnaNpu2ArrayFamily" in array_contents
    assert "kLoomXdnaRegisterPatterns" in register_contents
    assert "kLoomXdnaRegisterFieldCount = 173" in register_contents
    assert "kLoomXdnaDeviceProfiles" in profile_contents
    assert "0x535848414c4f0001" in profile_contents
