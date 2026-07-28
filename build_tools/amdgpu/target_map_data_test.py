# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from build_tools.amdgpu.target_map_data import (
    AMDGPU_CODE_OBJECT_COMPATIBILITY_INFOS,
    AmdgpuCodeObjectCompatibilityInfo,
    AmdgpuGenericCodeObjectInfo,
    generic_code_object_current_version,
    validate_code_object_compatibility,
)


class TargetMapDataTest(unittest.TestCase):
    def test_canonical_map_is_closed_and_versioned(self):
        validate_code_object_compatibility()
        self.assertEqual(generic_code_object_current_version("gfx11-generic"), 1)
        gfx1151 = next(
            info
            for info in AMDGPU_CODE_OBJECT_COMPATIBILITY_INFOS
            if info.exact_processor == "gfx1151"
        )
        self.assertEqual(gfx1151.code_object_processor, "gfx11-generic")
        self.assertEqual(gfx1151.generic_introduction_version, 1)

    def test_rejects_member_newer_than_generic_code_object(self):
        with self.assertRaisesRegex(ValueError, "outside .* supported range"):
            validate_code_object_compatibility(
                (AmdgpuGenericCodeObjectInfo("gfx11-generic", 1),),
                (AmdgpuCodeObjectCompatibilityInfo("gfx1151", "gfx11-generic", 2),),
            )

    def test_rejects_undeclared_generic_family(self):
        with self.assertRaisesRegex(ValueError, "unknown generic"):
            validate_code_object_compatibility(
                (AmdgpuGenericCodeObjectInfo("gfx11-generic", 1),),
                (AmdgpuCodeObjectCompatibilityInfo("gfx1200", "gfx12-generic", 1),),
            )

    def test_rejects_unreferenced_generic_family(self):
        with self.assertRaisesRegex(ValueError, "have no exact members"):
            validate_code_object_compatibility(
                (
                    AmdgpuGenericCodeObjectInfo("gfx11-generic", 1),
                    AmdgpuGenericCodeObjectInfo("gfx12-generic", 1),
                ),
                (AmdgpuCodeObjectCompatibilityInfo("gfx1151", "gfx11-generic", 1),),
            )


if __name__ == "__main__":
    unittest.main()
