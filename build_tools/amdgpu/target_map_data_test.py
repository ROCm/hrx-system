# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from build_tools.amdgpu.target_map_data import (
    AMDGPU_DEVICE_BINARY_VARIANTS,
    AMDGPU_EXACT_TARGET_INFOS,
    AMDGPU_GENERIC_CODE_OBJECT_INFOS,
    TARGET_ID_FEATURE_SRAMECC,
    TARGET_ID_FEATURE_XNACK,
    AmdgpuAsicRevisionInfo,
    AmdgpuDeviceBinaryTarget,
    AmdgpuDeviceBinaryTargetMatch,
    AmdgpuExactTargetInfo,
    AmdgpuGenericCodeObjectInfo,
    generic_code_object_current_version,
    target_id_features_for_processor,
    validate_target_map_data,
)


class TargetMapDataTest(unittest.TestCase):
    def test_canonical_map_is_closed_and_versioned(self):
        validate_target_map_data()
        self.assertEqual(
            {
                info.processor: generic_code_object_current_version(info.processor)
                for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS
            },
            {
                info.processor: info.current_version
                for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS
            },
        )

    def test_rejects_unknown_generic_version_query(self):
        with self.assertRaisesRegex(ValueError, "unknown AMDGPU generic"):
            generic_code_object_current_version("gfx-future-generic")

    def test_rejects_member_newer_than_generic_code_object(self):
        with self.assertRaisesRegex(ValueError, "outside .* supported range"):
            validate_target_map_data(
                (AmdgpuGenericCodeObjectInfo("gfx-test-generic", 1),),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-test-generic", 2),),
                (),
            )

    def test_rejects_undeclared_generic_family(self):
        with self.assertRaisesRegex(ValueError, "unknown generic"):
            validate_target_map_data(
                (AmdgpuGenericCodeObjectInfo("gfx-test-generic", 1),),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-undeclared-generic", 1),),
                (),
            )

    def test_rejects_unreferenced_generic_family(self):
        with self.assertRaisesRegex(ValueError, "have no exact members"):
            validate_target_map_data(
                (
                    AmdgpuGenericCodeObjectInfo("gfx-test-generic", 1),
                    AmdgpuGenericCodeObjectInfo("gfx-unused-generic", 1),
                ),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-test-generic", 1),),
                (),
            )

    def test_target_id_features_cover_exact_and_generic_processors(self):
        self.assertEqual(
            target_id_features_for_processor("gfx942"),
            (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
        )
        self.assertEqual(
            target_id_features_for_processor("gfx9-4-generic"),
            (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
        )
        self.assertEqual(target_id_features_for_processor("gfx1151"), ())

    def test_asic_revisions_are_finite_and_have_an_explicit_default(self):
        gfx1250 = next(
            info
            for info in AMDGPU_EXACT_TARGET_INFOS
            if info.exact_processor == "gfx1250"
        )
        self.assertEqual(
            gfx1250.asic_revisions,
            (
                AmdgpuAsicRevisionInfo(0, "a0"),
                AmdgpuAsicRevisionInfo(1, "b0"),
            ),
        )
        self.assertEqual(gfx1250.default_asic_revision, 1)
        self.assertEqual(gfx1250.target_id_features, ())

    def test_rejects_undeclared_default_asic_revision(self):
        with self.assertRaisesRegex(ValueError, "default ASIC revision.*not declared"):
            validate_target_map_data(
                (),
                (
                    AmdgpuExactTargetInfo(
                        "gfx-test",
                        "gfx-test",
                        0,
                        asic_revisions=(AmdgpuAsicRevisionInfo(0, "a0"),),
                        default_asic_revision=1,
                    ),
                ),
                (),
            )

    def test_rejects_out_of_range_asic_revision(self):
        with self.assertRaisesRegex(ValueError, "outside the uint32 range"):
            validate_target_map_data(
                (),
                (
                    AmdgpuExactTargetInfo(
                        "gfx-test",
                        "gfx-test",
                        0,
                        asic_revisions=(AmdgpuAsicRevisionInfo(2**32, "future"),),
                        default_asic_revision=2**32,
                    ),
                ),
                (),
            )

    def test_rejects_noncanonical_asic_revision_order(self):
        with self.assertRaisesRegex(ValueError, "ascending value order"):
            validate_target_map_data(
                (),
                (
                    AmdgpuExactTargetInfo(
                        "gfx-test",
                        "gfx-test",
                        0,
                        asic_revisions=(
                            AmdgpuAsicRevisionInfo(1, "b0"),
                            AmdgpuAsicRevisionInfo(0, "a0"),
                        ),
                        default_asic_revision=1,
                    ),
                ),
                (),
            )

    def test_rejects_variant_with_unsupported_asic_revision(self):
        with self.assertRaisesRegex(ValueError, "unsupported ASIC revision"):
            validate_target_map_data(
                (),
                (
                    AmdgpuExactTargetInfo(
                        "gfx-test",
                        "gfx-test",
                        0,
                        asic_revisions=(AmdgpuAsicRevisionInfo(0, "a0"),),
                        default_asic_revision=0,
                    ),
                ),
                (
                    AmdgpuDeviceBinaryTarget(
                        "gfx-test-qualified",
                        "gfx-test",
                        (AmdgpuDeviceBinaryTargetMatch("gfx-test", 1),),
                    ),
                ),
            )

    def test_multiple_revisioned_processors_are_data_only(self):
        exact_infos = (
            AmdgpuExactTargetInfo(
                "gfx-test-a",
                "gfx-test-a",
                0,
                asic_revisions=(
                    AmdgpuAsicRevisionInfo(0, "a0"),
                    AmdgpuAsicRevisionInfo(1, "b0"),
                ),
                default_asic_revision=1,
            ),
            AmdgpuExactTargetInfo(
                "gfx-test-b",
                "gfx-test-b",
                0,
                asic_revisions=(
                    AmdgpuAsicRevisionInfo(7, "c0"),
                    AmdgpuAsicRevisionInfo(9, "d0"),
                ),
                default_asic_revision=9,
            ),
        )
        variants = (
            AmdgpuDeviceBinaryTarget(
                "gfx-test-a-a0",
                "gfx-test-a",
                (AmdgpuDeviceBinaryTargetMatch("gfx-test-a", 0),),
            ),
            AmdgpuDeviceBinaryTarget(
                "gfx-test-b-c0",
                "gfx-test-b",
                (AmdgpuDeviceBinaryTargetMatch("gfx-test-b", 7),),
            ),
        )

        validate_target_map_data((), exact_infos, variants)

    def test_rejects_noncanonical_asic_revision_name(self):
        with self.assertRaisesRegex(ValueError, "canonical artifact coordinate"):
            validate_target_map_data(
                (),
                (
                    AmdgpuExactTargetInfo(
                        "gfx-test",
                        "gfx-test",
                        0,
                        asic_revisions=(AmdgpuAsicRevisionInfo(0, "A 0"),),
                        default_asic_revision=0,
                    ),
                ),
                (),
            )

    def test_canonical_device_binary_variants_validate(self):
        validate_target_map_data(
            AMDGPU_GENERIC_CODE_OBJECT_INFOS,
            AMDGPU_EXACT_TARGET_INFOS,
            AMDGPU_DEVICE_BINARY_VARIANTS,
        )


if __name__ == "__main__":
    unittest.main()
