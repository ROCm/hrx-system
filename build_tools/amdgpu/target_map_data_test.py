# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from build_tools.amdgpu.target_map_data import (
    AMDGPU_EXACT_TARGET_INFOS,
    AMDGPU_GENERIC_CODE_OBJECT_INFOS,
    AMDGPU_PHYSICAL_TARGET_INFOS,
    AMDGPU_TARGET_OVERLAY_INFOS,
    TARGET_ID_FEATURE_SRAMECC,
    TARGET_ID_FEATURE_XNACK,
    AmdgpuExactTargetInfo,
    AmdgpuGenericCodeObjectInfo,
    AmdgpuPhysicalTargetInfo,
    AmdgpuTargetOverlayInfo,
    generic_code_object_current_version,
    physical_target_info,
    processor_has_physical_target_infos,
    target_id_features_for_processor,
    target_processor,
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
                (),
            )

    def test_rejects_undeclared_generic_family(self):
        with self.assertRaisesRegex(ValueError, "unknown generic"):
            validate_target_map_data(
                (AmdgpuGenericCodeObjectInfo("gfx-test-generic", 1),),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-undeclared-generic", 1),),
                (),
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

    def test_target_overlay_selects_one_backend_processor(self):
        self.assertEqual(
            AMDGPU_TARGET_OVERLAY_INFOS,
            (
                AmdgpuTargetOverlayInfo(
                    target="gfx1250-a0",
                    processor="gfx1250",
                    compile_options=(
                        "-mllvm",
                        "-amdgpu-gfx1250-b0-specific=false",
                    ),
                    link_options=("-plugin-opt=-amdgpu-gfx1250-b0-specific=false",),
                ),
            ),
        )
        self.assertEqual(target_processor("gfx1250-a0"), "gfx1250")
        self.assertEqual(target_processor("gfx1250"), "gfx1250")
        self.assertEqual(target_processor("gfx12-5-generic"), "gfx12-5-generic")
        self.assertIsNone(target_processor("gfx-future"))

    def test_rejects_overlay_for_unknown_processor(self):
        with self.assertRaisesRegex(ValueError, "unknown exact processor"):
            validate_target_map_data(
                (),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-test", 0),),
                (AmdgpuTargetOverlayInfo("gfx-test-a0", "gfx-future"),),
                (),
            )

    def test_physical_revision_resolves_to_a_canonical_target(self):
        self.assertTrue(processor_has_physical_target_infos("gfx1250"))
        self.assertFalse(processor_has_physical_target_infos("gfx1100"))
        self.assertEqual(
            physical_target_info("gfx1250", 0),
            AmdgpuPhysicalTargetInfo("gfx1250", 0, "gfx1250-a0"),
        )
        self.assertEqual(
            physical_target_info("gfx1250", 1),
            AmdgpuPhysicalTargetInfo("gfx1250", 1, "gfx1250"),
        )
        self.assertIsNone(physical_target_info("gfx1250", 2))
        self.assertIsNone(physical_target_info("gfx1100", 4))

    def test_rejects_out_of_range_asic_revision(self):
        with self.assertRaisesRegex(ValueError, "outside the uint32 range"):
            validate_target_map_data(
                (),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-test", 0),),
                (),
                (AmdgpuPhysicalTargetInfo("gfx-test", 2**32, "gfx-test"),),
            )

    def test_rejects_noncanonical_physical_target_order(self):
        with self.assertRaisesRegex(ValueError, "canonical processor and ASIC"):
            validate_target_map_data(
                (),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-test", 0),),
                (),
                (
                    AmdgpuPhysicalTargetInfo("gfx-test", 1, "gfx-test"),
                    AmdgpuPhysicalTargetInfo("gfx-test", 0, "gfx-test"),
                ),
            )

    def test_multiple_revisioned_processors_are_data_only(self):
        exact_infos = (
            AmdgpuExactTargetInfo("gfx-test-a", "gfx-test-a", 0),
            AmdgpuExactTargetInfo("gfx-test-b", "gfx-test-b", 0),
        )
        overlays = (
            AmdgpuTargetOverlayInfo("gfx-test-a-a0", "gfx-test-a"),
            AmdgpuTargetOverlayInfo("gfx-test-b-c0", "gfx-test-b"),
        )
        physical_targets = (
            AmdgpuPhysicalTargetInfo("gfx-test-a", 0, "gfx-test-a-a0"),
            AmdgpuPhysicalTargetInfo("gfx-test-a", 1, "gfx-test-a"),
            AmdgpuPhysicalTargetInfo("gfx-test-b", 7, "gfx-test-b-c0"),
            AmdgpuPhysicalTargetInfo("gfx-test-b", 9, "gfx-test-b"),
        )

        validate_target_map_data((), exact_infos, overlays, physical_targets)

    def test_rejects_noncanonical_overlay_name(self):
        with self.assertRaisesRegex(ValueError, "canonical target coordinate"):
            validate_target_map_data(
                (),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-test", 0),),
                (AmdgpuTargetOverlayInfo("gfx-test A0", "gfx-test"),),
                (),
            )

    def test_rejects_physical_target_from_another_processor(self):
        with self.assertRaisesRegex(ValueError, "from another processor"):
            validate_target_map_data(
                (),
                (
                    AmdgpuExactTargetInfo("gfx-test-a", "gfx-test-a", 0),
                    AmdgpuExactTargetInfo("gfx-test-b", "gfx-test-b", 0),
                ),
                (AmdgpuTargetOverlayInfo("gfx-test-a-a0", "gfx-test-a"),),
                (
                    AmdgpuPhysicalTargetInfo("gfx-test-b", 0, "gfx-test-a-a0"),
                    AmdgpuPhysicalTargetInfo("gfx-test-b", 1, "gfx-test-b"),
                ),
            )

    def test_rejects_physical_mapping_without_canonical_base_revision(self):
        with self.assertRaisesRegex(ValueError, "no revision selecting"):
            validate_target_map_data(
                (),
                (AmdgpuExactTargetInfo("gfx-test", "gfx-test", 0),),
                (AmdgpuTargetOverlayInfo("gfx-test-a0", "gfx-test"),),
                (AmdgpuPhysicalTargetInfo("gfx-test", 0, "gfx-test-a0"),),
            )

    def test_canonical_target_overlays_validate(self):
        validate_target_map_data(
            AMDGPU_GENERIC_CODE_OBJECT_INFOS,
            AMDGPU_EXACT_TARGET_INFOS,
            AMDGPU_TARGET_OVERLAY_INFOS,
            AMDGPU_PHYSICAL_TARGET_INFOS,
        )


if __name__ == "__main__":
    unittest.main()
