# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import cast

from build_tools.spirv.registry.dependency_expression import (
    DependencyExpression,
    format_dependency_expression,
)
from build_tools.spirv.registry.vulkan_spirv_availability import (
    VulkanApi,
    VulkanAvailabilityBlockKind,
    VulkanSpirvCoreEnable,
    VulkanSpirvExtensionEnable,
    VulkanSpirvFeatureEnable,
    VulkanSpirvPropertyEnable,
    VulkanVersion,
    load_vulkan_spirv_availability,
)

_REGISTRY_PATH: Path | None = None


class VulkanSpirvAvailabilityRegistryTest(unittest.TestCase):
    def test_imports_complete_pinned_availability_surface(self):
        self.assertIsNotNone(_REGISTRY_PATH)
        registry_path = cast(Path, _REGISTRY_PATH)
        registry = load_vulkan_spirv_availability(registry_path)

        self.assertEqual(len(registry.core_features), 21)
        self.assertEqual(len(registry.core_features_by_name), 21)
        self.assertEqual(
            registry.core_features_by_name["VK_VERSION_1_4"].version,
            VulkanVersion(1, 4),
        )
        self.assertEqual(
            registry.core_features_by_name["VKSC_VERSION_1_0"].apis,
            {VulkanApi.VULKAN_SC},
        )

        self.assertEqual(len(registry.extensions), 697)
        self.assertEqual(len(registry.extensions_by_name), 697)
        self.assertEqual(len(registry.extensions_by_number), 697)
        self.assertEqual(
            sum(not extension.supported_apis for extension in registry.extensions),
            220,
        )
        self.assertEqual(
            sum(extension.provisional for extension in registry.extensions), 5
        )

        blocks = tuple(
            block
            for owner in (*registry.core_features, *registry.extensions)
            for block in owner.blocks
        )
        self.assertEqual(
            Counter(block.kind for block in blocks),
            {
                VulkanAvailabilityBlockKind.REQUIRE: 1129,
                VulkanAvailabilityBlockKind.REMOVE: 3,
                VulkanAvailabilityBlockKind.DEPRECATE: 23,
            },
        )
        feature_references = tuple(
            feature for block in blocks for feature in block.feature_references
        )
        self.assertEqual(len(feature_references), 454)
        self.assertEqual(
            sum(len(feature.member_names) for feature in feature_references), 510
        )
        self.assertEqual(sum(len(block.type_references) for block in blocks), 2810)
        self.assertEqual(
            sum(
                owner.dependencies is not None
                for owner in (*registry.core_features, *registry.extensions)
            ),
            412,
        )
        self.assertEqual(sum(block.dependencies is not None for block in blocks), 267)

        self.assertEqual(len(registry.structures), 286)
        self.assertEqual(len(registry.structures_by_name), 358)
        self.assertEqual(
            sum(len(structure.members) for structure in registry.structures), 1347
        )
        self.assertEqual(len(registry.structure_members_by_name), 1671)
        self.assertEqual(
            sum(
                member.availability_alias is not None
                for structure in registry.structures
                for member in structure.members
            ),
            159,
        )
        self.assertEqual(
            sum(
                structure.structure_type is not None
                for structure in registry.structures
            ),
            285,
        )
        self.assertEqual(len(registry.constants), 10)
        self.assertEqual(len(registry.constants_by_name), 11)

        self.assertEqual(len(registry.spirv_extensions), 113)
        self.assertEqual(len(registry.spirv_extensions_by_name), 113)
        self.assertEqual(len(registry.spirv_capabilities), 200)
        self.assertEqual(len(registry.spirv_capabilities_by_name), 200)
        enables = tuple(
            enable
            for availability in (
                *registry.spirv_extensions,
                *registry.spirv_capabilities,
            )
            for enable in availability.enables
        )
        self.assertEqual(
            Counter(type(enable) for enable in enables),
            {
                VulkanSpirvCoreEnable: 33,
                VulkanSpirvExtensionEnable: 142,
                VulkanSpirvFeatureEnable: 194,
                VulkanSpirvPropertyEnable: 24,
            },
        )
        self.assertEqual(
            sum(
                isinstance(
                    enable, (VulkanSpirvFeatureEnable, VulkanSpirvPropertyEnable)
                )
                for enable in enables
            ),
            218,
        )

        bfloat16_extension = registry.extensions_by_name["VK_KHR_shader_bfloat16"]
        self.assertEqual(
            format_dependency_expression(
                cast(DependencyExpression, bfloat16_extension.dependencies)
            ),
            "VK_KHR_get_physical_device_properties2,VK_VERSION_1_1",
        )
        self.assertIsInstance(
            registry.spirv_extensions_by_name["SPV_KHR_bfloat16"].enables[0],
            VulkanSpirvExtensionEnable,
        )
        self.assertEqual(
            len(
                registry.spirv_capabilities_by_name[
                    "BFloat16CooperativeMatrixKHR"
                ].enables
            ),
            1,
        )

        subgroup = registry.spirv_capabilities_by_name[
            "GroupNonUniformArithmetic"
        ].enables[0]
        self.assertIsInstance(subgroup, VulkanSpirvPropertyEnable)
        self.assertEqual(subgroup.expected_value, 1 << 2)
        self.assertEqual(
            subgroup.canonical_constant_name,
            "VK_SUBGROUP_FEATURE_ARITHMETIC_BIT",
        )
        partitioned = registry.constants_by_name[
            "VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV"
        ]
        self.assertIs(
            registry.constants_by_name["VK_SUBGROUP_FEATURE_PARTITIONED_BIT_EXT"],
            partitioned,
        )
        self.assertEqual(partitioned.value, 1 << 8)

        physical_storage = registry.spirv_capabilities_by_name[
            "PhysicalStorageBufferAddresses"
        ].enables[1]
        self.assertIsInstance(physical_storage, VulkanSpirvFeatureEnable)
        self.assertEqual(
            physical_storage.profile_member_alias, "bufferDeviceAddressEXT"
        )
        vulkan11_properties = registry.structures_by_name[
            "VkPhysicalDeviceVulkan11Properties"
        ]
        self.assertEqual(
            vulkan11_properties.members_by_name[
                "subgroupSupportedOperations"
            ].availability_alias,
            "VkPhysicalDeviceSubgroupProperties::supportedOperations",
        )
        self.assertIn("VkPhysicalDeviceSubgroupProperties", registry.structures_by_name)
        self.assertIn(
            "VkPhysicalDeviceSubgroupProperties::supportedOperations",
            registry.structure_members_by_name,
        )


def _registry_path_from_argv(argv: list[str]) -> Path:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <vk.xml>")
    registry_path = Path(argv[1])
    if not registry_path.is_file():
        raise SystemExit(f"Vulkan registry does not exist: {registry_path}")
    return registry_path


if __name__ == "__main__":
    _REGISTRY_PATH = _registry_path_from_argv(sys.argv)
    unittest.main(argv=[sys.argv[0]])
