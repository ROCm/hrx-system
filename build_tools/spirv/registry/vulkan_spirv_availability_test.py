# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from typing import cast

from build_tools.spirv.registry.dependency_expression import (
    DependencyAnyOf,
    DependencyExpression,
    format_dependency_expression,
)
from build_tools.spirv.registry.registry_source import RegistrySourceError
from build_tools.spirv.registry.vulkan_spirv_availability import (
    VulkanApi,
    VulkanAvailabilityBlockKind,
    VulkanExtensionType,
    VulkanSpirvCoreEnable,
    VulkanSpirvExtensionEnable,
    VulkanSpirvFeatureEnable,
    VulkanSpirvPropertyEnable,
    VulkanVersion,
    parse_vulkan_spirv_availability,
)


def _minimal_registry() -> str:
    return """\
<registry>
  <comment>Test-only registry.</comment>
  <types comment="Vulkan type definitions">
    <type name="VkUnused" />
    <type category="struct" name="VkPhysicalDeviceTestFeaturesEXT"
          structextends="VkPhysicalDeviceFeatures2,VkDeviceCreateInfo">
      <member values="VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEST_FEATURES_EXT"><type>VkStructureType</type> <name>sType</name></member>
      <member optional="true"><type>void</type>* <name>pNext</name></member>
      <member><type>VkBool32</type> <name alias="VK_EXT_test">testFeature</name></member>
    </type>
    <type category="struct" name="VkPhysicalDeviceTestFeaturesKHR"
          alias="VkPhysicalDeviceTestFeaturesEXT" />
    <type category="struct" name="VkPhysicalDeviceTestProperties"
          returnedonly="true" structextends="VkPhysicalDeviceProperties2">
      <member values="VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEST_PROPERTIES"><type>VkStructureType</type> <name>sType</name></member>
      <member optional="true"><type>void</type>* <name>pNext</name></member>
      <member><type>VkFlags</type> <name>testFlags</name></member>
    </type>
  </types>
  <enums name="VkTestFlagBits" type="bitmask">
    <enum bitpos="2" name="VK_TEST_BIT" comment="Test bit" />
  </enums>
  <commands><future-command-schema /></commands>
  <feature api="vulkan" name="VK_VERSION_1_0" number="1.0"
           comment="Vulkan 1.0">
    <require comment="Test feature declarations">
      <type name="VkUnused" />
      <feature struct="VkPhysicalDeviceTestFeaturesKHR" name="testFeature" />
      <enum name="VK_TEST_BIT" />
    </require>
  </feature>
  <extensions comment="Vulkan extension interface definitions">
    <extension name="VK_EXT_test" number="1" type="device"
               supported="vulkan" depends="VK_VERSION_1_0"
               provisional="true">
      <require depends="VK_VERSION_1_0">
        <enum extends="VkTestFlagBits" name="VK_TEST_BIT_ALIAS"
              alias="VK_TEST_BIT" />
        <feature struct="VkPhysicalDeviceTestFeaturesEXT"
                 name="testFeature" />
      </require>
    </extension>
  </extensions>
  <spirvextensions comment="SPIR-V extensions allowed in Vulkan">
    <spirvextension name="SPV_TEST_extension">
      <enable extension="VK_EXT_test" />
    </spirvextension>
  </spirvextensions>
  <spirvcapabilities comment="SPIR-V capabilities allowed in Vulkan">
    <spirvcapability name="TestCapability">
      <enable version="VK_VERSION_1_0" />
      <enable struct="VkPhysicalDeviceTestFeaturesKHR" feature="testFeature"
              alias="testFeatureEXT"
              requires="VK_VERSION_1_0,VK_EXT_test" />
    </spirvcapability>
    <spirvcapability name="TestProperty">
      <enable property="VkPhysicalDeviceTestProperties" member="testFlags"
              value="VK_TEST_BIT_ALIAS" requires="VK_EXT_test" />
    </spirvcapability>
  </spirvcapabilities>
</registry>
"""


class VulkanSpirvAvailabilityTest(unittest.TestCase):
    def test_builds_consumer_closed_model_and_immutable_indexes(self):
        registry = parse_vulkan_spirv_availability(
            _minimal_registry(), source="test_vk.xml"
        )

        self.assertEqual(len(registry.core_features), 1)
        core = registry.core_features_by_name["VK_VERSION_1_0"]
        self.assertEqual(core.version, VulkanVersion(1, 0))
        self.assertEqual(core.apis, {VulkanApi.VULKAN})
        self.assertEqual(core.blocks[0].kind, VulkanAvailabilityBlockKind.REQUIRE)
        self.assertEqual(
            core.blocks[0].feature_references[0].canonical_structure_name,
            "VkPhysicalDeviceTestFeaturesEXT",
        )

        extension = registry.extensions_by_name["VK_EXT_test"]
        self.assertIs(registry.extensions_by_number[1], extension)
        self.assertEqual(extension.extension_type, VulkanExtensionType.DEVICE)
        self.assertEqual(extension.supported_apis, {VulkanApi.VULKAN})
        self.assertTrue(extension.provisional)
        self.assertEqual(
            format_dependency_expression(
                cast(DependencyExpression, extension.dependencies)
            ),
            "VK_VERSION_1_0",
        )

        structure = registry.structures_by_name["VkPhysicalDeviceTestFeaturesEXT"]
        self.assertIs(
            registry.structures_by_name["VkPhysicalDeviceTestFeaturesKHR"],
            structure,
        )
        self.assertEqual(
            structure.structure_type,
            "VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEST_FEATURES_EXT",
        )
        self.assertEqual(
            structure.members_by_name["testFeature"].availability_alias,
            "VK_EXT_test",
        )
        self.assertIs(
            registry.structure_members_by_name[
                "VkPhysicalDeviceTestFeaturesKHR::testFeature"
            ],
            structure.members_by_name["testFeature"],
        )

        constant = registry.constants_by_name["VK_TEST_BIT"]
        self.assertIs(registry.constants_by_name["VK_TEST_BIT_ALIAS"], constant)
        self.assertEqual(constant.value, 4)

        spirv_extension = registry.spirv_extensions_by_name["SPV_TEST_extension"]
        self.assertIsInstance(spirv_extension.enables[0], VulkanSpirvExtensionEnable)
        capability = registry.spirv_capabilities_by_name["TestCapability"]
        self.assertIsInstance(capability.enables[0], VulkanSpirvCoreEnable)
        feature_enable = capability.enables[1]
        self.assertIsInstance(feature_enable, VulkanSpirvFeatureEnable)
        self.assertEqual(feature_enable.profile_member_alias, "testFeatureEXT")
        self.assertIsInstance(feature_enable.requirements, DependencyAnyOf)
        property_enable = registry.spirv_capabilities_by_name["TestProperty"].enables[0]
        self.assertIsInstance(property_enable, VulkanSpirvPropertyEnable)
        self.assertEqual(property_enable.canonical_constant_name, "VK_TEST_BIT")
        self.assertEqual(property_enable.expected_value, 4)

        with self.assertRaises(TypeError):
            registry.extensions_by_name["VK_EXT_future"] = extension  # type: ignore[index]
        with self.assertRaises(TypeError):
            structure.members_by_name["future"] = structure.members[0]  # type: ignore[index]

    def test_rejects_xml_syntax_doctype_and_top_level_schema_drift(self):
        with self.assertRaisesRegex(
            RegistrySourceError, r"broken\.xml:line 1, column 11"
        ):
            parse_vulkan_spirv_availability("<registry>", source="broken.xml")

        with self.assertRaisesRegex(RegistrySourceError, "document type declaration"):
            parse_vulkan_spirv_availability(
                "<!DOCTYPE registry><registry />", source="doctype.xml"
            )

        unknown_section = _minimal_registry().replace(
            "<commands><future-command-schema /></commands>",
            "<future><entry /></future>",
        )
        with self.assertRaisesRegex(
            RegistrySourceError, "unknown top-level section 'future'"
        ):
            parse_vulkan_spirv_availability(
                unknown_section, source="unknown_section.xml"
            )

    def test_rejects_unknown_modeled_attributes_and_enable_forms(self):
        unknown_extension_attribute = _minimal_registry().replace(
            'supported="vulkan" depends=',
            'supported="vulkan" future="true" depends=',
        )
        with self.assertRaisesRegex(RegistrySourceError, "unknown attributes: future"):
            parse_vulkan_spirv_availability(
                unknown_extension_attribute, source="unknown_attribute.xml"
            )

        unknown_enum_attribute = _minimal_registry().replace(
            '<enum name="VK_TEST_BIT" />',
            '<enum name="VK_TEST_BIT" future="true" />',
        )
        with self.assertRaisesRegex(
            RegistrySourceError, "unknown enum attributes: future"
        ):
            parse_vulkan_spirv_availability(
                unknown_enum_attribute, source="unknown_enum_attribute.xml"
            )

        unsupported_enable = _minimal_registry().replace(
            '<enable extension="VK_EXT_test" />',
            '<enable extension="VK_EXT_test" version="VK_VERSION_1_0" />',
        )
        with self.assertRaisesRegex(
            RegistrySourceError,
            "unsupported SPIR-V enable attributes: extension, version",
        ):
            parse_vulkan_spirv_availability(
                unsupported_enable, source="unsupported_enable.xml"
            )

    def test_rejects_dangling_availability_references(self):
        cases = (
            (
                _minimal_registry().replace(
                    'depends="VK_VERSION_1_0"\n               provisional=',
                    'depends="VK_VERSION_FUTURE"\n               provisional=',
                ),
                "unknown Vulkan dependency 'VK_VERSION_FUTURE'",
            ),
            (
                _minimal_registry().replace(
                    'struct="VkPhysicalDeviceTestFeaturesKHR" name="testFeature"',
                    'struct="VkPhysicalDeviceFutureFeatures" name="testFeature"',
                    1,
                ),
                "unknown Vulkan structure 'VkPhysicalDeviceFutureFeatures'",
            ),
            (
                _minimal_registry().replace(
                    'struct="VkPhysicalDeviceTestFeaturesKHR" feature="testFeature"',
                    'struct="VkPhysicalDeviceTestFeaturesKHR" feature="futureFeature"',
                ),
                "unknown member VkPhysicalDeviceTestFeaturesKHR::futureFeature",
            ),
            (
                _minimal_registry().replace(
                    'value="VK_TEST_BIT_ALIAS"', 'value="VK_FUTURE_BIT"'
                ),
                "Vulkan constant 'VK_FUTURE_BIT' has no definition",
            ),
            (
                _minimal_registry().replace(
                    'alias="VK_EXT_test">testFeature',
                    'alias="VK_EXT_future">testFeature',
                ),
                "unknown Vulkan availability alias 'VK_EXT_future'",
            ),
        )
        for contents, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RegistrySourceError, message):
                    parse_vulkan_spirv_availability(
                        contents, source="dangling_reference.xml"
                    )

    def test_scopes_alias_validation_and_rejects_reachable_cycles(self):
        type_alias_cycle = _minimal_registry().replace(
            "  </types>",
            """\
    <type category="struct" name="VkCycleA" alias="VkCycleB" />
    <type category="struct" name="VkCycleB" alias="VkCycleA" />
  </types>""",
        )
        registry = parse_vulkan_spirv_availability(
            type_alias_cycle, source="unrelated_type_alias_cycle.xml"
        )
        self.assertNotIn("VkCycleA", registry.structures_by_name)

        requested_type_alias_cycle = type_alias_cycle.replace(
            'struct="VkPhysicalDeviceTestFeaturesKHR" name="testFeature"',
            'struct="VkCycleA" name="testFeature"',
            1,
        )
        with self.assertRaisesRegex(RegistrySourceError, "type alias cycle"):
            parse_vulkan_spirv_availability(
                requested_type_alias_cycle, source="requested_type_alias_cycle.xml"
            )

        constant_alias_cycle = _minimal_registry().replace(
            '<enum bitpos="2" name="VK_TEST_BIT" comment="Test bit" />',
            '<enum name="VK_TEST_BIT" alias="VK_TEST_BIT_ALIAS" />',
        )
        with self.assertRaisesRegex(
            RegistrySourceError, "property constant alias cycle"
        ):
            parse_vulkan_spirv_availability(
                constant_alias_cycle, source="constant_alias_cycle.xml"
            )

    def test_rejects_duplicate_spirv_names(self):
        duplicate_spirv_name = _minimal_registry().replace(
            "  </spirvextensions>",
            """\
    <spirvextension name="SPV_TEST_extension">
      <enable extension="VK_EXT_test" />
    </spirvextension>
  </spirvextensions>""",
        )
        with self.assertRaisesRegex(
            RegistrySourceError, "SPIR-V name 'SPV_TEST_extension' collides"
        ):
            parse_vulkan_spirv_availability(
                duplicate_spirv_name, source="duplicate_spirv_name.xml"
            )


if __name__ == "__main__":
    unittest.main()
