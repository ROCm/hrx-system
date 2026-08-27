# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Immutable representation of Vulkan SPIR-V availability facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from build_tools.spirv.registry.dependency_expression import DependencyExpression
from build_tools.spirv.registry.registry_source import RegistryOrigin


@dataclass(frozen=True, order=True, slots=True)
class VulkanVersion:
    """One Vulkan registry major/minor version."""

    # Major API version.
    major: int
    # Minor API version.
    minor: int

    def __post_init__(self) -> None:
        if type(self.major) is not int or self.major <= 0:
            raise ValueError("Vulkan major version must be a positive integer")
        if type(self.minor) is not int or self.minor < 0:
            raise ValueError("Vulkan minor version must be a nonnegative integer")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


class VulkanApi(Enum):
    """API variants named by Vulkan availability metadata."""

    VULKAN = "vulkan"
    VULKAN_SC = "vulkansc"
    VULKAN_BASE = "vulkanbase"


class VulkanExtensionType(Enum):
    """Scope at which a Vulkan extension is enabled."""

    DEVICE = "device"
    INSTANCE = "instance"


class VulkanAvailabilityBlockKind(Enum):
    """Effect of one version or extension availability block."""

    REQUIRE = "require"
    REMOVE = "remove"
    DEPRECATE = "deprecate"


@dataclass(frozen=True, slots=True)
class VulkanStructureMember:
    """One named member retained from an availability-reachable structure."""

    # Vulkan member name used by profiles and dependency predicates.
    name: str
    # Equivalent feature/member path or extension availability name.
    availability_alias: str | None
    # Canonical source definition of this member.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanStructure:
    """One canonical availability-reachable Vulkan structure."""

    # Canonical Vulkan type name.
    name: str
    # Equivalent Vulkan type aliases targeting this definition.
    aliases: tuple[str, ...]
    # Ordered structure members retained for direct predicate resolution.
    members: tuple[VulkanStructureMember, ...]
    # Symbolic VkStructureType identity, absent for legacy unchained structures.
    structure_type: str | None
    # Canonical source definition of this structure.
    origin: RegistryOrigin
    # Direct member lookup within the canonical structure.
    members_by_name: Mapping[str, VulkanStructureMember] = field(
        repr=False, compare=False, hash=False
    )


@dataclass(frozen=True, slots=True)
class VulkanConstant:
    """One property-predicate constant and its integer value."""

    # Canonical Vulkan constant name.
    name: str
    # Equivalent names resolving to this constant.
    aliases: tuple[str, ...]
    # Integer value used by equality and bit predicates.
    value: int
    # Canonical source definition of this constant.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanFeatureReference:
    """One explicit structure/member feature reference in a block."""

    # Structure spelling used by the availability block.
    structure_name: str
    # Canonical structure name after Vulkan type-alias resolution.
    canonical_structure_name: str
    # Ordered feature-member names carried by this row.
    member_names: tuple[str, ...]
    # Canonical source definition of this feature reference.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanTypeReference:
    """One Vulkan type reference retained from an availability block."""

    # Referenced Vulkan type name.
    name: str
    # Replacement type named by a removal/deprecation row, when present.
    superseded_by: str | None
    # Canonical source definition of this type reference.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanAvailabilityBlock:
    """One conditional require, remove, or deprecate block."""

    # Effect this block has on its referenced declarations.
    kind: VulkanAvailabilityBlockKind
    # Explicit API restriction, or empty when the owner supplies it.
    apis: frozenset[VulkanApi]
    # Structured condition guarding this block, when present.
    dependencies: DependencyExpression | None
    # Ordered explicit feature structure/member references.
    feature_references: tuple[VulkanFeatureReference, ...]
    # Ordered Vulkan type references.
    type_references: tuple[VulkanTypeReference, ...]
    # Canonical source definition of this block.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanCoreFeature:
    """One Vulkan core or internal layered-version declaration."""

    # Registry feature name used by dependency expressions.
    name: str
    # Major/minor version declared by the feature.
    version: VulkanVersion
    # API variants to which this feature applies.
    apis: frozenset[VulkanApi]
    # True for the registry's base, compute, and graphics layering features.
    internal: bool
    # Structured dependencies of this feature, when present.
    dependencies: DependencyExpression | None
    # Ordered availability blocks owned by this feature.
    blocks: tuple[VulkanAvailabilityBlock, ...]
    # Canonical source definition of this feature.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanExtension:
    """One Vulkan extension and its availability relationships."""

    # Canonical extension name.
    name: str
    # Stable Vulkan extension number.
    number: int
    # Instance/device enablement scope, when one is declared.
    extension_type: VulkanExtensionType | None
    # Supported API variants, or empty for a disabled reservation.
    supported_apis: frozenset[VulkanApi]
    # Structured dependencies of this extension, when present.
    dependencies: DependencyExpression | None
    # Version or extension that promoted this extension, when present.
    promoted_to: str | None
    # Replacement for a deprecated extension; an empty string means none.
    deprecated_by: str | None
    # Extension that made this extension obsolete, when present.
    obsoleted_by: str | None
    # Platform guard name, when this extension is platform-specific.
    platform: str | None
    # True when the extension is provisional rather than ratified.
    provisional: bool
    # Ordered availability blocks owned by this extension.
    blocks: tuple[VulkanAvailabilityBlock, ...]
    # Canonical source definition of this extension.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanSpirvCoreEnable:
    """A SPIR-V entry enabled by a Vulkan core feature."""

    # Referenced Vulkan core-feature name.
    core_feature_name: str
    # Canonical source definition of this alternative.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanSpirvExtensionEnable:
    """A SPIR-V entry enabled by a Vulkan extension."""

    # Referenced Vulkan extension name.
    extension_name: str
    # Canonical source definition of this alternative.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanSpirvFeatureEnable:
    """A SPIR-V entry enabled by a Boolean Vulkan feature member."""

    # Structure spelling used by the enable row.
    structure_name: str
    # Canonical structure name after Vulkan type-alias resolution.
    canonical_structure_name: str
    # Boolean feature member that must be enabled.
    member_name: str
    # Alternative profile-member spelling, when declared by the registry.
    profile_member_alias: str | None
    # Structured version/extension prerequisites for the member.
    requirements: DependencyExpression
    # Canonical source definition of this alternative.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanSpirvPropertyEnable:
    """A SPIR-V entry enabled by a Vulkan property value or bit."""

    # Property structure spelling used by the enable row.
    structure_name: str
    # Canonical structure name after Vulkan type-alias resolution.
    canonical_structure_name: str
    # Property member tested by this alternative.
    member_name: str
    # Constant spelling used by the enable row.
    constant_name: str
    # Canonical constant name after alias resolution.
    canonical_constant_name: str
    # Integer equality value or bit mask represented by the constant.
    expected_value: int
    # Structured version/extension prerequisites for the property.
    requirements: DependencyExpression
    # Canonical source definition of this alternative.
    origin: RegistryOrigin


VulkanSpirvEnable: TypeAlias = (
    VulkanSpirvCoreEnable
    | VulkanSpirvExtensionEnable
    | VulkanSpirvFeatureEnable
    | VulkanSpirvPropertyEnable
)


@dataclass(frozen=True, slots=True)
class VulkanSpirvAvailability:
    """One SPIR-V extension or capability with ordered enable alternatives."""

    # SPIR-V extension or capability name.
    name: str
    # Ordered alternatives, any one of which makes the entry available.
    enables: tuple[VulkanSpirvEnable, ...]
    # Canonical source definition of this entry.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class VulkanSpirvAvailabilityRegistry:
    """Complete consumer-closed Vulkan SPIR-V availability model."""

    # Root identity of the canonical source document.
    source_origin: RegistryOrigin
    # Core and internal layered features in canonical source order.
    core_features: tuple[VulkanCoreFeature, ...]
    # Vulkan extensions in canonical source order.
    extensions: tuple[VulkanExtension, ...]
    # Canonical structures reached by availability metadata.
    structures: tuple[VulkanStructure, ...]
    # Canonical constants reached by property-predicate alternatives.
    constants: tuple[VulkanConstant, ...]
    # SPIR-V extension mappings in canonical source order.
    spirv_extensions: tuple[VulkanSpirvAvailability, ...]
    # SPIR-V capability mappings in canonical source order.
    spirv_capabilities: tuple[VulkanSpirvAvailability, ...]
    # Direct lookup by Vulkan core-feature name.
    core_features_by_name: Mapping[str, VulkanCoreFeature] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by Vulkan extension name.
    extensions_by_name: Mapping[str, VulkanExtension] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by Vulkan extension number.
    extensions_by_number: Mapping[int, VulkanExtension] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by every retained canonical and alias structure name.
    structures_by_name: Mapping[str, VulkanStructure] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by ``Structure::member`` for all retained type aliases.
    structure_members_by_name: Mapping[str, VulkanStructureMember] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by every retained canonical and alias constant name.
    constants_by_name: Mapping[str, VulkanConstant] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by SPIR-V extension name.
    spirv_extensions_by_name: Mapping[str, VulkanSpirvAvailability] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by SPIR-V capability name.
    spirv_capabilities_by_name: Mapping[str, VulkanSpirvAvailability] = field(
        repr=False, compare=False, hash=False
    )


__all__ = [
    "VulkanApi",
    "VulkanAvailabilityBlock",
    "VulkanAvailabilityBlockKind",
    "VulkanConstant",
    "VulkanCoreFeature",
    "VulkanExtension",
    "VulkanExtensionType",
    "VulkanFeatureReference",
    "VulkanSpirvAvailability",
    "VulkanSpirvAvailabilityRegistry",
    "VulkanSpirvCoreEnable",
    "VulkanSpirvEnable",
    "VulkanSpirvExtensionEnable",
    "VulkanSpirvFeatureEnable",
    "VulkanSpirvPropertyEnable",
    "VulkanStructure",
    "VulkanStructureMember",
    "VulkanTypeReference",
    "VulkanVersion",
]
