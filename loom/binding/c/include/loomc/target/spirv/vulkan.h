// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_VULKAN_H_
#define LOOMC_TARGET_SPIRV_VULKAN_H_

#include "loomc/target/spirv/profile.h"
#include "vulkan/vulkan.h"

/// @file
/// Raw Vulkan SPIR-V profile adapter.
///
/// This optional leaf derives normalized SPIR-V profile facts from a
/// caller-owned `VkPhysicalDevice` and a small table of Vulkan query functions.
/// It includes Vulkan headers but does not load or link a Vulkan loader.
/// Embedders that already own an instance/device stack can pass their resolved
/// function pointers directly, while offline and cross-compilation flows can
/// keep using header-light profile facts or saved vulkaninfo importers.
///
/// The adapter is a convenience layer over `loomc_target_profile_create_spirv`.
/// It does not expose a second target model: queried Vulkan properties and
/// features are normalized into the same feature, limit, and environment fact
/// rows used by saved profiles and programmatic profile construction.
///
/// @par Example
/// Create a profile from a live Vulkan physical device:
///
/// @code{.c}
/// loomc_spirv_vulkan_function_table_t functions = {
///     .type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE,
///     .structure_size = sizeof(loomc_spirv_vulkan_function_table_t),
///     .get_physical_device_properties2 =
///         resolved_vkGetPhysicalDeviceProperties2,
///     .get_physical_device_features2 = resolved_vkGetPhysicalDeviceFeatures2,
///     .enumerate_device_extension_properties =
///         resolved_vkEnumerateDeviceExtensionProperties,
/// };
/// loomc_spirv_vulkan_profile_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_PROFILE_OPTIONS,
///     .structure_size = sizeof(loomc_spirv_vulkan_profile_options_t),
///     .identifier = loomc_make_cstring_view("live-vulkan-device"),
///     .physical_device = physical_device,
///     .functions = &functions,
/// };
/// loomc_target_profile_t* profile = NULL;
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_target_profile_create_spirv_vulkan(
///     target_environment, &options, loomc_allocator_system(), &profile,
///     &result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(result)) {
///   // Inspect result diagnostics; profile is NULL on profile failure.
/// } else {
///   // Use profile for target selection/refinement.
///   loomc_target_profile_release(profile);
/// }
/// loomc_result_release(result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Vulkan query functions used by the raw Vulkan SPIR-V profile adapter.
typedef struct loomc_spirv_vulkan_function_table_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Vulkan physical-device properties query function.
  PFN_vkGetPhysicalDeviceProperties2 get_physical_device_properties2;

  /// Vulkan physical-device features query function.
  PFN_vkGetPhysicalDeviceFeatures2 get_physical_device_features2;

  /// Optional Vulkan physical-device extension enumeration function.
  ///
  /// When provided, extension feature structs are queried only when the
  /// physical device reports the corresponding extension. When omitted, the
  /// adapter queries only core-promoted Vulkan feature structs implied by the
  /// physical device API version.
  PFN_vkEnumerateDeviceExtensionProperties
      enumerate_device_extension_properties;
} loomc_spirv_vulkan_function_table_t;

/// Raw Vulkan profile creation options.
typedef struct loomc_spirv_vulkan_profile_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_PROFILE_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Stable profile identifier used in diagnostics.
  loomc_string_view_t identifier;

  /// Vulkan physical device to query.
  VkPhysicalDevice physical_device;

  /// Borrowed Vulkan function table.
  const loomc_spirv_vulkan_function_table_t* functions;
} loomc_spirv_vulkan_profile_options_t;

/// Creates a SPIR-V target profile from a raw Vulkan physical device.
///
/// @param target_environment SPIR-V-capable target environment.
/// @param options Vulkan physical device and function table. Required.
/// @param allocator Host allocator used for profile and result storage.
/// @param out_profile Receives one retained profile when `out_result`
/// succeeds.
/// @param out_result Receives one retained result describing profile
/// preparation.
/// @return OK when the Vulkan query and profile preparation completed far
/// enough to report a result. Non-OK statuses represent API misuse or
/// infrastructure failures before a result could be produced.
///
/// The adapter queries core physical-device properties and feature structs,
/// plus extension structs exposed by the Vulkan headers used to build this
/// leaf. Unsupported or header-absent facts remain unknown. Recognized queried
/// facts use `vulkan:` provenance strings and are then validated by the shared
/// SPIR-V profile layer.
///
/// @ownership
/// The caller owns `out_profile` when the returned result succeeds and releases
/// it with `loomc_target_profile_release`. The caller always owns `out_result`
/// on an OK return and releases it with `loomc_result_release`.
///
/// @thread_safety
/// The adapter does not retain the Vulkan physical device or function table.
/// Concurrent calls are safe when the caller's Vulkan implementation allows the
/// supplied query functions to run concurrently for the supplied physical
/// device.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_spirv_vulkan(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_vulkan_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_SPIRV_VULKAN_H_
