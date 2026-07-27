// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_VULKANINFO_H_
#define LOOMC_TARGET_SPIRV_VULKANINFO_H_

#include "loomc/target/spirv/profile.h"

/// @file
/// Saved vulkaninfo profile importer for SPIR-V targets.
///
/// This leaf translates saved Vulkan profile data into normalized SPIR-V
/// profile facts. It does not include Vulkan headers, load a Vulkan loader, or
/// require hardware. The importer accepts immutable `loomc_source_t` handles so
/// embedders can choose borrowed, copied, memory-mapped, cached, or
/// binding-owned storage using the ordinary source API.
/// Filesystem input should be loaded with `loomc_source_create_from_path` and
/// then passed to this importer.
///
/// The imported facts are intentionally the same public fact rows accepted by
/// `loomc_target_profile_create_spirv`. The importer is a convenience adapter,
/// not a separate target model. Unknown JSON fields are ignored. Recognized
/// malformed fields produce `SPIRV/VULKANINFO` diagnostics on the returned
/// result. Contradictory or unsupported normalized facts are then diagnosed by
/// the SPIR-V profile layer using `SPIRV/PROFILE` diagnostics.
///
/// @par Example
/// Import a saved profile from caller-owned bytes:
///
/// @code{.c}
/// loomc_source_options_t source_options = {
///     .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
///     .structure_size = sizeof(loomc_source_options_t),
///     .format = LOOMC_SOURCE_FORMAT_UNKNOWN,
///     .identifier = loomc_make_cstring_view("vulkaninfo.json"),
///     .contents = loomc_make_byte_span(json_data, json_data_length),
///     .storage = LOOMC_SOURCE_STORAGE_BORROWED,
/// };
/// loomc_source_t* source = NULL;
/// loomc_status_t status = loomc_source_create(
///     &source_options, loomc_allocator_system(), &source);
/// if (!loomc_status_is_ok(status)) return status;
///
/// loomc_spirv_vulkaninfo_profile_options_t import_options = {
///     .type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKANINFO_PROFILE_OPTIONS,
///     .structure_size = sizeof(loomc_spirv_vulkaninfo_profile_options_t),
///     .identifier = loomc_make_cstring_view("offline-vulkan-profile"),
/// };
/// loomc_target_profile_t* profile = NULL;
/// loomc_result_t* result = NULL;
/// status = loomc_target_profile_create_spirv_vulkaninfo(
///     target_environment, source, &import_options, loomc_allocator_system(),
///     &profile, &result);
/// loomc_source_release(source);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(result)) {
///   // Inspect result diagnostics; profile is NULL on import failure.
/// }
/// loomc_result_release(result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Saved vulkaninfo profile import options.
typedef struct loomc_spirv_vulkaninfo_profile_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_SPIRV_VULKANINFO_PROFILE_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Stable profile identifier. Empty uses the source identifier.
  loomc_string_view_t identifier;

  /// Optional profile key for GPUInfo/Vulkan Profiles wrapper JSON.
  ///
  /// Empty selects the first profile entry when a wrapper contains `profiles`.
  /// Raw vulkaninfo-style `devices` JSON ignores profile selection and uses
  /// `device_index`.
  loomc_string_view_t profile_name;

  /// Device index for raw vulkaninfo-style `devices` arrays.
  ///
  /// Ignored when importing GPUInfo/Vulkan Profiles wrapper JSON that selects
  /// a named capability object through `profiles`.
  uint32_t device_index;
} loomc_spirv_vulkaninfo_profile_options_t;

/// Creates a SPIR-V target profile from saved vulkaninfo JSON.
///
/// @param target_environment SPIR-V-capable target environment.
/// @param source Immutable JSON/JSONC source bytes.
/// @param options Import options, or `NULL` for defaults.
/// @param allocator Host allocator used for profile and result storage.
/// @param out_profile Receives one retained profile when `out_result`
/// succeeds.
/// @param out_result Receives one retained result describing import and profile
/// preparation.
/// @return OK when import ran far enough to report a result. Non-OK statuses
/// represent API misuse or infrastructure failures before a result could be
/// produced.
///
/// Supported input shapes are the common raw vulkaninfo shape with a top-level
/// `devices` array and the GPUInfo/Vulkan Profiles wrapper shape with
/// top-level `capabilities` and `profiles` objects. The importer recognizes
/// currently modeled Loom facts only; unrecognized Vulkan fields are ignored so
/// newer saved profiles remain loadable by older Loom builds.
///
/// @ownership
/// The caller owns `out_profile` when the returned result succeeds and releases
/// it with `loomc_target_profile_release`. The caller always owns `out_result`
/// on an OK return and releases it with `loomc_result_release`.
///
/// @lifetime
/// The returned profile and result do not borrow from `source`. Diagnostics
/// retain `source` when they reference imported JSON.
///
/// @thread_safety
/// The importer only reads `source` and `target_environment`; concurrent
/// imports using shared immutable handles are safe.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_spirv_vulkaninfo(
    loomc_target_environment_t* target_environment,
    const loomc_source_t* source,
    const loomc_spirv_vulkaninfo_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_SPIRV_VULKANINFO_H_
