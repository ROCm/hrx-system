// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_AMDGPU_PROFILE_H_
#define LOOMC_TARGET_AMDGPU_PROFILE_H_

#include "loomc/target/amdgpu/base.h"

/// @file
/// AMDGPU target profile facts.
///
/// The AMDGPU public profile selects one exact or generic AMDGPU processor row
/// such as `gfx1151`, `gfx942`, or `gfx11-generic`, plus any silicon identity
/// required to compile for that row. The profile determines the
/// descriptor-family target bundle, native HSACO support, default wavefront
/// size, HSA target id, and target-record family used by compilation and
/// emission. Live HSA/HIP/HRX adapters should derive an exact processor and
/// silicon identity from their runtime agent query and then create a normal
/// `loomc_target_profile_t`.

#ifdef __cplusplus
extern "C" {
#endif

/// gfx1250 silicon revision selected by an AMDGPU target profile.
typedef enum loomc_amdgpu_gfx1250_revision_e {
  /// Uses the default revision for the selected processor. gfx1250 defaults to
  /// B0; the value has no effect on other processors.
  LOOMC_AMDGPU_GFX1250_REVISION_DEFAULT = 0,

  /// Selects gfx1250 A0 silicon behavior and errata.
  LOOMC_AMDGPU_GFX1250_REVISION_A0 = 1,

  /// Selects gfx1250 B0 silicon behavior.
  LOOMC_AMDGPU_GFX1250_REVISION_B0 = 2,
} loomc_amdgpu_gfx1250_revision_t;

/// AMDGPU processor profile creation options.
typedef struct loomc_amdgpu_profile_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Stable profile identifier used in diagnostics.
  ///
  /// Empty uses `processor`.
  loomc_string_view_t identifier;

  /// Exact or generic AMDGPU processor name, such as `gfx1151`, `gfx942`, or
  /// `gfx11-generic`.
  loomc_string_view_t processor;

  /// gfx1250 silicon revision. Explicit A0/B0 values require processor
  /// `gfx1250`; DEFAULT selects B0 for gfx1250 and is ignored otherwise.
  loomc_amdgpu_gfx1250_revision_t gfx1250_revision;
} loomc_amdgpu_profile_options_t;

/// Creates an AMDGPU target profile from an exact or generic processor name.
///
/// @param target_environment AMDGPU-capable target environment.
/// @param options Profile options. Required.
/// @param allocator Host allocator used for profile storage.
/// @param out_profile Receives one retained profile on success.
/// @return OK when the processor is known, HSACO-capable, and has a supported
/// Loom descriptor-family bundle.
///
/// @ownership
/// The caller owns the returned profile and releases it with
/// `loomc_target_profile_release`.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_amdgpu(
    loomc_target_environment_t* target_environment,
    const loomc_amdgpu_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile);

/// Returns the processor selected by an AMDGPU target profile.
///
/// @param profile AMDGPU target profile to query.
/// @return Empty when `profile` is `NULL` or was not created by the AMDGPU
/// profile API.
LOOMC_API_EXPORT loomc_string_view_t
loomc_amdgpu_target_profile_processor(const loomc_target_profile_t* profile);

/// Returns the gfx1250 revision selected by an AMDGPU target profile.
///
/// @param profile AMDGPU target profile to query.
/// @return A0 or B0 for gfx1250 profiles. DEFAULT is returned when `profile`
/// is `NULL`, was not created by the AMDGPU profile API, or selects another
/// processor.
LOOMC_API_EXPORT loomc_amdgpu_gfx1250_revision_t
loomc_amdgpu_target_profile_gfx1250_revision(
    const loomc_target_profile_t* profile);

/// Extracts an AMDGPU processor name from an HSA ISA target id.
///
/// HSA agents report ISA names such as
/// `amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-`. Loom target profiles use the
/// processor component, such as `gfx942`. The returned view is owned by Loom's
/// static AMDGPU target tables and remains valid for the lifetime of the
/// process.
///
/// @param hsa_isa_name HSA ISA target id reported by the runtime.
/// @param out_processor Receives a borrowed processor-name view on success.
/// @return OK when the target id is an AMDHSA target id with a known processor.
LOOMC_API_EXPORT loomc_status_t loomc_amdgpu_processor_from_hsa_isa_name(
    loomc_string_view_t hsa_isa_name, loomc_string_view_t* out_processor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMDGPU_PROFILE_H_
