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
/// The AMDGPU public profile selects one exact, generic, or overlay AMDGPU
/// target such as `gfx1151`, `gfx11-generic`, or `gfx1250-a0`. Each target
/// binds a backend processor to all semantic overlays required to compile for
/// that target. The profile determines the descriptor-family target bundle,
/// native HSACO support, default wavefront size, HSA target id, and
/// target-record family used by compilation and emission. Live HSA/HIP/HRX
/// adapters resolve their physical device observations to a canonical target
/// and then create a normal `loomc_target_profile_t`.

#ifdef __cplusplus
extern "C" {
#endif

/// Normalized state of one AMDHSA target-ID feature.
typedef enum loomc_amdgpu_target_feature_state_e {
  /// The target identity does not constrain the feature.
  LOOMC_AMDGPU_TARGET_FEATURE_ANY = 0,

  /// The selected target is known not to support the feature.
  LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED = 1,

  /// The feature is explicitly disabled, such as `xnack-`.
  LOOMC_AMDGPU_TARGET_FEATURE_OFF = 2,

  /// The feature is explicitly enabled, such as `sramecc+`.
  LOOMC_AMDGPU_TARGET_FEATURE_ON = 3,
} loomc_amdgpu_target_feature_state_t;

/// Structured AMDHSA target-ID feature states.
typedef struct loomc_amdgpu_amdhsa_feature_states_t {
  /// SRAM ECC target-ID feature state.
  loomc_amdgpu_target_feature_state_t sramecc;

  /// XNACK target-ID feature state.
  loomc_amdgpu_target_feature_state_t xnack;
} loomc_amdgpu_amdhsa_feature_states_t;

/// Complete structured AMDGPU identity used to construct a target profile.
typedef struct loomc_amdgpu_target_identity_t {
  /// Exact, generic, or overlay AMDGPU target selector, such as `gfx1151`,
  /// `gfx11-generic`, or `gfx1250-a0`.
  loomc_string_view_t target;

  /// Structured AMDHSA target-ID feature states.
  loomc_amdgpu_amdhsa_feature_states_t amdhsa_features;
} loomc_amdgpu_target_identity_t;

/// Parses a canonical AMDGPU artifact key into a target identity.
///
/// The key begins with an exact, generic, or overlay target selector and may
/// carry AMDHSA feature coordinates such as `:sramecc+:xnack-`.
///
/// @param artifact_key Canonical compiler artifact-selection key.
/// @param out_identity Receives the structured AMDGPU target identity.
/// @return OK when the key names a supported compiler target and all feature
/// coordinates are valid for its processor.
///
/// @lifetime
/// `out_identity->target` is owned by Loom's static AMDGPU target tables and
/// remains valid for the lifetime of the process.
LOOMC_API_EXPORT loomc_status_t loomc_amdgpu_target_identity_parse_artifact_key(
    loomc_string_view_t artifact_key,
    loomc_amdgpu_target_identity_t* out_identity);

/// AMDGPU target profile creation options.
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
  /// Empty uses `identity.target`.
  loomc_string_view_t identifier;

  /// Structured AMDGPU target identity.
  loomc_amdgpu_target_identity_t identity;
} loomc_amdgpu_profile_options_t;

/// Creates an AMDGPU target profile from a canonical target selector.
///
/// @param target_environment AMDGPU-capable target environment.
/// @param options Profile options. Required.
/// @param allocator Host allocator used for profile storage.
/// @param out_profile Receives one retained profile on success.
/// @return OK when the target is known, HSACO-capable, and has a supported Loom
/// descriptor-family bundle.
///
/// @ownership
/// The caller owns the returned profile and releases it with
/// `loomc_target_profile_release`.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_amdgpu(
    loomc_target_environment_t* target_environment,
    const loomc_amdgpu_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile);

/// Returns the complete identity selected by an AMDGPU target profile.
///
/// @param profile AMDGPU target profile to query.
/// @param out_identity Receives the structured AMDGPU identity.
/// @return OK when `profile` is an AMDGPU target profile.
///
/// @lifetime
/// `out_identity->target` remains valid until `profile` is released.
LOOMC_API_EXPORT loomc_status_t loomc_amdgpu_target_profile_query_identity(
    const loomc_target_profile_t* profile,
    loomc_amdgpu_target_identity_t* out_identity);

/// Resolves an HSA ISA name and physical ASIC revision to an AMDGPU identity.
///
/// HSA agents report ISA names such as
/// `amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-`. The adapter preserves every
/// structured target-ID feature and resolves the physical processor and ASIC
/// revision through Loom's generated target map. The resulting identity uses a
/// canonical target selector and carries no independent revision coordinate.
///
/// @param hsa_isa_name HSA ISA target id reported by the runtime.
/// @param asic_revision Physical ASIC revision reported by HSA.
/// @param out_identity Receives the structured AMDGPU target identity.
/// @return OK when the target id names a known processor and its physical
/// observation maps to a supported canonical target.
///
/// @lifetime
/// `out_identity->target` is owned by Loom's static AMDGPU target tables and
/// remains valid for the lifetime of the process.
LOOMC_API_EXPORT loomc_status_t loomc_amdgpu_target_identity_from_hsa_isa_name(
    loomc_string_view_t hsa_isa_name, uint32_t asic_revision,
    loomc_amdgpu_target_identity_t* out_identity);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMDGPU_PROFILE_H_
