// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_EXECUTABLE_AMDGPU_TARGET_ID_H_
#define IREE_HAL_EXECUTABLE_AMDGPU_TARGET_ID_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// AMDGPU Target IDs
//===----------------------------------------------------------------------===//

// Parsed gfx IP version.
typedef struct iree_hal_amdgpu_gfxip_version_t {
  // Major gfx ISA version, such as 9, 10, 11, or 12.
  uint32_t major;
  // Minor gfx ISA version within |major|.
  uint32_t minor;
  // Stepping digit within |major|.|minor|.
  uint32_t stepping;
} iree_hal_amdgpu_gfxip_version_t;

// Target feature selector state from AMDGPU target IDs.
typedef enum iree_hal_amdgpu_target_feature_state_e {
  // Feature is not represented by the parsed target ID.
  IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY = 0,
  // Feature is known not to be supported by the target.
  IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED,
  // Feature is explicitly disabled, such as `:xnack-`.
  IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF,
  // Feature is explicitly enabled, such as `:sramecc+`.
  IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON,
} iree_hal_amdgpu_target_feature_state_t;

// Target processor name class.
typedef enum iree_hal_amdgpu_target_kind_e {
  // Exact target processor such as `gfx942` or `gfx1100`.
  IREE_HAL_AMDGPU_TARGET_KIND_EXACT = 0,
  // Generic target processor such as `gfx9-4-generic` or `gfx11-generic`.
  IREE_HAL_AMDGPU_TARGET_KIND_GENERIC,
} iree_hal_amdgpu_target_kind_t;

// Structured AMDHSA target-ID feature states.
typedef struct iree_hal_amdgpu_amdhsa_feature_states_t {
  // SRAM ECC selector state.
  iree_hal_amdgpu_target_feature_state_t sramecc;
  // XNACK selector state.
  iree_hal_amdgpu_target_feature_state_t xnack;
} iree_hal_amdgpu_amdhsa_feature_states_t;

// Structured AMDGPU target identity.
typedef struct iree_hal_amdgpu_target_identity_t {
  // Target class used to interpret |version|.
  iree_hal_amdgpu_target_kind_t kind;
  // Parsed processor gfx IP version or generic family version.
  iree_hal_amdgpu_gfxip_version_t version;
  // Generic code-object format version from ELF e_flags, or 0 if unspecified.
  uint32_t generic_version;
  // Normalized AMDHSA target-ID feature states.
  iree_hal_amdgpu_amdhsa_feature_states_t amdhsa_features;
  // Borrowed canonical target selector without feature suffixes.
  iree_string_view_t target;
  // Borrowed backend processor selected by |target|.
  iree_string_view_t processor;
} iree_hal_amdgpu_target_identity_t;

// Compatibility reasons reported by
// iree_hal_amdgpu_target_identity_check_compatible.
typedef enum iree_hal_amdgpu_target_compatibility_bits_e {
  // Code object target identity is compatible with the agent target identity.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE = 0u,
  // Exact target selectors do not match.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_TARGET = 1u << 0,
  // Generic processor family does not match the agent's mapped family.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY = 1u << 1,
  // Generic code-object version is older than the agent's supported floor.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_VERSION = 1u << 2,
  // Explicit SRAM ECC mode does not match the agent.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_SRAMECC = 1u << 3,
  // Explicit XNACK mode does not match the agent.
  IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK = 1u << 4,
} iree_hal_amdgpu_target_compatibility_bits_t;
typedef uint32_t iree_hal_amdgpu_target_compatibility_t;

// Wavefront-size support flags for AMDGPU processors.
typedef uint32_t iree_hal_amdgpu_wavefront_size_flags_t;
typedef enum iree_hal_amdgpu_wavefront_size_flag_bits_e {
  // No wavefront-size flag bits are present.
  IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_NONE = 0u,
  // Wavefront-size-32 mode.
  IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_32 = 1u << 0,
  // Wavefront-size-64 mode.
  IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_64 = 1u << 1,
} iree_hal_amdgpu_wavefront_size_flag_bits_t;

// Wavefront-size facts derived from the AMDGPU processor table.
typedef struct iree_hal_amdgpu_wavefront_size_support_t {
  // Default wavefront size in lanes.
  uint32_t default_size;
  // Explicitly selectable wavefront-size modes from the kernel descriptor ABI.
  iree_hal_amdgpu_wavefront_size_flags_t explicit_supported_sizes;
} iree_hal_amdgpu_wavefront_size_support_t;

// Parses one exact or generic processor name into its same-named target.
//
// Feature modes remain unconstrained or unsupported according to the generated
// target table. No feature coordinates are accepted.
iree_status_t iree_hal_amdgpu_target_identity_parse_processor(
    iree_string_view_t processor,
    iree_hal_amdgpu_target_identity_t* out_identity);

// Parses a canonical AMDGPU artifact target key.
//
// The canonical target selector is followed by real AMDHSA feature suffixes
// using target-ID syntax. Returned string views either borrow from |value| or
// reference generated static target rows.
iree_status_t iree_hal_amdgpu_target_identity_parse_artifact_key(
    iree_string_view_t value, iree_hal_amdgpu_target_identity_t* out_identity);

// Parses an HSA ISA name reported by HSA_ISA_INFO_NAME.
iree_status_t iree_hal_amdgpu_target_identity_parse_hsa_isa_name(
    iree_string_view_t value, iree_hal_amdgpu_target_identity_t* out_identity);

// Returns true when physical discovery must resolve |identity| to a canonical
// target.
bool iree_hal_amdgpu_target_identity_requires_physical_resolution(
    const iree_hal_amdgpu_target_identity_t* identity);

// Resolves the HSA-reported physical ASIC revision to a canonical target.
//
// Processors without physical target rows are unchanged. Processors with
// physical target rows reject unknown values rather than retaining a revision
// coordinate with no target semantics.
iree_status_t iree_hal_amdgpu_target_identity_resolve_physical_target(
    uint32_t asic_revision, iree_hal_amdgpu_target_identity_t* identity);

// Returns true when |lhs| and |rhs| describe the same target identity.
//
// Borrowed string storage addresses are ignored; processor names and all
// structured target properties are compared by value.
bool iree_hal_amdgpu_target_identity_equal(
    const iree_hal_amdgpu_target_identity_t* lhs,
    const iree_hal_amdgpu_target_identity_t* rhs);

// Formats |identity| as a canonical AMDGPU artifact target key.
//
// If |buffer_capacity| is insufficient, |out_buffer_length| still receives the
// required character length excluding the NUL terminator.
iree_status_t iree_hal_amdgpu_target_identity_format_artifact_key(
    const iree_hal_amdgpu_target_identity_t* identity,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length);

// Projects an exact target identity into its AMDHSA code-object identity.
//
// The code-object processor mapping and AMDHSA feature modes are preserved.
// Target-overlay identity is deliberately projected away. If no generic
// code-object mapping is known, the backend processor is retained as the
// code-object target.
iree_status_t iree_hal_amdgpu_target_identity_project_code_object(
    const iree_hal_amdgpu_target_identity_t* exact_identity,
    iree_hal_amdgpu_target_identity_t* out_code_object_identity);

// Returns the wavefront-size flag for |wavefront_size|, or zero if unsupported.
iree_hal_amdgpu_wavefront_size_flags_t iree_hal_amdgpu_wavefront_size_flag(
    uint32_t wavefront_size);

// Looks up wavefront-size facts for an exact processor identity.
//
// The returned support preserves the processor table's explicit mode support.
// The implicit default mode is executable even when it is not explicitly
// selectable by the kernel descriptor ABI.
bool iree_hal_amdgpu_target_identity_lookup_wavefront_size_support(
    const iree_hal_amdgpu_target_identity_t* exact_identity,
    iree_hal_amdgpu_wavefront_size_support_t* out_support);

// Checks whether |artifact_identity| can execute on |agent_identity|.
iree_hal_amdgpu_target_compatibility_t
iree_hal_amdgpu_target_identity_check_compatible(
    const iree_hal_amdgpu_target_identity_t* artifact_identity,
    const iree_hal_amdgpu_target_identity_t* agent_identity);

// Formats compatibility mismatch bits into a comma-separated diagnostic string.
//
// If |buffer_capacity| is insufficient, |out_buffer_length| still receives the
// required character length excluding the NUL terminator.
iree_status_t iree_hal_amdgpu_target_compatibility_format(
    iree_hal_amdgpu_target_compatibility_t compatibility,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_EXECUTABLE_AMDGPU_TARGET_ID_H_
