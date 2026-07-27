// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low sanitizer access-check builders.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_ACCESS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_ACCESS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

typedef struct loom_amdgpu_sanitizer_access_config_t {
  // VGPRx2 device-visible ASAN shadow base loaded from the runtime config.
  loom_value_id_t shadow_base;
  // VGPRx1 zero constant used by shadow-byte predicates.
  loom_value_id_t zero;
  // VGPRx1 minimum ASAN full-poison shadow byte value.
  loom_value_id_t poison_min;
} loom_amdgpu_sanitizer_access_config_t;

typedef struct loom_amdgpu_sanitizer_access_check_t {
  // Native lane mask identifying lanes that failed the access check.
  loom_value_id_t failure_mask;
  // Shadow address selected for reporting a failed access.
  loom_value_id_t shadow_address;
  // Shadow byte selected for reporting a failed access.
  loom_value_id_t shadow_value;
} loom_amdgpu_sanitizer_access_check_t;

// Loads the per-kernel ASAN shadow configuration and materializes constants
// shared by access checks emitted at the current insertion point.
iree_status_t loom_amdgpu_build_sanitizer_access_config(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t asan_config_symbol, loom_location_id_t location,
    loom_amdgpu_sanitizer_access_config_t* out_config);

// Emits the hot-path AMDGPU ASAN-style failure predicate for one static access
// range.
//
// |fault_address| must be a 64-bit SGPR or VGPR register range. |access_size|
// is packetized into <=8 byte chunks so every touched shadow byte is covered.
// |wavefront_size| must be 32 or 64 and controls whether the returned SGPRx2
// |out_failure_mask| needs wave32 zero-extension before it is consumed as an
// EXEC-width lane mask. This helper only returns the assertion predicate and
// intentionally does not preserve report payload values across the hot/cold
// split.
iree_status_t loom_amdgpu_build_sanitizer_access_failure_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t asan_config_symbol, loom_value_id_t fault_address,
    uint32_t access_size, uint32_t wavefront_size, loom_location_id_t location,
    loom_value_id_t* out_failure_mask);

// Emits the hot-path AMDGPU ASAN-style failure predicate using an already
// loaded sanitizer access configuration.
iree_status_t loom_amdgpu_build_sanitizer_access_failure_mask_with_config(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_config_t* config,
    loom_value_id_t fault_address, uint32_t access_size,
    uint32_t minimum_alignment, uint32_t wavefront_size,
    loom_location_id_t location, loom_value_id_t* out_failure_mask);

// Emits a native SGPRx2 lane-mask union for two sanitizer failure masks.
iree_status_t loom_amdgpu_build_sanitizer_access_failure_mask_union(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs_failure_mask, loom_value_id_t rhs_failure_mask,
    loom_location_id_t location, loom_value_id_t* out_failure_mask);

// Selects a 64-bit VGPR value by native SGPRx2 lane mask.
iree_status_t loom_amdgpu_build_sanitizer_access_vgpr64_select(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition_mask, loom_location_id_t location,
    loom_value_id_t* out_value);

// Emits an AMDGPU ASAN-style shadow check for one static access range.
//
// |fault_address| must be a 64-bit SGPR or VGPR register range. |access_size|
// must be non-zero and is packetized into <=8 byte chunks so every touched
// shadow byte is covered.
// |wavefront_size| must be 32 or 64 and controls how the returned SGPRx2
// |failure_mask| is canonicalized before it is consumed as an EXEC-width lane
// mask by loom_amdgpu_build_sanitizer_access_report_failure_mask_branch.
// Report payload fields are selected for the failing lanes and should be built
// on the already-failing path when possible.
iree_status_t loom_amdgpu_build_sanitizer_access_check(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t asan_config_symbol, loom_value_id_t fault_address,
    uint32_t access_size, uint32_t wavefront_size, loom_location_id_t location,
    loom_amdgpu_sanitizer_access_check_t* out_check);

// Emits an AMDGPU ASAN-style shadow check using an already loaded sanitizer
// access configuration.
iree_status_t loom_amdgpu_build_sanitizer_access_check_with_config(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_config_t* config,
    loom_value_id_t fault_address, uint32_t access_size,
    uint32_t minimum_alignment, uint32_t wavefront_size,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_access_check_t* out_check);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_ACCESS_H_
