// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU descriptor-set-local register-class helpers.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_REGISTER_CLASS_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_REGISTER_CLASS_H_

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scalar operand-encoding locations reserved for TTMP0 through TTMP15.
#define LOOM_AMDGPU_TTMP_SGPR_LOCATION_BASE 108u
#define LOOM_AMDGPU_TTMP_SGPR_LOCATION_COUNT 16u

// Returns true when the non-empty SGPR location range is wholly within the
// architectural TTMP encoding window.
static inline bool loom_amdgpu_sgpr_location_range_is_ttmp(
    uint32_t location_base, uint32_t location_count) {
  const uint64_t location_end = (uint64_t)location_base + location_count;
  return location_count != 0 &&
         location_base >= LOOM_AMDGPU_TTMP_SGPR_LOCATION_BASE &&
         location_end <= LOOM_AMDGPU_TTMP_SGPR_LOCATION_BASE +
                             LOOM_AMDGPU_TTMP_SGPR_LOCATION_COUNT;
}

// Returns true when |reg_class_id| names the accumulator register file in
// |descriptor_set|. Register-class IDs are descriptor-set-local; RDNA sets use
// the same numeric ID for M0 that CDNA4 uses for AGPR.
static inline bool loom_amdgpu_register_class_is_agpr(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  return iree_any_bit_set(
      loom_amdgpu_reg_class_traits(descriptor_set, reg_class_id),
      LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR);
}

// Returns true when |reg_class_id| names the M0 special-register class in
// |descriptor_set|.
static inline bool loom_amdgpu_register_class_is_m0(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  return iree_any_bit_set(
      loom_amdgpu_reg_class_traits(descriptor_set, reg_class_id),
      LOOM_AMDGPU_REG_CLASS_TRAIT_M0);
}

// Returns true when |reg_class_id| names the VCC condition-mask register class
// in |descriptor_set|.
static inline bool loom_amdgpu_register_class_is_vcc(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  return iree_any_bit_set(
      loom_amdgpu_reg_class_traits(descriptor_set, reg_class_id),
      LOOM_AMDGPU_REG_CLASS_TRAIT_VCC);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_REGISTER_CLASS_H_
