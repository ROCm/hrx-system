// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/collective_combine.h"

#include <stdint.h>

#include "loom/target/arch/amdgpu/lower/collective_payload.h"

bool loom_amdgpu_collective_combine_descriptor_ref(
    loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32;
      return true;
    case LOOM_COMBINING_KIND_MULI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32;
      return true;
    case LOOM_COMBINING_KIND_MINSI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_I32;
      return true;
    case LOOM_COMBINING_KIND_MAXSI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_I32;
      return true;
    case LOOM_COMBINING_KIND_MINUI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32;
      return true;
    case LOOM_COMBINING_KIND_MAXUI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_U32;
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32;
      return true;
    case LOOM_COMBINING_KIND_ORI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32;
      return true;
    case LOOM_COMBINING_KIND_XORI:
      if (!loom_amdgpu_collective_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32;
      return true;
    case LOOM_COMBINING_KIND_ADDF:
      if (!loom_amdgpu_collective_payload_is_float(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32;
      return true;
    case LOOM_COMBINING_KIND_MULF:
      if (!loom_amdgpu_collective_payload_is_float(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32;
      return true;
    case LOOM_COMBINING_KIND_MINNUMF:
      if (!loom_amdgpu_collective_payload_is_float(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32;
      return true;
    case LOOM_COMBINING_KIND_MAXNUMF:
      if (!loom_amdgpu_collective_payload_is_float(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32;
      return true;
    default:
      return false;
  }
}

bool loom_amdgpu_collective_combine_identity_bits(loom_combining_kind_t kind,
                                                  uint32_t* out_bits) {
  *out_bits = 0;
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
    case LOOM_COMBINING_KIND_ORI:
    case LOOM_COMBINING_KIND_XORI:
    case LOOM_COMBINING_KIND_ADDF:
      *out_bits = 0u;
      return true;
    case LOOM_COMBINING_KIND_MULI:
      *out_bits = 1u;
      return true;
    case LOOM_COMBINING_KIND_MINSI:
      *out_bits = (uint32_t)INT32_MAX;
      return true;
    case LOOM_COMBINING_KIND_MAXSI:
      *out_bits = (uint32_t)INT32_MIN;
      return true;
    case LOOM_COMBINING_KIND_MINUI:
      *out_bits = UINT32_MAX;
      return true;
    case LOOM_COMBINING_KIND_MAXUI:
      *out_bits = 0u;
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      *out_bits = UINT32_MAX;
      return true;
    case LOOM_COMBINING_KIND_MULF:
      *out_bits = 0x3f800000u;
      return true;
    case LOOM_COMBINING_KIND_MINNUMF:
      *out_bits = 0x7f800000u;
      return true;
    case LOOM_COMBINING_KIND_MAXNUMF:
      *out_bits = 0xff800000u;
      return true;
    default:
      return false;
  }
}
