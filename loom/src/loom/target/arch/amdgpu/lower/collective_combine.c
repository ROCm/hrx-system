// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/collective_combine.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/lower/collective_payload.h"

typedef struct loom_amdgpu_collective_dpp_combine_descriptor_row_t {
  // Source combining kind lowered by this native DPP packet row.
  loom_combining_kind_t kind;
  // Legacy VOP2 DPP descriptor reference for pre-DPP16 targets.
  loom_amdgpu_descriptor_ref_t legacy_descriptor_ref;
  // DPP16 descriptor reference for RDNA3 and newer targets.
  loom_amdgpu_descriptor_ref_t dpp16_descriptor_ref;
} loom_amdgpu_collective_dpp_combine_descriptor_row_t;

static const loom_amdgpu_collective_dpp_combine_descriptor_row_t
    kLoomAmdgpuCollectiveDppCombineDescriptorRows[] = {
        {
            .kind = LOOM_COMBINING_KIND_ADDF,
            .legacy_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32_DPP,
            .dpp16_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32_DPP16,
        },
        {
            .kind = LOOM_COMBINING_KIND_MULF,
            .legacy_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32_DPP,
            .dpp16_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32_DPP16,
        },
        {
            .kind = LOOM_COMBINING_KIND_MINNUMF,
            .legacy_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32_DPP,
            .dpp16_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32_DPP16,
        },
        {
            .kind = LOOM_COMBINING_KIND_MAXNUMF,
            .legacy_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32_DPP,
            .dpp16_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32_DPP16,
        },
};

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

bool loom_amdgpu_collective_combine_dpp_descriptor_ref(
    loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_amdgpu_collective_combine_dpp_form_t dpp_form,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_collective_payload_is_float(payload_kind)) {
    return false;
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuCollectiveDppCombineDescriptorRows); ++i) {
    const loom_amdgpu_collective_dpp_combine_descriptor_row_t* row =
        &kLoomAmdgpuCollectiveDppCombineDescriptorRows[i];
    if (row->kind != kind) {
      continue;
    }
    switch (dpp_form) {
      case LOOM_AMDGPU_COLLECTIVE_COMBINE_DPP_FORM_LEGACY:
        *out_descriptor_ref = row->legacy_descriptor_ref;
        return true;
      case LOOM_AMDGPU_COLLECTIVE_COMBINE_DPP_FORM_DPP16:
        *out_descriptor_ref = row->dpp16_descriptor_ref;
        return true;
      default:
        return false;
    }
  }
  return false;
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
