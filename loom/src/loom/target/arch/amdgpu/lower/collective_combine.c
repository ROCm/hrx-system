// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/collective_combine.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/lower/collective_payload.h"

typedef uint8_t loom_amdgpu_collective_combine_row_flags_t;

enum loom_amdgpu_collective_combine_row_flags_e {
  LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_INTEGER_PAYLOAD = 1u << 0,
  LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_FLOAT_PAYLOAD = 1u << 1,
  LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_HAS_IDENTITY = 1u << 2,
  LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_INTEGER =
      LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_INTEGER_PAYLOAD |
      LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_HAS_IDENTITY,
  LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_FLOAT =
      LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_FLOAT_PAYLOAD |
      LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_HAS_IDENTITY,
};

typedef struct loom_amdgpu_collective_combine_row_t {
  // Payload family accepted by the row.
  loom_amdgpu_collective_combine_row_flags_t flags;
  // Native VGPR descriptor reference used by reductions and scans.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // 32-bit identity bit pattern for guarded reductions and scans.
  uint32_t identity_bits;
  // Legacy VOP2 DPP descriptor reference for pre-DPP16 row reductions.
  loom_amdgpu_descriptor_ref_t legacy_descriptor_ref;
  // DPP16 descriptor reference for RDNA3 and newer row reductions.
  loom_amdgpu_descriptor_ref_t dpp16_descriptor_ref;
} loom_amdgpu_collective_combine_row_t;

#define LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(descriptor, identity) \
  {                                                                      \
      .flags = LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_INTEGER,               \
      .descriptor_ref = descriptor,                                      \
      .identity_bits = identity,                                         \
      .legacy_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,          \
      .dpp16_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,           \
  }

#define LOOM_AMDGPU_COLLECTIVE_COMBINE_FLOAT_ROW(descriptor, identity, \
                                                 legacy_dpp, dpp16)    \
  {                                                                    \
      .flags = LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_FLOAT,               \
      .descriptor_ref = descriptor,                                    \
      .identity_bits = identity,                                       \
      .legacy_descriptor_ref = legacy_dpp,                             \
      .dpp16_descriptor_ref = dpp16,                                   \
  }

static const loom_amdgpu_collective_combine_row_t
    kLoomAmdgpuCollectiveCombineRows[LOOM_COMBINING_KIND_COUNT_] = {
        [LOOM_COMBINING_KIND_ADDI] = LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
            LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, 0u),
        [LOOM_COMBINING_KIND_ADDF] = LOOM_AMDGPU_COLLECTIVE_COMBINE_FLOAT_ROW(
            LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32, 0u,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32_DPP,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32_DPP16),
        [LOOM_COMBINING_KIND_MULI] = LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32, 1u),
        [LOOM_COMBINING_KIND_MULF] = LOOM_AMDGPU_COLLECTIVE_COMBINE_FLOAT_ROW(
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, 0x3f800000u,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32_DPP,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32_DPP16),
        [LOOM_COMBINING_KIND_MINSI] =
            LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_I32, (uint32_t)INT32_MAX),
        [LOOM_COMBINING_KIND_MAXSI] =
            LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_I32, (uint32_t)INT32_MIN),
        [LOOM_COMBINING_KIND_MINUI] =
            LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32, UINT32_MAX),
        [LOOM_COMBINING_KIND_MAXUI] =
            LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_U32, 0u),
        [LOOM_COMBINING_KIND_ANDI] = LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
            LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32, UINT32_MAX),
        [LOOM_COMBINING_KIND_ORI] = LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
            LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, 0u),
        [LOOM_COMBINING_KIND_XORI] = LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW(
            LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32, 0u),
        [LOOM_COMBINING_KIND_MINNUMF] =
            LOOM_AMDGPU_COLLECTIVE_COMBINE_FLOAT_ROW(
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32, 0x7f800000u,
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32_DPP,
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32_DPP16),
        [LOOM_COMBINING_KIND_MAXNUMF] =
            LOOM_AMDGPU_COLLECTIVE_COMBINE_FLOAT_ROW(
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32, 0xff800000u,
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32_DPP,
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32_DPP16),
};

#undef LOOM_AMDGPU_COLLECTIVE_COMBINE_FLOAT_ROW
#undef LOOM_AMDGPU_COLLECTIVE_COMBINE_INTEGER_ROW

static const loom_amdgpu_collective_combine_row_t*
loom_amdgpu_collective_combine_row(loom_combining_kind_t kind) {
  if (!loom_combining_kind_is_valid(kind)) {
    return NULL;
  }
  const loom_amdgpu_collective_combine_row_t* row =
      &kLoomAmdgpuCollectiveCombineRows[kind];
  return row->flags == 0 ? NULL : row;
}

static bool loom_amdgpu_collective_combine_row_accepts_payload(
    const loom_amdgpu_collective_combine_row_t* row,
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  if (iree_all_bits_set(row->flags,
                        LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_INTEGER_PAYLOAD)) {
    return loom_amdgpu_collective_payload_is_integer(payload_kind);
  }
  if (iree_all_bits_set(row->flags,
                        LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_FLOAT_PAYLOAD)) {
    return loom_amdgpu_collective_payload_is_float(payload_kind);
  }
  return false;
}

bool loom_amdgpu_collective_combine_descriptor_ref(
    loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_amdgpu_collective_combine_row_t* row =
      loom_amdgpu_collective_combine_row(kind);
  if (row == NULL ||
      !loom_amdgpu_collective_combine_row_accepts_payload(row, payload_kind)) {
    return false;
  }
  *out_descriptor_ref = row->descriptor_ref;
  return true;
}

bool loom_amdgpu_collective_combine_dpp_descriptor_ref(
    loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_amdgpu_collective_combine_dpp_form_t dpp_form,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_amdgpu_collective_combine_row_t* row =
      loom_amdgpu_collective_combine_row(kind);
  if (row == NULL ||
      !loom_amdgpu_collective_combine_row_accepts_payload(row, payload_kind)) {
    return false;
  }
  switch (dpp_form) {
    case LOOM_AMDGPU_COLLECTIVE_COMBINE_DPP_FORM_LEGACY:
      *out_descriptor_ref = row->legacy_descriptor_ref;
      break;
    case LOOM_AMDGPU_COLLECTIVE_COMBINE_DPP_FORM_DPP16:
      *out_descriptor_ref = row->dpp16_descriptor_ref;
      break;
    default:
      return false;
  }
  return *out_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

bool loom_amdgpu_collective_combine_identity_bits(loom_combining_kind_t kind,
                                                  uint32_t* out_bits) {
  *out_bits = 0;
  const loom_amdgpu_collective_combine_row_t* row =
      loom_amdgpu_collective_combine_row(kind);
  if (row == NULL ||
      !iree_all_bits_set(row->flags,
                         LOOM_AMDGPU_COLLECTIVE_COMBINE_ROW_HAS_IDENTITY)) {
    return false;
  }
  *out_bits = row->identity_bits;
  return true;
}
