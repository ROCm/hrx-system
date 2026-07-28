// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/analysis/lds_bank_service.h"

#include <stddef.h>

static const loom_amdgpu_lds_bank_service_model_t kGfx1250ReadB128Model = {
    .key = IREE_SVL("gfx1250.rocroller.ds_read_b128"),
    .revision = IREE_SVL("ROCm/rocm-libraries@a7e3879c8847:LDSModel.cpp"),
    .evidence_class =
        LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED,
    .direction = LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ,
    .request_policy = LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH,
    .wave_size = 32,
    .bank_count = 32,
    .bank_word_byte_count = 4,
    .packet_word_count = 4,
    .phase_count = 8,
    .phase_lane_masks =
        {
            UINT64_C(0x0000000F),
            UINT64_C(0x000000F0),
            UINT64_C(0x00000F00),
            UINT64_C(0x0000F000),
            UINT64_C(0x000F0000),
            UINT64_C(0x00F00000),
            UINT64_C(0x0F000000),
            UINT64_C(0xF0000000),
        },
};

static const loom_amdgpu_lds_bank_service_model_t kGfx1250WriteB128Model = {
    .key = IREE_SVL("gfx1250.rocroller.ds_write_b128"),
    .revision = IREE_SVL("ROCm/rocm-libraries@a7e3879c8847:LDSModel.cpp"),
    .evidence_class =
        LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED,
    .direction = LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE,
    .request_policy = LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH,
    .wave_size = 32,
    .bank_count = 32,
    .bank_word_byte_count = 4,
    .packet_word_count = 4,
    .phase_count = 8,
    .phase_lane_masks =
        {
            UINT64_C(0x0000000F),
            UINT64_C(0x000000F0),
            UINT64_C(0x00000F00),
            UINT64_C(0x0000F000),
            UINT64_C(0x000F0000),
            UINT64_C(0x00F00000),
            UINT64_C(0x0F000000),
            UINT64_C(0xF0000000),
        },
};

const loom_amdgpu_lds_bank_service_model_t*
loom_amdgpu_lds_bank_service_gfx1250_b128_model(
    loom_amdgpu_lds_bank_service_direction_t direction) {
  switch (direction) {
    case LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ:
      return &kGfx1250ReadB128Model;
    case LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE:
      return &kGfx1250WriteB128Model;
    default:
      return NULL;
  }
}

static bool loom_amdgpu_lds_bank_service_word_already_requested(
    const uint64_t* requested_words, uint16_t requested_word_count,
    uint64_t address_word) {
  for (uint16_t i = 0; i < requested_word_count; ++i) {
    if (requested_words[i] == address_word) return true;
  }
  return false;
}

void loom_amdgpu_lds_bank_service_evaluate(
    const loom_amdgpu_lds_bank_service_model_t* model,
    uint64_t active_lane_mask,
    const uint64_t
        lane_base_byte_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE],
    loom_amdgpu_lds_bank_service_result_t* out_result) {
  IREE_ASSERT(
      model->wave_size > 0 &&
      model->wave_size <= LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE &&
      model->bank_count > 0 &&
      model->bank_count <= LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_BANK_COUNT &&
      model->bank_word_byte_count > 0 && model->packet_word_count > 0 &&
      model->packet_word_count <=
          LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_PACKET_WORD_COUNT &&
      model->phase_count > 0 &&
      model->phase_count <= LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_PHASE_COUNT);
  const uint64_t valid_lane_mask =
      model->wave_size == 64 ? UINT64_MAX
                             : (UINT64_C(1) << model->wave_size) - UINT64_C(1);
  IREE_ASSERT((active_lane_mask & ~valid_lane_mask) == 0);
  const bool coalesce_identical_reads =
      model->direction == LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ &&
      model->request_policy ==
          LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS;

  *out_result = (loom_amdgpu_lds_bank_service_result_t){
      .phase_count = model->phase_count,
      .base_residue_count = model->bank_count,
  };
  for (uint8_t phase = 0; phase < model->phase_count; ++phase) {
    uint16_t bank_request_counts[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_BANK_COUNT] =
        {0};
    uint64_t
        requested_words[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE *
                        LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_PACKET_WORD_COUNT] = {
            0};
    uint16_t requested_word_count = 0;
    const uint64_t phase_active_lanes =
        active_lane_mask & model->phase_lane_masks[phase];
    if (phase_active_lanes != 0) ++out_result->uncontended_rounds;

    for (uint8_t lane = 0; lane < model->wave_size; ++lane) {
      if ((phase_active_lanes & (UINT64_C(1) << lane)) == 0) continue;
      const uint64_t lane_base_byte_offset = lane_base_byte_offsets[lane];
      IREE_ASSERT((lane_base_byte_offset % model->bank_word_byte_count) == 0);
      const uint64_t lane_base_word =
          lane_base_byte_offset / model->bank_word_byte_count;
      for (uint8_t packet_word = 0; packet_word < model->packet_word_count;
           ++packet_word) {
        const uint64_t address_word = lane_base_word + packet_word;
        if (coalesce_identical_reads &&
            loom_amdgpu_lds_bank_service_word_already_requested(
                requested_words, requested_word_count, address_word)) {
          continue;
        }
        if (coalesce_identical_reads) {
          requested_words[requested_word_count++] = address_word;
        }
        const uint8_t bank = (uint8_t)(address_word % model->bank_count);
        const uint16_t request_count = ++bank_request_counts[bank];
        out_result->maximum_request_multiplicity =
            iree_max(out_result->maximum_request_multiplicity, request_count);
        out_result->phase_required_rounds[phase] =
            iree_max(out_result->phase_required_rounds[phase], request_count);
      }
    }
    out_result->required_rounds += out_result->phase_required_rounds[phase];
  }
  out_result->extra_rounds =
      out_result->required_rounds - out_result->uncontended_rounds;
}
