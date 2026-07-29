// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact structural LDS bank-service analysis under named target models.
//
// The results are bank service rounds, not hardware cycles. Model provenance
// and address proof are deliberately separate: this layer evaluates explicit
// lane addresses under a model, while callers prove how those addresses follow
// from compiler-owned source or fragment layouts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ANALYSIS_LDS_BANK_SERVICE_H_
#define LOOM_TARGET_ARCH_AMDGPU_ANALYSIS_LDS_BANK_SERVICE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE 64
#define LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_BANK_COUNT 64
#define LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_PHASE_COUNT 8
#define LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_PACKET_WORD_COUNT 4

typedef enum loom_amdgpu_lds_bank_service_direction_e {
  LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ = 0,
  LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE = 1,
} loom_amdgpu_lds_bank_service_direction_t;

typedef enum loom_amdgpu_lds_bank_service_evidence_class_e {
  LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION = 0,
  LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED = 1,
  LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL = 2,
} loom_amdgpu_lds_bank_service_evidence_class_t;

typedef enum loom_amdgpu_lds_bank_service_request_policy_e {
  // Every packet bank-word request consumes service independently.
  LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH = 0,
  // Identical read addresses within one phase are coalesced.
  LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS = 1,
} loom_amdgpu_lds_bank_service_request_policy_t;

// One immutable target packet service model.
typedef struct loom_amdgpu_lds_bank_service_model_t {
  // Stable semantic key for the target, direction, and packet family.
  iree_string_view_t key;
  // Immutable source revision defining the model.
  iree_string_view_t revision;
  // Provenance strength of the model rather than the evaluated addresses.
  loom_amdgpu_lds_bank_service_evidence_class_t evidence_class;
  // Packet read or write direction.
  loom_amdgpu_lds_bank_service_direction_t direction;
  // Treatment of requests to identical bank-word addresses.
  loom_amdgpu_lds_bank_service_request_policy_t request_policy;
  // Number of lanes represented by the phase masks.
  uint8_t wave_size;
  // Number of independently serviced LDS banks.
  uint8_t bank_count;
  // Byte width of one LDS bank word.
  uint8_t bank_word_byte_count;
  // Number of consecutive bank words requested by each active lane.
  uint8_t packet_word_count;
  // Number of populated lane-service phase masks.
  uint8_t phase_count;
  // Active lane membership for each hardware service phase.
  uint64_t phase_lane_masks[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_PHASE_COUNT];
} loom_amdgpu_lds_bank_service_model_t;

// Exact service profile for one known active-lane set and lane-address map.
typedef struct loom_amdgpu_lds_bank_service_result_t {
  // Required bank service rounds for each model phase.
  uint16_t phase_required_rounds[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_PHASE_COUNT];
  // Number of populated phase results.
  uint8_t phase_count;
  // Number of bank-word base translations covered by the profile.
  uint8_t base_residue_count;
  // Sum of phase_required_rounds.
  uint16_t required_rounds;
  // One round for each phase containing at least one active lane.
  uint16_t uncontended_rounds;
  // Difference between required_rounds and uncontended_rounds.
  uint16_t extra_rounds;
  // Maximum requests assigned to one bank in one phase.
  uint16_t maximum_request_multiplicity;
} loom_amdgpu_lds_bank_service_result_t;

// Returns the immutable model selected by |model_set_ordinal| and
// |descriptor_ref|.
//
// Generated target rows intern complete, unambiguous model sets. Multiple
// processors and revisions may select the same set without introducing
// processor-specific query functions.
const loom_amdgpu_lds_bank_service_model_t*
loom_amdgpu_lds_bank_service_model_lookup(
    loom_amdgpu_lds_bank_service_model_set_ordinal_t model_set_ordinal,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

// Returns the stable report key for |evidence_class|.
iree_string_view_t loom_amdgpu_lds_bank_service_evidence_class_name(
    loom_amdgpu_lds_bank_service_evidence_class_t evidence_class);

// Returns the stable report key for |request_policy|.
iree_string_view_t loom_amdgpu_lds_bank_service_request_policy_name(
    loom_amdgpu_lds_bank_service_request_policy_t request_policy);

// Evaluates explicit lane-relative byte addresses under |model|.
//
// The first |model->wave_size| lane addresses must be aligned to the model bank
// word. All addresses share one unknown additive LDS base. Because adding a
// common bank-word residue only rotates bank indices, the returned profile
// proves translation invariance across every bank residue.
void loom_amdgpu_lds_bank_service_evaluate(
    const loom_amdgpu_lds_bank_service_model_t* model,
    uint64_t active_lane_mask,
    const uint64_t
        lane_base_byte_offsets[LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE],
    loom_amdgpu_lds_bank_service_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ANALYSIS_LDS_BANK_SERVICE_H_
