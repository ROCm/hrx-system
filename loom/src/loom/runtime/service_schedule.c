// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/service_schedule.h"

#include <string.h>

struct loom_runtime_service_plan_t {
  // Eight little-endian u64 words in plan_layout_t field order.
  uint64_t words[8];
};

typedef struct loom_service_schedule_decision_t {
  // Service ordinal/incarnation, offer ordinal/revision, and choice ordinal.
  uint64_t words[5];
} loom_service_schedule_decision_t;

typedef struct loom_service_schedule_outcome_t {
  // Acceptance, numeric status, operation ordinal, and operation serial.
  uint64_t words[4];
} loom_service_schedule_outcome_t;

static_assert(sizeof(loom_runtime_service_plan_t) == 64, "plan header ABI");
static_assert(sizeof(loom_service_schedule_decision_t) == 40, "decision ABI");
static_assert(sizeof(loom_service_schedule_outcome_t) == 32, "outcome ABI");

static uint64_t loom_service_schedule_word(uint64_t value) {
  return iree_unaligned_convert_native_le_u64(value);
}

static iree_status_t loom_service_schedule_layout(
    iree_host_size_t decision_capacity,
    loom_runtime_service_plan_layout_t* out_layout) {
  *out_layout = (loom_runtime_service_plan_layout_t){0};
  iree_host_size_t length = 0, decisions_offset = 0, results_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_runtime_service_plan_t), &length,
      IREE_STRUCT_FIELD_ALIGNED(decision_capacity,
                                loom_service_schedule_decision_t, 8,
                                &decisions_offset),
      IREE_STRUCT_FIELD_ALIGNED(decision_capacity,
                                loom_service_schedule_outcome_t, 8,
                                &results_offset)));
  *out_layout = (loom_runtime_service_plan_layout_t){
      .byte_length = length,
      .decision_capacity = decision_capacity,
      .decisions_offset = decisions_offset,
      .results_offset = results_offset};
  return iree_ok_status();
}

iree_status_t loom_runtime_service_plan_size(
    iree_host_size_t decision_capacity, iree_host_size_t* out_byte_length) {
  *out_byte_length = 0;
  loom_runtime_service_plan_layout_t layout;
  IREE_RETURN_IF_ERROR(
      loom_service_schedule_layout(decision_capacity, &layout));
  *out_byte_length = layout.byte_length;
  return iree_ok_status();
}

iree_status_t loom_runtime_service_plan_initialize(
    iree_host_size_t decision_capacity, iree_byte_span_t storage,
    loom_runtime_service_plan_t** out_plan) {
  *out_plan = NULL;
  loom_runtime_service_plan_layout_t layout;
  IREE_RETURN_IF_ERROR(
      loom_service_schedule_layout(decision_capacity, &layout));
  if (!storage.data || ((uintptr_t)storage.data & 7) ||
      storage.data_length < layout.byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "service plan requires aligned storage of at least %" PRIu64 " bytes",
        layout.byte_length);
  }
  loom_runtime_service_plan_t* plan = (void*)storage.data;
  memset(plan, 0, sizeof(*plan));
  plan->words[0] = loom_service_schedule_word(layout.byte_length);
  plan->words[1] = loom_service_schedule_word(layout.decision_capacity);
  plan->words[4] = loom_service_schedule_word(layout.decisions_offset);
  plan->words[5] = loom_service_schedule_word(layout.results_offset);
  *out_plan = plan;
  return iree_ok_status();
}

loom_runtime_service_plan_layout_t loom_runtime_service_plan_layout(
    const loom_runtime_service_plan_t* plan) {
  return (loom_runtime_service_plan_layout_t){
      .byte_length = loom_service_schedule_word(plan->words[0]),
      .decision_capacity = loom_service_schedule_word(plan->words[1]),
      .decision_count = loom_service_schedule_word(plan->words[2]),
      .applied_count = loom_service_schedule_word(plan->words[3]),
      .decisions_offset = loom_service_schedule_word(plan->words[4]),
      .results_offset = loom_service_schedule_word(plan->words[5]),
      .selection_flags = loom_service_schedule_word(plan->words[6]),
      .reserved = loom_service_schedule_word(plan->words[7])};
}

void loom_runtime_service_plan_reset(loom_runtime_service_plan_t* plan) {
  plan->words[2] = 0;
  plan->words[3] = 0;
  plan->words[6] = 0;
}

static loom_service_schedule_decision_t* loom_service_schedule_decisions(
    const loom_runtime_service_plan_t* plan) {
  return (void*)((uint8_t*)plan + loom_service_schedule_word(plan->words[4]));
}

static loom_service_schedule_outcome_t* loom_service_schedule_outcomes(
    const loom_runtime_service_plan_t* plan) {
  return (void*)((uint8_t*)plan + loom_service_schedule_word(plan->words[5]));
}

bool loom_runtime_service_plan_append(
    loom_runtime_service_plan_t* plan,
    const loom_runtime_service_decision_t* decision) {
  const uint64_t count = loom_service_schedule_word(plan->words[2]);
  if (count == loom_service_schedule_word(plan->words[1])) {
    plan->words[6] =
        loom_service_schedule_word(LOOM_RUNTIME_SERVICE_SELECTION_CAPACITY);
    return false;
  }
  loom_service_schedule_decision_t* target =
      &loom_service_schedule_decisions(plan)[count];
  target->words[0] = loom_service_schedule_word(decision->service_ordinal);
  target->words[1] = loom_service_schedule_word(decision->service_incarnation);
  target->words[2] = loom_service_schedule_word(decision->offer.ordinal);
  target->words[3] = loom_service_schedule_word(decision->offer.revision);
  target->words[4] = loom_service_schedule_word(decision->choice_ordinal);
  plan->words[2] = loom_service_schedule_word(count + 1);
  return true;
}

loom_runtime_service_decision_t loom_runtime_service_plan_decision(
    const loom_runtime_service_plan_t* plan, uint64_t ordinal) {
  const loom_service_schedule_decision_t* source =
      &loom_service_schedule_decisions(plan)[ordinal];
  return (loom_runtime_service_decision_t){
      .service_ordinal = loom_service_schedule_word(source->words[0]),
      .service_incarnation = loom_service_schedule_word(source->words[1]),
      .offer = {.ordinal = loom_service_schedule_word(source->words[2]),
                .revision = loom_service_schedule_word(source->words[3])},
      .choice_ordinal = loom_service_schedule_word(source->words[4])};
}

void loom_runtime_service_plan_complete(
    loom_runtime_service_plan_t* plan, iree_status_code_t status_code,
    loom_runtime_service_operation_key_t operation) {
  const uint64_t count = loom_service_schedule_word(plan->words[3]);
  loom_service_schedule_outcome_t* target =
      &loom_service_schedule_outcomes(plan)[count];
  target->words[0] = loom_service_schedule_word(operation.serial != 0);
  target->words[1] = loom_service_schedule_word(status_code);
  target->words[2] = loom_service_schedule_word(operation.ordinal);
  target->words[3] = loom_service_schedule_word(operation.serial);
  plan->words[3] = loom_service_schedule_word(count + 1);
}

loom_runtime_service_schedule_outcome_t loom_runtime_service_plan_outcome(
    const loom_runtime_service_plan_t* plan, uint64_t ordinal) {
  const loom_service_schedule_outcome_t* source =
      &loom_service_schedule_outcomes(plan)[ordinal];
  return (loom_runtime_service_schedule_outcome_t){
      .accepted = loom_service_schedule_word(source->words[0]) != 0,
      .status_code =
          (iree_status_code_t)loom_service_schedule_word(source->words[1]),
      .operation = {.ordinal = loom_service_schedule_word(source->words[2]),
                    .serial = loom_service_schedule_word(source->words[3])}};
}
