// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/representation_observer.h"

#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/op_defs.h"

typedef struct loom_low_lower_representation_observer_state_t {
  // Target policy supplying participation and boundary constraints.
  const loom_low_lower_representation_provider_t* provider;
  // Function-local source value domain used for compact ordinals.
  const loom_local_value_domain_t* value_domain;
  // Sparse equality and exact-candidate selection state.
  loom_low_representation_plan_t plan;
  // First or joined boundary-observation failure.
  iree_status_t terminal_status;
} loom_low_lower_representation_observer_state_t;

struct loom_low_lower_representation_recorder_t {
  // Active function-local representation state.
  loom_low_lower_representation_observer_state_t* state;
};

static const uint8_t kLoomLowLowerRepresentationStateKey;

static iree_status_t loom_low_lower_representation_take_status(
    loom_low_lower_representation_observer_state_t* state) {
  iree_status_t status = state->terminal_status;
  state->terminal_status = iree_ok_status();
  return status;
}

IREE_ATTRIBUTE_NOINLINE void loom_low_lower_representation_record_failure(
    loom_low_lower_representation_recorder_t* recorder, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(recorder);
  recorder->state->terminal_status =
      iree_status_join(recorder->state->terminal_status, status);
}

IREE_ATTRIBUTE_NOINLINE static loom_value_ordinal_t
loom_low_lower_representation_ordinal(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(recorder->state->value_domain,
                                          source_value_id);
  IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID,
                 "representation relation values must belong to the active "
                 "function domain");
  return value_ordinal;
}

void loom_low_lower_representation_record_union(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t left_value_id, loom_value_id_t right_value_id) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (!iree_status_is_ok(recorder->state->terminal_status)) return;
  const loom_value_ordinal_t left_ordinal =
      loom_low_lower_representation_ordinal(recorder, left_value_id);
  const loom_value_ordinal_t right_ordinal =
      loom_low_lower_representation_ordinal(recorder, right_value_id);
  iree_status_t status = loom_low_representation_plan_union(
      &recorder->state->plan, left_ordinal, right_ordinal);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
  }
}

void loom_low_lower_representation_record_candidates(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id,
    const loom_low_representation_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (!iree_status_is_ok(recorder->state->terminal_status)) return;
  IREE_ASSERT_ARGUMENT(candidates);
  IREE_ASSERT_GT(candidate_count, 0u);
  const loom_value_ordinal_t value_ordinal =
      loom_low_lower_representation_ordinal(recorder, source_value_id);
  iree_status_t status = loom_low_representation_plan_constrain(
      &recorder->state->plan, value_ordinal, candidates, candidate_count);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
  }
}

bool loom_low_lower_representation_component_is_constrained(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (!iree_status_is_ok(recorder->state->terminal_status)) return false;
  const loom_value_ordinal_t value_ordinal =
      loom_low_lower_representation_ordinal(recorder, source_value_id);
  return loom_low_representation_plan_component_is_constrained(
      &recorder->state->plan, value_ordinal);
}

static void loom_low_lower_representation_try_relation(
    loom_low_lower_representation_recorder_t* recorder,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_relation_t* relation) {
  if (relation->source_value_id == LOOM_VALUE_ID_INVALID ||
      relation->destination_value_id == LOOM_VALUE_ID_INVALID ||
      !iree_status_is_ok(recorder->state->terminal_status)) {
    return;
  }
  const loom_low_lower_representation_provider_t* provider =
      recorder->state->provider;
  if (provider->relation(provider->user_data, context, source_op, relation)) {
    loom_low_lower_representation_record_union(
        recorder, relation->source_value_id, relation->destination_value_id);
  }
}

static const loom_low_lower_representation_boundary_t*
loom_low_lower_representation_find_boundary(
    const loom_low_lower_representation_provider_t* provider,
    loom_op_kind_t op_kind) {
  if (provider->boundary_count == 0) return NULL;
  if (op_kind < provider->boundaries[0].op_kind) return NULL;
  const loom_low_lower_representation_boundary_t* last =
      &provider->boundaries[provider->boundary_count - 1];
  if (op_kind >= last->op_kind) return op_kind == last->op_kind ? last : NULL;
  uint16_t begin = 0;
  uint16_t end = provider->boundary_count - 1;
  while (begin < end) {
    const uint16_t middle = begin + (uint16_t)((end - begin) / 2);
    const loom_low_lower_representation_boundary_t* boundary =
        &provider->boundaries[middle];
    if (op_kind == boundary->op_kind) return boundary;
    if (op_kind < boundary->op_kind) {
      end = middle;
    } else {
      begin = middle + 1;
    }
  }
  return provider->boundaries[begin].op_kind == op_kind
             ? &provider->boundaries[begin]
             : NULL;
}

iree_status_t loom_low_lower_representation_observer_begin(
    void* user_data, loom_low_lower_context_t* context,
    void** out_observer_state) {
  IREE_ASSERT_ARGUMENT(out_observer_state);
  *out_observer_state = NULL;
  const loom_low_lower_representation_provider_t* provider =
      (const loom_low_lower_representation_provider_t*)user_data;
  IREE_ASSERT(
      provider != NULL &&
          (provider->relation_mask != 0) == (provider->relation != NULL) &&
          (provider->relation_mask & ~LOOM_VALUE_RELATION_MASK_ALL) == 0 &&
          (provider->boundary_count == 0 ||
           (provider->boundaries != NULL &&
            provider->observe_boundary != NULL)),
      "source representation provider must be internally valid");
  loom_low_lower_representation_observer_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &kLoomLowLowerRepresentationStateKey, sizeof(*state),
      (void**)&state));
  IREE_ASSERT(state->provider == NULL,
              "source representation observer must begin exactly once");
  const loom_local_value_domain_t* value_domain =
      loom_low_lower_context_value_domain(context);
  IREE_ASSERT(
      value_domain != NULL && loom_local_value_domain_is_acquired(value_domain),
      "source representation observation requires the value domain");
  *state = (loom_low_lower_representation_observer_state_t){
      .provider = provider,
      .value_domain = value_domain,
      .terminal_status = iree_ok_status(),
  };
  loom_low_representation_plan_initialize(
      value_domain->value_count, loom_low_lower_context_function_arena(context),
      &state->plan);
  *out_observer_state = state;
  loom_low_lower_representation_observer_observe(
      state, context, loom_low_lower_context_source_function(context).op);
  if (!iree_status_is_ok(state->terminal_status)) {
    *out_observer_state = NULL;
    return loom_low_lower_representation_take_status(state);
  }
  return iree_ok_status();
}

void loom_low_lower_representation_observer_observe(
    void* observer_state, loom_low_lower_context_t* context,
    const loom_op_t* source_op) {
  loom_low_lower_representation_observer_state_t* state =
      (loom_low_lower_representation_observer_state_t*)observer_state;
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT(state->provider != NULL);
  IREE_ASSERT(!state->plan.solved);
  loom_low_lower_representation_recorder_t recorder = {
      .state = state,
  };
  if (!iree_status_is_ok(state->terminal_status)) return;

  if (state->provider->relation_mask != 0) {
    loom_value_relation_iterator_t relation_iterator;
    loom_value_relation_iterator_initialize(
        loom_low_lower_context_module(context), source_op,
        state->provider->relation_mask, &relation_iterator);
    loom_value_relation_t relation;
    while (loom_value_relation_iterator_next(&relation_iterator, &relation)) {
      loom_low_lower_representation_try_relation(&recorder, context, source_op,
                                                 &relation);
    }
  }
  if (!iree_status_is_ok(state->terminal_status)) return;

  const loom_low_lower_representation_boundary_t* boundary =
      loom_low_lower_representation_find_boundary(state->provider,
                                                  source_op->kind);
  if (boundary != NULL) {
    IREE_ASSERT((boundary->flags &
                 ~LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL) == 0);
    state->provider->observe_boundary(state->provider->user_data,
                                      boundary->action, boundary->flags,
                                      context, source_op, &recorder);
  }
}

iree_status_t loom_low_lower_representation_observer_end(
    void* observer_state, loom_low_lower_context_t* context) {
  (void)context;
  loom_low_lower_representation_observer_state_t* state =
      (loom_low_lower_representation_observer_state_t*)observer_state;
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT(state->provider != NULL);
  IREE_ASSERT(!state->plan.solved);
  if (!iree_status_is_ok(state->terminal_status)) {
    return loom_low_lower_representation_take_status(state);
  }

  loom_low_representation_conflict_t plan_conflict;
  const bool exact =
      loom_low_representation_plan_solve(&state->plan, &plan_conflict);
  if (exact) return iree_ok_status();

  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "source value %u has incompatible physical representation domains",
      (unsigned)state->value_domain->value_ids[plan_conflict.value_ordinal]);
}

IREE_ATTRIBUTE_NOINLINE static void loom_low_lower_representation_state_lookup(
    loom_low_lower_representation_observer_state_t* state,
    loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation) {
  IREE_ASSERT(state->provider != NULL && state->plan.solved,
              "source representation lookup requires a solved plan");
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(state->value_domain, source_value_id);
  IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID,
                 "representation query values must belong to the active "
                 "function domain");
  loom_low_representation_plan_lookup(&state->plan, value_ordinal,
                                      out_representation);
}

iree_status_t loom_low_lower_representation_lookup(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation) {
  IREE_ASSERT_ARGUMENT(out_representation);
  *out_representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_observer_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &kLoomLowLowerRepresentationStateKey, sizeof(*state),
      (void**)&state));
  loom_low_lower_representation_state_lookup(state, source_value_id,
                                             out_representation);
  return iree_ok_status();
}

iree_status_t loom_low_lower_representation_query_lookup(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(out_representation);
  *out_representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_observer_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_target_contract_query_get_or_allocate_target_state(
      environment, &kLoomLowLowerRepresentationStateKey, sizeof(*state),
      (void**)&state));
  if (state == NULL || state->provider == NULL) return iree_ok_status();
  loom_low_lower_representation_state_lookup(state, source_value_id,
                                             out_representation);
  return iree_ok_status();
}
