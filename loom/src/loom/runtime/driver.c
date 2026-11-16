// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/driver.h"

void loom_runtime_driver_initialize(iree_host_size_t capacity,
                                    const loom_runtime_service_t* services,
                                    loom_runtime_driver_t* out_driver) {
  *out_driver = (loom_runtime_driver_t){
      .services = services,
      .capacity = capacity,
  };
}

void loom_runtime_driver_set_observer(loom_runtime_driver_t* driver,
                                      loom_runtime_wake_callback_t observer) {
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const loom_runtime_service_t* service = &driver->services[i];
    if (service->vtable) {
      service->vtable->set_observer(service->self, observer);
    }
  }
}

loom_runtime_service_flags_t loom_runtime_driver_shared_activity(
    const loom_runtime_driver_t* driver, iree_host_size_t selected) {
  const void* capacity_domain = driver->services[selected].capacity_domain;
  if (!capacity_domain) {
    return 0;
  }
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const loom_runtime_service_t* service = &driver->services[i];
    if (i != selected && service->vtable &&
        service->capacity_domain == capacity_domain &&
        service->vtable->is_active(service->self)) {
      return LOOM_RUNTIME_SERVICE_FLAG_SHARED_ACTIVITY;
    }
  }
  return 0;
}

void loom_runtime_driver_advance(loom_runtime_driver_t* driver,
                                 bool* out_progress) {
  *out_progress = false;
  driver->stopping |= !iree_status_is_ok(driver->status);
  driver->stopping_applied = driver->stopping;
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const iree_host_size_t index = (driver->cursor + i) % driver->capacity;
    const loom_runtime_service_t* service = &driver->services[index];
    if (!service->vtable) {
      continue;
    }
    const loom_runtime_service_flags_t flags =
        loom_runtime_driver_shared_activity(driver, index) |
        (driver->stopping ? LOOM_RUNTIME_SERVICE_FLAG_STOPPING : 0);
    bool progress = false;
    iree_status_t status =
        service->vtable->advance(service->self, flags, &progress);
    if (!iree_status_is_ok(status)) {
      // A first failure schedules a stopping turn. Repeated terminal failure
      // is not progress while accepted borrowers await their retirement wake.
      progress |= iree_status_is_ok(driver->status);
      driver->status = iree_status_join(driver->status, status);
      driver->stopping = true;
    }
    *out_progress |= progress;
  }
  if (++driver->cursor >= driver->capacity) {
    driver->cursor = 0;
  }
}

iree_status_t loom_runtime_driver_try_schedule(
    loom_runtime_driver_t* driver,
    const loom_runtime_service_decision_t* decision,
    loom_runtime_service_schedule_outcome_t* out_outcome) {
  *out_outcome = (loom_runtime_service_schedule_outcome_t){0};
  if (driver->stopping || !iree_status_is_ok(driver->status) ||
      decision->service_ordinal >= driver->capacity) {
    return iree_ok_status();
  }
  const loom_runtime_service_t* service =
      &driver->services[decision->service_ordinal];
  if (!service->vtable || !service->control ||
      loom_runtime_service_control_layout(service->control).incarnation !=
          decision->service_incarnation) {
    return iree_ok_status();
  }
  return service->vtable->try_schedule(
      service->self, decision->offer, decision->choice_ordinal,
      loom_runtime_driver_shared_activity(driver, decision->service_ordinal),
      out_outcome);
}

void loom_runtime_driver_apply_plan(loom_runtime_driver_t* driver,
                                    loom_runtime_service_plan_t* plan,
                                    uint64_t* inout_decisions,
                                    bool* out_progress) {
  *out_progress = false;
  const loom_runtime_service_plan_layout_t layout =
      loom_runtime_service_plan_layout(plan);
  uint64_t ordinal = layout.applied_count;
  while (ordinal < layout.decision_count && *inout_decisions) {
    --*inout_decisions;
    const loom_runtime_service_decision_t decision =
        loom_runtime_service_plan_decision(plan, ordinal);
    loom_runtime_service_schedule_outcome_t outcome;
    iree_status_t status =
        loom_runtime_driver_try_schedule(driver, &decision, &outcome);
    loom_runtime_service_plan_complete(plan, outcome.status_code,
                                       outcome.operation);
    ++ordinal;
    *out_progress |= outcome.accepted;
    if (!iree_status_is_ok(status)) {
      *out_progress |= iree_status_is_ok(driver->status);
      driver->status = iree_status_join(driver->status, status);
      driver->stopping = true;
    }
  }
  *out_progress |= ordinal < layout.decision_count;
}

static void loom_runtime_service_default_select(
    void* self, const loom_runtime_driver_t* driver,
    loom_runtime_service_schedule_context_t* context,
    loom_runtime_service_plan_t* plan) {
  loom_runtime_service_default_policy_t* state = self;
  loom_runtime_service_plan_reset(plan);
  if (!driver->capacity) {
    return;
  }
  state->service_ordinal %= driver->capacity;
  const iree_host_size_t first_service = state->service_ordinal;
  const loom_runtime_service_t* first = &driver->services[first_service];
  uint64_t first_offer = 0;
  if (first->vtable && first->control) {
    const uint64_t capacity =
        loom_runtime_service_control_layout(first->control).offer_capacity;
    if (state->offer_ordinal < capacity) {
      first_offer = state->offer_ordinal;
    }
  }
  // Complete one traversal of the coordinates, including the first service's
  // prefix last. Wrapping its offers before visiting peers could starve them
  // when that service continuously fills the output buffer.
  for (iree_host_size_t i = 0;
       i < driver->capacity || (i == driver->capacity && first_offer); ++i) {
    const iree_host_size_t service_ordinal =
        (first_service + i) % driver->capacity;
    const loom_runtime_service_t* service = &driver->services[service_ordinal];
    if (service->vtable && service->control) {
      const loom_runtime_service_control_layout_t layout =
          loom_runtime_service_control_layout(service->control);
      const uint64_t begin = i == 0 ? first_offer : 0;
      const uint64_t end =
          i == driver->capacity ? first_offer : layout.offer_capacity;
      for (uint64_t ordinal = begin; ordinal < end; ++ordinal) {
        loom_runtime_service_offer_t offer;
        if (loom_runtime_service_read_offer(service->control, ordinal,
                                            &offer) !=
                LOOM_RUNTIME_SERVICE_READ_COMPLETE ||
            !offer.choice_count) {
          continue;
        }
        const loom_runtime_service_decision_t decision = {
            .service_ordinal = service_ordinal,
            .service_incarnation = layout.incarnation,
            .offer = offer.key};
        if (!loom_runtime_service_plan_append(plan, &decision)) {
          state->service_ordinal = service_ordinal;
          state->offer_ordinal = ordinal;
          return;
        }
      }
    }
  }
  state->service_ordinal = (first_service + 1) % driver->capacity;
  state->offer_ordinal = 0;
}

loom_runtime_service_policy_t loom_runtime_service_default_policy(
    loom_runtime_service_default_policy_t* state) {
  return (loom_runtime_service_policy_t){
      .self = state, .select = loom_runtime_service_default_select};
}

void loom_runtime_driver_schedule(
    loom_runtime_driver_t* driver, loom_runtime_service_policy_t policy,
    loom_runtime_service_plan_t* plan,
    loom_runtime_service_schedule_budget_t* budget,
    loom_runtime_service_schedule_context_t* context, bool* out_progress) {
  *out_progress = false;
  context->wake_time_ns = IREE_TIME_INFINITE_FUTURE;
  if (driver->stopping || !iree_status_is_ok(driver->status)) {
    return;
  }
  const bool may_evaluate = budget->evaluations != 0;
  for (;;) {
    loom_runtime_service_plan_layout_t layout =
        loom_runtime_service_plan_layout(plan);
    if (layout.applied_count == layout.decision_count) {
      if (!budget->evaluations) {
        if (may_evaluate) {
          context->wake_time_ns = context->now_ns;
        }
        break;
      }
      --budget->evaluations;
      loom_runtime_service_schedule_context_t invocation = {
          .now_ns = context->now_ns, .wake_time_ns = IREE_TIME_INFINITE_FUTURE};
      policy.select(policy.self, driver, &invocation, plan);
      context->wake_time_ns =
          iree_min(context->wake_time_ns, invocation.wake_time_ns);
      layout = loom_runtime_service_plan_layout(plan);
      if (!layout.decision_count) {
        break;
      }
    }
    // Withheld application credit preserves the plan without manufacturing
    // progress. Consuming granted credit below instead yields another turn.
    if (!budget->decisions) {
      break;
    }
    bool progress = false;
    loom_runtime_driver_apply_plan(driver, plan, &budget->decisions, &progress);
    *out_progress |= progress;
    if (driver->stopping) {
      break;
    }
    if (!budget->decisions) {
      context->wake_time_ns = context->now_ns;
      break;
    }
  }
}

bool loom_runtime_driver_collect(loom_runtime_driver_t* driver,
                                 bool* out_progress) {
  driver->stopping |= !iree_status_is_ok(driver->status);
  bool pending =
      driver->capacity && driver->stopping && !driver->stopping_applied;
  *out_progress |= pending;
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const loom_runtime_service_t* service = &driver->services[i];
    pending |= service->vtable && service->vtable->is_pending(service->self);
  }
  return pending;
}
