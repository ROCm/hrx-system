// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/scheduler.h"

#include "iree/base/internal/math.h"

typedef struct loom_serve_execution_resource_t {
  // Zero adds no software admission limit; positive values count accepted
  // units.
  uint32_t max_active;
  // Accepted units by request class, independent of model registration count.
  uint32_t active[2];
  // Resolved union of exclusion scopes touching this resource.
  loom_serve_execution_resources_t exclusions;
} loom_serve_execution_resource_t;

struct loom_serve_execution_scheduler_t {
  // Allocation owner for the cold resource table and controller.
  iree_allocator_t host_allocator;
  // First live observation; observations borrow only serialized-owner state.
  loom_serve_execution_source_t* sources;
  // First live permit, preserving offer order independently of router rotation.
  loom_serve_execution_permit_t* head;
  // Last live permit; NULL exactly when head is NULL.
  loom_serve_execution_permit_t* tail;
  // Immutable policy and owner-mutated accepted accounting for each resource.
  loom_serve_execution_resource_t resources[];
};

iree_status_t loom_serve_execution_scheduler_allocate(
    iree_host_size_t resource_count,
    const loom_serve_execution_policy_t* policy,
    iree_allocator_t host_allocator,
    loom_serve_execution_scheduler_t** out_scheduler) {
  *out_scheduler = NULL;
  if (!resource_count || resource_count > 64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "execution policy needs between 1 and 64 resources");
  }
  const uint64_t all_resources = UINT64_MAX >> (64 - resource_count);
  if (policy) {
    uint64_t covered = 0;
    for (iree_host_size_t i = 0; i < policy->scope_count; ++i) {
      const loom_serve_execution_scope_t* scope = &policy->scopes[i];
      if (!scope->resources || (scope->resources & ~all_resources) ||
          (scope->coexistence != LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE &&
           scope->coexistence != LOOM_SERVE_EXECUTION_COEXISTENCE_OVERLAP)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid execution interference scope %" PRIhsz,
                                i);
      }
      covered |= scope->resources;
    }
    if (covered != all_resources) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "execution policy must cover every resource");
    }
  }
  loom_serve_execution_scheduler_t* scheduler = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      sizeof(*scheduler) + resource_count * sizeof(scheduler->resources[0]),
      (void**)&scheduler));
  scheduler->host_allocator = host_allocator;
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    loom_serve_execution_resource_t* resource = &scheduler->resources[i];
    if (!policy) {
      resource->exclusions = all_resources;
      continue;
    }
    resource->max_active = policy->max_active ? policy->max_active[i] : 0;
    for (iree_host_size_t j = 0; j < policy->scope_count; ++j) {
      const loom_serve_execution_scope_t* scope = &policy->scopes[j];
      if ((scope->resources & (UINT64_C(1) << i)) &&
          scope->coexistence == LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE) {
        resource->exclusions |= scope->resources;
      }
    }
  }
  *out_scheduler = scheduler;
  return iree_ok_status();
}

void loom_serve_execution_scheduler_free(
    loom_serve_execution_scheduler_t* scheduler) {
  if (scheduler) {
    iree_allocator_free(scheduler->host_allocator, scheduler);
  }
}

void loom_serve_execution_source_initialize(
    loom_serve_execution_scheduler_t* scheduler,
    loom_serve_execution_demand_fn_t query, void* self,
    loom_serve_execution_source_t* out_source) {
  *out_source = (loom_serve_execution_source_t){
      .scheduler = scheduler, .query = query, .self = self};
  if (scheduler) {
    out_source->next = scheduler->sources;
    if (scheduler->sources) {
      scheduler->sources->previous = out_source;
    }
    scheduler->sources = out_source;
  }
}

void loom_serve_execution_source_deinitialize(
    loom_serve_execution_source_t* source) {
  if (!source->scheduler) {
    return;
  }
  if (source->previous) {
    source->previous->next = source->next;
  } else {
    source->scheduler->sources = source->next;
  }
  if (source->next) {
    source->next->previous = source->previous;
  }
  *source = (loom_serve_execution_source_t){0};
}

void loom_serve_execution_permit_initialize(
    loom_serve_execution_scheduler_t* scheduler,
    loom_serve_execution_resources_t resources,
    loom_serve_execution_permit_t* out_permit) {
  *out_permit = (loom_serve_execution_permit_t){.scheduler = scheduler,
                                                .resources = resources};
}

static void loom_serve_execution_permit_unlink(
    loom_serve_execution_permit_t* permit) {
  loom_serve_execution_scheduler_t* scheduler = permit->scheduler;
  if (permit->previous) {
    permit->previous->next = permit->next;
  } else {
    scheduler->head = permit->next;
  }
  if (permit->next) {
    permit->next->previous = permit->previous;
  } else {
    scheduler->tail = permit->previous;
  }
  permit->previous = NULL;
  permit->next = NULL;
}

void loom_serve_execution_permit_request(loom_serve_execution_permit_t* permit,
                                         loom_serve_execution_qos_t qos) {
  if (!permit) {
    return;
  }
  permit->qos = qos;
  permit->phase = LOOM_SERVE_EXECUTION_PERMIT_WAITING;
  loom_serve_execution_scheduler_t* scheduler = permit->scheduler;
  if (scheduler) {
    permit->previous = scheduler->tail;
    if (scheduler->tail) {
      scheduler->tail->next = permit;
    } else {
      scheduler->head = permit;
    }
    scheduler->tail = permit;
  }
}

static uint64_t loom_serve_execution_realtime_demand(
    const loom_serve_execution_scheduler_t* scheduler) {
  uint64_t resources = 0;
  for (const loom_serve_execution_source_t* source = scheduler->sources; source;
       source = source->next) {
    resources |= source->query(source->self);
  }
  for (const loom_serve_execution_permit_t* permit = scheduler->head; permit;
       permit = permit->next) {
    if (permit->qos == LOOM_SERVE_EXECUTION_QOS_REALTIME) {
      resources |= permit->resources;
    }
  }
  return resources;
}

static uint64_t loom_serve_execution_exclusions(
    const loom_serve_execution_scheduler_t* scheduler, uint64_t resources) {
  uint64_t exclusions = 0;
  while (resources) {
    const uint32_t index = iree_math_count_trailing_zeros_u64(resources);
    resources &= resources - 1;
    exclusions |= scheduler->resources[index].exclusions;
  }
  return exclusions;
}

bool loom_serve_execution_is_preempted(
    loom_serve_execution_scheduler_t* scheduler,
    loom_serve_execution_resources_t resources,
    loom_serve_execution_qos_t qos) {
  return scheduler && qos == LOOM_SERVE_EXECUTION_QOS_BACKGROUND &&
         (loom_serve_execution_exclusions(scheduler, resources) &
          loom_serve_execution_realtime_demand(scheduler));
}

static bool loom_serve_execution_permit_can_acquire(
    const loom_serve_execution_permit_t* permit, uint64_t realtime_demand) {
  const loom_serve_execution_scheduler_t* scheduler = permit->scheduler;
  const uint64_t exclusions =
      loom_serve_execution_exclusions(scheduler, permit->resources);
  if (permit->qos == LOOM_SERVE_EXECUTION_QOS_BACKGROUND &&
      (exclusions & realtime_demand)) {
    return false;
  }
  uint64_t resources = exclusions;
  const uint32_t peer_class = 1 - permit->qos;
  while (resources) {
    const uint32_t index = iree_math_count_trailing_zeros_u64(resources);
    resources &= resources - 1;
    if (scheduler->resources[index].active[peer_class]) {
      return false;
    }
  }
  resources = permit->resources;
  while (resources) {
    const uint32_t index = iree_math_count_trailing_zeros_u64(resources);
    resources &= resources - 1;
    const loom_serve_execution_resource_t* resource =
        &scheduler->resources[index];
    if (!resource->max_active) {
      continue;
    }
    const uint64_t active = (uint64_t)resource->active[0] + resource->active[1];
    if (active >= resource->max_active) {
      return false;
    }
    // Preserve one slot for older same-class work that needs this resource and
    // others together. Residual capacity can still serve compatible later work.
    for (const loom_serve_execution_permit_t* peer = scheduler->head;
         peer != permit; peer = peer->next) {
      if (peer->phase == LOOM_SERVE_EXECUTION_PERMIT_WAITING &&
          peer->qos == permit->qos &&
          (peer->resources & (UINT64_C(1) << index)) &&
          active + 1 == resource->max_active) {
        return false;
      }
    }
  }
  // In a shared scope, ready realtime work gets the next contested slot.
  if (permit->qos == LOOM_SERVE_EXECUTION_QOS_BACKGROUND) {
    for (const loom_serve_execution_permit_t* peer = scheduler->head; peer;
         peer = peer->next) {
      if (peer->phase == LOOM_SERVE_EXECUTION_PERMIT_WAITING &&
          peer->qos == LOOM_SERVE_EXECUTION_QOS_REALTIME &&
          (peer->resources & permit->resources) &&
          loom_serve_execution_permit_can_acquire(peer, realtime_demand)) {
        return false;
      }
    }
  }
  return true;
}

bool loom_serve_execution_scheduler_is_ready(
    loom_serve_execution_scheduler_t* scheduler) {
  const uint64_t demand = loom_serve_execution_realtime_demand(scheduler);
  for (const loom_serve_execution_permit_t* permit = scheduler->head; permit;
       permit = permit->next) {
    if (permit->phase == LOOM_SERVE_EXECUTION_PERMIT_WAITING &&
        loom_serve_execution_permit_can_acquire(permit, demand)) {
      return true;
    }
  }
  return false;
}

bool loom_serve_execution_permit_try_acquire(
    loom_serve_execution_permit_t* permit) {
  if (!permit) {
    return true;
  }
  if (permit->phase != LOOM_SERVE_EXECUTION_PERMIT_WAITING) {
    return false;
  }
  loom_serve_execution_scheduler_t* scheduler = permit->scheduler;
  if (scheduler) {
    if (!loom_serve_execution_permit_can_acquire(
            permit, loom_serve_execution_realtime_demand(scheduler))) {
      return false;
    }
    uint64_t resources = permit->resources;
    while (resources) {
      const uint32_t index = iree_math_count_trailing_zeros_u64(resources);
      resources &= resources - 1;
      ++scheduler->resources[index].active[permit->qos];
    }
  }
  permit->phase = LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED;
  return true;
}

static void loom_serve_execution_permit_withdraw(
    loom_serve_execution_permit_t* permit) {
  loom_serve_execution_scheduler_t* scheduler = permit->scheduler;
  if (scheduler && permit->phase != LOOM_SERVE_EXECUTION_PERMIT_IDLE) {
    loom_serve_execution_permit_unlink(permit);
    if (permit->phase == LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED) {
      uint64_t resources = permit->resources;
      while (resources) {
        const uint32_t index = iree_math_count_trailing_zeros_u64(resources);
        resources &= resources - 1;
        --scheduler->resources[index].active[permit->qos];
      }
    }
  }
  permit->phase = LOOM_SERVE_EXECUTION_PERMIT_IDLE;
}

void loom_serve_execution_permit_complete(
    loom_serve_execution_permit_t* permit) {
  if (permit) {
    loom_serve_execution_permit_withdraw(permit);
  }
}

void loom_serve_execution_permit_cancel(loom_serve_execution_permit_t* permit) {
  if (permit && permit->phase != LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED) {
    loom_serve_execution_permit_withdraw(permit);
  }
}

void loom_serve_execution_permit_deinitialize(
    loom_serve_execution_permit_t* permit) {
  if (permit) {
    loom_serve_execution_permit_withdraw(permit);
    permit->scheduler = NULL;
  }
}
