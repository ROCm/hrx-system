// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/scheduler.h"

void loom_serve_execution_scheduler_initialize(
    uint32_t concurrency, loom_serve_execution_scheduler_t* out_scheduler) {
  *out_scheduler =
      (loom_serve_execution_scheduler_t){.concurrency = concurrency};
}

void loom_serve_execution_permit_initialize(
    loom_serve_execution_scheduler_t* scheduler,
    loom_serve_execution_permit_t* out_permit) {
  *out_permit = (loom_serve_execution_permit_t){.scheduler = scheduler};
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
                                         uint32_t weight) {
  permit->weight = weight;
  loom_serve_execution_scheduler_t* scheduler = permit->scheduler;
  if (!scheduler || !scheduler->concurrency) {
    permit->phase = LOOM_SERVE_EXECUTION_PERMIT_GRANTED;
    return;
  }
  permit->service_time = iree_max(permit->service_time, scheduler->minimum);
  permit->phase = LOOM_SERVE_EXECUTION_PERMIT_WAITING;
  permit->previous = scheduler->tail;
  if (scheduler->tail) {
    scheduler->tail->next = permit;
  } else {
    scheduler->head = permit;
  }
  scheduler->tail = permit;
}

static uint64_t loom_serve_execution_permit_accounted_service(
    const loom_serve_execution_permit_t* permit, iree_time_t now) {
  const uint64_t elapsed = (uint64_t)(now - permit->start_time);
  // Charge at least one nanosecond even within the clock's resolution. The
  // same observation rule applies while accepted and at numerical completion.
  const uint64_t charge =
      elapsed / permit->weight + (elapsed % permit->weight != 0);
  return permit->service_time + iree_max(charge, UINT64_C(1));
}

bool loom_serve_execution_scheduler_advance(
    loom_serve_execution_scheduler_t* scheduler, iree_time_t now) {
  if (!scheduler->concurrency ||
      scheduler->outstanding_count == scheduler->concurrency) {
    return false;
  }
  loom_serve_execution_permit_t* selected = NULL;
  uint64_t minimum = UINT64_MAX;
  uint64_t active_minimum = UINT64_MAX;
  for (loom_serve_execution_permit_t* permit = scheduler->head; permit;
       permit = permit->next) {
    const uint64_t service_time =
        permit->phase == LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED
            ? loom_serve_execution_permit_accounted_service(permit, now)
            : permit->service_time;
    minimum = iree_min(minimum, service_time);
    if (permit->phase != LOOM_SERVE_EXECUTION_PERMIT_WAITING) {
      active_minimum = iree_min(active_minimum, service_time);
    }
    if (permit->phase == LOOM_SERVE_EXECUTION_PERMIT_WAITING &&
        (!selected || permit->service_time < selected->service_time)) {
      selected = permit;
    }
  }
  if (!selected) {
    return false;
  }
  scheduler->minimum = iree_max(scheduler->minimum, minimum);
  // Spare capacity does not entitle a more-served producer to interfere with
  // accepted work. Peer completion or another owner wake reevaluates
  // eligibility; without granted or accepted peers, ready work always advances.
  if (selected->service_time > active_minimum) {
    return false;
  }
  selected->phase = LOOM_SERVE_EXECUTION_PERMIT_GRANTED;
  ++scheduler->outstanding_count;
  return true;
}

bool loom_serve_execution_permit_try_acquire(
    loom_serve_execution_permit_t* permit, iree_time_t now) {
  if (permit->phase != LOOM_SERVE_EXECUTION_PERMIT_GRANTED) {
    return false;
  }
  permit->phase = LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED;
  permit->start_time = now;
  return true;
}

void loom_serve_execution_permit_complete(loom_serve_execution_permit_t* permit,
                                          iree_time_t now) {
  loom_serve_execution_scheduler_t* scheduler = permit->scheduler;
  if (scheduler && scheduler->concurrency) {
    permit->service_time =
        loom_serve_execution_permit_accounted_service(permit, now);
    loom_serve_execution_permit_unlink(permit);
    --scheduler->outstanding_count;
    uint64_t minimum = permit->service_time;
    for (loom_serve_execution_permit_t* live = scheduler->head; live;
         live = live->next) {
      minimum = iree_min(minimum, live->service_time);
    }
    scheduler->minimum = iree_max(scheduler->minimum, minimum);
  }
  permit->phase = LOOM_SERVE_EXECUTION_PERMIT_IDLE;
}

static void loom_serve_execution_permit_withdraw(
    loom_serve_execution_permit_t* permit) {
  loom_serve_execution_scheduler_t* scheduler = permit->scheduler;
  if (scheduler && scheduler->concurrency &&
      permit->phase != LOOM_SERVE_EXECUTION_PERMIT_IDLE) {
    loom_serve_execution_permit_unlink(permit);
    if (permit->phase != LOOM_SERVE_EXECUTION_PERMIT_WAITING) {
      --scheduler->outstanding_count;
    }
  }
  permit->phase = LOOM_SERVE_EXECUTION_PERMIT_IDLE;
}

void loom_serve_execution_permit_cancel(loom_serve_execution_permit_t* permit) {
  // Cancelling caller interest never revokes an already accepted model loan.
  IREE_ASSERT(permit->phase != LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED);
  loom_serve_execution_permit_withdraw(permit);
}

void loom_serve_execution_permit_deinitialize(
    loom_serve_execution_permit_t* permit) {
  loom_serve_execution_permit_withdraw(permit);
  permit->scheduler = NULL;
}
