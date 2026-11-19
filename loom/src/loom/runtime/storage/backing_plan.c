// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/backing_plan.h"

#include "loom/util/adaptive_sort.h"

void loom_runtime_backing_plan_initialize(loom_runtime_backing_plan_t* plan) {
  *plan = (loom_runtime_backing_plan_t){.capacity = 1, .runs = &plan->first};
}

void loom_runtime_backing_plan_deinitialize(loom_runtime_backing_plan_t* plan,
                                            iree_allocator_t allocator) {
  if (plan->runs != &plan->first) {
    iree_allocator_free(allocator, plan->runs);
  }
}

iree_status_t loom_runtime_backing_plan_append(
    loom_runtime_backing_plan_t* plan, iree_allocator_t allocator,
    loom_runtime_backing_run_t run) {
  if (plan->count == plan->capacity) {
    // The current allocation already fits capacity elements. Doubling the
    // element count fits size_t because each element is larger than two bytes.
    const iree_host_size_t capacity = 2 * plan->capacity;
    if (plan->runs == &plan->first) {
      loom_runtime_backing_run_t* runs = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
          allocator, capacity, sizeof(*runs), (void**)&runs));
      runs[0] = plan->first;
      plan->runs = runs;
    } else {
      IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
          allocator, capacity, sizeof(*plan->runs), (void**)&plan->runs));
    }
    plan->capacity = capacity;
  }
  plan->runs[plan->count++] = run;
  return iree_ok_status();
}

static bool loom_runtime_backing_run_precedes(
    bool by_ordinal, const loom_runtime_backing_run_t* lhs,
    const loom_runtime_backing_run_t* rhs) {
  if (by_ordinal) {
    return lhs->ordinal < rhs->ordinal;
  }
  return lhs->buffer == rhs->buffer
             ? lhs->index < rhs->index
             : (uintptr_t)lhs->buffer < (uintptr_t)rhs->buffer;
}
LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_runtime_backing_run_sort,
                                       loom_runtime_backing_run_t, bool,
                                       loom_runtime_backing_run_precedes)

void loom_runtime_backing_plan_order_by_ordinal(
    loom_runtime_backing_plan_t* plan) {
  loom_runtime_backing_run_sort(true, plan->runs, plan->count);
}
void loom_runtime_backing_plan_order_by_target(
    loom_runtime_backing_plan_t* plan) {
  loom_runtime_backing_run_sort(false, plan->runs, plan->count);
}
const loom_runtime_backing_run_t* loom_runtime_backing_plan_find(
    const loom_runtime_backing_plan_t* plan, iree_host_size_t ordinal) {
  iree_host_size_t first = 0, end = plan->count;
  while (first + 1 < end) {
    const iree_host_size_t middle = first + (end - first) / 2;
    if (plan->runs[middle].ordinal <= ordinal) {
      first = middle;
    } else {
      end = middle;
    }
  }
  return &plan->runs[first];
}
