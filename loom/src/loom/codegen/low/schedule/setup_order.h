// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Static dependency ordering for exclusive storage and repair materializations.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_SETUP_ORDER_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_SETUP_ORDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_schedule_setup_order_t {
  // Whether node classification found setup requiring dependency ordering.
  bool has_members;
  // During graph construction, the sole complete successor, UINT32_MAX for
  // none, or the producer itself for fan-out. Finish replaces these facts
  // with transitive setup completion nodes. NULL disables setup ordering.
  uint32_t* completion_nodes;
  // First setup node for each completion, populated by finish.
  uint32_t* entry_nodes;
} loom_low_schedule_setup_order_t;

// Allocates graph scratch after membership classification.
iree_status_t loom_low_schedule_setup_order_initialize(
    uint32_t node_count, iree_arena_allocator_t* arena,
    loom_low_schedule_setup_order_t* out_order);

// Records complete dependency fan-out at its producer, including non-SSA edges.
static inline void loom_low_schedule_setup_order_record_dependency(
    loom_low_schedule_setup_order_t* order, uint32_t producer,
    uint32_t consumer) {
  if (order->completion_nodes == NULL) {
    return;
  }
  uint32_t* completion = &order->completion_nodes[producer];
  if (*completion == UINT32_MAX) {
    *completion = consumer;
  } else if (*completion != consumer) {
    *completion = producer;
  }
}

struct loom_low_schedule_build_state_t;
struct loom_low_schedule_node_t;

// Classifies exclusive storage setup from the retained pressure domains.
// Repair also orders structural setup and data-dependent rematerializations.
// The existing node walk records membership before dependency construction.
void loom_low_schedule_setup_order_classify_node(
    struct loom_low_schedule_build_state_t* state,
    struct loom_low_schedule_node_t* node, bool is_repair);

// Closes setup dependencies before adjacency construction. Every member has
// one complete successor and remains inside its source-order range, so its
// external prerequisites can precede the entire chain without creating a cycle.
iree_status_t loom_low_schedule_setup_order_finish(
    struct loom_low_schedule_build_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_SETUP_ORDER_H_
