// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_QUEUES_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_QUEUES_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fixed provisioned queues shared by independently retained model executions.
// Borrower counts describe live execution domains, not active requests, device
// occupancy, or priority. Binding never creates or destroys a native queue.
typedef struct loom_serve_execution_queues_t loom_serve_execution_queues_t;

// Captures the provisioned inventory of two selected, nonempty families on the
// same device. Retains that device; its live group must enclose every binding.
// A shared family contributes one inventory and one borrower per execution.
iree_status_t loom_serve_execution_queues_create(
    const iree_hal_queue_family_t* dispatch_family,
    const iree_hal_queue_family_t* transfer_family,
    iree_allocator_t host_allocator,
    loom_serve_execution_queues_t** out_queues);

// Reference ownership; NULL is accepted.
void loom_serve_execution_queues_retain(loom_serve_execution_queues_t* queues);
void loom_serve_execution_queues_release(loom_serve_execution_queues_t* queues);

typedef struct loom_serve_execution_binding_t {
  // Retained inventory owning both provisioned queues and borrower counters.
  loom_serve_execution_queues_t* owner;
  // Exact command queue, borrowed through binding release.
  iree_hal_queue_t* dispatch;
  // Exact transfer queue, borrowed through binding release.
  iree_hal_queue_t* transfer;
  // Borrowed queue's ordinal within the selected dispatch family.
  iree_hal_queue_ordinal_t dispatch_ordinal;
  // Borrowed queue's ordinal within the selected transfer family.
  iree_hal_queue_ordinal_t transfer_ordinal;
} loom_serve_execution_binding_t;

// Selects the least observed borrower count in each family, breaking ties by
// ordinal. The device owner serializes acquisitions; releases may occur on
// other threads. The same family supplies one exact queue for both roles.
// Cold model binding only: no allocation, native queue creation, or host wait.
loom_serve_execution_binding_t loom_serve_execution_queues_acquire(
    loom_serve_execution_queues_t* queues);

// Consumes one acquired binding after its final execution reference retires.
// May run on a completion callback. Counts are placement hints, not completion
// signals, and returning a binding neither waits nor destroys its queues.
void loom_serve_execution_binding_release(
    loom_serve_execution_binding_t binding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_QUEUES_H_
