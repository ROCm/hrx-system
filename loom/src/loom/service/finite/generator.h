// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_FINITE_GENERATOR_H_
#define LOOM_SERVICE_FINITE_GENERATOR_H_

#include "iree/tokenizer/tokenizer.h"
#include "loom/runtime/parameter/residency.h"
#include "loom/service/execution_reconfiguration.h"
#include "loom/service/finite/result.h"
#include "loom/service/prepared_request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source policy and resources for serialized finite epochs. The service owns
// membership and receipts; the generator owns numerical state and progress.
typedef struct loom_service_finite_generator_vtable_t {
  // Immutable additive capacity of one compatible numerical cohort.
  uint64_t (*batch_limit)(void* self);
  // Optional immutable tokenizer, borrowed through model retirement.
  const iree_tokenizer_t* (*tokenizer)(void* self);
  // Current batch's parameter and transient admission identity.
  loom_runtime_residency_t* (*residency)(void* self);
  // Captured executable reconfiguration over the same source and state.
  loom_service_execution_update_t (*execution_update)(void* self);
  // Installs a wake-only observer on the serialized owner.
  void (*set_observer)(void* self, loom_runtime_wake_callback_t observer);
  // Reports source/JIT retirement independent of request interest.
  bool (*is_pending)(void* self);
  // Consumes asynchronous source/platform failures on the owner.
  iree_status_t (*query_status)(void* self);
  // CPU preparation creates one owned request state and its first work offer.
  // Failure creates no state. Equal tags and additive units permit batching.
  iree_status_t (*prepare_request)(void* self, iree_vm_buffer_t* input,
                                   loom_service_prepared_request_t* out_work);
  // Releases service ownership after its CPU borrowers retire. Accepted device
  // work retains its own source captures, including through failure retirement.
  void (*release_request)(void* self, loom_service_prepared_request_t* work);
  // Optional request-capacity admission before selection. A miss leaves this
  // request pending and permits other candidates, including continuations.
  // The source owner retains acquired capacity across epochs and releases it
  // with the request. NULL means no retained request-capacity requirement.
  iree_status_t (*try_admit_request)(void* self,
                                     loom_service_prepared_request_t* work,
                                     bool* out_acquired);
  // Required with try_admit_request: changes when previously denied capacity
  // can be acquired. Its observer wakes the owner after each such change.
  uint64_t (*capacity_generation)(void* self);
  // CPU preparation borrows stable original ordinals until release_batch or
  // complete. Output rows are either absent or the complete requested result.
  iree_status_t (*prepare_batch)(void* self, iree_host_size_t count,
                                 loom_service_prepared_request_t* work,
                                 const iree_host_size_t* expected_rows);
  // Owner-only asynchronous specialization; complete permits admission.
  iree_status_t (*advance_batch)(void* self, bool cancel, bool* out_complete);
  // Queues accepted work once. Results retain output and a dependency, not
  // request state, parameters, programs, or the generator itself.
  iree_status_t (*submit)(void* self, loom_service_finite_result_t** out_result,
                          loom_runtime_execution_frontier_t* out_frontier);
  // CPU collection after successful device readiness, in original ordinal
  // order including cancelled peers. Updates source-authoritative next work;
  // zero units means terminal and requires a complete published result. Empty
  // output alone never determines progress.
  // Every state remains owned until release_request; no payload download is
  // implied by collection. Failure retains accepted state for outer retirement.
  iree_status_t (*collect)(void* self, iree_host_size_t count,
                           loom_service_prepared_request_t* work,
                           const loom_service_finite_result_t* result);
  // Owner consumes successful collection and releases batch residency/captures.
  void (*complete)(void* self);
  // Releases an unsubmitted plan; never substitutes for failed-work retirement.
  void (*release_batch)(void* self);
} loom_service_finite_generator_vtable_t;

typedef struct loom_service_finite_generator_t {
  // Borrowed source owner, outliving the service and every accepted borrower.
  void* self;
  // Immutable source operations, outliving self.
  const loom_service_finite_generator_vtable_t* vtable;
} loom_service_finite_generator_t;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_FINITE_GENERATOR_H_
