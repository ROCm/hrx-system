// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_FINITE_BATCH_H_
#define LOOM_SERVICE_FINITE_BATCH_H_

#include "iree/hal/vm/types.h"
#include "iree/vm/environment.h"
#include "iree/vm/module.h"
#include "iree/vm/variant.h"
#include "loom/service/finite/result.h"
#include "loom/service/prepared_request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the process-lifetime finite.batch identity before source loading.
iree_status_t loom_service_finite_batch_register_types(
    iree_vm_environment_t* environment);
iree_vm_ref_type_t loom_service_finite_batch_type(void);

// One reusable, serialized cohort capability. The "finite" module exports:
//   batch_count(ptr<batch>) -> i64
//   batch_state(ptr<batch>, i64 ordinal) -> buffer
//   batch_pack(ptr<batch>) -> (buffer ranges, buffer packed)
//   batch_origin(ptr<batch>, i64 ordinal) -> i64
//   batch_buffer(ptr<batch>, i64 ordinal, i32 root) -> hal.buffer
//   batch_release(ptr<batch>, i64 ordinal)
//   batch_offer(ptr<batch>, i64 ordinal, i64 units, i64 tag)
//   batch_finish(ptr<batch>)
// State buffers retain their identity across epochs. batch_pack explicitly
// copies them into the ordinary upload directory; continuing sources can
// instead construct only the next epoch's control packet. batch_offer and
// batch_finish are valid only during successful collection. batch_release
// relinquishes a cohort interval without ending its logical request.
// batch_buffer is submission-only and retains that interval through actual
// queue ownership. batch_origin is in source units. Zero units finishes a
// member; positive units describe its next epoch, with equal tags permitting
// batching.
iree_status_t loom_service_finite_batch_module_create(
    iree_vm_environment_t* environment, const iree_hal_vm_types_t* hal_types,
    uint64_t batch_limit, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

typedef enum loom_service_finite_batch_phase_e {
  LOOM_SERVICE_FINITE_BATCH_PREPARE = 0,
  LOOM_SERVICE_FINITE_BATCH_SUBMIT,
  LOOM_SERVICE_FINITE_BATCH_COLLECT,
} loom_service_finite_batch_phase_t;

// Opens a borrowed member view for one synchronous source invocation. No
// allocation, state copy, or ownership transfer occurs. Original ordinals
// include cancelled members until the accepted cohort retires. Only collection
// can update offers; preparation borrows the states without changing them.
// result is NULL for preparation/submission and the accepted result for
// collection.
iree_vm_variant_t loom_service_finite_batch_begin(
    iree_vm_module_t* module, loom_service_finite_batch_phase_t phase,
    iree_host_size_t count, loom_service_prepared_request_t* requests,
    const loom_service_finite_result_t* result);

// Invalidates the view before member storage can be reused. On successful
// collection, checks that terminal offers agree with the published rows.
// Consumes the source invocation status, preserving its error on failure.
iree_status_t loom_service_finite_batch_end(iree_vm_module_t* module,
                                            iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_FINITE_BATCH_H_
