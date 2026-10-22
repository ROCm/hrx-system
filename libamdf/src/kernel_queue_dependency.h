// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_KERNEL_QUEUE_DEPENDENCY_H_
#define AMDF_SRC_KERNEL_QUEUE_DEPENDENCY_H_

#include "amdf/amdf.h"
#include "libamdf/src/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable, producer-owned native progress identity. A protocol-specific
// representation embeds this prefix and supplies its native transport facts.
typedef struct amdf_kernel_queue_native_progress_t {
  // Native protocol qualifying the representation that follows this prefix.
  amdf_kernel_queue_dependency_protocol_t protocol;
  // Extra native signals per producer point, shared by every consumer.
  uint32_t signal_operation_count;
} amdf_kernel_queue_native_progress_t;

// Consumer-owned native view imported during cold binding construction.
typedef struct amdf_kernel_queue_native_dependency_t {
  // Consumes an idle native view and its metadata, including on cleanup error.
  amdf_status_t (*destroy)(
      struct amdf_kernel_queue_native_dependency_t* native);
  // Native wait operations needed for one validated dependency operand.
  uint32_t wait_operation_count;
} amdf_kernel_queue_native_dependency_t;

struct amdf_kernel_queue_dependency_t {
  // Allocator copied for terminal teardown after releasing both queue borrows.
  amdf_allocator_t host_allocator;
  // Producer borrowed until this binding is destroyed.
  amdf_kernel_queue_t* producer;
  // Consumer borrowed until this binding is destroyed.
  amdf_kernel_queue_t* consumer;
  // Native view whose exact representation belongs to the consumer provider.
  amdf_kernel_queue_native_dependency_t* native;
  // Immutable native mechanism and publication costs.
  amdf_kernel_queue_dependency_info_t info;
  // Last accepted use, written by the consumer's exclusive publication owner.
  amdf_atomic_uint64_t last_consumer_submission;
};

amdf_status_t AMDF_CALL amdf_kernel_queue_dependency_create(
    amdf_kernel_queue_t* producer, amdf_kernel_queue_t* consumer,
    const amdf_kernel_queue_dependency_create_info_t* create_info,
    amdf_kernel_queue_dependency_t** out_dependency);

amdf_status_t AMDF_CALL amdf_kernel_queue_dependency_query_info(
    amdf_kernel_queue_dependency_t* dependency,
    amdf_kernel_queue_dependency_info_t* out_info);

amdf_status_t AMDF_CALL amdf_kernel_queue_dependency_destroy(
    amdf_kernel_queue_dependency_t* dependency);

// Public submission boundary: validates every borrowed operand before native
// publication. Reads the producer's accepted-prefix fact without native work.
amdf_status_t amdf_kernel_queue_dependency_validate_waits(
    amdf_kernel_queue_t* consumer, uint32_t wait_count,
    const amdf_kernel_queue_dependency_wait_t* waits);

// Records only the native-accepted wait prefix under the consumer's existing
// publication claim. Does not allocate or retain a per-submission owner list.
void amdf_kernel_queue_dependency_record_waits(
    uint32_t accepted_wait_count,
    const amdf_kernel_queue_dependency_wait_t* waits,
    uint64_t consumer_submission);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_KERNEL_QUEUE_DEPENDENCY_H_
