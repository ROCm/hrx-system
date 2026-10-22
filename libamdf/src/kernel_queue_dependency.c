// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/kernel_queue_dependency.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/kernel_queue.h"
#include "libamdf/src/structure.h"

amdf_status_t AMDF_CALL amdf_kernel_queue_dependency_create(
    amdf_kernel_queue_t* producer, amdf_kernel_queue_t* consumer,
    const amdf_kernel_queue_dependency_create_info_t* create_info,
    amdf_kernel_queue_dependency_t** out_dependency) {
  if (producer == NULL || consumer == NULL || out_dependency == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_CREATE_INFO,
      sizeof(*create_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (create_info->reserved != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((producer->info.capabilities & consumer->info.capabilities &
       AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_kernel_queue_native_progress_t* progress =
      producer->vtable->query_native_progress(producer);
  if (create_info->required_protocol !=
          AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_ANY &&
      create_info->required_protocol != progress->protocol) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_kernel_queue_dependency_t* dependency = NULL;
  status = amdf_calloc(consumer->host_allocator, sizeof(*dependency),
                       amdf_alignof(amdf_kernel_queue_dependency_t),
                       (void**)&dependency);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  dependency->host_allocator = consumer->host_allocator;
  dependency->producer = producer;
  dependency->consumer = consumer;
  amdf_atomic_uint64_initialize(&dependency->last_consumer_submission, 0);
  status = amdf_child_tracker_register(&producer->children);
  if (!amdf_status_is_ok(status)) {
    amdf_free(dependency->host_allocator, dependency);
    return status;
  }
  status = amdf_child_tracker_register(&consumer->children);
  const bool consumer_registered = amdf_status_is_ok(status);
  if (consumer_registered) {
    status = consumer->vtable->create_native_dependency(consumer, progress,
                                                        &dependency->native);
  }
  if (!amdf_status_is_ok(status)) {
    if (consumer_registered) {
      amdf_child_tracker_unregister(&consumer->children);
    }
    amdf_child_tracker_unregister(&producer->children);
    amdf_free(dependency->host_allocator, dependency);
    return status;
  }
  dependency->info = (amdf_kernel_queue_dependency_info_t){
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_INFO,
      .structure_size = sizeof(dependency->info),
      .protocol = progress->protocol,
      .producer_signal_operation_count = progress->signal_operation_count,
      .consumer_wait_operation_count = dependency->native->wait_operation_count,
  };
  *out_dependency = dependency;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_kernel_queue_dependency_query_info(
    amdf_kernel_queue_dependency_t* dependency,
    amdf_kernel_queue_dependency_info_t* out_info) {
  if (dependency == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_INFO,
      sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = dependency->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_kernel_queue_dependency_destroy(
    amdf_kernel_queue_dependency_t* dependency) {
  if (dependency == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_kernel_queue_status_t consumer_status = {0};
  dependency->consumer->vtable->query_status(dependency->consumer,
                                             &consumer_status);
  if (consumer_status.retired_submission <
      amdf_atomic_uint64_load_acquire(&dependency->last_consumer_submission)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = dependency->native->destroy(dependency->native);
  amdf_child_tracker_unregister(&dependency->consumer->children);
  amdf_child_tracker_unregister(&dependency->producer->children);
  amdf_free(dependency->host_allocator, dependency);
  return status;
}

amdf_status_t amdf_kernel_queue_dependency_validate_waits(
    amdf_kernel_queue_t* consumer, uint32_t wait_count,
    const amdf_kernel_queue_dependency_wait_t* waits) {
  if (wait_count != 0 && waits == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < wait_count && amdf_status_is_ok(status); ++i) {
    const amdf_kernel_queue_dependency_wait_t* wait = &waits[i];
    if (wait->dependency == NULL || wait->dependency->consumer != consumer ||
        wait->producer_submission == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    } else {
      amdf_kernel_queue_t* producer = wait->dependency->producer;
      if (wait->producer_submission >
          producer->vtable->query_last_submission(producer)) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
      } else {
        amdf_kernel_queue_status_t producer_status = {0};
        producer->vtable->query_status(producer, &producer_status);
        if (producer_status.retired_submission < wait->producer_submission) {
          status = producer_status.terminal_status;
        }
      }
    }
  }
  return status;
}

void amdf_kernel_queue_dependency_record_waits(
    uint32_t accepted_wait_count,
    const amdf_kernel_queue_dependency_wait_t* waits,
    uint64_t consumer_submission) {
  for (uint32_t i = 0; i < accepted_wait_count; ++i) {
    amdf_atomic_uint64_store_release(
        &waits[i].dependency->last_consumer_submission, consumer_submission);
  }
}
