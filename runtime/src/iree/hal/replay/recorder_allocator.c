// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_allocator.h"

#include <string.h>

#include "iree/hal/replay/recorder_record.h"

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_allocator_t {
  // HAL resource header for the recording wrapper allocator.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying allocator receiving forwarded HAL calls.
  iree_hal_allocator_t* base_allocator;
  // Session-local device object id associated with this allocator.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this allocator.
  iree_hal_replay_object_id_t allocator_id;
} iree_hal_replay_recorder_allocator_t;

static const iree_hal_allocator_vtable_t
    iree_hal_replay_recorder_allocator_vtable;

static iree_hal_replay_recorder_allocator_t*
iree_hal_replay_recorder_allocator_cast(iree_hal_allocator_t* base_allocator) {
  IREE_HAL_ASSERT_TYPE(base_allocator,
                       &iree_hal_replay_recorder_allocator_vtable);
  return (iree_hal_replay_recorder_allocator_t*)base_allocator;
}

static iree_hal_allocator_t* iree_hal_replay_recorder_allocator_base_or_self(
    iree_hal_allocator_t* allocator) {
  return iree_hal_resource_is(allocator,
                              &iree_hal_replay_recorder_allocator_vtable)
             ? iree_hal_replay_recorder_allocator_cast(allocator)
                   ->base_allocator
             : allocator;
}

static void iree_hal_replay_recorder_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(allocator->base_allocator);
  iree_hal_replay_recorder_release(allocator->recorder);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_replay_recorder_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  const iree_hal_replay_recorder_allocator_t* allocator =
      (const iree_hal_replay_recorder_allocator_t*)base_allocator;
  return iree_hal_allocator_host_allocator(allocator->base_allocator);
}

static void iree_hal_replay_recorder_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_allocator_query_statistics(allocator->base_allocator,
                                      out_statistics);
}

static iree_status_t iree_hal_replay_recorder_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_replay_recorder_allocator_t* allocator =
      iree_hal_replay_recorder_allocator_cast(base_allocator);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      allocator->recorder, allocator->device_id, allocator->allocator_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_QUERY_MEMORY_HEAPS,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_allocator_query_memory_heaps(allocator->base_allocator, capacity,
                                            heaps, out_count));
}

iree_status_t iree_hal_replay_recorder_wrap_allocator(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_allocator_t* base_allocator, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;

  iree_hal_replay_recorder_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                             (void**)&allocator));
  memset(allocator, 0, sizeof(*allocator));
  iree_hal_resource_initialize(&iree_hal_replay_recorder_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->recorder = recorder;
  iree_hal_replay_recorder_retain(allocator->recorder);
  allocator->base_allocator =
      iree_hal_replay_recorder_allocator_base_or_self(base_allocator);
  iree_hal_allocator_retain(allocator->base_allocator);
  allocator->device_id = device_id;

  iree_status_t status = iree_hal_replay_recorder_record_object(
      recorder, device_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, 0, NULL, &allocator->allocator_id);
  if (iree_status_is_ok(status)) {
    *out_allocator = (iree_hal_allocator_t*)allocator;
  } else {
    iree_hal_allocator_release(allocator->base_allocator);
    iree_hal_replay_recorder_release(allocator->recorder);
    iree_allocator_free(host_allocator, allocator);
  }
  return status;
}

static const iree_hal_allocator_vtable_t
    iree_hal_replay_recorder_allocator_vtable = {
        .destroy = iree_hal_replay_recorder_allocator_destroy,
        .host_allocator = iree_hal_replay_recorder_allocator_host_allocator,
        .query_statistics = iree_hal_replay_recorder_allocator_query_statistics,
        .query_memory_heaps =
            iree_hal_replay_recorder_allocator_query_memory_heaps,
};
