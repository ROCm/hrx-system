// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Stream implementation. Each stream owns a timeline semaphore and a pending
// command buffer. Operations accumulate in the CB and flush on explicit call.
// Adapted from iree-hal-streaming's stream.c timeline semaphore pattern.

#include "pyre_internal.h"

#include <stdlib.h>

// Create a fresh one-shot command buffer for recording.
static pyre_status_t pyre_stream_begin_cb(pyre_stream_t stream) {
  if (stream->pending_cb) return pyre_ok_status();

  iree_status_t status = iree_hal_command_buffer_create(
      stream->device->hal_device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER |
          IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0,
      &stream->pending_cb);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  status = iree_hal_command_buffer_begin(stream->pending_cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(stream->pending_cb);
    stream->pending_cb = NULL;
    return pyre_status_from_iree(status);
  }

  return pyre_ok_status();
}

pyre_status_t pyre_stream_create(pyre_device_t device, uint32_t flags,
                                 pyre_stream_t* stream) {
  if (!device || !stream) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device or stream is NULL");
  }

  pyre_stream_s* s = (pyre_stream_s*)calloc(1, sizeof(pyre_stream_s));
  if (!s) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate stream");
  }

  iree_atomic_ref_count_init(&s->ref_count);
  s->device = device;
  s->flags = flags;
  s->timepoint = 0;
  s->has_pending_work = false;
  s->pending_cb = NULL;

  // Create the stream's timeline semaphore.
  pyre_status_t status =
      pyre_semaphore_create(device, /*initial_value=*/0, &s->semaphore);
  if (!pyre_status_is_ok(status)) {
    free(s);
    return status;
  }

  *stream = s;
  return pyre_ok_status();
}

pyre_status_t pyre_stream_retain(pyre_stream_t stream) {
  if (!stream) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }
  iree_atomic_ref_count_inc(&stream->ref_count);
  return pyre_ok_status();
}

pyre_status_t pyre_stream_release(pyre_stream_t stream) {
  if (!stream) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }
  if (iree_atomic_ref_count_dec(&stream->ref_count) == 1) {
    // Flush pending work before destruction.
    if (stream->has_pending_work) {
      pyre_status_ignore(pyre_stream_flush(stream));
    }
    // Wait for all submitted work.
    if (stream->semaphore && stream->timepoint > 0) {
      pyre_status_ignore(pyre_semaphore_wait(
          stream->semaphore, stream->timepoint, UINT64_MAX));
    }
    if (stream->pending_cb) {
      iree_hal_command_buffer_release(stream->pending_cb);
    }
    if (stream->semaphore) {
      pyre_status_ignore(pyre_semaphore_release(stream->semaphore));
    }
    free(stream);
  }
  return pyre_ok_status();
}

pyre_status_t pyre_stream_flush(pyre_stream_t stream) {
  if (!stream) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }
  if (!stream->has_pending_work || !stream->pending_cb) {
    return pyre_ok_status();
  }

  // End the command buffer recording.
  iree_status_t status = iree_hal_command_buffer_end(stream->pending_cb);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  // Build wait/signal semaphore lists.
  // Wait on current timepoint, signal next.
  uint64_t wait_value = stream->timepoint;
  uint64_t signal_value = stream->timepoint + 1;

  iree_hal_semaphore_t* sem = stream->semaphore->hal_semaphore;

  iree_hal_semaphore_list_t wait_list = {
      .count = (stream->timepoint > 0) ? 1 : 0,
      .semaphores = &sem,
      .payload_values = &wait_value,
  };
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &sem,
      .payload_values = &signal_value,
  };

  iree_hal_buffer_binding_table_t binding_table = {0};
  status = iree_hal_device_queue_execute(
      stream->device->hal_device,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      wait_list, signal_list,
      stream->pending_cb,
      binding_table, /*flags=*/0);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  stream->timepoint = signal_value;
  iree_hal_command_buffer_release(stream->pending_cb);
  stream->pending_cb = NULL;
  stream->has_pending_work = false;
  return pyre_ok_status();
}

pyre_status_t pyre_stream_synchronize(pyre_stream_t stream) {
  if (!stream) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }

  // Flush any pending work first.
  pyre_status_t status = pyre_stream_flush(stream);
  if (!pyre_status_is_ok(status)) return status;

  if (stream->timepoint == 0) return pyre_ok_status();

  return pyre_semaphore_wait(stream->semaphore, stream->timepoint, UINT64_MAX);
}

pyre_status_t pyre_stream_query(pyre_stream_t stream, bool* complete) {
  if (!stream || !complete) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "stream or complete is NULL");
  }
  if (stream->timepoint == 0) {
    *complete = true;
    return pyre_ok_status();
  }
  uint64_t current = 0;
  pyre_status_t status = pyre_semaphore_query(stream->semaphore, &current);
  if (!pyre_status_is_ok(status)) return status;
  *complete = (current >= stream->timepoint);
  return pyre_ok_status();
}

pyre_status_t pyre_stream_get_semaphore(pyre_stream_t stream,
                                        pyre_semaphore_t* semaphore) {
  if (!stream || !semaphore) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "stream or semaphore is NULL");
  }
  *semaphore = stream->semaphore;
  return pyre_ok_status();
}

pyre_status_t pyre_stream_get_timeline_position(
    pyre_stream_t stream, pyre_timeline_point_t* position) {
  if (!stream || !position) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "stream or position is NULL");
  }
  position->semaphore = stream->semaphore;
  position->value = stream->timepoint;
  return pyre_ok_status();
}

pyre_status_t pyre_stream_wait_on(pyre_stream_t stream,
                                  pyre_timeline_point_t position) {
  if (!stream) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }
  if (!position.semaphore) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "position semaphore is NULL");
  }

  // Flush current pending work with a barrier that waits on the given point.
  pyre_status_t status = pyre_stream_flush(stream);
  if (!pyre_status_is_ok(status)) return status;

  // Insert a queue barrier that waits on the external semaphore
  // and signals our next timepoint.
  uint64_t signal_value = stream->timepoint + 1;
  iree_hal_semaphore_t* wait_sem = position.semaphore->hal_semaphore;
  uint64_t wait_val = position.value;
  iree_hal_semaphore_t* sig_sem = stream->semaphore->hal_semaphore;

  iree_hal_semaphore_list_t wait_list = {
      .count = 1,
      .semaphores = &wait_sem,
      .payload_values = &wait_val,
  };
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &sig_sem,
      .payload_values = &signal_value,
  };

  iree_status_t iree_status = iree_hal_device_queue_barrier(
      stream->device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      wait_list, signal_list, /*flags=*/0);
  if (!iree_status_is_ok(iree_status)) {
    return pyre_status_from_iree(iree_status);
  }

  stream->timepoint = signal_value;
  return pyre_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream operations (record into pending CB)
//===----------------------------------------------------------------------===//

pyre_status_t pyre_stream_fill_buffer(pyre_stream_t stream,
                                      pyre_buffer_t buffer,
                                      size_t offset, size_t size,
                                      const void* pattern,
                                      size_t pattern_size) {
  if (!stream || !buffer || !pattern) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "stream, buffer, or pattern is NULL");
  }

  pyre_status_t status = pyre_stream_begin_cb(stream);
  if (!pyre_status_is_ok(status)) return status;

  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      buffer->hal_buffer, (iree_device_size_t)offset, (iree_device_size_t)size);
  iree_status_t iree_status = iree_hal_command_buffer_fill_buffer(
      stream->pending_cb, target_ref,
      pattern, (iree_host_size_t)pattern_size, /*flags=*/0);
  if (!iree_status_is_ok(iree_status)) {
    return pyre_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return pyre_ok_status();
}

pyre_status_t pyre_stream_copy_buffer(pyre_stream_t stream,
                                      pyre_buffer_t src, size_t src_offset,
                                      pyre_buffer_t dst, size_t dst_offset,
                                      size_t size) {
  if (!stream || !src || !dst) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "stream, src, or dst is NULL");
  }

  pyre_status_t status = pyre_stream_begin_cb(stream);
  if (!pyre_status_is_ok(status)) return status;

  iree_hal_buffer_ref_t source_ref = iree_hal_make_buffer_ref(
      src->hal_buffer, (iree_device_size_t)src_offset, (iree_device_size_t)size);
  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      dst->hal_buffer, (iree_device_size_t)dst_offset, (iree_device_size_t)size);
  iree_status_t iree_status = iree_hal_command_buffer_copy_buffer(
      stream->pending_cb, source_ref, target_ref, /*flags=*/0);
  if (!iree_status_is_ok(iree_status)) {
    return pyre_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return pyre_ok_status();
}

pyre_status_t pyre_stream_update_buffer(pyre_stream_t stream,
                                        const void* host_data,
                                        size_t host_data_size,
                                        pyre_buffer_t dst,
                                        size_t dst_offset) {
  if (!stream || !host_data || !dst) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "stream, host_data, or dst is NULL");
  }

  pyre_status_t status = pyre_stream_begin_cb(stream);
  if (!pyre_status_is_ok(status)) return status;

  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      dst->hal_buffer, (iree_device_size_t)dst_offset,
      (iree_device_size_t)host_data_size);
  iree_status_t iree_status = iree_hal_command_buffer_update_buffer(
      stream->pending_cb, host_data, (iree_host_size_t)0,
      target_ref, /*flags=*/0);
  if (!iree_status_is_ok(iree_status)) {
    return pyre_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return pyre_ok_status();
}
