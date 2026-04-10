// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Stream implementation. Each stream owns a timeline semaphore and a pending
// command buffer. Operations accumulate in the CB and flush on explicit call.
// Adapted from iree-hal-streaming's stream.c timeline semaphore pattern.

#include "hrx_internal.h"

#include <stdlib.h>

// Create a fresh one-shot command buffer for recording.
static hrx_status_t hrx_stream_begin_cb(hrx_stream_t stream) {
  if (stream->pending_cb) return hrx_ok_status();

  iree_status_t status = iree_hal_command_buffer_create(
      stream->device->hal_device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER |
          IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0,
      &stream->pending_cb);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  status = iree_hal_command_buffer_begin(stream->pending_cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(stream->pending_cb);
    stream->pending_cb = NULL;
    return hrx_status_from_iree(status);
  }

  return hrx_ok_status();
}

hrx_status_t hrx_stream_create(hrx_device_t device, uint32_t flags,
                                 hrx_stream_t* stream) {
  if (!device || !stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "device or stream is NULL");
  }

  hrx_stream_s* s = (hrx_stream_s*)calloc(1, sizeof(hrx_stream_s));
  if (!s) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                            "failed to allocate stream");
  }

  iree_atomic_ref_count_init(&s->ref_count);
  s->device = device;
  hrx_device_retain(s->device);
  s->flags = flags;
  s->timepoint = 0;
  s->has_pending_work = false;
  s->pending_cb = NULL;

  // Create the stream's timeline semaphore.
  hrx_status_t status =
      hrx_semaphore_create(device, /*initial_value=*/0, &s->semaphore);
  if (!hrx_status_is_ok(status)) {
    free(s);
    return status;
  }

  *stream = s;
  return hrx_ok_status();
}

void hrx_stream_retain(hrx_stream_t stream) {
  hrx_device_retain(stream->device);
  hrx_semaphore_retain(stream->semaphore);
  iree_atomic_ref_count_inc(&stream->ref_count);
}

void hrx_stream_release(hrx_stream_t stream) {
  hrx_device_t device = stream->device;
  hrx_semaphore_t semaphore = stream->semaphore;
  if (iree_atomic_ref_count_dec(&stream->ref_count) == 1) {
    if (stream->has_pending_work) {
      hrx_status_ignore(hrx_stream_flush(stream));
    }
    if (stream->timepoint > 0) {
      hrx_status_ignore(hrx_semaphore_wait(
          stream->semaphore, stream->timepoint, UINT64_MAX));
    }
    if (stream->pending_cb) {
      iree_hal_command_buffer_release(stream->pending_cb);
    }
    free(stream);
  }
  hrx_semaphore_release(semaphore);
  hrx_device_release(device);
}

hrx_status_t hrx_stream_flush(hrx_stream_t stream) {
  if (!stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }
  if (!stream->has_pending_work || !stream->pending_cb) {
    return hrx_ok_status();
  }

  // End the command buffer recording.
  iree_status_t status = iree_hal_command_buffer_end(stream->pending_cb);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
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
    return hrx_status_from_iree(status);
  }

  stream->timepoint = signal_value;
  iree_hal_command_buffer_release(stream->pending_cb);
  stream->pending_cb = NULL;
  stream->has_pending_work = false;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_synchronize(hrx_stream_t stream) {
  if (!stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }

  // Flush any pending work first.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) return status;

  if (stream->timepoint == 0) return hrx_ok_status();

  return hrx_semaphore_wait(stream->semaphore, stream->timepoint, UINT64_MAX);
}

hrx_status_t hrx_stream_query(hrx_stream_t stream, bool* complete) {
  if (!stream || !complete) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream or complete is NULL");
  }
  if (stream->timepoint == 0) {
    *complete = true;
    return hrx_ok_status();
  }
  uint64_t current = 0;
  hrx_status_t status = hrx_semaphore_query(stream->semaphore, &current);
  if (!hrx_status_is_ok(status)) return status;
  *complete = (current >= stream->timepoint);
  return hrx_ok_status();
}

hrx_status_t hrx_stream_get_semaphore(hrx_stream_t stream,
                                        hrx_semaphore_t* semaphore) {
  if (!stream || !semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream or semaphore is NULL");
  }
  *semaphore = stream->semaphore;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_get_device(hrx_stream_t stream,
                                     hrx_device_t* device) {
  if (!stream || !device) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream or device is NULL");
  }
  *device = stream->device;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_get_timeline_position(
    hrx_stream_t stream, hrx_timeline_point_t* position) {
  if (!stream || !position) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream or position is NULL");
  }
  position->semaphore = stream->semaphore;
  position->value = stream->timepoint;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_advance_timeline(hrx_stream_t stream,
                                           uint64_t* value) {
  if (!stream || !value) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream or value is NULL");
  }
  *value = ++stream->timepoint;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_wait_on(hrx_stream_t stream,
                                  hrx_timeline_point_t position) {
  if (!stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }
  if (!position.semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "position semaphore is NULL");
  }

  // Flush current pending work with a barrier that waits on the given point.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) return status;

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
    return hrx_status_from_iree(iree_status);
  }

  stream->timepoint = signal_value;
  return hrx_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream operations (record into pending CB)
//===----------------------------------------------------------------------===//

hrx_status_t hrx_stream_fill_buffer(hrx_stream_t stream,
                                      hrx_buffer_t buffer,
                                      size_t offset, size_t size,
                                      const void* pattern,
                                      size_t pattern_size) {
  if (!stream || !buffer || !pattern) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream, buffer, or pattern is NULL");
  }

  hrx_status_t status = hrx_stream_begin_cb(stream);
  if (!hrx_status_is_ok(status)) return status;

  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      buffer->hal_buffer, (iree_device_size_t)offset, (iree_device_size_t)size);
  iree_status_t iree_status = iree_hal_command_buffer_fill_buffer(
      stream->pending_cb, target_ref,
      pattern, (iree_host_size_t)pattern_size, /*flags=*/0);
  if (!iree_status_is_ok(iree_status)) {
    return hrx_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_copy_buffer(hrx_stream_t stream,
                                      hrx_buffer_t src, size_t src_offset,
                                      hrx_buffer_t dst, size_t dst_offset,
                                      size_t size) {
  if (!stream || !src || !dst) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream, src, or dst is NULL");
  }

  hrx_status_t status = hrx_stream_begin_cb(stream);
  if (!hrx_status_is_ok(status)) return status;

  iree_hal_buffer_ref_t source_ref = iree_hal_make_buffer_ref(
      src->hal_buffer, (iree_device_size_t)src_offset, (iree_device_size_t)size);
  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      dst->hal_buffer, (iree_device_size_t)dst_offset, (iree_device_size_t)size);
  iree_status_t iree_status = iree_hal_command_buffer_copy_buffer(
      stream->pending_cb, source_ref, target_ref, /*flags=*/0);
  if (!iree_status_is_ok(iree_status)) {
    return hrx_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_update_buffer(hrx_stream_t stream,
                                        const void* host_data,
                                        size_t host_data_size,
                                        hrx_buffer_t dst,
                                        size_t dst_offset) {
  if (!stream || !host_data || !dst) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream, host_data, or dst is NULL");
  }

  hrx_status_t status = hrx_stream_begin_cb(stream);
  if (!hrx_status_is_ok(status)) return status;

  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      dst->hal_buffer, (iree_device_size_t)dst_offset,
      (iree_device_size_t)host_data_size);
  iree_status_t iree_status = iree_hal_command_buffer_update_buffer(
      stream->pending_cb, host_data, (iree_host_size_t)0,
      target_ref, /*flags=*/0);
  if (!iree_status_is_ok(iree_status)) {
    return hrx_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_dispatch(
    hrx_stream_t stream, hrx_executable_t executable,
    uint32_t export_ordinal, const hrx_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const hrx_buffer_ref_t* bindings, size_t binding_count,
    uint32_t flags) {
  if (!stream || !executable || !config ||
      (binding_count > 0 && !bindings) ||
      (constants_size > 0 && !constants)) {
    return hrx_make_status(
        HRX_STATUS_INVALID_ARGUMENT,
        "stream, executable, config, constants, or bindings are invalid");
  }

  hrx_status_t status = hrx_stream_begin_cb(stream);
  if (!hrx_status_is_ok(status)) return status;

  iree_hal_buffer_ref_t* hal_bindings = NULL;
  if (binding_count > 0) {
    hal_bindings = (iree_hal_buffer_ref_t*)calloc(
        binding_count, sizeof(iree_hal_buffer_ref_t));
    if (!hal_bindings) {
      return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                              "failed to allocate dispatch bindings");
    }
    for (size_t i = 0; i < binding_count; ++i) {
      if (!bindings[i].buffer) {
        free(hal_bindings);
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                "binding buffer is NULL");
      }
      hal_bindings[i] = iree_hal_make_buffer_ref(
          bindings[i].buffer->hal_buffer,
          (iree_device_size_t)bindings[i].offset,
          (iree_device_size_t)bindings[i].length);
    }
  }

  iree_hal_dispatch_config_t hal_config = {
      .workgroup_size = {
          config->workgroup_size[0],
          config->workgroup_size[1],
          config->workgroup_size[2],
      },
      .workgroup_count = {
          config->workgroup_count[0],
          config->workgroup_count[1],
          config->workgroup_count[2],
      },
  };
  iree_const_byte_span_t hal_constants =
      iree_make_const_byte_span((const uint8_t*)constants, constants_size);
  iree_hal_buffer_ref_list_t hal_binding_list = {
      .count = (iree_host_size_t)binding_count,
      .values = hal_bindings,
  };

  iree_status_t iree_status = iree_hal_command_buffer_dispatch(
      stream->pending_cb, executable->hal_executable,
      (iree_hal_executable_export_ordinal_t)export_ordinal,
      hal_config, hal_constants, hal_binding_list,
      (iree_hal_dispatch_flags_t)flags);
  free(hal_bindings);
  if (!iree_status_is_ok(iree_status)) {
    return hrx_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_execution_barrier(hrx_stream_t stream) {
  if (!stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }

  hrx_status_t status = hrx_stream_begin_cb(stream);
  if (!hrx_status_is_ok(status)) return status;

  iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_MEMORY_ACCESS_ALL,
      .target_scope = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  iree_status_t iree_status = iree_hal_command_buffer_execution_barrier(
      stream->pending_cb,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      1, &memory_barrier,
      0, NULL);
  if (!iree_status_is_ok(iree_status)) {
    return hrx_status_from_iree(iree_status);
  }

  stream->has_pending_work = true;
  return hrx_ok_status();
}
