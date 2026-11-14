// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Stream implementation. Each stream owns a timeline semaphore and a pending
// command buffer. Operations accumulate in the CB and flush on explicit call.
// Adapted from iree-hal-streaming's stream.c timeline semaphore pattern.

#include "stream.h"

#include <stdlib.h>

#include "barrier.h"
#include "hrx_internal.h"

// Create a fresh one-shot command buffer for recording.
static hrx_status_t hrx_stream_begin_cb(
    hrx_stream_t stream, iree_hal_command_category_t required_categories) {
  if (!iree_all_bits_set(stream->command_categories, required_categories)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "stream queue does not support this operation");
  }
  if (stream->pending_cb) {
    return hrx_ok_status();
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      iree_hal_queue_family(stream->hal_queue),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          hrx_device_profile_command_buffer_mode(stream->queue->device),
      stream->command_categories,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    // Queue waits order external producers. Acquire their payload once before
    // this recording's device-only sequence begins.
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
    };
    status = iree_hal_command_buffer_barrier(command_buffer, &barrier);
  }
  if (iree_status_is_ok(status)) {
    stream->pending_cb = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return hrx_status_from_iree(status);
}

static iree_status_t hrx_stream_record_memory_barrier(
    hrx_stream_t stream, iree_hal_access_scope_t access) {
  iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = access,
      .target_scope = access,
  };
  const iree_hal_barrier_t execution_barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .memory_barrier_count = 1,
      .memory_barriers = &memory_barrier,
  };
  return iree_hal_command_buffer_barrier(stream->pending_cb,
                                         &execution_barrier);
}

// A local dependency precedes its consumer. Host and other-device consumers
// observe the recording's release at flush, not each intermediate operation.
static iree_status_t hrx_stream_record_ordering_barrier(hrx_stream_t stream) {
  if (!stream->has_pending_work) {
    return iree_ok_status();
  }
  return hrx_stream_record_memory_barrier(
      stream, IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
                  IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                  IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                  IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                  IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE);
}

hrx_status_t hrx_stream_create(hrx_queue_t queue, uint32_t flags,
                               hrx_stream_t* out_stream) {
  if (!out_stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "stream output is NULL");
  }
  *out_stream = NULL;
  if (!queue) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "queue is NULL");
  }

  hrx_stream_s* s = (hrx_stream_s*)calloc(1, sizeof(hrx_stream_s));
  if (!s) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate stream");
  }

  iree_atomic_ref_count_init(&s->ref_count);
  s->queue = queue;
  hrx_queue_retain(queue);
  s->hal_queue = queue->hal_queue;
  s->command_categories =
      hrx_queue_family_command_categories(iree_hal_queue_family(s->hal_queue));
  s->flags = flags;
  s->timepoint = 0;
  s->has_pending_work = false;
  s->pending_cb = NULL;

  // Create the stream's timeline semaphore.
  hrx_status_t status =
      hrx_semaphore_create(queue->device, /*initial_value=*/0, &s->semaphore);
  if (!hrx_status_is_ok(status)) {
    hrx_queue_release(s->queue);
    free(s);
    return status;
  }

  *out_stream = s;
  return hrx_ok_status();
}

void hrx_stream_retain(hrx_stream_t stream) {
  if (!stream) {
    return;
  }
  iree_atomic_ref_count_inc(&stream->ref_count);
}

void hrx_stream_release(hrx_stream_t stream) {
  if (!stream) {
    return;
  }
  if (iree_atomic_ref_count_dec(&stream->ref_count) == 1) {
    iree_status_t status = iree_ok_status();
    if (stream->has_pending_work) {
      status = hrx_status_to_iree(hrx_stream_flush(stream));
    }
    if (stream->timepoint > 0) {
      status = iree_status_join(
          status, hrx_status_to_iree(hrx_semaphore_wait(
                      stream->semaphore, stream->timepoint, UINT64_MAX)));
    }
    hrx_device_report_failure(stream->queue->device, status);
    iree_hal_command_buffer_release(stream->pending_cb);
    hrx_queue_release(stream->queue);
    hrx_semaphore_release(stream->semaphore);
    free(stream);
  }
}

hrx_queue_t hrx_stream_get_queue(hrx_stream_t stream) {
  return stream ? stream->queue : NULL;
}

hrx_status_t hrx_stream_flush(hrx_stream_t stream) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_flush");
  if (!stream) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL"));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, stream->has_pending_work ? 1 : 0);
  if (!stream->has_pending_work || !stream->pending_cb) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
  }

  // Ending a one-shot recording consumes it even if sealing or submission
  // fails. It cannot remain available for appending commands or ending again.
  iree_hal_command_buffer_t* command_buffer = stream->pending_cb;
  stream->pending_cb = NULL;
  stream->has_pending_work = false;
  // The published timeline point may be observed by the host or another queue.
  // Release the complete recording before signaling that point.
  const iree_hal_barrier_t release_barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  iree_status_t status =
      iree_hal_command_buffer_barrier(command_buffer, &release_barrier);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
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

  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_execute(stream->hal_queue, wait_list, signal_list,
                                    command_buffer, binding_table,
                                    IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    stream->timepoint = signal_value;
  }

  iree_hal_command_buffer_release(command_buffer);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_stream_synchronize(hrx_stream_t stream) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_synchronize");
  if (!stream) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL"));
  }

  // Flush any pending work first.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  HRX_RETURN_AND_END_ZONE(z0, hrx_stream_wait(stream));
}

hrx_status_t hrx_stream_wait(hrx_stream_t stream) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_wait");
  if (!stream) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL"));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, stream->timepoint);
  iree_status_t status = iree_ok_status();
  if (stream->timepoint != 0) {
    status = hrx_status_to_iree(
        hrx_semaphore_wait(stream->semaphore, stream->timepoint, UINT64_MAX));
  }
  status = iree_status_join(status,
                            hrx_runtime_profile_flush(stream->queue->device));
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
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
  if (!hrx_status_is_ok(status)) {
    return status;
  }
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

hrx_status_t hrx_stream_get_device(hrx_stream_t stream, hrx_device_t* device) {
  if (!stream || !device) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "stream or device is NULL");
  }
  *device = stream->queue->device;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_get_timeline_position(hrx_stream_t stream,
                                              hrx_timeline_point_t* position) {
  if (!stream || !position) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "stream or position is NULL");
  }
  position->semaphore = stream->semaphore;
  position->value = stream->timepoint;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_advance_timeline(hrx_stream_t stream, uint64_t* value) {
  if (!stream || !value) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "stream or value is NULL");
  }
  *value = ++stream->timepoint;
  return hrx_ok_status();
}

hrx_status_t hrx_stream_wait_on(hrx_stream_t stream,
                                hrx_timeline_point_t position) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_wait_on");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, position.value);
  if (!stream) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL"));
  }
  if (!position.semaphore) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "position semaphore is NULL"));
  }

  // Flush current pending work with a barrier that waits on the given point.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  // HAL queues are out of order: adding an external dependency must also
  // preserve the stream's prior tail before publishing its next point.
  uint64_t signal_value = stream->timepoint + 1;
  iree_hal_semaphore_t* sig_sem = stream->semaphore->hal_semaphore;
  iree_hal_semaphore_t* wait_semaphores[2] = {position.semaphore->hal_semaphore,
                                              sig_sem};
  uint64_t wait_values[2] = {position.value, stream->timepoint};
  iree_host_size_t wait_count = 1;
  if (wait_semaphores[0] == sig_sem) {
    wait_values[0] = iree_max(wait_values[0], stream->timepoint);
  } else if (stream->timepoint > 0) {
    wait_count = 2;
  }

  iree_hal_semaphore_list_t wait_list = {
      .count = wait_count,
      .semaphores = wait_semaphores,
      .payload_values = wait_values,
  };
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &sig_sem,
      .payload_values = &signal_value,
  };

  iree_status_t iree_status = iree_hal_queue_barrier(
      stream->hal_queue, wait_list, signal_list, /*barriers=*/NULL,
      IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  if (!iree_status_is_ok(iree_status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
  }

  stream->timepoint = signal_value;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

//===----------------------------------------------------------------------===//
// Stream operations (record into pending CB)
//===----------------------------------------------------------------------===//

hrx_status_t hrx_stream_fill_buffer(hrx_stream_t stream, hrx_buffer_t buffer,
                                    size_t offset, size_t size,
                                    const void* pattern, size_t pattern_size) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_fill_buffer");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!stream || !buffer || !pattern) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream, buffer, or pattern is NULL"));
  }

  hrx_status_t status =
      hrx_stream_begin_cb(stream, IREE_HAL_COMMAND_CATEGORY_TRANSFER);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      buffer->hal_buffer, (iree_device_size_t)offset, (iree_device_size_t)size);
  iree_status_t iree_status = hrx_stream_record_ordering_barrier(stream);
  if (iree_status_is_ok(iree_status)) {
    iree_status = iree_hal_command_buffer_fill_buffer(
        stream->pending_cb, target_ref, pattern, (iree_host_size_t)pattern_size,
        /*flags=*/0);
  }
  if (!iree_status_is_ok(iree_status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
  }

  stream->has_pending_work = true;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

hrx_status_t hrx_stream_copy_buffer(hrx_stream_t stream, hrx_buffer_t src,
                                    size_t src_offset, hrx_buffer_t dst,
                                    size_t dst_offset, size_t size) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_copy_buffer");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!stream || !src || !dst) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "stream, src, or dst is NULL"));
  }

  hrx_status_t status =
      hrx_stream_begin_cb(stream, IREE_HAL_COMMAND_CATEGORY_TRANSFER);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  iree_hal_buffer_ref_t source_ref =
      iree_hal_make_buffer_ref(src->hal_buffer, (iree_device_size_t)src_offset,
                               (iree_device_size_t)size);
  iree_hal_buffer_ref_t target_ref =
      iree_hal_make_buffer_ref(dst->hal_buffer, (iree_device_size_t)dst_offset,
                               (iree_device_size_t)size);
  iree_status_t iree_status = hrx_stream_record_ordering_barrier(stream);
  if (iree_status_is_ok(iree_status)) {
    iree_status = iree_hal_command_buffer_copy_buffer(
        stream->pending_cb, source_ref, target_ref, /*flags=*/0);
  }
  if (!iree_status_is_ok(iree_status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
  }

  stream->has_pending_work = true;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

hrx_status_t hrx_stream_update_buffer(hrx_stream_t stream,
                                      const void* host_data,
                                      size_t host_data_size, hrx_buffer_t dst,
                                      size_t dst_offset) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_update_buffer");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, host_data_size);
  if (!stream || !host_data || !dst) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream, host_data, or dst is NULL"));
  }

  hrx_status_t status =
      hrx_stream_begin_cb(stream, IREE_HAL_COMMAND_CATEGORY_TRANSFER);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  iree_hal_buffer_ref_t target_ref =
      iree_hal_make_buffer_ref(dst->hal_buffer, (iree_device_size_t)dst_offset,
                               (iree_device_size_t)host_data_size);
  iree_status_t iree_status = hrx_stream_record_ordering_barrier(stream);
  if (iree_status_is_ok(iree_status)) {
    iree_status = iree_hal_command_buffer_update_buffer(
        stream->pending_cb, host_data, (iree_host_size_t)0, target_ref,
        /*flags=*/0);
  }
  if (!iree_status_is_ok(iree_status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
  }

  stream->has_pending_work = true;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

hrx_status_t hrx_stream_dispatch(hrx_stream_t stream,
                                 hrx_executable_t executable,
                                 uint32_t export_ordinal,
                                 const hrx_dispatch_config_t* config,
                                 const void* constants, size_t constants_size,
                                 const hrx_buffer_ref_t* bindings,
                                 size_t binding_count, uint32_t flags) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_dispatch");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, export_ordinal);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, binding_count);
  if (!stream || !executable || !config || (binding_count > 0 && !bindings) ||
      (constants_size > 0 && !constants)) {
    HRX_RETURN_AND_END_ZONE(
        z0,
        hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "stream, executable, config, constants, or bindings are invalid"));
  }

  iree_hal_dispatch_flags_t hal_flags = IREE_HAL_DISPATCH_FLAG_NONE;
  hrx_status_t status = hrx_iree_dispatch_flags_from_hrx(flags, &hal_flags);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  status = hrx_stream_begin_cb(stream, IREE_HAL_COMMAND_CATEGORY_DISPATCH);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  iree_hal_buffer_ref_t* hal_bindings = NULL;
  if (binding_count > 0) {
    hal_bindings = (iree_hal_buffer_ref_t*)calloc(
        binding_count, sizeof(iree_hal_buffer_ref_t));
    if (!hal_bindings) {
      HRX_RETURN_AND_END_ZONE(
          z0, hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                              "failed to allocate dispatch bindings"));
    }
    for (size_t i = 0; i < binding_count; ++i) {
      if (!bindings[i].buffer) {
        free(hal_bindings);
        HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                    "binding buffer is NULL"));
      }
      hal_bindings[i] =
          iree_hal_make_buffer_ref(bindings[i].buffer->hal_buffer,
                                   (iree_device_size_t)bindings[i].offset,
                                   (iree_device_size_t)bindings[i].length);
    }
  }

  iree_hal_dispatch_config_t hal_config = {
      .workgroup_size =
          {
              config->workgroup_size[0],
              config->workgroup_size[1],
              config->workgroup_size[2],
          },
      .workgroup_count =
          {
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

  iree_status_t iree_status = hrx_stream_record_ordering_barrier(stream);
  if (iree_status_is_ok(iree_status)) {
    iree_status = iree_hal_command_buffer_dispatch(
        stream->pending_cb, executable->hal_executable,
        iree_hal_executable_function_from_index(export_ordinal), hal_config,
        hal_constants, hal_binding_list, hal_flags);
  }
  free(hal_bindings);
  if (!iree_status_is_ok(iree_status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
  }

  stream->has_pending_work = true;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

hrx_status_t hrx_stream_barrier(hrx_stream_t stream,
                                const hrx_barrier_t* barrier) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_barrier");
  if (!stream) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL"));
  }

  hrx_status_t status = hrx_stream_begin_cb(stream, /*required_categories=*/0);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  iree_status_t iree_status = iree_ok_status();
  if (barrier) {
    const hrx_barrier_list_t list = {.count = 1, .values = barrier};
    const hrx_queue_barriers_t barriers = {.before = &list};
    hrx_barrier_storage_t storage;
    iree_status = hrx_barrier_storage_initialize(&barriers, &storage);
    if (iree_status_is_ok(iree_status)) {
      iree_status = iree_hal_command_buffer_barrier(
          stream->pending_cb, storage.barriers.before->values);
    }
    hrx_barrier_storage_deinitialize(&storage);
  } else {
    iree_status = hrx_stream_record_memory_barrier(
        stream,
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE);
  }
  if (!iree_status_is_ok(iree_status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
  }

  stream->has_pending_work = true;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}
