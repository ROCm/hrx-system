// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Each stream batches operations on one queue and publishes a timeline point
// when that queue accepts the batch.

#include "stream.h"

#include <stdlib.h>

#include "barrier.h"
#include "hrx_internal.h"
#include "transfer.h"

// Acquire the preceding stream point before the device sequence and release
// its writes before signaling the next point. Both submission paths use the
// same descriptors.
static const iree_hal_barrier_t hrx_stream_acquire_from_system = {
    .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
    .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
    .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
};
static const iree_hal_barrier_t hrx_stream_release_to_system = {
    .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
    .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
};
static const iree_hal_memory_barrier_t hrx_stream_full_access = {
    .source_scope =
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
    .target_scope =
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
};
static const iree_hal_barrier_t hrx_stream_full_barrier = {
    .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
    .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
    .memory_barrier_count = 1,
    .memory_barriers = &hrx_stream_full_access,
};

// A local dependency precedes its consumer. Host and other-device consumers
// observe the recording's release at flush, not each intermediate operation.
static iree_status_t hrx_stream_record_ordering_barrier(
    iree_hal_command_buffer_t* command_buffer) {
  const iree_hal_access_scope_t access = IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
                                         IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                                         IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                                         IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                                         IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;
  const iree_hal_memory_barrier_t memory = {.source_scope = access,
                                            .target_scope = access};
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .memory_barrier_count = 1,
      .memory_barriers = &memory,
  };
  return iree_hal_command_buffer_barrier(command_buffer, &barrier);
}

// A new recording belongs to the pending batch only once its first command
// succeeds. Until then the caller owns it and any leading barrier stays
// pending.
static iree_status_t hrx_stream_prepare_recording(
    hrx_stream_t stream, iree_hal_command_category_t categories,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;
  if (!iree_all_bits_set(stream->command_categories, categories)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stream queue does not support this operation");
  }
  if (stream->pending_cb) {
    *out_command_buffer = stream->pending_cb;
    // A payload needs an edge from the preceding command. An explicit barrier
    // supplies its own dependency.
    return categories ? hrx_stream_record_ordering_barrier(stream->pending_cb)
                      : iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      iree_hal_queue_family(stream->hal_queue),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          hrx_device_profile_command_buffer_mode(stream->queue->device),
      stream->command_categories, /*binding_capacity=*/0, out_command_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(*out_command_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_barrier(
      *out_command_buffer, &hrx_stream_acquire_from_system));
  if (stream->pending_full_barrier) {
    return iree_hal_command_buffer_barrier(*out_command_buffer,
                                           &hrx_stream_full_barrier);
  }
  return iree_ok_status();
}

static void hrx_stream_finish_command(hrx_stream_t stream,
                                      iree_hal_command_buffer_t* command_buffer,
                                      bool accepted) {
  if (accepted) {
    stream->pending_cb = command_buffer;
    stream->pending_full_barrier = false;
  } else if (command_buffer != stream->pending_cb) {
    iree_hal_command_buffer_release(command_buffer);
  }
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
    if (stream->pending_cb || stream->pending_full_barrier) {
      status = hrx_status_to_iree(hrx_stream_flush(stream));
    }
    if (stream->timepoint > 0) {
      status =
          iree_status_join(status, hrx_status_to_iree(hrx_semaphore_wait(
                                       stream->semaphore, stream->timepoint,
                                       UINT64_MAX, HRX_WAIT_FLAG_NONE)));
    }
    hrx_device_report_failure(stream->queue->device, status);
    hrx_queue_release(stream->queue);
    hrx_semaphore_release(stream->semaphore);
    free(stream);
  }
}

hrx_queue_t hrx_stream_get_queue(hrx_stream_t stream) {
  return stream ? stream->queue : NULL;
}

static iree_status_t hrx_stream_submit_recording(
    hrx_stream_t stream, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, const iree_hal_barrier_t* terminal) {
  iree_hal_command_buffer_t* command_buffer = stream->pending_cb;
  if (terminal) {
    // An invalid terminal descriptor leaves the earlier batch available.
    IREE_RETURN_IF_ERROR(
        iree_hal_command_buffer_barrier(command_buffer, terminal));
  }
  // Sealing consumes the one-shot recording even on failure.
  stream->pending_cb = NULL;
  iree_status_t status = iree_hal_command_buffer_barrier(
      command_buffer, &hrx_stream_release_to_system);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_execute(stream->hal_queue, waits, signals,
                                    command_buffer,
                                    iree_hal_buffer_binding_table_empty(),
                                    IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  }
  iree_hal_command_buffer_release(command_buffer);
  return status;
}

static iree_status_t hrx_stream_submit_barrier(
    hrx_stream_t stream, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, const iree_hal_barrier_t* terminal) {
  iree_hal_barrier_t entry[3];
  entry[0] = hrx_stream_acquire_from_system;
  iree_host_size_t count = 1;
  if (stream->pending_full_barrier && terminal != &hrx_stream_full_barrier) {
    entry[count++] = hrx_stream_full_barrier;
  }
  if (terminal) {
    entry[count++] = *terminal;
  }
  const iree_hal_barrier_list_t before = {.count = count, .values = entry};
  const iree_hal_barrier_list_t after = {
      .count = 1, .values = &hrx_stream_release_to_system};
  const iree_hal_queue_barriers_t barriers = {.before = &before,
                                              .after = &after};
  iree_status_t status =
      iree_hal_queue_barrier(stream->hal_queue, waits, signals, &barriers,
                             IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  // A rejected terminal descriptor has not consumed the preceding full barrier.
  if (iree_status_is_ok(status) || !terminal) {
    stream->pending_full_barrier = false;
  }
  return status;
}

static iree_status_t hrx_stream_flush_pending(
    hrx_stream_t stream, const iree_hal_barrier_t* terminal) {
  if (!stream->pending_cb && !stream->pending_full_barrier && !terminal) {
    return iree_ok_status();
  }
  uint64_t wait_value = stream->timepoint;
  uint64_t signal_value = wait_value + 1;
  iree_hal_semaphore_t* semaphore = stream->semaphore->hal_semaphore;
  const iree_hal_semaphore_list_t waits = {.count = wait_value != 0,
                                           .semaphores = &semaphore,
                                           .payload_values = &wait_value};
  const iree_hal_semaphore_list_t signals = {
      .count = 1, .semaphores = &semaphore, .payload_values = &signal_value};
  iree_status_t status =
      stream->pending_cb
          ? hrx_stream_submit_recording(stream, waits, signals, terminal)
          : hrx_stream_submit_barrier(stream, waits, signals, terminal);
  if (iree_status_is_ok(status)) {
    stream->timepoint = signal_value;
  }
  return status;
}

hrx_status_t hrx_stream_flush(hrx_stream_t stream) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_flush");
  if (!stream) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL"));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, stream->pending_cb || stream->pending_full_barrier);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(hrx_stream_flush_pending(
                                  stream, /*terminal=*/NULL)));
}

hrx_status_t hrx_stream_synchronize(hrx_stream_t stream,
                                    hrx_wait_flags_t flags) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_synchronize");
  if (!stream) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL"));
  }
  if (flags & ~(HRX_WAIT_FLAG_YIELD | HRX_WAIT_FLAG_ACTIVE)) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "invalid wait flags"));
  }

  // Flush any pending work first.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, status);
  }

  HRX_RETURN_AND_END_ZONE(z0, hrx_stream_wait(stream, flags));
}

hrx_status_t hrx_stream_wait(hrx_stream_t stream, hrx_wait_flags_t flags) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_wait");
  if (!stream) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL"));
  }
  if (flags & ~(HRX_WAIT_FLAG_YIELD | HRX_WAIT_FLAG_ACTIVE)) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "invalid wait flags"));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, stream->timepoint);
  iree_status_t status = iree_ok_status();
  if (stream->timepoint != 0) {
    status = hrx_status_to_iree(hrx_semaphore_wait(
        stream->semaphore, stream->timepoint, UINT64_MAX, flags));
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
                                    size_t offset, const void* pattern,
                                    size_t pattern_size,
                                    const hrx_fill_region_t* region) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_fill_buffer");
  if (!stream || !buffer || !pattern || !region) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream, buffer, pattern, or region is NULL"));
  }
  const iree_hal_fill_region_t hal_region = hrx_fill_region_to_hal(region);
  if (!iree_hal_transfer_range_fits(offset, region->row_length,
                                    region->row_count, region->slice_count,
                                    &hal_region.target_strides, buffer->size)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                                                "fill exceeds buffer size"));
  }
  HRX_TRACE_ZONE_APPEND_BYTES(
      z0, region->row_length * region->row_count * region->slice_count);
  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      buffer->hal_buffer, offset,
      iree_hal_transfer_range_length(region->row_length, region->row_count,
                                     region->slice_count,
                                     &hal_region.target_strides));
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t iree_status = hrx_stream_prepare_recording(
      stream, IREE_HAL_COMMAND_CATEGORY_TRANSFER, &command_buffer);
  if (iree_status_is_ok(iree_status)) {
    iree_status = iree_hal_command_buffer_fill_buffer(
        command_buffer, target_ref, pattern, (iree_host_size_t)pattern_size,
        &hal_region, /*flags=*/0);
  }
  hrx_stream_finish_command(stream, command_buffer,
                            iree_status_is_ok(iree_status));
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
}

hrx_status_t hrx_stream_fill_buffer_linear(hrx_stream_t stream,
                                           hrx_buffer_t buffer, size_t offset,
                                           size_t size, const void* pattern,
                                           size_t pattern_size) {
  const hrx_fill_region_t region = {
      .row_length = size, .row_count = 1, .slice_count = 1};
  return hrx_stream_fill_buffer(stream, buffer, offset, pattern, pattern_size,
                                &region);
}

hrx_status_t hrx_stream_copy_buffer(hrx_stream_t stream, hrx_buffer_t src,
                                    size_t src_offset, hrx_buffer_t dst,
                                    size_t dst_offset,
                                    const hrx_transfer_region_t* region) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_copy_buffer");
  if (!stream || !src || !dst || !region) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream, src, dst, or region is NULL"));
  }
  const iree_hal_transfer_region_t hal_region =
      hrx_transfer_region_to_hal(region);
  if (!iree_hal_transfer_range_fits(src_offset, region->row_length,
                                    region->row_count, region->slice_count,
                                    &hal_region.source_strides, src->size) ||
      !iree_hal_transfer_range_fits(dst_offset, region->row_length,
                                    region->row_count, region->slice_count,
                                    &hal_region.target_strides, dst->size)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                                                "copy exceeds buffer size"));
  }
  HRX_TRACE_ZONE_APPEND_BYTES(z0, iree_hal_transfer_region_length(&hal_region));
  iree_hal_buffer_ref_t source_ref = iree_hal_make_buffer_ref(
      src->hal_buffer, src_offset,
      iree_hal_transfer_range_length(region->row_length, region->row_count,
                                     region->slice_count,
                                     &hal_region.source_strides));
  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      dst->hal_buffer, dst_offset,
      iree_hal_transfer_range_length(region->row_length, region->row_count,
                                     region->slice_count,
                                     &hal_region.target_strides));
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t iree_status = hrx_stream_prepare_recording(
      stream, IREE_HAL_COMMAND_CATEGORY_TRANSFER, &command_buffer);
  if (iree_status_is_ok(iree_status)) {
    iree_status = iree_hal_command_buffer_copy_buffer(
        command_buffer, source_ref, target_ref, &hal_region, /*flags=*/0);
  }
  hrx_stream_finish_command(stream, command_buffer,
                            iree_status_is_ok(iree_status));
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
}

hrx_status_t hrx_stream_copy_buffer_linear(hrx_stream_t stream,
                                           hrx_buffer_t src, size_t src_offset,
                                           hrx_buffer_t dst, size_t dst_offset,
                                           size_t size) {
  const hrx_transfer_region_t region = {
      .row_length = size, .row_count = 1, .slice_count = 1};
  return hrx_stream_copy_buffer(stream, src, src_offset, dst, dst_offset,
                                &region);
}

hrx_status_t hrx_stream_update_buffer(hrx_stream_t stream,
                                      const void* host_data,
                                      size_t host_data_size, hrx_buffer_t dst,
                                      size_t dst_offset,
                                      const hrx_transfer_region_t* region) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_update_buffer");
  if (!stream || !host_data || !dst || !region) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream, host_data, dst, or region is NULL"));
  }
  const iree_hal_transfer_region_t hal_region =
      hrx_transfer_region_to_hal(region);
  if (!iree_hal_transfer_range_fits(
          0, region->row_length, region->row_count, region->slice_count,
          &hal_region.source_strides, host_data_size) ||
      !iree_hal_transfer_range_fits(dst_offset, region->row_length,
                                    region->row_count, region->slice_count,
                                    &hal_region.target_strides, dst->size)) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                            "update exceeds source or destination size"));
  }
  HRX_TRACE_ZONE_APPEND_BYTES(z0, iree_hal_transfer_region_length(&hal_region));
  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      dst->hal_buffer, dst_offset,
      iree_hal_transfer_range_length(region->row_length, region->row_count,
                                     region->slice_count,
                                     &hal_region.target_strides));
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t iree_status = hrx_stream_prepare_recording(
      stream, IREE_HAL_COMMAND_CATEGORY_TRANSFER, &command_buffer);
  if (iree_status_is_ok(iree_status)) {
    iree_status = iree_hal_command_buffer_update_buffer(
        command_buffer, host_data, (iree_host_size_t)0, target_ref, &hal_region,
        /*flags=*/0);
  }
  hrx_stream_finish_command(stream, command_buffer,
                            iree_status_is_ok(iree_status));
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
}

hrx_status_t hrx_stream_update_buffer_linear(hrx_stream_t stream,
                                             const void* host_data,
                                             size_t host_data_size,
                                             hrx_buffer_t dst,
                                             size_t dst_offset) {
  const hrx_transfer_region_t region = {
      .row_length = host_data_size, .row_count = 1, .slice_count = 1};
  return hrx_stream_update_buffer(stream, host_data, host_data_size, dst,
                                  dst_offset, &region);
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

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t iree_status = hrx_stream_prepare_recording(
      stream, IREE_HAL_COMMAND_CATEGORY_DISPATCH, &command_buffer);
  if (iree_status_is_ok(iree_status)) {
    iree_status = iree_hal_command_buffer_dispatch(
        command_buffer, executable->hal_executable,
        iree_hal_executable_function_from_index(export_ordinal), hal_config,
        hal_constants, hal_binding_list, hal_flags);
  }
  free(hal_bindings);
  hrx_stream_finish_command(stream, command_buffer,
                            iree_status_is_ok(iree_status));
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(iree_status));
}

static hrx_status_t hrx_stream_apply_barrier(hrx_stream_t stream,
                                             const hrx_barrier_t* barrier,
                                             bool flush) {
  if (!stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "stream is NULL");
  }
  if (!flush && !barrier && !stream->pending_cb) {
    // Full barriers have no borrowed operands and are idempotent until the
    // next command. Defer the choice of submission path until then.
    stream->pending_full_barrier = true;
    return hrx_ok_status();
  }

  hrx_barrier_storage_t storage;
  const iree_hal_barrier_t* native = &hrx_stream_full_barrier;
  iree_status_t status = iree_ok_status();
  if (barrier) {
    const hrx_barrier_list_t list = {.count = 1, .values = barrier};
    const hrx_queue_barriers_t barriers = {.before = &list};
    status = hrx_barrier_storage_initialize(&barriers, &storage);
    if (!iree_status_is_ok(status)) {
      return hrx_status_from_iree(status);
    }
    native = storage.barriers.before->values;
  }
  if (flush) {
    status = hrx_stream_flush_pending(stream, native);
  } else {
    iree_hal_command_buffer_t* command_buffer = NULL;
    status =
        hrx_stream_prepare_recording(stream, /*categories=*/0, &command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_barrier(command_buffer, native);
    }
    hrx_stream_finish_command(stream, command_buffer,
                              iree_status_is_ok(status));
  }
  if (barrier) {
    hrx_barrier_storage_deinitialize(&storage);
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_stream_barrier(hrx_stream_t stream,
                                const hrx_barrier_t* barrier) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_barrier");
  HRX_RETURN_AND_END_ZONE(z0, hrx_stream_apply_barrier(stream, barrier, false));
}

hrx_status_t hrx_stream_flush_with_barrier(hrx_stream_t stream,
                                           const hrx_barrier_t* barrier) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_stream_flush_with_barrier");
  HRX_RETURN_AND_END_ZONE(z0, hrx_stream_apply_barrier(stream, barrier, true));
}
