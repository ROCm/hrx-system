// Copyright 2026 The IREE Authors
//
// Licensed under the Apache v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_virtual_buffer.h"

#include <string.h>

#include "iree/hal/replay/recorder_buffer.h"
#include "iree/hal/replay/recorder_pool.h"
#include "iree/hal/replay/recorder_record.h"

iree_status_t iree_hal_replay_recorder_create_virtual_buffer(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_virtual_buffer_create_callback_t next,
    iree_hal_device_group_t* request_group, const iree_hal_pool_scope_t* scope,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;
  iree_hal_replay_recorder_buffer_t* buffer_proxy = NULL;
  iree_status_t status = iree_hal_replay_recorder_buffer_allocate_proxy(
      host_allocator, &buffer_proxy);
  iree_hal_replay_pool_family_access_payload_t* family_payloads = NULL;
  if (iree_status_is_ok(status) && scope->family_count) {
    status = iree_allocator_malloc_array(host_allocator, scope->family_count,
                                         sizeof(*family_payloads),
                                         (void**)&family_payloads);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_pool_capture_families(scope, family_payloads);
  }
  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_reserve_object_id(recorder, &buffer_id);
  }
  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        recorder, IREE_HAL_REPLAY_OBJECT_ID_NONE,
        IREE_HAL_REPLAY_OBJECT_ID_NONE, buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
        IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_CREATE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_CREATE, &pending_record);
  }
  iree_hal_buffer_t* native_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = next.fn(next.user_data, request_group, scope, options,
                     host_allocator, &native_buffer);
  }
  iree_hal_replay_buffer_object_payload_t buffer_payload = {0};
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_make_object_payload(native_buffer,
                                                        &buffer_payload);
  }
  if (pending_record.recorder) {
    const iree_hal_replay_virtual_buffer_create_payload_t payload = {
        .byte_length = options->byte_length,
        .min_alignment = options->min_alignment,
        .mapping_granularity = options->mapping_granularity,
        .family_count = scope->family_count,
        .host_access = scope->host.access,
        .host_mapping_modes = scope->host.modes,
        .host_cacheability = scope->host.cacheability,
        .host_atomic_device_scope_32 = scope->host.atomics.device_scope_32,
        .host_atomic_device_scope_64 = scope->host.atomics.device_scope_64,
        .host_atomic_system_scope_32 = scope->host.atomics.system_scope_32,
        .host_atomic_system_scope_64 = scope->host.atomics.system_scope_64,
    };
    const iree_const_byte_span_t operation_iovecs[] = {
        iree_make_const_byte_span(&payload, sizeof(payload)),
        iree_make_const_byte_span(
            family_payloads, scope->family_count * sizeof(*family_payloads)),
    };
    const iree_const_byte_span_t buffer_iovec =
        iree_make_const_byte_span(&buffer_payload, sizeof(buffer_payload));
    const iree_hal_replay_created_object_record_t created_objects[] = {
        {
            .object_id = buffer_id,
            .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
            .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT,
            .iovec_count = 1,
            .iovecs = &buffer_iovec,
        },
    };
    status = iree_hal_replay_recorder_end_creation_operation_list(
        &pending_record, status, IREE_ARRAYSIZE(operation_iovecs),
        operation_iovecs, IREE_ARRAYSIZE(created_objects), created_objects);
  }
  iree_allocator_free(host_allocator, family_payloads);
  if (iree_status_is_ok(status)) {
    *out_buffer = iree_hal_replay_recorder_buffer_initialize_proxy(
        recorder, IREE_HAL_REPLAY_OBJECT_ID_NONE, buffer_id,
        IREE_HAL_REPLAY_OBJECT_ID_NONE, NULL, native_buffer, host_allocator,
        buffer_proxy);
    buffer_proxy = NULL;
  }
  iree_hal_replay_recorder_buffer_free_proxy(host_allocator, buffer_proxy);
  iree_hal_buffer_release(native_buffer);
  return status;
}

iree_status_t iree_hal_replay_recorder_queue_map(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t queue_id, iree_hal_queue_t* base_queue,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t mapping_count, const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags, iree_allocator_t host_allocator) {
  iree_hal_replay_memory_mapping_payload_t* mapping_payloads = NULL;
  iree_status_t status = iree_ok_status();
  if (mapping_count) {
    status = iree_allocator_malloc_array(host_allocator, mapping_count,
                                         sizeof(*mapping_payloads),
                                         (void**)&mapping_payloads);
  }
  bool can_record = iree_hal_replay_recorder_has_captured_semaphores(
                        recorder, wait_semaphore_list) &&
                    iree_hal_replay_recorder_has_captured_semaphores(
                        recorder, signal_semaphore_list);
  for (iree_host_size_t i = 0; i < mapping_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_replay_memory_mapping_payload_t* payload = &mapping_payloads[i];
    memset(payload, 0, sizeof(*payload));
    const iree_hal_buffer_ref_t target_ref = {
        .buffer = mappings[i].target,
        .offset = mappings[i].target_range.byte_offset,
        .length = mappings[i].target_range.byte_length,
    };
    iree_hal_replay_buffer_ref_payload_t target_payload;
    iree_hal_replay_recorder_buffer_ref_make_payload(target_ref,
                                                     &target_payload);
    payload->target_buffer_id = target_payload.buffer_id;
    payload->target_offset = target_payload.offset;
    payload->byte_length = target_payload.length;
    can_record &=
        iree_hal_replay_recorder_find_buffer_id(recorder, target_ref.buffer) !=
        IREE_HAL_REPLAY_OBJECT_ID_NONE;
    payload->scope_id = mappings[i].scope.id;
    payload->operation = mappings[i].operation;
    payload->access = mappings[i].access;
    iree_hal_buffer_ref_t source_ref = {
        .buffer = mappings[i].source,
        .offset = mappings[i].source_offset,
        .length = mappings[i].target_range.byte_length,
    };
    iree_hal_replay_recorder_buffer_ref_make_payload(source_ref,
                                                     &payload->source_ref);
    if (source_ref.buffer) {
      can_record &=
          iree_hal_replay_recorder_find_buffer_id(
              recorder, source_ref.buffer) != IREE_HAL_REPLAY_OBJECT_ID_NONE;
    }
  }
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        recorder, wait_semaphore_list, host_allocator, &wait_payloads,
        &wait_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        recorder, signal_semaphore_list, host_allocator, &signal_payloads,
        &signal_payloads_size);
  }
  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        recorder, device_id, queue_id, IREE_HAL_REPLAY_OBJECT_ID_NONE,
        IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_MAP,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_MAP
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (pending_record.recorder && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_map(base_queue, wait_semaphore_list,
                                signal_semaphore_list, mapping_count, mappings,
                                flags);
  }
  if (pending_record.recorder) {
    const iree_hal_replay_queue_map_payload_t payload = {
        .wait_semaphore_count = wait_semaphore_list.count,
        .signal_semaphore_count = signal_semaphore_list.count,
        .mapping_count = mapping_count,
        .flags = flags,
    };
    const iree_const_byte_span_t iovecs[] = {
        iree_make_const_byte_span(&payload, sizeof(payload)),
        iree_make_const_byte_span(wait_payloads, wait_payloads_size),
        iree_make_const_byte_span(signal_payloads, signal_payloads_size),
        iree_make_const_byte_span(mapping_payloads,
                                  mapping_count * sizeof(*mapping_payloads)),
    };
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, can_record ? IREE_ARRAYSIZE(iovecs) : 0,
        can_record ? iovecs : NULL);
  }
  iree_allocator_free(host_allocator, signal_payloads);
  iree_allocator_free(host_allocator, wait_payloads);
  iree_allocator_free(host_allocator, mapping_payloads);
  return status;
}
