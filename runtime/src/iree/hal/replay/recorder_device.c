// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_device.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/api.h"
#include "iree/hal/replay/recorder.h"
#include "iree/hal/replay/recorder_allocator.h"
#include "iree/hal/replay/recorder_buffer.h"
#include "iree/hal/replay/recorder_command_buffer.h"
#include "iree/hal/replay/recorder_executable.h"
#include "iree/hal/replay/recorder_file.h"
#include "iree/hal/replay/recorder_queue.h"
#include "iree/hal/replay/recorder_record.h"

//===----------------------------------------------------------------------===//
// iree_hal_replay_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_queue_family_t {
  // Canonical family identity owned by the recording wrapper device.
  iree_hal_queue_family_t base;

  // Start of this family's queues in the wrapper's flat queue table.
  iree_host_size_t queue_offset;

  // Number of provisioned queues in this family.
  uint32_t queue_count;
} iree_hal_replay_queue_family_t;

typedef struct iree_hal_replay_device_t {
  // HAL resource header for the recording wrapper device.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Source group retaining the underlying device and original topology.
  iree_hal_device_group_t* base_group;
  // Underlying device receiving forwarded HAL calls.
  iree_hal_device_t* base_device;
  // Recording allocator returned from iree_hal_device_allocator.
  iree_hal_allocator_t* allocator;
  // Session-local object id assigned to this wrapper.
  iree_hal_replay_object_id_t device_id;
  // Topology information assigned to this wrapper during group creation.
  iree_hal_device_topology_info_t topology_info;

  // Number of canonical queue-family records.
  iree_host_size_t queue_family_count;

  // Wrapper-owned canonical queue-family records in ordinal order.
  iree_hal_replay_queue_family_t* queue_families;

  // Number of successfully initialized queue proxies.
  iree_host_size_t initialized_queue_count;

  // Wrapper-owned provisioned queue proxies grouped by canonical family.
  iree_hal_replay_recorder_queue_t* queues;
} iree_hal_replay_device_t;

static const iree_hal_device_vtable_t iree_hal_replay_device_vtable;

static bool iree_hal_replay_device_isa(iree_hal_device_t* base_device) {
  return iree_hal_resource_is(base_device, &iree_hal_replay_device_vtable);
}

static iree_hal_replay_device_t* iree_hal_replay_device_cast(
    iree_hal_device_t* base_device) {
  IREE_HAL_ASSERT_TYPE(base_device, &iree_hal_replay_device_vtable);
  return (iree_hal_replay_device_t*)base_device;
}

static iree_hal_device_t* iree_hal_replay_device_base_or_self(
    iree_hal_device_t* device) {
  return iree_hal_replay_device_isa(device)
             ? iree_hal_replay_device_cast(device)->base_device
             : device;
}

static iree_status_t iree_hal_replay_device_begin_operation(
    iree_hal_replay_device_t* device,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      operation_code, IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, out_pending_record);
}

static iree_status_t iree_hal_replay_device_complete_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status) {
  return iree_hal_replay_recorder_end_operation(pending_record,
                                                operation_status);
}

static void iree_hal_replay_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < device->initialized_queue_count; ++i) {
    iree_hal_queue_t* queue = &device->queues[i].base;
    iree_atomic_ref_count_abort_if_uses(&queue->resource.ref_count);
    iree_hal_queue_release(queue);
  }
  iree_hal_allocator_release(device->allocator);
  iree_hal_device_release(device->base_device);
  iree_hal_device_group_release(device->base_group);
  iree_hal_replay_recorder_release(device->recorder);
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_replay_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_device_id(device->base_device);
}

static iree_allocator_t iree_hal_replay_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_device_host_allocator(device->base_device);
}

static iree_hal_allocator_t* iree_hal_replay_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return device->allocator;
}

static void iree_hal_replay_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_device_replace_channel_provider(device->base_device, new_provider);
}

static iree_status_t iree_hal_replay_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record = {0};
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_TRIM, &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_trim(device->base_device));
}

static const iree_hal_device_spec_t* iree_hal_replay_device_spec(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_device_spec(device->base_device);
}

static const iree_hal_queue_family_t* iree_hal_replay_device_queue_family(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  if (family_ordinal >= device->queue_family_count) return NULL;
  return &device->queue_families[family_ordinal].base;
}

static iree_hal_queue_t* iree_hal_replay_device_queue(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal,
    iree_hal_queue_ordinal_t queue_ordinal) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  if (family_ordinal >= device->queue_family_count) return NULL;
  const iree_hal_replay_queue_family_t* family =
      &device->queue_families[family_ordinal];
  if (queue_ordinal >= family->queue_count) return NULL;
  return &device->queues[family->queue_offset + queue_ordinal].base;
}

static iree_status_t iree_hal_replay_device_sample_observation(
    iree_hal_device_t* base_device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_device_sample_observation(device->base_device,
                                            requested_flags, out_observation);
}

static const iree_hal_device_topology_info_t*
iree_hal_replay_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_replay_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(src_device);
  iree_hal_device_t* base_dst_device =
      iree_hal_replay_device_base_or_self(dst_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_REFINE_TOPOLOGY_EDGE,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_refine_topology_edge(
                           device->base_device, base_dst_device, edge));
}

static iree_status_t iree_hal_replay_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO,
      &pending_record));
  iree_status_t status = iree_ok_status();
  if (topology_info) {
    device->topology_info = *topology_info;
  } else {
    memset(&device->topology_info, 0, sizeof(device->topology_info));
  }
  return iree_hal_replay_device_complete_operation(&pending_record, status);
}

static iree_status_t iree_hal_replay_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_CHANNEL,
      &pending_record));
  iree_hal_replay_recorder_mark_unsupported(&pending_record);
  iree_status_t status = iree_hal_channel_create(
      device->base_device, queue_affinity, params, out_channel);
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  if (!iree_status_is_ok(status) && out_channel && *out_channel) {
    iree_hal_channel_release(*out_channel);
    *out_channel = NULL;
  }
  return status;
}

static iree_status_t iree_hal_replay_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_command_buffer = NULL;

  iree_hal_replay_object_id_t command_buffer_id =
      IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      device->recorder, &command_buffer_id));

  iree_hal_replay_command_buffer_object_payload_t payload;
  iree_hal_replay_recorder_command_buffer_make_object_payload(
      mode, command_categories, queue_affinity, binding_capacity, &payload);
  iree_const_byte_span_t payload_iovec =
      iree_make_const_byte_span(&payload, sizeof(payload));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id, command_buffer_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT, &pending_record));

  iree_hal_command_buffer_t* base_command_buffer = NULL;
  iree_hal_command_buffer_t* replay_command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device->base_device, mode, command_categories, queue_affinity,
      binding_capacity, &base_command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_command_buffer_create_proxy(
        device->recorder, device->device_id, command_buffer_id,
        iree_hal_device_allocator(device->base_device), base_command_buffer,
        device->host_allocator, &replay_command_buffer);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, 1, &payload_iovec,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, command_buffer_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT, 1, &payload_iovec);

  if (iree_status_is_ok(status)) {
    *out_command_buffer = replay_command_buffer;
  } else {
    iree_hal_command_buffer_release(replay_command_buffer);
  }
  iree_hal_command_buffer_release(base_command_buffer);
  return status;
}

static iree_status_t iree_hal_replay_device_load_executable(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* params,
    iree_hal_executable_t** out_executable) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_replay_recorder_device_load_executable(
      device->recorder, device->device_id, device->base_device, queue_affinity,
      target, params, device->host_allocator, out_executable);
}

static iree_status_t iree_hal_replay_device_import_file(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_file = NULL;

  iree_hal_replay_object_id_t file_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_reserve_object_id(device->recorder, &file_id));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id, file_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT, &pending_record));

  iree_hal_file_t* base_file = NULL;
  iree_hal_file_t* replay_file = NULL;
  iree_status_t status =
      iree_hal_file_import(device->base_device, queue_family_affinity, access,
                           handle, flags, &base_file);

  char reference_storage[IREE_MAX_PATH];
  iree_byte_span_t allocated_reference_storage = iree_byte_span_empty();
  iree_hal_replay_file_object_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_string_view_t reference = iree_string_view_empty();
  const iree_hal_replay_recorder_options_t* recorder_options =
      iree_hal_replay_recorder_options(device->recorder);
  iree_status_t payload_status =
      iree_hal_replay_recorder_file_make_object_payload(
          handle, queue_family_affinity, access, flags, base_file,
          recorder_options->external_file_policy,
          recorder_options->external_file_validation, device->host_allocator,
          iree_make_byte_span((uint8_t*)reference_storage,
                              sizeof(reference_storage)),
          &allocated_reference_storage, &payload, &reference);
  status = iree_status_join(status, payload_status);
  iree_const_byte_span_t payload_iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(reference.data, reference.size),
  };

  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_file_create_proxy(
        device->recorder, device->device_id, file_id, handle, base_file,
        device->host_allocator, &replay_file);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, IREE_ARRAYSIZE(payload_iovecs), payload_iovecs,
      IREE_HAL_REPLAY_OBJECT_TYPE_FILE, file_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT, IREE_ARRAYSIZE(payload_iovecs),
      payload_iovecs);
  iree_allocator_free(device->host_allocator, allocated_reference_storage.data);

  if (iree_status_is_ok(status)) {
    *out_file = replay_file;
  } else {
    iree_hal_file_release(replay_file);
  }
  iree_hal_file_release(base_file);
  return status;
}

static iree_status_t iree_hal_replay_device_create_semaphore(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_semaphore = NULL;

  iree_hal_replay_object_id_t semaphore_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      device->recorder, &semaphore_id));

  iree_hal_replay_semaphore_object_payload_t payload = {
      .queue_family_affinity = queue_family_affinity,
      .initial_value = initial_value,
      .flags = flags,
  };
  iree_const_byte_span_t payload_iovec =
      iree_make_const_byte_span(&payload, sizeof(payload));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id, semaphore_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT, &pending_record));

  iree_hal_semaphore_t* semaphore = NULL;
  iree_status_t status =
      iree_hal_semaphore_create(device->base_device, queue_family_affinity,
                                initial_value, flags, &semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_register_semaphore(
        &pending_record, semaphore, semaphore_id);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, 1, &payload_iovec,
      IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE, semaphore_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT, 1, &payload_iovec);

  if (iree_status_is_ok(status)) {
    *out_semaphore = semaphore;
  } else {
    iree_hal_semaphore_release(semaphore);
  }
  return status;
}

static iree_hal_semaphore_compatibility_t
iree_hal_replay_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_device_query_semaphore_compatibility(device->base_device,
                                                       semaphore);
}

static iree_status_t iree_hal_replay_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_QUEUE_POOL_BACKEND,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_query_queue_pool_backend(
                           device->base_device, queue_affinity, out_backend));
}

static iree_status_t iree_hal_replay_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_buffer = NULL;

  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_reserve_object_id(device->recorder, &buffer_id));

  iree_hal_buffer_params_t canonical_params = params;
  iree_hal_buffer_params_canonicalize(&canonical_params);
  iree_hal_replay_device_queue_alloca_payload_t operation_payload;
  memset(&operation_payload, 0, sizeof(operation_payload));
  iree_hal_replay_recorder_allocator_make_allocate_buffer_payload(
      &canonical_params, allocation_size, &operation_payload.allocation);
  operation_payload.queue_affinity = queue_affinity;
  operation_payload.flags = flags;
  operation_payload.wait_semaphore_count = wait_semaphore_list.count;
  operation_payload.signal_semaphore_count = signal_semaphore_list.count;
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t operation_iovecs[3] = {
      iree_make_const_byte_span(&operation_payload, sizeof(operation_payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id, buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA, &pending_record);
  }

  iree_hal_buffer_t* base_buffer = NULL;
  iree_hal_buffer_t* replay_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(device->base_device, queue_affinity,
                                          wait_semaphore_list,
                                          signal_semaphore_list, pool, params,
                                          allocation_size, flags, &base_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_create_proxy(
        device->recorder, device->device_id, buffer_id, base_device,
        base_buffer, device->host_allocator, &replay_buffer);
  }

  iree_hal_replay_buffer_object_payload_t object_payload;
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_make_object_payload(base_buffer,
                                                        &object_payload);
  } else {
    memset(&object_payload, 0, sizeof(object_payload));
  }
  iree_const_byte_span_t object_iovec = iree_make_const_byte_span(
      (const uint8_t*)&object_payload, sizeof(object_payload));
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_creation_operation(
        &pending_record, status, IREE_ARRAYSIZE(operation_iovecs),
        operation_iovecs, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, buffer_id,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT, 1, &object_iovec);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = replay_buffer;
  } else {
    iree_hal_buffer_release(replay_buffer);
  }
  iree_hal_buffer_release(base_buffer);
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_dealloca_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(buffer, 0, iree_hal_buffer_byte_length(buffer)),
      &payload.buffer_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.buffer_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA, &pending_record);
  }
  iree_hal_buffer_t* base_buffer = NULL;
  iree_hal_buffer_t* temporary_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        buffer, device->host_allocator, &base_buffer, &temporary_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_buffer, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_fill_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset, length),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.pattern_length = pattern_length;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      iree_make_const_byte_span(pattern, pattern_length),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FILL,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_FILL, &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_fill(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_target_buffer, target_offset, length,
        pattern, pattern_length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue update length exceeds host size");
  }
  if (IREE_UNLIKELY(length != 0 && !source_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay queue update source buffer is required");
  }

  iree_hal_replay_device_queue_update_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset, length),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.source_offset = source_offset;
  payload.data_length = length;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  const uint8_t* source_data =
      length ? (const uint8_t*)source_buffer + source_offset : NULL;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      iree_make_const_byte_span(source_data, (iree_host_size_t)length),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE, &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, source_buffer, source_offset, base_target_buffer,
        target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_copy_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(source_buffer, source_offset, length),
      &payload.source_ref);
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset, length),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_COPY,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_COPY, &pending_record);
  }
  iree_hal_buffer_t* base_source_buffer = NULL;
  iree_hal_buffer_t* temporary_source_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        source_buffer, device->host_allocator, &base_source_buffer,
        &temporary_source_buffer);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_source_buffer, source_offset,
        base_target_buffer, target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  iree_hal_replay_recorder_buffer_release_temporary(temporary_source_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_read_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.source_file_id =
      iree_hal_replay_recorder_file_id_or_none(source_file);
  if (IREE_UNLIKELY(payload.source_file_id == IREE_HAL_REPLAY_OBJECT_ID_NONE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot record replay queue_read with an unwrapped file");
  }
  payload.source_offset = source_offset;
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset, length),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_byte_span_t captured_data_storage = iree_byte_span_empty();
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  const iree_hal_replay_recorder_options_t* recorder_options =
      iree_hal_replay_recorder_options(device->recorder);
  if (iree_status_is_ok(status) &&
      recorder_options->external_file_policy ==
          IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_RANGES) {
    status = iree_hal_replay_recorder_file_capture_read_data(
        source_file, source_offset, length, device->host_allocator,
        &captured_data_storage);
    payload.captured_data_length = captured_data_storage.data_length;
  }
  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      iree_make_const_byte_span(captured_data_storage.data,
                                captured_data_storage.data_length),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ, &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  iree_hal_file_t* base_source_file = NULL;
  if (iree_status_is_ok(status)) {
    base_source_file = iree_hal_replay_recorder_file_base_or_self(source_file);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_read(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_source_file, source_offset,
        base_target_buffer, target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, captured_data_storage.data);
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_write_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(source_buffer, source_offset, length),
      &payload.source_ref);
  payload.target_file_id =
      iree_hal_replay_recorder_file_id_or_none(target_file);
  if (IREE_UNLIKELY(payload.target_file_id == IREE_HAL_REPLAY_OBJECT_ID_NONE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot record replay queue_write with an unwrapped file");
  }
  payload.target_offset = target_offset;
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.source_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE, &pending_record);
  }
  iree_hal_buffer_t* base_source_buffer = NULL;
  iree_hal_buffer_t* temporary_source_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        source_buffer, device->host_allocator, &base_source_buffer,
        &temporary_source_buffer);
  }
  iree_hal_file_t* base_target_file = NULL;
  if (iree_status_is_ok(status)) {
    base_target_file = iree_hal_replay_recorder_file_base_or_self(target_file);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_source_buffer, source_offset,
        base_target_file, target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_source_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_HOST_CALL,
      &pending_record));
  iree_hal_replay_recorder_mark_unsupported(&pending_record);
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_host_call(
          device->base_device, queue_affinity, wait_semaphore_list,
          signal_semaphore_list, call, args, flags));
}

static iree_status_t iree_hal_replay_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);

  iree_hal_replay_dispatch_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.executable_id =
      iree_hal_replay_recorder_executable_id_or_none(executable);
  payload.queue_affinity = queue_affinity;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_executable_recorded_ordinal(
      executable, function, &payload.function_ordinal));
  payload.flags = flags;
  memcpy(payload.workgroup_size, config.workgroup_size,
         sizeof(payload.workgroup_size));
  memcpy(payload.workgroup_count, config.workgroup_count,
         sizeof(payload.workgroup_count));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      config.workgroup_count_ref, &payload.workgroup_count_ref);
  payload.dynamic_workgroup_local_memory =
      config.dynamic_workgroup_local_memory;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.constants_length = constants.data_length;
  payload.binding_count = bindings.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_hal_replay_buffer_ref_payload_t* binding_payloads = NULL;
  iree_host_size_t binding_payloads_size = 0;
  iree_hal_buffer_ref_list_t base_bindings = bindings;
  iree_hal_buffer_ref_t* binding_storage = NULL;
  iree_hal_buffer_t** temporary_buffers = NULL;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  if (iree_status_is_ok(status) && bindings.count) {
    iree_host_size_t binding_storage_size = 0;
    iree_host_size_t temporary_buffers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*binding_payloads),
                                                  &binding_payloads_size) ||
                      !iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*binding_storage),
                                                  &binding_storage_size) ||
                      !iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*temporary_buffers),
                                                  &temporary_buffers_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay queue dispatch binding count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, binding_payloads_size,
                                (void**)&binding_payloads);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, binding_storage_size,
                                (void**)&binding_storage);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, temporary_buffers_size,
                                (void**)&temporary_buffers);
    }
    if (iree_status_is_ok(status)) {
      memset(temporary_buffers, 0, temporary_buffers_size);
      memcpy(binding_storage, bindings.values, binding_storage_size);
      for (iree_host_size_t i = 0;
           i < bindings.count && iree_status_is_ok(status); ++i) {
        iree_hal_replay_recorder_buffer_ref_make_payload(bindings.values[i],
                                                         &binding_payloads[i]);
        if (binding_storage[i].buffer) {
          status = iree_hal_replay_recorder_buffer_unwrap_for_call(
              binding_storage[i].buffer, device->host_allocator,
              &binding_storage[i].buffer, &temporary_buffers[i]);
        }
      }
      base_bindings.values = binding_storage;
    }
  }

  iree_const_byte_span_t iovecs[5] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      constants,
      iree_make_const_byte_span(binding_payloads, binding_payloads_size),
  };
  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.executable_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH, &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_device_queue_dispatch(
            device->base_device, queue_affinity, wait_semaphore_list,
            signal_semaphore_list,
            iree_hal_replay_recorder_executable_base_or_self(executable),
            function, config, constants, base_bindings, flags),
        IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (temporary_buffers) {
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(temporary_buffers[i]);
    }
  }
  iree_allocator_free(device->host_allocator, temporary_buffers);
  iree_allocator_free(device->host_allocator, binding_storage);
  iree_allocator_free(device->host_allocator, binding_payloads);
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);

  iree_hal_replay_device_queue_execute_payload_t payload = {
      .command_buffer_id =
          iree_hal_replay_recorder_command_buffer_id_or_none(command_buffer),
      .queue_affinity = queue_affinity,
      .flags = flags,
      .wait_semaphore_count = wait_semaphore_list.count,
      .signal_semaphore_count = signal_semaphore_list.count,
      .binding_count = binding_table.count,
  };
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_hal_replay_buffer_ref_payload_t* binding_payloads = NULL;
  iree_host_size_t binding_payloads_size = 0;
  iree_hal_buffer_binding_table_t base_binding_table = binding_table;
  iree_hal_buffer_binding_t* binding_storage = NULL;
  iree_hal_buffer_t** temporary_buffers = NULL;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  if (iree_status_is_ok(status) && binding_table.count) {
    iree_host_size_t binding_storage_size = 0;
    iree_host_size_t temporary_buffers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(binding_table.count,
                                                  sizeof(*binding_payloads),
                                                  &binding_payloads_size) ||
                      !iree_host_size_checked_mul(binding_table.count,
                                                  sizeof(*binding_storage),
                                                  &binding_storage_size) ||
                      !iree_host_size_checked_mul(binding_table.count,
                                                  sizeof(*temporary_buffers),
                                                  &temporary_buffers_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay queue execute binding count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, binding_payloads_size,
                                (void**)&binding_payloads);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, binding_storage_size,
                                (void**)&binding_storage);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, temporary_buffers_size,
                                (void**)&temporary_buffers);
    }
    if (iree_status_is_ok(status)) {
      memset(temporary_buffers, 0, temporary_buffers_size);
      memcpy(binding_storage, binding_table.bindings, binding_storage_size);
      for (iree_host_size_t i = 0;
           i < binding_table.count && iree_status_is_ok(status); ++i) {
        iree_hal_buffer_ref_t binding_ref = iree_hal_make_buffer_ref(
            binding_table.bindings[i].buffer, binding_table.bindings[i].offset,
            binding_table.bindings[i].length);
        iree_hal_replay_recorder_buffer_ref_make_payload(binding_ref,
                                                         &binding_payloads[i]);
        if (binding_storage[i].buffer) {
          status = iree_hal_replay_recorder_buffer_unwrap_for_call(
              binding_storage[i].buffer, device->host_allocator,
              &binding_storage[i].buffer, &temporary_buffers[i]);
        }
      }
      base_binding_table.bindings = binding_storage;
    }
  }

  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      iree_make_const_byte_span(binding_payloads, binding_payloads_size),
  };
  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.command_buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE, &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_device_queue_execute(
            device->base_device, queue_affinity, wait_semaphore_list,
            signal_semaphore_list,
            iree_hal_replay_recorder_command_buffer_base_or_self(
                command_buffer),
            base_binding_table, flags),
        IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (temporary_buffers) {
    for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(temporary_buffers[i]);
    }
  }
  iree_allocator_free(device->host_allocator, temporary_buffers);
  iree_allocator_free(device->host_allocator, binding_storage);
  iree_allocator_free(device->host_allocator, binding_payloads);
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_atomic_wait(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_atomic_wait_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset,
                               iree_hal_atomic_width_byte_count(params.width)),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.params.value = params.value;
  payload.params.mask = params.mask;
  payload.params.flags = params.flags;
  payload.params.width = params.width;
  payload.params.condition = params.condition;
  payload.params.reserved0 = params.reserved;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_WAIT,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT, &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_atomic_wait(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_target_buffer, target_offset, params);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_atomic_store(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_atomic_store_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset,
                               iree_hal_atomic_width_byte_count(params.width)),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.params.value = params.value;
  payload.params.flags = params.flags;
  payload.params.width = params.width;
  memcpy(payload.params.reserved0, params.reserved,
         sizeof(payload.params.reserved0));

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
        &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_atomic_store(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_target_buffer, target_offset, params);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_atomic_rmw(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_atomic_rmw_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset,
                               iree_hal_atomic_width_byte_count(params.width)),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.params.operand = params.operand;
  payload.params.flags = params.flags;
  payload.params.width = params.width;
  payload.params.operation = params.operation;
  payload.params.reserved0 = params.reserved;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_RMW,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_RMW, &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_atomic_rmw(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_target_buffer, target_offset, params);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_timestamp(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_TIMESTAMP,
      &pending_record));
  iree_hal_replay_recorder_mark_unsupported(&pending_record);
  iree_hal_buffer_t* base_buffer = NULL;
  iree_hal_buffer_t* temporary_buffer = NULL;
  iree_status_t status = iree_hal_replay_recorder_buffer_unwrap_for_call(
      target_buffer, device->host_allocator, &base_buffer, &temporary_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_timestamp(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_buffer, target_offset, flags);
  }
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  iree_hal_replay_recorder_buffer_release_temporary(temporary_buffer);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FLUSH,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_flush(device->base_device, queue_affinity));
}

static iree_status_t iree_hal_replay_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_BEGIN,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_profiling_begin(device->base_device, options));
}

static iree_status_t iree_hal_replay_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_FLUSH,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_profiling_flush(device->base_device));
}

static iree_status_t iree_hal_replay_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_END,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_profiling_end(device->base_device));
}

static iree_status_t iree_hal_replay_wrap_device(
    iree_hal_replay_recorder_t* recorder, iree_hal_device_group_t* base_group,
    iree_hal_device_t* base_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_group);
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_device_spec_t* base_device_spec =
      iree_hal_device_spec(base_device);
  const iree_hal_device_queue_spec_t* base_queue_spec =
      base_device_spec ? iree_hal_device_spec_queues(base_device_spec) : NULL;
  const iree_host_size_t queue_family_count =
      base_queue_spec ? base_queue_spec->family_count : 0;
  iree_host_size_t queue_count = 0;
  for (iree_host_size_t i = 0; i < queue_family_count; ++i) {
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            queue_count, base_queue_spec->families[i].provisioned_queue_count,
            &queue_count))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "recording queue table size overflow");
    }
  }

  iree_host_size_t queue_families_offset = 0;
  iree_host_size_t queues_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          sizeof(iree_hal_replay_device_t), &total_size,
          IREE_STRUCT_FIELD_ALIGNED(
              queue_family_count, iree_hal_replay_queue_family_t,
              iree_alignof(iree_hal_replay_queue_family_t),
              &queue_families_offset),
          IREE_STRUCT_FIELD_ALIGNED(
              queue_count, iree_hal_replay_recorder_queue_t,
              iree_alignof(iree_hal_replay_recorder_queue_t), &queues_offset)));
  iree_hal_replay_device_t* device = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_replay_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  device->recorder = recorder;
  iree_hal_replay_recorder_retain(device->recorder);
  device->base_group = base_group;
  iree_hal_device_group_retain(device->base_group);
  device->base_device = base_device;
  iree_hal_device_retain(device->base_device);

  device->queue_family_count = queue_family_count;
  device->queue_families =
      queue_family_count
          ? (iree_hal_replay_queue_family_t*)((uint8_t*)device +
                                              queue_families_offset)
          : NULL;
  device->queues = queue_count
                       ? (iree_hal_replay_recorder_queue_t*)((uint8_t*)device +
                                                             queues_offset)
                       : NULL;

  iree_status_t status = iree_ok_status();
  iree_host_size_t flat_queue_ordinal = 0;
  for (iree_host_size_t i = 0;
       i < queue_family_count && iree_status_is_ok(status); ++i) {
    const iree_hal_queue_family_ordinal_t family_ordinal =
        (iree_hal_queue_family_ordinal_t)i;
    if (IREE_UNLIKELY(
            !iree_hal_device_queue_family(base_device, family_ordinal))) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "wrapped device queue spec advertises family %" PRIhsz
          " but the device did not expose it",
          i);
      break;
    }
    const uint32_t family_queue_count =
        base_queue_spec->families[i].provisioned_queue_count;
    iree_hal_replay_queue_family_t* queue_family = &device->queue_families[i];
    iree_hal_queue_family_initialize(family_ordinal, &queue_family->base);
    queue_family->queue_offset = flat_queue_ordinal;
    queue_family->queue_count = family_queue_count;
    for (uint32_t j = 0; j < family_queue_count && iree_status_is_ok(status);
         ++j) {
      iree_hal_queue_t* base_queue = iree_hal_device_queue(
          base_device, family_ordinal, (iree_hal_queue_ordinal_t)j);
      if (IREE_UNLIKELY(!base_queue)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "wrapped device queue spec advertises family %" PRIhsz
            " queue %u but the device did not expose it",
            i, j);
        break;
      }
      ++flat_queue_ordinal;
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_record_object(
        recorder, IREE_HAL_REPLAY_OBJECT_ID_NONE,
        IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE, IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        0, NULL, &device->device_id);
  }
  flat_queue_ordinal = 0;
  for (iree_host_size_t i = 0;
       i < queue_family_count && iree_status_is_ok(status); ++i) {
    const iree_hal_queue_family_ordinal_t family_ordinal =
        (iree_hal_queue_family_ordinal_t)i;
    iree_hal_replay_queue_family_t* queue_family = &device->queue_families[i];
    for (uint32_t j = 0;
         j < queue_family->queue_count && iree_status_is_ok(status); ++j) {
      const iree_hal_queue_ordinal_t queue_ordinal =
          (iree_hal_queue_ordinal_t)j;
      const iree_hal_replay_provisioned_queue_object_payload_t payload = {
          .family_ordinal = family_ordinal,
          .queue_ordinal = queue_ordinal,
      };
      const iree_const_byte_span_t payload_iovec =
          iree_make_const_byte_span(&payload, sizeof(payload));
      iree_hal_replay_object_id_t queue_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
      status = iree_hal_replay_recorder_record_object(
          recorder, device->device_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
          IREE_HAL_REPLAY_PAYLOAD_TYPE_PROVISIONED_QUEUE_OBJECT, 1,
          &payload_iovec, &queue_id);
      if (iree_status_is_ok(status)) {
        iree_hal_queue_t* base_queue =
            iree_hal_device_queue(base_device, family_ordinal, queue_ordinal);
        iree_hal_replay_recorder_queue_initialize(
            &queue_family->base, recorder, device->device_id, queue_id,
            base_queue, host_allocator, &device->queues[flat_queue_ordinal++]);
        ++device->initialized_queue_count;
      }
    }
  }
  iree_hal_allocator_t* base_allocator =
      iree_status_is_ok(status) ? iree_hal_device_allocator(device->base_device)
                                : NULL;
  if (iree_status_is_ok(status) && base_allocator) {
    status = iree_hal_replay_recorder_wrap_allocator(
        recorder, device->device_id, (iree_hal_device_t*)device, base_allocator,
        host_allocator, &device->allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_replay_wrap_device_group_context_t {
  // Shared recorder receiving all wrapper records.
  iree_hal_replay_recorder_t* recorder;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
} iree_hal_replay_wrap_device_group_context_t;

static iree_status_t iree_hal_replay_wrap_device_group_device(
    iree_hal_device_group_t* source_group, iree_host_size_t device_index,
    iree_hal_device_t* source_device, void* user_data,
    iree_hal_device_t** out_replacement_device) {
  (void)device_index;
  iree_hal_replay_wrap_device_group_context_t* context =
      (iree_hal_replay_wrap_device_group_context_t*)user_data;
  return iree_hal_replay_wrap_device(context->recorder, source_group,
                                     source_device, context->host_allocator,
                                     out_replacement_device);
}

IREE_API_EXPORT iree_status_t iree_hal_replay_wrap_device_group(
    iree_hal_replay_recorder_t* recorder, iree_hal_device_group_t* base_group,
    iree_allocator_t host_allocator, iree_hal_device_group_t** out_group) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_group);
  IREE_ASSERT_ARGUMENT(out_group);
  *out_group = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_replay_wrap_device_group_context_t context = {
      .recorder = recorder,
      .host_allocator = host_allocator,
  };
  iree_hal_device_group_replacement_callback_t replacement_callback = {
      .fn = iree_hal_replay_wrap_device_group_device,
      .user_data = &context,
  };
  iree_status_t status = iree_hal_device_group_create_with_replacements(
      base_group, replacement_callback, host_allocator, out_group);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_replay_device_vtable = {
    .destroy = iree_hal_replay_device_destroy,
    .id = iree_hal_replay_device_id,
    .host_allocator = iree_hal_replay_device_host_allocator,
    .device_allocator = iree_hal_replay_device_allocator,
    .replace_channel_provider = iree_hal_replay_replace_channel_provider,
    .trim = iree_hal_replay_device_trim,
    .device_spec = iree_hal_replay_device_spec,
    .queue_family = iree_hal_replay_device_queue_family,
    .queue = iree_hal_replay_device_queue,
    .sample_observation = iree_hal_replay_device_sample_observation,
    .topology_info = iree_hal_replay_device_topology_info,
    .refine_topology_edge = iree_hal_replay_device_refine_topology_edge,
    .assign_topology_info = iree_hal_replay_device_assign_topology_info,
    .create_channel = iree_hal_replay_device_create_channel,
    .create_command_buffer = iree_hal_replay_device_create_command_buffer,
    .load_executable = iree_hal_replay_device_load_executable,
    .import_file = iree_hal_replay_device_import_file,
    .create_semaphore = iree_hal_replay_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_replay_device_query_semaphore_compatibility,
    .query_queue_pool_backend = iree_hal_replay_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_replay_device_queue_alloca,
    .queue_dealloca = iree_hal_replay_device_queue_dealloca,
    .queue_fill = iree_hal_replay_device_queue_fill,
    .queue_update = iree_hal_replay_device_queue_update,
    .queue_copy = iree_hal_replay_device_queue_copy,
    .queue_read = iree_hal_replay_device_queue_read,
    .queue_write = iree_hal_replay_device_queue_write,
    .queue_host_call = iree_hal_replay_device_queue_host_call,
    .queue_dispatch = iree_hal_replay_device_queue_dispatch,
    .queue_execute = iree_hal_replay_device_queue_execute,
    .queue_atomic_wait = iree_hal_replay_device_queue_atomic_wait,
    .queue_atomic_store = iree_hal_replay_device_queue_atomic_store,
    .queue_atomic_rmw = iree_hal_replay_device_queue_atomic_rmw,
    .queue_timestamp = iree_hal_replay_device_queue_timestamp,
    .queue_flush = iree_hal_replay_device_queue_flush,
    .profiling_begin = iree_hal_replay_device_profiling_begin,
    .profiling_flush = iree_hal_replay_device_profiling_flush,
    .profiling_end = iree_hal_replay_device_profiling_end,
};
