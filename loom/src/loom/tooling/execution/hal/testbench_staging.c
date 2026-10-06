// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_staging.h"

static iree_status_t loom_run_hal_testbench_staging_transfer(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_testbench_staging_t* staging,
    iree_hal_transfer_operation_type_t type) {
  if (staging->transfer_count == 0) {
    return iree_ok_status();
  }
  iree_hal_semaphore_t* completion = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      runtime->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion));
  iree_hal_transfer_operation_t* operations = NULL;
  iree_hal_buffer_mapping_t* mappings = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      staging->host_allocator, staging->transfer_count, sizeof(*operations),
      (void**)&operations);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(staging->host_allocator,
                                         staging->transfer_count,
                                         sizeof(*mappings), (void**)&mappings);
  }
  iree_host_size_t mapping_count = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < staging->transfer_count; ++i) {
    const iree_hal_transfer_operation_t* transfer = &staging->transfers[i];
    const iree_hal_memory_access_t access =
        type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD
            ? IREE_HAL_MEMORY_ACCESS_READ
            : IREE_HAL_MEMORY_ACCESS_WRITE;
    status = iree_hal_buffer_map_range(
        transfer->copy.source_buffer, IREE_HAL_MAPPING_MODE_SCOPED, access,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, transfer->copy.length, &mappings[i]);
    if (iree_status_is_ok(status)) {
      ++mapping_count;
      operations[i].type = type;
      if (type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
        operations[i].upload.source = mappings[i].contents.data;
        operations[i].upload.target_buffer = transfer->copy.target_buffer;
        operations[i].upload.length = transfer->copy.length;
      } else {
        operations[i].download.source_buffer = transfer->copy.target_buffer;
        operations[i].download.target = mappings[i].contents.data;
        operations[i].download.length = transfer->copy.length;
      }
    }
  }
  uint64_t completion_value = 1;
  const iree_hal_semaphore_list_t signal_semaphore_list = {
      .count = 1,
      .semaphores = &completion,
      .payload_values = &completion_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_transfer(
        runtime->transfer_queue, iree_hal_semaphore_list_empty(),
        signal_semaphore_list, staging->transfer_count, operations,
        /*barriers=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_semaphore_wait(completion, completion_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE);
  }
  for (iree_host_size_t i = 0; i < mapping_count; ++i) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&mappings[i]));
  }
  iree_allocator_free(staging->host_allocator, mappings);
  iree_allocator_free(staging->host_allocator, operations);
  iree_hal_semaphore_release(completion);
  return status;
}

iree_status_t loom_run_hal_testbench_staging_initialize(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t binding_count,
    iree_hal_buffer_binding_t* bindings, iree_allocator_t host_allocator,
    loom_run_hal_testbench_staging_t* out_staging) {
  *out_staging = (loom_run_hal_testbench_staging_t){
      .host_allocator = host_allocator,
  };
  const iree_hal_buffer_params_t device_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < binding_count;
       ++i) {
    iree_hal_buffer_binding_t* binding = &bindings[i];
    if (iree_hal_buffer_allocation_placement(binding->buffer).device ==
            runtime->device &&
        iree_all_bits_set(iree_hal_buffer_memory_type(binding->buffer),
                          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
      continue;
    }
    if (out_staging->transfers == NULL) {
      status = iree_allocator_malloc_array(host_allocator, binding_count,
                                           sizeof(*out_staging->transfers),
                                           (void**)&out_staging->transfers);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }

    iree_hal_buffer_t* allocation =
        iree_hal_buffer_allocated_buffer(binding->buffer);
    iree_host_size_t transfer_index = 0;
    while (transfer_index < out_staging->transfer_count &&
           out_staging->transfers[transfer_index].copy.source_buffer !=
               allocation) {
      ++transfer_index;
    }
    iree_hal_transfer_operation_t* transfer =
        &out_staging->transfers[transfer_index];
    if (transfer_index == out_staging->transfer_count) {
      *transfer = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .copy =
              {
                  .source_buffer = allocation,
                  .length = iree_hal_buffer_byte_length(allocation),
              },
      };
      status = iree_hal_allocator_allocate_buffer(
          iree_hal_device_allocator(runtime->device), device_params,
          transfer->copy.length, &transfer->copy.target_buffer);
      if (iree_status_is_ok(status)) {
        ++out_staging->transfer_count;
      }
    }
    if (iree_status_is_ok(status)) {
      binding->offset += iree_hal_buffer_byte_offset(binding->buffer);
      binding->buffer = transfer->copy.target_buffer;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_staging_transfer(
        runtime, out_staging, IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD);
  }
  return status;
}

iree_status_t loom_run_hal_testbench_staging_readback(
    const loom_run_hal_runtime_t* runtime,
    loom_run_hal_testbench_staging_t* staging) {
  return loom_run_hal_testbench_staging_transfer(
      runtime, staging, IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD);
}

void loom_run_hal_testbench_staging_deinitialize(
    loom_run_hal_testbench_staging_t* staging) {
  for (iree_host_size_t i = 0; i < staging->transfer_count; ++i) {
    iree_hal_buffer_release(staging->transfers[i].copy.target_buffer);
  }
  iree_allocator_free(staging->host_allocator, staging->transfers);
  *staging = (loom_run_hal_testbench_staging_t){0};
}
