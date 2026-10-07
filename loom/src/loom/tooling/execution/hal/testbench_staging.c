// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_staging.h"

enum {
  LOOM_RUN_HAL_STAGING_ALLOCATION_READY = 1,
  LOOM_RUN_HAL_STAGING_UPLOAD_READY = 2,
  LOOM_RUN_HAL_STAGING_DOWNLOAD_READY = 3,
  LOOM_RUN_HAL_STAGING_RETIREMENT_READY = 1,
};

static bool loom_run_hal_testbench_binding_is_local(
    const loom_run_hal_runtime_t* runtime,
    const iree_hal_buffer_binding_t* binding) {
  return iree_hal_buffer_allocation_placement(binding->buffer).device ==
             runtime->device &&
         iree_all_bits_set(iree_hal_buffer_memory_type(binding->buffer),
                           IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
}

static iree_status_t loom_run_hal_testbench_staging_allocate_metadata(
    iree_host_size_t capacity, loom_run_hal_testbench_staging_t* staging) {
  iree_host_size_t storage_size = 0;
  iree_host_size_t transfers_offset = 0;
  iree_host_size_t operations_offset = 0;
  iree_host_size_t mappings_offset = 0;
  iree_host_size_t requests_offset = 0;
  iree_host_size_t buffers_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(capacity, iree_hal_transfer_operation_t,
                        &transfers_offset),
      IREE_STRUCT_FIELD(capacity, iree_hal_transfer_operation_t,
                        &operations_offset),
      IREE_STRUCT_FIELD(capacity, iree_hal_buffer_mapping_t, &mappings_offset),
      IREE_STRUCT_FIELD(capacity, iree_hal_pool_reservation_request_t,
                        &requests_offset),
      IREE_STRUCT_FIELD(capacity, iree_hal_buffer_t*, &buffers_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(staging->host_allocator,
                                             storage_size, (void**)&storage));
  memset(storage, 0, storage_size);
  staging->copies.storage = storage;
  staging->copies.transfers =
      (iree_hal_transfer_operation_t*)(storage + transfers_offset);
  staging->copies.operations =
      (iree_hal_transfer_operation_t*)(storage + operations_offset);
  staging->copies.mappings =
      (iree_hal_buffer_mapping_t*)(storage + mappings_offset);
  staging->copies.requests =
      (iree_hal_pool_reservation_request_t*)(storage + requests_offset);
  staging->copies.buffers = (iree_hal_buffer_t**)(storage + buffers_offset);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_staging_prepare_transfers(
    loom_run_hal_testbench_staging_t* staging,
    iree_hal_transfer_operation_type_t type,
    iree_host_size_t* out_mapping_count) {
  *out_mapping_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < staging->copies.count; ++i) {
    const iree_hal_transfer_operation_t* transfer =
        &staging->copies.transfers[i];
    const iree_hal_memory_access_t access =
        type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD
            ? IREE_HAL_MEMORY_ACCESS_READ
            : IREE_HAL_MEMORY_ACCESS_WRITE;
    status = iree_hal_buffer_map_range(
        transfer->copy.source_buffer, IREE_HAL_MAPPING_MODE_SCOPED, access,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, transfer->copy.length,
        &staging->copies.mappings[i]);
    if (iree_status_is_ok(status)) {
      ++*out_mapping_count;
      iree_hal_transfer_operation_t* operation = &staging->copies.operations[i];
      *operation = (iree_hal_transfer_operation_t){.type = type};
      if (type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
        operation->upload.source = staging->copies.mappings[i].contents.data;
        operation->upload.target_buffer = transfer->copy.target_buffer;
        operation->upload.length = transfer->copy.length;
      } else {
        operation->download.source_buffer = transfer->copy.target_buffer;
        operation->download.target = staging->copies.mappings[i].contents.data;
        operation->download.length = transfer->copy.length;
      }
    }
  }
  return status;
}

static iree_status_t loom_run_hal_testbench_staging_unmap(
    loom_run_hal_testbench_staging_t* staging, iree_host_size_t mapping_count,
    iree_status_t status) {
  for (iree_host_size_t i = 0; i < mapping_count; ++i) {
    status = iree_status_join(
        status, iree_hal_buffer_unmap_range(&staging->copies.mappings[i]));
  }
  return status;
}

static iree_status_t loom_run_hal_testbench_staging_upload(
    loom_run_hal_testbench_staging_t* staging) {
  iree_host_size_t mapping_count = 0;
  iree_status_t status = loom_run_hal_testbench_staging_prepare_transfers(
      staging, IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD, &mapping_count);
  bool transfer_submitted = false;
  uint64_t allocation_value = LOOM_RUN_HAL_STAGING_ALLOCATION_READY;
  uint64_t upload_value = LOOM_RUN_HAL_STAGING_UPLOAD_READY;
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_transfer(
        staging->runtime->transfer_queue,
        (iree_hal_semaphore_list_t){
            .count = 1,
            .semaphores = &staging->allocation.progress_semaphore,
            .payload_values = &allocation_value,
        },
        (iree_hal_semaphore_list_t){
            .count = 1,
            .semaphores = &staging->allocation.progress_semaphore,
            .payload_values = &upload_value,
        },
        staging->copies.count, staging->copies.operations,
        /*barriers=*/NULL);
    transfer_submitted = iree_status_is_ok(status);
  }
  const uint64_t terminal_value =
      transfer_submitted ? upload_value : allocation_value;
  status = iree_status_join(
      status, loom_run_hal_semaphore_wait(
                  staging->allocation.progress_semaphore, terminal_value,
                  iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  return loom_run_hal_testbench_staging_unmap(staging, mapping_count, status);
}

iree_status_t loom_run_hal_testbench_staging_initialize(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t binding_count,
    iree_hal_buffer_binding_t* bindings, iree_allocator_t host_allocator,
    loom_run_hal_testbench_staging_t* out_staging) {
  *out_staging = (loom_run_hal_testbench_staging_t){
      .runtime = runtime,
      .host_allocator = host_allocator,
  };

  bool requires_staging = false;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    requires_staging |=
        !loom_run_hal_testbench_binding_is_local(runtime, &bindings[i]);
  }
  if (!requires_staging) {
    return iree_ok_status();
  }

  iree_status_t status = loom_run_hal_testbench_staging_allocate_metadata(
      binding_count, out_staging);
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < binding_count;
       ++i) {
    iree_hal_buffer_binding_t* binding = &bindings[i];
    if (loom_run_hal_testbench_binding_is_local(runtime, binding)) {
      continue;
    }
    iree_hal_buffer_t* allocation =
        iree_hal_buffer_allocated_buffer(binding->buffer);
    iree_host_size_t transfer_index = 0;
    while (transfer_index < out_staging->copies.count &&
           out_staging->copies.transfers[transfer_index].copy.source_buffer !=
               allocation) {
      ++transfer_index;
    }
    if (transfer_index == out_staging->copies.count) {
      out_staging->copies.transfers[transfer_index] =
          (iree_hal_transfer_operation_t){
              .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
              .copy =
                  {
                      .source_buffer = allocation,
                      .length = iree_hal_buffer_byte_length(allocation),
                  },
          };
      out_staging->copies.requests[transfer_index] =
          (iree_hal_pool_reservation_request_t){
              .params =
                  {
                      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
                      .access = IREE_HAL_MEMORY_ACCESS_ALL,
                      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                      .queue_family_affinity =
                          IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                  },
              .allocation_size = iree_hal_buffer_byte_length(allocation),
          };
      ++out_staging->copies.count;
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        runtime->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &out_staging->allocation.progress_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        runtime->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &out_staging->allocation.retirement_semaphore);
  }
  uint64_t allocation_value = LOOM_RUN_HAL_STAGING_ALLOCATION_READY;
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_alloca(
        runtime->transfer_queue, iree_hal_semaphore_list_empty(),
        (iree_hal_semaphore_list_t){
            .count = 1,
            .semaphores = &out_staging->allocation.progress_semaphore,
            .payload_values = &allocation_value,
        },
        runtime->staging_pool, out_staging->copies.count,
        out_staging->copies.requests, out_staging->copies.buffers);
    out_staging->allocation.live = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < out_staging->copies.count; ++i) {
      out_staging->copies.transfers[i].copy.target_buffer =
          out_staging->copies.buffers[i];
    }
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      iree_hal_buffer_binding_t* binding = &bindings[i];
      if (loom_run_hal_testbench_binding_is_local(runtime, binding)) {
        continue;
      }
      iree_hal_buffer_t* allocation =
          iree_hal_buffer_allocated_buffer(binding->buffer);
      iree_host_size_t transfer_index = 0;
      while (transfer_index < out_staging->copies.count &&
             out_staging->copies.transfers[transfer_index].copy.source_buffer !=
                 allocation) {
        ++transfer_index;
      }
      IREE_ASSERT(transfer_index < out_staging->copies.count);
      binding->offset += iree_hal_buffer_byte_offset(binding->buffer);
      binding->buffer = out_staging->copies.buffers[transfer_index];
    }
    status = loom_run_hal_testbench_staging_upload(out_staging);
    if (!iree_status_is_ok(status)) {
      // The upload helper waits for every accepted operation to become
      // terminal. Wrapper destruction can therefore return any committed
      // reservation without manufacturing a second failed dependency edge.
      out_staging->allocation.live = false;
    }
  }
  return status;
}

iree_status_t loom_run_hal_testbench_staging_readback(
    const loom_run_hal_runtime_t* runtime,
    loom_run_hal_testbench_staging_t* staging) {
  if (staging->copies.count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT(runtime == staging->runtime);
  IREE_ASSERT(staging->allocation.live);

  iree_host_size_t mapping_count = 0;
  iree_status_t status = loom_run_hal_testbench_staging_prepare_transfers(
      staging, IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD, &mapping_count);
  bool download_submitted = false;
  uint64_t download_value = LOOM_RUN_HAL_STAGING_DOWNLOAD_READY;
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_transfer(
        runtime->transfer_queue, iree_hal_semaphore_list_empty(),
        (iree_hal_semaphore_list_t){
            .count = 1,
            .semaphores = &staging->allocation.progress_semaphore,
            .payload_values = &download_value,
        },
        staging->copies.count, staging->copies.operations,
        /*barriers=*/NULL);
    download_submitted = iree_status_is_ok(status);
  }

  bool deallocation_submitted = false;
  uint64_t retirement_value = LOOM_RUN_HAL_STAGING_RETIREMENT_READY;
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_dealloca(
        runtime->transfer_queue,
        (iree_hal_semaphore_list_t){
            .count = 1,
            .semaphores = &staging->allocation.progress_semaphore,
            .payload_values = &download_value,
        },
        (iree_hal_semaphore_list_t){
            .count = 1,
            .semaphores = &staging->allocation.retirement_semaphore,
            .payload_values = &retirement_value,
        },
        staging->copies.count, staging->copies.buffers);
    deallocation_submitted = iree_status_is_ok(status);
  }
  if (deallocation_submitted) {
    status = iree_status_join(
        status, loom_run_hal_semaphore_wait(
                    staging->allocation.retirement_semaphore, retirement_value,
                    iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    staging->allocation.live = false;
  } else if (download_submitted) {
    status = iree_status_join(
        status, loom_run_hal_semaphore_wait(
                    staging->allocation.progress_semaphore, download_value,
                    iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
  return loom_run_hal_testbench_staging_unmap(staging, mapping_count, status);
}

iree_status_t loom_run_hal_testbench_staging_deinitialize(
    loom_run_hal_testbench_staging_t* staging) {
  iree_status_t status = iree_ok_status();
  if (staging->allocation.live) {
    uint64_t retirement_value = LOOM_RUN_HAL_STAGING_RETIREMENT_READY;
    status = iree_hal_queue_dealloca(
        staging->runtime->transfer_queue, iree_hal_semaphore_list_empty(),
        (iree_hal_semaphore_list_t){
            .count = 1,
            .semaphores = &staging->allocation.retirement_semaphore,
            .payload_values = &retirement_value,
        },
        staging->copies.count, staging->copies.buffers);
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_semaphore_wait(
          staging->allocation.retirement_semaphore, retirement_value,
          iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    }
    staging->allocation.live = false;
  }
  for (iree_host_size_t i = 0; i < staging->copies.count; ++i) {
    iree_hal_buffer_release(staging->copies.buffers[i]);
  }
  iree_hal_semaphore_release(staging->allocation.retirement_semaphore);
  iree_hal_semaphore_release(staging->allocation.progress_semaphore);
  iree_allocator_free(staging->host_allocator, staging->copies.storage);
  *staging = (loom_run_hal_testbench_staging_t){0};
  return status;
}
