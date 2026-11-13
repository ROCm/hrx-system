// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/code.h"

#include "iree/hal/driver/amd/status.h"

struct iree_hal_amd_gpu_code_upload_t {
  // Allocator owning this temporary initialization image.
  iree_allocator_t host_allocator;
  // Initialized extent, rounded to a DWORD for native writes.
  iree_host_size_t byte_length;
  // Ordinary host storage; only captured command bytes become GPU operands.
  uint32_t contents[];
};

void iree_hal_amd_gpu_code_upload_free(iree_hal_amd_gpu_code_upload_t* upload) {
  if (upload) {
    iree_allocator_free(upload->host_allocator, upload);
  }
}

iree_status_t iree_hal_amd_gpu_code_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* resident, uint64_t byte_length,
    uint64_t alignment, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_memory_t* out_memory,
    iree_hal_amd_gpu_code_upload_t** out_upload,
    iree_byte_span_t* out_contents) {
  *out_memory = (iree_hal_amd_fabric_memory_t){0};
  *out_upload = NULL;
  *out_contents = iree_byte_span_empty();
  iree_host_size_t storage_length = 0;
  if (byte_length > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_add(sizeof(iree_hal_amd_gpu_code_upload_t) + 3,
                                  (iree_host_size_t)byte_length,
                                  &storage_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "code initialization exceeds host address range");
  }
  const iree_host_size_t padded_length =
      iree_host_align((iree_host_size_t)byte_length, 4);
  iree_hal_amd_gpu_code_upload_t* upload = NULL;
  if (!resident->host_access) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        host_allocator, storage_length, (void**)&upload));
    upload->host_allocator = host_allocator;
    upload->byte_length = padded_length;
    memset((uint8_t*)upload->contents + byte_length, 0,
           padded_length - byte_length);
  }
  iree_hal_amd_fabric_memory_t memory = {0};
  // Native serialization orders this one allocation independently of slab
  // batches. Unmapped code needs no additional pin, map, or device attachment.
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      context->api, resident, padded_length, alignment, context->event_sink,
      &memory, NULL);
  if (iree_status_is_ok(status)) {
    *out_memory = memory;
    *out_upload = upload;
    *out_contents = iree_make_byte_span(
        upload ? (uint8_t*)upload->contents : memory.contents.data,
        (iree_host_size_t)byte_length);
  } else {
    iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                            &memory);
    iree_hal_amd_gpu_code_upload_free(upload);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_code_publish(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_code_queue_t* queue,
    iree_timeout_t timeout, iree_hal_amd_fabric_memory_t* memory,
    iree_hal_amd_gpu_code_upload_t** upload_ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_convert_timeout_to_absolute(&timeout);
  iree_hal_amd_gpu_code_upload_t* upload = *upload_ptr;
  *upload_ptr = NULL;
  iree_const_byte_span_t contents = iree_const_byte_span_empty();
  iree_status_t status = iree_ok_status();
  if (upload) {
    // Construction selects the publisher before exposing unmapped code.
    IREE_ASSERT(queue);
    contents = iree_make_const_byte_span(upload->contents, upload->byte_length);
  } else {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->host_mapping_cache_control(
            memory->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
            memory->contents.data_length),
        "publish code host writes");
  }
  if (iree_status_is_ok(status) && queue) {
    status =
        iree_hal_amd_gpu_code_queue_publish(queue, timeout, memory, contents);
  }
  iree_hal_amd_gpu_code_upload_free(upload);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
