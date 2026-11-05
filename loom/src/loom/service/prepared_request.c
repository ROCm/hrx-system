// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/prepared_request.h"

#include <string.h>

iree_status_t loom_service_prepared_requests_pack(
    iree_host_size_t count, const loom_service_prepared_request_t* requests,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_ranges,
    iree_vm_buffer_t** out_packed) {
  *out_ranges = NULL;
  *out_packed = NULL;
  if (count > IREE_HOST_SIZE_MAX / 16) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "prepared request ranges exceed host storage");
  }
  const iree_host_size_t directory_length = count * 16;
  iree_host_size_t length = directory_length;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t size = iree_vm_buffer_length(requests[i].bytes);
    if (size > IREE_HOST_SIZE_MAX - length) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "prepared cohort exceeds host address space");
    }
    length += size;
  }
  iree_vm_buffer_t* ranges = NULL;
  iree_vm_buffer_t* packed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_create(directory_length, 8, host_allocator, &ranges));
  iree_status_t status =
      iree_vm_buffer_create(length, 8, host_allocator, &packed);
  iree_byte_span_t records = iree_byte_span_empty();
  iree_byte_span_t target = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(ranges, 0, directory_length, &records);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(packed, 0, length, &target);
  }
  iree_host_size_t offset = directory_length;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_const_byte_span_t input;
    status = iree_vm_buffer_map_read(
        requests[i].bytes, 0, iree_vm_buffer_length(requests[i].bytes), &input);
    if (iree_status_is_ok(status)) {
      iree_unaligned_store_le_u64(records.data + i * 16, offset);
      iree_unaligned_store_le_u64(records.data + i * 16 + 8, input.data_length);
      if (input.data_length) {
        memcpy(target.data + offset, input.data, input.data_length);
      }
      offset += input.data_length;
    }
  }
  if (iree_status_is_ok(status)) {
    if (directory_length) {
      memcpy(target.data, records.data, directory_length);
    }
    *out_ranges = ranges;
    *out_packed = packed;
  } else {
    iree_vm_buffer_release(ranges);
    iree_vm_buffer_release(packed);
  }
  return status;
}
