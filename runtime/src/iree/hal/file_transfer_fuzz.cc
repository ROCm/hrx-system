// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "iree/hal/file.h"

// Division supplies an independent overflow-safe oracle for addressed bounds.
static bool Fits(uint64_t offset, uint64_t row_length, uint64_t row_count,
                 uint64_t slice_count, iree_hal_transfer_strides_t strides,
                 uint64_t length) {
  if (offset > length || row_length > length - offset) {
    return false;
  }
  uint64_t remaining = length - offset - row_length;
  if (row_count > 1) {
    if (strides.row_stride > remaining / (row_count - 1)) {
      return false;
    }
    remaining -= strides.row_stride * (row_count - 1);
  }
  return slice_count == 1 ||
         strides.slice_stride <= remaining / (slice_count - 1);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < sizeof(iree_hal_file_transfer_record_t)) {
    return 0;
  }
  iree_hal_file_transfer_record_t record;
  memcpy(&record, data, sizeof(record));
  iree_hal_file_transfer_t transfer;
  iree_status_t status =
      iree_hal_file_transfer_record_decode(&record, &transfer);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return 0;
  }
  uint64_t file_length = UINT64_MAX;
  iree_device_size_t buffer_length = IREE_DEVICE_SIZE_MAX;
  if (size >= sizeof(record) + 16) {
    file_length = iree_unaligned_load_le_u64(data + sizeof(record));
    buffer_length = (iree_device_size_t)iree_unaligned_load_le_u64(
        data + sizeof(record) + 8);
  }
  const iree_hal_file_binding_t file = {nullptr, 0, file_length};
  const iree_hal_buffer_binding_t buffer = {nullptr, 0, buffer_length};
  const bool expected =
      !transfer.row_length ||
      (transfer.file_binding == 0 && transfer.buffer_binding == 0 &&
       Fits(transfer.file_offset, transfer.row_length, transfer.row_count,
            transfer.slice_count, transfer.file_strides, file_length) &&
       Fits(transfer.buffer_offset, transfer.row_length, transfer.row_count,
            transfer.slice_count, transfer.buffer_strides, buffer_length));
  status = iree_hal_file_transfer_validate({1, &file}, {1, &buffer}, &transfer);
  const bool valid = iree_status_is_ok(status);
  iree_status_free(status);
  if (valid != expected) {
    abort();
  }
  return 0;
}
