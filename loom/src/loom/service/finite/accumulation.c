// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/finite/accumulation.h"

#include <string.h>

void loom_service_finite_accumulation_deinitialize(
    loom_service_finite_accumulation_t* accumulation) {
  for (iree_host_size_t i = 0; i < accumulation->count; ++i) {
    if (accumulation->device_rows) {
      iree_hal_buffer_release(accumulation->device_rows[i].buffer);
    }
    if (accumulation->host_owners) {
      loom_service_finite_snapshot_release(accumulation->host_owners[i]);
    }
  }
  loom_service_finite_result_release(accumulation->latest);
  iree_allocator_free(accumulation->host_allocator, accumulation->storage);
  memset(accumulation, 0, sizeof(*accumulation));
}

iree_status_t loom_service_finite_accumulation_append(
    loom_service_finite_accumulation_t* accumulation,
    iree_host_size_t expected_count, loom_service_finite_output_flags_t flags,
    loom_service_finite_result_t* result,
    loom_service_finite_snapshot_t* snapshot, iree_host_size_t request,
    iree_allocator_t host_allocator) {
  const bool device =
      iree_any_bit_set(flags, LOOM_SERVICE_FINITE_OUTPUT_FLAG_DEVICE);
  const bool host =
      iree_any_bit_set(flags, LOOM_SERVICE_FINITE_OUTPUT_FLAG_HOST);
  if (flags && !accumulation->storage) {
    const iree_host_size_t stride =
        (device ? sizeof(iree_hal_buffer_ref_t) : 0) +
        (host ? sizeof(iree_const_byte_span_t) + sizeof(snapshot) : 0);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, expected_count, stride, &accumulation->storage));
    accumulation->host_allocator = host_allocator;
    uint8_t* cursor = accumulation->storage;
    if (device) {
      accumulation->device_rows = (iree_hal_buffer_ref_t*)cursor;
      cursor += expected_count * sizeof(iree_hal_buffer_ref_t);
    }
    if (host) {
      accumulation->host_rows = (iree_const_byte_span_t*)cursor;
      cursor += expected_count * sizeof(iree_const_byte_span_t);
      accumulation->host_owners = (loom_service_finite_snapshot_t**)cursor;
    }
  }
  const iree_host_size_t count =
      loom_service_finite_result_row_count(result, request);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t index = accumulation->count + i;
    if (device) {
      accumulation->device_rows[index] =
          loom_service_finite_result_row(result, request, i);
      iree_hal_buffer_retain(accumulation->device_rows[index].buffer);
    }
    if (host) {
      accumulation->host_rows[index] =
          loom_service_finite_snapshot_row(snapshot, request, i);
      accumulation->host_owners[index] = snapshot;
      loom_service_finite_snapshot_retain(snapshot);
    }
  }
  accumulation->count += count;
  if (device) {
    loom_service_finite_result_retain(result);
    loom_service_finite_result_release(accumulation->latest);
    accumulation->latest = result;
  }
  return iree_ok_status();
}

iree_status_t loom_service_finite_accumulation_result(
    const loom_service_finite_accumulation_t* accumulation,
    uint64_t input_tokens, loom_service_finite_result_t** out_result) {
  return loom_service_finite_result_create_rows(
      accumulation->latest, accumulation->count, accumulation->device_rows,
      input_tokens, accumulation->host_allocator, out_result);
}
