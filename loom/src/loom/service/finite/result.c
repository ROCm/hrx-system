// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/finite/result.h"

struct loom_service_finite_result_t {
  // Shared ownership across original requests and asynchronous consumers.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this metadata object.
  iree_allocator_t host_allocator;
  // Source-declared output extent, including required padding.
  iree_host_size_t output_length;
  // Live result prefix containing every checked row.
  iree_host_size_t length;
  // Original request count, independent of later cancellation or scheduling.
  iree_host_size_t request_count;
  // Checked request and row metadata extent in the trailing allocation.
  iree_host_size_t records_length;
  // Retained producer timeline and shared workspace, not its source model.
  loom_runtime_execution_t* execution;
  // Exact output view, including storage ownership but not producer tracking.
  iree_hal_buffer_t* buffer;
  // Owned row references for a logical result spanning several cohort buffers.
  // NULL denotes the ordinary single-output source result.
  iree_hal_buffer_ref_t* gathered_rows;
  // Number of owned references in gathered_rows.
  iree_host_size_t gathered_row_count;
  // Accepted producer frontier, whose semaphores are owned by execution.
  loom_runtime_execution_frontier_t ready;
  // Checked request and row metadata, independent of source buffer aliases.
  uint8_t records[];
};

struct loom_service_finite_snapshot_t {
  // Shared by the caller and the download's actual queue capture.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this header and trailing host storage together.
  iree_allocator_t host_allocator;
  // Retained timeline and source-pool ownership, independent of the model.
  loom_runtime_execution_t* execution;
  // Original request count used by checked row metadata.
  iree_host_size_t request_count;
  // Copied metadata extent preceding the exact downloaded result prefix.
  iree_host_size_t records_length;
  // Independent checked records followed by the downloaded host result bytes.
  uint8_t storage[];
};

iree_status_t loom_service_finite_result_create(
    iree_vm_buffer_t* plan, iree_host_size_t output_capacity,
    iree_host_size_t result_capacity, iree_host_size_t request_count,
    const iree_host_size_t* expected_rows, iree_allocator_t host_allocator,
    loom_service_finite_result_t** out_result) {
  *out_result = NULL;
  iree_const_byte_span_t bytes;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_map_read(plan, 0, iree_vm_buffer_length(plan), &bytes));
  if (bytes.data_length < 32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source result plan requires a complete header");
  }
  const uint64_t output_length = iree_unaligned_load_le_u64(bytes.data);
  const uint64_t length = iree_unaligned_load_le_u64(bytes.data + 8);
  const uint64_t count = iree_unaligned_load_le_u64(bytes.data + 16);
  const uint64_t rows = iree_unaligned_load_le_u64(bytes.data + 24);
  if (!output_length || output_length > output_capacity ||
      length > result_capacity || length > output_length ||
      count != request_count || count > (bytes.data_length - 32) / 24 ||
      rows != (bytes.data_length - 32 - count * 24) / 16 ||
      bytes.data_length != 32 + count * 24 + rows * 16) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source result plan has invalid extents");
  }
  uint64_t first_row = 0;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    const uint8_t* record = bytes.data + 32 + i * 24;
    const uint64_t row_count = iree_unaligned_load_le_u64(record + 8);
    if (iree_unaligned_load_le_u64(record) != first_row ||
        row_count > expected_rows[i] || row_count > rows - first_row) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source result rows differ from request %zu", i);
    }
    first_row += row_count;
  }
  if (first_row != rows) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source result plan contains unclaimed rows");
  }
  for (uint64_t i = 0; i < rows; ++i) {
    const uint8_t* record = bytes.data + 32 + count * 24 + i * 16;
    const uint64_t offset = iree_unaligned_load_le_u64(record);
    const uint64_t width = iree_unaligned_load_le_u64(record + 8);
    if (offset % 4 || offset > length || !width ||
        width > (length - offset) / 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source vector lies outside result storage");
    }
  }
  loom_service_finite_result_t* result = NULL;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(sizeof(*result), bytes.data_length - 32,
                                  &allocation_size) ||
      !iree_host_size_checked_add(allocation_size,
                                  request_count * sizeof(uint64_t),
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "result metadata exceeds host address space");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, allocation_size, (void**)&result));
  memset(result, 0, sizeof(*result));
  iree_atomic_ref_count_init(&result->ref_count);
  result->host_allocator = host_allocator;
  memcpy(result->records, bytes.data + 32, bytes.data_length - 32);
  result->output_length = (iree_host_size_t)output_length;
  result->length = (iree_host_size_t)length;
  result->request_count = request_count;
  result->records_length = bytes.data_length - 32;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    iree_unaligned_store_le_u64(
        result->records + result->records_length + i * sizeof(uint64_t),
        expected_rows[i]);
  }
  *out_result = result;
  return iree_ok_status();
}

iree_status_t loom_service_finite_result_create_rows(
    const loom_service_finite_result_t* frontier, iree_host_size_t row_count,
    const iree_hal_buffer_ref_t* rows, uint64_t input_tokens,
    iree_allocator_t host_allocator,
    loom_service_finite_result_t** out_result) {
  *out_result = NULL;
  iree_host_size_t row_storage = 0;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(row_count, 16 + sizeof(iree_hal_buffer_ref_t),
                                  &row_storage) ||
      !iree_host_size_checked_add(sizeof(loom_service_finite_result_t) + 32,
                                  row_storage, &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "logical result metadata exceeds address space");
  }
  iree_host_size_t length = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
       ++i) {
    if (!iree_host_size_checked_add(length, rows[i].length, &length)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "logical result bytes exceed address space");
    }
  }
  IREE_RETURN_IF_ERROR(status);
  loom_service_finite_result_t* result = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, allocation_size, (void**)&result));
  memset(result, 0, sizeof(*result));
  iree_atomic_ref_count_init(&result->ref_count);
  result->host_allocator = host_allocator;
  result->request_count = 1;
  result->records_length = 24 + row_count * 16;
  result->output_length = length;
  result->length = length;
  result->execution = frontier->execution;
  loom_runtime_execution_retain(result->execution);
  result->ready = frontier->ready;
  result->gathered_row_count = row_count;
  result->gathered_rows =
      (iree_hal_buffer_ref_t*)(result->records + result->records_length +
                               sizeof(uint64_t));
  iree_unaligned_store_le_u64(result->records, 0);
  iree_unaligned_store_le_u64(result->records + 8, row_count);
  iree_unaligned_store_le_u64(result->records + 16, input_tokens);
  iree_unaligned_store_le_u64(result->records + result->records_length,
                              row_count);
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    result->gathered_rows[i] = rows[i];
    iree_hal_buffer_retain(rows[i].buffer);
    iree_unaligned_store_le_u64(result->records + 24 + i * 16, offset);
    iree_unaligned_store_le_u64(result->records + 24 + i * 16 + 8,
                                rows[i].length / sizeof(float));
    offset += rows[i].length;
  }
  *out_result = result;
  return iree_ok_status();
}

void loom_service_finite_result_bind(loom_service_finite_result_t* result,
                                     loom_runtime_execution_t* execution,
                                     iree_hal_buffer_t* output) {
  result->execution = execution;
  loom_runtime_execution_retain(execution);
  result->ready = loom_runtime_execution_frontier(execution);
  result->buffer = output;
  iree_hal_buffer_retain(result->buffer);
}

void loom_service_finite_result_retain(loom_service_finite_result_t* result) {
  if (result) {
    iree_atomic_ref_count_inc(&result->ref_count);
  }
}

void loom_service_finite_result_release(loom_service_finite_result_t* result) {
  if (result && iree_atomic_ref_count_dec(&result->ref_count) == 1) {
    for (iree_host_size_t i = 0; i < result->gathered_row_count; ++i) {
      iree_hal_buffer_release(result->gathered_rows[i].buffer);
    }
    iree_hal_buffer_release(result->buffer);
    loom_runtime_execution_release(result->execution);
    iree_allocator_free(result->host_allocator, result);
  }
}

iree_host_size_t loom_service_finite_result_output_length(
    const loom_service_finite_result_t* result) {
  return result->output_length;
}

iree_host_size_t loom_service_finite_result_length(
    const loom_service_finite_result_t* result) {
  return result->length;
}

loom_runtime_execution_frontier_t loom_service_finite_result_ready(
    const loom_service_finite_result_t* result) {
  return result->ready;
}

iree_host_size_t loom_service_finite_result_row_count(
    const loom_service_finite_result_t* result, iree_host_size_t request) {
  return (iree_host_size_t)iree_unaligned_load_le_u64(result->records +
                                                      request * 24 + 8);
}

bool loom_service_finite_result_completes_request(
    const loom_service_finite_result_t* result, iree_host_size_t request) {
  const uint64_t remaining = iree_unaligned_load_le_u64(
      result->records + result->records_length + request * sizeof(uint64_t));
  return loom_service_finite_result_row_count(result, request) == remaining;
}

uint64_t loom_service_finite_result_input_tokens(
    const loom_service_finite_result_t* result, iree_host_size_t request) {
  return iree_unaligned_load_le_u64(result->records + request * 24 + 16);
}

iree_hal_buffer_ref_t loom_service_finite_result_row(
    const loom_service_finite_result_t* result, iree_host_size_t request,
    iree_host_size_t row) {
  const uint8_t* records = result->records;
  const uint64_t first = iree_unaligned_load_le_u64(records + request * 24);
  if (result->gathered_rows) {
    return result->gathered_rows[first + row];
  }
  const uint8_t* vector =
      records + result->request_count * 24 + (first + row) * 16;
  return (iree_hal_buffer_ref_t){
      .buffer = result->buffer,
      .offset = iree_unaligned_load_le_u64(vector),
      .length = iree_unaligned_load_le_u64(vector + 8) * 4,
  };
}

void loom_service_finite_snapshot_retain(
    loom_service_finite_snapshot_t* snapshot) {
  if (snapshot) {
    iree_atomic_ref_count_inc(&snapshot->ref_count);
  }
}

void loom_service_finite_snapshot_release(
    loom_service_finite_snapshot_t* snapshot) {
  if (snapshot && iree_atomic_ref_count_dec(&snapshot->ref_count) == 1) {
    loom_runtime_execution_release(snapshot->execution);
    iree_allocator_free(snapshot->host_allocator, snapshot);
  }
}

static void loom_service_finite_snapshot_retired(void* user_data,
                                                 iree_hal_buffer_t* buffer) {
  loom_service_finite_snapshot_release(user_data);
}

iree_status_t loom_service_finite_result_download(
    const loom_service_finite_result_t* result,
    loom_service_finite_snapshot_t** out_snapshot,
    loom_runtime_execution_frontier_t* out_frontier) {
  *out_snapshot = NULL;
  *out_frontier = (loom_runtime_execution_frontier_t){0};
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(sizeof(loom_service_finite_snapshot_t),
                                  result->records_length, &allocation_size) ||
      !iree_host_size_checked_add(allocation_size, result->length,
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "result snapshot exceeds host address space");
  }
  loom_service_finite_snapshot_t* snapshot = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      result->host_allocator, allocation_size, (void**)&snapshot));
  iree_atomic_ref_count_init(&snapshot->ref_count);
  snapshot->host_allocator = result->host_allocator;
  snapshot->execution = result->execution;
  loom_runtime_execution_retain(snapshot->execution);
  snapshot->request_count = result->request_count;
  snapshot->records_length = result->records_length;
  memcpy(snapshot->storage, result->records, result->records_length);
  const iree_host_size_t count =
      result->gathered_rows ? result->gathered_row_count : 1;
  iree_hal_transfer_operation_t inline_operation;
  iree_hal_transfer_operation_t* operations = &inline_operation;
  iree_hal_transfer_operation_t* allocated_operations = NULL;
  iree_status_t status = iree_ok_status();
  if (count > 1) {
    status = iree_allocator_malloc_array(result->host_allocator, count,
                                         sizeof(*operations),
                                         (void**)&allocated_operations);
    operations = allocated_operations;
  }
  iree_host_size_t initialized = 0;
  iree_host_size_t target_offset = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_buffer_ref_t row =
        result->gathered_rows
            ? result->gathered_rows[i]
            : (iree_hal_buffer_ref_t){.buffer = result->buffer,
                                      .length = result->length};
    iree_hal_buffer_t* source = NULL;
    status = iree_hal_subspan_buffer_create_with_callback(
        row.buffer, iree_hal_buffer_byte_offset(row.buffer) + row.offset,
        row.length,
        (iree_hal_buffer_release_callback_t){
            loom_service_finite_snapshot_retired, snapshot},
        result->host_allocator, &source);
    if (iree_status_is_ok(status)) {
      loom_service_finite_snapshot_retain(snapshot);
      operations[initialized++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {.source_buffer = source,
                       .target = snapshot->storage + snapshot->records_length +
                                 target_offset,
                       .length = row.length}};
      target_offset += row.length;
    }
  }
  if (iree_status_is_ok(status)) {
    uint64_t value = 0;
    status = loom_runtime_execution_feedback(result->execution, count,
                                             operations, &value);
  }
  for (iree_host_size_t i = 0; i < initialized; ++i) {
    iree_hal_buffer_release(operations[i].download.source_buffer);
  }
  iree_allocator_free(result->host_allocator, allocated_operations);
  if (iree_status_is_ok(status)) {
    *out_snapshot = snapshot;
    *out_frontier = loom_runtime_execution_frontier(result->execution);
  } else {
    loom_service_finite_snapshot_release(snapshot);
  }
  return status;
}

uint64_t loom_service_finite_snapshot_input_tokens(
    const loom_service_finite_snapshot_t* snapshot, iree_host_size_t request) {
  return iree_unaligned_load_le_u64(snapshot->storage + request * 24 + 16);
}

iree_const_byte_span_t loom_service_finite_snapshot_row(
    const loom_service_finite_snapshot_t* snapshot, iree_host_size_t request,
    iree_host_size_t row) {
  const uint64_t first =
      iree_unaligned_load_le_u64(snapshot->storage + request * 24);
  const uint8_t* vector =
      snapshot->storage + snapshot->request_count * 24 + (first + row) * 16;
  return iree_make_const_byte_span(
      snapshot->storage + snapshot->records_length +
          iree_unaligned_load_le_u64(vector),
      (iree_host_size_t)iree_unaligned_load_le_u64(vector + 8) * 4);
}
