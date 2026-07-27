// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/row_list.h"

#include <string.h>

enum {
  // Default allocation block size for compile report detail rows.
  LOOM_TARGET_COMPILE_REPORT_ROW_BLOCK_BYTE_LENGTH = 4096,
};

void loom_target_compile_report_row_list_deinitialize(
    iree_allocator_t allocator, loom_target_compile_report_row_list_t* list) {
  loom_target_compile_report_vec_t* vec = list->head;
  while (vec != NULL) {
    loom_target_compile_report_vec_t* next = vec->next;
    iree_allocator_free(allocator, vec);
    vec = next;
  }
  *list = (loom_target_compile_report_row_list_t){0};
}

iree_status_t loom_target_compile_report_row_list_append(
    loom_target_compile_report_row_list_t* list, iree_host_size_t row_size,
    iree_allocator_t allocator, const void* row) {
  IREE_ASSERT(row_size != 0);
  if (iree_allocator_is_null(allocator)) {
    return iree_ok_status();
  }
  if (list->tail == NULL || list->tail->count == list->tail->capacity) {
    iree_host_size_t capacity =
        (LOOM_TARGET_COMPILE_REPORT_ROW_BLOCK_BYTE_LENGTH -
         sizeof(loom_target_compile_report_vec_t)) /
        row_size;
    capacity = iree_max((iree_host_size_t)1, capacity);
    iree_host_size_t row_bytes = 0;
    if (!iree_host_size_checked_mul(capacity, row_size, &row_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "compile report row block is too large");
    }
    iree_host_size_t block_bytes = 0;
    if (!iree_host_size_checked_add(sizeof(loom_target_compile_report_vec_t),
                                    row_bytes, &block_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "compile report row block is too large");
    }
    loom_target_compile_report_vec_t* vec = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, block_bytes, (void**)&vec));
    *vec = (loom_target_compile_report_vec_t){
        .capacity = capacity,
    };
    if (list->tail != NULL) {
      list->tail->next = vec;
    } else {
      list->head = vec;
    }
    list->tail = vec;
  }
  uint8_t* rows = (uint8_t*)loom_target_compile_report_vec_rows(list->tail);
  memcpy(rows + list->tail->count * row_size, row, row_size);
  ++list->tail->count;
  ++list->count;
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_row_list_append_all(
    loom_target_compile_report_row_list_t* target,
    const loom_target_compile_report_row_list_t* source,
    iree_host_size_t row_size, iree_allocator_t allocator) {
  for (const loom_target_compile_report_vec_t* vec = source->head; vec != NULL;
       vec = vec->next) {
    const uint8_t* rows =
        (const uint8_t*)loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
          target, row_size, allocator, rows + i * row_size));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_row_list_clone(
    const loom_target_compile_report_row_list_t* source,
    iree_host_size_t row_size, iree_allocator_t allocator,
    loom_target_compile_report_row_list_t* target) {
  *target = (loom_target_compile_report_row_list_t){0};
  return loom_target_compile_report_row_list_append_all(target, source,
                                                        row_size, allocator);
}
