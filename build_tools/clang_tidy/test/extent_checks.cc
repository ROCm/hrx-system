// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

using iree_host_size_t = unsigned long;

typedef struct {
  const char* data;
  iree_host_size_t size;
} iree_string_view_t;
static inline iree_string_view_t iree_string_view_empty(void) {
  iree_string_view_t view = {nullptr, 0};
  return view;
}

typedef struct {
  const unsigned char* data;
  iree_host_size_t data_length;
} iree_const_byte_span_t;
static inline iree_const_byte_span_t iree_const_byte_span_empty(void) {
  iree_const_byte_span_t span = {nullptr, 0};
  return span;
}

typedef struct {
  void** operations;
  iree_host_size_t count;
} iree_async_operation_list_t;
static inline iree_async_operation_list_t iree_async_operation_list_empty(
    void) {
  iree_async_operation_list_t list = {nullptr, 0};
  return list;
}

void iree_clang_tidy_extent_cpp_aggregate_zero_initializers(void) {
  iree_string_view_t string_view = {};
  iree_const_byte_span_t const_byte_span = {};
  (void)string_view;
  (void)const_byte_span;
}

void iree_clang_tidy_extent_cpp_lambda_initializers(void) {
  void* operation = nullptr;
  [&]() {
    void* operations[] = {operation};
    iree_async_operation_list_t list = {operations, 1};
    (void)list;
  }();
}
