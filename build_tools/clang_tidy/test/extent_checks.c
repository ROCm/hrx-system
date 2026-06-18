// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NULL
#define NULL ((void*)0)
#endif  // NULL

typedef unsigned long iree_host_size_t;

typedef struct {
  unsigned char* data;
  iree_host_size_t data_length;
} iree_byte_span_t;
static inline iree_byte_span_t iree_byte_span_empty(void) {
  iree_byte_span_t span = {NULL, 0};
  return span;
}

typedef struct {
  const unsigned char* data;
  iree_host_size_t data_length;
} iree_const_byte_span_t;
static inline iree_const_byte_span_t iree_const_byte_span_empty(void) {
  iree_const_byte_span_t span = {NULL, 0};
  return span;
}

typedef struct {
  const char* data;
  iree_host_size_t size;
} iree_string_view_t;
static inline iree_string_view_t iree_string_view_empty(void) {
  iree_string_view_t view = {NULL, 0};
  return view;
}

typedef struct {
  char* data;
  iree_host_size_t size;
} iree_mutable_string_view_t;
static inline iree_mutable_string_view_t iree_mutable_string_view_empty(void) {
  iree_mutable_string_view_t view = {NULL, 0};
  return view;
}

typedef struct {
  iree_host_size_t count;
  void** semaphores;
  unsigned long* payload_values;
} iree_hal_semaphore_list_t;
static inline iree_hal_semaphore_list_t iree_hal_semaphore_list_empty(void) {
  iree_hal_semaphore_list_t list = {0};
  return list;
}

int iree_string_view_is_empty(iree_string_view_t view);
int iree_const_byte_span_is_empty(iree_const_byte_span_t span);
int iree_clang_tidy_extent_other_predicate(void);

void iree_clang_tidy_extent_aggregate_zero_initializers(void) {
  iree_byte_span_t byte_span = {0};
  iree_const_byte_span_t const_byte_span = {NULL, 0};
  iree_string_view_t string_view = {0};
  iree_mutable_string_view_t mutable_string_view = {NULL, 0};
  iree_hal_semaphore_list_t semaphore_list = {0};
  (void)byte_span;
  (void)const_byte_span;
  (void)string_view;
  (void)mutable_string_view;
  (void)semaphore_list;
}

int iree_clang_tidy_extent_string_view_is_empty(iree_string_view_t view) {
  if (view.data == NULL || view.size == 0) return 1;
  return 0;
}

void iree_clang_tidy_extent_pointer_empty_predicates(
    iree_string_view_t view, iree_const_byte_span_t span) {
  if (!view.data && !view.size) return;
  if (!span.data && !span.data_length) return;
}

void iree_clang_tidy_extent_allowed_predicates(iree_string_view_t view,
                                               iree_const_byte_span_t span) {
  if (!view.data || !view.size) return;
  if (span.data == NULL || span.data_length == 0) return;
  if (view.size > 0 && view.data == NULL) return;
  if (span.data_length > 0 && !span.data) return;
  if (iree_string_view_is_empty(view)) return;
  if (iree_const_byte_span_is_empty(span)) return;
  if (iree_clang_tidy_extent_other_predicate()) return;
}

void iree_clang_tidy_extent_allowed_initializers(void) {
  iree_byte_span_t byte_span = iree_byte_span_empty();
  iree_const_byte_span_t const_byte_span = iree_const_byte_span_empty();
  iree_string_view_t string_view = iree_string_view_empty();
  iree_mutable_string_view_t mutable_string_view =
      iree_mutable_string_view_empty();
  iree_hal_semaphore_list_t semaphore_list = iree_hal_semaphore_list_empty();
  (void)byte_span;
  (void)const_byte_span;
  (void)string_view;
  (void)mutable_string_view;
  (void)semaphore_list;
}
