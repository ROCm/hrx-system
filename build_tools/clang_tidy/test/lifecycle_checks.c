// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typedef int iree_status_t;

typedef struct iree_clang_tidy_in_place_resource_t {
  int value;
} iree_clang_tidy_in_place_resource_t;

typedef struct iree_clang_tidy_heap_resource_t {
  int value;
} iree_clang_tidy_heap_resource_t;

iree_status_t iree_ok_status(void);

iree_status_t iree_clang_tidy_in_place_resource_allocate(
    int value, iree_clang_tidy_in_place_resource_t* out_resource) {
  out_resource->value = value;
  return iree_ok_status();
}

void iree_clang_tidy_in_place_resource_deinitialize(
    iree_clang_tidy_in_place_resource_t* resource) {
  resource->value = 0;
}

iree_status_t iree_clang_tidy_in_place_resource_initialize(
    int value, iree_clang_tidy_in_place_resource_t* out_resource) {
  out_resource->value = value;
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_heap_resource_allocate(
    iree_clang_tidy_heap_resource_t** out_resource) {
  (void)out_resource;
  return iree_ok_status();
}

void iree_clang_tidy_heap_resource_free(
    iree_clang_tidy_heap_resource_t* resource) {
  (void)resource;
}

iree_status_t iree_clang_tidy_arena_bitset_allocate(
    int word_count, iree_clang_tidy_in_place_resource_t* out_bitset) {
  out_bitset->value = word_count;
  return iree_ok_status();
}

void iree_clang_tidy_other_resource_deinitialize(
    iree_clang_tidy_heap_resource_t* resource) {
  (void)resource;
}

iree_status_t iree_clang_tidy_view_initialize(
    iree_clang_tidy_heap_resource_t** out_resource) {
  (void)out_resource;
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_view_with_storage_initialize(
    void* storage, iree_clang_tidy_heap_resource_t** out_resource) {
  (void)storage;
  (void)out_resource;
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_view_with_named_storage_initialize(
    iree_clang_tidy_in_place_resource_t* out_storage,
    iree_clang_tidy_heap_resource_t** out_resource) {
  (void)out_storage;
  (void)out_resource;
  return iree_ok_status();
}
