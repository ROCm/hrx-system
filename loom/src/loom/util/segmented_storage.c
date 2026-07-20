// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/segmented_storage.h"

#include <string.h>

static iree_status_t loom_segmented_storage_allocate_pointer_page(
    iree_arena_allocator_t* arena, void*** out_page) {
  void** page = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE,
                                sizeof(*page), (void**)&page));
  memset(page, 0, LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE * sizeof(*page));
  *out_page = page;
  return iree_ok_status();
}

static iree_status_t loom_segmented_storage_expand_primary_page(
    loom_segmented_storage_t* storage, iree_arena_allocator_t* arena) {
  if (storage->primary_page != NULL) {
    return iree_ok_status();
  }
  void** primary_page = NULL;
  IREE_RETURN_IF_ERROR(
      loom_segmented_storage_allocate_pointer_page(arena, &primary_page));
  memcpy(primary_page, storage->inline_segments,
         sizeof(storage->inline_segments));
  storage->primary_page = primary_page;
  return iree_ok_status();
}

static iree_status_t loom_segmented_storage_ensure_page(
    loom_segmented_storage_t* storage, iree_arena_allocator_t* arena,
    uint32_t page_index, void*** out_page) {
  if (page_index == 0) {
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_expand_primary_page(storage, arena));
    *out_page = storage->primary_page;
    return iree_ok_status();
  }

  if (!storage->page_directory) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE,
        sizeof(*storage->page_directory), (void**)&storage->page_directory));
    memset(storage->page_directory, 0,
           LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE *
               sizeof(*storage->page_directory));
    storage->page_directory[0] = storage->primary_page;
  }
  if (!storage->page_directory[page_index]) {
    IREE_RETURN_IF_ERROR(loom_segmented_storage_allocate_pointer_page(
        arena, &storage->page_directory[page_index]));
  }
  *out_page = storage->page_directory[page_index];
  return iree_ok_status();
}

void loom_segmented_storage_initialize(iree_host_size_t segment_size,
                                       iree_host_size_t segment_alignment,
                                       loom_segmented_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(out_storage);
  IREE_ASSERT(segment_size > 0);
  IREE_ASSERT(iree_host_size_is_power_of_two(segment_alignment));
  memset(out_storage, 0, sizeof(*out_storage));
  out_storage->segment_size = segment_size;
  out_storage->segment_alignment = segment_alignment;
}

void loom_segmented_storage_move(loom_segmented_storage_t* source,
                                 loom_segmented_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_storage = *source;
  *source = (loom_segmented_storage_t){0};
}

iree_status_t loom_segmented_storage_append(loom_segmented_storage_t* storage,
                                            iree_arena_allocator_t* arena,
                                            void** out_segment) {
  IREE_ASSERT_ARGUMENT(storage);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_segment);
  *out_segment = NULL;

  if (IREE_UNLIKELY(storage->segment_count >=
                    LOOM_SEGMENTED_STORAGE_MAX_SEGMENT_COUNT)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "segmented storage exceeds %u segments",
                            LOOM_SEGMENTED_STORAGE_MAX_SEGMENT_COUNT);
  }

  const uint32_t segment_index = storage->segment_count;
  const uint32_t page_index =
      segment_index >> LOOM_SEGMENTED_STORAGE_PAGE_SHIFT;
  void** page = storage->primary_page != NULL ? storage->primary_page
                                              : storage->inline_segments;
  if (segment_index >= LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT) {
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_ensure_page(storage, arena, page_index, &page));
  }

  void* segment = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_aligned(
      arena, storage->segment_size, storage->segment_alignment, &segment));
  page[segment_index & LOOM_SEGMENTED_STORAGE_PAGE_MASK] = segment;
  ++storage->segment_count;
  *out_segment = segment;
  return iree_ok_status();
}
