// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable arena-backed storage for fixed-size segments.
//
// Segments are appended and never moved. Typed owners choose the payload
// layout within each segment and map their logical indexes to segment indexes.
// The directory keeps a small number of segment pointers inline, expands once
// to a fixed primary page, and allocates further fixed pages through a fixed
// top-level directory. Payload storage is never copied during directory growth.

#ifndef LOOM_UTIL_SEGMENTED_STORAGE_H_
#define LOOM_UTIL_SEGMENTED_STORAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of segment pointers stored directly in the container.
#define LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT 16u

// Number of segment pointers stored in each allocated pointer page.
#define LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE 4096u

// Shift mapping a segment index to its pointer-page index.
#define LOOM_SEGMENTED_STORAGE_PAGE_SHIFT 12u

// Mask mapping a segment index to its position within a pointer page.
#define LOOM_SEGMENTED_STORAGE_PAGE_MASK \
  (LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE - 1u)

// Maximum number of segments represented by the two-level directory.
#define LOOM_SEGMENTED_STORAGE_MAX_SEGMENT_COUNT \
  (LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE *    \
   LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE)

static_assert((1u << LOOM_SEGMENTED_STORAGE_PAGE_SHIFT) ==
                  LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE,
              "segment pointer page size must match its index shift");

// Arena-backed stable segment directory.
//
// The storage object does not own or reset its arena. Every pointer becomes
// invalid when the owner resets that arena, after which the owner must
// reinitialize the storage object before reuse. Initialized storage may be
// copied for read-only use; appends must continue through one authoritative
// instance.
typedef struct loom_segmented_storage_t {
  // Number of initialized segment pointers.
  uint32_t segment_count;
  // Size in bytes of every segment payload.
  iree_host_size_t segment_size;
  // Minimum power-of-two alignment of every segment payload.
  iree_host_size_t segment_alignment;
  // Allocated segment pointer page covering the first page of segment indexes,
  // or NULL while the inline pointer directory is active.
  void** primary_page;
  // Lazily allocated pointer-page directory, or NULL while one page suffices.
  void*** page_directory;
  // Inline pointer page used before the first allocated page is required.
  void* inline_segments[LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT];
} loom_segmented_storage_t;

// Initializes empty storage for fixed-size, fixed-alignment segments.
// No allocation is performed until the first segment is appended.
void loom_segmented_storage_initialize(iree_host_size_t segment_size,
                                       iree_host_size_t segment_alignment,
                                       loom_segmented_storage_t* out_storage);

// Moves initialized |source| storage into uninitialized |out_storage| without
// moving segment payloads. |source| is left uninitialized and must be
// initialized again before reuse.
void loom_segmented_storage_move(loom_segmented_storage_t* source,
                                 loom_segmented_storage_t* out_storage);

// Appends one uninitialized segment allocated from |arena| and returns it.
// Existing segment payload pointers remain stable.
iree_status_t loom_segmented_storage_append(loom_segmented_storage_t* storage,
                                            iree_arena_allocator_t* arena,
                                            void** out_segment);

// Returns a mutable segment payload by its zero-based index.
static inline void* loom_segmented_storage_segment(
    loom_segmented_storage_t* storage, uint32_t segment_index) {
  IREE_ASSERT(segment_index < storage->segment_count);
  if (IREE_LIKELY(storage->primary_page == NULL)) {
    return storage->inline_segments[segment_index];
  }
  const uint32_t page_index =
      segment_index >> LOOM_SEGMENTED_STORAGE_PAGE_SHIFT;
  void** page = page_index == 0 ? storage->primary_page
                                : storage->page_directory[page_index];
  return page[segment_index & LOOM_SEGMENTED_STORAGE_PAGE_MASK];
}

// Returns a const segment payload by its zero-based index.
static inline const void* loom_segmented_storage_const_segment(
    const loom_segmented_storage_t* storage, uint32_t segment_index) {
  IREE_ASSERT(segment_index < storage->segment_count);
  if (IREE_LIKELY(storage->primary_page == NULL)) {
    return storage->inline_segments[segment_index];
  }
  const uint32_t page_index =
      segment_index >> LOOM_SEGMENTED_STORAGE_PAGE_SHIFT;
  void* const* page = page_index == 0 ? storage->primary_page
                                      : storage->page_directory[page_index];
  return page[segment_index & LOOM_SEGMENTED_STORAGE_PAGE_MASK];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_SEGMENTED_STORAGE_H_
