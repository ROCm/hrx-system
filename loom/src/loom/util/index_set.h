// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ordered membership over a fixed 32-bit index domain.

#ifndef LOOM_UTIL_INDEX_SET_H_
#define LOOM_UTIL_INDEX_SET_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Six 64-way levels cover every index below UINT32_MAX.
#define LOOM_INDEX_SET_LEVEL_CAPACITY 6u

// Selection result when the set is empty; never a member of an index domain.
#define LOOM_INDEX_SET_NONE UINT32_MAX

// Shared layout for caller-owned membership words. Leaves start at word zero;
// each higher level marks nonempty words in the preceding level. A nonempty
// domain has one root word. The empty domain has no words or levels.
//
// Callers allocate and initially zero word_count words for each independent
// set. The same layout can describe several adjacent sets. Membership storage
// and its lifetime remain with the caller; operations neither allocate nor
// retain pointers. Insert, erase and selection visit at most six levels.
typedef struct loom_index_set_layout_t {
  // Word offset of each summary level, leaves first.
  uint32_t level_starts[LOOM_INDEX_SET_LEVEL_CAPACITY];
  // Total number of membership and summary words in each set.
  uint32_t word_count;
  // Number of populated level_starts entries.
  uint8_t level_count;
} loom_index_set_layout_t;

// Calculates storage for indices in [0, capacity). The complete uint32_t
// capacity domain is representable without overflow.
loom_index_set_layout_t loom_index_set_calculate_layout(uint32_t capacity);

// Inserts an in-domain index and returns whether its membership was new.
bool loom_index_set_insert(const loom_index_set_layout_t* layout,
                           uint64_t* words, uint32_t index);

// Inserts an in-domain interval [first, first + count). Work is proportional
// to its membership words, not its member count. Empty intervals do nothing.
void loom_index_set_insert_range(const loom_index_set_layout_t* layout,
                                 uint64_t* words, uint32_t first,
                                 uint32_t count);

// Erases an in-domain index, including one that is not currently a member.
void loom_index_set_erase(const loom_index_set_layout_t* layout,
                          uint64_t* words, uint32_t index);

// Clears nonempty leaf words and their summaries. Work follows occupied words;
// clearing an empty set takes constant time regardless of its domain size.
void loom_index_set_clear(const loom_index_set_layout_t* layout,
                          uint64_t* words);

// Selects the first member at or after start, wrapping to the lowest member
// when none follows start. Returns NONE for an empty set. start is in-domain,
// or zero for the empty domain. A start of zero selects the lowest member.
uint32_t loom_index_set_select(const loom_index_set_layout_t* layout,
                               const uint64_t* words, uint32_t start);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_INDEX_SET_H_
