// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_loop_reads.h"

#include <string.h>

iree_status_t loom_value_fact_loop_reads_begin(
    loom_value_fact_loop_reads_t* reads,
    const loom_value_fact_loop_subtree_t* subtree,
    iree_arena_allocator_t* arena) {
  if (!reads->bits) {
    const iree_host_size_t word_count =
        (reads->owners->root_range.capacity + 63) / 64;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, word_count, sizeof(*reads->words), (void**)&reads->words));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, word_count, sizeof(*reads->bits), (void**)&reads->bits));
    memset(reads->bits, 0, word_count * sizeof(*reads->bits));
  }
  reads->subtree = subtree;
  return iree_ok_status();
}

void loom_value_fact_loop_record_read(loom_value_fact_loop_reads_t* reads,
                                      loom_value_id_t value_id) {
  if (!loom_value_fact_range_contains(reads->owners->root_range, value_id)) {
    return;
  }
  const iree_host_size_t storage_offset =
      loom_value_fact_range_offset(reads->owners->storage_range, value_id);
  const loom_value_fact_loop_owner_t entry =
      reads->owners->entries[storage_offset];
  if (loom_value_fact_loop_owner_generation(entry) !=
      reads->owners->generation) {
    return;
  }
  const uint32_t owner = loom_value_fact_loop_owner_ordinal(entry);
  if (owner >= reads->subtree->ordinal && owner < reads->subtree->end) {
    return;
  }
  const iree_host_size_t offset =
      loom_value_fact_range_offset(reads->owners->root_range, value_id);
  const uint32_t word = offset / 64;
  const uint64_t bit = UINT64_C(1) << (offset % 64);
  if (!reads->bits[word]) {
    reads->words[reads->word_count++] = word;
  }
  reads->bits[word] |= bit;
}

void loom_value_fact_loop_reads_reset(loom_value_fact_loop_reads_t* reads) {
  for (iree_host_size_t i = 0; i < reads->word_count; ++i) {
    reads->bits[reads->words[i]] = 0;
  }
  reads->word_count = 0;
  reads->subtree = NULL;
}
