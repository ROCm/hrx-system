// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/participants.h"

#include "iree/base/internal/math.h"
#include "loom/util/adaptive_sort.h"

struct loom_pipeline_participants_t {
  // Lowest 64-engine neighborhood represented by this subtree.
  iree_host_size_t prefix;
  // Number of present engines in this subtree.
  iree_host_size_t count;
  // Highest differing engine bit, or zero at a packed leaf.
  iree_host_size_t branch;
  union {
    // Children partitioned by branch, with zero-bit engines first.
    const loom_pipeline_participants_t* children[2];
    // Packed associations when branch is zero.
    struct {
      // Present engines within this leaf's 64-engine neighborhood.
      uint64_t mask;
      // Packed representative workers, ordered by engine, owned by the arena.
      const iree_host_size_t* workers;
    } leaf;
  };
};

iree_host_size_t loom_pipeline_participants_count(
    const loom_pipeline_participants_t* participants) {
  return participants ? participants->count : 0;
}

loom_pipeline_participant_location_t loom_pipeline_participants_find(
    const loom_pipeline_participants_t* participants, iree_host_size_t engine) {
  iree_host_size_t index = 0;
  if (participants) {
    while (participants->branch) {
      const bool right = (engine & participants->branch) != 0;
      if (right) {
        index += participants->children[0]->count;
      }
      participants = participants->children[right];
    }
    const uint64_t bit = UINT64_C(1) << (engine % 64);
    if (participants->prefix == (engine & ~(iree_host_size_t)63) &&
        (participants->leaf.mask & bit)) {
      const iree_host_size_t local =
          iree_math_count_ones_u64(participants->leaf.mask & (bit - 1));
      return (loom_pipeline_participant_location_t){
          .index = index + local, .worker = participants->leaf.workers[local]};
    }
  }
  return (loom_pipeline_participant_location_t){IREE_HOST_SIZE_MAX,
                                                IREE_HOST_SIZE_MAX};
}

static iree_status_t loom_pipeline_participants_leaf(
    iree_arena_allocator_t* arena, iree_host_size_t prefix, uint64_t mask,
    loom_pipeline_participants_t** out_leaf, iree_host_size_t** out_workers) {
  const iree_host_size_t count = iree_math_count_ones_u64(mask);
  loom_pipeline_participants_t* leaf = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, sizeof(*leaf) + count * sizeof(**out_workers), (void**)&leaf));
  *out_workers = (iree_host_size_t*)(leaf + 1);
  *leaf = (loom_pipeline_participants_t){
      .prefix = prefix,
      .count = count,
      .leaf = {.mask = mask, .workers = *out_workers}};
  *out_leaf = leaf;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_participants_branch(
    iree_arena_allocator_t* arena, iree_host_size_t bit,
    const loom_pipeline_participants_t* left,
    const loom_pipeline_participants_t* right,
    const loom_pipeline_participants_t** out_participants) {
  loom_pipeline_participants_t* branch = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*branch), (void**)&branch));
  *branch = (loom_pipeline_participants_t){.prefix = left->prefix,
                                           .count = left->count + right->count,
                                           .branch = bit,
                                           .children = {left, right}};
  *out_participants = branch;
  return iree_ok_status();
}

static iree_host_size_t loom_pipeline_participants_high_bit(
    iree_host_size_t value) {
  return value
             ? (iree_host_size_t)(UINT64_C(1)
                                  << (63 -
                                      iree_math_count_leading_zeros_u64(value)))
             : 0;
}

static bool loom_pipeline_participant_less(
    const loom_pipeline_participant_t* a,
    const loom_pipeline_participant_t* b) {
  return a->engine != b->engine ? a->engine < b->engine : a->worker < b->worker;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_participants_sort,
                          loom_pipeline_participant_t,
                          loom_pipeline_participant_less)

// Recursion is bounded by ordinal bit width, independently of pipeline depth.
static iree_status_t loom_pipeline_participants_build_sorted(
    iree_arena_allocator_t* arena, const loom_pipeline_participant_t* entries,
    iree_host_size_t count,
    const loom_pipeline_participants_t** out_participants) {
  const iree_host_size_t bit = loom_pipeline_participants_high_bit(
      (entries[0].engine ^ entries[count - 1].engine) & ~(iree_host_size_t)63);
  if (!bit) {
    uint64_t mask = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      mask |= UINT64_C(1) << (entries[i].engine % 64);
    }
    loom_pipeline_participants_t* leaf = NULL;
    iree_host_size_t* workers = NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_participants_leaf(
        arena, entries[0].engine & ~(iree_host_size_t)63, mask, &leaf,
        &workers));
    for (iree_host_size_t i = 0; i < count; ++i) {
      workers[i] = entries[i].worker;
    }
    *out_participants = leaf;
    return iree_ok_status();
  }
  iree_host_size_t begin = 1, end = count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (entries[middle].engine & bit) {
      end = middle;
    } else {
      begin = middle + 1;
    }
  }
  const loom_pipeline_participants_t* children[2];
  IREE_RETURN_IF_ERROR(loom_pipeline_participants_build_sorted(
      arena, entries, begin, &children[0]));
  IREE_RETURN_IF_ERROR(loom_pipeline_participants_build_sorted(
      arena, entries + begin, count - begin, &children[1]));
  return loom_pipeline_participants_branch(arena, bit, children[0], children[1],
                                           out_participants);
}

iree_status_t loom_pipeline_participants_build(
    iree_arena_allocator_t* arena, loom_pipeline_participant_t* entries,
    iree_host_size_t count,
    const loom_pipeline_participants_t** out_participants,
    loom_pipeline_execution_conflict_t* out_conflict) {
  *out_participants = NULL;
  *out_conflict = (loom_pipeline_execution_conflict_t){IREE_HOST_SIZE_MAX,
                                                       IREE_HOST_SIZE_MAX};
  if (!count) {
    return iree_ok_status();
  }
  loom_pipeline_participants_sort(entries, count);
  for (iree_host_size_t i = 1; i < count; ++i) {
    if (entries[i - 1].engine == entries[i].engine &&
        entries[i].worker < out_conflict->second_worker) {
      *out_conflict = (loom_pipeline_execution_conflict_t){
          entries[i - 1].worker, entries[i].worker};
    }
  }
  if (out_conflict->second_worker != IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }
  return loom_pipeline_participants_build_sorted(arena, entries, count,
                                                 out_participants);
}

iree_status_t loom_pipeline_participants_merge(
    iree_arena_allocator_t* arena, const loom_pipeline_participants_t* first,
    const loom_pipeline_participants_t* second, bool concurrent,
    loom_pipeline_execution_conflict_t* conflict,
    const loom_pipeline_participants_t** out_participants) {
  if (!first || !second) {
    *out_participants = first ? first : second;
    return iree_ok_status();
  }
  if (first == second && !concurrent) {
    *out_participants = first;
    return iree_ok_status();
  }
  const iree_host_size_t difference =
      loom_pipeline_participants_high_bit(first->prefix ^ second->prefix);
  if (difference > first->branch && difference > second->branch) {
    const bool reverse = first->prefix > second->prefix;
    return loom_pipeline_participants_branch(
        arena, difference, reverse ? second : first, reverse ? first : second,
        out_participants);
  }
  if (!first->branch && !second->branch) {
    const uint64_t overlap = first->leaf.mask & second->leaf.mask;
    if (concurrent && overlap) {
      const iree_host_size_t engine =
          first->prefix + iree_math_count_trailing_zeros_u64(overlap);
      *conflict = (loom_pipeline_execution_conflict_t){
          loom_pipeline_participants_worker(first, engine),
          loom_pipeline_participants_worker(second, engine)};
      return iree_ok_status();
    }
    if (!(second->leaf.mask & ~first->leaf.mask)) {
      *out_participants = first;
      return iree_ok_status();
    }
    loom_pipeline_participants_t* leaf = NULL;
    iree_host_size_t* workers = NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_participants_leaf(
        arena, first->prefix, first->leaf.mask | second->leaf.mask, &leaf,
        &workers));
    iree_host_size_t a = 0, b = 0, index = 0;
    for (uint64_t remaining = leaf->leaf.mask; remaining;
         remaining &= remaining - 1) {
      const uint64_t bit = remaining & (~remaining + 1);
      workers[index++] = first->leaf.mask & bit ? first->leaf.workers[a]
                                                : second->leaf.workers[b];
      a += (first->leaf.mask & bit) != 0;
      b += (second->leaf.mask & bit) != 0;
    }
    *out_participants = leaf;
    return iree_ok_status();
  }
  const iree_host_size_t branch = iree_max(first->branch, second->branch);
  const loom_pipeline_participants_t* children[2];
  for (unsigned side = 0; side < 2; ++side) {
    const loom_pipeline_participants_t* a =
        first->branch == branch                   ? first->children[side]
        : ((first->prefix & branch) != 0) == side ? first
                                                  : NULL;
    const loom_pipeline_participants_t* b =
        second->branch == branch                   ? second->children[side]
        : ((second->prefix & branch) != 0) == side ? second
                                                   : NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_participants_merge(
        arena, a, b, concurrent, conflict, &children[side]));
    if (concurrent && conflict->second_worker != IREE_HOST_SIZE_MAX) {
      return iree_ok_status();
    }
  }
  if (first->branch == branch && first->children[0] == children[0] &&
      first->children[1] == children[1]) {
    *out_participants = first;
    return iree_ok_status();
  }
  if (second->branch == branch && second->children[0] == children[0] &&
      second->children[1] == children[1]) {
    *out_participants = second;
    return iree_ok_status();
  }
  return loom_pipeline_participants_branch(arena, branch, children[0],
                                           children[1], out_participants);
}

iree_status_t loom_pipeline_participants_insert(
    iree_arena_allocator_t* arena, loom_pipeline_participant_t participant,
    const loom_pipeline_participants_t** participants) {
  if (loom_pipeline_participants_worker(*participants, participant.engine) !=
      IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }
  loom_pipeline_participants_t* leaf = NULL;
  iree_host_size_t* workers = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_participants_leaf(
      arena, participant.engine & ~(iree_host_size_t)63,
      UINT64_C(1) << (participant.engine % 64), &leaf, &workers));
  workers[0] = participant.worker;
  return loom_pipeline_participants_merge(arena, *participants, leaf, false,
                                          NULL, participants);
}

void loom_pipeline_participants_begin(
    const loom_pipeline_participants_t* participants,
    loom_pipeline_participant_iterator_t* iterator) {
  iterator->pending_count = participants ? 1 : 0;
  iterator->pending[0] = participants;
  iterator->remaining = 0;
}

bool loom_pipeline_participants_next(
    loom_pipeline_participant_iterator_t* iterator,
    loom_pipeline_participant_t* out_participant) {
  if (!iterator->remaining) {
    if (!iterator->pending_count) {
      return false;
    }
    const loom_pipeline_participants_t* node =
        iterator->pending[--iterator->pending_count];
    while (node->branch) {
      iterator->pending[iterator->pending_count++] = node->children[1];
      node = node->children[0];
    }
    iterator->leaf = node;
    iterator->remaining = node->leaf.mask;
    iterator->worker_index = 0;
  }
  *out_participant = (loom_pipeline_participant_t){
      .engine = iterator->leaf->prefix +
                iree_math_count_trailing_zeros_u64(iterator->remaining),
      .worker = iterator->leaf->leaf.workers[iterator->worker_index++]};
  iterator->remaining &= iterator->remaining - 1;
  return true;
}

int loom_pipeline_participants_compare_members(
    const loom_pipeline_participants_t* first,
    const loom_pipeline_participants_t* second) {
  if (first == second) {
    return 0;
  }
  if (!first || !second) {
    return first ? 1 : -1;
  }
  if (!first->branch && !second->branch && first->prefix == second->prefix) {
    const uint64_t difference = first->leaf.mask ^ second->leaf.mask;
    if (!difference) {
      return 0;
    }
    return first->leaf.mask & (difference & (~difference + 1)) ? 1 : -1;
  }
  loom_pipeline_participant_iterator_t a, b;
  loom_pipeline_participants_begin(first, &a);
  loom_pipeline_participants_begin(second, &b);
  loom_pipeline_participant_t left, right;
  for (;;) {
    const bool has_left = loom_pipeline_participants_next(&a, &left);
    const bool has_right = loom_pipeline_participants_next(&b, &right);
    if (!has_left || !has_right) {
      return has_left - has_right;
    }
    if (left.engine != right.engine) {
      return left.engine < right.engine ? 1 : -1;
    }
  }
}
