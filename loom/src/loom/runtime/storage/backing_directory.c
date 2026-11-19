// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/backing_directory.h"

static int loom_runtime_backing_rank(uint64_t mask, int position) {
  return iree_math_count_ones_u64_width(mask, position);
}

loom_runtime_backing_slot_t loom_runtime_backing_group_slot(
    const loom_runtime_backing_group_t* group, int position,
    iree_device_size_t stride) {
  if (group->fragment_capacity) {
    return group->sources
        .fragments[loom_runtime_backing_rank(group->assigned, position)];
  }
  return (loom_runtime_backing_slot_t){
      group->sources.origin.parent,
      group->sources.origin.offset + position * stride};
}

iree_status_t loom_runtime_backing_group_assign(
    loom_runtime_backing_group_t* group, iree_allocator_t allocator,
    uint64_t mask, iree_device_size_t stride,
    loom_runtime_backing_slot_t first) {
  const int position = iree_math_count_trailing_zeros_u64(mask);
  const loom_runtime_backing_slot_t origin = {first.parent,
                                              first.offset - position * stride};
  if (!group->assigned || (!group->fragment_capacity &&
                           group->sources.origin.parent == origin.parent &&
                           group->sources.origin.offset == origin.offset)) {
    group->sources.origin = origin;
  } else {
    const int previous = iree_math_count_ones_u64(group->assigned);
    const int added = iree_math_count_ones_u64(mask);
    if (previous + added > group->fragment_capacity) {
      // Admission has already marked the whole accepted interval. Once this
      // group fragments, reserve its outstanding pages in the same allocation.
      const uint32_t capacity = iree_math_round_up_to_pow2_u32(
          iree_math_count_ones_u64(group->assigned | group->in_flight | mask));
      loom_runtime_backing_slot_t* slots =
          group->fragment_capacity ? group->sources.fragments : NULL;
      if (!slots) {
        IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
            allocator, capacity, sizeof(*slots), (void**)&slots));
        uint64_t existing = group->assigned;
        for (int i = 0; existing; ++i, existing &= existing - 1) {
          slots[i] = loom_runtime_backing_group_slot(
              group, iree_math_count_trailing_zeros_u64(existing), stride);
        }
      } else {
        IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
            allocator, capacity, sizeof(*slots), (void**)&slots));
      }
      group->sources.fragments = slots;
      group->fragment_capacity = (uint8_t)capacity;
    }
    const int insert = loom_runtime_backing_rank(group->assigned, position);
    memmove(group->sources.fragments + insert + added,
            group->sources.fragments + insert,
            (previous - insert) * sizeof(*group->sources.fragments));
    for (int i = 0; i < added; ++i) {
      group->sources.fragments[insert + i] = (loom_runtime_backing_slot_t){
          first.parent, first.offset + i * stride};
    }
  }
  group->assigned |= mask;
  return iree_ok_status();
}

void loom_runtime_backing_group_clear(loom_runtime_backing_group_t* group,
                                      iree_allocator_t allocator, uint64_t mask,
                                      iree_device_size_t stride) {
  const uint64_t removed = group->assigned & mask;
  if (!removed) {
    return;
  }
  const uint64_t remaining = group->assigned & ~mask;
  if (group->fragment_capacity) {
    if (!(remaining & (remaining - 1))) {
      loom_runtime_backing_slot_t* fragments = group->sources.fragments;
      if (remaining) {
        const int position = iree_math_count_trailing_zeros_u64(remaining);
        group->sources.origin =
            loom_runtime_backing_group_slot(group, position, stride);
        group->sources.origin.offset -= position * stride;
      }
      iree_allocator_free(allocator, fragments);
      group->fragment_capacity = 0;
    } else {
      const int count = iree_math_count_ones_u64(removed);
      const int start = loom_runtime_backing_rank(
          group->assigned, iree_math_count_trailing_zeros_u64(removed));
      memmove(group->sources.fragments + start,
              group->sources.fragments + start + count,
              (iree_math_count_ones_u64(remaining) - start) *
                  sizeof(*group->sources.fragments));
    }
  }
  group->assigned = remaining;
}

iree_host_size_t loom_runtime_backing_group_take_lease(
    const loom_runtime_backing_group_t* group, uint64_t* mask,
    iree_device_size_t stride, int* out_position,
    loom_runtime_backing_slot_t* out_slot) {
  uint64_t available = *mask;
  iree_host_size_t count =
      loom_runtime_backing_take_run(&available, out_position);
  *out_slot = loom_runtime_backing_group_slot(group, *out_position, stride);
  if (group->fragment_capacity) {
    const int first = loom_runtime_backing_rank(group->assigned, *out_position);
    for (iree_host_size_t i = 1; i < count; ++i) {
      const loom_runtime_backing_slot_t next =
          group->sources.fragments[first + i];
      if (next.parent != out_slot->parent ||
          next.offset != out_slot->offset + i * stride) {
        count = i;
        break;
      }
    }
  }
  *mask &=
      ~(iree_math_mask_low_bits_u64(UINT64_MAX, (int)count) << *out_position);
  return count;
}

static iree_host_size_t loom_runtime_backing_directory_hash(
    iree_host_size_t index) {
  return index * (iree_host_size_t)UINT64_C(11400714819323198485);
}

static void loom_runtime_backing_directory_release_storage(
    loom_runtime_backing_directory_t* directory) {
  iree_allocator_free(directory->host_allocator, directory->groups);
  directory->groups = NULL;
  directory->capacity = 0;
  directory->tombstone_count = 0;
}

static const loom_runtime_backing_group_t* loom_runtime_backing_directory_probe(
    const loom_runtime_backing_directory_t* directory, iree_host_size_t index,
    bool for_insert) {
  if (!directory->capacity) {
    return NULL;
  }
  const iree_host_size_t mask = directory->capacity - 1;
  iree_host_size_t position = loom_runtime_backing_directory_hash(index) & mask;
  const loom_runtime_backing_group_t* first_tombstone = NULL;
  for (;;) {
    const loom_runtime_backing_group_t* entry = &directory->groups[position];
    if (entry->occupied) {
      if (entry->index == index) {
        return entry;
      }
    } else if (entry->tombstone) {
      if (for_insert && !first_tombstone) {
        first_tombstone = entry;
      }
    } else {
      if (!for_insert) {
        return NULL;
      }
      return first_tombstone ? first_tombstone : entry;
    }
    position = (position + 1) & mask;
  }
}

const loom_runtime_backing_group_t* loom_runtime_backing_directory_find_const(
    const loom_runtime_backing_directory_t* directory, iree_host_size_t index) {
  return loom_runtime_backing_directory_probe(directory, index, false);
}

loom_runtime_backing_group_t* loom_runtime_backing_directory_find(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index) {
  return (loom_runtime_backing_group_t*)loom_runtime_backing_directory_probe(
      directory, index, false);
}

static void loom_runtime_backing_directory_place(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_group_t* entry, loom_runtime_backing_group_t value) {
  if (entry->tombstone) {
    --directory->tombstone_count;
  }
  value.occupied = true;
  value.tombstone = false;
  *entry = value;
  ++directory->count;
}

static iree_status_t loom_runtime_backing_directory_rehash(
    loom_runtime_backing_directory_t* directory, iree_host_size_t capacity) {
  loom_runtime_backing_group_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      directory->host_allocator, capacity, sizeof(*entries), (void**)&entries));
  loom_runtime_backing_group_t* old_entries = directory->groups;
  const iree_host_size_t old_capacity = directory->capacity;
  directory->groups = entries;
  directory->capacity = capacity;
  directory->count = 0;
  directory->tombstone_count = 0;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (old_entries[i].occupied) {
      loom_runtime_backing_group_t* entry =
          (loom_runtime_backing_group_t*)loom_runtime_backing_directory_probe(
              directory, old_entries[i].index, true);
      loom_runtime_backing_directory_place(directory, entry, old_entries[i]);
    }
  }
  iree_allocator_free(directory->host_allocator, old_entries);
  return iree_ok_status();
}

static iree_host_size_t loom_runtime_backing_directory_limit(
    iree_host_size_t capacity) {
  return capacity - iree_max((iree_host_size_t)1, capacity / 4);
}

void loom_runtime_backing_directory_initialize(
    iree_allocator_t host_allocator,
    loom_runtime_backing_directory_t* out_directory) {
  *out_directory = (loom_runtime_backing_directory_t){
      .host_allocator = host_allocator,
  };
}

void loom_runtime_backing_directory_deinitialize(
    loom_runtime_backing_directory_t* directory) {
  for (iree_host_size_t i = 0; i < directory->capacity; ++i) {
    if (directory->groups[i].fragment_capacity) {
      iree_allocator_free(directory->host_allocator,
                          directory->groups[i].sources.fragments);
    }
  }
  iree_allocator_free(directory->host_allocator, directory->groups);
  *directory = (loom_runtime_backing_directory_t){0};
}

iree_status_t loom_runtime_backing_directory_insert(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index,
    loom_runtime_backing_group_t** out_entry) {
  loom_runtime_backing_group_t* entry =
      (loom_runtime_backing_group_t*)loom_runtime_backing_directory_probe(
          directory, index, true);
  if (!entry || !entry->occupied) {
    // Reusing this probe chain's tombstone consumes no vacancy. A tombstone
    // elsewhere cannot terminate this chain: keep a genuinely empty bucket.
    if (!entry ||
        (!entry->tombstone &&
         directory->count + directory->tombstone_count + 1 >
             loom_runtime_backing_directory_limit(directory->capacity))) {
      iree_host_size_t capacity = directory->capacity ? directory->capacity : 2;
      if (directory->count + 1 >
          loom_runtime_backing_directory_limit(capacity)) {
        capacity *= 2;
      }
      IREE_RETURN_IF_ERROR(
          loom_runtime_backing_directory_rehash(directory, capacity));
      entry =
          (loom_runtime_backing_group_t*)loom_runtime_backing_directory_probe(
              directory, index, true);
    }
    loom_runtime_backing_directory_place(
        directory, entry, (loom_runtime_backing_group_t){.index = index});
  }
  *out_entry = entry;
  return iree_ok_status();
}

void loom_runtime_backing_directory_remove(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_group_t* entry) {
  IREE_ASSERT(entry->occupied);
  *entry = (loom_runtime_backing_group_t){.tombstone = true};
  --directory->count;
  ++directory->tombstone_count;
}

void loom_runtime_backing_directory_prune(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_group_t* group) {
  if (!(group->pending_commit | group->in_flight | group->published |
        group->selected)) {
    loom_runtime_backing_directory_remove(directory, group);
  }
}

loom_runtime_backing_cursor_t loom_runtime_backing_directory_iterate(
    loom_runtime_backing_directory_t* directory, iree_host_size_t begin,
    iree_host_size_t end) {
  const iree_host_size_t first = begin / LOOM_RUNTIME_BACKING_GROUP_SIZE;
  const iree_host_size_t last = end / LOOM_RUNTIME_BACKING_GROUP_SIZE +
                                !!(end % LOOM_RUNTIME_BACKING_GROUP_SIZE);
  const bool sparse = last - first > directory->capacity;
  return (loom_runtime_backing_cursor_t){directory,
                                         begin,
                                         end,
                                         sparse ? 0 : first,
                                         begin == end ? 0
                                         : sparse     ? directory->capacity
                                                      : last,
                                         sparse};
}

loom_runtime_backing_group_t* loom_runtime_backing_directory_next(
    loom_runtime_backing_cursor_t* cursor, uint64_t* out_mask) {
  while (cursor->position < cursor->limit) {
    const iree_host_size_t position = cursor->position++;
    loom_runtime_backing_group_t* group =
        cursor->sparse
            ? &cursor->directory->groups[position]
            : loom_runtime_backing_directory_find(cursor->directory, position);
    if (!group || !group->occupied ||
        group->index < cursor->begin / LOOM_RUNTIME_BACKING_GROUP_SIZE ||
        group->index > (cursor->end - 1) / LOOM_RUNTIME_BACKING_GROUP_SIZE) {
      continue;
    }
    const iree_host_size_t begin =
        iree_max(cursor->begin, group->index * LOOM_RUNTIME_BACKING_GROUP_SIZE);
    *out_mask = loom_runtime_backing_group_mask(begin, cursor->end);
    return group;
  }
  return NULL;
}

void loom_runtime_backing_directory_discard_empty(
    loom_runtime_backing_directory_t* directory) {
  for (iree_host_size_t i = 0; i < directory->capacity; ++i) {
    loom_runtime_backing_group_t* entry = &directory->groups[i];
    if (entry->occupied &&
        !(entry->assigned | entry->in_flight | entry->published)) {
      IREE_ASSERT(!entry->pending_commit && !entry->selected);
      loom_runtime_backing_directory_remove(directory, entry);
    }
  }
  if (!directory->count) {
    loom_runtime_backing_directory_release_storage(directory);
  }
}

iree_status_t loom_runtime_backing_directory_shrink(
    loom_runtime_backing_directory_t* directory) {
  if (!directory->count) {
    loom_runtime_backing_directory_release_storage(directory);
    return iree_ok_status();
  }
  iree_host_size_t capacity = 2;
  while (directory->count > loom_runtime_backing_directory_limit(capacity)) {
    capacity *= 2;
  }
  if (capacity >= directory->capacity) {
    return iree_ok_status();
  }
  return loom_runtime_backing_directory_rehash(directory, capacity);
}
