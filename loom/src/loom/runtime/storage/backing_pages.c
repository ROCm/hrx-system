// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/backing_pages.h"

#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"

// Hierarchy levels index free slots, then nonempty words, with fanout 64. The
// geometry determines the depth; neither claim nor return scans bitmap words.
typedef struct loom_runtime_backing_level_t {
  // Word offset from the first trailing bitmap word.
  iree_host_size_t offset;
} loom_runtime_backing_level_t;

typedef enum loom_runtime_backing_state_e {
  LOOM_RUNTIME_BACKING_PREPARED,
  LOOM_RUNTIME_BACKING_PENDING,
  LOOM_RUNTIME_BACKING_READY,
  LOOM_RUNTIME_BACKING_FAILED,
} loom_runtime_backing_state_t;

typedef struct loom_runtime_backing_list_t {
  // Next parent in this list.
  loom_runtime_backing_parent_t* next;
  // Previous link naming this parent, or NULL when unlinked.
  loom_runtime_backing_parent_t** previous;
} loom_runtime_backing_list_t;

typedef struct loom_runtime_backing_class_t {
  // Exact native parent geometry, independent of logical mapping lengths.
  iree_hal_pool_reservation_request_t request;
  // Immutable source-quoted physical cost of one parent.
  uint64_t cost;
  // Fixed physical slot width; whole-parent claims bypass the bitmap.
  iree_device_size_t slot_size;
  // Number of available physical slots across usable parents.
  uint64_t free_count;
  // Free slots whose native acquisitions have completed successfully.
  uint64_t ready_free_count;
  // Number of entirely unused parents, a subset of free_count.
  uint64_t idle_count;
  // Non-full parents partitioned into pending and ready acquisition states.
  loom_runtime_backing_parent_t* available[2];
  // Entirely unused parents in the same acquisition-state partitions.
  loom_runtime_backing_parent_t* idle[2];
} loom_runtime_backing_class_t;

struct loom_runtime_backing_pages_t {
  // Allocator for this owner and cold native-parent metadata.
  iree_allocator_t host_allocator;
  // Borrowed native source outliving all reservations and callbacks.
  iree_hal_pool_t* source;
  // Borrowed device constructing private allocation readiness semaphores.
  iree_hal_device_t* device;
  // Parent size classes sharing the same packed mapping granularity.
  loom_runtime_backing_class_t classes[2];
  // Class whose full extent can satisfy a weight mapping.
  uint8_t weight_class;
  // All retained parents, including failed or retiring acquisitions.
  loom_runtime_backing_parent_t* parents;
  // Accepted source callbacks awaiting serialized publication.
  iree_atomic_slist_t completed;
  // Physical cost retained by this inventory rather than native retirement.
  uint64_t retained_bytes;
  // Number of native acquisitions not yet published by collect.
  iree_host_size_t pending_count;
  // Serialized transaction identity for unique parent observations.
  uint64_t observation;
  // Fixed nonblocking owner wake, joined by callback publication.
  void (*wake)(void*);
  // Borrowed scheduling owner alive through all native completions.
  void* wake_user_data;
};

struct loom_runtime_backing_parent_t {
  // Serialized inventory owning this parent through native publication.
  loom_runtime_backing_pages_t* pages;
  // Reusable physical class, or NULL for a dedicated extent.
  loom_runtime_backing_class_t* category;
  // Intrusive ownership list.
  loom_runtime_backing_list_t all;
  // Intrusive non-full-parent list.
  loom_runtime_backing_list_t available;
  // Intrusive unused-parent list.
  loom_runtime_backing_list_t idle;
  // Captured growth list, used only before submission.
  loom_runtime_backing_parent_t* growth_next;
  // Callback-to-owner publication link.
  iree_atomic_slist_entry_t completed_entry;
  // Joins the callback's final access before owner publication/destruction.
  iree_slim_mutex_t completion_mutex;
  // Owned exact allocation request, borrowed by the source until completion.
  iree_hal_pool_reservation_request_t request;
  // Owned native reservation, returned after the buffer view is released.
  iree_hal_pool_reservation_t reservation;
  // Source acquisition result storage.
  iree_hal_pool_acquire_info_t info;
  // Prepared physical credit transferred before native acquisition starts.
  iree_hal_memory_credit_t credit;
  // Native payload represented by this parent, including source padding.
  uint64_t cost;
  // Owned source view published before readiness, shared by every slot.
  iree_hal_buffer_t* buffer;
  // Owned private allocation completion, independent of caller cancellation.
  iree_hal_semaphore_t* ready;
  // Owned native acquisition/materialization/readiness result.
  iree_status_t status;
  // Last serialized transaction that observed this pending parent.
  uint64_t observation;
  // Number of leases, including mappings whose private unmap is pending.
  iree_host_size_t occupied;
  // Usable packed slots; dedicated parents have one.
  iree_host_size_t slot_count;
  // Number of hierarchy levels in trailing storage.
  iree_host_size_t level_count;
  // Source publication state; mutated only by the serialized owner.
  loom_runtime_backing_state_t state;
  // A whole-parent lease excludes its otherwise-empty bitmap.
  bool whole;
  // Idle parent selected for physical disposal, no longer eligible for claims.
  bool retiring;
  // Whether retained_bytes still includes this parent's physical cost.
  bool retained;
  // Hierarchy descriptors followed by the bitmap words.
  loom_runtime_backing_level_t levels[];
};

static uint64_t* loom_runtime_backing_words(
    loom_runtime_backing_parent_t* parent, iree_host_size_t level) {
  return (uint64_t*)(parent->levels + parent->level_count) +
         parent->levels[level].offset;
}

static void loom_runtime_backing_unlink(loom_runtime_backing_list_t* link,
                                        iree_host_size_t member_offset) {
  if (!link->previous) {
    return;
  }
  *link->previous = link->next;
  if (link->next) {
    loom_runtime_backing_list_t* next =
        (loom_runtime_backing_list_t*)((uint8_t*)link->next + member_offset);
    next->previous = link->previous;
  }
  *link = (loom_runtime_backing_list_t){0};
}

static void loom_runtime_backing_link(loom_runtime_backing_parent_t* parent,
                                      loom_runtime_backing_parent_t** head,
                                      iree_host_size_t member_offset) {
  loom_runtime_backing_list_t* link =
      (loom_runtime_backing_list_t*)((uint8_t*)parent + member_offset);
  link->next = *head;
  link->previous = head;
  if (*head) {
    ((loom_runtime_backing_list_t*)((uint8_t*)*head + member_offset))
        ->previous = &link->next;
  }
  *head = parent;
}

static bool loom_runtime_backing_usable(
    const loom_runtime_backing_parent_t* parent) {
  return parent->category && !parent->retiring &&
         parent->state != LOOM_RUNTIME_BACKING_FAILED;
}

static void loom_runtime_backing_add_available(
    loom_runtime_backing_parent_t* parent) {
  if (!loom_runtime_backing_usable(parent)) {
    return;
  }
  const int ready = parent->state == LOOM_RUNTIME_BACKING_READY;
  if (parent->occupied < parent->slot_count && !parent->available.previous) {
    loom_runtime_backing_link(
        parent, &parent->category->available[ready],
        offsetof(loom_runtime_backing_parent_t, available));
  }
  if (!parent->occupied && !parent->idle.previous) {
    loom_runtime_backing_link(parent, &parent->category->idle[ready],
                              offsetof(loom_runtime_backing_parent_t, idle));
  }
}

static void loom_runtime_backing_remove_available(
    loom_runtime_backing_parent_t* parent) {
  loom_runtime_backing_unlink(
      &parent->available, offsetof(loom_runtime_backing_parent_t, available));
  loom_runtime_backing_unlink(&parent->idle,
                              offsetof(loom_runtime_backing_parent_t, idle));
}

static void loom_runtime_backing_forget_capacity(
    loom_runtime_backing_parent_t* parent) {
  if (loom_runtime_backing_usable(parent)) {
    parent->category->free_count -= parent->slot_count - parent->occupied;
    if (parent->state == LOOM_RUNTIME_BACKING_READY) {
      parent->category->ready_free_count -=
          parent->slot_count - parent->occupied;
    }
    parent->category->idle_count -= parent->occupied == 0;
  }
  loom_runtime_backing_remove_available(parent);
}

static void loom_runtime_backing_forget_charge(
    loom_runtime_backing_parent_t* parent) {
  if (parent->retained) {
    parent->pages->retained_bytes -= parent->cost;
    parent->retained = false;
  }
}

static void loom_runtime_backing_parent_free(
    loom_runtime_backing_parent_t* parent) {
  loom_runtime_backing_pages_t* pages = parent->pages;
  loom_runtime_backing_forget_capacity(parent);
  loom_runtime_backing_forget_charge(parent);
  loom_runtime_backing_unlink(&parent->all,
                              offsetof(loom_runtime_backing_parent_t, all));
  iree_hal_buffer_release(parent->buffer);
  if (parent->reservation.block_handle) {
    iree_hal_pool_release_reservations(pages->source, 1, &parent->reservation,
                                       NULL);
  }
  iree_hal_memory_credit_return(&parent->credit);
  iree_hal_semaphore_release(parent->ready);
  iree_status_free(parent->status);
  iree_slim_mutex_deinitialize(&parent->completion_mutex);
  iree_allocator_free(pages->host_allocator, parent);
}

iree_status_t loom_runtime_backing_pages_allocate(
    iree_hal_pool_t* source, iree_hal_device_t* device,
    iree_hal_pool_reservation_request_t packed,
    iree_hal_pool_reservation_request_t parent,
    iree_hal_pool_reservation_request_t weight, void (*wake)(void*),
    void* wake_user_data, iree_allocator_t host_allocator,
    loom_runtime_backing_pages_t** out_pages) {
  *out_pages = NULL;
  loom_runtime_backing_pages_t* pages = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*pages), (void**)&pages));
  pages->host_allocator = host_allocator;
  pages->source = source;
  pages->device = device;
  pages->wake = wake;
  pages->wake_user_data = wake_user_data;
  iree_atomic_slist_initialize(&pages->completed);
  pages->weight_class = parent.allocation_size != weight.allocation_size;
  if (!pages->weight_class) {
    parent.min_alignment = iree_max(parent.min_alignment, weight.min_alignment);
    weight = parent;
  }
  pages->classes[0].request = parent;
  pages->classes[0].slot_size = packed.allocation_size;
  pages->classes[1].request = weight;
  pages->classes[1].slot_size = packed.allocation_size;
  iree_status_t status = iree_ok_status();
  for (int i = 0; i <= pages->weight_class && iree_status_is_ok(status); ++i) {
    status = iree_hal_slab_pool_query_allocation_cost(
        source, 1, &pages->classes[i].request, &pages->classes[i].cost);
  }
  if (iree_status_is_ok(status)) {
    *out_pages = pages;
  } else {
    loom_runtime_backing_pages_free(pages);
  }
  return status;
}

void loom_runtime_backing_pages_free(loom_runtime_backing_pages_t* pages) {
  if (!pages) {
    return;
  }
  IREE_ASSERT(!pages->pending_count);
  while (pages->parents) {
    IREE_ASSERT(!pages->parents->occupied);
    loom_runtime_backing_parent_free(pages->parents);
  }
  iree_atomic_slist_deinitialize(&pages->completed);
  iree_allocator_free(pages->host_allocator, pages);
}

static uint64_t loom_runtime_backing_add_cost(uint64_t total, uint64_t count,
                                              uint64_t cost) {
  if (cost && count > (UINT64_MAX - total) / cost) {
    return UINT64_MAX;
  }
  return total + count * cost;
}

uint64_t loom_runtime_backing_pages_growth_cost(
    const loom_runtime_backing_pages_t* pages, uint64_t packed_count,
    uint64_t weight_count, uint64_t dedicated_cost) {
  const loom_runtime_backing_class_t* packed = &pages->classes[0];
  const loom_runtime_backing_class_t* weight =
      &pages->classes[pages->weight_class];
  const uint64_t reused_weight = iree_min(weight_count, weight->idle_count);
  uint64_t total = loom_runtime_backing_add_cost(
      dedicated_cost, weight_count - reused_weight, weight->cost);
  const uint64_t slots = packed->request.allocation_size / packed->slot_size;
  const uint64_t weight_slots =
      weight->request.allocation_size / weight->slot_size;
  const uint64_t available = pages->classes[0].free_count +
                             pages->classes[1].free_count -
                             reused_weight * weight_slots;
  const uint64_t missing =
      packed_count > available ? packed_count - available : 0;
  return loom_runtime_backing_add_cost(
      total, missing / slots + (missing % slots != 0), packed->cost);
}

static iree_status_t loom_runtime_backing_parent_allocate(
    loom_runtime_backing_pages_t* pages, loom_runtime_backing_class_t* category,
    iree_hal_pool_reservation_request_t request, uint64_t cost,
    loom_runtime_backing_growth_t* growth,
    loom_runtime_backing_parent_t** out_parent) {
  const iree_host_size_t slots =
      category ? request.allocation_size / category->slot_size : 1;
  iree_host_size_t level_count = 0, word_count = 0, count = slots;
  do {
    count = 1 + (count - 1) / 64;
    ++level_count;
    if (!iree_host_size_checked_add(word_count, count, &word_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "page bitmap exceeds host size");
    }
  } while (count > 1);
  iree_host_size_t metadata_size = 0;
  if (!iree_host_size_checked_mul_add(
          sizeof(loom_runtime_backing_parent_t) +
              level_count * sizeof(loom_runtime_backing_level_t),
          word_count, sizeof(uint64_t), &metadata_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "page bitmap exceeds host size");
  }
  loom_runtime_backing_parent_t* parent = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pages->host_allocator,
                                             metadata_size, (void**)&parent));
  parent->pages = pages;
  parent->category = category;
  parent->request = request;
  parent->cost = cost;
  parent->slot_count = slots;
  parent->level_count = level_count;
  iree_slim_mutex_initialize(&parent->completion_mutex);
  iree_status_t status = iree_hal_semaphore_create(
      pages->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &parent->ready);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_deinitialize(&parent->completion_mutex);
    iree_allocator_free(pages->host_allocator, parent);
    return status;
  }
  count = slots;
  iree_host_size_t offset = 0;
  for (iree_host_size_t level = 0; level < level_count; ++level) {
    const iree_host_size_t words = 1 + (count - 1) / 64;
    parent->levels[level] = (loom_runtime_backing_level_t){offset};
    uint64_t* bitmap = loom_runtime_backing_words(parent, level);
    for (iree_host_size_t i = 0; i < words; ++i) {
      const iree_host_size_t valid =
          iree_min((iree_host_size_t)64, count - i * 64);
      bitmap[i] = UINT64_MAX >> (64 - valid);
    }
    offset += words;
    count = words;
  }
  parent->credit = iree_hal_memory_credit_split(&growth->credit, cost);
  parent->retained = true;
  pages->retained_bytes += cost;
  parent->growth_next = growth->parents;
  growth->parents = parent;
  loom_runtime_backing_link(parent, &pages->parents,
                            offsetof(loom_runtime_backing_parent_t, all));
  if (category) {
    category->free_count += slots;
    ++category->idle_count;
  }
  loom_runtime_backing_add_available(parent);
  *out_parent = parent;
  return iree_ok_status();
}

iree_status_t loom_runtime_backing_pages_prepare_packed(
    loom_runtime_backing_pages_t* pages, uint64_t packed_count,
    loom_runtime_backing_growth_t* growth) {
  loom_runtime_backing_class_t* category = &pages->classes[0];
  const uint64_t cost =
      loom_runtime_backing_pages_growth_cost(pages, packed_count, 0, 0);
  const uint64_t count = cost / category->cost;
  iree_status_t status = iree_ok_status();
  for (uint64_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    loom_runtime_backing_parent_t* parent = NULL;
    status = loom_runtime_backing_parent_allocate(
        pages, category, category->request, category->cost, growth, &parent);
  }
  return status;
}

// Takes the first contiguous prefix of free slots from the chosen parent.
// Empty leaf words clear their summary bit once, regardless of run length.
static iree_host_size_t loom_runtime_backing_parent_claim_packed(
    loom_runtime_backing_parent_t* parent, iree_host_size_t maximum_count,
    iree_host_size_t* out_index) {
  iree_host_size_t index = 0;
  for (iree_host_size_t level = parent->level_count; level-- > 0;) {
    index = index * 64 + iree_math_count_trailing_zeros_u64(
                             loom_runtime_backing_words(parent, level)[index]);
  }
  *out_index = index;
  const iree_host_size_t limit =
      iree_min(maximum_count, parent->slot_count - index);
  uint64_t* leaves = loom_runtime_backing_words(parent, 0);
  iree_host_size_t count = 0;
  while (count < limit) {
    iree_host_size_t position = (index + count) / 64;
    const int32_t bit = (int32_t)((index + count) % 64);
    const int32_t width =
        (int32_t)iree_min(limit - count, (iree_host_size_t)(64 - bit));
    const int32_t take = iree_math_count_trailing_zeros_u64_width(
        ~(leaves[position] >> bit), width);
    leaves[position] &= ~(iree_math_mask_low_bits_u64(UINT64_MAX, take) << bit);
    if (take && !leaves[position]) {
      for (iree_host_size_t level = 1; level < parent->level_count; ++level) {
        uint64_t* word =
            &loom_runtime_backing_words(parent, level)[position / 64];
        *word &= ~(UINT64_C(1) << (position % 64));
        if (*word) {
          break;
        }
        position /= 64;
      }
    }
    count += take;
    if (take < width) {
      break;
    }
  }
  return count;
}

iree_status_t loom_runtime_backing_pages_claim(
    loom_runtime_backing_pages_t* pages, loom_runtime_backing_kind_t kind,
    iree_hal_pool_reservation_request_t request, uint64_t dedicated_cost,
    iree_host_size_t maximum_count, loom_runtime_backing_growth_t* growth,
    loom_runtime_backing_slot_t* out_slot, iree_host_size_t* out_count) {
  *out_count = 0;
  *out_slot = (loom_runtime_backing_slot_t){0};
  loom_runtime_backing_class_t* category =
      kind == LOOM_RUNTIME_BACKING_DEDICATED
          ? NULL
          : &pages->classes[kind == LOOM_RUNTIME_BACKING_WEIGHT
                                ? pages->weight_class
                                : 0];
  loom_runtime_backing_parent_t* parent = NULL;
  if (kind == LOOM_RUNTIME_BACKING_PACKED) {
    // This cohort already needs every newly prepared parent. Starting with
    // their aligned free prefixes keeps consecutive target pages contiguous;
    // older holes still satisfy the remainder without extra physical growth.
    // PREPARED parents belong to this serialized capture, not prior work.
    parent = pages->classes[0].available[0];
    if (parent && parent->state != LOOM_RUNTIME_BACKING_PREPARED) {
      parent = NULL;
    }
    // Existing ready pages precede earlier pending acquisitions, regardless
    // of the native allocation size. An unrelated pending parent cannot delay
    // a cohort that fits ready inventory.
    for (int ready = 1; ready >= 0 && !parent; --ready) {
      for (int ordinal = 0; ordinal <= pages->weight_class && !parent;
           ++ordinal) {
        parent = pages->classes[ordinal].available[ready];
      }
    }
    if (parent) {
      category = parent->category;
    }
  } else if (category) {
    parent = category->idle[1] ? category->idle[1] : category->idle[0];
  }
  if (!parent) {
    IREE_RETURN_IF_ERROR(loom_runtime_backing_parent_allocate(
        pages, category, category ? category->request : request,
        category ? category->cost : dedicated_cost, growth, &parent));
  }
  if (!parent->occupied && category) {
    --category->idle_count;
    loom_runtime_backing_unlink(&parent->idle,
                                offsetof(loom_runtime_backing_parent_t, idle));
  }
  iree_host_size_t index = 0;
  iree_host_size_t count = 1;
  if (kind != LOOM_RUNTIME_BACKING_PACKED) {
    parent->whole = true;
    parent->occupied = parent->slot_count;
    if (category) {
      category->free_count -= parent->slot_count;
      if (parent->state == LOOM_RUNTIME_BACKING_READY) {
        category->ready_free_count -= parent->slot_count;
      }
    }
  } else {
    count =
        loom_runtime_backing_parent_claim_packed(parent, maximum_count, &index);
    parent->occupied += count;
    category->free_count -= count;
    if (parent->state == LOOM_RUNTIME_BACKING_READY) {
      category->ready_free_count -= count;
    }
  }
  if (parent->occupied == parent->slot_count) {
    loom_runtime_backing_unlink(
        &parent->available, offsetof(loom_runtime_backing_parent_t, available));
  }
  *out_count = count;
  *out_slot = (loom_runtime_backing_slot_t){
      parent, category ? index * category->slot_size : 0};
  return iree_ok_status();
}

void loom_runtime_backing_pages_return(loom_runtime_backing_slot_t slot,
                                       iree_host_size_t slot_count) {
  loom_runtime_backing_parent_t* parent = slot.parent;
  if (!parent) {
    return;
  }
  const uint64_t count = parent->whole ? parent->slot_count : slot_count;
  if (parent->whole) {
    parent->whole = false;
  } else {
    iree_host_size_t index = slot.offset / parent->category->slot_size;
    uint64_t* leaves = loom_runtime_backing_words(parent, 0);
    for (iree_host_size_t remaining = slot_count; remaining;) {
      iree_host_size_t position = index / 64;
      const int32_t bit = (int32_t)(index % 64);
      const int32_t width =
          (int32_t)iree_min(remaining, (iree_host_size_t)(64 - bit));
      const uint64_t previous = leaves[position];
      leaves[position] |= iree_math_mask_low_bits_u64(UINT64_MAX, width) << bit;
      if (!previous) {
        for (iree_host_size_t level = 1; level < parent->level_count; ++level) {
          uint64_t* word =
              &loom_runtime_backing_words(parent, level)[position / 64];
          const uint64_t occupied = *word;
          *word |= UINT64_C(1) << (position % 64);
          if (occupied) {
            break;
          }
          position /= 64;
        }
      }
      index += width;
      remaining -= width;
    }
  }
  parent->occupied -= count;
  if (loom_runtime_backing_usable(parent)) {
    parent->category->free_count += count;
    if (parent->state == LOOM_RUNTIME_BACKING_READY) {
      parent->category->ready_free_count += count;
    }
    parent->category->idle_count += parent->occupied == 0;
    loom_runtime_backing_add_available(parent);
  }
  if (!parent->occupied && parent->state != LOOM_RUNTIME_BACKING_PREPARED &&
      (!parent->category || parent->state == LOOM_RUNTIME_BACKING_FAILED)) {
    if (parent->state == LOOM_RUNTIME_BACKING_PENDING) {
      parent->retiring = true;
      loom_runtime_backing_forget_charge(parent);
    } else {
      loom_runtime_backing_parent_free(parent);
    }
  }
}

static void loom_runtime_backing_acquired(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  loom_runtime_backing_parent_t* parent = user_data;
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_materialize_reservations(
        parent->pages->source, 1, &parent->request, &parent->reservation,
        IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &parent->buffer);
  }
  iree_slim_mutex_lock(&parent->completion_mutex);
  parent->status = status;
  iree_atomic_slist_push(&parent->pages->completed, &parent->completed_entry);
  if (iree_status_is_ok(parent->status)) {
    parent->status = iree_hal_semaphore_signal(parent->ready, 1, NULL);
  }
  if (!iree_status_is_ok(parent->status)) {
    iree_hal_semaphore_fail(parent->ready, iree_status_clone(parent->status));
  }
  if (parent->pages->wake) {
    parent->pages->wake(parent->pages->wake_user_data);
  }
  iree_slim_mutex_unlock(&parent->completion_mutex);
}

void loom_runtime_backing_pages_submit(loom_runtime_backing_growth_t* growth) {
  loom_runtime_backing_parent_t* parent = growth->parents;
  growth->parents = NULL;
  while (parent) {
    loom_runtime_backing_parent_t* next = parent->growth_next;
    parent->growth_next = NULL;
    parent->state = LOOM_RUNTIME_BACKING_PENDING;
    ++parent->pages->pending_count;
    iree_status_t status = iree_hal_slab_pool_acquire_with_credit(
        parent->pages->source, 1, &parent->request, &parent->credit,
        (iree_hal_pool_acquire_callback_t){loom_runtime_backing_acquired,
                                           parent},
        &parent->reservation, &parent->info);
    if (!iree_status_is_ok(status)) {
      iree_hal_memory_credit_return(&parent->credit);
      loom_runtime_backing_acquired(parent, status, IREE_HAL_POOL_ACQUIRE_NONE);
    }
    parent = next;
  }
  iree_hal_memory_credit_return(&growth->credit);
}

void loom_runtime_backing_pages_abort(loom_runtime_backing_growth_t* growth) {
  while (growth->parents) {
    loom_runtime_backing_parent_t* parent = growth->parents;
    growth->parents = parent->growth_next;
    IREE_ASSERT(!parent->occupied);
    loom_runtime_backing_parent_free(parent);
  }
  iree_hal_memory_credit_return(&growth->credit);
}

uint64_t loom_runtime_backing_pages_begin_observation(
    loom_runtime_backing_pages_t* pages) {
  return ++pages->observation;
}

iree_hal_semaphore_t* loom_runtime_backing_pages_observe(
    loom_runtime_backing_slot_t slot, uint64_t observation) {
  loom_runtime_backing_parent_t* parent = slot.parent;
  if (parent->state == LOOM_RUNTIME_BACKING_READY ||
      parent->observation == observation) {
    return NULL;
  }
  parent->observation = observation;
  return parent->ready;
}

iree_hal_buffer_t* loom_runtime_backing_pages_buffer(
    loom_runtime_backing_slot_t slot) {
  return slot.parent->buffer;
}

void loom_runtime_backing_pages_collect(loom_runtime_backing_pages_t* pages) {
  iree_atomic_slist_entry_t* completed = NULL;
  iree_atomic_slist_flush(&pages->completed,
                          IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                          &completed, NULL);
  while (completed) {
    iree_atomic_slist_entry_t* next = completed->next;
    loom_runtime_backing_parent_t* parent = iree_containerof(
        completed, loom_runtime_backing_parent_t, completed_entry);
    iree_slim_mutex_lock(&parent->completion_mutex);
    iree_slim_mutex_unlock(&parent->completion_mutex);
    loom_runtime_backing_forget_capacity(parent);
    parent->state = iree_status_is_ok(parent->status)
                        ? LOOM_RUNTIME_BACKING_READY
                        : LOOM_RUNTIME_BACKING_FAILED;
    --pages->pending_count;
    if (loom_runtime_backing_usable(parent)) {
      parent->category->free_count += parent->slot_count - parent->occupied;
      parent->category->ready_free_count +=
          parent->slot_count - parent->occupied;
      parent->category->idle_count += parent->occupied == 0;
      loom_runtime_backing_add_available(parent);
    }
    if (!parent->buffer) {
      loom_runtime_backing_forget_charge(parent);
    }
    if (!parent->occupied &&
        (parent->retiring || parent->state == LOOM_RUNTIME_BACKING_FAILED)) {
      loom_runtime_backing_parent_free(parent);
    }
    completed = next;
  }
}

void loom_runtime_backing_pages_trim(loom_runtime_backing_pages_t* pages,
                                     uint64_t native_bytes_to_return,
                                     uint64_t protected_packed_count,
                                     uint64_t protected_weight_count) {
  // Reserve whole-parent reuse for the incoming weight cohort before spending
  // the remaining free slots on its packed pages. Selection is owner-local.
  const loom_runtime_backing_class_t* weight =
      &pages->classes[pages->weight_class];
  const uint64_t protected_weights =
      iree_min(protected_weight_count, weight->idle_count);
  uint64_t available =
      pages->classes[0].free_count + pages->classes[1].free_count -
      protected_weights * (weight->request.allocation_size / weight->slot_size);
  for (int ordinal = 1; ordinal >= 0 && native_bytes_to_return; --ordinal) {
    loom_runtime_backing_class_t* category = &pages->classes[ordinal];
    const uint64_t slots =
        category->request.allocation_size / category->slot_size;
    const uint64_t weights =
        ordinal == pages->weight_class
            ? iree_min(protected_weight_count, category->idle_count)
            : 0;
    while (native_bytes_to_return && category->idle_count > weights &&
           available >= slots && available - slots >= protected_packed_count) {
      loom_runtime_backing_parent_t* parent =
          category->idle[1] ? category->idle[1] : category->idle[0];
      const uint64_t cost = parent->cost;
      loom_runtime_backing_forget_capacity(parent);
      available -= slots;
      parent->retiring = true;
      loom_runtime_backing_forget_charge(parent);
      if (parent->state != LOOM_RUNTIME_BACKING_PENDING) {
        loom_runtime_backing_parent_free(parent);
      }
      native_bytes_to_return -= iree_min(native_bytes_to_return, cost);
    }
  }
}

uint64_t loom_runtime_backing_pages_retained(
    const loom_runtime_backing_pages_t* pages) {
  return pages->retained_bytes;
}
uint64_t loom_runtime_backing_pages_cached(
    const loom_runtime_backing_pages_t* pages) {
  return pages->classes[0].ready_free_count * pages->classes[0].slot_size +
         pages->classes[1].ready_free_count * pages->classes[1].slot_size;
}
bool loom_runtime_backing_pages_is_pending(
    const loom_runtime_backing_pages_t* pages) {
  return pages->pending_count != 0;
}
iree_status_t loom_runtime_backing_pages_await(
    loom_runtime_backing_pages_t* pages) {
  iree_status_t status = iree_ok_status();
  for (loom_runtime_backing_parent_t* parent = pages->parents; parent;
       parent = parent->all.next) {
    if (parent->state == LOOM_RUNTIME_BACKING_PENDING) {
      status = iree_status_join(
          status,
          iree_hal_semaphore_wait(parent->ready, 1, iree_infinite_timeout(),
                                  IREE_ASYNC_WAIT_FLAG_NONE));
    }
  }
  loom_runtime_backing_pages_collect(pages);
  return status;
}

iree_status_t loom_runtime_backing_pages_prewarm(
    loom_runtime_backing_pages_t* pages, iree_hal_memory_budget_t* budget,
    uint64_t minimum_bytes) {
  const uint64_t slot_size = pages->classes[0].slot_size;
  const uint64_t slots =
      minimum_bytes / slot_size + (minimum_bytes % slot_size != 0);
  const uint64_t cost =
      loom_runtime_backing_pages_growth_cost(pages, slots, 0, 0);
  loom_runtime_backing_growth_t growth = {0};
  if (!iree_hal_memory_budget_try_reserve(budget, cost, &growth.credit)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "prewarming %" PRIu64 " bytes requires %" PRIu64
        " bytes of physical backing; pool limit is %" PRIu64 " bytes",
        minimum_bytes, cost, iree_hal_memory_budget_limit(budget));
  }
  iree_status_t status =
      loom_runtime_backing_pages_prepare_packed(pages, slots, &growth);
  if (!iree_status_is_ok(status)) {
    loom_runtime_backing_pages_abort(&growth);
    return status;
  }
  loom_runtime_backing_pages_submit(&growth);
  return loom_runtime_backing_pages_await(pages);
}
