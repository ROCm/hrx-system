// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/backing_pages.h"

#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"

// Hierarchy levels index free slots, then nonempty words, with fanout 64. The
// geometry determines the depth; neither claim nor return scans bitmap words.
typedef struct loom_serve_backing_level_t {
  // Word offset from the first trailing bitmap word.
  iree_host_size_t offset;
} loom_serve_backing_level_t;

typedef enum loom_serve_backing_state_e {
  LOOM_SERVE_BACKING_PREPARED,
  LOOM_SERVE_BACKING_PENDING,
  LOOM_SERVE_BACKING_READY,
  LOOM_SERVE_BACKING_FAILED,
} loom_serve_backing_state_t;

typedef struct loom_serve_backing_list_t {
  // Next parent in this list.
  loom_serve_backing_parent_t* next;
  // Previous link naming this parent, or NULL when unlinked.
  loom_serve_backing_parent_t** previous;
} loom_serve_backing_list_t;

typedef struct loom_serve_backing_class_t {
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
  loom_serve_backing_parent_t* available[2];
  // Entirely unused parents in the same acquisition-state partitions.
  loom_serve_backing_parent_t* idle[2];
} loom_serve_backing_class_t;

struct loom_serve_backing_pages_t {
  // Allocator for this owner and cold native-parent metadata.
  iree_allocator_t host_allocator;
  // Borrowed native source outliving all reservations and callbacks.
  iree_hal_pool_t* source;
  // Borrowed device constructing private allocation readiness semaphores.
  iree_hal_device_t* device;
  // Packed mapping parents and optional distinct whole-weight parents.
  loom_serve_backing_class_t classes[2];
  // Weight class ordinal; zero enables whole-parent/page reuse.
  uint8_t weight_class;
  // All retained parents, including failed or retiring acquisitions.
  loom_serve_backing_parent_t* parents;
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

struct loom_serve_backing_parent_t {
  // Serialized inventory owning this parent through native publication.
  loom_serve_backing_pages_t* pages;
  // Reusable physical class, or NULL for a dedicated extent.
  loom_serve_backing_class_t* category;
  // Intrusive ownership list.
  loom_serve_backing_list_t all;
  // Intrusive non-full-parent list.
  loom_serve_backing_list_t available;
  // Intrusive unused-parent list.
  loom_serve_backing_list_t idle;
  // Captured growth list, used only before submission.
  loom_serve_backing_parent_t* growth_next;
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
  // Usable fixed slots; dedicated and distinct whole parents have one.
  iree_host_size_t slot_count;
  // Number of hierarchy levels in trailing storage.
  iree_host_size_t level_count;
  // Source publication state; mutated only by the serialized owner.
  loom_serve_backing_state_t state;
  // A whole-parent lease excludes its otherwise-empty bitmap.
  bool whole;
  // Idle parent selected for physical disposal, no longer eligible for claims.
  bool retiring;
  // Whether retained_bytes still includes this parent's physical cost.
  bool retained;
  // Hierarchy descriptors followed by the bitmap words.
  loom_serve_backing_level_t levels[];
};

static uint64_t* loom_serve_backing_words(loom_serve_backing_parent_t* parent,
                                          iree_host_size_t level) {
  return (uint64_t*)(parent->levels + parent->level_count) +
         parent->levels[level].offset;
}

static void loom_serve_backing_unlink(loom_serve_backing_list_t* link,
                                      iree_host_size_t member_offset) {
  if (!link->previous) {
    return;
  }
  *link->previous = link->next;
  if (link->next) {
    loom_serve_backing_list_t* next =
        (loom_serve_backing_list_t*)((uint8_t*)link->next + member_offset);
    next->previous = link->previous;
  }
  *link = (loom_serve_backing_list_t){0};
}

static void loom_serve_backing_link(loom_serve_backing_parent_t* parent,
                                    loom_serve_backing_parent_t** head,
                                    iree_host_size_t member_offset) {
  loom_serve_backing_list_t* link =
      (loom_serve_backing_list_t*)((uint8_t*)parent + member_offset);
  link->next = *head;
  link->previous = head;
  if (*head) {
    ((loom_serve_backing_list_t*)((uint8_t*)*head + member_offset))->previous =
        &link->next;
  }
  *head = parent;
}

static bool loom_serve_backing_usable(
    const loom_serve_backing_parent_t* parent) {
  return parent->category && !parent->retiring &&
         parent->state != LOOM_SERVE_BACKING_FAILED;
}

static void loom_serve_backing_add_available(
    loom_serve_backing_parent_t* parent) {
  if (!loom_serve_backing_usable(parent)) {
    return;
  }
  const int ready = parent->state == LOOM_SERVE_BACKING_READY;
  if (parent->occupied < parent->slot_count && !parent->available.previous) {
    loom_serve_backing_link(parent, &parent->category->available[ready],
                            offsetof(loom_serve_backing_parent_t, available));
  }
  if (!parent->occupied && !parent->idle.previous) {
    loom_serve_backing_link(parent, &parent->category->idle[ready],
                            offsetof(loom_serve_backing_parent_t, idle));
  }
}

static void loom_serve_backing_remove_available(
    loom_serve_backing_parent_t* parent) {
  loom_serve_backing_unlink(&parent->available,
                            offsetof(loom_serve_backing_parent_t, available));
  loom_serve_backing_unlink(&parent->idle,
                            offsetof(loom_serve_backing_parent_t, idle));
}

static void loom_serve_backing_forget_capacity(
    loom_serve_backing_parent_t* parent) {
  if (loom_serve_backing_usable(parent)) {
    parent->category->free_count -= parent->slot_count - parent->occupied;
    if (parent->state == LOOM_SERVE_BACKING_READY) {
      parent->category->ready_free_count -=
          parent->slot_count - parent->occupied;
    }
    parent->category->idle_count -= parent->occupied == 0;
  }
  loom_serve_backing_remove_available(parent);
}

static void loom_serve_backing_forget_charge(
    loom_serve_backing_parent_t* parent) {
  if (parent->retained) {
    parent->pages->retained_bytes -= parent->cost;
    parent->retained = false;
  }
}

static void loom_serve_backing_parent_free(
    loom_serve_backing_parent_t* parent) {
  loom_serve_backing_pages_t* pages = parent->pages;
  loom_serve_backing_forget_capacity(parent);
  loom_serve_backing_forget_charge(parent);
  loom_serve_backing_unlink(&parent->all,
                            offsetof(loom_serve_backing_parent_t, all));
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

iree_status_t loom_serve_backing_pages_allocate(
    iree_hal_pool_t* source, iree_hal_device_t* device,
    iree_hal_pool_reservation_request_t packed,
    iree_hal_pool_reservation_request_t parent,
    iree_hal_pool_reservation_request_t weight, void (*wake)(void*),
    void* wake_user_data, iree_allocator_t host_allocator,
    loom_serve_backing_pages_t** out_pages) {
  *out_pages = NULL;
  loom_serve_backing_pages_t* pages = NULL;
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
  pages->classes[1].slot_size = weight.allocation_size;
  iree_status_t status = iree_ok_status();
  for (int i = 0; i <= pages->weight_class && iree_status_is_ok(status); ++i) {
    status = iree_hal_slab_pool_query_allocation_cost(
        source, 1, &pages->classes[i].request, &pages->classes[i].cost);
  }
  if (iree_status_is_ok(status)) {
    *out_pages = pages;
  } else {
    loom_serve_backing_pages_free(pages);
  }
  return status;
}

void loom_serve_backing_pages_free(loom_serve_backing_pages_t* pages) {
  if (!pages) {
    return;
  }
  IREE_ASSERT(!pages->pending_count);
  while (pages->parents) {
    IREE_ASSERT(!pages->parents->occupied);
    loom_serve_backing_parent_free(pages->parents);
  }
  iree_atomic_slist_deinitialize(&pages->completed);
  iree_allocator_free(pages->host_allocator, pages);
}

static uint64_t loom_serve_backing_add_cost(uint64_t total, uint64_t count,
                                            uint64_t cost) {
  if (cost && count > (UINT64_MAX - total) / cost) {
    return UINT64_MAX;
  }
  return total + count * cost;
}

uint64_t loom_serve_backing_pages_growth_cost(
    const loom_serve_backing_pages_t* pages, uint64_t packed_count,
    uint64_t weight_count, uint64_t dedicated_cost) {
  const loom_serve_backing_class_t* packed = &pages->classes[0];
  const loom_serve_backing_class_t* weight =
      &pages->classes[pages->weight_class];
  const uint64_t reused_weight = iree_min(weight_count, weight->idle_count);
  uint64_t total = loom_serve_backing_add_cost(
      dedicated_cost, weight_count - reused_weight, weight->cost);
  const uint64_t slots = packed->request.allocation_size / packed->slot_size;
  uint64_t available = packed->free_count;
  if (!pages->weight_class) {
    available -= reused_weight * slots;
  }
  const uint64_t missing =
      packed_count > available ? packed_count - available : 0;
  return loom_serve_backing_add_cost(
      total, missing / slots + (missing % slots != 0), packed->cost);
}

static iree_status_t loom_serve_backing_parent_allocate(
    loom_serve_backing_pages_t* pages, loom_serve_backing_class_t* category,
    iree_hal_pool_reservation_request_t request, uint64_t cost,
    loom_serve_backing_growth_t* growth,
    loom_serve_backing_parent_t** out_parent) {
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
          sizeof(loom_serve_backing_parent_t) +
              level_count * sizeof(loom_serve_backing_level_t),
          word_count, sizeof(uint64_t), &metadata_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "page bitmap exceeds host size");
  }
  loom_serve_backing_parent_t* parent = NULL;
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
    parent->levels[level] = (loom_serve_backing_level_t){offset};
    uint64_t* bitmap = loom_serve_backing_words(parent, level);
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
  loom_serve_backing_link(parent, &pages->parents,
                          offsetof(loom_serve_backing_parent_t, all));
  if (category) {
    category->free_count += slots;
    ++category->idle_count;
  }
  loom_serve_backing_add_available(parent);
  *out_parent = parent;
  return iree_ok_status();
}

iree_status_t loom_serve_backing_pages_claim(
    loom_serve_backing_pages_t* pages, loom_serve_backing_kind_t kind,
    iree_hal_pool_reservation_request_t request, uint64_t dedicated_cost,
    loom_serve_backing_growth_t* growth, loom_serve_backing_slot_t* out_slot) {
  *out_slot = (loom_serve_backing_slot_t){0};
  loom_serve_backing_class_t* category =
      kind == LOOM_SERVE_BACKING_DEDICATED
          ? NULL
          : &pages->classes[kind == LOOM_SERVE_BACKING_WEIGHT
                                ? pages->weight_class
                                : 0];
  loom_serve_backing_parent_t* parent = NULL;
  if (category) {
    loom_serve_backing_parent_t** list = kind == LOOM_SERVE_BACKING_WEIGHT
                                             ? category->idle
                                             : category->available;
    parent = list[1] ? list[1] : list[0];
  }
  if (!parent) {
    IREE_RETURN_IF_ERROR(loom_serve_backing_parent_allocate(
        pages, category, category ? category->request : request,
        category ? category->cost : dedicated_cost, growth, &parent));
  }
  if (!parent->occupied && category) {
    --category->idle_count;
    loom_serve_backing_unlink(&parent->idle,
                              offsetof(loom_serve_backing_parent_t, idle));
  }
  iree_host_size_t index = 0;
  if (kind != LOOM_SERVE_BACKING_PACKED) {
    parent->whole = true;
    parent->occupied = parent->slot_count;
    if (category) {
      category->free_count -= parent->slot_count;
      if (parent->state == LOOM_SERVE_BACKING_READY) {
        category->ready_free_count -= parent->slot_count;
      }
    }
  } else {
    for (iree_host_size_t level = parent->level_count; level-- > 0;) {
      index = index * 64 + iree_math_count_trailing_zeros_u64(
                               loom_serve_backing_words(parent, level)[index]);
    }
    iree_host_size_t position = index;
    for (iree_host_size_t level = 0; level < parent->level_count; ++level) {
      uint64_t* word = &loom_serve_backing_words(parent, level)[position / 64];
      *word &= ~(UINT64_C(1) << (position % 64));
      if (*word) {
        break;
      }
      position /= 64;
    }
    ++parent->occupied;
    --category->free_count;
    category->ready_free_count -= parent->state == LOOM_SERVE_BACKING_READY;
  }
  if (parent->occupied == parent->slot_count) {
    loom_serve_backing_unlink(&parent->available,
                              offsetof(loom_serve_backing_parent_t, available));
  }
  *out_slot = (loom_serve_backing_slot_t){
      parent, category ? index * category->slot_size : 0};
  return iree_ok_status();
}

void loom_serve_backing_pages_return(loom_serve_backing_slot_t slot) {
  loom_serve_backing_parent_t* parent = slot.parent;
  if (!parent) {
    return;
  }
  const uint64_t count = parent->whole ? parent->slot_count : 1;
  if (parent->whole) {
    parent->whole = false;
  } else {
    iree_host_size_t index = slot.offset / parent->category->slot_size;
    for (iree_host_size_t level = 0; level < parent->level_count; ++level) {
      uint64_t* word = &loom_serve_backing_words(parent, level)[index / 64];
      const uint64_t previous = *word;
      *word |= UINT64_C(1) << (index % 64);
      if (previous) {
        break;
      }
      index /= 64;
    }
  }
  parent->occupied -= count;
  if (loom_serve_backing_usable(parent)) {
    parent->category->free_count += count;
    if (parent->state == LOOM_SERVE_BACKING_READY) {
      parent->category->ready_free_count += count;
    }
    parent->category->idle_count += parent->occupied == 0;
    loom_serve_backing_add_available(parent);
  }
  if (!parent->occupied && parent->state != LOOM_SERVE_BACKING_PREPARED &&
      (!parent->category || parent->state == LOOM_SERVE_BACKING_FAILED)) {
    if (parent->state == LOOM_SERVE_BACKING_PENDING) {
      parent->retiring = true;
      loom_serve_backing_forget_charge(parent);
    } else {
      loom_serve_backing_parent_free(parent);
    }
  }
}

static void loom_serve_backing_acquired(void* user_data, iree_status_t status,
                                        iree_hal_pool_acquire_result_t result) {
  loom_serve_backing_parent_t* parent = user_data;
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

void loom_serve_backing_pages_submit(loom_serve_backing_growth_t* growth) {
  loom_serve_backing_parent_t* parent = growth->parents;
  growth->parents = NULL;
  while (parent) {
    loom_serve_backing_parent_t* next = parent->growth_next;
    parent->growth_next = NULL;
    parent->state = LOOM_SERVE_BACKING_PENDING;
    ++parent->pages->pending_count;
    iree_status_t status = iree_hal_slab_pool_acquire_with_credit(
        parent->pages->source, 1, &parent->request, &parent->credit,
        &parent->reservation, &parent->info,
        (iree_hal_pool_acquire_callback_t){loom_serve_backing_acquired,
                                           parent});
    if (!iree_status_is_ok(status)) {
      iree_hal_memory_credit_return(&parent->credit);
      loom_serve_backing_acquired(parent, status, IREE_HAL_POOL_ACQUIRE_NONE);
    }
    parent = next;
  }
  iree_hal_memory_credit_return(&growth->credit);
}

void loom_serve_backing_pages_abort(loom_serve_backing_growth_t* growth) {
  while (growth->parents) {
    loom_serve_backing_parent_t* parent = growth->parents;
    growth->parents = parent->growth_next;
    IREE_ASSERT(!parent->occupied);
    loom_serve_backing_parent_free(parent);
  }
  iree_hal_memory_credit_return(&growth->credit);
}

uint64_t loom_serve_backing_pages_begin_observation(
    loom_serve_backing_pages_t* pages) {
  return ++pages->observation;
}

iree_hal_semaphore_t* loom_serve_backing_pages_observe(
    loom_serve_backing_slot_t slot, uint64_t observation) {
  loom_serve_backing_parent_t* parent = slot.parent;
  if (parent->state == LOOM_SERVE_BACKING_READY ||
      parent->observation == observation) {
    return NULL;
  }
  parent->observation = observation;
  return parent->ready;
}

iree_hal_buffer_t* loom_serve_backing_pages_buffer(
    loom_serve_backing_slot_t slot) {
  return slot.parent->buffer;
}

void loom_serve_backing_pages_collect(loom_serve_backing_pages_t* pages) {
  iree_atomic_slist_entry_t* completed = NULL;
  iree_atomic_slist_flush(&pages->completed,
                          IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                          &completed, NULL);
  while (completed) {
    iree_atomic_slist_entry_t* next = completed->next;
    loom_serve_backing_parent_t* parent = iree_containerof(
        completed, loom_serve_backing_parent_t, completed_entry);
    iree_slim_mutex_lock(&parent->completion_mutex);
    iree_slim_mutex_unlock(&parent->completion_mutex);
    loom_serve_backing_forget_capacity(parent);
    parent->state = iree_status_is_ok(parent->status)
                        ? LOOM_SERVE_BACKING_READY
                        : LOOM_SERVE_BACKING_FAILED;
    --pages->pending_count;
    if (loom_serve_backing_usable(parent)) {
      parent->category->free_count += parent->slot_count - parent->occupied;
      parent->category->ready_free_count +=
          parent->slot_count - parent->occupied;
      parent->category->idle_count += parent->occupied == 0;
      loom_serve_backing_add_available(parent);
    }
    if (!parent->buffer) {
      loom_serve_backing_forget_charge(parent);
    }
    if (!parent->occupied &&
        (parent->retiring || parent->state == LOOM_SERVE_BACKING_FAILED)) {
      loom_serve_backing_parent_free(parent);
    }
    completed = next;
  }
}

void loom_serve_backing_pages_trim(loom_serve_backing_pages_t* pages,
                                   uint64_t native_bytes_to_return,
                                   uint64_t protected_packed_count,
                                   uint64_t protected_weight_count) {
  // Reserve whole-parent reuse for the incoming weight cohort before spending
  // the remaining free slots on its packed pages. Selection is owner-local.
  for (int ordinal = 1; ordinal >= 0 && native_bytes_to_return; --ordinal) {
    loom_serve_backing_class_t* category = &pages->classes[ordinal];
    const uint64_t slots =
        category->request.allocation_size / category->slot_size;
    const uint64_t weights =
        ordinal == pages->weight_class
            ? iree_min(protected_weight_count, category->idle_count)
            : 0;
    uint64_t protected_slots = weights * slots;
    if (!ordinal) {
      protected_slots += protected_packed_count;
    }
    while (native_bytes_to_return && category->idle_count > weights &&
           category->free_count >= slots &&
           category->free_count - slots >= protected_slots) {
      loom_serve_backing_parent_t* parent =
          category->idle[1] ? category->idle[1] : category->idle[0];
      const uint64_t cost = parent->cost;
      loom_serve_backing_forget_capacity(parent);
      parent->retiring = true;
      loom_serve_backing_forget_charge(parent);
      if (parent->state != LOOM_SERVE_BACKING_PENDING) {
        loom_serve_backing_parent_free(parent);
      }
      native_bytes_to_return -= iree_min(native_bytes_to_return, cost);
    }
  }
}

uint64_t loom_serve_backing_pages_retained(
    const loom_serve_backing_pages_t* pages) {
  return pages->retained_bytes;
}
uint64_t loom_serve_backing_pages_cached(
    const loom_serve_backing_pages_t* pages) {
  return pages->classes[0].ready_free_count * pages->classes[0].slot_size +
         pages->classes[1].ready_free_count * pages->classes[1].slot_size;
}
bool loom_serve_backing_pages_is_pending(
    const loom_serve_backing_pages_t* pages) {
  return pages->pending_count != 0;
}
void loom_serve_backing_pages_await(loom_serve_backing_pages_t* pages) {
  for (loom_serve_backing_parent_t* parent = pages->parents; parent;
       parent = parent->all.next) {
    if (parent->state == LOOM_SERVE_BACKING_PENDING) {
      // The operation and parent retain the diagnostic. This join only protects
      // borrowed source/group lifetimes during outer-owner destruction.
      iree_status_ignore(iree_hal_semaphore_wait(parent->ready, 1,
                                                 iree_infinite_timeout(),
                                                 IREE_ASYNC_WAIT_FLAG_NONE));
    }
  }
  loom_serve_backing_pages_collect(pages);
}
