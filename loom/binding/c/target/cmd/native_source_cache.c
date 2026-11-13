// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/target/cmd/native_source_cache.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/binding/c/src/result.h"
#include "loom/transforms/kernel/kernel_specialization_key.h"
#include "loomc/iree.h"

typedef struct loomc_cmd_native_source_t loomc_cmd_native_source_t;
typedef struct loomc_cmd_native_plan_t loomc_cmd_native_plan_t;

// Preserves the allocator's natural alignment after the accounting header.
typedef struct iree_alignas(iree_max_align_t) loomc_cmd_source_allocation_t {
  // Total requested bytes, including this header.
  iree_host_size_t length;
} loomc_cmd_source_allocation_t;

typedef struct loomc_cmd_native_entry_t {
  // Allocator owning this entry and its accounting allocations.
  iree_allocator_t allocator;
  // Live requested bytes including this entry, allocation headers, and pool.
  iree_host_size_t owned_bytes;
  // Independently reclaimable backing owned by this immutable entry.
  iree_arena_block_pool_t block_pool;
  // Key and projection storage, never used by concurrent invocations.
  iree_arena_allocator_t arena;
  // Selects the source or compact-plan ownership contract.
  bool is_plan;
  // True once retained; detached entries stay charged until destruction ends.
  bool retained;
  // Publisher leases, plus one source lease for each retained child plan.
  iree_host_size_t use_count;
  // Earlier idle entry eligible for eviction.
  struct loomc_cmd_native_entry_t* idle_previous;
  // Later idle entry eligible for eviction.
  struct loomc_cmd_native_entry_t* idle_next;
} loomc_cmd_native_entry_t;

struct loomc_cmd_native_source_t {
  // Common accounting and independently reclaimable source storage.
  loomc_cmd_native_entry_t entry;
  // Native module, candidate headers, and common classifier/models.
  loom_kernel_request_preparation_t* preparation;
  // Successful preparation diagnostics, or NULL when none were emitted.
  loomc_result_t* diagnostics;
  // Immutable identity within the owner's exact source index.
  iree_host_size_t symbol_ordinal;
  // Resolution policy used to prepare this source.
  loomc_config_policy_flags_t config_policy_flags;
  // Homogeneous specialization semantics applied during preparation.
  loomc_link_deployment_specialization_phase_t deployment_phase;
  // Retained homogeneous execution profile, or NULL.
  loomc_target_profile_t* deployment_profile;
  // Owned source-index assignments with individually retained profiles.
  loomc_link_target_specializations_t targets;
  // One value per preparation config key; empty means explicitly absent.
  iree_string_view_t* config_values;
  // Intrusive source bucket link.
  loomc_cmd_native_source_t* next;
  // Previous bucket link for constant-time removal.
  loomc_cmd_native_source_t** previous_link;
  // Root of the bounded hash-bit index of compact staged plans.
  loomc_cmd_native_plan_t* plans;
};

struct loomc_cmd_native_plan_t {
  // Common accounting and independently reclaimable compact storage.
  loomc_cmd_native_entry_t entry;
  // Source pinned by retention, or borrowed from the active publisher.
  loomc_cmd_native_source_t* source;
  // Exact owned cross-module identity of the staged tuple.
  loom_kernel_specialization_key_t key;
  // Immutable compact projections borrowing only this plan and its source.
  loom_kernel_class_classifier_t classifier;
  // Lookup accelerator; equality always compares the complete semantic key.
  uint32_t hash;
  // Subsequent plans with exactly the same hash, newest first.
  loomc_cmd_native_plan_t* next;
  // Hash-bit children; only the equal-hash chain head owns these links.
  loomc_cmd_native_plan_t* children[2];
};

struct loomc_cmd_native_source_cache_t {
  // Immutable provider universe borrowed from the owner.
  const loom_link_module_index_t* index;
  // Allocator for cache and entry storage.
  iree_allocator_t allocator;
  // Protects completed publication, leases, indices, and byte accounting.
  iree_slim_mutex_t mutex;
  // Single maximum for completed sources and compact plans together.
  iree_host_size_t byte_limit;
  // Charged bytes, including detached entries until destruction finishes.
  iree_host_size_t retained_bytes;
  // Source-index buckets for completed entries.
  loomc_cmd_native_source_t** sources;
  // Oldest idle source or plan eligible for eviction.
  loomc_cmd_native_entry_t* idle_head;
  // Most recently released idle source or plan.
  loomc_cmd_native_entry_t* idle_tail;
};

static iree_status_t loomc_cmd_entry_allocator_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  loomc_cmd_native_entry_t* entry = self;
  if (command == IREE_ALLOCATOR_COMMAND_TRIM) {
    return iree_allocator_trim(entry->allocator);
  }
  loomc_cmd_source_allocation_t* allocation =
      *inout_ptr ? (loomc_cmd_source_allocation_t*)*inout_ptr - 1 : NULL;
  if (command == IREE_ALLOCATOR_COMMAND_FREE) {
    if (allocation != NULL) {
      entry->owned_bytes -= allocation->length;
      iree_allocator_free(entry->allocator, allocation);
      *inout_ptr = NULL;
    }
    return iree_ok_status();
  }
  const iree_allocator_alloc_params_t* request = params;
  iree_host_size_t length = 0;
  if (!iree_host_size_checked_add(request->byte_length, sizeof(*allocation),
                                  &length)) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  const iree_host_size_t old_length = allocation ? allocation->length : 0;
  const iree_allocator_alloc_params_t expanded = {.byte_length = length};
  void* storage = allocation;
  IREE_RETURN_IF_ERROR(entry->allocator.ctl(entry->allocator.self, command,
                                            &expanded, &storage));
  allocation = storage;
  allocation->length = length;
  entry->owned_bytes = entry->owned_bytes - old_length + length;
  *inout_ptr = allocation + 1;
  return iree_ok_status();
}

static iree_allocator_t loomc_cmd_entry_allocator(
    loomc_cmd_native_entry_t* entry) {
  return (iree_allocator_t){.self = entry,
                            .ctl = loomc_cmd_entry_allocator_ctl};
}

static void loomc_cmd_entry_free(loomc_cmd_native_entry_t* entry) {
  iree_arena_deinitialize(&entry->arena);
  iree_arena_block_pool_deinitialize(&entry->block_pool);
  iree_allocator_free(entry->allocator, entry);
}

static void loomc_cmd_native_source_free(loomc_cmd_native_source_t* source) {
  if (source == NULL) {
    return;
  }
  loomc_result_release(source->diagnostics);
  for (iree_host_size_t i = 0; i < source->targets.count; ++i) {
    loomc_target_profile_release(source->targets.assignments[i].profile);
  }
  loomc_target_profile_release(source->deployment_profile);
  loom_kernel_request_preparation_free(source->preparation);
  loomc_cmd_entry_free(&source->entry);
}

static iree_string_view_t loomc_cmd_source_config_value(
    const loom_config_text_binding_set_t* bindings, iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < bindings->binding_count; ++i) {
    if (iree_string_view_equal(bindings->bindings[i].key, key)) {
      return bindings->bindings[i].value;
    }
  }
  // Normalization rejects empty bound values, making empty an exact absence.
  return iree_string_view_empty();
}

static bool loomc_cmd_native_source_matches(
    const loomc_cmd_native_source_t* source,
    const loomc_link_materialization_state_t* state) {
  if (source->config_policy_flags !=
          state->specialization.config_policy_flags ||
      source->deployment_phase != state->specialization.deployment_phase ||
      !loomc_target_profile_equal(source->deployment_profile,
                                  state->specialization.deployment_profile)) {
    return false;
  }
  const loomc_link_target_specializations_t* targets =
      state->specialization.indexed;
  const iree_host_size_t target_count = targets ? targets->count : 0;
  if (source->targets.count != target_count ||
      source->targets.function_count !=
          (targets ? targets->function_count : 0)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loomc_link_target_assignment_t* lhs = &source->targets.assignments[i];
    const loomc_link_target_assignment_t* rhs = &targets->assignments[i];
    if (lhs->symbol_ordinal != rhs->symbol_ordinal ||
        !loomc_target_profile_equal(lhs->profile, rhs->profile)) {
      return false;
    }
  }
  const iree_string_view_list_t keys =
      loom_kernel_request_preparation_config_keys(source->preparation);
  for (iree_host_size_t i = 0; i < keys.count; ++i) {
    if (!iree_string_view_equal(
            source->config_values[i],
            loomc_cmd_source_config_value(
                &state->specialization.config_bindings, keys.values[i]))) {
      return false;
    }
  }
  return true;
}

static iree_status_t loomc_cmd_native_source_capture_key(
    loomc_cmd_native_source_t* source,
    const loomc_link_materialization_state_t* state) {
  source->config_policy_flags = state->specialization.config_policy_flags;
  source->deployment_phase = state->specialization.deployment_phase;
  source->deployment_profile =
      (loomc_target_profile_t*)state->specialization.deployment_profile;
  loomc_target_profile_retain(source->deployment_profile);
  const loomc_link_target_specializations_t* targets =
      state->specialization.indexed;
  if (targets != NULL) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&source->entry.arena, targets->count,
                                  sizeof(*source->targets.assignments),
                                  (void**)&source->targets.assignments));
    source->targets.count = targets->count;
    source->targets.function_count = targets->function_count;
    for (iree_host_size_t i = 0; i < targets->count; ++i) {
      source->targets.assignments[i] = targets->assignments[i];
      loomc_target_profile_retain(source->targets.assignments[i].profile);
    }
  }
  const iree_string_view_list_t keys =
      loom_kernel_request_preparation_config_keys(source->preparation);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &source->entry.arena, keys.count, sizeof(*source->config_values),
      (void**)&source->config_values));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < keys.count && iree_status_is_ok(status);
       ++i) {
    const iree_string_view_t value = loomc_cmd_source_config_value(
        &state->specialization.config_bindings, keys.values[i]);
    char* storage = NULL;
    status =
        iree_arena_allocate(&source->entry.arena, value.size, (void**)&storage);
    if (iree_status_is_ok(status)) {
      if (value.size != 0) {
        memcpy(storage, value.data, value.size);
      }
      source->config_values[i] = iree_make_string_view(storage, value.size);
    }
  }
  return status;
}

static iree_status_t loomc_cmd_native_source_prepare(
    loomc_cmd_native_source_cache_t* cache,
    loomc_link_materialization_state_t* state,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t symbol_ordinal, loomc_cmd_native_source_t** out_source) {
  *out_source = NULL;
  loomc_cmd_native_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(cache->allocator, sizeof(*source),
                                             (void**)&source));
  source->entry.allocator = cache->allocator;
  source->entry.owned_bytes = sizeof(*source);
  source->symbol_ordinal = symbol_ordinal;
  const iree_allocator_t allocator = loomc_cmd_entry_allocator(&source->entry);
  iree_arena_block_pool_initialize(environment->block_pool->total_block_size,
                                   allocator, &source->entry.block_pool);
  iree_arena_initialize(&source->entry.block_pool, &source->entry.arena);
  iree_status_t status = iree_status_from_loomc(loomc_result_create(
      LOOMC_STATUS_OK, loomc_result_source_retention(state->diagnostics.result),
      loomc_allocator_from_iree(allocator), &source->diagnostics));
  // This stack view borrows normalized configuration and resolved target rows;
  // it never takes ownership or deinitializes the invocation's state. Only its
  // fresh diagnostics and source backing may survive preparation.
  loomc_link_materialization_state_t preparation_state = *state;
  preparation_state.diagnostics.result = source->diagnostics;
  preparation_state.diagnostics.source = NULL;
  preparation_state.arena = &source->entry.arena;
  preparation_state.allocator = loomc_allocator_from_iree(allocator);
  loom_link_plan_materialization_environment_t preparation_environment =
      *environment;
  preparation_environment.block_pool = &source->entry.block_pool;
  preparation_environment.allocator = allocator;
  preparation_environment.user_data = &preparation_state;
  if (iree_status_is_ok(status)) {
    status = loom_kernel_request_preparation_allocate(
        cache->index, &preparation_environment, symbol_ordinal,
        &source->preparation);
  }
  if (iree_status_is_ok(status)) {
    status = loomc_cmd_native_source_capture_key(source, &preparation_state);
  }
  if (iree_status_is_ok(status)) {
    // Empty results retain neither storage nor the first invocation's result.
    if (loomc_result_diagnostic_count(source->diagnostics) == 0) {
      loomc_result_release(source->diagnostics);
      source->diagnostics = NULL;
    }
    // Temporary link/planner blocks are not part of the retained preparation.
    iree_arena_block_pool_trim(&source->entry.block_pool);
    *out_source = source;
  } else {
    status = iree_status_join(
        status, iree_status_from_loomc(loomc_result_copy_diagnostics(
                    state->diagnostics.result, source->diagnostics)));
    loomc_cmd_native_source_free(source);
  }
  return status;
}

static void loomc_cmd_entry_idle_remove(loomc_cmd_native_source_cache_t* cache,
                                        loomc_cmd_native_entry_t* entry) {
  if (entry->idle_previous) {
    entry->idle_previous->idle_next = entry->idle_next;
  } else {
    cache->idle_head = entry->idle_next;
  }
  if (entry->idle_next) {
    entry->idle_next->idle_previous = entry->idle_previous;
  } else {
    cache->idle_tail = entry->idle_previous;
  }
  entry->idle_previous = NULL;
  entry->idle_next = NULL;
}

static void loomc_cmd_entry_acquire_locked(
    loomc_cmd_native_source_cache_t* cache, loomc_cmd_native_entry_t* entry) {
  if (entry->use_count++ == 0) {
    loomc_cmd_entry_idle_remove(cache, entry);
  }
}

static void loomc_cmd_entry_release_locked(
    loomc_cmd_native_source_cache_t* cache, loomc_cmd_native_entry_t* entry) {
  if (--entry->use_count == 0) {
    entry->idle_previous = cache->idle_tail;
    if (cache->idle_tail) {
      cache->idle_tail->idle_next = entry;
    } else {
      cache->idle_head = entry;
    }
    cache->idle_tail = entry;
  }
}

// One hash bit per occupied level bounds lookup by the hash width. No index
// allocation or rehashing is needed under the cache mutex.
static loomc_cmd_native_plan_t** loomc_cmd_plan_bucket(
    loomc_cmd_native_source_t* source, uint32_t hash) {
  loomc_cmd_native_plan_t** link = &source->plans;
  uint32_t depth = 0;
  while (*link && (*link)->hash != hash) {
    IREE_ASSERT_LT(depth, 32u);
    link = &(*link)->children[(hash >> depth++) & 1u];
  }
  return link;
}

static void loomc_cmd_plan_insert(loomc_cmd_native_plan_t* plan) {
  loomc_cmd_native_plan_t** link =
      loomc_cmd_plan_bucket(plan->source, plan->hash);
  plan->next = *link;
  if (*link) {
    plan->children[0] = (*link)->children[0];
    plan->children[1] = (*link)->children[1];
    (*link)->children[0] = NULL;
    (*link)->children[1] = NULL;
  }
  *link = plan;
}

static void loomc_cmd_plan_remove(loomc_cmd_native_plan_t* plan) {
  loomc_cmd_native_plan_t** link =
      loomc_cmd_plan_bucket(plan->source, plan->hash);
  if (*link != plan) {
    while (*link != plan) {
      link = &(*link)->next;
    }
    *link = plan->next;
  } else if (plan->next) {
    *link = plan->next;
    (*link)->children[0] = plan->children[0];
    (*link)->children[1] = plan->children[1];
  } else if (!plan->children[0] && !plan->children[1]) {
    *link = NULL;
  } else {
    // Any descendant leaf shares the removed node's complete ancestor prefix.
    // Move its whole equal-hash chain; no key or borrowed entry address
    // changes.
    loomc_cmd_native_plan_t** leaf_link =
        &plan->children[plan->children[0] ? 0 : 1];
    while ((*leaf_link)->children[0] || (*leaf_link)->children[1]) {
      leaf_link = &(*leaf_link)->children[(*leaf_link)->children[0] ? 0 : 1];
    }
    loomc_cmd_native_plan_t* leaf = *leaf_link;
    *leaf_link = NULL;
    leaf->children[0] = plan->children[0];
    leaf->children[1] = plan->children[1];
    *link = leaf;
  }
}

// Locked on entry/return. The detached entry remains charged and its parent
// remains pinned until destruction has completed outside the mutex.
static void loomc_cmd_entry_retire_locked(
    loomc_cmd_native_source_cache_t* cache, loomc_cmd_native_entry_t* entry) {
  loomc_cmd_entry_idle_remove(cache, entry);
  const iree_host_size_t bytes = entry->owned_bytes;
  loomc_cmd_native_source_t* parent = NULL;
  if (entry->is_plan) {
    loomc_cmd_native_plan_t* plan = (loomc_cmd_native_plan_t*)entry;
    parent = plan->source;
    loomc_cmd_plan_remove(plan);
  } else {
    loomc_cmd_native_source_t* source = (loomc_cmd_native_source_t*)entry;
    *source->previous_link = source->next;
    if (source->next) {
      source->next->previous_link = source->previous_link;
    }
  }
  iree_slim_mutex_unlock(&cache->mutex);
  if (parent) {
    loomc_cmd_entry_free(entry);
  } else {
    loomc_cmd_native_source_free((loomc_cmd_native_source_t*)entry);
  }
  iree_slim_mutex_lock(&cache->mutex);
  cache->retained_bytes -= bytes;
  if (parent) {
    loomc_cmd_entry_release_locked(cache, &parent->entry);
  }
}

static void loomc_cmd_cache_make_room_locked(
    loomc_cmd_native_source_cache_t* cache, iree_host_size_t bytes) {
  const iree_host_size_t available = cache->byte_limit - cache->retained_bytes;
  if (bytes <= available) {
    return;
  }
  const iree_host_size_t needed = bytes - available;
  iree_host_size_t reclaimed = 0;
  while (reclaimed < needed && cache->idle_head) {
    loomc_cmd_native_entry_t* entry = cache->idle_head;
    reclaimed += entry->owned_bytes;
    loomc_cmd_entry_retire_locked(cache, entry);
  }
  // Concurrent publication can claim freed capacity. The caller checks actual
  // space once more and uses its private entry if retention is no longer
  // possible.
}

static loomc_cmd_native_source_t* loomc_cmd_source_lookup(
    loomc_cmd_native_source_cache_t* cache,
    const loomc_link_materialization_state_t* state,
    iree_host_size_t symbol_ordinal) {
  for (loomc_cmd_native_source_t* source = cache->sources[symbol_ordinal];
       source; source = source->next) {
    if (loomc_cmd_native_source_matches(source, state)) {
      loomc_cmd_entry_acquire_locked(cache, &source->entry);
      return source;
    }
  }
  return NULL;
}

static loomc_cmd_native_source_t* loomc_cmd_source_publish_completed(
    loomc_cmd_native_source_cache_t* cache,
    const loomc_link_materialization_state_t* state,
    loomc_cmd_native_source_t* candidate) {
  if (candidate->entry.owned_bytes > cache->byte_limit) {
    return candidate;
  }
  iree_slim_mutex_lock(&cache->mutex);
  loomc_cmd_native_source_t* winner =
      loomc_cmd_source_lookup(cache, state, candidate->symbol_ordinal);
  if (!winner) {
    loomc_cmd_cache_make_room_locked(cache, candidate->entry.owned_bytes);
    winner = loomc_cmd_source_lookup(cache, state, candidate->symbol_ordinal);
  }
  if (!winner && candidate->entry.owned_bytes <=
                     cache->byte_limit - cache->retained_bytes) {
    candidate->next = cache->sources[candidate->symbol_ordinal];
    candidate->previous_link = &cache->sources[candidate->symbol_ordinal];
    if (candidate->next) {
      candidate->next->previous_link = &candidate->next;
    }
    *candidate->previous_link = candidate;
    candidate->entry.retained = true;
    candidate->entry.use_count = 1;
    cache->retained_bytes += candidate->entry.owned_bytes;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  if (winner) {
    loomc_cmd_native_source_free(candidate);
    return winner;
  }
  return candidate;
}

static void loomc_cmd_source_release(loomc_cmd_native_source_cache_t* cache,
                                     loomc_cmd_native_source_t* source) {
  if (!source->entry.retained) {
    loomc_cmd_native_source_free(source);
  } else {
    iree_slim_mutex_lock(&cache->mutex);
    loomc_cmd_entry_release_locked(cache, &source->entry);
    iree_slim_mutex_unlock(&cache->mutex);
  }
}

// Locked on entry/return. Exact payload comparison borrows pinned chain heads
// and candidates, and runs unlocked. Only concurrently inserted prefixes need
// another visit; eviction can remove unvisited entries but cannot retire a pin.
static loomc_cmd_native_plan_t* loomc_cmd_plan_lookup_locked(
    loomc_cmd_native_source_cache_t* cache, loomc_cmd_native_source_t* source,
    const loom_kernel_specialization_query_t* query) {
  const uint32_t hash = loom_kernel_specialization_query_hash(query);
  loomc_cmd_native_plan_t* head = *loomc_cmd_plan_bucket(source, hash);
  if (!head) {
    return NULL;
  }
  loomc_cmd_entry_acquire_locked(cache, &head->entry);
  loomc_cmd_native_plan_t* boundary = NULL;
  loomc_cmd_native_plan_t* current = head;
  loomc_cmd_native_plan_t* winner = NULL;
  while (true) {
    while (current != boundary) {
      if (current->key.length == loom_kernel_specialization_query_size(query)) {
        loomc_cmd_entry_acquire_locked(cache, &current->entry);
        iree_slim_mutex_unlock(&cache->mutex);
        const bool equal =
            loom_kernel_specialization_key_matches(query, current->key);
        iree_slim_mutex_lock(&cache->mutex);
        if (equal) {
          winner = current;
          break;
        }
        loomc_cmd_native_plan_t* next = current->next;
        loomc_cmd_entry_release_locked(cache, &current->entry);
        current = next;
      } else {
        current = current->next;
      }
    }
    if (boundary) {
      loomc_cmd_entry_release_locked(cache, &boundary->entry);
    }
    loomc_cmd_native_plan_t* latest = *loomc_cmd_plan_bucket(source, hash);
    if (winner || latest == head) {
      break;
    }
    boundary = head;
    head = latest;
    loomc_cmd_entry_acquire_locked(cache, &head->entry);
    current = head;
  }
  loomc_cmd_entry_release_locked(cache, &head->entry);
  return winner;
}

typedef struct loomc_cmd_plan_publication_t {
  // Cache owner, borrowed through this synchronous publication.
  loomc_cmd_native_source_cache_t* cache;
  // Common source leased before any plan is acquired.
  loomc_cmd_native_source_t* source;
} loomc_cmd_plan_publication_t;

static iree_status_t loomc_cmd_plan_prepare(
    loomc_cmd_plan_publication_t* publication, const loom_module_t* site_module,
    const loom_kernel_specialization_value_t* values,
    const loom_kernel_specialization_query_t* query,
    loomc_cmd_native_plan_t** out_plan) {
  *out_plan = NULL;
  loomc_cmd_native_source_cache_t* cache = publication->cache;
  loomc_cmd_native_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(cache->allocator, sizeof(*plan), (void**)&plan));
  plan->entry.allocator = cache->allocator;
  plan->entry.owned_bytes = sizeof(*plan);
  plan->entry.is_plan = true;
  plan->source = publication->source;
  // Small independently evictable projections do not inherit the much larger
  // source-construction block size. Larger arrays use ordinary arena oversize
  // allocations; this granularity is not a separate capacity limit.
  iree_arena_block_pool_initialize(4 * 1024,
                                   loomc_cmd_entry_allocator(&plan->entry),
                                   &plan->entry.block_pool);
  iree_arena_initialize(&plan->entry.block_pool, &plan->entry.arena);
  iree_status_t status = loom_kernel_request_preparation_bind(
      publication->source->preparation, site_module, values, &plan->entry.arena,
      &plan->classifier);
  iree_arena_block_pool_trim(&plan->entry.block_pool);
  if (iree_status_is_ok(status) && query) {
    const iree_host_size_t source_bytes =
        publication->source->entry.owned_bytes;
    const iree_host_size_t available = cache->byte_limit - source_bytes;
    if (plan->entry.owned_bytes <= available) {
      // The current arena head is already charged, so its unused aligned
      // prefix can hold a key even when no new allocation fits the budget.
      // Otherwise refuse oversized payloads before independently copying them.
      const iree_host_size_t head_capacity =
          plan->entry.arena.block_bytes_remaining &
          ~(iree_host_size_t)(iree_max_align_t - 1);
      status = loom_kernel_specialization_key_capture(
          query, iree_max(head_capacity, available - plan->entry.owned_bytes),
          &plan->entry.arena, &plan->key);
      plan->hash = loom_kernel_specialization_query_hash(query);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    loomc_cmd_entry_free(&plan->entry);
  }
  return status;
}

static iree_status_t loomc_cmd_plan_acquire(
    void* user_data, const loom_module_t* site_module,
    const loom_kernel_specialization_value_t* values, uint16_t value_count,
    iree_arena_allocator_t* scratch_arena,
    const loom_kernel_class_classifier_t** out_classifier, void** out_lease) {
  *out_classifier = NULL;
  *out_lease = NULL;
  loomc_cmd_plan_publication_t* publication = user_data;
  loomc_cmd_native_source_cache_t* cache = publication->cache;
  loom_kernel_specialization_query_t* query = NULL;
  loomc_cmd_native_plan_t* plan = NULL;
  if (publication->source->entry.retained) {
    IREE_RETURN_IF_ERROR(loom_kernel_specialization_query_allocate(
        site_module, values, value_count, scratch_arena, &query));
    iree_slim_mutex_lock(&cache->mutex);
    plan = loomc_cmd_plan_lookup_locked(cache, publication->source, query);
    iree_slim_mutex_unlock(&cache->mutex);
  }
  if (!plan) {
    IREE_RETURN_IF_ERROR(
        loomc_cmd_plan_prepare(publication, site_module, values, query, &plan));
    if (plan->key.data &&
        plan->entry.owned_bytes <=
            cache->byte_limit - publication->source->entry.owned_bytes) {
      iree_slim_mutex_lock(&cache->mutex);
      loomc_cmd_native_plan_t* winner =
          loomc_cmd_plan_lookup_locked(cache, publication->source, query);
      if (!winner) {
        loomc_cmd_cache_make_room_locked(cache, plan->entry.owned_bytes);
        winner =
            loomc_cmd_plan_lookup_locked(cache, publication->source, query);
      }
      if (!winner && plan->entry.owned_bytes <=
                         cache->byte_limit - cache->retained_bytes) {
        loomc_cmd_plan_insert(plan);
        plan->entry.retained = true;
        plan->entry.use_count = 1;
        cache->retained_bytes += plan->entry.owned_bytes;
        loomc_cmd_entry_acquire_locked(cache, &plan->source->entry);
      }
      iree_slim_mutex_unlock(&cache->mutex);
      if (winner) {
        loomc_cmd_entry_free(&plan->entry);
        plan = winner;
      }
    }
  }
  *out_classifier = &plan->classifier;
  *out_lease = plan;
  return iree_ok_status();
}

static void loomc_cmd_plan_release(void* user_data, void* lease) {
  loomc_cmd_plan_publication_t* publication = user_data;
  loomc_cmd_native_plan_t* plan = lease;
  if (!plan->entry.retained) {
    loomc_cmd_entry_free(&plan->entry);
  } else {
    iree_slim_mutex_lock(&publication->cache->mutex);
    loomc_cmd_entry_release_locked(publication->cache, &plan->entry);
    iree_slim_mutex_unlock(&publication->cache->mutex);
  }
}

iree_status_t loomc_cmd_native_source_cache_allocate(
    const loom_link_module_index_t* index, iree_host_size_t retained_byte_limit,
    iree_allocator_t allocator, loomc_cmd_native_source_cache_t** out_cache) {
  *out_cache = NULL;
  loomc_cmd_native_source_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*cache), (void**)&cache));
  cache->index = index;
  cache->allocator = allocator;
  cache->byte_limit = retained_byte_limit;
  iree_slim_mutex_initialize(&cache->mutex);
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(index);
  iree_status_t status = iree_ok_status();
  if (symbol_count != 0) {
    status = iree_allocator_malloc_array(allocator, symbol_count,
                                         sizeof(*cache->sources),
                                         (void**)&cache->sources);
  }
  if (iree_status_is_ok(status)) {
    *out_cache = cache;
  } else {
    loomc_cmd_native_source_cache_free(cache);
  }
  return status;
}

void loomc_cmd_native_source_cache_free(
    loomc_cmd_native_source_cache_t* cache) {
  if (cache == NULL) {
    return;
  }
  iree_slim_mutex_lock(&cache->mutex);
  while (cache->idle_head) {
    loomc_cmd_entry_retire_locked(cache, cache->idle_head);
  }
  iree_slim_mutex_unlock(&cache->mutex);
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->allocator, cache->sources);
  iree_allocator_free(cache->allocator, cache);
}

iree_status_t loomc_cmd_native_source_cache_publish(
    loomc_cmd_native_source_cache_t* cache,
    loomc_link_materialization_state_t* state,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes) {
  *out_site_classes = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&cache->mutex);
  loomc_cmd_native_source_t* source =
      loomc_cmd_source_lookup(cache, state, source_symbol_ordinal);
  iree_slim_mutex_unlock(&cache->mutex);
  iree_status_t status = iree_ok_status();
  if (source == NULL) {
    status = loomc_cmd_native_source_prepare(cache, state, environment,
                                             source_symbol_ordinal, &source);
    if (iree_status_is_ok(status)) {
      source = loomc_cmd_source_publish_completed(cache, state, source);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_result_copy_diagnostics(
        state->diagnostics.result, source->diagnostics));
  }
  if (iree_status_is_ok(status)) {
    loomc_cmd_plan_publication_t publication = {.cache = cache,
                                                .source = source};
    const loom_kernel_request_plan_provider_t plans = {
        .acquire = loomc_cmd_plan_acquire,
        .release = loomc_cmd_plan_release,
        .user_data = &publication,
    };
    status = loom_kernel_request_preparation_publish(
        source->preparation, site_module, sites, site_count, collection_options,
        plans, sink, scratch_arena, out_site_classes);
  }
  if (source != NULL) {
    loomc_cmd_source_release(cache, source);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
