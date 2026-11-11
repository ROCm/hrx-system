// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/target/cmd/native_source_cache.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/binding/c/src/result.h"
#include "loomc/iree.h"

typedef struct loomc_cmd_native_source_t loomc_cmd_native_source_t;

// Preserves the allocator's natural alignment after the accounting header.
typedef struct iree_alignas(iree_max_align_t) loomc_cmd_source_allocation_t {
  // Total requested bytes, including this header.
  iree_host_size_t length;
} loomc_cmd_source_allocation_t;

struct loomc_cmd_native_source_t {
  // Allocator owning this entry and its accounting allocations.
  iree_allocator_t allocator;
  // Requested live bytes, including this entry, headers, and pool capacity.
  iree_host_size_t owned_bytes;
  // Independently reclaimable backing for the immutable source and its key.
  iree_arena_block_pool_t block_pool;
  // Owned target assignment rows and copied config values.
  iree_arena_allocator_t arena;
  // Native module, candidate headers, and classifier.
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
  // Protected by the cache mutex; true only while indexed by the cache.
  bool retained;
  // Number of synchronous publishers holding this retained source.
  iree_host_size_t use_count;
  // Intrusive source bucket link, reused for privately retired entries.
  loomc_cmd_native_source_t* next;
  // Previous bucket link, allowing constant-time unlink from the index.
  loomc_cmd_native_source_t** previous_link;
  // Earlier idle entry, or NULL at the oldest end.
  loomc_cmd_native_source_t* idle_previous;
  // Later idle entry, or NULL at the newest end.
  loomc_cmd_native_source_t* idle_next;
};

struct loomc_cmd_native_source_cache_t {
  // Immutable provider universe borrowed from the owner.
  const loom_link_module_index_t* index;
  // Allocator for cache and entry storage.
  iree_allocator_t allocator;
  // Protects completed publication, leases, and byte accounting only.
  iree_slim_mutex_t mutex;
  // Maximum bytes retained by completed source entries.
  iree_host_size_t byte_limit;
  // Charged entry bytes, including detached entries until destruction finishes.
  iree_host_size_t retained_bytes;
  // Source-index buckets for completed entries.
  loomc_cmd_native_source_t** sources;
  // Oldest idle entry eligible for eviction.
  loomc_cmd_native_source_t* idle_head;
  // Most recently released idle entry.
  loomc_cmd_native_source_t* idle_tail;
};

static iree_status_t loomc_cmd_source_allocator_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  loomc_cmd_native_source_t* source = self;
  if (command == IREE_ALLOCATOR_COMMAND_TRIM) {
    return iree_allocator_trim(source->allocator);
  }
  loomc_cmd_source_allocation_t* allocation =
      *inout_ptr ? (loomc_cmd_source_allocation_t*)*inout_ptr - 1 : NULL;
  if (command == IREE_ALLOCATOR_COMMAND_FREE) {
    if (allocation != NULL) {
      source->owned_bytes -= allocation->length;
      iree_allocator_free(source->allocator, allocation);
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
  IREE_RETURN_IF_ERROR(source->allocator.ctl(source->allocator.self, command,
                                             &expanded, &storage));
  allocation = storage;
  allocation->length = length;
  source->owned_bytes = source->owned_bytes - old_length + length;
  *inout_ptr = allocation + 1;
  return iree_ok_status();
}

static iree_allocator_t loomc_cmd_source_allocator(
    loomc_cmd_native_source_t* source) {
  return (iree_allocator_t){
      .self = source,
      .ctl = loomc_cmd_source_allocator_ctl,
  };
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
  iree_arena_deinitialize(&source->arena);
  iree_arena_block_pool_deinitialize(&source->block_pool);
  iree_allocator_free(source->allocator, source);
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
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &source->arena, targets->count, sizeof(*source->targets.assignments),
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
      &source->arena, keys.count, sizeof(*source->config_values),
      (void**)&source->config_values));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < keys.count && iree_status_is_ok(status);
       ++i) {
    const iree_string_view_t value = loomc_cmd_source_config_value(
        &state->specialization.config_bindings, keys.values[i]);
    char* storage = NULL;
    status = iree_arena_allocate(&source->arena, value.size, (void**)&storage);
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
  source->allocator = cache->allocator;
  source->owned_bytes = sizeof(*source);
  source->symbol_ordinal = symbol_ordinal;
  const iree_allocator_t allocator = loomc_cmd_source_allocator(source);
  iree_arena_block_pool_initialize(environment->block_pool->total_block_size,
                                   allocator, &source->block_pool);
  iree_arena_initialize(&source->block_pool, &source->arena);
  iree_status_t status = iree_status_from_loomc(loomc_result_create(
      LOOMC_STATUS_OK, loomc_result_source_retention(state->diagnostics.result),
      loomc_allocator_from_iree(allocator), &source->diagnostics));
  // This stack view borrows normalized configuration and resolved target rows;
  // it never takes ownership or deinitializes the invocation's state. Only its
  // fresh diagnostics and source backing may survive preparation.
  loomc_link_materialization_state_t preparation_state = *state;
  preparation_state.diagnostics.result = source->diagnostics;
  preparation_state.diagnostics.source = NULL;
  preparation_state.arena = &source->arena;
  preparation_state.allocator = loomc_allocator_from_iree(allocator);
  loom_link_plan_materialization_environment_t preparation_environment =
      *environment;
  preparation_environment.block_pool = &source->block_pool;
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
    iree_arena_block_pool_trim(&source->block_pool);
    *out_source = source;
  } else {
    status = iree_status_join(
        status, iree_status_from_loomc(loomc_result_copy_diagnostics(
                    state->diagnostics.result, source->diagnostics)));
    loomc_cmd_native_source_free(source);
  }
  return status;
}

static void loomc_cmd_source_idle_remove(loomc_cmd_native_source_cache_t* cache,
                                         loomc_cmd_native_source_t* source) {
  if (source->idle_previous != NULL) {
    source->idle_previous->idle_next = source->idle_next;
  } else {
    cache->idle_head = source->idle_next;
  }
  if (source->idle_next != NULL) {
    source->idle_next->idle_previous = source->idle_previous;
  } else {
    cache->idle_tail = source->idle_previous;
  }
  source->idle_previous = NULL;
  source->idle_next = NULL;
}

static loomc_cmd_native_source_t* loomc_cmd_source_lookup(
    loomc_cmd_native_source_cache_t* cache,
    const loomc_link_materialization_state_t* state,
    iree_host_size_t symbol_ordinal) {
  for (loomc_cmd_native_source_t* source = cache->sources[symbol_ordinal];
       source != NULL; source = source->next) {
    if (loomc_cmd_native_source_matches(source, state)) {
      if (source->use_count++ == 0) {
        loomc_cmd_source_idle_remove(cache, source);
      }
      return source;
    }
  }
  return NULL;
}

static loomc_cmd_native_source_t* loomc_cmd_source_publish_completed(
    loomc_cmd_native_source_cache_t* cache,
    const loomc_link_materialization_state_t* state,
    loomc_cmd_native_source_t* candidate) {
  if (candidate->owned_bytes > cache->byte_limit) {
    return candidate;
  }
  loomc_cmd_native_source_t* retired = NULL;
  iree_host_size_t retired_bytes = 0;
  iree_slim_mutex_lock(&cache->mutex);
  loomc_cmd_native_source_t* winner =
      loomc_cmd_source_lookup(cache, state, candidate->symbol_ordinal);
  while (winner == NULL && cache->idle_head != NULL &&
         candidate->owned_bytes >
             cache->byte_limit - (cache->retained_bytes - retired_bytes)) {
    loomc_cmd_native_source_t* source = cache->idle_head;
    loomc_cmd_source_idle_remove(cache, source);
    *source->previous_link = source->next;
    if (source->next != NULL) {
      source->next->previous_link = source->previous_link;
    }
    retired_bytes += source->owned_bytes;
    source->next = retired;
    retired = source;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  while (retired != NULL) {
    loomc_cmd_native_source_t* next = retired->next;
    loomc_cmd_native_source_free(retired);
    retired = next;
  }
  iree_slim_mutex_lock(&cache->mutex);
  cache->retained_bytes -= retired_bytes;
  if (winner == NULL) {
    // Another builder may have completed while idle sources were freed.
    winner = loomc_cmd_source_lookup(cache, state, candidate->symbol_ordinal);
  }
  if (winner == NULL &&
      candidate->owned_bytes <= cache->byte_limit - cache->retained_bytes) {
    candidate->next = cache->sources[candidate->symbol_ordinal];
    candidate->previous_link = &cache->sources[candidate->symbol_ordinal];
    if (candidate->next != NULL) {
      candidate->next->previous_link = &candidate->next;
    }
    *candidate->previous_link = candidate;
    candidate->retained = true;
    candidate->use_count = 1;
    cache->retained_bytes += candidate->owned_bytes;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  if (winner != NULL) {
    loomc_cmd_native_source_free(candidate);
    return winner;
  }
  return candidate;
}

static void loomc_cmd_source_release(loomc_cmd_native_source_cache_t* cache,
                                     loomc_cmd_native_source_t* source) {
  if (!source->retained) {
    loomc_cmd_native_source_free(source);
    return;
  }
  iree_slim_mutex_lock(&cache->mutex);
  if (--source->use_count == 0) {
    source->idle_previous = cache->idle_tail;
    if (cache->idle_tail != NULL) {
      cache->idle_tail->idle_next = source;
    } else {
      cache->idle_head = source;
    }
    cache->idle_tail = source;
  }
  iree_slim_mutex_unlock(&cache->mutex);
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
  loomc_cmd_native_source_t* source = cache->idle_head;
  while (source != NULL) {
    loomc_cmd_native_source_t* next = source->idle_next;
    loomc_cmd_native_source_free(source);
    source = next;
  }
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
    status = loom_kernel_request_preparation_publish(
        source->preparation, site_module, sites, site_count, collection_options,
        sink, scratch_arena, out_site_classes);
  }
  if (source != NULL) {
    loomc_cmd_source_release(cache, source);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
