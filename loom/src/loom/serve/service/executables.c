// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/executables.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/serve/runtime/parameter/module.h"
#include "loom/serve/runtime/program/stage_set.h"
#include "loomc/iree.h"

typedef struct loom_serve_generation_t {
  // Management, preferred selection, captured batch, and bound runner owners.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this generation and its execution tables.
  iree_allocator_t host_allocator;
  // Retained immutable source snapshot and exact-input native reuse.
  loom_serve_jit_t* jit;
  // Original model evidence or owned trailing storage for a new generation.
  loom_serve_compilation_report_t* report;
  // Cold declaration count, including partial construction on failure.
  iree_host_size_t count;
  // Owned replacement stages; NULL for the model-owned original stages.
  loom_serve_jit_stage_t** compiled;
  // Borrowed original table or owned recordings paired with compiled stages.
  const loom_serve_stage_table_t* table;
  // Native payload retained by the immutable cold stages, excluding index
  // space.
  iree_host_size_t cold_native_bytes;
  // Owner-side request-cache accounting for this source generation.
  struct {
    // Request choices retaining this generation, including construction.
    iree_host_size_t count;
    // Additional native keys/images charged once across those choices.
    iree_host_size_t native_bytes;
  } choices;
  // Prepared checkpoint entries retaining native preparers in this generation.
  iree_host_size_t parameters;
} loom_serve_generation_t;

typedef struct loom_serve_checkpoint_t {
  // Intrusive idle-retention or retirement list, owned by the model.
  struct loom_serve_checkpoint_t* next;
  // Previous retained entry; NULL at the MRU head and on the retired list.
  struct loom_serve_checkpoint_t* previous;
  // Source generation retaining the preparer's JIT and compilation report.
  loom_serve_generation_t* generation;
  // Owned checkpoint identity; path arrays/bytes follow this allocation.
  loom_serve_preparation_parameter_t declaration;
  // Owned single-domain bank, independent of any compiled request product.
  loom_serve_parameter_bank_t* bank;
  // Recorded choices borrowing this bank, including the active construction.
  iree_host_size_t users;
  // Final packed root bytes, independent of physical commitment and host cost.
  uint64_t bytes;
  // Source/index/native preparation succeeded; incomplete entries cannot hit.
  bool ready;
} loom_serve_checkpoint_t;

typedef struct loom_serve_choice_job_t loom_serve_choice_job_t;

typedef struct loom_serve_choice_code_t {
  // Serialized choice owners; compilation completes before a second borrower.
  iree_host_size_t users;
  // Initialized products, including partial construction on failure.
  iree_host_size_t count;
  // Immutable compiler products shared independently of checkpoint bindings.
  loom_serve_jit_stage_t** compiled;
  // Requested product and array storage, excluding shared native images.
  iree_host_size_t storage_size;
  // Owner collection has charged this bundle once, not once per choice.
  bool accounted;
} loom_serve_choice_code_t;

typedef struct loom_serve_choice_t {
  // Borrowed serialized owner, alive through producer and cache retirement.
  loom_serve_executables_t* owner;
  // Retained source snapshot and cold descriptors preceding request stages.
  loom_serve_generation_t* generation;
  // Owned exact request key, borrowed by the compiler until completion.
  loom_serve_preparation_t* preparation;
  // Owned indexed sources, retaining the model's immutable cold prefix.
  loom_serve_parameter_sources_t* parameter_sources;
  // Pool entry receiving the terminal producer result.
  loom_serve_variant_entry_t* entry;
  // Shared immutable code, independent of this choice's checkpoint selection.
  loom_serve_choice_code_t* code;
  // Owned recordings with a borrowed cold prefix from the retained generation.
  loom_serve_stage_table_t* table;
  // Owned admission for cold roots plus request-selected checkpoint banks.
  // NULL borrows the model's original residency.
  loom_serve_residency_t* residency;
  // Number of borrowed cache entries in the trailing pointer array.
  iree_host_size_t checkpoint_count;
  // Requested owned payload bytes, finalized before terminal publication.
  iree_host_size_t storage_size;
  // Owner collection has added storage_size to the retained payload total.
  bool accounted;
  // Checkpoint owners outliving this choice's fixed recordings and residency.
  loom_serve_checkpoint_t* checkpoints[];
} loom_serve_choice_t;

struct loom_serve_executables_t {
  // Allocator owning this serialized model's code policy.
  iree_allocator_t host_allocator;
  // Immutable borrowed model resources, live through all compiler callbacks.
  loom_serve_executable_options_t options;
  // Owned original policy/checkpoint namespace, not replacement providers.
  loom_serve_sources_t* sources;
  // Original source snapshot and descriptors, available for restoration.
  loom_serve_generation_t* original;
  // Retained source generation selected for subsequent batches.
  loom_serve_generation_t* selected;
  // Retained source generation captured by the current batch.
  loom_serve_generation_t* captured;
  // Retained owner of the runner's current copied execution table.
  loom_serve_generation_t* bound;
  // Sole invocation's request declaration and compilation ownership.
  struct {
    // Unresolved declarations transferred from the source capability.
    loom_serve_preparation_t* declarations;
    // Accepted producer, or retained ready request choice after collection.
    loom_serve_variant_entry_t* entry;
    // Accepted producer still owns model/compiler borrows.
    bool pending;
    // Intermediate producer handoffs; NULL after terminal recording is sent.
    loom_serve_choice_job_t* job;
  } invocation;
  // Lazy bounded reuse; cold-only models allocate no pool or entry storage.
  struct {
    // Existing variant publication and consumer-borrow mechanism.
    loom_serve_variant_pool_t* pool;
    // Ready provenance anchor, with no additional executable ownership.
    loom_serve_variant_entry_t* original;
    // Runner's independent consumer borrow, retained between invocations.
    loom_serve_variant_entry_t* bound;
    // Nonblocking scheduling callback installed before any producer starts.
    loom_serve_wake_callback_t observer;
    // Fixed pool and provenance metadata, excluded from idle payload limits.
    iree_host_size_t fixed_bytes;
    // Sum of collected, independently owned request payloads.
    iree_host_size_t payload_bytes;
    // Shared native keys/images, charged once per source generation.
    iree_host_size_t native_bytes;
  } cache;
  // Prepared checkpoint retention is independent of compiled-choice retention.
  struct {
    // Most recently selected entry first; only ready entries are reusable.
    loom_serve_checkpoint_t* entries;
    // Unborrowed banks awaiting asynchronous consumer/native retirement.
    loom_serve_checkpoint_t* retired;
    // Cumulative activity of destroyed entries, independent of cache retention.
    loom_serve_weight_activity_t activity;
    // Cumulative physical returns from destroyed banks.
    uint64_t released_bytes;
    // Physical retention policy inherited by newly prepared cache entries.
    bool pinned;
  } parameters;
  // Terminal owner retirement has released runner and compiled consumers.
  bool retiring;
};

static iree_host_size_t loom_serve_native_bytes(loom_serve_jit_t* jit) {
  const loom_serve_jit_cache_storage_t storage =
      loom_serve_jit_cache_storage(jit);
  return storage.host_bytes - storage.index_bytes + storage.code_bytes;
}

static void loom_serve_choice_free(void* value);
static void loom_serve_executables_trim(loom_serve_executables_t* executables);
static void loom_serve_checkpoints_trim(loom_serve_executables_t* executables);

static void loom_serve_checkpoint_touch(loom_serve_executables_t* executables,
                                        loom_serve_checkpoint_t* checkpoint) {
  if (!checkpoint->previous) {
    return;
  }
  checkpoint->previous->next = checkpoint->next;
  if (checkpoint->next) {
    checkpoint->next->previous = checkpoint->previous;
  }
  checkpoint->previous = NULL;
  checkpoint->next = executables->parameters.entries;
  checkpoint->next->previous = checkpoint;
  executables->parameters.entries = checkpoint;
}

static void loom_serve_weight_activity_add(loom_serve_weight_activity_t* total,
                                           loom_serve_weight_activity_t value) {
  total->activations += value.activations;
  total->deactivations += value.deactivations;
  total->read_bytes += value.read_bytes;
  total->prepared_bytes += value.prepared_bytes;
  total->groups += value.groups;
}

static void loom_serve_generation_retain(loom_serve_generation_t* generation) {
  if (generation) {
    iree_atomic_ref_count_inc(&generation->ref_count);
  }
}

static void loom_serve_generation_retire(loom_serve_generation_t* generation) {
  if (generation && generation->compiled) {
    // Recordings borrow parameter roots. Compiled products and native preparers
    // remain alive until the banks have joined their consumers and retired.
    loom_serve_stage_table_free((loom_serve_stage_table_t*)generation->table);
    generation->table = NULL;
  }
}

static void loom_serve_generation_release(void* value) {
  loom_serve_generation_t* generation = value;
  if (!generation) {
    return;
  }
  if (iree_atomic_ref_count_dec(&generation->ref_count) != 1) {
    return;
  }
  loom_serve_generation_retire(generation);
  if (generation->compiled) {
    for (iree_host_size_t i = 0; i < generation->count; ++i) {
      loom_serve_jit_stage_free(generation->compiled[i]);
    }
  }
  loom_serve_jit_release(generation->jit);
  iree_allocator_free(generation->host_allocator, generation->compiled);
  iree_allocator_free(generation->host_allocator, generation);
}

static iree_status_t loom_serve_generation_create(
    loom_serve_jit_t* original, iree_allocator_t host_allocator,
    loom_serve_generation_t** out_generation) {
  *out_generation = NULL;
  loom_serve_generation_t* generation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      iree_sizeof_struct(*generation) +
          (original ? 0 : sizeof(loom_serve_compilation_report_t)),
      (void**)&generation));
  iree_atomic_ref_count_init(&generation->ref_count);
  generation->host_allocator = host_allocator;
  generation->jit = original;
  loom_serve_jit_retain(original);
  generation->report =
      original
          ? loom_serve_jit_report(original)
          : (loom_serve_compilation_report_t*)((uint8_t*)generation +
                                               iree_sizeof_struct(*generation));
  *out_generation = generation;
  return iree_ok_status();
}

iree_status_t loom_serve_executables_allocate(
    const loom_serve_executable_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_executables_t** out_executables) {
  *out_executables = NULL;
  if (options->cache.capacity > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "request cache capacity is not representable");
  }
  loom_serve_executables_t* executables = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executables), (void**)&executables));
  executables->host_allocator = host_allocator;
  executables->options = *options;
  iree_status_t status = loom_serve_sources_clone(
      options->sources, host_allocator, &executables->sources);
  if (iree_status_is_ok(status)) {
    status = loom_serve_generation_create(options->jit, host_allocator,
                                          &executables->original);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_generation_t* original = executables->original;
    original->count = loom_serve_preparation_stage_count(options->preparation);
    original->table = options->stages;
    original->cold_native_bytes = loom_serve_native_bytes(original->jit);
    executables->selected = original;
    loom_serve_generation_retain(original);
    executables->bound = original;
    loom_serve_generation_retain(original);
  }
  if (iree_status_is_ok(status)) {
    *out_executables = executables;
  } else {
    status = iree_status_join(status, loom_serve_executables_free(executables));
  }
  return status;
}

void loom_serve_executables_retire(loom_serve_executables_t* executables) {
  if (!executables || executables->retiring) {
    return;
  }
  executables->retiring = true;
  loom_serve_preparation_free(executables->invocation.declarations);
  loom_serve_module_clear_stages(executables->options.runner);
  loom_serve_parameters_module_bind_sources(
      executables->options.parameter_module, NULL);
  loom_serve_variant_entry_release(executables->invocation.entry);
  loom_serve_variant_entry_release(executables->cache.bound);
  loom_serve_variant_pool_free(executables->cache.pool);
  executables->cache.pool = NULL;
  executables->invocation.entry = NULL;
  executables->invocation.declarations = NULL;
  executables->cache.bound = NULL;
  loom_serve_generation_retire(executables->original);
  loom_serve_generation_retire(executables->selected);
  loom_serve_generation_retire(executables->captured);
  loom_serve_generation_retire(executables->bound);
  for (loom_serve_checkpoint_t* entry = executables->parameters.entries; entry;
       entry = entry->next) {
    loom_serve_generation_retire(entry->generation);
  }
  for (loom_serve_checkpoint_t* entry = executables->parameters.retired; entry;
       entry = entry->next) {
    loom_serve_generation_retire(entry->generation);
  }
  loom_serve_checkpoints_trim(executables);
}

bool loom_serve_executables_is_retired(loom_serve_executables_t* executables) {
  bool retired = true;
  for (loom_serve_checkpoint_t* entry =
           executables ? executables->parameters.retired : NULL;
       entry; entry = entry->next) {
    retired &= loom_serve_parameter_bank_is_retired(entry->bank);
  }
  return retired;
}

iree_status_t loom_serve_executables_await_retirement(
    loom_serve_executables_t* executables) {
  iree_status_t status = iree_ok_status();
  for (loom_serve_checkpoint_t* entry =
           executables ? executables->parameters.retired : NULL;
       entry; entry = entry->next) {
    status = iree_status_join(
        status, loom_serve_parameter_bank_await_retirement(entry->bank));
  }
  return status;
}

iree_status_t loom_serve_executables_free(
    loom_serve_executables_t* executables) {
  if (!executables) {
    return iree_ok_status();
  }
  loom_serve_executables_retire(executables);
  IREE_RETURN_IF_ERROR(loom_serve_executables_query_status(executables));
  IREE_ASSERT(!executables->parameters.retired);
  loom_serve_generation_release(executables->captured);
  loom_serve_generation_release(executables->bound);
  loom_serve_generation_release(executables->selected);
  loom_serve_generation_release(executables->original);
  loom_serve_sources_free(executables->sources, executables->host_allocator);
  iree_allocator_free(executables->host_allocator, executables);
  return iree_ok_status();
}

void loom_serve_executables_capture(loom_serve_executables_t* executables) {
  executables->captured = executables->selected;
  loom_serve_generation_retain(executables->captured);
}

void loom_serve_executables_declare(loom_serve_executables_t* executables,
                                    loom_serve_preparation_t* preparation) {
  executables->invocation.declarations = preparation;
}

void loom_serve_executables_set_observer(loom_serve_executables_t* executables,
                                         loom_serve_wake_callback_t observer) {
  executables->cache.observer = observer;
  if (executables->cache.pool) {
    loom_serve_variant_pool_set_observer(executables->cache.pool, observer);
  }
  for (loom_serve_checkpoint_t* entry = executables->parameters.retired; entry;
       entry = entry->next) {
    loom_serve_parameter_bank_retire(entry->bank, observer.wake,
                                     observer.user_data);
  }
}

static iree_status_t loom_serve_executables_bind(
    loom_serve_executables_t* executables) {
  loom_serve_generation_t* generation = executables->captured;
  loom_serve_variant_entry_t* entry = executables->invocation.entry;
  if (executables->bound == generation && executables->cache.bound == entry) {
    return iree_ok_status();
  }
  const loom_serve_choice_t* choice =
      entry ? loom_serve_variant_entry_value(entry) : NULL;
  const loom_serve_stage_table_t* table =
      choice ? choice->table : generation->table;
  IREE_RETURN_IF_ERROR(loom_serve_module_bind_stages(
      executables->options.runner, table->part_count, table->parts));
  loom_serve_parameters_module_bind_sources(
      executables->options.parameter_module,
      choice ? choice->parameter_sources
             : executables->options.parameter_sources);
  loom_serve_variant_entry_retain(entry);
  loom_serve_variant_entry_release(executables->cache.bound);
  executables->cache.bound = entry;
  loom_serve_generation_retain(generation);
  loom_serve_generation_release(executables->bound);
  executables->bound = generation;
  return iree_ok_status();
}

void loom_serve_executables_release(loom_serve_executables_t* executables) {
  loom_serve_residency_cancel_admission(
      loom_serve_executables_residency(executables));
  loom_serve_preparation_free(executables->invocation.declarations);
  loom_serve_variant_entry_release(executables->invocation.entry);
  memset(&executables->invocation, 0, sizeof(executables->invocation));
  loom_serve_generation_release(executables->captured);
  executables->captured = NULL;
  loom_serve_executables_trim(executables);
  loom_serve_checkpoints_trim(executables);
}

typedef struct loom_serve_generation_job_t {
  // Finite source-indexing task, retired before asynchronous stage compilation.
  loomc_task_t task;
  // Borrowed immutable model resources through terminal completion.
  loom_serve_executables_t* executables;
  // Owned generation transferred at terminal completion, including failure.
  loom_serve_generation_t* generation;
  // Provider namespace borrowed through source indexing only.
  loom_serve_sources_t sources;
  // Indexing outcome consumed after the task relinquishes compiler access.
  iree_status_t status;
  // Terminal ownership transfer after all model and source accesses retire.
  loom_serve_variant_completion_fn_t completion;
  // Borrowed terminal consumer state.
  void* user_data;
} loom_serve_generation_job_t;

static iree_status_t loom_serve_executables_record(
    const loom_serve_executable_options_t* options,
    const loom_serve_stage_table_t* prefix, iree_host_size_t count,
    loom_serve_jit_stage_t* const* stages, iree_allocator_t host_allocator,
    loom_serve_stage_table_t** out_table) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = loom_serve_stage_record_check(&options->recording, stages[i]);
    const loom_cmd_program_t* program = loom_serve_jit_stage_program(stages[i]);
    if (iree_status_is_ok(status) && program) {
      status =
          loom_serve_parameter_bank_check_layout(options->parameters, program);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_stage_table_allocate(&options->recording, prefix, count,
                                             stages, host_allocator, out_table);
  }
  return status;
}

static void loom_serve_account_natives(loom_serve_executables_t* executables,
                                       loom_serve_generation_t* generation) {
  const iree_host_size_t bytes =
      generation->choices.count || generation->parameters
          ? loom_serve_native_bytes(generation->jit) -
                generation->cold_native_bytes
          : 0;
  executables->cache.native_bytes -= generation->choices.native_bytes;
  executables->cache.native_bytes += bytes;
  generation->choices.native_bytes = bytes;
}

static void loom_serve_choice_free(void* value) {
  loom_serve_choice_t* choice = value;
  if (!choice) {
    return;
  }
  loom_serve_executables_t* executables = choice->owner;
  loom_serve_generation_t* generation = choice->generation;
  loom_serve_stage_table_free(choice->table);
  loom_serve_residency_free(choice->residency);
  for (iree_host_size_t i = 0; i < choice->checkpoint_count; ++i) {
    --choice->checkpoints[i]->users;
  }
  loom_serve_choice_code_t* code = choice->code;
  if (code && --code->users == 0) {
    for (iree_host_size_t i = 0; i < code->count; ++i) {
      loom_serve_jit_stage_free(code->compiled[i]);
    }
    if (code->accounted) {
      executables->cache.payload_bytes -= code->storage_size;
    }
    iree_allocator_free(executables->host_allocator, code->compiled);
    iree_allocator_free(executables->host_allocator, code);
  }
  loom_serve_preparation_free(choice->preparation);
  loom_serve_parameter_sources_release(choice->parameter_sources);
  if (choice->accounted) {
    executables->cache.payload_bytes -= choice->storage_size;
  }
  --generation->choices.count;
  loom_serve_jit_trim(generation->jit);
  loom_serve_account_natives(executables, generation);
  loom_serve_generation_release(generation);
  iree_allocator_free(executables->host_allocator, choice);
}

static void loom_serve_choice_discard(loom_serve_executables_t* executables,
                                      loom_serve_variant_entry_t* entry) {
  if (entry == executables->cache.bound) {
    loom_serve_module_clear_stages(executables->options.runner);
    loom_serve_parameters_module_bind_sources(
        executables->options.parameter_module, NULL);
    loom_serve_variant_entry_release(executables->cache.bound);
    executables->cache.bound = NULL;
    loom_serve_generation_release(executables->bound);
    executables->bound = NULL;
  }
  loom_serve_variant_entry_discard(entry);
}

static void loom_serve_executables_trim(loom_serve_executables_t* executables) {
  loom_serve_variant_pool_t* pool = executables->cache.pool;
  if (!pool) {
    return;
  }
  while (loom_serve_variant_pool_count(pool) > 1 &&
         (loom_serve_variant_pool_count(pool) - 1 >
              executables->options.cache.capacity ||
          executables->cache.payload_bytes + executables->cache.native_bytes +
                  loom_serve_variant_pool_storage_size(pool) -
                  executables->cache.fixed_bytes >
              executables->options.cache.max_bytes)) {
    loom_serve_variant_entry_t* entry =
        loom_serve_variant_entry_next(executables->cache.original);
    loom_serve_choice_discard(executables, entry);
  }
}

static void loom_serve_checkpoints_trim(loom_serve_executables_t* executables) {
  loom_serve_checkpoint_t** cursor = &executables->parameters.entries;
  iree_host_size_t count = 0;
  uint64_t bytes = 0;
  while (*cursor) {
    loom_serve_checkpoint_t* checkpoint = *cursor;
    const bool retain =
        !executables->retiring && checkpoint->ready &&
        count < executables->options.parameter_cache.capacity &&
        checkpoint->bytes <=
            executables->options.parameter_cache.max_bytes - bytes;
    if (retain) {
      ++count;
      bytes += checkpoint->bytes;
      cursor = &checkpoint->next;
      continue;
    }
    // Logical parameter eviction invalidates dependent fixed recordings, not
    // shared code or other entries. This runs only between invocations.
    for (loom_serve_variant_entry_t* entry =
             executables->cache.pool
                 ? loom_serve_variant_entry_next(executables->cache.original)
                 : NULL;
         entry;) {
      loom_serve_variant_entry_t* next = loom_serve_variant_entry_next(entry);
      const loom_serve_choice_t* choice = loom_serve_variant_entry_value(entry);
      for (iree_host_size_t i = 0; i < choice->checkpoint_count; ++i) {
        if (choice->checkpoints[i] == checkpoint) {
          loom_serve_choice_discard(executables, entry);
          break;
        }
      }
      entry = next;
    }
    IREE_ASSERT(!checkpoint->users);
    *cursor = checkpoint->next;
    if (checkpoint->next) {
      checkpoint->next->previous = checkpoint->previous;
    }
    checkpoint->previous = NULL;
    checkpoint->next = executables->parameters.retired;
    executables->parameters.retired = checkpoint;
    const loom_serve_wake_callback_t observer = executables->cache.observer;
    loom_serve_parameter_bank_retire(checkpoint->bank, observer.wake,
                                     observer.user_data);
    if (observer.wake) {
      observer.wake(observer.user_data);
    }
  }
}

iree_status_t loom_serve_executables_query_status(
    loom_serve_executables_t* executables) {
  if (!executables || executables->invocation.pending) {
    return iree_ok_status();
  }
  loom_serve_checkpoint_t** cursor = &executables->parameters.retired;
  iree_status_t status = iree_ok_status();
  while (*cursor && iree_status_is_ok(status)) {
    loom_serve_checkpoint_t* checkpoint = *cursor;
    if (!loom_serve_parameter_bank_is_retired(checkpoint->bank)) {
      cursor = &checkpoint->next;
      continue;
    }
    loom_serve_weights_t* weights =
        checkpoint->bank
            ? loom_serve_parameter_bank_weights(checkpoint->bank, 0)
            : NULL;
    const loom_serve_weight_activity_t activity =
        weights ? loom_serve_weights_activity(weights)
                : (loom_serve_weight_activity_t){0};
    const loom_serve_memory_statistics_t memory =
        weights ? loom_serve_weights_statistics(weights)
                : (loom_serve_memory_statistics_t){0};
    status = loom_serve_parameter_bank_free(checkpoint->bank);
    if (iree_status_is_ok(status)) {
      loom_serve_weight_activity_add(&executables->parameters.activity,
                                     activity);
      executables->parameters.released_bytes +=
          memory.released_bytes + memory.committed_bytes;
      *cursor = checkpoint->next;
      loom_serve_generation_t* generation = checkpoint->generation;
      --generation->parameters;
      loom_serve_jit_trim(generation->jit);
      loom_serve_account_natives(executables, generation);
      loom_serve_generation_release(generation);
      iree_allocator_free(executables->host_allocator, checkpoint);
    }
  }
  return status;
}

bool loom_serve_executables_is_pending(
    const loom_serve_executables_t* executables) {
  for (const loom_serve_checkpoint_t* entry =
           executables ? executables->parameters.retired : NULL;
       entry; entry = entry->next) {
    if (!loom_serve_parameter_bank_is_retired(entry->bank)) {
      return true;
    }
  }
  return false;
}

loom_serve_residency_t* loom_serve_executables_residency(
    const loom_serve_executables_t* executables) {
  const loom_serve_choice_t* choice =
      executables->invocation.entry && !executables->invocation.pending
          ? loom_serve_variant_entry_value(executables->invocation.entry)
          : NULL;
  return choice && choice->residency ? choice->residency
                                     : executables->options.residency;
}

loom_serve_residency_snapshot_t loom_serve_executables_parameters(
    const loom_serve_executables_t* executables) {
  loom_serve_residency_snapshot_t result = loom_serve_residency_snapshot(
      loom_serve_executables_residency(executables));
  result.memory =
      loom_serve_residency_statistics(executables->options.residency);
  result.activity =
      loom_serve_residency_activity(executables->options.residency);
  loom_serve_weight_activity_add(&result.activity,
                                 executables->parameters.activity);
  result.memory.released_bytes += executables->parameters.released_bytes;
  const loom_serve_checkpoint_t* lists[] = {executables->parameters.entries,
                                            executables->parameters.retired};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lists); ++i) {
    for (const loom_serve_checkpoint_t* entry = lists[i]; entry;
         entry = entry->next) {
      loom_serve_weights_t* weights =
          entry->bank ? loom_serve_parameter_bank_weights(entry->bank, 0)
                      : NULL;
      if (!weights) {
        continue;
      }
      const loom_serve_memory_statistics_t memory =
          loom_serve_weights_statistics(weights);
      result.memory.reserved_bytes += memory.reserved_bytes;
      result.memory.committed_bytes += memory.committed_bytes;
      result.memory.pending_bytes += memory.pending_bytes;
      result.memory.releasing_bytes += memory.releasing_bytes;
      result.memory.released_bytes += memory.released_bytes;
      loom_serve_weight_activity_add(&result.activity,
                                     loom_serve_weights_activity(weights));
    }
  }
  return result;
}

void loom_serve_executables_set_pinned(loom_serve_executables_t* executables,
                                       bool pinned) {
  executables->parameters.pinned = pinned;
  loom_serve_residency_set_pinned(executables->options.residency, pinned);
  for (loom_serve_checkpoint_t* entry = executables->parameters.entries; entry;
       entry = entry->next) {
    loom_serve_residency_group_t* group =
        loom_serve_parameter_bank_group(entry->bank);
    if (group) {
      loom_serve_residency_group_set_pinned(group, pinned);
    }
  }
}

iree_status_t loom_serve_executables_deactivate(
    loom_serve_executables_t* executables, bool* out_complete) {
  *out_complete = false;
  bool pinned = loom_serve_residency_is_pinned(executables->options.residency);
  for (loom_serve_checkpoint_t* entry = executables->parameters.entries; entry;
       entry = entry->next) {
    const loom_serve_residency_group_t* group =
        loom_serve_parameter_bank_group(entry->bank);
    pinned |= group && loom_serve_residency_group_is_pinned(group);
  }
  if (pinned) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "deactivation requires unpinned parameters");
  }
  bool complete = false;
  IREE_RETURN_IF_ERROR(loom_serve_residency_deactivate(
      executables->options.residency, &complete));
  iree_status_t status = iree_ok_status();
  for (loom_serve_checkpoint_t* entry = executables->parameters.entries;
       entry && iree_status_is_ok(status); entry = entry->next) {
    if (entry->ready) {
      bool released = false;
      status = loom_serve_residency_group_deactivate(
          loom_serve_parameter_bank_group(entry->bank), &released);
      complete &= released;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_executables_query_status(executables);
  }
  *out_complete =
      complete && !executables->parameters.retired && iree_status_is_ok(status);
  return status;
}

static iree_status_t loom_serve_checkpoint_allocate(
    loom_serve_choice_t* choice,
    const loom_serve_preparation_parameter_t* declaration,
    const loom_cmd_program_t* program, uint32_t domain,
    loom_serve_checkpoint_t** out_checkpoint) {
  *out_checkpoint = NULL;
  iree_host_size_t size = 0;
  iree_host_size_t shard_count = 0;
  bool fits = iree_host_size_checked_mul_add(
      sizeof(loom_serve_checkpoint_t), declaration->source_count,
      sizeof(*declaration->sources), &size);
  fits &= iree_host_size_checked_add(size, declaration->policy.size, &size);
  for (uint32_t i = 0; i < declaration->source_count && fits; ++i) {
    const loom_model_file_list_t source = declaration->sources[i];
    fits = iree_host_size_checked_add(shard_count, source.count, &shard_count);
    for (iree_host_size_t j = 0; j < source.count && fits; ++j) {
      fits =
          iree_host_size_checked_add(size, source.values[j].path.size, &size) &&
          iree_host_size_checked_add(size, source.values[j].format.size, &size);
    }
  }
  fits &= iree_host_size_checked_mul_add(size, shard_count,
                                         sizeof(loom_model_file_t), &size);
  fits &= iree_host_size_checked_mul_add(size, declaration->source_count,
                                         sizeof(uint32_t), &size);
  if (!fits) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "checkpoint identity storage is not representable");
  }
  loom_serve_executables_t* executables = choice->owner;
  loom_serve_checkpoint_t* checkpoint = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executables->host_allocator, size,
                                             (void**)&checkpoint));
  loom_model_file_list_t* sources = (loom_model_file_list_t*)(checkpoint + 1);
  loom_model_file_t* shards =
      (loom_model_file_t*)(sources + declaration->source_count);
  uint32_t* ordinals = (uint32_t*)(shards + shard_count);
  char* bytes = (char*)(ordinals + declaration->source_count);
  checkpoint->declaration = *declaration;
  checkpoint->declaration.sources = sources;
  checkpoint->declaration.source_ordinals = ordinals;
  memcpy(ordinals, declaration->source_ordinals,
         declaration->source_count * sizeof(*ordinals));
  checkpoint->declaration.policy =
      iree_make_string_view(bytes, declaration->policy.size);
  memcpy(bytes, declaration->policy.data, declaration->policy.size);
  bytes += declaration->policy.size;
  for (uint32_t i = 0; i < declaration->source_count; ++i) {
    sources[i] =
        (loom_model_file_list_t){declaration->sources[i].count, shards};
    for (iree_host_size_t j = 0; j < sources[i].count; ++j) {
      const loom_model_file_t file = declaration->sources[i].values[j];
      const iree_string_view_t path = file.path;
      shards->path = iree_make_string_view(bytes, path.size);
      memcpy(bytes, path.data, path.size);
      bytes += path.size;
      shards->format = iree_make_string_view(bytes, file.format.size);
      if (file.format.size) {
        memcpy(bytes, file.format.data, file.format.size);
      }
      bytes += file.format.size;
      ++shards;
    }
  }
  checkpoint->generation = choice->generation;
  loom_serve_generation_retain(checkpoint->generation);
  ++checkpoint->generation->parameters;
  checkpoint->bytes =
      loom_cmd_program_parameter_root_at(program, domain).required_byte_length;
  checkpoint->next = executables->parameters.entries;
  if (checkpoint->next) {
    checkpoint->next->previous = checkpoint;
  }
  executables->parameters.entries = checkpoint;
  *out_checkpoint = checkpoint;
  iree_status_t status = loom_serve_parameter_bank_allocate_domain(
      executables->options.device, program, domain, executables->host_allocator,
      &checkpoint->bank);
  if (iree_status_is_ok(status)) {
    loom_serve_residency_group_set_pinned(
        loom_serve_parameter_bank_group(checkpoint->bank),
        executables->parameters.pinned);
  }
  return status;
}

struct loom_serve_choice_job_t {
  // Cold source-indexing task, reused for final recording after compilation.
  loomc_task_t task;
  // Unpublished choice, retained through terminal publication.
  loom_serve_choice_t* choice;
  // Joins each producer's last access before owner advancement/destruction.
  iree_slim_mutex_t mutex;
  // Accepted compiler or preparer still owns its exclusive borrows.
  bool pending;
  // A declaration needs owner-side bank resolution after compilation.
  bool needs_parameters;
  // No existing request-code product matches the captured declarations.
  bool needs_compilation;
  // Accepted compilation also completed recording using retained cold banks.
  bool recorded;
  // Owned producer outcome, consumed on the next serialized owner turn.
  iree_status_t status;
  // First domain not yet resolved; all preceding bindings are initialized.
  iree_host_size_t index;
  // Newly allocated entry whose preparation is being joined, or NULL.
  loom_serve_checkpoint_t* preparing;
  // Selected roots followed by buffer pointers and distinct group pointers.
  loom_serve_parameter_binding_t bindings[];
};

static void loom_serve_choice_job_free(loom_serve_choice_job_t* job) {
  iree_slim_mutex_deinitialize(&job->mutex);
  iree_allocator_free(job->choice->owner->host_allocator, job);
}

static void loom_serve_choice_prepared(void* user_data, iree_status_t status) {
  loom_serve_choice_job_t* job = user_data;
  iree_slim_mutex_lock(&job->mutex);
  job->status = status;
  job->pending = false;
  const loom_serve_wake_callback_t observer =
      job->choice->owner->cache.observer;
  if (observer.wake) {
    observer.wake(observer.user_data);
  }
  iree_slim_mutex_unlock(&job->mutex);
}

// Identity is resolved before placement: equal names/shapes in independent
// checkpoints never establish interchangeable contents. Only the serialized
// owner allocates banks or changes residency registrations.
static iree_status_t loom_serve_choice_bind_parameter(
    loom_serve_choice_job_t* job, uint32_t index,
    loom_serve_checkpoint_t** out_preparing) {
  *out_preparing = NULL;
  loom_serve_choice_t* choice = job->choice;
  const loom_serve_preparation_stage_t* selected =
      loom_serve_preparation_stage(choice->preparation, 0);
  loom_serve_executables_t* executables = choice->owner;
  loom_serve_checkpoint_t** cursor = &executables->parameters.entries;
  while (*cursor) {
    loom_serve_checkpoint_t* checkpoint = *cursor;
    if (checkpoint->ready && checkpoint->generation == choice->generation &&
        loom_serve_preparation_parameter_equal(&selected->parameters[index],
                                               &checkpoint->declaration) &&
        loom_serve_parameter_bank_matches_domain(
            checkpoint->bank,
            loom_serve_jit_stage_program(choice->code->compiled[0]), index)) {
      loom_serve_checkpoint_touch(executables, checkpoint);
      ++checkpoint->users;
      choice->checkpoints[choice->checkpoint_count++] = checkpoint;
      job->bindings[index] =
          (loom_serve_parameter_binding_t){checkpoint->bank, 0};
      return iree_ok_status();
    }
    cursor = &checkpoint->next;
  }
  loom_serve_checkpoint_t* checkpoint = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_checkpoint_allocate(
      choice, &selected->parameters[index],
      loom_serve_jit_stage_program(choice->code->compiled[0]), index,
      &checkpoint));
  ++checkpoint->users;
  choice->checkpoints[choice->checkpoint_count++] = checkpoint;
  job->bindings[index] = (loom_serve_parameter_binding_t){checkpoint->bank, 0};
  *out_preparing = checkpoint;
  return iree_ok_status();
}

static iree_status_t loom_serve_choice_check(loom_serve_choice_t* choice) {
  loom_serve_executables_t* executables = choice->owner;
  const loom_serve_choice_code_t* code = choice->code;
  if (!code->count) {
    return iree_ok_status();
  }
  const loom_serve_preparation_stage_t* canonical =
      loom_serve_preparation_stage(choice->preparation, 0);
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(code->compiled[0]);
  const iree_host_size_t count = canonical->parameter_count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < code->count && iree_status_is_ok(status);
       ++i) {
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(choice->preparation, i);
    const loom_cmd_program_t* candidate =
        loom_serve_jit_stage_program(code->compiled[i]);
    if (count) {
      status = loom_serve_stage_record_check(&executables->options.recording,
                                             code->compiled[i]);
    }
    if (iree_status_is_ok(status) && candidate) {
      status = loom_serve_preparation_stage_check_workspace(
          declaration, candidate->requirements.transient.required_byte_length);
    }
    if (iree_status_is_ok(status) && count) {
      status = loom_serve_preparation_stage_check_parameters(
          declaration, candidate, canonical, program);
    } else if (iree_status_is_ok(status) && declaration->parameter_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "request stages must share the first stage's "
                                "checkpoint declarations");
    }
  }
  return status;
}

// A source-only request preserves the cold stage table and resident bindings.
// Otherwise all request stages share the first stage's checkpoint map.
static iree_host_size_t loom_serve_choice_parameter_count(
    const loom_serve_preparation_t* preparation) {
  return loom_serve_preparation_stage_count(preparation)
             ? loom_serve_preparation_stage(preparation, 0)->parameter_count
             : 0;
}

static iree_status_t loom_serve_choice_record(loom_serve_choice_job_t* job) {
  loom_serve_choice_t* choice = job->choice;
  loom_serve_executables_t* executables = choice->owner;
  const loom_serve_choice_code_t* code = choice->code;
  const iree_host_size_t count =
      loom_serve_choice_parameter_count(choice->preparation);
  if (!count) {
    return loom_serve_executables_record(
        &executables->options, choice->generation->table, code->count,
        code->compiled, executables->host_allocator, &choice->table);
  }

  iree_hal_buffer_t** buffers = (iree_hal_buffer_t**)(job->bindings + count);
  for (iree_host_size_t i = 0; i < count; ++i) {
    buffers[i] = loom_serve_parameter_bank_buffers(
        job->bindings[i].bank)[job->bindings[i].domain];
  }
  iree_status_t status = loom_serve_parameter_bindings_check_layout(
      count, job->bindings, loom_serve_jit_stage_program(code->compiled[0]),
      executables->host_allocator);
  if (iree_status_is_ok(status)) {
    loom_serve_stage_record_options_t recording =
        executables->options.recording;
    recording.parameters = buffers;
    status = loom_serve_stage_table_allocate(
        &recording, choice->generation->table, code->count, code->compiled,
        executables->host_allocator, &choice->table);
  }
  return status;
}

static void loom_serve_choice_complete(loom_serve_choice_t* choice,
                                       iree_status_t status) {
  if (iree_status_is_ok(status)) {
    choice->storage_size =
        sizeof(*choice) +
        loom_serve_choice_parameter_count(choice->preparation) *
            sizeof(*choice->checkpoints) +
        loom_serve_preparation_storage_size(choice->preparation) +
        loom_serve_stage_table_storage_size(choice->table);
  }
  loom_serve_variant_entry_complete(choice->entry, status, choice,
                                    choice->generation->report);
}

static void loom_serve_choice_compiled(void* user_data, iree_status_t status,
                                       loom_serve_stage_set_t* stages) {
  loom_serve_choice_job_t* job = user_data;
  loom_serve_choice_t* choice = job->choice;
  loom_serve_choice_code_t* code = choice->code;
  code->count = stages->count;
  code->compiled = stages->values;
  stages->count = 0;
  stages->values = NULL;
  loom_serve_stage_set_free(stages);
  if (iree_status_is_ok(status)) {
    code->storage_size = sizeof(*code) + code->count * sizeof(*code->compiled);
    for (iree_host_size_t i = 0; i < code->count; ++i) {
      code->storage_size +=
          loom_serve_jit_stage_storage_size(code->compiled[i]);
    }
    if (!job->needs_parameters) {
      status = loom_serve_choice_check(choice);
      if (iree_status_is_ok(status)) {
        status = loom_serve_choice_record(job);
      }
      job->recorded = true;
    }
  }
  loom_serve_choice_prepared(job, status);
}

static void loom_serve_choice_index_execute(loomc_task_t* task,
                                            loomc_host_size_t worker) {
  loom_serve_choice_job_t* job = (loom_serve_choice_job_t*)task;
  loom_serve_choice_t* choice = job->choice;
  const loom_model_file_list_t* paths = NULL;
  const uint32_t count =
      loom_serve_preparation_local_sources(choice->preparation, &paths);
  job->status = loom_serve_parameter_sources_create(
      choice->owner->options.parameter_sources, choice->owner->sources, count,
      paths, choice->owner->host_allocator, &choice->parameter_sources);
}

static void loom_serve_choice_index_complete(loomc_task_t* task) {
  loom_serve_choice_job_t* job = (loom_serve_choice_job_t*)task;
  iree_status_t status = job->status;
  job->status = iree_ok_status();
  if (iree_status_is_ok(status) && job->needs_compilation) {
    status = loom_serve_stage_set_compile_catalog_async(
        job->choice->generation->jit, job->choice->preparation, NULL,
        loom_serve_choice_compiled, job, job->choice->owner->host_allocator);
    if (iree_status_is_ok(status)) {
      return;
    }
  }
  loom_serve_choice_prepared(job, status);
}

static const loomc_task_vtable_t loom_serve_choice_index_vtable = {
    .execute = loom_serve_choice_index_execute,
    .destroy = loom_serve_choice_index_complete,
};

static void loom_serve_choice_record_execute(loomc_task_t* task,
                                             loomc_host_size_t worker) {
  loom_serve_choice_job_t* job = (loom_serve_choice_job_t*)task;
  job->status = loom_serve_choice_record(job);
}

static void loom_serve_choice_record_complete(loomc_task_t* task) {
  loom_serve_choice_job_t* job = (loom_serve_choice_job_t*)task;
  loom_serve_choice_t* choice = job->choice;
  iree_status_t status = job->status;
  loom_serve_choice_job_free(job);
  loom_serve_choice_complete(choice, status);
}

static const loomc_task_vtable_t loom_serve_choice_record_vtable = {
    .execute = loom_serve_choice_record_execute,
    .destroy = loom_serve_choice_record_complete,
};

static iree_status_t loom_serve_choice_record_async(
    loom_serve_choice_job_t* job) {
  loom_serve_executables_t* executables = job->choice->owner;
  loomc_task_initialize(&loom_serve_choice_record_vtable, &job->task);
  iree_status_t status = iree_status_from_loomc(loomc_task_sink_submit(
      loom_serve_host_compiler_sink(
          loom_serve_device_host(executables->options.device)),
      &job->task));
  return status;
}

static int loom_serve_group_compare(const void* lhs, const void* rhs) {
  const uintptr_t a = (uintptr_t)*(loom_serve_residency_group_t* const*)lhs;
  const uintptr_t b = (uintptr_t)*(loom_serve_residency_group_t* const*)rhs;
  return (a > b) - (a < b);
}

static iree_status_t loom_serve_choice_allocate_residency(
    loom_serve_choice_job_t* job) {
  loom_serve_choice_t* choice = job->choice;
  if (!choice->checkpoint_count) {
    return iree_ok_status();
  }
  loom_serve_executables_t* executables = choice->owner;
  const iree_host_size_t count =
      loom_serve_choice_parameter_count(choice->preparation);
  iree_hal_buffer_t** buffers = (iree_hal_buffer_t**)(job->bindings + count);
  loom_serve_residency_group_t** groups =
      (loom_serve_residency_group_t**)(buffers + count);
  iree_host_size_t group_count = 0;
  loom_serve_residency_group_t* base =
      loom_serve_parameter_bank_group(executables->options.parameters);
  if (base) {
    groups[group_count++] = base;
  }
  for (iree_host_size_t i = 0; i < choice->checkpoint_count; ++i) {
    groups[group_count++] =
        loom_serve_parameter_bank_group(choice->checkpoints[i]->bank);
  }
  qsort(groups, group_count, sizeof(*groups), loom_serve_group_compare);
  iree_host_size_t distinct_count = 0;
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    if (!distinct_count || groups[i] != groups[distinct_count - 1]) {
      groups[distinct_count++] = groups[i];
    }
  }
  return loom_serve_residency_allocate(
      loom_serve_device_residency_cache(executables->options.device),
      executables->options.execution, distinct_count, groups,
      executables->host_allocator, &choice->residency);
}

static void loom_serve_choice_advance(loom_serve_executables_t* executables,
                                      bool cancel) {
  loom_serve_choice_job_t* job = executables->invocation.job;
  iree_slim_mutex_lock(&job->mutex);
  if (job->pending) {
    iree_slim_mutex_unlock(&job->mutex);
    return;
  }
  iree_status_t status = job->status;
  job->status = iree_ok_status();
  iree_slim_mutex_unlock(&job->mutex);
  loom_serve_choice_t* choice = job->choice;
  if (job->recorded) {
    executables->invocation.job = NULL;
    loom_serve_choice_job_free(job);
    loom_serve_choice_complete(choice, status);
    return;
  }
  if (job->preparing && iree_status_is_ok(status)) {
    job->preparing->ready = true;
    job->preparing = NULL;
  }
  if (cancel && iree_status_is_ok(status)) {
    status = iree_make_status(IREE_STATUS_CANCELLED);
  }
  if (!job->index && iree_status_is_ok(status)) {
    status = loom_serve_choice_check(choice);
  }
  const iree_host_size_t count =
      loom_serve_choice_parameter_count(choice->preparation);
  while (job->index < count && iree_status_is_ok(status)) {
    if (!job->bindings[job->index].bank) {
      status = loom_serve_choice_bind_parameter(job, (uint32_t)job->index,
                                                &job->preparing);
    }
    ++job->index;
    if (iree_status_is_ok(status) && job->preparing) {
      job->pending = true;
      status = loom_serve_weights_prepare_async(
          loom_serve_parameter_bank_weights(job->preparing->bank, 0),
          loom_serve_device_host(executables->options.device),
          choice->generation->jit, executables->options.recording.command_mode,
          executables->sources, &job->preparing->declaration,
          choice->parameter_sources, loom_serve_choice_prepared, job);
      if (iree_status_is_ok(status)) {
        return;
      }
      job->pending = false;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_choice_allocate_residency(job);
  }
  // Accepted recording may finish synchronously. It owns the job from here;
  // only the variant publication remains visible to the invocation owner.
  executables->invocation.job = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_choice_record_async(job);
    if (iree_status_is_ok(status)) {
      return;
    }
  }
  loom_serve_choice_job_free(job);
  loom_serve_choice_complete(choice, status);
}

static iree_status_t loom_serve_choice_begin(
    loom_serve_executables_t* executables) {
  if (!executables->cache.pool) {
    IREE_RETURN_IF_ERROR(loom_serve_variant_pool_allocate(
        executables->options.cache.capacity + 2, NULL, loom_serve_choice_free,
        executables->host_allocator, &executables->cache.original,
        &executables->cache.pool));
    executables->cache.fixed_bytes =
        loom_serve_variant_pool_storage_size(executables->cache.pool);
    loom_serve_variant_pool_set_observer(executables->cache.pool,
                                         executables->cache.observer);
  }
  loom_serve_choice_code_t* shared = NULL;
  for (loom_serve_variant_entry_t* entry =
           loom_serve_variant_entry_next(executables->cache.original);
       entry; entry = loom_serve_variant_entry_next(entry)) {
    const loom_serve_choice_t* choice = loom_serve_variant_entry_value(entry);
    if (choice->generation == executables->captured &&
        loom_serve_preparation_equal(choice->preparation,
                                     executables->invocation.declarations)) {
      loom_serve_variant_entry_retain(entry);
      loom_serve_variant_entry_touch(entry);
      executables->invocation.entry = entry;
      for (iree_host_size_t i = 0; i < choice->checkpoint_count; ++i) {
        loom_serve_checkpoint_touch(executables, choice->checkpoints[i]);
      }
      loom_serve_preparation_free(executables->invocation.declarations);
      executables->invocation.declarations = NULL;
      return iree_ok_status();
    }
    if (choice->generation == executables->captured &&
        loom_serve_preparation_code_equal(
            choice->preparation, executables->invocation.declarations)) {
      shared = choice->code;
    }
  }
  loom_serve_choice_t* choice = NULL;
  const iree_host_size_t count =
      loom_serve_choice_parameter_count(executables->invocation.declarations);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      executables->host_allocator, sizeof(*choice), count,
      sizeof(*choice->checkpoints), (void**)&choice));
  choice->owner = executables;
  choice->generation = executables->captured;
  loom_serve_generation_retain(choice->generation);
  ++choice->generation->choices.count;
  choice->code = shared;
  if (shared) {
    ++shared->users;
  }
  iree_status_t status = loom_serve_variant_pool_begin(
      executables->cache.pool,
      loom_serve_variant_entry_id(executables->cache.original), &choice->entry);
  if (!iree_status_is_ok(status)) {
    loom_serve_choice_free(choice);
    return status;
  }
  choice->preparation = executables->invocation.declarations;
  executables->invocation.declarations = NULL;
  executables->invocation.entry = choice->entry;
  executables->invocation.pending = true;
  loom_serve_choice_job_t* job = NULL;
  status = iree_allocator_malloc_struct_array(
      executables->host_allocator,
      sizeof(*job) + sizeof(loom_serve_residency_group_t*), count,
      sizeof(*job->bindings) + sizeof(iree_hal_buffer_t*) +
          sizeof(loom_serve_residency_group_t*),
      (void**)&job);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_initialize(&job->mutex);
    job->choice = choice;
    const loom_serve_preparation_stage_t* selected =
        count ? loom_serve_preparation_stage(choice->preparation, 0) : NULL;
    const loom_serve_preparation_stage_t* retained =
        loom_serve_preparation_stage(executables->options.preparation, 0);
    for (iree_host_size_t i = 0; i < count; ++i) {
      for (uint32_t domain = 0; domain < retained->parameter_count; ++domain) {
        if (loom_serve_preparation_parameter_equal(
                &selected->parameters[i], &retained->parameters[domain])) {
          job->bindings[i] = (loom_serve_parameter_binding_t){
              executables->options.parameters, domain};
          break;
        }
      }
      job->needs_parameters |= !job->bindings[i].bank;
    }
    executables->invocation.job = job;
  }
  if (iree_status_is_ok(status) && !shared) {
    status =
        iree_allocator_malloc(executables->host_allocator,
                              sizeof(*choice->code), (void**)&choice->code);
    if (iree_status_is_ok(status)) {
      choice->code->users = 1;
      job->needs_compilation = true;
    }
  }
  if (iree_status_is_ok(status)) {
    job->pending = true;
    loomc_task_initialize(&loom_serve_choice_index_vtable, &job->task);
    status = iree_status_from_loomc(loomc_task_sink_submit(
        loom_serve_host_preparation_sink(
            loom_serve_device_host(executables->options.device)),
        &job->task));
  }
  if (!iree_status_is_ok(status)) {
    executables->invocation.job = NULL;
    if (job) {
      loom_serve_choice_job_free(job);
    }
    loom_serve_variant_entry_complete(choice->entry, status, choice, NULL);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_executables_advance(
    loom_serve_executables_t* executables, bool cancel, bool* out_complete) {
  *out_complete = true;
  IREE_RETURN_IF_ERROR(loom_serve_executables_query_status(executables));
  if (!cancel && executables->invocation.declarations) {
    if (executables->parameters.retired) {
      *out_complete = false;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_serve_choice_begin(executables));
  }
  if (executables->invocation.job) {
    loom_serve_choice_advance(executables, cancel);
  }
  if (executables->invocation.pending) {
    loom_serve_variant_pool_collect(executables->cache.pool);
    loom_serve_variant_entry_t* entry = executables->invocation.entry;
    const loom_serve_variant_snapshot_t snapshot =
        loom_serve_variant_entry_snapshot(entry);
    if (snapshot.phase == LOOM_SERVE_VARIANT_COMPILING) {
      *out_complete = false;
      return iree_ok_status();
    }
    executables->invocation.pending = false;
    if (snapshot.phase == LOOM_SERVE_VARIANT_FAILED) {
      iree_status_t status = loom_serve_variant_entry_status(entry);
      loom_serve_variant_entry_discard(entry);
      executables->invocation.entry = NULL;
      if (cancel && iree_status_is_cancelled(status)) {
        iree_status_free(status);
        return iree_ok_status();
      }
      return status;
    }
    loom_serve_choice_t* choice = loom_serve_variant_entry_value(entry);
    choice->accounted = true;
    executables->cache.payload_bytes += choice->storage_size;
    if (!choice->code->accounted) {
      choice->code->accounted = true;
      executables->cache.payload_bytes += choice->code->storage_size;
    }
    loom_serve_account_natives(executables, choice->generation);
    loom_serve_variant_entry_retain(entry);
  }
  return cancel ? iree_ok_status() : loom_serve_executables_bind(executables);
}

const loom_serve_stage_launch_t* loom_serve_executables_stage(
    const loom_serve_executables_t* executables, uint32_t ordinal) {
  const loom_serve_choice_t* choice =
      executables->invocation.entry
          ? loom_serve_variant_entry_value(executables->invocation.entry)
          : NULL;
  const loom_serve_stage_table_t* table =
      choice ? choice->table : executables->captured->table;
  return &table->launches[ordinal];
}

static void loom_serve_generation_complete(loom_serve_generation_job_t* job,
                                           iree_status_t status) {
  loom_serve_generation_t* generation = job->generation;
  const loom_serve_variant_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(generation->host_allocator, job);
  completion(user_data, status, generation, generation->report);
}

static void loom_serve_generation_compiled(void* user_data,
                                           iree_status_t status,
                                           loom_serve_stage_set_t* stages) {
  loom_serve_generation_job_t* job = user_data;
  job->generation->count = stages->count;
  job->generation->compiled = stages->values;
  stages->count = 0;
  stages->values = NULL;
  loom_serve_stage_set_free(stages);
  if (iree_status_is_ok(status)) {
    loom_serve_stage_table_t* table = NULL;
    status = loom_serve_executables_record(
        &job->executables->options, NULL, job->generation->count,
        job->generation->compiled, job->generation->host_allocator, &table);
    job->generation->table = table;
  }
  if (iree_status_is_ok(status)) {
    job->generation->cold_native_bytes =
        loom_serve_native_bytes(job->generation->jit);
  }
  loom_serve_generation_complete(job, status);
}

static void loom_serve_generation_index(loomc_task_t* task,
                                        loomc_host_size_t worker) {
  loom_serve_generation_job_t* job =
      (loom_serve_generation_job_t*)((uint8_t*)task -
                                     offsetof(loom_serve_generation_job_t,
                                              task));
  loom_serve_generation_t* generation = job->generation;
  loom_serve_device_t* device = job->executables->options.device;
  job->status = loom_serve_jit_create(
      loom_serve_device_handle(device),
      loom_serve_device_dispatch_queue(device), loom_serve_device_host(device),
      &job->sources, NULL, generation->report, generation->host_allocator,
      &generation->jit);
}

static void loom_serve_generation_indexed(loomc_task_t* task) {
  loom_serve_generation_job_t* job =
      (loom_serve_generation_job_t*)((uint8_t*)task -
                                     offsetof(loom_serve_generation_job_t,
                                              task));
  iree_status_t status = job->status;
  if (iree_status_is_ok(status)) {
    status = loom_serve_stage_set_compile_catalog_async(
        job->generation->jit, job->executables->options.preparation, NULL,
        loom_serve_generation_compiled, job, job->generation->host_allocator);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_generation_complete(job, status);
  }
}

static const loomc_task_vtable_t executable_generation_task_vtable = {
    .execute = loom_serve_generation_index,
    .destroy = loom_serve_generation_indexed,
};

static iree_status_t loom_serve_generation_compile(
    void* self, const loom_serve_sources_t* sources,
    loom_serve_variant_completion_fn_t completion, void* user_data) {
  loom_serve_executables_t* executables = self;
  loom_serve_generation_t* generation = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_generation_create(
      NULL, executables->host_allocator, &generation));
  loom_serve_generation_job_t* job = NULL;
  iree_status_t status = iree_allocator_malloc(executables->host_allocator,
                                               sizeof(*job), (void**)&job);
  if (iree_status_is_ok(status)) {
    job->executables = executables;
    job->generation = generation;
    job->sources = *sources;
    job->completion = completion;
    job->user_data = user_data;
    loomc_task_initialize(&executable_generation_task_vtable, &job->task);
    status = iree_status_from_loomc(loomc_task_sink_submit(
        loom_serve_host_compiler_sink(
            loom_serve_device_host(executables->options.device)),
        &job->task));
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(executables->host_allocator, job);
    loom_serve_generation_release(generation);
  }
  return status;
}

static iree_status_t loom_serve_generation_select(void* self, void* value) {
  loom_serve_executables_t* executables = self;
  loom_serve_generation_t* generation = value ? value : executables->original;
  loom_serve_generation_retain(generation);
  loom_serve_generation_release(executables->selected);
  executables->selected = generation;
  return iree_ok_status();
}

loom_serve_execution_update_t loom_serve_executables_update(
    loom_serve_executables_t* executables) {
  return (loom_serve_execution_update_t){
      .self = executables,
      .compile = loom_serve_generation_compile,
      .select = loom_serve_generation_select,
      .dispose = loom_serve_generation_release};
}
