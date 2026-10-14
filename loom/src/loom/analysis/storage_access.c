// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/storage_access.h"

#include <string.h>

#include "loom/analysis/movement.h"
#include "loom/analysis/value_relation.h"
#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/util/walk.h"

typedef struct loom_storage_access_function_record_t {
  // Completed local graph consumed by allocation interference.
  loom_storage_access_function_t graph;
  // Source definition whose body remains alive throughout construction.
  loom_func_like_t function;
  // Module-local symbol indexing this record.
  uint16_t symbol_id;
  // Sparse symbol index branch; the first record needs only its leaf.
  struct {
    // Tagged branch or untagged leaf for each value of |bit|.
    uintptr_t children[2];
    // Discriminating symbol bit, decreasing along every branch path.
    uint8_t bit;
  } branch;
  // Pending body discovery; each reached definition is queued exactly once.
  struct loom_storage_access_function_record_t* next_pending;
  // Reference nodes in formal argument order; scalar entries are NULL.
  loom_storage_reference_t** arguments;
  // Last direct use retained in source walk order, or NULL.
  loom_storage_access_use_t* use_tail;
  // Last rendezvous retained in source order, or NULL.
  loom_storage_access_barrier_t* barrier_tail;
} loom_storage_access_function_record_t;

typedef struct loom_storage_access_state_t {
  // Borrowed immutable-source scope and its construction arena.
  loom_storage_access_scope_t* scope;
  // Sparse radix index over reached 16-bit module symbol IDs.
  uintptr_t function_index;
  // First reached definition awaiting body discovery.
  loom_storage_access_function_record_t* first_function;
  // Last reached definition awaiting body discovery.
  loom_storage_access_function_record_t* last_function;
  // Stable reference pointers indexed by the temporary module scratch map.
  loom_storage_reference_t** references;
  // Number of initialized reference pointers.
  iree_host_size_t reference_count;
  // Allocated reference pointer capacity.
  iree_host_size_t reference_capacity;
  // First reference awaiting monotone effect propagation.
  loom_storage_reference_t* first_effect;
  // Last reference awaiting monotone effect propagation.
  loom_storage_reference_t* last_effect;
  // Function whose source is currently being interpreted.
  loom_storage_access_function_record_t* current;
  // Reusable direct-operand classification scratch.
  loom_storage_access_effects_t* operand_effects;
  // Allocated operand classification capacity.
  iree_host_size_t operand_capacity;
} loom_storage_access_state_t;

static bool loom_storage_access_is_reference(const loom_module_t* module,
                                             loom_value_id_t value_id) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  return loom_type_is_buffer(type) || loom_type_is_view(type);
}

// Tagged branches and leaves share stable function-record storage. Every branch
// consumes a distinct bit of a 16-bit symbol, bounding lookup independently of
// source order without allocating slots for unqueried module symbols.
static loom_storage_access_function_record_t* loom_storage_access_find_function(
    uintptr_t index, uint16_t symbol_id) {
  while (index & 1) {
    const loom_storage_access_function_record_t* record =
        (const loom_storage_access_function_record_t*)(index & ~(uintptr_t)1);
    index = record->branch.children[(symbol_id >> record->branch.bit) & 1];
  }
  return (loom_storage_access_function_record_t*)index;
}

static void loom_storage_access_index_function(
    loom_storage_access_state_t* state,
    loom_storage_access_function_record_t* record,
    const loom_storage_access_function_record_t* previous) {
  uintptr_t* edge = &state->function_index;
  if (previous == NULL) {
    *edge = (uintptr_t)record;
    return;
  }
  uint16_t difference = record->symbol_id ^ previous->symbol_id;
  uint8_t bit = 0;
  while ((difference >>= 1) != 0) {
    ++bit;
  }
  while (*edge & 1) {
    loom_storage_access_function_record_t* branch =
        (loom_storage_access_function_record_t*)(*edge & ~(uintptr_t)1);
    if (branch->branch.bit < bit) {
      break;
    }
    edge =
        &branch->branch.children[(record->symbol_id >> branch->branch.bit) & 1];
  }
  const uint8_t direction = (record->symbol_id >> bit) & 1;
  record->branch.bit = bit;
  record->branch.children[direction] = (uintptr_t)record;
  record->branch.children[direction ^ 1] = *edge;
  *edge = (uintptr_t)record | 1;
}

static iree_status_t loom_storage_access_add_reference(
    loom_storage_access_state_t* state,
    loom_storage_access_function_record_t* function, loom_value_id_t value_id,
    loom_storage_reference_t** out_reference) {
  if (state->reference_count == UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "storage access graph exceeds reference index range");
  }
  if (state->reference_count == state->reference_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->scope->arena, state->reference_count, state->reference_count + 1,
        sizeof(*state->references), &state->reference_capacity,
        (void**)&state->references));
  }
  loom_storage_reference_t* reference = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      state->scope->arena, sizeof(*reference), (void**)&reference));
  *reference = (loom_storage_reference_t){
      .value_id = value_id,
      .index = (uint32_t)state->reference_count,
      .next = function->graph.references,
  };
  state->references[state->reference_count++] = reference;
  function->graph.references = reference;
  *out_reference = reference;
  return iree_ok_status();
}

static iree_status_t loom_storage_access_enqueue_function(
    loom_storage_access_state_t* state, loom_func_like_t function,
    loom_storage_access_function_record_t** out_record) {
  const uint16_t symbol_id = loom_func_like_callee(function).symbol_id;
  loom_storage_access_function_record_t* previous =
      loom_storage_access_find_function(state->function_index, symbol_id);
  if (previous && previous->symbol_id == symbol_id) {
    *out_record = previous;
    return iree_ok_status();
  }
  loom_storage_access_function_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(state->scope->arena, sizeof(*record),
                                           (void**)&record));
  *record = (loom_storage_access_function_record_t){
      .function = function,
      .symbol_id = symbol_id,
  };
  const loom_block_t* entry =
      loom_region_entry_block(loom_func_like_body(function));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scope->arena, entry->arg_count, sizeof(*record->arguments),
      (void**)&record->arguments));
  for (uint16_t i = 0; i < entry->arg_count; ++i) {
    record->arguments[i] = NULL;
    const loom_value_id_t value_id = loom_block_arg_id(entry, i);
    if (loom_storage_access_is_reference(state->scope->module, value_id)) {
      IREE_RETURN_IF_ERROR(loom_storage_access_add_reference(
          state, record, value_id, &record->arguments[i]));
    }
  }
  loom_storage_access_index_function(state, record, previous);
  if (state->last_function) {
    state->last_function->next_pending = record;
  } else {
    state->first_function = record;
  }
  state->last_function = record;
  *out_record = record;
  return iree_ok_status();
}

static iree_status_t loom_storage_access_reference(
    loom_storage_access_state_t* state, loom_value_id_t value_id,
    loom_storage_reference_t** out_reference) {
  loom_value_u32_scratch_t* scratch = &state->scope->module->scratch.values;
  uint32_t index = loom_value_u32_scratch_load(scratch, value_id);
  if (index != UINT32_MAX) {
    *out_reference = state->references[index];
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_storage_access_add_reference(
      state, state->current, value_id, out_reference));
  loom_value_u32_scratch_store(scratch, value_id, (*out_reference)->index);
  return iree_ok_status();
}

static void loom_storage_access_include(loom_storage_access_state_t* state,
                                        loom_storage_reference_t* reference,
                                        loom_storage_access_effects_t effects) {
  if ((reference->effects | effects) == reference->effects) {
    return;
  }
  reference->effects |= effects;
  if (reference->queued) {
    return;
  }
  reference->queued = true;
  reference->next_pending = NULL;
  if (state->last_effect) {
    state->last_effect->next_pending = reference;
  } else {
    state->first_effect = reference;
  }
  state->last_effect = reference;
}

enum loom_storage_reference_edge_flag_bits_e {
  LOOM_STORAGE_REFERENCE_EDGE_LOCAL_TRANSPORT = 1u << 0,
};
typedef uint8_t loom_storage_reference_edge_flags_t;

static iree_status_t loom_storage_access_add_edge(
    loom_storage_access_state_t* state, loom_storage_reference_t* origin,
    loom_storage_reference_t* destination,
    loom_storage_reference_edge_flags_t flags) {
  if (origin == destination) {
    return iree_ok_status();
  }
  loom_storage_reference_edge_t* edge = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(state->scope->arena, sizeof(*edge), (void**)&edge));
  *edge = (loom_storage_reference_edge_t){
      .origin = origin,
      .destination = destination,
      .next_origin = destination->origins,
  };
  destination->origins = edge;
  if (iree_any_bit_set(flags, LOOM_STORAGE_REFERENCE_EDGE_LOCAL_TRANSPORT)) {
    edge->next_destination = origin->destinations;
    origin->destinations = edge;
  }
  // A new caller can depend on an already completed callee. Retain its current
  // effects as well as the edge carrying any remaining construction updates.
  loom_storage_access_include(state, origin, destination->effects);
  return iree_ok_status();
}

static iree_status_t loom_storage_access_record_use(
    loom_storage_access_state_t* state, const loom_op_t* op,
    loom_value_id_t value_id, loom_storage_access_effects_t effects,
    const loom_storage_reference_t* callee_argument) {
  if (effects == 0 && callee_argument == NULL) {
    return iree_ok_status();
  }
  loom_storage_reference_t* reference = NULL;
  IREE_RETURN_IF_ERROR(
      loom_storage_access_reference(state, value_id, &reference));
  loom_storage_access_use_t* use = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(state->scope->arena, sizeof(*use), (void**)&use));
  *use = (loom_storage_access_use_t){
      .operation = op,
      .value_id = value_id,
      .effects = effects,
      .callee_argument = callee_argument,
      .next = reference->uses,
  };
  reference->uses = use;
  if (state->current->use_tail) {
    state->current->use_tail->next_in_function = use;
  } else {
    state->current->graph.uses = use;
  }
  state->current->use_tail = use;
  if (iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_ASYNC)) {
    effects =
        (effects & ~LOOM_STORAGE_ACCESS_ASYNC) | LOOM_STORAGE_ACCESS_ESCAPE;
  }
  loom_storage_access_include(state, reference, effects);
  return iree_ok_status();
}

static iree_status_t loom_storage_access_record_barrier(
    loom_storage_access_state_t* state, const loom_op_t* op) {
  if (!loom_kernel_barrier_isa(op) ||
      loom_kernel_barrier_memory_space(op) !=
          LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      loom_kernel_barrier_scope(op) != LOOM_ATOMIC_SCOPE_WORKGROUP ||
      loom_kernel_barrier_ordering(op) != LOOM_ATOMIC_ORDERING_ACQ_REL) {
    return iree_ok_status();
  }
  loom_storage_access_barrier_t* barrier = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(state->scope->arena,
                                           sizeof(*barrier), (void**)&barrier));
  *barrier = (loom_storage_access_barrier_t){.operation = op};
  if (state->current->barrier_tail) {
    state->current->barrier_tail->next = barrier;
  } else {
    state->current->graph.barriers = barrier;
  }
  state->current->barrier_tail = barrier;
  return iree_ok_status();
}

static iree_status_t loom_storage_access_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_storage_access_state_t* state = user_data;
  loom_module_t* module = state->scope->module;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  IREE_RETURN_IF_ERROR(loom_storage_access_record_barrier(state, op));
  if (op->operand_count > state->operand_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->scope->arena, 0, op->operand_count,
        sizeof(*state->operand_effects), &state->operand_capacity,
        (void**)&state->operand_effects));
  }
  if (op->operand_count) {
    memset(state->operand_effects, LOOM_STORAGE_ACCESS_ESCAPE,
           op->operand_count * sizeof(*state->operand_effects));
  }
  const loom_call_like_t call = loom_call_like_cast(module, op);
  loom_storage_access_function_record_t* callee_record = NULL;
  if (loom_call_like_is_direct_semantic(call)) {
    const loom_symbol_ref_t callee = loom_call_like_callee(call);
    if (callee.module_id == 0) {
      const loom_func_like_t function = loom_func_like_cast(
          module, module->symbols.entries[callee.symbol_id].defining_op);
      if (!loom_func_like_is_kernel(function) &&
          loom_func_like_repr_contract(function) == LOOM_STRING_ID_INVALID &&
          loom_func_like_body(function)) {
        IREE_RETURN_IF_ERROR(loom_storage_access_enqueue_function(
            state, function, &callee_record));
      }
    }
  }
  // Calls cannot access fresh caller storage without receiving a reference.
  // Opaque calls keep the default escape effect on each exposed operand;
  // ambient effects in a callee do not reach an unexposed caller root.
  if (callee_record) {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      loom_storage_reference_t* formal = callee_record->arguments[i];
      if (!formal) {
        continue;
      }
      loom_storage_reference_t* actual = NULL;
      IREE_RETURN_IF_ERROR(
          loom_storage_access_reference(state, operands[i], &actual));
      IREE_RETURN_IF_ERROR(
          loom_storage_access_add_edge(state, actual, formal, 0));
      IREE_RETURN_IF_ERROR(
          loom_storage_access_record_use(state, op, operands[i], 0, formal));
      state->operand_effects[i] = 0;
    }
  } else if (!loom_call_like_isa(call)) {
    bool described_read = false;
    bool described_write = false;
    const bool asynchronous = loom_movement_op_kind_is_async(op->kind);
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      const loom_operand_descriptor_t* descriptor = NULL;
      if (!loom_op_operand_descriptor_at(vtable, op, i, &descriptor, NULL,
                                         NULL)) {
        continue;
      }
      if (iree_any_bit_set(descriptor->flags,
                           LOOM_OPERAND_OBSERVES_REFERENCE)) {
        state->operand_effects[i] = 0;
      }
      loom_storage_access_effects_t effects = 0;
      if (iree_any_bit_set(descriptor->flags, LOOM_OPERAND_READS)) {
        effects |= LOOM_STORAGE_ACCESS_READ;
        described_read = true;
      }
      if (iree_any_bit_set(descriptor->flags, LOOM_OPERAND_WRITES)) {
        effects |= LOOM_STORAGE_ACCESS_WRITE;
        described_write = true;
      }
      if (effects) {
        state->operand_effects[i] =
            effects | (asynchronous ? LOOM_STORAGE_ACCESS_ASYNC : 0);
      }
    }
    const bool stream_completion =
        loom_kernel_async_group_isa(op) || loom_kernel_async_wait_isa(op);
    if (!stream_completion &&
        (iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS) ||
         (iree_any_bit_set(traits, LOOM_TRAIT_READS_MEMORY) &&
          !described_read) ||
         (iree_any_bit_set(traits, LOOM_TRAIT_WRITES_MEMORY) &&
          !described_write))) {
      state->current->graph.has_unknown_memory_access = true;
    }
    const loom_value_relation_mask_t mask =
        LOOM_VALUE_RELATION_MASK_ALL &
        ~LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_ELEMENTWISE);
    loom_value_relation_iterator_t iterator;
    loom_value_relation_iterator_initialize(module, op, mask, &iterator);
    loom_value_relation_t relation;
    while (loom_value_relation_iterator_next(&iterator, &relation)) {
      if (!loom_storage_access_is_reference(module, relation.source_value_id) ||
          !loom_storage_access_is_reference(module,
                                            relation.destination_value_id)) {
        continue;
      }
      loom_storage_reference_t* origin = NULL;
      loom_storage_reference_t* destination = NULL;
      IREE_RETURN_IF_ERROR(loom_storage_access_reference(
          state, relation.source_value_id, &origin));
      IREE_RETURN_IF_ERROR(loom_storage_access_reference(
          state, relation.destination_value_id, &destination));
      IREE_RETURN_IF_ERROR(loom_storage_access_add_edge(
          state, origin, destination,
          LOOM_STORAGE_REFERENCE_EDGE_LOCAL_TRANSPORT));
      if (relation.source_operand_index !=
          LOOM_VALUE_RELATION_OPERAND_INDEX_NONE) {
        state->operand_effects[relation.source_operand_index] &=
            ~LOOM_STORAGE_ACCESS_ESCAPE;
      }
    }
  }
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (loom_storage_access_is_reference(module, operands[i])) {
      IREE_RETURN_IF_ERROR(loom_storage_access_record_use(
          state, op, operands[i], state->operand_effects[i], NULL));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_storage_access_build_pending(
    loom_storage_access_state_t* state) {
  loom_module_t* module = state->scope->module;
  iree_status_t status = iree_ok_status();
  while (state->first_function && iree_status_is_ok(status)) {
    state->current = state->first_function;
    state->first_function = state->current->next_pending;
    if (!state->first_function) {
      state->last_function = NULL;
    }
    loom_module_value_ordinal_scratch_acquire(module);
    for (const loom_storage_reference_t* reference =
             state->current->graph.references;
         reference; reference = reference->next) {
      loom_value_u32_scratch_store(&module->scratch.values, reference->value_id,
                                   reference->index);
    }
    loom_walk_result_t walk_result;
    status =
        loom_walk_region(module, loom_func_like_body(state->current->function),
                         LOOM_WALK_PRE_ORDER,
                         (loom_walk_callback_t){.fn = loom_storage_access_visit,
                                                .user_data = state},
                         &walk_result);
    if (iree_status_is_ok(status) &&
        state->current->graph.has_unknown_memory_access) {
      // An unmodeled local effect can reach every formal reference. Project
      // that uncertainty through argument edges, never through an unrelated
      // caller allocation. This runs once after the body's uses are classified.
      const loom_block_t* entry = loom_region_entry_block(
          loom_func_like_body(state->current->function));
      for (uint16_t i = 0; i < entry->arg_count; ++i) {
        if (state->current->arguments[i]) {
          loom_storage_access_include(state, state->current->arguments[i],
                                      LOOM_STORAGE_ACCESS_ESCAPE);
        }
      }
    }
    for (const loom_storage_reference_t* reference =
             state->current->graph.references;
         reference; reference = reference->next) {
      loom_value_u32_scratch_store(&module->scratch.values, reference->value_id,
                                   LOOM_VALUE_ORDINAL_INVALID);
    }
    loom_module_value_ordinal_scratch_release(module);
  }
  return status;
}

static void loom_storage_access_solve(loom_storage_access_state_t* state) {
  while (state->first_effect) {
    loom_storage_reference_t* reference = state->first_effect;
    state->first_effect = reference->next_pending;
    if (!state->first_effect) {
      state->last_effect = NULL;
    }
    reference->queued = false;
    for (const loom_storage_reference_edge_t* edge = reference->origins; edge;
         edge = edge->next_origin) {
      loom_storage_access_include(state, edge->origin, reference->effects);
    }
  }
}

void loom_storage_access_scope_initialize(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_storage_access_scope_t* out_scope) {
  *out_scope = (loom_storage_access_scope_t){.module = module, .arena = arena};
}

iree_status_t loom_storage_access_require_function(
    loom_storage_access_scope_t* scope, loom_func_like_t function,
    loom_local_value_domain_t* value_domain,
    const loom_storage_access_function_t** out_function) {
  *out_function = NULL;
  if (scope->state == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        scope->arena, sizeof(*scope->state), (void**)&scope->state));
    *scope->state = (loom_storage_access_state_t){.scope = scope};
  }
  loom_storage_access_function_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(
      loom_storage_access_enqueue_function(scope->state, function, &record));
  iree_status_t status = iree_ok_status();
  if (scope->state->first_function) {
    loom_local_value_domain_release(value_domain);
    status = loom_storage_access_build_pending(scope->state);
    loom_local_value_domain_restore(value_domain);
    if (iree_status_is_ok(status)) {
      loom_storage_access_solve(scope->state);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_function = &record->graph;
  }
  return status;
}
