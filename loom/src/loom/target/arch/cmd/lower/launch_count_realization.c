// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/launch_count_realization.h"

#include <string.h>

#include "loom/analysis/liveness_uses.h"
#include "loom/analysis/motion.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/type_dependencies.h"
#include "loom/link/linker.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_extensions.h"

typedef enum loom_cmd_launch_count_value_kind_e {
  LOOM_CMD_LAUNCH_COUNT_VALUE_UNSEEN = 0,
  LOOM_CMD_LAUNCH_COUNT_VALUE_VISITING = 1,
  LOOM_CMD_LAUNCH_COUNT_VALUE_CONSTANT = 2,
  LOOM_CMD_LAUNCH_COUNT_VALUE_ARGUMENT = 3,
  LOOM_CMD_LAUNCH_COUNT_VALUE_PRODUCED = 4,
} loom_cmd_launch_count_value_kind_t;

typedef struct loom_cmd_launch_count_value_change_t {
  // Dense root-local value ordinal whose state changed.
  loom_value_ordinal_t ordinal;

  // Generation present before the speculative change.
  uint32_t generation;

  // Classification present before the speculative change.
  uint8_t kind;
} loom_cmd_launch_count_value_change_t;

typedef struct loom_cmd_launch_count_read_t {
  // Original memory observation retained by closure construction.
  const loom_op_t* operation;
  // Reference read at that observation.
  loom_value_id_t reference;
} loom_cmd_launch_count_read_t;

typedef struct loom_cmd_launch_count_closure_t {
  // Source module containing the normalized command root.
  loom_module_t* module;

  // Facts computed before any realization rewrite.
  const loom_value_fact_table_t* facts;

  // Dense root-local value domain shared across closures in this root.
  const loom_local_value_domain_t* domain;

  // Current nonzero closure generation.
  uint32_t generation;

  // Generation by local value ordinal.
  uint32_t* value_generations;

  // Closure classification by local value ordinal in |generation|.
  uint8_t* value_kinds;

  // Scratch arena owning closure lists.
  iree_arena_allocator_t* arena;

  // Root-owned read-motion index, allocated on the first captured read.
  loom_motion_read_barrier_table_t** read_barriers;

  // Root analysis storage, independent of speculative closure checkpoints.
  iree_arena_allocator_t* analysis_arena;

  // Original consumer of the count expression currently being captured.
  const loom_op_t* consumer;

  // First read preventing a common legal producer placement.
  const loom_op_t* invalid_observation;

  // A captured read needs a producer at its source observation.
  bool needs_source_placement;

  // Exact values materialized before cloning source operations.
  loom_value_id_t* constants;
  iree_host_size_t constant_count;
  iree_host_size_t constant_capacity;

  // Buffer or statically typed view values passed to the generated kernel.
  loom_value_id_t* arguments;
  iree_host_size_t argument_count;
  iree_host_size_t argument_capacity;

  // Source operations in dependency order.
  loom_op_t** operations;
  iree_host_size_t operation_count;
  iree_host_size_t operation_capacity;

  // Memory observations and workload-rooted references read by the closure.
  loom_cmd_launch_count_read_t* read_references;
  iree_host_size_t read_reference_count;
  iree_host_size_t read_reference_capacity;

  // Whether value-state writes are being recorded for rollback.
  bool record_value_changes;

  // Value-state undo log for one speculative cluster extension.
  loom_cmd_launch_count_value_change_t* value_changes;
  iree_host_size_t value_change_count;
  iree_host_size_t value_change_capacity;
} loom_cmd_launch_count_closure_t;

typedef struct loom_cmd_launch_count_root_t {
  // Normalized command function.
  loom_func_like_t function;

  // Dense values in the function region tree.
  loom_local_value_domain_t domain;

  // Generation by local value ordinal.
  uint32_t* value_generations;

  // Per-generation value classification by local value ordinal.
  uint8_t* value_kinds;

  // Active-cluster possible-write generation by alias-scope ordinal.
  uint32_t* write_scope_generations;

  // Next nonzero closure generation.
  uint32_t next_generation;

  // Canonical source-order read-motion index, shared by this root's closures.
  loom_motion_read_barrier_table_t* read_barriers;

  // Storage for analysis retained across closure checkpoint restoration.
  iree_arena_allocator_t* analysis_arena;
} loom_cmd_launch_count_root_t;

typedef struct loom_cmd_launch_count_transaction_t {
  // Arena position before speculative closure storage was allocated.
  iree_arena_checkpoint_t checkpoint;

  // Closure lists and capacities before speculative extension.
  loom_cmd_launch_count_closure_t prior_closure;
} loom_cmd_launch_count_transaction_t;

typedef struct loom_cmd_launch_count_dispatch_list_t {
  loom_op_t** values;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_arena_allocator_t* arena;
} loom_cmd_launch_count_dispatch_list_t;

typedef struct loom_cmd_launch_count_tuple_t {
  // Normalized XYZ source values. INVALID denotes the implicit value one.
  loom_value_id_t values[3];
} loom_cmd_launch_count_tuple_t;

typedef struct loom_cmd_launch_count_demand_t {
  // Original direct dispatch rewritten after its producer is inserted.
  loom_op_t* consumer;

  // Index of the unique XYZ tuple written for this consumer.
  iree_host_size_t tuple_ordinal;
} loom_cmd_launch_count_demand_t;

typedef struct loom_cmd_launch_count_cluster_t {
  // Target shared by every generated count tuple.
  loom_symbol_ref_t target;

  // Shared consumer block used to admit ordinary clustered configurations.
  loom_block_t* block;

  // First residual consumer in the cluster.
  loom_op_t* first_consumer;

  // Validated producer insertion point, usually the first consumer.
  loom_op_t* producer_before;

  // Last command crossed by producer placement.
  loom_op_t* frontier;

  // Shared transitive closure for every unique count tuple.
  loom_cmd_launch_count_closure_t closure;

  // Residual consumers retained in source order.
  loom_cmd_launch_count_demand_t* demands;
  iree_host_size_t demand_count;
  iree_host_size_t demand_capacity;

  // Unique normalized count tuples.
  loom_cmd_launch_count_tuple_t* tuples;
  iree_host_size_t tuple_count;
  iree_host_size_t tuple_capacity;

  // Open-addressed tuple ordinals plus one; zero denotes an empty bucket.
  iree_host_size_t* tuple_buckets;
  iree_host_size_t tuple_bucket_capacity;

  // Whether a crossed command may have written through a reference argument.
  bool has_possible_writes;

  // Whether a source-placement or alias constraint prevents later joins.
  bool closed;
} loom_cmd_launch_count_cluster_t;

// Exact emission inputs retained after root-local planning storage is released.
typedef struct loom_cmd_launch_count_plan_t {
  // Next cluster in planning order, or NULL for the final cluster.
  struct loom_cmd_launch_count_plan_t* next;
  // Target shared by every generated count tuple.
  loom_symbol_ref_t target;
  // Source insertion point for the generated producer.
  loom_op_t* producer_before;
  // First consumer, retained for diagnostics and execution-scope selection.
  loom_op_t* first_consumer;
  // Exact source values to materialize in the producer.
  loom_value_id_t* constants;
  // Number of exact source values.
  iree_host_size_t constant_count;
  // Source references bound as producer arguments.
  loom_value_id_t* arguments;
  // Number of source reference arguments.
  iree_host_size_t argument_count;
  // Source operations in dependency order, borrowed until all rewrites finish.
  loom_op_t** operations;
  // Number of captured source operations.
  iree_host_size_t operation_count;
  // Consumers and their tuple ordinals in source order.
  loom_cmd_launch_count_demand_t* demands;
  // Number of consumers.
  iree_host_size_t demand_count;
  // Unique normalized count tuples.
  loom_cmd_launch_count_tuple_t* tuples;
  // Number of unique count tuples.
  iree_host_size_t tuple_count;
  // Validated type of the contiguous output tuples.
  loom_type_t slab_view_type;
  // Validated output allocation size, in bytes.
  int64_t slab_byte_length;
} loom_cmd_launch_count_plan_t;

static iree_status_t loom_cmd_launch_count_root_prepare(
    loom_module_t* module, loom_cmd_launch_count_root_t* root,
    iree_arena_allocator_t* scratch_arena) {
  if (loom_local_value_domain_is_acquired(&root->domain)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      module, loom_func_like_body(root->function), scratch_arena,
      &root->domain));
  if (root->domain.value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, root->domain.value_count, sizeof(*root->value_generations),
      (void**)&root->value_generations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, root->domain.value_count, sizeof(*root->value_kinds),
      (void**)&root->value_kinds));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, root->domain.value_count,
                                sizeof(*root->write_scope_generations),
                                (void**)&root->write_scope_generations));
  memset(root->value_generations, 0,
         root->domain.value_count * sizeof(*root->value_generations));
  memset(root->value_kinds, 0,
         root->domain.value_count * sizeof(*root->value_kinds));
  memset(root->write_scope_generations, 0,
         root->domain.value_count * sizeof(*root->write_scope_generations));
  return iree_ok_status();
}

static uint32_t loom_cmd_launch_count_next_generation(
    loom_cmd_launch_count_root_t* root) {
  if (++root->next_generation == 0) {
    memset(root->value_generations, 0,
           root->domain.value_count * sizeof(*root->value_generations));
    memset(root->write_scope_generations, 0,
           root->domain.value_count * sizeof(*root->write_scope_generations));
    root->next_generation = 1;
  }
  return root->next_generation;
}

static iree_string_view_t loom_cmd_launch_count_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_string_id_t name_id =
      module->symbols.entries[symbol_ref.symbol_id].name_id;
  IREE_ASSERT_LT(name_id, module->strings.count);
  return loom_string_table_get(&module->strings, name_id);
}

static iree_status_t loom_cmd_launch_count_emit_range_error(
    const loom_module_t* module, const loom_op_t* dispatch_op,
    uint32_t dimension, iree_string_view_t requirement,
    iree_diagnostic_emitter_t emitter) {
  const loom_symbol_ref_t callee = loom_kernel_dispatch_callee(dispatch_op);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_cmd_launch_count_symbol_name(module, callee)),
      loom_param_with_field_ref(
          loom_param_u32(dimension),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, dimension)),
      loom_param_string(requirement),
  };
  const loom_diagnostic_emission_t emission = {
      .module = module,
      .op = dispatch_op,
      .error = LOOM_ERR_LOWERING_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_cmd_launch_count_append_value(
    iree_arena_allocator_t* arena, loom_value_id_t value,
    loom_value_id_t** values, iree_host_size_t* count,
    iree_host_size_t* capacity) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, *count, *count + 1, sizeof(**values), capacity, (void**)values));
  (*values)[(*count)++] = value;
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_append_operation(
    loom_cmd_launch_count_closure_t* closure, loom_op_t* operation) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      closure->arena, closure->operation_count, closure->operation_count + 1,
      sizeof(*closure->operations), &closure->operation_capacity,
      (void**)&closure->operations));
  closure->operations[closure->operation_count++] = operation;
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_set_value_state(
    loom_cmd_launch_count_closure_t* closure, loom_value_ordinal_t ordinal,
    loom_cmd_launch_count_value_kind_t kind) {
  if (closure->record_value_changes) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        closure->arena, closure->value_change_count,
        closure->value_change_count + 1, sizeof(*closure->value_changes),
        &closure->value_change_capacity, (void**)&closure->value_changes));
    closure->value_changes[closure->value_change_count++] =
        (loom_cmd_launch_count_value_change_t){
            .ordinal = ordinal,
            .generation = closure->value_generations[ordinal],
            .kind = closure->value_kinds[ordinal],
        };
  }
  closure->value_generations[ordinal] = closure->generation;
  closure->value_kinds[ordinal] = (uint8_t)kind;
  return iree_ok_status();
}

// Returns true when |operation| projects an existing reference without creating
// a new storage root. These operations must be cloned into a generated count
// producer so local views and their refinement facts remain local instead of
// leaking into its ABI as independently bindable arguments.
static bool loom_cmd_launch_count_is_reference_projection(
    const loom_op_t* operation) {
  return loom_buffer_assume_alignment_isa(operation) ||
         loom_buffer_assume_memory_space_isa(operation) ||
         loom_buffer_assume_noalias_isa(operation) ||
         loom_buffer_assume_same_root_isa(operation) ||
         loom_buffer_view_isa(operation) || loom_view_subview_isa(operation) ||
         loom_view_refine_isa(operation) || loom_view_bitcast_isa(operation);
}

static iree_status_t loom_cmd_launch_count_visit_value(
    loom_cmd_launch_count_closure_t* closure, loom_value_id_t value_id);

static iree_status_t loom_cmd_launch_count_visit_value_callback(
    void* user_data, loom_value_id_t value_id) {
  return loom_cmd_launch_count_visit_value(
      (loom_cmd_launch_count_closure_t*)user_data, value_id);
}

static iree_status_t loom_cmd_launch_count_visit_attribute_uses(
    loom_cmd_launch_count_closure_t* closure, const loom_op_t* operation) {
  const uint32_t* attribute_owners = loom_op_attribute_owners(operation);
  for (uint8_t i = 0; i < operation->attribute_count; ++i) {
    if (attribute_owners[i] == 0) {
      continue;
    }
    loom_type_use_iterator_t dependencies;
    loom_attribute_dependencies_begin(&closure->module->type_uses, operation, i,
                                      &dependencies);
    for (loom_value_id_t dependency =
             loom_type_dependencies_next(&dependencies);
         dependency != LOOM_VALUE_ID_INVALID;
         dependency = loom_type_dependencies_next(&dependencies)) {
      IREE_RETURN_IF_ERROR(
          loom_cmd_launch_count_visit_value(closure, dependency));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_visit_operation(
    loom_cmd_launch_count_closure_t* closure, loom_op_t* operation) {
  if (closure->invalid_observation != NULL) {
    return iree_ok_status();
  }
  if (operation->result_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "launch-count value has a resultless producer");
  }
  const loom_value_id_t state_value = loom_op_const_results(operation)[0];
  const loom_value_ordinal_t state_ordinal =
      loom_local_value_domain_ordinal(closure->domain, state_value);
  if (closure->value_generations[state_ordinal] == closure->generation) {
    const uint8_t state = closure->value_kinds[state_ordinal];
    if (state == LOOM_CMD_LAUNCH_COUNT_VALUE_VISITING) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "launch-count dataflow contains a cycle");
    }
    if (state == LOOM_CMD_LAUNCH_COUNT_VALUE_PRODUCED) {
      return iree_ok_status();
    }
  }

  const loom_trait_flags_t traits =
      loom_op_effective_traits(closure->module, operation);
  if (loom_traits_has_side_effects(traits) ||
      loom_traits_has_unique_identity(traits) ||
      iree_any_bit_set(traits, LOOM_TRAIT_COMMAND_EFFECT)) {
    const iree_string_view_t operation_name =
        loom_op_name(closure->module, operation);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch-count device realization cannot capture effectful operation "
        "'%.*s'",
        (int)operation_name.size, operation_name.data);
  }

  if (iree_any_bit_set(traits, LOOM_TRAIT_READS_MEMORY)) {
    bool can_move = false;
    const bool ordinary =
        loom_motion_op_is_ordinary_load(closure->module, operation);
    if (ordinary) {
      if (*closure->read_barriers == NULL) {
        IREE_RETURN_IF_ERROR(loom_motion_read_barrier_table_allocate(
            closure->module, closure->analysis_arena, closure->read_barriers));
      }
      IREE_RETURN_IF_ERROR(loom_motion_read_barrier_table_can_cross(
          *closure->read_barriers, operation, closure->consumer, &can_move));
    }
    if (!ordinary) {
      closure->invalid_observation = operation;
      return iree_ok_status();
    }
    closure->needs_source_placement |= !can_move;
    const loom_memory_access_t access =
        loom_memory_access_cast(closure->module, operation);
    IREE_ASSERT(loom_memory_access_isa(access));
    IREE_ASSERT_EQ(loom_memory_access_operation_kind(access),
                   LOOM_MEMORY_ACCESS_OPERATION_LOAD);
    const loom_value_id_t reference = loom_memory_access_view(access);
    IREE_ASSERT_NE(reference, LOOM_VALUE_ID_INVALID);
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        closure->arena, closure->read_reference_count,
        closure->read_reference_count + 1, sizeof(*closure->read_references),
        &closure->read_reference_capacity, (void**)&closure->read_references));
    closure->read_references[closure->read_reference_count++] =
        (loom_cmd_launch_count_read_t){
            .operation = operation,
            .reference = reference,
        };
  }

  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_set_value_state(
      closure, state_ordinal, LOOM_CMD_LAUNCH_COUNT_VALUE_VISITING));
  IREE_RETURN_IF_ERROR(loom_liveness_for_each_op_use(
      closure->module, operation,
      loom_liveness_value_callback_make(
          loom_cmd_launch_count_visit_value_callback, closure)));
  IREE_RETURN_IF_ERROR(
      loom_cmd_launch_count_visit_attribute_uses(closure, operation));
  IREE_RETURN_IF_ERROR(
      loom_cmd_launch_count_append_operation(closure, operation));

  const loom_value_id_t* results = loom_op_const_results(operation);
  for (uint16_t i = 0; i < operation->result_count; ++i) {
    const loom_value_ordinal_t result_ordinal =
        loom_local_value_domain_ordinal(closure->domain, results[i]);
    IREE_RETURN_IF_ERROR(loom_cmd_launch_count_set_value_state(
        closure, result_ordinal, LOOM_CMD_LAUNCH_COUNT_VALUE_PRODUCED));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_visit_value(
    loom_cmd_launch_count_closure_t* closure, loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(closure->domain, value_id);
  if (closure->value_generations[ordinal] == closure->generation) {
    const uint8_t kind = closure->value_kinds[ordinal];
    if (kind == LOOM_CMD_LAUNCH_COUNT_VALUE_VISITING) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "launch-count dataflow contains a cycle");
    }
    return iree_ok_status();
  }

  const loom_type_t type = loom_module_value_type(closure->module, value_id);
  const loom_value_t* value = loom_module_value(closure->module, value_id);
  loom_op_t* defining_op =
      loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
  if (defining_op != NULL && defining_op->result_count > 1) {
    return loom_cmd_launch_count_visit_operation(closure, defining_op);
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(closure->facts, value_id);
  if (loom_value_facts_can_materialize_constant(facts, type)) {
    IREE_RETURN_IF_ERROR(loom_cmd_launch_count_set_value_state(
        closure, ordinal, LOOM_CMD_LAUNCH_COUNT_VALUE_CONSTANT));
    return loom_cmd_launch_count_append_value(
        closure->arena, value_id, &closure->constants, &closure->constant_count,
        &closure->constant_capacity);
  }

  if (defining_op != NULL &&
      loom_cmd_launch_count_is_reference_projection(defining_op)) {
    return loom_cmd_launch_count_visit_operation(closure, defining_op);
  }

  if (loom_type_is_buffer(type) ||
      (loom_type_is_view(type) && !loom_type_may_reference_values(type))) {
    IREE_RETURN_IF_ERROR(loom_cmd_launch_count_set_value_state(
        closure, ordinal, LOOM_CMD_LAUNCH_COUNT_VALUE_ARGUMENT));
    return loom_cmd_launch_count_append_value(
        closure->arena, value_id, &closure->arguments, &closure->argument_count,
        &closure->argument_capacity);
  }

  if (loom_value_is_block_arg(value)) {
    if (loom_type_is_view(type)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "launch-count device realization cannot bind a dependent view "
          "block argument without its signature dependencies");
    }
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch-count device realization requires non-exact scalar block "
        "arguments to be resolved before command lowering");
  }
  if (defining_op == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch-count dataflow contains an unowned declaration value");
  }
  return loom_cmd_launch_count_visit_operation(closure, defining_op);
}

static iree_status_t loom_cmd_launch_count_collect_dispatches(
    loom_region_t* region, loom_cmd_launch_count_dispatch_list_t* list) {
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* operation = NULL;
    loom_block_for_each_op(block, operation) {
      if (loom_kernel_dispatch_isa(operation)) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            list->arena, list->count, list->count + 1, sizeof(*list->values),
            &list->capacity, (void**)&list->values));
        list->values[list->count++] = operation;
      }
      loom_region_t* const* regions = loom_op_regions(operation);
      for (uint8_t i = 0; i < operation->region_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_cmd_launch_count_collect_dispatches(regions[i], list));
      }
    }
  }
  return iree_ok_status();
}

static bool loom_cmd_launch_count_symbol_refs_equal(loom_symbol_ref_t left,
                                                    loom_symbol_ref_t right) {
  return left.module_id == right.module_id && left.symbol_id == right.symbol_id;
}

static loom_symbol_ref_t loom_cmd_launch_count_dispatch_target(
    const loom_module_t* module, const loom_op_t* dispatch) {
  const loom_symbol_ref_t consumer_ref = loom_kernel_dispatch_callee(dispatch);
  IREE_ASSERT(loom_symbol_ref_is_valid(consumer_ref));
  IREE_ASSERT_EQ(consumer_ref.module_id, 0u);
  IREE_ASSERT_LT(consumer_ref.symbol_id, module->symbols.count);
  const loom_op_t* consumer_entry =
      module->symbols.entries[consumer_ref.symbol_id].defining_op;
  IREE_ASSERT(loom_kernel_entry_decl_isa(consumer_entry));
  return loom_kernel_entry_decl_target(consumer_entry);
}

static bool loom_cmd_launch_count_has_serial_parent(const loom_op_t* dispatch) {
  const loom_op_t* parent = dispatch->parent_op;
  return parent != NULL &&
         (loom_command_def_isa(parent) || loom_command_serial_isa(parent) ||
          loom_kernel_launch_serial_isa(parent));
}

static bool loom_cmd_launch_count_can_cross_operations(
    const loom_module_t* module, const loom_op_t* left,
    const loom_op_t* right) {
  if (left->parent_block != right->parent_block) {
    return false;
  }
  const loom_trait_flags_t forbidden_traits =
      LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_NON_DETERMINISTIC |
      LOOM_TRAIT_UNKNOWN_EFFECTS | LOOM_TRAIT_UNIQUE_IDENTITY |
      LOOM_TRAIT_CONVERGENT | LOOM_TRAIT_MEMORY_FENCE |
      LOOM_TRAIT_OBSERVABLE_EFFECT | LOOM_TRAIT_COMMAND_EFFECT;
  for (const loom_op_t* operation = left->next_op; operation != NULL;
       operation = operation->next_op) {
    if (operation == right) {
      return true;
    }
    if (operation->flags & LOOM_OP_FLAG_DEAD) {
      continue;
    }
    if (operation->region_count != 0 ||
        iree_any_bit_set(loom_op_effective_traits(module, operation),
                         forbidden_traits)) {
      return false;
    }
  }
  return false;
}

static bool loom_cmd_launch_count_reference_alias_scope(
    const loom_value_fact_table_t* facts, loom_value_id_t reference,
    loom_value_fact_alias_scope_id_t* out_alias_scope) {
  *out_alias_scope = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
  const loom_value_facts_t reference_facts =
      loom_value_fact_table_lookup(facts, reference);
  loom_value_fact_buffer_reference_t buffer_reference = {0};
  if (loom_value_facts_query_buffer_reference(&facts->context, reference_facts,
                                              &buffer_reference)) {
    *out_alias_scope = buffer_reference.alias_scope_id;
  } else {
    loom_value_fact_view_reference_t view_reference = {0};
    if (!loom_value_facts_query_view_reference(&facts->context, reference_facts,
                                               &view_reference)) {
      return false;
    }
    *out_alias_scope = view_reference.alias_scope_id;
  }
  return *out_alias_scope != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
}

static uint64_t loom_cmd_launch_count_tuple_hash(
    const loom_cmd_launch_count_tuple_t* tuple) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (uint8_t i = 0; i < 3; ++i) {
    hash ^= tuple->values[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static bool loom_cmd_launch_count_tuples_equal(
    const loom_cmd_launch_count_tuple_t* left,
    const loom_cmd_launch_count_tuple_t* right) {
  return left->values[0] == right->values[0] &&
         left->values[1] == right->values[1] &&
         left->values[2] == right->values[2];
}

static void loom_cmd_launch_count_tuple_table_insert(
    loom_cmd_launch_count_cluster_t* cluster, iree_host_size_t tuple_ordinal) {
  IREE_ASSERT_LT(tuple_ordinal, cluster->tuple_count);
  const iree_host_size_t mask = cluster->tuple_bucket_capacity - 1;
  iree_host_size_t bucket =
      (iree_host_size_t)(loom_cmd_launch_count_tuple_hash(
                             &cluster->tuples[tuple_ordinal]) &
                         mask);
  while (cluster->tuple_buckets[bucket] != 0) {
    bucket = (bucket + 1) & mask;
  }
  cluster->tuple_buckets[bucket] = tuple_ordinal + 1;
}

static iree_status_t loom_cmd_launch_count_tuple_table_grow(
    loom_cmd_launch_count_cluster_t* cluster) {
  iree_host_size_t new_capacity = 8;
  if (cluster->tuple_bucket_capacity != 0 &&
      !iree_host_size_checked_mul(cluster->tuple_bucket_capacity, 2,
                                  &new_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "launch-count tuple table is too large");
  }
  iree_host_size_t* new_buckets = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(cluster->closure.arena, new_capacity,
                                sizeof(*new_buckets), (void**)&new_buckets));
  memset(new_buckets, 0, new_capacity * sizeof(*new_buckets));
  cluster->tuple_buckets = new_buckets;
  cluster->tuple_bucket_capacity = new_capacity;
  for (iree_host_size_t i = 0; i < cluster->tuple_count; ++i) {
    loom_cmd_launch_count_tuple_table_insert(cluster, i);
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_intern_tuple(
    loom_cmd_launch_count_cluster_t* cluster, loom_value_slice_t counts,
    iree_host_size_t* out_tuple_ordinal) {
  loom_cmd_launch_count_tuple_t tuple = {
      .values =
          {
              LOOM_VALUE_ID_INVALID,
              LOOM_VALUE_ID_INVALID,
              LOOM_VALUE_ID_INVALID,
          },
  };
  memcpy(tuple.values, counts.values, counts.count * sizeof(*tuple.values));

  if (cluster->tuple_bucket_capacity == 0 ||
      cluster->tuple_count + 1 > cluster->tuple_bucket_capacity / 2) {
    IREE_RETURN_IF_ERROR(loom_cmd_launch_count_tuple_table_grow(cluster));
  }
  const iree_host_size_t mask = cluster->tuple_bucket_capacity - 1;
  iree_host_size_t bucket =
      (iree_host_size_t)(loom_cmd_launch_count_tuple_hash(&tuple) & mask);
  while (cluster->tuple_buckets[bucket] != 0) {
    const iree_host_size_t tuple_ordinal = cluster->tuple_buckets[bucket] - 1;
    if (loom_cmd_launch_count_tuples_equal(&cluster->tuples[tuple_ordinal],
                                           &tuple)) {
      *out_tuple_ordinal = tuple_ordinal;
      return iree_ok_status();
    }
    bucket = (bucket + 1) & mask;
  }

  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      cluster->closure.arena, cluster->tuple_count, cluster->tuple_count + 1,
      sizeof(*cluster->tuples), &cluster->tuple_capacity,
      (void**)&cluster->tuples));
  const iree_host_size_t tuple_ordinal = cluster->tuple_count++;
  cluster->tuples[tuple_ordinal] = tuple;
  cluster->tuple_buckets[bucket] = tuple_ordinal + 1;
  *out_tuple_ordinal = tuple_ordinal;
  return iree_ok_status();
}

static void loom_cmd_launch_count_cluster_initialize(
    loom_module_t* module, loom_cmd_launch_count_root_t* root,
    const loom_value_fact_table_t* facts, loom_op_t* first_consumer,
    loom_symbol_ref_t target, iree_arena_allocator_t* scratch_arena,
    loom_cmd_launch_count_cluster_t* out_cluster) {
  const uint32_t generation = loom_cmd_launch_count_next_generation(root);
  *out_cluster = (loom_cmd_launch_count_cluster_t){
      .target = target,
      .block = first_consumer->parent_block,
      .first_consumer = first_consumer,
      .producer_before = first_consumer,
      .frontier = first_consumer,
      .closure =
          {
              .module = module,
              .facts = facts,
              .domain = &root->domain,
              .generation = generation,
              .value_generations = root->value_generations,
              .value_kinds = root->value_kinds,
              .arena = scratch_arena,
              .read_barriers = &root->read_barriers,
              .analysis_arena = root->analysis_arena,
          },
  };
}

static bool loom_cmd_launch_count_new_arguments_available_before_cluster(
    const loom_cmd_launch_count_cluster_t* cluster,
    iree_host_size_t first_new_argument,
    const loom_dominance_info_t* dominance) {
  const loom_cmd_launch_count_closure_t* closure = &cluster->closure;
  for (iree_host_size_t i = first_new_argument; i < closure->argument_count;
       ++i) {
    if (!loom_value_is_available_before_op(dominance, closure->arguments[i],
                                           cluster->first_consumer)) {
      return false;
    }
  }
  return true;
}

static bool loom_cmd_launch_count_new_reads_conflict_with_writes(
    const loom_cmd_launch_count_root_t* root,
    const loom_cmd_launch_count_cluster_t* cluster,
    iree_host_size_t first_new_read) {
  if (!cluster->has_possible_writes) {
    return false;
  }
  const loom_cmd_launch_count_closure_t* closure = &cluster->closure;
  for (iree_host_size_t i = first_new_read; i < closure->read_reference_count;
       ++i) {
    loom_value_fact_alias_scope_id_t alias_scope =
        LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
    if (!loom_cmd_launch_count_reference_alias_scope(
            closure->facts, closure->read_references[i].reference,
            &alias_scope)) {
      return true;
    }
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_try_ordinal(&root->domain, alias_scope);
    if (ordinal == LOOM_VALUE_ORDINAL_INVALID ||
        root->write_scope_generations[ordinal] == cluster->closure.generation) {
      return true;
    }
  }
  return false;
}

static void loom_cmd_launch_count_cluster_record_possible_writes(
    loom_cmd_launch_count_root_t* root,
    loom_cmd_launch_count_cluster_t* cluster, const loom_op_t* dispatch) {
  const loom_value_slice_t arguments = loom_kernel_dispatch_arguments(dispatch);
  for (uint16_t i = 0; i < arguments.count; ++i) {
    const loom_value_id_t argument = arguments.values[i];
    const loom_type_t type =
        loom_module_value_type(cluster->closure.module, argument);
    if (!loom_type_is_buffer(type) && !loom_type_is_view(type)) {
      continue;
    }
    cluster->has_possible_writes = true;
    loom_value_fact_alias_scope_id_t alias_scope =
        LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
    if (!loom_cmd_launch_count_reference_alias_scope(cluster->closure.facts,
                                                     argument, &alias_scope)) {
      cluster->closed = true;
      continue;
    }
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_try_ordinal(&root->domain, alias_scope);
    if (ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      cluster->closed = true;
      continue;
    }
    root->write_scope_generations[ordinal] = cluster->closure.generation;
  }
}

static void loom_cmd_launch_count_transaction_begin(
    loom_cmd_launch_count_cluster_t* cluster,
    iree_arena_allocator_t* scratch_arena,
    loom_cmd_launch_count_transaction_t* out_transaction) {
  *out_transaction = (loom_cmd_launch_count_transaction_t){
      .checkpoint = iree_arena_checkpoint_save(scratch_arena),
      .prior_closure = cluster->closure,
  };
  cluster->closure.record_value_changes = true;
  cluster->closure.value_changes = NULL;
  cluster->closure.value_change_count = 0;
  cluster->closure.value_change_capacity = 0;
}

static void loom_cmd_launch_count_transaction_commit(
    loom_cmd_launch_count_cluster_t* cluster) {
  cluster->closure.record_value_changes = false;
  cluster->closure.value_changes = NULL;
  cluster->closure.value_change_count = 0;
  cluster->closure.value_change_capacity = 0;
}

static void loom_cmd_launch_count_transaction_rollback(
    loom_cmd_launch_count_cluster_t* cluster,
    const loom_cmd_launch_count_transaction_t* transaction) {
  loom_cmd_launch_count_closure_t* closure = &cluster->closure;
  for (iree_host_size_t i = closure->value_change_count; i > 0; --i) {
    const loom_cmd_launch_count_value_change_t* change =
        &closure->value_changes[i - 1];
    closure->value_generations[change->ordinal] = change->generation;
    closure->value_kinds[change->ordinal] = change->kind;
  }
  iree_arena_checkpoint_restore(&transaction->checkpoint);
  cluster->closure = transaction->prior_closure;
}

static iree_status_t loom_cmd_launch_count_cluster_visit_counts(
    loom_cmd_launch_count_cluster_t* cluster, const loom_op_t* dispatch,
    const loom_dominance_info_t* dominance,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid) {
  cluster->closure.consumer = dispatch;
  const loom_value_slice_t counts =
      loom_kernel_dispatch_workgroup_counts(dispatch);
  for (uint16_t i = 0;
       i < counts.count && cluster->closure.invalid_observation == NULL; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_cmd_launch_count_visit_value(&cluster->closure, counts.values[i]));
  }
  loom_cmd_launch_count_closure_t* closure = &cluster->closure;
  if (closure->invalid_observation == NULL && closure->needs_source_placement &&
      cluster->demand_count == 0) {
    // A scalar sampled before a write stays at that observation. Such a
    // producer cannot join a later configuration cluster across the write.
    const loom_op_t* last_read = closure->read_references[0].operation;
    for (iree_host_size_t i = 1; i < closure->read_reference_count; ++i) {
      const loom_op_t* read = closure->read_references[i].operation;
      if (read->parent_block != last_read->parent_block) {
        closure->invalid_observation = read;
      } else if (read->block_ordinal > last_read->block_ordinal) {
        last_read = read;
      }
    }
    loom_op_t* before = last_read->next_op;
    if (!loom_cmd_launch_count_has_serial_parent(last_read)) {
      closure->invalid_observation = last_read;
    }
    for (iree_host_size_t i = 0; i < closure->read_reference_count &&
                                 closure->invalid_observation == NULL;
         ++i) {
      const loom_op_t* read = closure->read_references[i].operation;
      bool can_move = false;
      IREE_RETURN_IF_ERROR(loom_motion_read_barrier_table_can_cross(
          *closure->read_barriers, read, before, &can_move));
      if (!can_move) {
        closure->invalid_observation = read;
      }
    }
    for (iree_host_size_t i = 0;
         i < closure->argument_count && closure->invalid_observation == NULL;
         ++i) {
      if (!loom_value_is_available_before_op(dominance, closure->arguments[i],
                                             before)) {
        closure->invalid_observation = last_read;
      }
    }
    cluster->producer_before = before;
    cluster->closed = true;
  }
  if (cluster->closure.invalid_observation != NULL) {
    *out_valid = false;
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_cmd_launch_count_symbol_name(
            cluster->closure.module, loom_kernel_dispatch_callee(dispatch))),
    };
    const loom_diagnostic_emission_t emission = {
        .module = cluster->closure.module,
        .op = cluster->closure.invalid_observation,
        .error = LOOM_ERR_LOWERING_068,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(diagnostic_emitter, &emission);
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_cluster_commit_demand(
    loom_cmd_launch_count_root_t* root,
    loom_cmd_launch_count_cluster_t* cluster, loom_op_t* dispatch) {
  const loom_value_slice_t counts =
      loom_kernel_dispatch_workgroup_counts(dispatch);
  iree_host_size_t tuple_ordinal = 0;
  IREE_RETURN_IF_ERROR(
      loom_cmd_launch_count_intern_tuple(cluster, counts, &tuple_ordinal));
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      cluster->closure.arena, cluster->demand_count, cluster->demand_count + 1,
      sizeof(*cluster->demands), &cluster->demand_capacity,
      (void**)&cluster->demands));
  cluster->demands[cluster->demand_count++] = (loom_cmd_launch_count_demand_t){
      .consumer = dispatch,
      .tuple_ordinal = tuple_ordinal,
  };
  cluster->frontier = dispatch;
  loom_cmd_launch_count_cluster_record_possible_writes(root, cluster, dispatch);
  if (!loom_cmd_launch_count_has_serial_parent(dispatch)) {
    cluster->closed = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_cluster_add_demand(
    loom_cmd_launch_count_root_t* root,
    loom_cmd_launch_count_cluster_t* cluster, loom_op_t* dispatch,
    const loom_dominance_info_t* dominance,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid) {
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_cluster_visit_counts(
      cluster, dispatch, dominance, diagnostic_emitter, out_valid));
  return *out_valid ? loom_cmd_launch_count_cluster_commit_demand(root, cluster,
                                                                  dispatch)
                    : iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_reserve_symbol(
    loom_module_t* module, uint32_t* next_name, loom_symbol_ref_t* out_symbol) {
  *out_symbol = loom_symbol_ref_null();
  char text[64];
  while (true) {
    iree_snprintf(text, sizeof(text), "__command_launch_counts_%u",
                  (*next_name)++);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, iree_make_cstring_view(text), &name_id));
    if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_symbol = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    return iree_ok_status();
  }
}

static iree_status_t loom_cmd_launch_count_build_index_constant(
    loom_builder_t* builder, int64_t value, loom_scalar_type_t scalar_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, loom_attr_i64(value),
                                                 loom_type_scalar(scalar_type),
                                                 location, &constant_op));
  *out_value = loom_index_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_build_kernel(
    loom_module_t* module, const loom_cmd_launch_count_plan_t* plan,
    loom_symbol_ref_t kernel_ref, const loom_value_fact_table_t* facts,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* scratch_arena,
    loom_rewriter_t* rewriter, loom_type_t count_view_type,
    loom_op_t** out_kernel_op, loom_kernel_class_product_t* out_product) {
  *out_kernel_op = NULL;
  *out_product = (loom_kernel_class_product_t){0};

  const loom_location_id_t location = plan->first_consumer->location;

  loom_type_t* argument_types = NULL;
  const iree_host_size_t argument_count = plan->argument_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, argument_count,
                                                 sizeof(*argument_types),
                                                 (void**)&argument_types));
  for (iree_host_size_t i = 0; i < plan->argument_count; ++i) {
    argument_types[i] = loom_module_value_type(module, plan->arguments[i]);
  }
  argument_types[plan->argument_count] = count_view_type;

  loom_kernel_def_build_flags_t build_flags = 0;
  if (loom_symbol_ref_is_valid(plan->target)) {
    build_flags |= LOOM_KERNEL_DEF_BUILD_FLAG_HAS_TARGET;
  }
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_block(builder, loom_module_block(module));
  loom_op_t* kernel_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_def_build(
      builder, build_flags, /*linkage=*/0, /*retain=*/0, plan->target,
      LOOM_STRING_ID_INVALID,
      /*export_linkage=*/0, kernel_ref, /*config_arg_types=*/NULL,
      /*config_arg_types_count=*/0, argument_types, argument_count,
      /*predicates=*/NULL, /*predicates_count=*/0, location, &kernel_op));
  const loom_func_like_t kernel = loom_func_like_cast(module, kernel_op);
  uint16_t kernel_argument_count = 0;
  const loom_value_id_t* kernel_arguments =
      loom_func_like_arg_ids(kernel, &kernel_argument_count);
  IREE_ASSERT_EQ(kernel_argument_count, argument_count);
  for (iree_host_size_t i = 0; i < plan->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(module, plan->arguments[i],
                                                     kernel_arguments[i]));
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      builder, kernel_op, loom_kernel_def_config(kernel_op));
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_build_index_constant(
      builder, 1, LOOM_SCALAR_TYPE_INDEX, location, &one));
  loom_op_t* config_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_build(
      builder, /*build_flags=*/0, one, one, one, one, one, one,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      location, &config_op));
  loom_builder_restore(builder, saved);

  saved = loom_builder_enter_region(builder, kernel_op,
                                    loom_kernel_def_body(kernel_op));
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, module, scratch_arena,
                                                /*options=*/NULL, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
      &remap, plan->arguments, kernel_arguments, plan->argument_count));
  for (iree_host_size_t i = 0; i < plan->constant_count; ++i) {
    const loom_value_id_t source = plan->constants[i];
    loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_constant_build(
        builder, loom_value_fact_table_lookup(facts, source),
        loom_module_value_type(module, source), location, &target_value));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(&remap, source, target_value));
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, source, target_value));
  }
  for (iree_host_size_t i = 0; i < plan->operation_count; ++i) {
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op(builder, plan->operations[i], &remap, &cloned_op));
  }

  for (iree_host_size_t tuple_ordinal = 0; tuple_ordinal < plan->tuple_count;
       ++tuple_ordinal) {
    const loom_cmd_launch_count_tuple_t* tuple = &plan->tuples[tuple_ordinal];
    for (uint8_t dimension = 0; dimension < 3; ++dimension) {
      loom_value_id_t count_value = LOOM_VALUE_ID_INVALID;
      if (tuple->values[dimension] != LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
            &remap, tuple->values[dimension], &count_value));
      } else {
        IREE_RETURN_IF_ERROR(loom_cmd_launch_count_build_index_constant(
            builder, 1, LOOM_SCALAR_TYPE_INDEX, location, &count_value));
      }
      const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
      const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
      loom_op_t* cast_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_cast_build(
          builder, count_value, index_type, i32_type, location, &cast_op));
      count_value = loom_index_cast_result(cast_op);
      const int64_t static_index = (int64_t)(tuple_ordinal * 3 + dimension);
      loom_op_t* store_op = NULL;
      IREE_RETURN_IF_ERROR(loom_view_store_build(
          builder, /*build_flags=*/0, /*instance_flags=*/0, count_value,
          kernel_arguments[plan->argument_count],
          /*indices=*/NULL, /*indices_count=*/0, &static_index,
          /*static_indices_count=*/1, /*cache_scope=*/0,
          /*cache_temporal=*/0, location, &store_op));
    }
  }
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_return_build(builder, location, &return_op));
  loom_builder_restore(builder, saved);

  const iree_string_view_t kernel_name =
      loom_cmd_launch_count_symbol_name(module, kernel_ref);
  loom_symbol_ref_t product_kernel = loom_symbol_ref_null();
  const loom_module_t* source_modules[] = {module};
  loom_module_t* product_module = NULL;
  IREE_RETURN_IF_ERROR(loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules),
      &(loom_link_options_t){
          .module_name = IREE_SV("command_launch_counts"),
          .root_symbols = {.values = &kernel_name, .count = 1},
          .root_target_symbols = {.values = &product_kernel, .count = 1},
      },
      block_pool, module->allocator, &product_module));
  *out_kernel_op = kernel_op;
  *out_product = (loom_kernel_class_product_t){
      .module = product_module,
      .kernel = product_kernel,
  };
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_replace_kernel_with_entry(
    loom_module_t* module, loom_op_t* kernel_op, loom_symbol_ref_t kernel_ref,
    const loom_cmd_launch_count_plan_t* plan, loom_type_t count_view_type,
    loom_rewriter_t* rewriter) {
  const loom_symbol_ref_t target = loom_kernel_def_target(kernel_op);
  loom_kernel_entry_decl_build_flags_t build_flags = 0;
  if (loom_symbol_ref_is_valid(target)) {
    build_flags |= LOOM_KERNEL_ENTRY_DECL_BUILD_FLAG_HAS_TARGET;
  }
  loom_type_t* argument_types = NULL;
  const iree_host_size_t argument_count = plan->argument_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, argument_count, sizeof(*argument_types),
      (void**)&argument_types));
  for (iree_host_size_t i = 0; i < plan->argument_count; ++i) {
    argument_types[i] = loom_module_value_type(module, plan->arguments[i]);
  }
  argument_types[plan->argument_count] = count_view_type;

  loom_builder_set_before(&rewriter->builder, kernel_op);
  loom_op_t* entry_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_entry_decl_build(
      &rewriter->builder, build_flags, /*retain=*/0, target, kernel_ref,
      argument_types, argument_count, kernel_op->location, &entry_op));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, kernel_op));
  loom_module_link_symbol_defining_op(module, entry_op,
                                      loom_op_vtable(module, entry_op));
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_build_indirect_consumer(
    loom_builder_t* builder, loom_op_t* consumer, loom_value_id_t count_view,
    loom_op_t** out_dispatch) {
  const loom_value_slice_t consumer_arguments =
      loom_kernel_dispatch_arguments(consumer);
  const loom_value_slice_t workgroup_size =
      loom_kernel_dispatch_workgroup_size(consumer);
  return loom_kernel_dispatch_build(
      builder, loom_kernel_dispatch_callee(consumer), &count_view, 1,
      consumer_arguments.values, consumer_arguments.count,
      workgroup_size.values, workgroup_size.count, consumer->location,
      out_dispatch);
}

static iree_status_t loom_cmd_launch_count_build_tuple_view(
    loom_builder_t* builder, const loom_cmd_launch_count_plan_t* plan,
    const loom_cmd_launch_count_demand_t* demand, loom_value_id_t slab_view,
    loom_type_t tuple_view_type, loom_value_id_t* out_tuple_view) {
  if (plan->tuple_count == 1) {
    *out_tuple_view = slab_view;
    return iree_ok_status();
  }
  const int64_t static_offset = (int64_t)(demand->tuple_ordinal * 3);
  loom_op_t* subview_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_subview_build(
      builder, slab_view, /*offsets=*/NULL, /*offsets_count=*/0, &static_offset,
      /*static_offsets_count=*/1, &tuple_view_type, /*result_count=*/1,
      demand->consumer->location, &subview_op));
  *out_tuple_view = loom_view_subview_result(subview_op);
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_rewrite_cluster(
    loom_module_t* module, const loom_cmd_launch_count_plan_t* plan,
    loom_symbol_ref_t producer_ref, loom_type_t slab_view_type,
    int64_t slab_byte_length, loom_rewriter_t* rewriter) {
  loom_op_t* first_consumer = plan->first_consumer;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, plan->producer_before);

  loom_value_id_t count_bytes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_build_index_constant(
      builder, slab_byte_length, LOOM_SCALAR_TYPE_OFFSET,
      first_consumer->location, &count_bytes));
  loom_op_t* alloca_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      builder, /*build_flags=*/0, LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
      LOOM_VALUE_ID_INVALID, /*base_alignment=*/4, count_bytes,
      loom_type_buffer(), first_consumer->location, &alloca_op));
  loom_value_id_t base_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_build_index_constant(
      builder, 0, LOOM_SCALAR_TYPE_OFFSET, first_consumer->location,
      &base_offset));
  loom_op_t* count_view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      builder, loom_buffer_alloca_result(alloca_op), base_offset,
      slab_view_type, first_consumer->location, &count_view_op));
  const loom_value_id_t slab_view = loom_buffer_view_result(count_view_op);
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_build_index_constant(
      builder, 1, LOOM_SCALAR_TYPE_INDEX, first_consumer->location, &one));

  loom_value_id_t* producer_arguments = NULL;
  const iree_host_size_t producer_argument_count = plan->argument_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, producer_argument_count, sizeof(*producer_arguments),
      (void**)&producer_arguments));
  if (plan->argument_count != 0) {
    memcpy(producer_arguments, plan->arguments,
           plan->argument_count * sizeof(*producer_arguments));
  }
  producer_arguments[plan->argument_count] = slab_view;

  const loom_type_t tuple_view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(3),
      /*encoding_id=*/0);
  if (loom_cmd_launch_count_has_serial_parent(first_consumer) ||
      plan->producer_before != first_consumer) {
    loom_op_t* producer_dispatch = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_dispatch_build(
        builder, producer_ref, &one, 1, producer_arguments,
        producer_argument_count, /*workgroup_size=*/NULL,
        /*workgroup_size_count=*/0, first_consumer->location,
        &producer_dispatch));
    for (iree_host_size_t i = 0; i < plan->demand_count; ++i) {
      const loom_cmd_launch_count_demand_t* demand = &plan->demands[i];
      loom_builder_set_before(builder, demand->consumer);
      loom_value_id_t tuple_view = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_cmd_launch_count_build_tuple_view(
          builder, plan, demand, slab_view, tuple_view_type, &tuple_view));
      loom_op_t* indirect_consumer = NULL;
      IREE_RETURN_IF_ERROR(loom_cmd_launch_count_build_indirect_consumer(
          builder, demand->consumer, tuple_view, &indirect_consumer));
    }
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(plan->demand_count, 1u);
  IREE_ASSERT_EQ(plan->tuple_count, 1u);
  loom_op_t* serial_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_command_serial_build(builder, first_consumer->location, &serial_op));
  const loom_builder_ip_t saved = loom_builder_enter_region(
      builder, serial_op, loom_command_serial_body(serial_op));
  loom_op_t* producer_dispatch = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_dispatch_build(
      builder, producer_ref, &one, 1, producer_arguments,
      producer_argument_count, /*workgroup_size=*/NULL,
      /*workgroup_size_count=*/0, first_consumer->location,
      &producer_dispatch));
  loom_op_t* indirect_consumer = NULL;
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_build_indirect_consumer(
      builder, first_consumer, slab_view, &indirect_consumer));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_command_yield_build(builder, first_consumer->location, &yield_op));
  loom_builder_restore(builder, saved);
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_append_kernel(
    loom_cmd_launch_count_realization_t* realization,
    loom_cmd_launch_count_kernel_t kernel,
    iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      scratch_arena, realization->kernel_count, realization->kernel_count + 1,
      sizeof(*realization->kernels), &realization->kernel_capacity,
      (void**)&realization->kernels));
  realization->kernels[realization->kernel_count++] = kernel;
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_classify_dispatch(
    const loom_module_t* module, const loom_op_t* dispatch,
    const loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    bool* out_residual) {
  *out_residual = false;
  const loom_value_slice_t counts =
      loom_kernel_dispatch_workgroup_counts(dispatch);
  if (counts.count == 1 &&
      loom_type_is_view(loom_module_value_type(module, counts.values[0]))) {
    return iree_ok_status();
  }
  IREE_ASSERT_GE(counts.count, 1u);
  IREE_ASSERT_LE(counts.count, 3u);

  bool all_exact = true;
  for (uint16_t i = 0; i < counts.count; ++i) {
    const loom_value_facts_t count_facts =
        loom_value_fact_table_lookup(facts, counts.values[i]);
    const bool is_exact = loom_value_facts_is_exact(count_facts);
    if (!loom_value_facts_fit_unsigned_bit_count(count_facts, 32)) {
      *out_valid = false;
      return loom_cmd_launch_count_emit_range_error(
          module, dispatch, i,
          is_exact ? IREE_SV("within the unsigned 32-bit range")
                   : IREE_SV("an exact unsigned 32-bit value"),
          diagnostic_emitter);
    }
    all_exact &= is_exact;
  }
  if (all_exact) {
    return iree_ok_status();
  }
  *out_residual = true;
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_cluster_layout(
    const loom_cmd_launch_count_cluster_t* cluster,
    loom_type_t* out_slab_view_type, int64_t* out_slab_byte_length) {
  if (cluster->tuple_count == 0 ||
      cluster->tuple_count > (iree_host_size_t)(INT64_MAX / 12)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "launch-count cluster output is too large");
  }
  const int64_t element_count = (int64_t)cluster->tuple_count * 3;
  *out_slab_view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(element_count),
      /*encoding_id=*/0);
  *out_slab_byte_length = element_count * 4;
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_apply_plan(
    loom_module_t* module, const loom_cmd_launch_count_plan_t* plan,
    const loom_value_fact_table_t* facts, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* scratch_arena, loom_rewriter_t* rewriter,
    uint32_t* next_name, loom_cmd_launch_count_kernel_t* out_kernel) {
  *out_kernel = (loom_cmd_launch_count_kernel_t){0};
  loom_symbol_ref_t producer_ref = loom_symbol_ref_null();
  loom_op_t* producer_op = NULL;
  loom_kernel_class_product_t product = {0};
  iree_status_t status =
      loom_cmd_launch_count_reserve_symbol(module, next_name, &producer_ref);
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_count_build_kernel(
        module, plan, producer_ref, facts, block_pool, scratch_arena, rewriter,
        plan->slab_view_type, &producer_op, &product);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_count_replace_kernel_with_entry(
        module, producer_op, producer_ref, plan, plan->slab_view_type,
        rewriter);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_count_rewrite_cluster(
        module, plan, producer_ref, plan->slab_view_type,
        plan->slab_byte_length, rewriter);
  }
  if (iree_status_is_ok(status)) {
    *out_kernel = (loom_cmd_launch_count_kernel_t){
        .entry = producer_ref,
        .product = product,
    };
  } else {
    loom_kernel_class_product_deinitialize(&product);
  }
  return status;
}

static iree_status_t loom_cmd_launch_count_copy_array(
    iree_arena_allocator_t* arena, const void* source, iree_host_size_t count,
    iree_host_size_t element_size, void** out_copy) {
  *out_copy = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, element_size, out_copy));
  memcpy(*out_copy, source, count * element_size);
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_finish_cluster(
    const loom_cmd_launch_count_cluster_t* cluster,
    const iree_arena_checkpoint_t* checkpoint,
    iree_arena_allocator_t* plan_arena,
    loom_cmd_launch_count_plan_t*** plan_tail) {
  loom_cmd_launch_count_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan_arena, sizeof(*plan), (void**)&plan));
  *plan = (loom_cmd_launch_count_plan_t){
      .target = cluster->target,
      .first_consumer = cluster->first_consumer,
      .producer_before = cluster->producer_before,
      .constant_count = cluster->closure.constant_count,
      .argument_count = cluster->closure.argument_count,
      .operation_count = cluster->closure.operation_count,
      .demand_count = cluster->demand_count,
      .tuple_count = cluster->tuple_count,
  };
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_cluster_layout(
      cluster, &plan->slab_view_type, &plan->slab_byte_length));
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_copy_array(
      plan_arena, cluster->closure.constants, plan->constant_count,
      sizeof(*plan->constants), (void**)&plan->constants));
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_copy_array(
      plan_arena, cluster->closure.arguments, plan->argument_count,
      sizeof(*plan->arguments), (void**)&plan->arguments));
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_copy_array(
      plan_arena, cluster->closure.operations, plan->operation_count,
      sizeof(*plan->operations), (void**)&plan->operations));
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_copy_array(
      plan_arena, cluster->demands, plan->demand_count, sizeof(*plan->demands),
      (void**)&plan->demands));
  IREE_RETURN_IF_ERROR(loom_cmd_launch_count_copy_array(
      plan_arena, cluster->tuples, plan->tuple_count, sizeof(*plan->tuples),
      (void**)&plan->tuples));
  **plan_tail = plan;
  *plan_tail = &plan->next;
  iree_arena_checkpoint_restore(checkpoint);
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_count_flush_cluster(
    loom_cmd_launch_count_cluster_t* cluster,
    const iree_arena_checkpoint_t* checkpoint, bool* active,
    iree_arena_allocator_t* plan_arena,
    loom_cmd_launch_count_plan_t*** plan_tail) {
  if (!*active) {
    return iree_ok_status();
  }
  const iree_status_t status = loom_cmd_launch_count_finish_cluster(
      cluster, checkpoint, plan_arena, plan_tail);
  *active = false;
  return status;
}

static iree_status_t loom_cmd_launch_count_plan_root(
    loom_module_t* module, loom_op_t* root_op,
    const loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* plan_arena, bool* out_valid,
    loom_cmd_launch_count_plan_t*** plan_tail) {
  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(block_pool, &analysis_arena);
  loom_cmd_launch_count_root_t root = {
      .function = loom_func_like_cast(module, root_op),
      .analysis_arena = &analysis_arena,
  };
  iree_status_t status = iree_ok_status();
  loom_cmd_launch_count_dispatch_list_t dispatches = {
      .arena = scratch_arena,
  };
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_count_collect_dispatches(
        loom_func_like_body(root.function), &dispatches);
  }

  bool root_prepared = false;
  loom_dominance_info_t dominance = {0};
  bool cluster_active = false;
  iree_arena_checkpoint_t cluster_checkpoint = {0};
  loom_cmd_launch_count_cluster_t cluster = {0};
  for (iree_host_size_t i = 0;
       *out_valid && i < dispatches.count && iree_status_is_ok(status); ++i) {
    loom_op_t* dispatch = dispatches.values[i];
    bool residual = false;
    status = loom_cmd_launch_count_classify_dispatch(
        module, dispatch, facts, diagnostic_emitter, out_valid, &residual);
    if (!iree_status_is_ok(status) || !*out_valid) {
      break;
    }

    if (!residual) {
      if (!cluster_active) {
        continue;
      }
      const bool can_cross =
          !cluster.closed &&
          loom_cmd_launch_count_has_serial_parent(dispatch) &&
          dispatch->parent_block == cluster.block &&
          loom_cmd_launch_count_can_cross_operations(module, cluster.frontier,
                                                     dispatch);
      if (!can_cross) {
        status = loom_cmd_launch_count_flush_cluster(
            &cluster, &cluster_checkpoint, &cluster_active, plan_arena,
            plan_tail);
        continue;
      }
      cluster.frontier = dispatch;
      loom_cmd_launch_count_cluster_record_possible_writes(&root, &cluster,
                                                           dispatch);
      if (cluster.closed) {
        status = loom_cmd_launch_count_flush_cluster(
            &cluster, &cluster_checkpoint, &cluster_active, plan_arena,
            plan_tail);
      }
      continue;
    }

    if (!root_prepared) {
      status =
          loom_cmd_launch_count_root_prepare(module, &root, &analysis_arena);
      if (iree_status_is_ok(status)) {
        status = loom_dominance_info_initialize_region(
            module, loom_func_like_body(root.function), &analysis_arena,
            &dominance);
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      root_prepared = true;
    }

    const loom_symbol_ref_t target =
        loom_cmd_launch_count_dispatch_target(module, dispatch);
    bool can_join =
        cluster_active && !cluster.closed &&
        loom_cmd_launch_count_has_serial_parent(dispatch) &&
        dispatch->parent_block == cluster.block &&
        loom_cmd_launch_count_symbol_refs_equal(target, cluster.target) &&
        loom_cmd_launch_count_can_cross_operations(module, cluster.frontier,
                                                   dispatch);
    bool demand_committed = false;
    if (can_join) {
      loom_cmd_launch_count_transaction_t transaction = {0};
      loom_cmd_launch_count_transaction_begin(&cluster, scratch_arena,
                                              &transaction);
      status = loom_cmd_launch_count_cluster_visit_counts(
          &cluster, dispatch, &dominance, diagnostic_emitter, out_valid);
      if (iree_status_is_ok(status) && *out_valid) {
        can_join = !cluster.closure.needs_source_placement &&
                   loom_cmd_launch_count_new_arguments_available_before_cluster(
                       &cluster, transaction.prior_closure.argument_count,
                       &dominance) &&
                   !loom_cmd_launch_count_new_reads_conflict_with_writes(
                       &root, &cluster,
                       transaction.prior_closure.read_reference_count);
      }
      if (!iree_status_is_ok(status) || !*out_valid) {
        loom_cmd_launch_count_transaction_rollback(&cluster, &transaction);
        break;
      }
      if (can_join) {
        loom_cmd_launch_count_transaction_commit(&cluster);
        demand_committed = true;
        status = loom_cmd_launch_count_cluster_commit_demand(&root, &cluster,
                                                             dispatch);
      } else {
        loom_cmd_launch_count_transaction_rollback(&cluster, &transaction);
      }
    }

    if (iree_status_is_ok(status) && !can_join && cluster_active) {
      status = loom_cmd_launch_count_flush_cluster(
          &cluster, &cluster_checkpoint, &cluster_active, plan_arena,
          plan_tail);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    if (iree_status_is_ok(status) && !cluster_active) {
      cluster_checkpoint = iree_arena_checkpoint_save(scratch_arena);
      loom_cmd_launch_count_cluster_initialize(module, &root, facts, dispatch,
                                               target, scratch_arena, &cluster);
      cluster_active = true;
    }
    if (iree_status_is_ok(status) && !demand_committed) {
      status = loom_cmd_launch_count_cluster_add_demand(
          &root, &cluster, dispatch, &dominance, diagnostic_emitter, out_valid);
    }
    if (*out_valid && iree_status_is_ok(status) && cluster.closed) {
      status = loom_cmd_launch_count_flush_cluster(
          &cluster, &cluster_checkpoint, &cluster_active, plan_arena,
          plan_tail);
    }
  }
  if (*out_valid && iree_status_is_ok(status) && cluster_active) {
    status = loom_cmd_launch_count_flush_cluster(
        &cluster, &cluster_checkpoint, &cluster_active, plan_arena, plan_tail);
  } else if (cluster_active) {
    iree_arena_checkpoint_restore(&cluster_checkpoint);
    cluster_active = false;
  }
  if (loom_local_value_domain_is_acquired(&root.domain)) {
    loom_local_value_domain_release(&root.domain);
  }
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

iree_status_t loom_cmd_launch_count_realize(
    loom_module_t* module, loom_op_t* const* root_ops,
    iree_host_size_t root_count, const loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* scratch_arena,
    bool* out_valid, loom_cmd_launch_count_realization_t* out_realization) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(root_ops);
  IREE_ASSERT_GT(root_count, 0u);
  IREE_ASSERT_ARGUMENT(facts);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_valid);
  IREE_ASSERT_ARGUMENT(out_realization);
  *out_valid = false;
  *out_realization = (loom_cmd_launch_count_realization_t){0};

  const iree_host_size_t initial_symbol_count = module->symbols.count;
  uint8_t* visited_roots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, initial_symbol_count, sizeof(*visited_roots),
      (void**)&visited_roots));
  memset(visited_roots, 0, initial_symbol_count * sizeof(*visited_roots));

  iree_arena_allocator_t plan_arena;
  iree_arena_initialize(block_pool, &plan_arena);
  loom_cmd_launch_count_plan_t* plans = NULL;
  loom_cmd_launch_count_plan_t** plan_tail = &plans;
  bool valid = true;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       valid && i < root_count && iree_status_is_ok(status); ++i) {
    const loom_symbol_ref_t root_ref =
        loom_func_like_callee(loom_func_like_cast(module, root_ops[i]));
    IREE_ASSERT(loom_symbol_ref_is_valid(root_ref));
    IREE_ASSERT_EQ(root_ref.module_id, 0u);
    IREE_ASSERT_LT(root_ref.symbol_id, initial_symbol_count);
    if (visited_roots[root_ref.symbol_id]) {
      continue;
    }
    visited_roots[root_ref.symbol_id] = 1;
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(scratch_arena);
    status = loom_cmd_launch_count_plan_root(
        module, root_ops[i], facts, diagnostic_emitter, block_pool,
        scratch_arena, &plan_arena, &valid, &plan_tail);
    iree_arena_checkpoint_restore(&checkpoint);
  }

  // Every source observation and cluster is validated before the first rewrite.
  // Keep source operations and insertion anchors alive until every replacement
  // is built; an early producer may be anchored at another plan's consumer.
  if (valid && iree_status_is_ok(status) && plans != NULL) {
    loom_rewriter_t rewriter;
    loom_rewriter_initialize(&rewriter, module, scratch_arena);
    status = loom_rewriter_enable_worklist(&rewriter);
    uint32_t next_name = 0;
    for (const loom_cmd_launch_count_plan_t* plan = plans;
         plan != NULL && iree_status_is_ok(status); plan = plan->next) {
      loom_cmd_launch_count_kernel_t kernel = {0};
      status = loom_cmd_launch_count_apply_plan(module, plan, facts, block_pool,
                                                scratch_arena, &rewriter,
                                                &next_name, &kernel);
      if (iree_status_is_ok(status)) {
        status = loom_cmd_launch_count_append_kernel(out_realization, kernel,
                                                     scratch_arena);
      }
      if (!iree_status_is_ok(status)) {
        loom_kernel_class_product_deinitialize(&kernel.product);
      }
    }
    for (const loom_cmd_launch_count_plan_t* plan = plans;
         plan != NULL && iree_status_is_ok(status); plan = plan->next) {
      for (iree_host_size_t i = 0;
           i < plan->demand_count && iree_status_is_ok(status); ++i) {
        status = loom_rewriter_erase(&rewriter, plan->demands[i].consumer);
      }
    }
    loom_op_t* operation = NULL;
    while (iree_status_is_ok(status) &&
           (operation = loom_rewriter_pop(&rewriter)) != NULL) {
      bool erased = false;
      status = loom_rewriter_erase_if_dead(&rewriter, operation, &erased);
    }
    loom_rewriter_deinitialize(&rewriter);
  }
  iree_arena_deinitialize(&plan_arena);

  if (valid && iree_status_is_ok(status) &&
      out_realization->kernel_count != 0) {
    out_realization->symbol_count = module->symbols.count;
    status =
        iree_arena_allocate_array(scratch_arena, out_realization->symbol_count,
                                  sizeof(*out_realization->kernel_by_symbol),
                                  (void**)&out_realization->kernel_by_symbol);
    if (iree_status_is_ok(status)) {
      memset(out_realization->kernel_by_symbol, 0,
             out_realization->symbol_count *
                 sizeof(*out_realization->kernel_by_symbol));
      for (iree_host_size_t i = 0; i < out_realization->kernel_count; ++i) {
        const loom_cmd_launch_count_kernel_t* kernel =
            &out_realization->kernels[i];
        IREE_ASSERT_LT(kernel->entry.symbol_id, out_realization->symbol_count);
        out_realization->kernel_by_symbol[kernel->entry.symbol_id] = kernel;
      }
    }
  }
  if (!iree_status_is_ok(status) || !valid) {
    loom_cmd_launch_count_realization_deinitialize(out_realization);
  }
  if (iree_status_is_ok(status)) {
    *out_valid = valid;
  }
  return status;
}

void loom_cmd_launch_count_realization_deinitialize(
    loom_cmd_launch_count_realization_t* realization) {
  if (realization == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < realization->kernel_count; ++i) {
    loom_kernel_class_product_deinitialize(&realization->kernels[i].product);
  }
  *realization = (loom_cmd_launch_count_realization_t){0};
}
