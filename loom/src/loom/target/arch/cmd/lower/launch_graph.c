// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/launch_graph.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/canonicalize.h"
#include "loom/transforms/cleanup/cse.h"

typedef struct loom_cmd_launch_graph_producer_frame_t {
  // Source pure operation awaiting dependency materialization.
  const loom_op_t* op;
  // Next operand to inspect before cloning |op|.
  uint16_t next_operand;
} loom_cmd_launch_graph_producer_frame_t;

// Non-recursive pure-producer traversal stack.
typedef struct loom_cmd_launch_graph_producer_stack_t {
  // Active frames in dependency traversal order.
  loom_cmd_launch_graph_producer_frame_t* frames;
  // Number of active frames in |frames|.
  iree_host_size_t count;
  // Allocated entry capacity of |frames|.
  iree_host_size_t capacity;
} loom_cmd_launch_graph_producer_stack_t;

// Final placement shared by launches with the same exact configuration
// identity.
typedef struct loom_cmd_launch_graph_placement_t {
  // Placement selecting the populated payload member.
  loom_cmd_launch_count_kind_t kind;
  union {
    // Exact workgroup count when |kind| is DIRECT.
    loom_target_dispatch_workgroup_count_t direct;
    // Dense xyz tuple ordinal returned by the host function when HOST.
    uint32_t host_tuple_ordinal;
  } payload;
} loom_cmd_launch_graph_placement_t;

// One exact kernel launch-configuration identity.
typedef struct loom_cmd_launch_graph_identity_t {
  // Concrete kernel definition containing the configuration body.
  const loom_op_t* kernel_op;
  // Ordered source-program workload values bound to the configuration body.
  loom_value_slice_t source_workloads;
  // Structural hash populated when the identity set becomes table-backed.
  uint32_t hash;
  // First scheduled launch with this identity, used for diagnostics.
  iree_host_size_t first_launch_index;
  // Final direct or host placement populated after configuration evaluation.
  loom_cmd_launch_graph_placement_t placement;
} loom_cmd_launch_graph_identity_t;

enum {
  LOOM_CMD_LAUNCH_GRAPH_INLINE_IDENTITY_CAPACITY = 4,
  LOOM_CMD_LAUNCH_GRAPH_INLINE_RESULT_CAPACITY =
      LOOM_CMD_LAUNCH_GRAPH_INLINE_IDENTITY_CAPACITY *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT,
};

// Scratch identity plan formed while assigning scheduled launch rows.
typedef struct loom_cmd_launch_graph_identity_plan_t {
  // Unique identities in first scheduled occurrence order.
  loom_cmd_launch_graph_identity_t* identities;
  // Number of entries populated in |identities|.
  iree_host_size_t identity_count;
  // Allocated entry capacity of |identities|.
  iree_host_size_t identity_capacity;
  // Open-addressed scratch slots, initially mapping exact identities.
  iree_host_size_t* slot_ordinals;
  // Power-of-two number of entries in |slot_ordinals|.
  iree_host_size_t slot_capacity;
  // Identity ordinal for each scheduled launch.
  iree_host_size_t* launch_identity_ordinals;
  // Flattened xyz results for each unique identity.
  loom_value_id_t* result_values;
} loom_cmd_launch_graph_identity_plan_t;

// Uninitialized stack storage used before an identity plan needs arena growth.
typedef struct loom_cmd_launch_graph_inline_storage_t {
  // Inline identity storage covering common small command programs.
  loom_cmd_launch_graph_identity_t
      inline_identities[LOOM_CMD_LAUNCH_GRAPH_INLINE_IDENTITY_CAPACITY];
  // Inline launch assignments covering common small command programs.
  iree_host_size_t inline_launch_identity_ordinals
      [LOOM_CMD_LAUNCH_GRAPH_INLINE_IDENTITY_CAPACITY];
  // Inline flattened xyz results covering common small command programs.
  loom_value_id_t
      inline_result_values[LOOM_CMD_LAUNCH_GRAPH_INLINE_RESULT_CAPACITY];
} loom_cmd_launch_graph_inline_storage_t;

// Immutable source inputs consumed by launch extraction.
typedef struct loom_cmd_launch_graph_source_t {
  // Immutable verified source module.
  const loom_module_t* module;
  // Source command program being factored.
  loom_func_like_t program;
  // Existing command schedule defining launch order.
  const loom_cmd_schedule_plan_t* schedule;
  // Borrowed source facts populated by the owning program plan.
  const loom_value_fact_table_t* facts;
} loom_cmd_launch_graph_source_t;

// Mutable aggregate host IR under construction.
typedef struct loom_cmd_launch_graph_target_t {
  // Owned aggregate host module under construction.
  loom_module_t* module;
  // Aggregate host function under construction.
  loom_func_like_t function;
  // Builder positioned in the aggregate host function body.
  loom_builder_t builder;
  // Source program value to aggregate host value mapping.
  loom_ir_remap_t program_remap;
} loom_cmd_launch_graph_target_t;

// Product tables retained with the aggregate host module.
typedef struct loom_cmd_launch_graph_product_t {
  // Launch placement rows owned by the target module.
  loom_cmd_launch_count_t* launches;
  // Ordered wave rows owned by the target module.
  loom_cmd_schedule_wave_t* waves;
} loom_cmd_launch_graph_product_t;

typedef struct loom_cmd_launch_graph_build_t {
  // Immutable source inputs.
  loom_cmd_launch_graph_source_t source;
  // Mutable aggregate host IR.
  loom_cmd_launch_graph_target_t target;
  // Scratch storage discarded after extraction.
  iree_arena_allocator_t* scratch_arena;
  // Pure source-producer materialization state.
  loom_cmd_launch_graph_producer_stack_t producer_stack;
  // Exact configuration identities and per-launch assignments.
  loom_cmd_launch_graph_identity_plan_t identity_plan;
  // Retained launch and wave product tables.
  loom_cmd_launch_graph_product_t product;
} loom_cmd_launch_graph_build_t;

static iree_string_view_t loom_cmd_launch_graph_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static loom_op_t* loom_cmd_launch_graph_resolve_kernel(
    const loom_module_t* module, const loom_op_t* launch_op) {
  IREE_ASSERT(loom_kernel_launch_isa(launch_op));
  const loom_symbol_ref_t callee = loom_kernel_launch_callee(launch_op);
  IREE_ASSERT(loom_symbol_ref_is_valid(callee));
  IREE_ASSERT_EQ(callee.module_id, 0u);
  IREE_ASSERT_LT(callee.symbol_id, module->symbols.count);
  loom_op_t* kernel_op = module->symbols.entries[callee.symbol_id].defining_op;
  IREE_ASSERT(kernel_op != NULL);
  IREE_ASSERT(loom_kernel_def_isa(kernel_op));
  return kernel_op;
}

// Extends an FNV-1a hash with one byte span.
static uint32_t loom_cmd_launch_graph_hash_bytes(uint32_t hash,
                                                 const void* data,
                                                 iree_host_size_t length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_cmd_launch_graph_hash_identity(
    const loom_op_t* kernel_op, loom_value_slice_t source_workloads) {
  uint32_t hash = 2166136261u;
  const uintptr_t kernel_bits = (uintptr_t)kernel_op;
  hash =
      loom_cmd_launch_graph_hash_bytes(hash, &kernel_bits, sizeof(kernel_bits));
  hash = loom_cmd_launch_graph_hash_bytes(hash, &source_workloads.count,
                                          sizeof(source_workloads.count));
  return loom_cmd_launch_graph_hash_bytes(
      hash, source_workloads.values,
      (iree_host_size_t)source_workloads.count * sizeof(loom_value_id_t));
}

static bool loom_cmd_launch_graph_identity_matches(
    const loom_cmd_launch_graph_identity_t* identity,
    const loom_op_t* kernel_op, loom_value_slice_t source_workloads) {
  return identity->kernel_op == kernel_op &&
         identity->source_workloads.count == source_workloads.count &&
         (source_workloads.count == 0 ||
          memcmp(identity->source_workloads.values, source_workloads.values,
                 (iree_host_size_t)source_workloads.count *
                     sizeof(loom_value_id_t)) == 0);
}

static bool loom_cmd_launch_graph_identity_equal(
    const loom_cmd_launch_graph_identity_t* identity,
    const loom_op_t* kernel_op, loom_value_slice_t source_workloads,
    uint32_t hash) {
  return identity->hash == hash && loom_cmd_launch_graph_identity_matches(
                                       identity, kernel_op, source_workloads);
}

static iree_status_t loom_cmd_launch_graph_reserve_identity(
    loom_cmd_launch_graph_build_t* build) {
  loom_cmd_launch_graph_identity_plan_t* plan = &build->identity_plan;
  if (plan->identity_count < plan->identity_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(
      build->scratch_arena, plan->identity_count,
      iree_max(plan->identity_count + 1, 16u), sizeof(*plan->identities),
      &plan->identity_capacity, (void**)&plan->identities);
}

static iree_status_t loom_cmd_launch_graph_grow_identity_slots(
    loom_cmd_launch_graph_build_t* build) {
  loom_cmd_launch_graph_identity_plan_t* plan = &build->identity_plan;
  const bool initialize_hashes = plan->slot_capacity == 0;
  iree_host_size_t new_capacity = 16;
  if (plan->slot_capacity != 0 &&
      !iree_host_size_checked_mul(plan->slot_capacity, 2, &new_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "launch configuration table is too large");
  }
  iree_host_size_t* new_slots = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(build->scratch_arena, new_capacity,
                                sizeof(*new_slots), (void**)&new_slots));
  memset(new_slots, 0xFF, new_capacity * sizeof(*new_slots));

  const iree_host_size_t slot_mask = new_capacity - 1;
  for (iree_host_size_t i = 0; i < plan->identity_count; ++i) {
    if (initialize_hashes) {
      plan->identities[i].hash = loom_cmd_launch_graph_hash_identity(
          plan->identities[i].kernel_op, plan->identities[i].source_workloads);
    }
    iree_host_size_t slot = plan->identities[i].hash & slot_mask;
    while (new_slots[slot] != IREE_HOST_SIZE_MAX) {
      slot = (slot + 1) & slot_mask;
    }
    new_slots[slot] = i;
  }
  plan->slot_ordinals = new_slots;
  plan->slot_capacity = new_capacity;
  return iree_ok_status();
}

// Forms exact configuration identities while assigning the existing scheduled
// launch rows. The schedule is visited once; later body construction visits
// only unique identities.
static iree_status_t loom_cmd_launch_graph_build_identity_plan(
    loom_cmd_launch_graph_build_t* build,
    loom_cmd_launch_graph_inline_storage_t* inline_storage) {
  const iree_host_size_t launch_count = build->source.schedule->command_count;
  loom_cmd_launch_graph_identity_plan_t* plan = &build->identity_plan;
  if (launch_count != 0) {
    plan->identities = inline_storage->inline_identities;
    plan->identity_capacity = IREE_ARRAYSIZE(inline_storage->inline_identities);
    plan->result_values = inline_storage->inline_result_values;
    if (launch_count <=
        IREE_ARRAYSIZE(inline_storage->inline_launch_identity_ordinals)) {
      plan->launch_identity_ordinals =
          inline_storage->inline_launch_identity_ordinals;
    } else {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(build->scratch_arena, launch_count,
                                    sizeof(*plan->launch_identity_ordinals),
                                    (void**)&plan->launch_identity_ordinals));
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &build->target.module->arena, launch_count,
        sizeof(*build->product.launches), (void**)&build->product.launches));
    memset(build->product.launches, 0,
           launch_count * sizeof(*build->product.launches));
  }
  if (build->source.schedule->wave_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &build->target.module->arena, build->source.schedule->wave_count,
        sizeof(*build->product.waves), (void**)&build->product.waves));
    memcpy(build->product.waves, build->source.schedule->waves,
           build->source.schedule->wave_count * sizeof(*build->product.waves));
  }

  // The common single-launch program cannot share a configuration identity and
  // therefore needs neither hashing nor table initialization.
  if (launch_count == 1) {
    const loom_op_t* launch_op = build->source.schedule->commands[0];
    plan->identities[0] = (loom_cmd_launch_graph_identity_t){
        .kernel_op = loom_cmd_launch_graph_resolve_kernel(build->source.module,
                                                          launch_op),
        .source_workloads = loom_kernel_launch_workloads(launch_op),
        .first_launch_index = 0,
    };
    plan->identity_count = 1;
    plan->launch_identity_ordinals[0] = 0;
    build->product.launches[0].source_op = launch_op;
    return iree_ok_status();
  }

  for (iree_host_size_t launch_index = 0; launch_index < launch_count;
       ++launch_index) {
    const loom_op_t* launch_op = build->source.schedule->commands[launch_index];
    loom_op_t* kernel_op =
        loom_cmd_launch_graph_resolve_kernel(build->source.module, launch_op);
    const loom_value_slice_t source_workloads =
        loom_kernel_launch_workloads(launch_op);
    uint32_t hash = 0;
    iree_host_size_t identity_ordinal = IREE_HOST_SIZE_MAX;
    iree_host_size_t slot = IREE_HOST_SIZE_MAX;
    if (plan->slot_capacity == 0) {
      // Keep bounded identity sets cheaper than hashing regardless of dispatch
      // count. The fifth unique identity promotes the plan to a table.
      for (iree_host_size_t candidate_ordinal = 0;
           candidate_ordinal < plan->identity_count; ++candidate_ordinal) {
        if (loom_cmd_launch_graph_identity_matches(
                &plan->identities[candidate_ordinal], kernel_op,
                source_workloads)) {
          identity_ordinal = candidate_ordinal;
          break;
        }
      }
    } else {
      hash = loom_cmd_launch_graph_hash_identity(kernel_op, source_workloads);
      const iree_host_size_t slot_mask = plan->slot_capacity - 1;
      slot = hash & slot_mask;
      while (plan->slot_ordinals[slot] != IREE_HOST_SIZE_MAX) {
        const iree_host_size_t candidate_ordinal = plan->slot_ordinals[slot];
        if (loom_cmd_launch_graph_identity_equal(
                &plan->identities[candidate_ordinal], kernel_op,
                source_workloads, hash)) {
          identity_ordinal = candidate_ordinal;
          break;
        }
        slot = (slot + 1) & slot_mask;
      }
    }
    if (identity_ordinal == IREE_HOST_SIZE_MAX) {
      if (plan->slot_capacity == 0 &&
          plan->identity_count ==
              LOOM_CMD_LAUNCH_GRAPH_INLINE_IDENTITY_CAPACITY) {
        IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_grow_identity_slots(build));
        hash = loom_cmd_launch_graph_hash_identity(kernel_op, source_workloads);
        slot = hash & (plan->slot_capacity - 1);
        while (plan->slot_ordinals[slot] != IREE_HOST_SIZE_MAX) {
          slot = (slot + 1) & (plan->slot_capacity - 1);
        }
      } else if (plan->slot_capacity != 0 &&
                 plan->identity_count + 1 > plan->slot_capacity / 2) {
        IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_grow_identity_slots(build));
        slot = hash & (plan->slot_capacity - 1);
        while (plan->slot_ordinals[slot] != IREE_HOST_SIZE_MAX) {
          slot = (slot + 1) & (plan->slot_capacity - 1);
        }
      }
      IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_reserve_identity(build));
      identity_ordinal = plan->identity_count++;
      plan->identities[identity_ordinal] = (loom_cmd_launch_graph_identity_t){
          .kernel_op = kernel_op,
          .source_workloads = source_workloads,
          .hash = hash,
          .first_launch_index = launch_index,
      };
      if (plan->slot_capacity != 0) {
        plan->slot_ordinals[slot] = identity_ordinal;
      }
    }
    plan->launch_identity_ordinals[launch_index] = identity_ordinal;
    build->product.launches[launch_index].source_op = launch_op;
  }
  return iree_ok_status();
}

static bool loom_cmd_launch_graph_op_is_nested_in_program(
    const loom_cmd_launch_graph_build_t* build, const loom_op_t* op) {
  for (const loom_op_t* ancestor = op ? op->parent_op : NULL; ancestor;
       ancestor = ancestor->parent_op) {
    if (ancestor == build->source.program.op) return true;
  }
  return false;
}

static iree_status_t loom_cmd_launch_graph_reserve_producer_frame(
    loom_cmd_launch_graph_build_t* build) {
  if (build->producer_stack.count < build->producer_stack.capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(
      build->scratch_arena, build->producer_stack.count,
      iree_max(build->producer_stack.count + 1, 16u),
      sizeof(*build->producer_stack.frames), &build->producer_stack.capacity,
      (void**)&build->producer_stack.frames);
}

static iree_status_t loom_cmd_launch_graph_push_producer(
    loom_cmd_launch_graph_build_t* build, loom_value_id_t source_value) {
  IREE_ASSERT_LT(source_value, build->source.module->values.count);
  const loom_value_t* value =
      loom_module_value(build->source.module, source_value);
  if (loom_value_is_block_arg(value)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch workload value %%%u is an unmapped block argument; "
        "buffer-sourced and residual control-flow values require a device "
        "launch slice",
        (unsigned)source_value);
  }

  const loom_op_t* producer = loom_value_def_op(value);
  IREE_ASSERT(producer != NULL);
  const loom_trait_flags_t traits =
      loom_op_effective_traits(build->source.module, producer);
  if (!loom_cmd_launch_graph_op_is_nested_in_program(build, producer) ||
      producer->region_count != 0 ||
      !iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    const iree_string_view_t op_name =
        loom_op_name(build->source.module, producer);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch workload value %%%u is produced by `%.*s`, which cannot be "
        "placed in the aggregate host launch function",
        (unsigned)source_value, (int)op_name.size, op_name.data);
  }

  IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_reserve_producer_frame(build));
  build->producer_stack.frames[build->producer_stack.count++] =
      (loom_cmd_launch_graph_producer_frame_t){
          .op = producer,
          .next_operand = 0,
      };
  return iree_ok_status();
}

// Materializes the pure producer closure for one source-program value.
//
// An explicit stack keeps launch extraction bounded for large scalar graphs.
// Cloning one operation maps all of its results, so later requests for another
// result of the same producer reuse the existing clone.
static iree_status_t loom_cmd_launch_graph_resolve_program_value(
    loom_cmd_launch_graph_build_t* build, loom_value_id_t source_value,
    loom_value_id_t* out_target_value) {
  *out_target_value = LOOM_VALUE_ID_INVALID;
  if (loom_ir_remap_try_lookup_value(&build->target.program_remap, source_value,
                                     out_target_value)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_cmd_launch_graph_push_producer(build, source_value));
  while (build->producer_stack.count != 0) {
    loom_cmd_launch_graph_producer_frame_t* frame =
        &build->producer_stack.frames[build->producer_stack.count - 1];

    bool pushed_dependency = false;
    const loom_value_id_t* operands = loom_op_const_operands(frame->op);
    while (frame->next_operand < frame->op->operand_count) {
      const loom_value_id_t operand = operands[frame->next_operand++];
      loom_value_id_t mapped_operand = LOOM_VALUE_ID_INVALID;
      if (loom_ir_remap_try_lookup_value(&build->target.program_remap, operand,
                                         &mapped_operand)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_push_producer(build, operand));
      pushed_dependency = true;
      break;
    }
    if (pushed_dependency) continue;

    const loom_value_id_t first_result =
        frame->op->result_count == 0 ? LOOM_VALUE_ID_INVALID
                                     : loom_op_const_results(frame->op)[0];
    loom_value_id_t mapped_first_result = LOOM_VALUE_ID_INVALID;
    if (first_result == LOOM_VALUE_ID_INVALID ||
        !loom_ir_remap_try_lookup_value(&build->target.program_remap,
                                        first_result, &mapped_first_result)) {
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(loom_ir_clone_op(&build->target.builder, frame->op,
                                            &build->target.program_remap,
                                            &cloned_op));
    }
    --build->producer_stack.count;
  }

  return loom_ir_remap_resolve_value(&build->target.program_remap, source_value,
                                     out_target_value);
}

static iree_status_t loom_cmd_launch_graph_copy_value_name(
    loom_cmd_launch_graph_build_t* build, loom_ir_remap_t* remap,
    loom_value_id_t source_value, loom_value_id_t target_value) {
  const loom_string_id_t source_name =
      loom_module_value(build->source.module, source_value)->name_id;
  if (source_name == LOOM_STRING_ID_INVALID) return iree_ok_status();
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(
      remap, source_name, /*allow_invalid=*/false, &target_name));
  return loom_module_set_value_name(build->target.module, target_value,
                                    target_name);
}

static iree_status_t loom_cmd_launch_graph_attach_program_predicates(
    loom_cmd_launch_graph_build_t* build) {
  uint16_t source_predicate_count = 0;
  const loom_predicate_t* source_predicates =
      loom_func_like_predicates(build->source.program, &source_predicate_count);
  if (source_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* host_source_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->scratch_arena, source_predicate_count,
      sizeof(*host_source_predicates), (void**)&host_source_predicates));
  uint16_t host_predicate_count = 0;
  for (uint16_t i = 0; i < source_predicate_count; ++i) {
    const loom_predicate_t predicate = source_predicates[i];
    bool all_values_mapped = true;
    for (uint8_t arg_index = 0; arg_index < predicate.arg_count; ++arg_index) {
      if (predicate.arg_tags[arg_index] != LOOM_PRED_ARG_VALUE) continue;
      loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
      if (!loom_ir_remap_try_lookup_value(
              &build->target.program_remap,
              (loom_value_id_t)predicate.args[arg_index], &target_value)) {
        all_values_mapped = false;
        break;
      }
    }
    if (all_values_mapped) {
      host_source_predicates[host_predicate_count++] = predicate;
    }
  }
  if (host_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* host_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      &build->target.program_remap, host_source_predicates,
      host_predicate_count, &host_predicates));
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, build->target.module,
                                                build->scratch_arena));
  iree_status_t status = loom_rewriter_set_attr(
      &rewriter, build->target.function.op, loom_func_def_predicates_ATTR_INDEX,
      loom_attr_predicate_list(host_predicates, host_predicate_count));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_cmd_launch_graph_build_host_function(
    loom_cmd_launch_graph_build_t* build) {
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(build->source.program, &source_argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(build->source.program);
  IREE_ASSERT_GE(specialization_count_i64, 0);
  IREE_ASSERT_LE(specialization_count_i64, source_argument_count);
  const uint16_t specialization_count = (uint16_t)specialization_count_i64;

  const iree_host_size_t result_count =
      build->identity_plan.identity_count *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
  if (result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "aggregate launch function requires %" PRIhsz
                            " scalar results, exceeding the %u-result IR limit",
                            result_count, (unsigned)UINT16_MAX);
  }

  loom_type_t* argument_types = NULL;
  if (specialization_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, specialization_count, sizeof(*argument_types),
        (void**)&argument_types));
    // Populate the target value map before remapping types because a later
    // argument type may reference an earlier specialization argument.
    for (uint16_t i = 0; i < specialization_count; ++i) {
      argument_types[i] = loom_type_none();
    }
  }
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, result_count, sizeof(*result_types),
        (void**)&result_types));
    for (iree_host_size_t i = 0; i < result_count; ++i) {
      result_types[i] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    }
  }

  const loom_symbol_ref_t source_callee =
      loom_func_like_callee(build->source.program);
  const iree_string_view_t source_name =
      loom_cmd_launch_graph_symbol_name(build->source.module, source_callee);
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(build->target.module,
                                                 source_name, &target_name));
  loom_symbol_id_t target_symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(build->target.module, target_name,
                                              &target_symbol));

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
      &build->target.program_remap, build->source.program.op->location,
      &target_location));
  loom_func_def_build_flags_t build_flags = LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY;
  uint8_t visibility = 0;
  if (loom_func_like_visibility(build->source.program) != 0) {
    build_flags |= LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
    visibility = LOOM_FUNC_VISIBILITY_PUBLIC;
  }

  loom_builder_initialize(build->target.module, &build->target.module->arena,
                          loom_module_block(build->target.module),
                          &build->target.builder);
  loom_op_t* host_function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &build->target.builder, build_flags, visibility, /*retain=*/0, /*cc=*/0,
      LOOM_FUNC_PURITY_PURE, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol},
      argument_types, specialization_count, result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, target_location,
      &host_function_op));
  build->target.function =
      loom_func_like_cast(build->target.module, host_function_op);
  IREE_ASSERT(loom_func_like_isa(build->target.function));

  uint16_t host_argument_count = 0;
  const loom_value_id_t* host_arguments =
      loom_func_like_arg_ids(build->target.function, &host_argument_count);
  IREE_ASSERT_EQ(host_argument_count, specialization_count);
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_values(&build->target.program_remap, source_arguments,
                               host_arguments, specialization_count));
  for (uint16_t i = 0; i < specialization_count; ++i) {
    loom_type_t host_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &build->target.program_remap,
        loom_module_value_type(build->source.module, source_arguments[i]),
        &host_type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        build->target.module, host_arguments[i], host_type));
    IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_copy_value_name(
        build, &build->target.program_remap, source_arguments[i],
        host_arguments[i]));
  }
  IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_attach_program_predicates(build));

  loom_builder_enter_region(&build->target.builder, host_function_op,
                            loom_func_like_body(build->target.function));
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_graph_clone_config_body(
    loom_cmd_launch_graph_build_t* build, const loom_op_t* kernel_op,
    loom_builder_t* builder, loom_ir_remap_t* config_remap,
    loom_value_id_t* out_result_values) {
  loom_region_t* config_region = loom_kernel_def_config(kernel_op);
  loom_block_t* config_block = loom_region_entry_block(config_region);
  const loom_op_t* launch_config = loom_kernel_def_launch_config_op(kernel_op);

  const loom_op_t* config_op = NULL;
  loom_block_for_each_op(config_block, config_op) {
    if (config_op == launch_config) continue;
    if (config_op->region_count != 0) {
      const iree_string_view_t op_name =
          loom_op_name(build->source.module, config_op);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "aggregate launch extraction does not support nested regions in "
          "kernel launch configuration operation `%.*s`",
          (int)op_name.size, op_name.data);
    }
    bool materialize_results = config_op->result_count != 0;
    const loom_value_id_t* source_results = loom_op_const_results(config_op);
    for (uint16_t i = 0; i < config_op->result_count; ++i) {
      const loom_value_facts_t facts =
          loom_value_fact_table_lookup(build->source.facts, source_results[i]);
      const loom_type_t type =
          loom_module_value_type(build->source.module, source_results[i]);
      materialize_results &=
          loom_value_facts_can_materialize_constant(facts, type);
    }
    if (!materialize_results) {
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_ir_clone_op(builder, config_op, config_remap, &cloned_op));
      continue;
    }

    loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
        config_remap, config_op->location, &target_location));
    for (uint16_t i = 0; i < config_op->result_count; ++i) {
      const loom_value_id_t source_result = source_results[i];
      loom_type_t target_type = {0};
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(
          config_remap,
          loom_module_value_type(build->source.module, source_result),
          &target_type));
      loom_value_id_t target_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_constant_build(
          builder,
          loom_value_fact_table_lookup(build->source.facts, source_result),
          target_type, target_location, &target_result));
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_map_value(config_remap, source_result, target_result));
      IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_copy_value_name(
          build, config_remap, source_result, target_result));
    }
  }
  for (uint8_t dimension = 0;
       dimension < LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT; ++dimension) {
    const loom_value_id_t source_count =
        loom_kernel_launch_config_workgroup_count_operand(
            launch_config, (loom_kernel_dimension_t)dimension);
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        config_remap, source_count, &out_result_values[dimension]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_graph_clone_config(
    loom_cmd_launch_graph_build_t* build, iree_host_size_t identity_ordinal) {
  const loom_cmd_launch_graph_identity_t* identity =
      &build->identity_plan.identities[identity_ordinal];
  const loom_value_slice_t config_arguments =
      loom_kernel_workload_arg_ids(build->source.module, identity->kernel_op);
  IREE_ASSERT_EQ(identity->source_workloads.count, config_arguments.count);

  loom_ir_remap_t config_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(build->source.module, build->target.module,
                               build->scratch_arena, NULL, &config_remap));
  for (uint16_t i = 0; i < identity->source_workloads.count; ++i) {
    loom_value_id_t target_workload = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_resolve_program_value(
        build, identity->source_workloads.values[i], &target_workload));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &config_remap, config_arguments.values[i], target_workload));
  }
  return loom_cmd_launch_graph_clone_config_body(
      build, identity->kernel_op, &build->target.builder, &config_remap,
      &build->identity_plan
           .result_values[identity_ordinal *
                          LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT]);
}

static iree_status_t loom_cmd_launch_graph_build_body(
    loom_cmd_launch_graph_build_t* build) {
  const iree_host_size_t result_count =
      build->identity_plan.identity_count *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
  if (result_count > LOOM_CMD_LAUNCH_GRAPH_INLINE_RESULT_CAPACITY) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(build->scratch_arena, result_count,
                                  sizeof(*build->identity_plan.result_values),
                                  (void**)&build->identity_plan.result_values));
  }

  for (iree_host_size_t i = 0; i < build->identity_plan.identity_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_clone_config(build, i));
  }

  loom_op_t* return_op = NULL;
  return loom_func_return_build(
      &build->target.builder, build->identity_plan.result_values, result_count,
      build->target.function.op->location, &return_op);
}

static iree_status_t loom_cmd_launch_graph_run_cse(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_block_pool_t* block_pool) {
  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(block_pool, &pass_arena);
  loom_pass_t pass = {0};
  pass.info = loom_cse_pass_info();
  pass.instance_arena = &pass_arena;
  pass.arena = &pass_arena;
  const loom_pass_statistic_layout_t* statistics = pass.info->statistic_layout;
  iree_status_t status = iree_ok_status();
  if (statistics && statistics->storage_size != 0) {
    status = iree_arena_allocate(&pass_arena, statistics->storage_size,
                                 &pass.statistic_storage);
    if (iree_status_is_ok(status)) {
      memset(pass.statistic_storage, 0, statistics->storage_size);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_cse_run(&pass, module, function);
  }
  iree_arena_deinitialize(&pass_arena);
  return status;
}

static bool loom_cmd_launch_graph_exact_u32(
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    uint32_t* out_value, bool* out_is_exact) {
  *out_value = 0;
  *out_is_exact = false;
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(facts, value_id), &value)) {
    return true;
  }
  *out_is_exact = true;
  if (value < 0 || value > UINT32_MAX) return false;
  *out_value = (uint32_t)value;
  return true;
}

static bool loom_cmd_launch_graph_tuples_equal(const loom_value_id_t* left,
                                               const loom_value_id_t* right) {
  return left[0] == right[0] && left[1] == right[1] && left[2] == right[2];
}

static uint32_t loom_cmd_launch_graph_hash_tuple(const loom_value_id_t* tuple) {
  return loom_cmd_launch_graph_hash_bytes(
      2166136261u, tuple,
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT * sizeof(*tuple));
}

static iree_status_t loom_cmd_launch_graph_compact_results(
    loom_cmd_launch_graph_build_t* build, const loom_value_fact_table_t* facts,
    uint32_t* out_host_tuple_count) {
  *out_host_tuple_count = 0;
  loom_block_t* host_block =
      loom_region_entry_block(loom_func_like_body(build->target.function));
  loom_op_t* old_return_op = host_block->last_op;
  IREE_ASSERT(loom_func_return_isa(old_return_op));
  const loom_value_slice_t return_values =
      loom_func_return_operands(old_return_op);
  const iree_host_size_t result_count =
      build->identity_plan.identity_count *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
  IREE_ASSERT_EQ(return_values.count, result_count);

  bool inline_remove_results[LOOM_CMD_LAUNCH_GRAPH_INLINE_RESULT_CAPACITY];
  bool* remove_results = inline_remove_results;
  // Body construction no longer needs this scratch array after the return op
  // copies its operands, so reuse it for the compacted return operands.
  loom_value_id_t* kept_values = build->identity_plan.result_values;
  if (result_count != 0) {
    if (result_count > LOOM_CMD_LAUNCH_GRAPH_INLINE_RESULT_CAPACITY) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          build->scratch_arena, result_count, sizeof(*remove_results),
          (void**)&remove_results));
    }
    memset(remove_results, 1, result_count * sizeof(*remove_results));
  }

  iree_host_size_t* tuple_identity_ordinals = NULL;
  iree_host_size_t first_host_identity_ordinal = IREE_HOST_SIZE_MAX;
  uint32_t host_tuple_count = 0;
  for (iree_host_size_t identity_ordinal = 0;
       identity_ordinal < build->identity_plan.identity_count;
       ++identity_ordinal) {
    loom_cmd_launch_graph_identity_t* identity =
        &build->identity_plan.identities[identity_ordinal];
    const loom_value_id_t* tuple =
        &return_values.values[identity_ordinal *
                              LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT];
    loom_target_dispatch_workgroup_count_t direct = {0};
    uint32_t* direct_values[] = {&direct.x, &direct.y, &direct.z};
    bool all_exact = true;
    for (uint8_t dimension = 0;
         dimension < LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
         ++dimension) {
      bool is_exact = false;
      if (!loom_cmd_launch_graph_exact_u32(
              facts, tuple[dimension], direct_values[dimension], &is_exact)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "launch %" PRIhsz " workgroup-count dimension %u is outside u32",
            identity->first_launch_index, (unsigned)dimension);
      }
      all_exact &= is_exact;
    }
    if (all_exact) {
      identity->placement.kind = LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT;
      identity->placement.payload.direct = direct;
      continue;
    }

    uint32_t tuple_ordinal = UINT32_MAX;
    bool is_new_tuple = false;
    if (build->identity_plan.identity_count <=
        LOOM_CMD_LAUNCH_GRAPH_INLINE_IDENTITY_CAPACITY) {
      // A short scan is cheaper than hashing for the bounded inline case.
      for (iree_host_size_t candidate_ordinal = 0;
           candidate_ordinal < identity_ordinal; ++candidate_ordinal) {
        const loom_cmd_launch_graph_identity_t* candidate_identity =
            &build->identity_plan.identities[candidate_ordinal];
        if (candidate_identity->placement.kind !=
            LOOM_CMD_LAUNCH_COUNT_KIND_HOST) {
          continue;
        }
        const loom_value_id_t* candidate_tuple =
            &return_values
                 .values[candidate_ordinal *
                         LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT];
        if (loom_cmd_launch_graph_tuples_equal(tuple, candidate_tuple)) {
          tuple_ordinal =
              candidate_identity->placement.payload.host_tuple_ordinal;
          break;
        }
      }
      is_new_tuple = tuple_ordinal == UINT32_MAX;
    } else if (host_tuple_count == 0) {
      // Defer table initialization until a second dynamic tuple can benefit.
      is_new_tuple = true;
    } else {
      if (!tuple_identity_ordinals) {
        tuple_identity_ordinals = build->identity_plan.slot_ordinals;
        memset(tuple_identity_ordinals, 0xFF,
               build->identity_plan.slot_capacity *
                   sizeof(*tuple_identity_ordinals));
        const loom_value_id_t* first_tuple =
            &return_values
                 .values[first_host_identity_ordinal *
                         LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT];
        const iree_host_size_t slot_mask =
            build->identity_plan.slot_capacity - 1;
        iree_host_size_t first_slot =
            loom_cmd_launch_graph_hash_tuple(first_tuple) & slot_mask;
        tuple_identity_ordinals[first_slot] = first_host_identity_ordinal;
      }

      const iree_host_size_t slot_mask = build->identity_plan.slot_capacity - 1;
      iree_host_size_t slot =
          loom_cmd_launch_graph_hash_tuple(tuple) & slot_mask;
      while (tuple_identity_ordinals[slot] != IREE_HOST_SIZE_MAX) {
        const iree_host_size_t candidate_identity_ordinal =
            tuple_identity_ordinals[slot];
        const loom_value_id_t* candidate_tuple =
            &return_values
                 .values[candidate_identity_ordinal *
                         LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT];
        if (loom_cmd_launch_graph_tuples_equal(tuple, candidate_tuple)) {
          tuple_ordinal =
              build->identity_plan.identities[candidate_identity_ordinal]
                  .placement.payload.host_tuple_ordinal;
          break;
        }
        slot = (slot + 1) & slot_mask;
      }
      if (tuple_ordinal == UINT32_MAX) {
        tuple_identity_ordinals[slot] = identity_ordinal;
        is_new_tuple = true;
      }
    }
    if (is_new_tuple) {
      tuple_ordinal = host_tuple_count++;
      if (first_host_identity_ordinal == IREE_HOST_SIZE_MAX) {
        first_host_identity_ordinal = identity_ordinal;
      }
      for (uint8_t dimension = 0;
           dimension < LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
           ++dimension) {
        remove_results[identity_ordinal *
                           LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT +
                       dimension] = false;
        kept_values[(iree_host_size_t)tuple_ordinal *
                        LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT +
                    dimension] = tuple[dimension];
      }
    }
    identity->placement.kind = LOOM_CMD_LAUNCH_COUNT_KIND_HOST;
    identity->placement.payload.host_tuple_ordinal = tuple_ordinal;
  }

  for (iree_host_size_t launch_index = 0;
       launch_index < build->source.schedule->command_count; ++launch_index) {
    loom_cmd_launch_count_t* launch = &build->product.launches[launch_index];
    const loom_cmd_launch_graph_placement_t* placement =
        &build->identity_plan
             .identities[build->identity_plan
                             .launch_identity_ordinals[launch_index]]
             .placement;
    launch->kind = placement->kind;
    if (placement->kind == LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT) {
      launch->payload.direct = placement->payload.direct;
    } else {
      IREE_ASSERT_EQ(placement->kind, LOOM_CMD_LAUNCH_COUNT_KIND_HOST);
      launch->payload.host_tuple_ordinal =
          placement->payload.host_tuple_ordinal;
    }
  }
  const iree_host_size_t kept_count =
      (iree_host_size_t)host_tuple_count *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;

  loom_builder_t builder;
  loom_builder_initialize(build->target.module, &build->target.module->arena,
                          host_block, &builder);
  loom_builder_set_before(&builder, old_return_op);
  loom_op_t* new_return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_return_build(&builder, kept_values, kept_count,
                                              old_return_op->location,
                                              &new_return_op));
  IREE_RETURN_IF_ERROR(loom_op_erase(build->target.module, old_return_op));

  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_op_remove_results(
      build->target.module, build->target.function.op, remove_results,
      build->scratch_arena, &removed_count));
  IREE_ASSERT_EQ(removed_count, result_count - kept_count);
  *out_host_tuple_count = host_tuple_count;
  return iree_ok_status();
}

iree_status_t loom_cmd_launch_graph_materialize(
    const loom_module_t* source_module, loom_op_t* source_program_op,
    const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* source_facts,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_cmd_launch_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(source_module);
  IREE_ASSERT_ARGUMENT(source_program_op);
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(source_facts);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_graph);
  memset(out_graph, 0, sizeof(*out_graph));

  loom_module_t* graph_module = NULL;
  iree_status_t status = loom_module_allocate(
      source_module->context, IREE_SV("command_launch_graph"), block_pool, NULL,
      allocator, &graph_module);
  iree_arena_allocator_t scratch_arena;
  bool scratch_arena_initialized = false;
  loom_pass_value_fact_owner_t fact_owner;
  bool fact_owner_initialized = false;
  loom_canonicalizer_t canonicalizer;
  bool canonicalizer_initialized = false;
  loom_cmd_launch_graph_inline_storage_t inline_storage;

  loom_cmd_launch_graph_build_t build = {
      .source =
          {
              .module = source_module,
              .program = loom_func_like_cast(source_module, source_program_op),
              .schedule = schedule,
              .facts = source_facts,
          },
      .target =
          {
              .module = graph_module,
          },
  };
  IREE_ASSERT(loom_func_like_isa(build.source.program));
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(block_pool, &scratch_arena);
    scratch_arena_initialized = true;
    build.scratch_arena = &scratch_arena;
    status =
        loom_ir_remap_initialize(source_module, graph_module, &scratch_arena,
                                 NULL, &build.target.program_remap);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_graph_build_identity_plan(&build, &inline_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_graph_build_host_function(&build);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_graph_build_body(&build);
  }
  if (iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_initialize(block_pool, &fact_owner);
    fact_owner_initialized = true;
    status = loom_canonicalizer_initialize(graph_module, &scratch_arena,
                                           &fact_owner, &canonicalizer);
    canonicalizer_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    loom_canonicalizer_result_t result = {0};
    status = loom_canonicalizer_run_function(
        &canonicalizer, build.target.function,
        &(loom_canonicalizer_options_t){0}, &result);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_graph_run_cse(graph_module, build.target.function,
                                           block_pool);
  }
  if (iree_status_is_ok(status)) {
    loom_canonicalizer_result_t result = {0};
    status = loom_canonicalizer_run_function(
        &canonicalizer, build.target.function,
        &(loom_canonicalizer_options_t){0}, &result);
  }

  uint32_t host_tuple_count = 0;
  if (iree_status_is_ok(status)) {
    const loom_value_fact_table_t* facts =
        loom_canonicalizer_fact_table(&canonicalizer);
    IREE_ASSERT(facts != NULL);
    status =
        loom_cmd_launch_graph_compact_results(&build, facts, &host_tuple_count);
  }
  if (iree_status_is_ok(status)) {
    loom_canonicalizer_result_t result = {0};
    status = loom_canonicalizer_run_function(
        &canonicalizer, build.target.function,
        &(loom_canonicalizer_options_t){0}, &result);
  }

  if (canonicalizer_initialized) {
    loom_canonicalizer_deinitialize(&canonicalizer);
  }
  if (fact_owner_initialized) {
    loom_pass_value_fact_owner_deinitialize(&fact_owner);
  }
  if (scratch_arena_initialized) {
    iree_arena_deinitialize(&scratch_arena);
  }
  if (!iree_status_is_ok(status)) {
    if (graph_module) loom_module_free(graph_module);
    memset(out_graph, 0, sizeof(*out_graph));
    return status;
  }

  *out_graph = (loom_cmd_launch_graph_t){
      .module = graph_module,
      .host_function_op = build.target.function.op,
      .launches = build.product.launches,
      .launch_count = schedule->command_count,
      .waves = build.product.waves,
      .wave_count = schedule->wave_count,
      .host_tuple_count = host_tuple_count,
  };
  return iree_ok_status();
}

void loom_cmd_launch_graph_deinitialize(loom_cmd_launch_graph_t* graph) {
  if (!graph) return;
  if (graph->module) loom_module_free(graph->module);
  memset(graph, 0, sizeof(*graph));
}
