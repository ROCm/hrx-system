// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/stage_loop_carried_fragments.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/control_uniformity.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/facts.h"
#include "loom/target/launch.h"
#include "loom/target/pass_environment.h"
#include "loom/target/reporting/report.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_STAGE_LOOP_CARRIED_FRAGMENTS_STATISTICS(V, statistics_type)     \
  V(statistics_type, candidate_loops, "candidate-loops",                     \
    "Number of structurally eligible scf.for loops discovered.")             \
  V(statistics_type, analysis_runs, "analysis-runs",                         \
    "Number of function value-fact analyses performed.")                     \
  V(statistics_type, loops_staged, "loops-staged",                           \
    "Number of scf.for loops rebuilt with workgroup-staged fragments.")      \
  V(statistics_type, fragments_staged, "fragments-staged",                   \
    "Number of loop-carried accumulator fragments staged through workgroup " \
    "memory.")

LOOM_PASS_STATISTICS_DEFINE(loom_stage_loop_carried_fragments_statistics,
                            loom_stage_loop_carried_fragments_statistics_t,
                            LOOM_STAGE_LOOP_CARRIED_FRAGMENTS_STATISTICS)

static const loom_pass_info_t
    loom_stage_loop_carried_fragments_pass_info_storage = {
        .name = IREE_SVL("stage-loop-carried-fragments"),
        .description = IREE_SVL(
            "Stage eligible loop-carried fragments through workgroup memory."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_layout =
            &loom_stage_loop_carried_fragments_statistics_layout,
};

const loom_pass_info_t* loom_stage_loop_carried_fragments_pass_info(void) {
  return &loom_stage_loop_carried_fragments_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Candidate model
//===----------------------------------------------------------------------===//

typedef struct loom_stage_loop_carried_fragment_t {
  // Loop-carried result/iter_arg ordinal.
  uint16_t ordinal;

  // Dense staging slot ordinal.
  uint16_t slot_ordinal;

  // Initial loop iter_arg value.
  loom_value_id_t initial_value;

  // Old loop body block argument value.
  loom_value_id_t body_value;

  // Old loop yielded value for this carried slot.
  loom_value_id_t yielded_value;

  // Static logical block count of the accumulator fragment.
  int64_t block_count;

  // Static logical row count of the accumulator fragment.
  int64_t row_count;

  // Static logical column count of the accumulator fragment.
  int64_t column_count;

  // Physical payload type of the accumulator fragment.
  loom_type_t payload_type;

  // Fragment contract carried by the slot.
  loom_vector_fragment_fact_t fact;
} loom_stage_loop_carried_fragment_t;

typedef struct loom_stage_loop_carried_fragment_list_t {
  // Dense candidate storage.
  loom_stage_loop_carried_fragment_t* values;

  // Number of structurally eligible accumulator fragments.
  uint16_t candidate_count;

  // Number of fragments selected for staging.
  uint16_t count;
} loom_stage_loop_carried_fragment_list_t;

typedef struct loom_stage_loop_carried_candidate_list_t {
  // Structurally eligible loops in source preorder.
  loom_op_t** values;

  // Number of loops in |values|.
  iree_host_size_t count;

  // Allocated capacity of |values|.
  iree_host_size_t capacity;
} loom_stage_loop_carried_candidate_list_t;

typedef struct loom_stage_loop_carried_fragments_layout_t {
  // Number of independent subgroup frames in the staging allocation.
  uint32_t subgroup_count;

  // Number of logical columns reserved for one subgroup frame.
  int64_t per_subgroup_column_count;

  // Total logical columns in the workgroup staging view.
  int64_t staged_column_count;

  // Total static allocation size in bytes.
  int64_t byte_count;
} loom_stage_loop_carried_fragments_layout_t;

typedef struct loom_stage_loop_carried_fragments_context_t {
  // Pass invocation.
  loom_pass_t* pass;

  // Pass statistics.
  loom_stage_loop_carried_fragments_statistics_t* statistics;

  // Module being transformed.
  loom_module_t* module;

  // Function being transformed.
  loom_func_like_t function;

  // Optional compile report receiving source-transform rows.
  loom_target_compile_report_t* report;

  // Rewriter owning IR mutations.
  loom_rewriter_t* rewriter;

  // Resettable storage for one candidate collection and rewrite attempt.
  iree_arena_allocator_t* scratch_arena;

  // Execution-uniformity analysis sharing the rewriter's value facts.
  loom_control_uniformity_info_t* control_uniformity;
} loom_stage_loop_carried_fragments_context_t;

static iree_status_t loom_stage_loop_carried_fragments_fact_scope(
    loom_pass_t* pass, const loom_module_t* module, loom_func_like_t function,
    loom_pass_value_fact_scope_t* out_scope) {
  *out_scope = loom_pass_value_fact_scope_function(function);
  const loom_target_facts_t* target_facts = NULL;
  bool resolved = false;
  IREE_RETURN_IF_ERROR(loom_target_pass_resolve_function_facts(
      pass, module, function, &resolved, &target_facts));
  if (resolved) {
    *out_scope =
        loom_pass_value_fact_scope_function_for_target(function, target_facts);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_stage_loop_carried_fragments_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_stage_loop_carried_fragments_function_name(
    const loom_stage_loop_carried_fragments_context_t* context) {
  if (!loom_func_like_isa(context->function)) return IREE_SV("<module>");
  return loom_stage_loop_carried_fragments_symbol_name(
      context->module, loom_func_like_callee(context->function));
}

static uint64_t
loom_stage_loop_carried_fragments_removed_payload_register_count(
    const loom_stage_loop_carried_fragment_list_t* fragments) {
  uint64_t total_register_count = 0;
  for (uint16_t i = 0; i < fragments->count; ++i) {
    const loom_type_t payload_type = fragments->values[i].payload_type;
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(payload_type, &element_count)) {
      return 0;
    }
    const int32_t element_bit_count =
        loom_scalar_type_bitwidth(loom_type_element_type(payload_type));
    if (element_bit_count <= 0 ||
        element_count > UINT64_MAX / (uint64_t)element_bit_count) {
      return 0;
    }
    const uint64_t payload_bit_count =
        element_count * (uint64_t)element_bit_count;
    if (payload_bit_count > UINT64_MAX - 31u) {
      return 0;
    }
    const uint64_t payload_register_count = (payload_bit_count + 31u) / 32u;
    if (total_register_count > UINT64_MAX - payload_register_count) {
      return 0;
    }
    total_register_count += payload_register_count;
  }
  return total_register_count;
}

static iree_status_t loom_stage_loop_carried_fragments_report(
    const loom_stage_loop_carried_fragments_context_t* context,
    iree_string_view_t source_op_name, uint32_t source_op_kind,
    const loom_stage_loop_carried_fragment_list_t* fragments,
    iree_string_view_t outcome, iree_string_view_t reason,
    uint64_t workgroup_memory_byte_count) {
  if (!context->report ||
      !loom_target_compile_report_wants_details(
          context->report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS) ||
      fragments->candidate_count == 0) {
    return iree_ok_status();
  }
  uint64_t row_count = 0;
  uint64_t column_count = 0;
  uint64_t block_count = 0;
  if (fragments->values != NULL && fragments->values[0].block_count > 0 &&
      fragments->values[0].row_count > 0 &&
      fragments->values[0].column_count > 0) {
    if (fragments->values[0].fact.shape_rank == 3) {
      block_count = (uint64_t)fragments->values[0].block_count;
    }
    row_count = (uint64_t)fragments->values[0].row_count;
    column_count = (uint64_t)fragments->values[0].column_count;
  }
  const loom_target_compile_report_source_low_transform_row_t row = {
      .function_name = loom_stage_loop_carried_fragments_function_name(context),
      .source_op_name = source_op_name,
      .source_op_kind = source_op_kind,
      .transform_key = IREE_SV("stage-loop-carried-fragments"),
      .outcome = outcome,
      .reason = reason,
      .candidate_value_count = fragments->candidate_count,
      .selected_value_count = fragments->count,
      .removed_loop_carried_value_count = fragments->count,
      .removed_loop_carried_payload_register_count =
          loom_stage_loop_carried_fragments_removed_payload_register_count(
              fragments),
      .block_count = block_count,
      .row_count = row_count,
      .column_count = column_count,
      .workgroup_memory_byte_count = workgroup_memory_byte_count,
      .inserted_load_op_count = (uint32_t)(fragments->count * 2),
      .inserted_store_op_count = (uint32_t)(fragments->count * 2),
      .inserted_barrier_op_count = fragments->count == 0 ? 0 : 2,
  };
  return loom_target_compile_report_record_source_low_transform_row(
      context->report, &row);
}

static bool loom_stage_loop_carried_fragments_match_structural_loop(
    loom_op_t* op, loom_op_t** out_yield) {
  if (out_yield) *out_yield = NULL;
  if (!loom_scf_for_isa(op) || op->tied_result_count != 0 ||
      op->result_count == 0) {
    return false;
  }

  loom_region_t* body = loom_scf_for_body(op);
  if (!body || body->block_count != 1) return false;
  loom_block_t* block = loom_region_entry_block(body);
  if (!block || block->arg_count != 1 + op->result_count) {
    return false;
  }
  loom_op_t* yield = block->last_op;
  if (!yield || !loom_scf_yield_isa(yield)) return false;

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return false;
  }

  if (out_yield) *out_yield = yield;
  return true;
}

static bool loom_stage_loop_carried_fragments_exact_i64(
    loom_rewriter_t* rewriter, loom_value_id_t value, int64_t* out_value) {
  return loom_value_facts_as_exact_i64(
      loom_rewriter_value_facts(rewriter, value), out_value);
}

static bool loom_stage_loop_carried_fragments_static_positive_shape(
    loom_rewriter_t* rewriter, loom_vector_fragment_fact_t fact,
    int64_t* out_blocks, int64_t* out_rows, int64_t* out_columns) {
  if (fact.shape_rank != 2 && fact.shape_rank != 3) return false;
  int64_t blocks = 1;
  int64_t rows = 0;
  int64_t columns = 0;
  const loom_value_id_t block_value =
      loom_vector_fragment_fact_block_value(fact);
  if (block_value != LOOM_VALUE_ID_INVALID &&
      !loom_stage_loop_carried_fragments_exact_i64(rewriter, block_value,
                                                   &blocks)) {
    return false;
  }
  if (!loom_stage_loop_carried_fragments_exact_i64(
          rewriter, loom_vector_fragment_fact_row_value(fact), &rows) ||
      !loom_stage_loop_carried_fragments_exact_i64(
          rewriter, loom_vector_fragment_fact_column_value(fact), &columns)) {
    return false;
  }
  if (blocks <= 0 || rows <= 0 || columns <= 0) return false;
  *out_blocks = blocks;
  *out_rows = rows;
  *out_columns = columns;
  return true;
}

static bool loom_stage_loop_carried_fragments_dense_accumulator_fact(
    loom_rewriter_t* rewriter, loom_value_id_t value,
    loom_vector_fragment_fact_t* out_fact, int64_t* out_blocks,
    int64_t* out_rows, int64_t* out_columns) {
  loom_vector_fragment_fact_t fact;
  if (!loom_vector_fragment_fact_query_value_facts(
          &rewriter->fact_table->context,
          loom_rewriter_value_facts(rewriter, value), &fact) ||
      !loom_vector_fragment_fact_is_accumulator_like(fact) ||
      iree_any_bit_set(fact.flags,
                       LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA |
                           LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_STATIC_SCHEMA)) {
    return false;
  }
  if (fact.auxiliary.present_keys != 0) {
    return false;
  }
  if (!loom_stage_loop_carried_fragments_static_positive_shape(
          rewriter, fact, out_blocks, out_rows, out_columns)) {
    return false;
  }
  *out_fact = fact;
  return true;
}

static bool loom_stage_loop_carried_fragments_same_candidate_contract(
    const loom_stage_loop_carried_fragment_t* base,
    const loom_stage_loop_carried_fragment_t* candidate) {
  loom_vector_fragment_fact_t base_fact = base->fact;
  loom_vector_fragment_fact_t candidate_fact = candidate->fact;
  memset(base_fact.shape_value_ids, 0, sizeof(base_fact.shape_value_ids));
  memset(candidate_fact.shape_value_ids, 0,
         sizeof(candidate_fact.shape_value_ids));
  return base->block_count == candidate->block_count &&
         base->row_count == candidate->row_count &&
         base->column_count == candidate->column_count &&
         loom_type_equal(base->payload_type, candidate->payload_type) &&
         loom_vector_fragment_facts_match_accumulator_contract(base_fact,
                                                               candidate_fact);
}

static bool loom_stage_loop_carried_fragments_exact_subgroup_count(
    const loom_stage_loop_carried_fragments_context_t* context,
    uint32_t* out_count) {
  *out_count = 0;
  if (!loom_kernel_def_isa(context->function.op)) {
    return false;
  }

  const loom_value_fact_table_t* fact_table = context->rewriter->fact_table;
  const loom_target_facts_t* target_facts =
      fact_table ? fact_table->context.target_facts : NULL;
  const loom_target_bundle_t* bundle = loom_target_facts_bundle(target_facts);
  if (!bundle || !bundle->snapshot || !bundle->export_plan ||
      bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL ||
      bundle->snapshot->subgroup_size == 0) {
    *out_count = 0;
    return false;
  }

  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_kernel_def_static_workgroup_size_from_facts(
          context->module, context->function.op, fact_table, &workgroup_size)) {
    workgroup_size = bundle->export_plan->hal_kernel.required_workgroup_size;
  }
  uint32_t flat_workgroup_size = 0;
  if (!loom_target_workgroup_size_flat_product_u32(&workgroup_size,
                                                   &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    *out_count = 0;
    return false;
  }

  const uint32_t subgroup_size = bundle->snapshot->subgroup_size;
  *out_count = flat_workgroup_size / subgroup_size +
               (flat_workgroup_size % subgroup_size != 0 ? 1u : 0u);
  return *out_count != 0;
}

static bool loom_stage_loop_carried_fragments_compute_layout(
    const loom_stage_loop_carried_fragment_list_t* staged_fragments,
    uint32_t subgroup_count,
    loom_stage_loop_carried_fragments_layout_t* out_layout) {
  memset(out_layout, 0, sizeof(*out_layout));
  const loom_stage_loop_carried_fragment_t* base = &staged_fragments->values[0];
  int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(base->payload_type));
  if (element_bit_count <= 0 || (element_bit_count % 8) != 0 ||
      subgroup_count == 0) {
    return false;
  }

  int64_t per_subgroup_column_count = 0;
  int64_t staged_column_count = 0;
  int64_t logical_element_count = 0;
  int64_t byte_count = 0;
  if (!iree_checked_mul_i64(base->column_count, staged_fragments->count,
                            &per_subgroup_column_count) ||
      !iree_checked_mul_i64(per_subgroup_column_count, subgroup_count,
                            &staged_column_count) ||
      !iree_checked_mul_i64(base->row_count, staged_column_count,
                            &logical_element_count) ||
      !iree_checked_mul_i64(base->block_count, logical_element_count,
                            &logical_element_count) ||
      !iree_checked_mul_i64(logical_element_count, element_bit_count / 8,
                            &byte_count)) {
    return false;
  }

  *out_layout = (loom_stage_loop_carried_fragments_layout_t){
      .subgroup_count = subgroup_count,
      .per_subgroup_column_count = per_subgroup_column_count,
      .staged_column_count = staged_column_count,
      .byte_count = byte_count,
  };
  return true;
}

static iree_status_t loom_stage_loop_carried_fragments_collect(
    loom_stage_loop_carried_fragments_context_t* context, loom_op_t* op,
    loom_op_t** out_yield, loom_stage_loop_carried_fragment_list_t* out_list) {
  *out_yield = NULL;
  memset(out_list, 0, sizeof(*out_list));

  if (!loom_stage_loop_carried_fragments_match_structural_loop(op, out_yield)) {
    return iree_ok_status();
  }

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(*out_yield);
  loom_block_t* block = loom_region_entry_block(loom_scf_for_body(op));

  loom_stage_loop_carried_fragment_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->scratch_arena, op->result_count,
                                sizeof(*candidates), (void**)&candidates));

  const loom_value_id_t* results = loom_op_const_results(op);
  uint16_t candidate_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_t payload_type =
        loom_module_value_type(context->module, results[i]);
    if (!loom_type_is_vector(payload_type)) continue;

    loom_stage_loop_carried_fragment_t candidate = {
        .ordinal = i,
        .slot_ordinal = candidate_count,
        .initial_value = iter_args.values[i],
        .body_value = loom_block_arg_id(block, (uint16_t)(1 + i)),
        .yielded_value = yielded_values.values[i],
        .payload_type = payload_type,
    };
    if (!loom_stage_loop_carried_fragments_dense_accumulator_fact(
            context->rewriter, candidate.initial_value, &candidate.fact,
            &candidate.block_count, &candidate.row_count,
            &candidate.column_count)) {
      continue;
    }
    loom_vector_fragment_fact_t yielded_fact;
    int64_t yielded_blocks = 0;
    int64_t yielded_rows = 0;
    int64_t yielded_columns = 0;
    if (!loom_stage_loop_carried_fragments_dense_accumulator_fact(
            context->rewriter, candidate.yielded_value, &yielded_fact,
            &yielded_blocks, &yielded_rows, &yielded_columns)) {
      continue;
    }
    loom_stage_loop_carried_fragment_t yielded_candidate = candidate;
    yielded_candidate.fact = yielded_fact;
    yielded_candidate.block_count = yielded_blocks;
    yielded_candidate.row_count = yielded_rows;
    yielded_candidate.column_count = yielded_columns;
    if (!loom_stage_loop_carried_fragments_same_candidate_contract(
            &candidate, &yielded_candidate)) {
      continue;
    }
    if (candidate_count != 0 &&
        !loom_stage_loop_carried_fragments_same_candidate_contract(
            &candidates[0], &candidate)) {
      continue;
    }
    candidates[candidate_count++] = candidate;
  }

  out_list->values = candidates;
  out_list->candidate_count = candidate_count;
  out_list->count = candidate_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

static iree_status_t loom_stage_loop_carried_fragments_build_index_constant(
    loom_stage_loop_carried_fragments_context_t* context, loom_op_t* anchor_op,
    int64_t value, loom_type_t type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &context->rewriter->builder, loom_attr_i64(value), type,
      anchor_op->location, &constant_op));
  *out_value = loom_index_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_stage_loop_carried_fragments_build_barrier(
    loom_stage_loop_carried_fragments_context_t* context,
    loom_op_t* anchor_op) {
  loom_op_t* barrier_op = NULL;
  return loom_kernel_barrier_build(
      &context->rewriter->builder, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
      LOOM_ATOMIC_SCOPE_WORKGROUP, LOOM_ATOMIC_ORDERING_ACQ_REL,
      anchor_op->location, &barrier_op);
}

static iree_status_t loom_stage_loop_carried_fragments_build_fragment_store(
    loom_stage_loop_carried_fragments_context_t* context, loom_op_t* anchor_op,
    loom_value_id_t value, loom_value_id_t view, loom_value_id_t zero_index,
    loom_value_id_t column_index,
    const loom_stage_loop_carried_fragment_t* fragment) {
  loom_vector_fragment_store_build_flags_t build_flags = 0;
  loom_value_id_t blocks =
      loom_vector_fragment_fact_block_value(fragment->fact);
  const bool has_blocks = blocks != LOOM_VALUE_ID_INVALID;
  if (has_blocks) {
    build_flags |= LOOM_VECTOR_FRAGMENT_STORE_BUILD_FLAG_HAS_BLOCKS;
  }
  int64_t static_indices[3] = {INT64_MIN, INT64_MIN, INT64_MIN};
  loom_value_id_t indices[3] = {zero_index, zero_index, column_index};
  const iree_host_size_t index_count = has_blocks ? 3 : 2;
  const loom_value_id_t* access_indices = has_blocks ? indices : &indices[1];
  loom_op_t* store_op = NULL;
  return loom_vector_fragment_store_build(
      &context->rewriter->builder, build_flags, LOOM_VECTOR_ROLE_RESULT, value,
      view, access_indices, index_count, static_indices, index_count, blocks,
      loom_vector_fragment_fact_row_value(fragment->fact),
      loom_vector_fragment_fact_column_value(fragment->fact),
      /*cache_scope=*/0, /*cache_temporal=*/0, anchor_op->location, &store_op);
}

static iree_status_t loom_stage_loop_carried_fragments_build_fragment_load(
    loom_stage_loop_carried_fragments_context_t* context, loom_op_t* anchor_op,
    loom_value_id_t view, loom_value_id_t zero_index,
    loom_value_id_t column_index,
    const loom_stage_loop_carried_fragment_t* fragment,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_vector_fragment_load_build_flags_t build_flags = 0;
  loom_value_id_t blocks =
      loom_vector_fragment_fact_block_value(fragment->fact);
  const bool has_blocks = blocks != LOOM_VALUE_ID_INVALID;
  if (has_blocks) {
    build_flags |= LOOM_VECTOR_FRAGMENT_LOAD_BUILD_FLAG_HAS_BLOCKS;
  }
  int64_t static_indices[3] = {INT64_MIN, INT64_MIN, INT64_MIN};
  loom_value_id_t indices[3] = {zero_index, zero_index, column_index};
  const iree_host_size_t index_count = has_blocks ? 3 : 2;
  const loom_value_id_t* access_indices = has_blocks ? indices : &indices[1];
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_fragment_load_build(
      &context->rewriter->builder, build_flags, LOOM_VECTOR_ROLE_INIT, view,
      access_indices, index_count, static_indices, index_count, blocks,
      loom_vector_fragment_fact_row_value(fragment->fact),
      loom_vector_fragment_fact_column_value(fragment->fact),
      /*auxiliary=*/NULL, /*auxiliary_count=*/0, /*cache_scope=*/0,
      /*cache_temporal=*/0, fragment->payload_type, anchor_op->location,
      &load_op));
  *out_value = loom_vector_fragment_load_result(load_op);
  return iree_ok_status();
}

static iree_status_t loom_stage_loop_carried_fragments_remap_value(
    loom_ir_remap_t* remap, loom_value_id_t source,
    loom_value_id_t* out_target) {
  if (loom_ir_remap_try_lookup_value(remap, source, out_target)) {
    return iree_ok_status();
  }
  return loom_ir_remap_resolve_value(remap, source, out_target);
}

static iree_status_t loom_stage_loop_carried_fragments_rewrite(
    loom_stage_loop_carried_fragments_context_t* context, loom_op_t* op,
    loom_op_t* yield,
    const loom_stage_loop_carried_fragment_list_t* staged_fragments,
    const loom_stage_loop_carried_fragments_layout_t* layout) {
  const loom_stage_loop_carried_fragment_t* base = &staged_fragments->values[0];
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  loom_block_t* old_block = loom_region_entry_block(loom_scf_for_body(op));

  uint16_t* staged_index_by_ordinal = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->scratch_arena, op->result_count,
      sizeof(*staged_index_by_ordinal), (void**)&staged_index_by_ordinal));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    staged_index_by_ordinal[i] = UINT16_MAX;
  }
  for (uint16_t i = 0; i < staged_fragments->count; ++i) {
    staged_index_by_ordinal[staged_fragments->values[i].ordinal] = i;
  }

  uint16_t kept_count = (uint16_t)(op->result_count - staged_fragments->count);
  loom_value_id_t* kept_iter_args = NULL;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->scratch_arena, kept_count, sizeof(*kept_iter_args),
        (void**)&kept_iter_args));
  }
  uint16_t kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (staged_index_by_ordinal[i] != UINT16_MAX) continue;
    kept_iter_args[kept_ordinal] = iter_args.values[i];
    ++kept_ordinal;
  }

  loom_scf_for_build_flags_t build_flags = 0;
  loom_value_id_t unroll_factor = LOOM_VALUE_ID_INVALID;
  if (loom_scf_for_unroll_factor_is_present(op)) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    unroll_factor = loom_scf_for_unroll_factor(op);
  }
  loom_scf_for_unroll_policy_t unroll_policy = 0;
  if (!loom_attr_is_absent(
          loom_op_attrs(op)[loom_scf_for_unroll_policy_ATTR_INDEX])) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    unroll_policy = loom_scf_for_unroll_policy(op);
  }
  loom_scf_for_unroll_schedule_t unroll_schedule = 0;
  if (!loom_attr_is_absent(
          loom_op_attrs(op)[loom_scf_for_unroll_schedule_ATTR_INDEX])) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_SCHEDULE;
    unroll_schedule = loom_scf_for_unroll_schedule(op);
  }

  loom_builder_set_before(&context->rewriter->builder, op);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);

  loom_value_id_t byte_count_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_build_index_constant(
      context, op, layout->byte_count,
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), &byte_count_value));
  loom_value_id_t zero_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_build_index_constant(
      context, op, 0, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), &zero_offset));
  loom_value_id_t zero_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_build_index_constant(
      context, op, 0, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &zero_index));

  loom_op_t* layout_op = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_layout_dense_build(
      &context->rewriter->builder,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
      op->location, &layout_op));
  loom_value_id_t layout_value = loom_encoding_layout_dense_result(layout_op);

  loom_op_t* buffer_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      &context->rewriter->builder, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
      /*base_alignment=*/16, byte_count_value, loom_type_buffer(), op->location,
      &buffer_op));
  loom_value_id_t buffer_value = loom_buffer_alloca_result(buffer_op);

  loom_overflow_dim_t view_dimensions[3] = {
      loom_dim_pack_static(base->block_count),
      loom_dim_pack_static(base->row_count),
      loom_dim_pack_static(layout->staged_column_count),
  };
  loom_type_t view_type = {0};
  if (base->fact.shape_rank == 3) {
    view_type.header = loom_type_make_header(
        LOOM_TYPE_VIEW, loom_type_element_type(base->payload_type), 3,
        LOOM_TYPE_FLAG_ALL_STATIC);
    view_type.dims[0] = (uint64_t)(uintptr_t)view_dimensions;
  } else {
    view_type = loom_type_shaped_2d(
        LOOM_TYPE_VIEW, loom_type_element_type(base->payload_type),
        view_dimensions[1], view_dimensions[2], /*encoding_id=*/0);
  }
  view_type.encoding_id = (uint16_t)layout_value;
  view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  loom_op_t* view_op = NULL;
  // Building the result canonicalizes the type and clones rank-3 dimensions
  // into the module arena before |view_dimensions| leaves scope.
  IREE_RETURN_IF_ERROR(
      loom_buffer_view_build(&context->rewriter->builder, buffer_value,
                             zero_offset, view_type, op->location, &view_op));
  loom_value_id_t staging_view = loom_buffer_view_result(view_op);

  loom_value_id_t* slot_columns = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->scratch_arena, staged_fragments->count,
                                sizeof(*slot_columns), (void**)&slot_columns));

  loom_value_id_t subgroup_column_base = LOOM_VALUE_ID_INVALID;
  if (layout->subgroup_count > 1) {
    loom_op_t* subgroup_id_op = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_subgroup_id_build(
        &context->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        op->location, &subgroup_id_op));
    loom_value_id_t per_subgroup_column_count = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_build_index_constant(
        context, op, layout->per_subgroup_column_count,
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &per_subgroup_column_count));
    loom_op_t* subgroup_column_base_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_mul_build(
        &context->rewriter->builder,
        loom_kernel_subgroup_id_result(subgroup_id_op),
        per_subgroup_column_count, op->location, &subgroup_column_base_op));
    subgroup_column_base = loom_index_mul_result(subgroup_column_base_op);
  }

  for (uint16_t i = 0; i < staged_fragments->count; ++i) {
    const loom_stage_loop_carried_fragment_t* fragment =
        &staged_fragments->values[i];
    const int64_t column = fragment->slot_ordinal * base->column_count;
    if (subgroup_column_base != LOOM_VALUE_ID_INVALID && column == 0) {
      slot_columns[i] = subgroup_column_base;
    } else {
      IREE_RETURN_IF_ERROR(
          loom_stage_loop_carried_fragments_build_index_constant(
              context, op, column, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
              &slot_columns[i]));
      if (subgroup_column_base != LOOM_VALUE_ID_INVALID) {
        loom_op_t* slot_column_op = NULL;
        IREE_RETURN_IF_ERROR(loom_index_add_build(
            &context->rewriter->builder, subgroup_column_base, slot_columns[i],
            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), op->location,
            &slot_column_op));
        slot_columns[i] = loom_index_add_result(slot_column_op);
      }
    }
    IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_build_fragment_store(
        context, op, fragment->initial_value, staging_view, zero_index,
        slot_columns[i], fragment));
  }
  IREE_RETURN_IF_ERROR(
      loom_stage_loop_carried_fragments_build_barrier(context, op));

  loom_op_t* new_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &context->rewriter->builder, build_flags, loom_scf_for_lower_bound(op),
      loom_scf_for_upper_bound(op), loom_scf_for_step(op), kept_iter_args,
      kept_count, /*tied_results=*/NULL, /*tied_result_count=*/0, unroll_factor,
      unroll_policy, unroll_schedule, op->location, &new_loop));

  loom_region_t* new_body = loom_scf_for_body(new_loop);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->rewriter->builder, new_loop, new_body);
  loom_block_t* new_block = loom_region_entry_block(new_body);

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, context->scratch_arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      &remap));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_value(&remap, loom_block_arg_id(old_block, 0),
                              loom_block_arg_id(new_block, 0)));

  kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t old_arg = loom_block_arg_id(old_block, (uint16_t)(1 + i));
    if (staged_index_by_ordinal[i] != UINT16_MAX) continue;
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &remap, old_arg,
        loom_block_arg_id(new_block, (uint16_t)(1 + kept_ordinal++))));
  }

  for (uint16_t i = 0; i < staged_fragments->count; ++i) {
    const loom_stage_loop_carried_fragment_t* fragment =
        &staged_fragments->values[i];
    loom_value_id_t initial_load = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_build_fragment_load(
        context, op, staging_view, zero_index, slot_columns[i], fragment,
        &initial_load));
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, fragment->body_value, initial_load));
  }

  IREE_RETURN_IF_ERROR(loom_ir_clone_block_ops(
      &context->rewriter->builder, old_block, &remap,
      &(loom_ir_clone_block_options_t){.omit_terminators = true}));

  loom_value_id_t* replacement_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->scratch_arena, op->result_count, sizeof(*replacement_values),
      (void**)&replacement_values));

  loom_value_id_t* kept_yield_values = NULL;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->scratch_arena, kept_count, sizeof(*kept_yield_values),
        (void**)&kept_yield_values));
  }

  kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_remap_value(
        &remap, yielded_values.values[i], &target_value));
    uint16_t staged_index = staged_index_by_ordinal[i];
    if (staged_index != UINT16_MAX) {
      IREE_RETURN_IF_ERROR(
          loom_stage_loop_carried_fragments_build_fragment_store(
              context, yield, target_value, staging_view, zero_index,
              slot_columns[staged_index],
              &staged_fragments->values[staged_index]));
      continue;
    }
    kept_yield_values[kept_ordinal++] = target_value;
  }

  IREE_RETURN_IF_ERROR(
      loom_stage_loop_carried_fragments_build_barrier(context, yield));
  loom_op_t* new_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&context->rewriter->builder,
                                            kept_yield_values, kept_count,
                                            yield->location, &new_yield));
  (void)new_yield;
  loom_builder_restore(&context->rewriter->builder, saved_ip);

  loom_builder_set_after(&context->rewriter->builder, new_loop);
  for (uint16_t i = staged_fragments->count; i > 0; --i) {
    const loom_stage_loop_carried_fragment_t* fragment =
        &staged_fragments->values[i - 1];
    loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_build_fragment_load(
        context, op, staging_view, zero_index, slot_columns[i - 1], fragment,
        &result_value));
    replacement_values[fragment->ordinal] = result_value;
  }

  kept_ordinal = 0;
  loom_value_slice_t new_results = loom_scf_for_results(new_loop);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (staged_index_by_ordinal[i] != UINT16_MAX) continue;
    replacement_values[i] = new_results.values[kept_ordinal++];
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, op, replacement_values, op->result_count,
      value_checkpoint));
  const iree_string_view_t source_op_name = loom_op_name(context->module, op);
  const uint32_t source_op_kind = op->kind;
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, op, replacement_values, op->result_count));

  IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_report(
      context, source_op_name, source_op_kind, staged_fragments,
      IREE_SV("selected"), IREE_SV("staged_workgroup_memory"),
      (uint64_t)layout->byte_count));

  ++context->statistics->loops_staged;
  context->statistics->fragments_staged += staged_fragments->count;
  loom_pass_mark_changed(context->pass);
  return iree_ok_status();
}

static iree_status_t loom_stage_loop_carried_fragments_try_rewrite(
    loom_stage_loop_carried_fragments_context_t* context, loom_op_t* op,
    bool* out_changed) {
  *out_changed = false;
  loom_op_t* yield = NULL;
  loom_stage_loop_carried_fragment_list_t staged_fragments = {0};
  IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_collect(
      context, op, &yield, &staged_fragments));
  if (staged_fragments.count == 0) {
    return iree_ok_status();
  }

  loom_control_uniformity_failure_t control_failure = {0};
  bool control_proven = false;
  IREE_RETURN_IF_ERROR(loom_control_uniformity_prove_execution(
      context->control_uniformity, yield,
      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP, &control_failure,
      &control_proven));
  if (!control_proven) {
    IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_report(
        context, loom_op_name(context->module, op), op->kind, &staged_fragments,
        IREE_SV("declined"), IREE_SV("control_not_workgroup_uniform"),
        /*workgroup_memory_byte_count=*/0));
    return iree_ok_status();
  }

  uint32_t subgroup_count = 1;
  if (!loom_stage_loop_carried_fragments_exact_subgroup_count(
          context, &subgroup_count)) {
    IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_report(
        context, loom_op_name(context->module, op), op->kind, &staged_fragments,
        IREE_SV("declined"), IREE_SV("subgroup_count_not_static"),
        /*workgroup_memory_byte_count=*/0));
    return iree_ok_status();
  }
  loom_stage_loop_carried_fragments_layout_t layout = {0};
  if (!loom_stage_loop_carried_fragments_compute_layout(
          &staged_fragments, subgroup_count, &layout)) {
    IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_report(
        context, loom_op_name(context->module, op), op->kind, &staged_fragments,
        IREE_SV("declined"), IREE_SV("staging_layout_not_static"),
        /*workgroup_memory_byte_count=*/0));
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_rewrite(
      context, op, yield, &staged_fragments, &layout));
  *out_changed = true;
  return iree_ok_status();
}

typedef struct loom_stage_loop_carried_candidate_collect_t {
  // Arena retaining the candidate list for the function pass invocation.
  iree_arena_allocator_t* arena;

  // Candidate list populated in source preorder.
  loom_stage_loop_carried_candidate_list_t* candidates;
} loom_stage_loop_carried_candidate_collect_t;

static iree_status_t loom_stage_loop_carried_fragments_collect_candidate(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_stage_loop_carried_fragments_match_structural_loop(op, NULL)) {
    return iree_ok_status();
  }

  loom_stage_loop_carried_candidate_collect_t* collect =
      (loom_stage_loop_carried_candidate_collect_t*)user_data;
  loom_stage_loop_carried_candidate_list_t* candidates = collect->candidates;
  if (candidates->count == candidates->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        collect->arena, candidates->count, candidates->count + 1,
        sizeof(*candidates->values), &candidates->capacity,
        (void**)&candidates->values));
  }
  candidates->values[candidates->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_stage_loop_carried_fragments_collect_candidates(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    loom_stage_loop_carried_candidate_list_t* out_candidates) {
  memset(out_candidates, 0, sizeof(*out_candidates));
  loom_stage_loop_carried_candidate_collect_t collect = {
      .arena = pass->arena,
      .candidates = out_candidates,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){
          .fn = loom_stage_loop_carried_fragments_collect_candidate,
          .user_data = &collect,
      },
      pass->arena, &walk_result);
}

iree_status_t loom_stage_loop_carried_fragments_run(loom_pass_t* pass,
                                                    loom_module_t* module,
                                                    loom_func_like_t function) {
  if (!loom_kernel_def_isa(function.op) || !loom_func_like_body(function)) {
    return iree_ok_status();
  }

  loom_stage_loop_carried_fragments_statistics_t* statistics =
      loom_stage_loop_carried_fragments_statistics(pass);
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  loom_target_compile_report_t* compile_report =
      loom_low_pass_capability_compile_report(low_capability);

  loom_stage_loop_carried_candidate_list_t candidates = {0};
  IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_collect_candidates(
      pass, module, function, &candidates));
  statistics->candidate_loops += (int64_t)candidates.count;
  if (candidates.count == 0) return iree_ok_status();

  loom_pass_value_fact_scope_t fact_scope =
      loom_pass_value_fact_scope_function(function);
  IREE_RETURN_IF_ERROR(loom_stage_loop_carried_fragments_fact_scope(
      pass, module, function, &fact_scope));

  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_value_facts_prepare(pass, module, fact_scope, &facts));

  iree_arena_allocator_t scratch_arena = {0};
  iree_arena_initialize(pass->arena->block_pool, &scratch_arena);
  // Retain one pool block across attempts while discarding any larger
  // candidate-specific growth at each checkpoint restore.
  void* scratch_anchor = NULL;
  iree_status_t status =
      iree_arena_allocate(&scratch_arena, 1, &scratch_anchor);
  (void)scratch_anchor;
  const iree_arena_checkpoint_t scratch_checkpoint =
      iree_arena_checkpoint_save(&scratch_arena);

  loom_rewriter_t rewriter = {0};
  bool rewriter_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_initialize(&rewriter, module, pass->arena);
  }
  if (iree_status_is_ok(status)) {
    rewriter_initialized = true;
    status = loom_rewriter_enable_analysis(&rewriter, function, facts);
  }
  if (iree_status_is_ok(status)) {
    ++statistics->analysis_runs;
    // The rewriter pops from the end. Seed in reverse source order so outer
    // loops are processed before their nested loops and siblings remain in
    // source order. Cloned descendants re-enter through the builder callback.
    for (iree_host_size_t i = candidates.count; i > 0; --i) {
      status =
          loom_rewriter_add_to_worklist(&rewriter, candidates.values[i - 1]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  bool changed = false;
  if (iree_status_is_ok(status)) {
    loom_control_uniformity_info_t control_uniformity;
    loom_control_uniformity_info_initialize(module, facts, pass->arena,
                                            &control_uniformity);
    loom_stage_loop_carried_fragments_context_t context = {
        .pass = pass,
        .statistics = statistics,
        .module = module,
        .function = function,
        .report = compile_report,
        .rewriter = &rewriter,
        .scratch_arena = &scratch_arena,
        .control_uniformity = &control_uniformity,
    };
    loom_op_t* op = NULL;
    while (iree_status_is_ok(status) && (op = loom_rewriter_pop(&rewriter))) {
      if (!loom_scf_for_isa(op)) continue;
      bool op_changed = false;
      status = loom_stage_loop_carried_fragments_try_rewrite(&context, op,
                                                             &op_changed);
      changed |= op_changed;
      iree_arena_checkpoint_restore(&scratch_checkpoint);
    }
  }

  if (rewriter_initialized) {
    loom_rewriter_deinitialize(&rewriter);
  }
  iree_arena_deinitialize(&scratch_arena);
  if ((changed || !iree_status_is_ok(status)) && pass->value_facts) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  return status;
}
