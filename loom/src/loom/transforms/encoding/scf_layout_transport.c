// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <string.h>

#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/canonicalize.h"
#include "loom/ops/scf/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/transforms/encoding/layout_transport.h"
#include "loom/util/walk.h"

#define LOOM_SCF_LAYOUT_TRANSPORT_STATISTICS(V, statistics_type)       \
  V(statistics_type, layouts_decomposed, "layouts-decomposed",         \
    "Number of structured strided-layout results decomposed.")         \
  V(statistics_type, stride_values_inserted, "stride-values-inserted", \
    "Number of index-valued stride results inserted.")                 \
  V(statistics_type, operations_rewritten, "operations-rewritten",     \
    "Number of structured control-flow operations rewritten.")

LOOM_PASS_STATISTICS_DEFINE(loom_scf_layout_transport_statistics,
                            loom_scf_layout_transport_statistics_t,
                            LOOM_SCF_LAYOUT_TRANSPORT_STATISTICS)

static const loom_pass_info_t kScfLayoutTransportPassInfo = {
    .name = IREE_SVL("decompose-scf-layout-transports"),
    .description =
        IREE_SVL("Carry dynamic strided layouts as scalar SCF payloads."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_scf_layout_transport_statistics_layout,
};

const loom_pass_info_t* loom_decompose_scf_layout_transports_pass_info(void) {
  return &kScfLayoutTransportPassInfo;
}

typedef enum loom_scf_layout_projection_kind_e {
  LOOM_SCF_LAYOUT_PROJECTION_VALUE = 0,
  LOOM_SCF_LAYOUT_PROJECTION_CONSTANT = 1,
  LOOM_SCF_LAYOUT_PROJECTION_CANDIDATE_AXIS = 2,
} loom_scf_layout_projection_kind_t;

typedef enum loom_scf_layout_endpoint_e {
  // Semantic result of the structured operation.
  LOOM_SCF_LAYOUT_ENDPOINT_RESULT = 0,
  // Counted-loop body or condition-loop before-region argument.
  LOOM_SCF_LAYOUT_ENDPOINT_ENTRY = 1,
  // Condition-loop after-region argument.
  LOOM_SCF_LAYOUT_ENDPOINT_BODY = 2,
} loom_scf_layout_endpoint_t;

typedef struct loom_scf_layout_projection_t {
  // How the scalar stride is obtained in the source scope.
  loom_scf_layout_projection_kind_t kind;
  union {
    // Existing index SSA value for VALUE.
    loom_value_id_t value;
    // Exact nonnegative element stride for CONSTANT.
    int64_t constant;
    struct {
      // Candidate whose scalar result supplies the stride.
      iree_host_size_t candidate;
      // Layout axis selecting the candidate's scalar result.
      uint8_t axis;
      // Structured endpoint supplying the scalar stride.
      loom_scf_layout_endpoint_t endpoint;
    } candidate_axis;
  } value;
} loom_scf_layout_projection_t;

typedef struct loom_scf_layout_dependency_t {
  // Candidate requiring the source candidate to be decomposed.
  iree_host_size_t target;
  // Next dependent candidate of the same source.
  iree_host_size_t next;
} loom_scf_layout_dependency_t;

typedef struct loom_scf_layout_candidate_reference_t {
  // Candidate represented by the indexed semantic value.
  iree_host_size_t candidate;
  // Candidate endpoint represented by the indexed semantic value.
  loom_scf_layout_endpoint_t endpoint;
} loom_scf_layout_candidate_reference_t;

typedef enum loom_scf_layout_rewrite_state_e {
  LOOM_SCF_LAYOUT_REWRITE_PENDING = 0,
  LOOM_SCF_LAYOUT_REWRITE_ACTIVE = 1,
  LOOM_SCF_LAYOUT_REWRITE_DONE = 2,
} loom_scf_layout_rewrite_state_t;

typedef struct loom_scf_layout_candidate_t {
  // Structured operation producing the semantic layout.
  loom_op_t* op;
  // Original layout-valued result.
  loom_value_id_t result;
  // Original role-qualified encoding type.
  loom_type_t type;
  // Static stride or INT64_MIN for each transported dynamic axis.
  int64_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Scalar operation result indexed by layout axis; static axes are invalid.
  loom_value_id_t axis_values[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Scalar loop-entry argument indexed by layout axis.
  loom_value_id_t entry_axis_values[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Scalar condition-loop body argument indexed by layout axis.
  loom_value_id_t body_axis_values[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Original counted-loop body or condition-loop before-region argument.
  loom_value_id_t entry_value;
  // Original condition-loop after-region argument.
  loom_value_id_t body_value;
  // Dynamic axes in transported tuple order.
  uint8_t dynamic_axes[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Source-major scalar projections for every dynamic axis.
  loom_scf_layout_projection_t* projections;
  // First dependent candidate, or IREE_HOST_SIZE_MAX.
  iree_host_size_t first_dependent;
  // Number of incoming descriptor sources.
  iree_host_size_t source_count;
  // First candidate belonging to the same structured operation.
  iree_host_size_t operation_begin;
  // Candidate after the last one belonging to the same operation.
  iree_host_size_t operation_end;
  // Result column within the original operation.
  uint16_t result_index;
  // Layout rank.
  uint8_t rank;
  // Number of non-exact axes carried by the structured operation.
  uint8_t dynamic_count;
  // Whether every source can provide this descriptor.
  bool selected;
  // Dependency-driven operation rewrite state.
  loom_scf_layout_rewrite_state_t rewrite_state;
} loom_scf_layout_candidate_t;

typedef struct loom_scf_layout_transport_plan_t {
  // Active pass instance for statistics and fact ownership.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // Rewriter maintaining use lists.
  loom_rewriter_t* rewriter;
  // Function-local scratch storage.
  iree_arena_allocator_t* arena;
  // Borrowed function value facts.
  loom_value_fact_table_t* facts;
  // Dense function-local value domain.
  loom_local_value_domain_t domain;
  // Candidate endpoint by local value ordinal.
  loom_scf_layout_candidate_reference_t* candidate_references;
  // Structured results that may be scalarized, in operation postorder.
  loom_scf_layout_candidate_t* candidates;
  // Number of populated candidates.
  iree_host_size_t candidate_count;
  // Allocated candidate capacity.
  iree_host_size_t candidate_capacity;
  // Reverse dependency edges used to reject incomplete structured rewrites.
  loom_scf_layout_dependency_t* dependencies;
  // Number of populated dependency edges.
  iree_host_size_t dependency_count;
  // Allocated dependency capacity.
  iree_host_size_t dependency_capacity;
  // Whether any structured operation was rewritten.
  bool changed;
} loom_scf_layout_transport_plan_t;

static bool loom_scf_layout_is_supported_operation(const loom_op_t* op) {
  return loom_scf_if_isa(op) || loom_scf_switch_isa(op) ||
         loom_scf_select_isa(op) || loom_scf_lookup_isa(op) ||
         loom_scf_for_isa(op) || loom_scf_while_isa(op);
}

static iree_host_size_t loom_scf_layout_source_count(const loom_op_t* op) {
  if (loom_scf_if_isa(op) || loom_scf_switch_isa(op)) {
    return op->region_count;
  }
  if (loom_scf_select_isa(op)) {
    return 2;
  }
  if (loom_scf_lookup_isa(op)) {
    return (iree_host_size_t)loom_scf_lookup_case_keys(op).count + 1;
  }
  if (loom_scf_for_isa(op)) {
    return 2;
  }
  if (loom_scf_while_isa(op)) {
    return 3;
  }
  return 0;
}

static bool loom_scf_layout_describe(
    const loom_scf_layout_transport_plan_t* plan, loom_value_id_t value_id,
    loom_scf_layout_candidate_t* out_candidate) {
  const loom_type_t type = loom_module_value_type(plan->module, value_id);
  if (!loom_type_is_encoding(type) ||
      loom_type_encoding_role(type) != LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
    return false;
  }

  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_value_address_layout(&plan->facts->context, value_id,
                                                &layout) ||
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      layout.rank == 0 || layout.rank > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK ||
      !layout.strides) {
    return false;
  }

  *out_candidate = (loom_scf_layout_candidate_t){
      .result = value_id,
      .type = type,
      .entry_value = LOOM_VALUE_ID_INVALID,
      .body_value = LOOM_VALUE_ID_INVALID,
      .first_dependent = IREE_HOST_SIZE_MAX,
      .rank = layout.rank,
      .selected = true,
  };
  for (uint8_t axis = 0; axis < layout.rank; ++axis) {
    out_candidate->axis_values[axis] = LOOM_VALUE_ID_INVALID;
    out_candidate->entry_axis_values[axis] = LOOM_VALUE_ID_INVALID;
    out_candidate->body_axis_values[axis] = LOOM_VALUE_ID_INVALID;
    int64_t exact_stride = 0;
    if (loom_value_facts_as_exact_i64(layout.strides[axis], &exact_stride)) {
      if (exact_stride < 0) {
        return false;
      }
      out_candidate->static_strides[axis] = exact_stride;
    } else {
      out_candidate->static_strides[axis] = INT64_MIN;
      out_candidate->dynamic_axes[out_candidate->dynamic_count++] = axis;
    }
  }
  return out_candidate->dynamic_count != 0;
}

static bool loom_scf_layout_result_is_candidate(
    const loom_scf_layout_transport_plan_t* plan, const loom_op_t* op,
    uint16_t result_index) {
  const loom_value_id_t result = loom_op_results(op)[result_index];
  const loom_type_t type = loom_module_value_type(plan->module, result);
  return loom_type_is_encoding(type) &&
         loom_type_encoding_role(type) == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT;
}

// The specialized transport represents a condition loop as one positional
// recurrence tuple. General condition loops own independent header (A) and
// body/result (B) schemes and are handled by boundary projection instead. Keep
// this while-only classification out of the ordinary per-op collection path.
IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static bool
loom_scf_layout_while_is_supported(const loom_scf_layout_transport_plan_t* plan,
                                   const loom_op_t* op) {
  bool has_layout_result = false;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    has_layout_result |= loom_scf_layout_result_is_candidate(plan, op, i);
  }
  if (!has_layout_result) {
    return true;
  }

  const loom_block_t* before =
      loom_region_const_entry_block(loom_scf_while_before(op));
  if (before == NULL || before->arg_count != op->result_count) {
    return false;
  }

  const loom_type_value_remap_t remap = {
      .source_values = before->arg_ids,
      .target_values = loom_op_const_results(op),
      .count = op->result_count,
      .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
  };
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_type_t header_type =
        loom_block_arg_type(plan->module, before, i);
    const loom_type_t result_type =
        loom_module_value_type(plan->module, loom_op_const_results(op)[i]);
    if (!loom_type_equal_after_value_remap(plan->module, header_type,
                                           result_type, &remap)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_scf_layout_add_candidate(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* op,
    uint16_t result_index) {
  if (!loom_scf_layout_result_is_candidate(plan, op, result_index)) {
    return iree_ok_status();
  }
  const loom_value_id_t result = loom_op_results(op)[result_index];
  const loom_type_t type = loom_module_value_type(plan->module, result);
  if (plan->candidate_count == plan->candidate_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->candidate_count, plan->candidate_count + 1,
        sizeof(*plan->candidates), &plan->candidate_capacity,
        (void**)&plan->candidates));
  }
  plan->candidates[plan->candidate_count++] = (loom_scf_layout_candidate_t){
      .op = op,
      .result = result,
      .type = type,
      .first_dependent = IREE_HOST_SIZE_MAX,
      .source_count = loom_scf_layout_source_count(op),
      .result_index = result_index,
  };
  return iree_ok_status();
}

static void loom_scf_layout_describe_candidates(
    loom_scf_layout_transport_plan_t* plan) {
  iree_host_size_t described_count = 0;
  for (iree_host_size_t i = 0; i < plan->candidate_count; ++i) {
    const loom_scf_layout_candidate_t collected = plan->candidates[i];
    loom_scf_layout_candidate_t described;
    if (!loom_scf_layout_describe(plan, collected.result, &described)) {
      continue;
    }
    described.op = collected.op;
    described.result_index = collected.result_index;
    described.source_count = collected.source_count;
    if (loom_scf_for_isa(collected.op)) {
      loom_block_t* body =
          loom_region_entry_block(loom_scf_for_body(collected.op));
      IREE_ASSERT(body);
      described.entry_value =
          loom_block_arg_id(body, (uint16_t)(1 + collected.result_index));
    } else if (loom_scf_while_isa(collected.op)) {
      loom_block_t* before =
          loom_region_entry_block(loom_scf_while_before(collected.op));
      loom_block_t* after =
          loom_region_entry_block(loom_scf_while_after(collected.op));
      IREE_ASSERT(before && after);
      described.entry_value = loom_block_arg_id(before, collected.result_index);
      described.body_value = loom_block_arg_id(after, collected.result_index);
    }
    plan->candidates[described_count++] = described;
  }
  plan->candidate_count = described_count;
  for (iree_host_size_t begin = 0; begin < described_count;) {
    iree_host_size_t end = begin + 1;
    while (end < described_count &&
           plan->candidates[end].op == plan->candidates[begin].op) {
      ++end;
    }
    for (iree_host_size_t i = begin; i < end; ++i) {
      plan->candidates[i].operation_begin = begin;
      plan->candidates[i].operation_end = end;
    }
    begin = end;
  }
}

static iree_status_t loom_scf_layout_collect(void* user_data, loom_op_t* op,
                                             const loom_walk_context_t* context,
                                             loom_walk_result_t* out_result) {
  (void)context;
  loom_scf_layout_transport_plan_t* plan = user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_scf_layout_is_supported_operation(op)) {
    return iree_ok_status();
  }
  if (loom_scf_while_isa(op) && !loom_scf_layout_while_is_supported(plan, op)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_layout_add_candidate(plan, op, i));
  }
  return iree_ok_status();
}

static loom_scf_layout_candidate_reference_t
loom_scf_layout_candidate_reference(
    const loom_scf_layout_transport_plan_t* plan, loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(&plan->domain, value_id);
  return ordinal == LOOM_VALUE_ORDINAL_INVALID
             ? (loom_scf_layout_candidate_reference_t){
                   .candidate = IREE_HOST_SIZE_MAX,
               }
             : plan->candidate_references[ordinal];
}

static iree_host_size_t loom_scf_layout_candidate_index(
    const loom_scf_layout_transport_plan_t* plan, loom_value_id_t value_id) {
  return loom_scf_layout_candidate_reference(plan, value_id).candidate;
}

static void loom_scf_layout_index_candidate_endpoint(
    loom_scf_layout_transport_plan_t* plan, loom_value_id_t value_id,
    iree_host_size_t candidate, loom_scf_layout_endpoint_t endpoint) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return;
  }
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(&plan->domain, value_id);
  IREE_ASSERT_EQ(plan->candidate_references[ordinal].candidate,
                 IREE_HOST_SIZE_MAX);
  plan->candidate_references[ordinal] = (loom_scf_layout_candidate_reference_t){
      .candidate = candidate,
      .endpoint = endpoint,
  };
}

static iree_status_t loom_scf_layout_index_candidates(
    loom_scf_layout_transport_plan_t* plan) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, plan->domain.value_count,
                                sizeof(*plan->candidate_references),
                                (void**)&plan->candidate_references));
  for (loom_value_ordinal_t i = 0; i < plan->domain.value_count; ++i) {
    plan->candidate_references[i] = (loom_scf_layout_candidate_reference_t){
        .candidate = IREE_HOST_SIZE_MAX,
    };
  }
  for (iree_host_size_t i = 0; i < plan->candidate_count; ++i) {
    loom_scf_layout_candidate_t* candidate = &plan->candidates[i];
    loom_scf_layout_index_candidate_endpoint(plan, candidate->result, i,
                                             LOOM_SCF_LAYOUT_ENDPOINT_RESULT);
    loom_scf_layout_index_candidate_endpoint(plan, candidate->entry_value, i,
                                             LOOM_SCF_LAYOUT_ENDPOINT_ENTRY);
    loom_scf_layout_index_candidate_endpoint(plan, candidate->body_value, i,
                                             LOOM_SCF_LAYOUT_ENDPOINT_BODY);
  }
  return iree_ok_status();
}

static bool loom_scf_layout_direct_projection(
    const loom_scf_layout_transport_plan_t* plan, loom_value_id_t source,
    uint8_t rank, uint8_t axis, loom_scf_layout_projection_t* out_projection) {
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_value_address_layout(&plan->facts->context, source,
                                                &layout) ||
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      layout.rank != rank || !layout.strides) {
    return false;
  }
  int64_t exact_stride = 0;
  if (loom_value_facts_as_exact_i64(layout.strides[axis], &exact_stride) &&
      exact_stride >= 0) {
    *out_projection = (loom_scf_layout_projection_t){
        .kind = LOOM_SCF_LAYOUT_PROJECTION_CONSTANT,
        .value.constant = exact_stride,
    };
    return true;
  }
  const loom_value_fact_layout_strides_t bindings =
      loom_encoding_query_value_layout_strides(&plan->facts->context, source);
  if (bindings.count != rank ||
      bindings.values[axis] == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  *out_projection = (loom_scf_layout_projection_t){
      .kind = LOOM_SCF_LAYOUT_PROJECTION_VALUE,
      .value.value = bindings.values[axis],
  };
  return true;
}

static bool loom_scf_layout_projection(
    const loom_scf_layout_transport_plan_t* plan, loom_value_id_t source,
    const loom_scf_layout_candidate_t* target, uint8_t axis,
    loom_scf_layout_projection_t* out_projection,
    iree_host_size_t* out_dependency) {
  *out_dependency = IREE_HOST_SIZE_MAX;
  const loom_scf_layout_candidate_reference_t source_reference =
      loom_scf_layout_candidate_reference(plan, source);
  if (source_reference.candidate != IREE_HOST_SIZE_MAX) {
    const loom_scf_layout_candidate_t* source_candidate =
        &plan->candidates[source_reference.candidate];
    if (source_candidate->rank != target->rank) {
      return false;
    }
    if (source_candidate->static_strides[axis] != INT64_MIN) {
      *out_projection = (loom_scf_layout_projection_t){
          .kind = LOOM_SCF_LAYOUT_PROJECTION_CONSTANT,
          .value.constant = source_candidate->static_strides[axis],
      };
      return true;
    }
    *out_projection = (loom_scf_layout_projection_t){
        .kind = LOOM_SCF_LAYOUT_PROJECTION_CANDIDATE_AXIS,
        .value.candidate_axis =
            {
                .candidate = source_reference.candidate,
                .axis = axis,
                .endpoint = source_reference.endpoint,
            },
    };
    *out_dependency = source_reference.candidate;
    return true;
  }
  return loom_scf_layout_direct_projection(plan, source, target->rank, axis,
                                           out_projection);
}

static iree_status_t loom_scf_layout_add_dependency(
    loom_scf_layout_transport_plan_t* plan, iree_host_size_t source,
    iree_host_size_t target) {
  if (source == target) {
    return iree_ok_status();
  }
  if (plan->dependency_count == plan->dependency_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->dependency_count, plan->dependency_count + 1,
        sizeof(*plan->dependencies), &plan->dependency_capacity,
        (void**)&plan->dependencies));
  }
  plan->dependencies[plan->dependency_count] = (loom_scf_layout_dependency_t){
      .target = target,
      .next = plan->candidates[source].first_dependent,
  };
  plan->candidates[source].first_dependent = plan->dependency_count++;
  return iree_ok_status();
}

static bool loom_scf_layout_candidate_source(
    const loom_scf_layout_candidate_t* candidate, iree_host_size_t source_index,
    loom_value_id_t* out_source) {
  loom_op_t* op = candidate->op;
  if (loom_scf_if_isa(op) || loom_scf_switch_isa(op)) {
    loom_op_t* yield =
        loom_scf_region_terminator(loom_op_regions(op)[(uint8_t)source_index]);
    if (!yield || candidate->result_index >= yield->operand_count) {
      return false;
    }
    *out_source = loom_op_const_operands(yield)[candidate->result_index];
    return true;
  }
  if (loom_scf_select_isa(op)) {
    *out_source = source_index == 0 ? loom_scf_select_true_value(op)
                                    : loom_scf_select_false_value(op);
    return true;
  }
  if (loom_scf_lookup_isa(op)) {
    const loom_value_slice_t values = loom_scf_lookup_values(op);
    const iree_host_size_t value_index =
        source_index * op->result_count + candidate->result_index;
    if (value_index >= values.count) {
      return false;
    }
    *out_source = values.values[value_index];
    return true;
  }
  if (loom_scf_for_isa(op)) {
    if (source_index == 0) {
      *out_source = loom_scf_for_iter_args(op).values[candidate->result_index];
      return true;
    }
    loom_op_t* yield = loom_scf_region_terminator(loom_scf_for_body(op));
    if (!yield || candidate->result_index >= yield->operand_count) {
      return false;
    }
    *out_source = loom_scf_yield_values(yield).values[candidate->result_index];
    return true;
  }
  if (loom_scf_while_isa(op)) {
    if (source_index == 0) {
      *out_source =
          loom_scf_while_iter_args(op).values[candidate->result_index];
      return true;
    }
    if (source_index == 1) {
      loom_block_t* before = loom_region_entry_block(loom_scf_while_before(op));
      loom_op_t* condition = before ? before->last_op : NULL;
      if (!condition || !loom_scf_condition_isa(condition) ||
          candidate->result_index >=
              loom_scf_condition_forwarded(condition).count) {
        return false;
      }
      *out_source = loom_scf_condition_forwarded(condition)
                        .values[candidate->result_index];
      return true;
    }
    loom_op_t* yield = loom_scf_region_terminator(loom_scf_while_after(op));
    if (!yield || candidate->result_index >= yield->operand_count) {
      return false;
    }
    *out_source = loom_scf_yield_values(yield).values[candidate->result_index];
    return true;
  }
  return false;
}

static bool loom_scf_layout_operation_owns_value(const loom_module_t* module,
                                                 const loom_op_t* op,
                                                 loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_value_is_block_arg(value)) {
    return loom_value_def_op(value) == op;
  }
  const loom_block_t* block = loom_value_def_block(value);
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_region_try_block_index(regions[i], block, NULL)) {
      return true;
    }
  }
  return false;
}

// Returns true when scalarizing |candidate| would strand an SSA reference in a
// type owned by the operation signature being rebuilt. External and nested
// carrier types can follow reconstructed values through ordinary replacement.
// Results and direct region arguments must instead be decomposed by their
// owning transport before this pass changes the tuple scheme.
static bool loom_scf_layout_has_internal_type_user(
    const loom_scf_layout_transport_plan_t* plan,
    const loom_scf_layout_candidate_t* candidate) {
  const loom_value_id_t endpoints[] = {
      candidate->result,
      candidate->entry_value,
      candidate->body_value,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(endpoints); ++i) {
    const loom_value_id_t endpoint = endpoints[i];
    if (endpoint == LOOM_VALUE_ID_INVALID ||
        !loom_module_value_has_type_uses(plan->module, endpoint)) {
      continue;
    }
    loom_type_use_iterator_t users;
    loom_module_value_type_users(plan->module, endpoint, &users);
    for (loom_value_id_t user = loom_type_users_next(&users);
         user != LOOM_VALUE_ID_INVALID; user = loom_type_users_next(&users)) {
      if (loom_scf_layout_operation_owns_value(plan->module, candidate->op,
                                               user)) {
        return true;
      }
    }
  }
  return false;
}

static iree_status_t loom_scf_layout_preflight_candidate(
    loom_scf_layout_transport_plan_t* plan, iree_host_size_t candidate_index) {
  loom_scf_layout_candidate_t* candidate = &plan->candidates[candidate_index];
  if (loom_scf_layout_has_internal_type_user(plan, candidate)) {
    // Rebuilding any column changes the operation's result and region-argument
    // identities, so retain the complete tuple until its dependent carriers
    // have been decomposed.
    for (iree_host_size_t i = candidate->operation_begin;
         i < candidate->operation_end; ++i) {
      plan->candidates[i].selected = false;
    }
  }
  const loom_tied_result_t* tied_results = loom_op_tied_results(candidate->op);
  for (uint16_t i = 0; i < candidate->op->tied_result_count; ++i) {
    if (tied_results[i].result_index == candidate->result_index) {
      candidate->selected = false;
    }
  }
  iree_host_size_t projection_count = 0;
  if (!iree_host_size_checked_mul(candidate->source_count,
                                  candidate->dynamic_count,
                                  &projection_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "layout projection count overflow");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, projection_count, sizeof(*candidate->projections),
      (void**)&candidate->projections));

  for (iree_host_size_t source_index = 0;
       source_index < candidate->source_count; ++source_index) {
    loom_value_id_t source = LOOM_VALUE_ID_INVALID;
    if (!loom_scf_layout_candidate_source(candidate, source_index, &source)) {
      candidate->selected = false;
      continue;
    }
    for (uint8_t i = 0; i < candidate->dynamic_count; ++i) {
      const uint8_t axis = candidate->dynamic_axes[i];
      loom_scf_layout_projection_t* projection =
          &candidate->projections[source_index * candidate->dynamic_count + i];
      iree_host_size_t dependency = IREE_HOST_SIZE_MAX;
      if (!loom_scf_layout_projection(plan, source, candidate, axis, projection,
                                      &dependency)) {
        candidate->selected = false;
      } else if (dependency != IREE_HOST_SIZE_MAX) {
        IREE_RETURN_IF_ERROR(
            loom_scf_layout_add_dependency(plan, dependency, candidate_index));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_layout_propagate_unavailable(
    loom_scf_layout_transport_plan_t* plan) {
  iree_host_size_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->candidate_count, sizeof(*queue), (void**)&queue));
  iree_host_size_t head = 0;
  iree_host_size_t tail = 0;
  for (iree_host_size_t i = 0; i < plan->candidate_count; ++i) {
    if (!plan->candidates[i].selected) {
      queue[tail++] = i;
    }
  }
  while (head < tail) {
    const iree_host_size_t source = queue[head++];
    for (iree_host_size_t edge = plan->candidates[source].first_dependent;
         edge != IREE_HOST_SIZE_MAX; edge = plan->dependencies[edge].next) {
      const iree_host_size_t target = plan->dependencies[edge].target;
      if (plan->candidates[target].selected) {
        plan->candidates[target].selected = false;
        queue[tail++] = target;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_layout_select_candidates(
    loom_scf_layout_transport_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->candidate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_layout_preflight_candidate(plan, i));
  }
  return loom_scf_layout_propagate_unavailable(plan);
}

static iree_status_t loom_scf_layout_name_axis(
    loom_scf_layout_transport_plan_t* plan, loom_value_id_t semantic_value,
    uint8_t axis, loom_value_id_t value) {
  char suffix[32] = {0};
  const int length =
      iree_snprintf(suffix, sizeof(suffix), "stride_%" PRIu8, axis);
  if (length <= 0 || (iree_host_size_t)length >= sizeof(suffix)) {
    return iree_ok_status();
  }
  return loom_rewriter_try_set_derived_value_name(
      plan->rewriter, semantic_value, value,
      iree_make_string_view(suffix, (iree_host_size_t)length));
}

static loom_value_id_t* loom_scf_layout_endpoint_axis_values(
    loom_scf_layout_candidate_t* candidate,
    loom_scf_layout_endpoint_t endpoint) {
  switch (endpoint) {
    case LOOM_SCF_LAYOUT_ENDPOINT_RESULT:
      return candidate->axis_values;
    case LOOM_SCF_LAYOUT_ENDPOINT_ENTRY:
      return candidate->entry_axis_values;
    case LOOM_SCF_LAYOUT_ENDPOINT_BODY:
      return candidate->body_axis_values;
    default:
      IREE_ASSERT(false);
      return candidate->axis_values;
  }
}

static iree_status_t loom_scf_layout_materialize_projection(
    loom_scf_layout_transport_plan_t* plan,
    const loom_scf_layout_projection_t* projection, loom_op_t* anchor,
    loom_value_id_t* out_value) {
  switch (projection->kind) {
    case LOOM_SCF_LAYOUT_PROJECTION_VALUE:
      *out_value = projection->value.value;
      return iree_ok_status();
    case LOOM_SCF_LAYOUT_PROJECTION_CONSTANT: {
      loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter->builder);
      loom_builder_set_before(&plan->rewriter->builder, anchor);
      loom_op_t* constant_op = NULL;
      iree_status_t status = loom_index_constant_build(
          &plan->rewriter->builder, loom_attr_i64(projection->value.constant),
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), anchor->location,
          &constant_op);
      loom_builder_restore(&plan->rewriter->builder, saved_ip);
      IREE_RETURN_IF_ERROR(status);
      *out_value = loom_index_constant_result(constant_op);
      return iree_ok_status();
    }
    case LOOM_SCF_LAYOUT_PROJECTION_CANDIDATE_AXIS: {
      loom_scf_layout_candidate_t* source =
          &plan->candidates[projection->value.candidate_axis.candidate];
      loom_value_id_t* axis_values = loom_scf_layout_endpoint_axis_values(
          source, projection->value.candidate_axis.endpoint);
      *out_value = axis_values[projection->value.candidate_axis.axis];
      IREE_ASSERT(*out_value != LOOM_VALUE_ID_INVALID);
      return iree_ok_status();
    }
    default:
      IREE_ASSERT(false);
      return iree_ok_status();
  }
}

static iree_status_t loom_scf_layout_materialize_source(
    loom_scf_layout_transport_plan_t* plan,
    const loom_scf_layout_candidate_t* candidate, iree_host_size_t source_index,
    loom_op_t* anchor, loom_value_id_t* out_values) {
  for (uint8_t i = 0; i < candidate->dynamic_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_layout_materialize_projection(
        plan,
        &candidate->projections[source_index * candidate->dynamic_count + i],
        anchor, &out_values[i]));
  }
  return iree_ok_status();
}

typedef enum loom_scf_layout_insertion_e {
  LOOM_SCF_LAYOUT_INSERT_BEFORE = 0,
  LOOM_SCF_LAYOUT_INSERT_AFTER = 1,
} loom_scf_layout_insertion_t;

static iree_status_t loom_scf_layout_build_reconstruction_at(
    loom_scf_layout_transport_plan_t* plan,
    loom_scf_layout_candidate_t* candidate, loom_value_id_t semantic_value,
    loom_value_id_t* axis_values, loom_op_t* anchor,
    loom_scf_layout_insertion_t insertion, loom_value_id_t* out_replacement) {
  loom_value_id_t dynamic_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  for (uint8_t i = 0; i < candidate->dynamic_count; ++i) {
    const uint8_t axis = candidate->dynamic_axes[i];
    dynamic_strides[i] = axis_values[axis];
    IREE_ASSERT(dynamic_strides[i] != LOOM_VALUE_ID_INVALID);
    IREE_RETURN_IF_ERROR(loom_scf_layout_name_axis(plan, semantic_value, axis,
                                                   dynamic_strides[i]));
  }
  loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter->builder);
  if (insertion == LOOM_SCF_LAYOUT_INSERT_BEFORE) {
    loom_builder_set_before(&plan->rewriter->builder, anchor);
  } else {
    loom_builder_set_after(&plan->rewriter->builder, anchor);
  }
  loom_op_t* layout_op = NULL;
  iree_status_t status = loom_encoding_layout_strided_build(
      &plan->rewriter->builder, dynamic_strides, candidate->dynamic_count,
      candidate->static_strides, candidate->rank, candidate->type,
      anchor->location, &layout_op);
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  *out_replacement = loom_encoding_layout_strided_result(layout_op);
  return iree_ok_status();
}

static iree_status_t loom_scf_layout_build_result_reconstruction(
    loom_scf_layout_transport_plan_t* plan,
    loom_scf_layout_candidate_t* candidate, loom_op_t* anchor,
    loom_value_id_t* out_replacement) {
  return loom_scf_layout_build_reconstruction_at(
      plan, candidate, candidate->result, candidate->axis_values, anchor,
      LOOM_SCF_LAYOUT_INSERT_AFTER, out_replacement);
}

static iree_status_t loom_scf_layout_copy_comments(loom_module_t* module,
                                                   const loom_op_t* source,
                                                   loom_op_t* target) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, source, &comment_count);
  return comment_count == 0 ? iree_ok_status()
                            : loom_module_attach_op_comments(
                                  module, target, comments, comment_count);
}

static iree_status_t loom_scf_layout_compute_repacking(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* op,
    uint16_t** out_new_indices, loom_type_t** out_result_types,
    uint16_t* out_result_count) {
  iree_host_size_t final_count = op->result_count;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(op)[i]);
    if (candidate_index != IREE_HOST_SIZE_MAX &&
        plan->candidates[candidate_index].selected) {
      final_count += plan->candidates[candidate_index].dynamic_count - 1;
    }
  }
  if (final_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "structured layout result tuple exceeds %u values",
                            (unsigned)UINT16_MAX);
  }

  uint16_t* new_indices = NULL;
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, op->result_count,
                                                 sizeof(*new_indices),
                                                 (void**)&new_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, final_count, sizeof(*result_types), (void**)&result_types));
  uint16_t next = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    new_indices[i] = next;
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(op)[i]);
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !plan->candidates[candidate_index].selected) {
      result_types[next++] =
          loom_module_value_type(plan->module, loom_op_const_results(op)[i]);
      continue;
    }
    const loom_scf_layout_candidate_t* candidate =
        &plan->candidates[candidate_index];
    for (uint8_t j = 0; j < candidate->dynamic_count; ++j) {
      result_types[next++] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    }
  }
  IREE_ASSERT_EQ(next, final_count);
  *out_new_indices = new_indices;
  *out_result_types = result_types;
  *out_result_count = (uint16_t)final_count;
  return iree_ok_status();
}

static iree_status_t loom_scf_layout_repack_source_values(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* op,
    iree_host_size_t source_index, loom_value_slice_t source_values,
    loom_op_t* anchor, uint16_t result_count, loom_value_id_t* out_values) {
  IREE_ASSERT_EQ(source_values.count, op->result_count);
  uint16_t next = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(op)[i]);
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !plan->candidates[candidate_index].selected) {
      out_values[next++] = source_values.values[i];
      continue;
    }
    const loom_scf_layout_candidate_t* candidate =
        &plan->candidates[candidate_index];
    IREE_RETURN_IF_ERROR(loom_scf_layout_materialize_source(
        plan, candidate, source_index, anchor, out_values + next));
    next += candidate->dynamic_count;
  }
  IREE_ASSERT_EQ(next, result_count);
  return iree_ok_status();
}

static void loom_scf_layout_bind_endpoint(
    loom_scf_layout_candidate_t* candidate, loom_scf_layout_endpoint_t endpoint,
    const loom_value_id_t* values) {
  loom_value_id_t* axis_values =
      loom_scf_layout_endpoint_axis_values(candidate, endpoint);
  for (uint8_t i = 0; i < candidate->dynamic_count; ++i) {
    axis_values[candidate->dynamic_axes[i]] = values[i];
  }
}

static iree_status_t loom_scf_layout_reconstruct_block_arguments(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* old_op,
    loom_block_t* old_block, loom_block_t* new_block, uint16_t argument_offset,
    loom_scf_layout_endpoint_t endpoint, const uint16_t* new_indices,
    loom_op_t* terminator) {
  for (uint16_t i = 0; i < old_op->result_count; ++i) {
    const loom_value_id_t old_argument =
        loom_block_arg_id(old_block, (uint16_t)(argument_offset + i));
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(old_op)[i]);
    loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !plan->candidates[candidate_index].selected) {
      replacement = loom_block_arg_id(
          new_block, (uint16_t)(argument_offset + new_indices[i]));
      const loom_type_t old_type =
          loom_module_value_type(plan->module, old_argument);
      if (!loom_type_equal(loom_module_value_type(plan->module, replacement),
                           old_type)) {
        IREE_RETURN_IF_ERROR(
            loom_module_set_value_type(plan->module, replacement, old_type));
      }
    } else {
      loom_scf_layout_candidate_t* candidate =
          &plan->candidates[candidate_index];
      IREE_RETURN_IF_ERROR(loom_scf_layout_build_reconstruction_at(
          plan, candidate, old_argument,
          loom_scf_layout_endpoint_axis_values(candidate, endpoint), terminator,
          LOOM_SCF_LAYOUT_INSERT_BEFORE, &replacement));
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
        plan->rewriter, old_argument, replacement));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        plan->rewriter, old_argument, replacement));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_layout_reconstruct_results(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* old_op,
    loom_op_t* new_op, const uint16_t* new_indices,
    loom_value_id_t* out_replacements) {
  for (uint16_t i = 0; i < old_op->result_count; ++i) {
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(old_op)[i]);
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !plan->candidates[candidate_index].selected) {
      out_replacements[i] = loom_op_results(new_op)[new_indices[i]];
      continue;
    }
    loom_scf_layout_candidate_t* candidate = &plan->candidates[candidate_index];
    loom_scf_layout_bind_endpoint(candidate, LOOM_SCF_LAYOUT_ENDPOINT_RESULT,
                                  loom_op_results(new_op) + new_indices[i]);
    IREE_RETURN_IF_ERROR(loom_scf_layout_build_result_reconstruction(
        plan, candidate, new_op, &out_replacements[i]));
  }
  return iree_ok_status();
}

typedef enum loom_scf_layout_tied_operand_mapping_e {
  LOOM_SCF_LAYOUT_TIED_OPERAND_UNCHANGED = 0,
  LOOM_SCF_LAYOUT_TIED_OPERAND_LOOKUP = 1,
  LOOM_SCF_LAYOUT_TIED_OPERAND_FOR = 2,
  LOOM_SCF_LAYOUT_TIED_OPERAND_WHILE = 3,
} loom_scf_layout_tied_operand_mapping_t;

static iree_status_t loom_scf_layout_adjust_tied_results(
    loom_scf_layout_transport_plan_t* plan, const loom_op_t* op,
    const uint16_t* new_indices, uint16_t result_count,
    loom_scf_layout_tied_operand_mapping_t operand_mapping,
    loom_tied_result_t** out_tied_results) {
  *out_tied_results = NULL;
  if (op->tied_result_count == 0) {
    return iree_ok_status();
  }
  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, op->tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));
  const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    const loom_tied_result_t old_tie = old_tied_results[i];
    IREE_ASSERT_LT(old_tie.result_index, op->result_count);
    tied_results[i] = old_tie;
    tied_results[i].result_index = new_indices[old_tie.result_index];
    iree_host_size_t new_operand_index = old_tie.operand_index;
    switch (operand_mapping) {
      case LOOM_SCF_LAYOUT_TIED_OPERAND_UNCHANGED:
        break;
      case LOOM_SCF_LAYOUT_TIED_OPERAND_LOOKUP:
        if (old_tie.operand_index != 0) {
          const uint16_t old_value_index = old_tie.operand_index - 1;
          const uint16_t row = old_value_index / op->result_count;
          const uint16_t column = old_value_index % op->result_count;
          new_operand_index =
              1 + (iree_host_size_t)row * result_count + new_indices[column];
        }
        break;
      case LOOM_SCF_LAYOUT_TIED_OPERAND_FOR: {
        const iree_host_size_t old_iter_arg_end =
            3 + (iree_host_size_t)op->result_count;
        if (old_tie.operand_index >= 3 &&
            old_tie.operand_index < old_iter_arg_end) {
          new_operand_index = 3 + new_indices[old_tie.operand_index - 3];
        } else if (old_tie.operand_index >= old_iter_arg_end) {
          new_operand_index = old_tie.operand_index +
                              (iree_host_size_t)result_count - op->result_count;
        }
        break;
      }
      case LOOM_SCF_LAYOUT_TIED_OPERAND_WHILE:
        IREE_ASSERT_LT(old_tie.operand_index, op->result_count);
        new_operand_index = new_indices[old_tie.operand_index];
        break;
      default:
        IREE_ASSERT(false);
        break;
    }
    if (new_operand_index > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "structured layout tie exceeds %u operands",
                              (unsigned)UINT16_MAX);
    }
    tied_results[i].operand_index = (uint16_t)new_operand_index;
  }
  *out_tied_results = tied_results;
  return iree_ok_status();
}

static iree_status_t loom_scf_layout_build_region_selection(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* old_op,
    const loom_type_t* result_types, uint16_t result_count,
    const loom_tied_result_t* tied_results, loom_op_t** out_new_op) {
  loom_builder_t* builder = &plan->rewriter->builder;
  if (loom_scf_if_isa(old_op)) {
    const loom_scf_if_build_flags_t flags =
        loom_scf_if_else_region(old_op) ? LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION
                                        : 0;
    IREE_RETURN_IF_ERROR(loom_scf_if_build(
        builder, flags, loom_scf_if_condition(old_op), result_types,
        result_count, tied_results, old_op->tied_result_count, old_op->location,
        out_new_op));
  } else {
    const loom_attribute_t case_keys = loom_scf_switch_case_keys(old_op);
    IREE_RETURN_IF_ERROR(loom_scf_switch_build(
        builder, loom_scf_switch_selector(old_op), result_types, result_count,
        tied_results, old_op->tied_result_count, case_keys.i64_array,
        case_keys.count, old_op->location, out_new_op));
  }
  (*out_new_op)->instance_flags = old_op->instance_flags;
  return loom_scf_layout_copy_comments(plan->module, old_op, *out_new_op);
}

static iree_status_t loom_scf_layout_rebuild_region_selection(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* old_op) {
  uint16_t* new_indices = NULL;
  loom_type_t* result_types = NULL;
  uint16_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_layout_compute_repacking(
      plan, old_op, &new_indices, &result_types, &result_count));
  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_layout_adjust_tied_results(
      plan, old_op, new_indices, result_count,
      LOOM_SCF_LAYOUT_TIED_OPERAND_UNCHANGED, &tied_results));

  loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter->builder);
  loom_builder_set_before(&plan->rewriter->builder, old_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(plan->rewriter);
  loom_op_t* new_op = NULL;
  iree_status_t status = loom_scf_layout_build_region_selection(
      plan, old_op, result_types, result_count, tied_results, &new_op);
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  for (uint8_t region_index = 0; region_index < old_op->region_count;
       ++region_index) {
    loom_region_t* old_region = loom_op_regions(old_op)[region_index];
    loom_op_t* old_yield = loom_scf_region_terminator(old_region);
    IREE_ASSERT(old_yield);
    loom_value_id_t* yielded = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, result_count, sizeof(*yielded), (void**)&yielded));
    uint16_t next = 0;
    for (uint16_t i = 0; i < old_op->result_count; ++i) {
      const iree_host_size_t candidate_index = loom_scf_layout_candidate_index(
          plan, loom_op_const_results(old_op)[i]);
      if (candidate_index == IREE_HOST_SIZE_MAX ||
          !plan->candidates[candidate_index].selected) {
        yielded[next++] = loom_op_const_operands(old_yield)[i];
        continue;
      }
      const loom_scf_layout_candidate_t* candidate =
          &plan->candidates[candidate_index];
      IREE_RETURN_IF_ERROR(loom_scf_layout_materialize_source(
          plan, candidate, region_index, old_yield, yielded + next));
      next += candidate->dynamic_count;
    }
    IREE_ASSERT_EQ(next, result_count);

    loom_region_t* new_region = loom_op_regions(new_op)[region_index];
    loom_builder_ip_t region_ip =
        loom_builder_enter_region(&plan->rewriter->builder, new_op, new_region);
    loom_op_t* new_yield = NULL;
    status =
        loom_scf_yield_build(&plan->rewriter->builder, yielded, result_count,
                             old_yield->location, &new_yield);
    loom_builder_restore(&plan->rewriter->builder, region_ip);
    IREE_RETURN_IF_ERROR(status);

    loom_block_t* old_block = loom_region_entry_block(old_region);
    loom_op_t* child_op = old_block->first_op;
    while (child_op && child_op != old_yield) {
      loom_op_t* next_child_op = child_op->next_op;
      IREE_RETURN_IF_ERROR(
          loom_rewriter_move_before(plan->rewriter, child_op, new_yield));
      child_op = next_child_op;
    }
  }

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, old_op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  for (uint16_t i = 0; i < old_op->result_count; ++i) {
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(old_op)[i]);
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !plan->candidates[candidate_index].selected) {
      replacements[i] = loom_op_results(new_op)[new_indices[i]];
      continue;
    }
    loom_scf_layout_candidate_t* candidate = &plan->candidates[candidate_index];
    for (uint8_t j = 0; j < candidate->dynamic_count; ++j) {
      const uint8_t axis = candidate->dynamic_axes[j];
      candidate->axis_values[axis] =
          loom_op_results(new_op)[new_indices[i] + j];
    }
    IREE_RETURN_IF_ERROR(loom_scf_layout_build_result_reconstruction(
        plan, candidate, new_op, &replacements[i]));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      plan->rewriter, old_op, replacements, old_op->result_count,
      value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      plan->rewriter, old_op, replacements, old_op->result_count);
}

static iree_status_t loom_scf_layout_rebuild_select(
    loom_scf_layout_transport_plan_t* plan,
    loom_scf_layout_candidate_t* candidate) {
  loom_op_t* old_op = candidate->op;
  loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter->builder);
  loom_builder_set_before(&plan->rewriter->builder, old_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(plan->rewriter);
  loom_value_id_t true_values[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  loom_value_id_t false_values[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  IREE_RETURN_IF_ERROR(loom_scf_layout_materialize_source(plan, candidate, 0,
                                                          old_op, true_values));
  IREE_RETURN_IF_ERROR(loom_scf_layout_materialize_source(
      plan, candidate, 1, old_op, false_values));
  loom_op_t* last_select = NULL;
  for (uint8_t i = 0; i < candidate->dynamic_count; ++i) {
    loom_op_t* scalar_select = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_select_build(
        &plan->rewriter->builder, loom_scf_select_condition(old_op),
        true_values[i], false_values[i],
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), old_op->location,
        &scalar_select));
    const uint8_t axis = candidate->dynamic_axes[i];
    candidate->axis_values[axis] = loom_scf_select_result(scalar_select);
    last_select = scalar_select;
  }
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scf_layout_build_result_reconstruction(
      plan, candidate, last_select, &replacement));
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      plan->rewriter, old_op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(plan->rewriter, old_op,
                                                  &replacement, 1);
}

static iree_status_t loom_scf_layout_rebuild_lookup(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* old_op) {
  uint16_t* new_indices = NULL;
  loom_type_t* result_types = NULL;
  uint16_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_layout_compute_repacking(
      plan, old_op, &new_indices, &result_types, &result_count));
  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_layout_adjust_tied_results(
      plan, old_op, new_indices, result_count,
      LOOM_SCF_LAYOUT_TIED_OPERAND_LOOKUP, &tied_results));
  const loom_attribute_t case_keys = loom_scf_lookup_case_keys(old_op);
  const iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  iree_host_size_t value_count = 0;
  if (!iree_host_size_checked_mul(row_count, result_count, &value_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "layout lookup payload count overflow");
  }
  if (value_count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "structured layout lookup exceeds %u operands",
                            (unsigned)UINT16_MAX);
  }
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, value_count, sizeof(*values), (void**)&values));
  const loom_value_slice_t old_values = loom_scf_lookup_values(old_op);
  iree_host_size_t next = 0;
  for (iree_host_size_t row = 0; row < row_count; ++row) {
    for (uint16_t i = 0; i < old_op->result_count; ++i) {
      const iree_host_size_t candidate_index = loom_scf_layout_candidate_index(
          plan, loom_op_const_results(old_op)[i]);
      if (candidate_index == IREE_HOST_SIZE_MAX ||
          !plan->candidates[candidate_index].selected) {
        values[next++] = old_values.values[row * old_op->result_count + i];
        continue;
      }
      const loom_scf_layout_candidate_t* candidate =
          &plan->candidates[candidate_index];
      IREE_RETURN_IF_ERROR(loom_scf_layout_materialize_source(
          plan, candidate, row, old_op, values + next));
      next += candidate->dynamic_count;
    }
  }
  IREE_ASSERT_EQ(next, value_count);

  loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter->builder);
  loom_builder_set_before(&plan->rewriter->builder, old_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(plan->rewriter);
  loom_op_t* new_op = NULL;
  iree_status_t status = loom_scf_lookup_build(
      &plan->rewriter->builder, loom_scf_lookup_selector(old_op),
      case_keys.i64_array, case_keys.count, values, value_count, result_types,
      result_count, tied_results, old_op->tied_result_count, old_op->location,
      &new_op);
  if (iree_status_is_ok(status)) {
    new_op->instance_flags = old_op->instance_flags;
    status = loom_scf_layout_copy_comments(plan->module, old_op, new_op);
  }
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, old_op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  for (uint16_t i = 0; i < old_op->result_count; ++i) {
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(old_op)[i]);
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !plan->candidates[candidate_index].selected) {
      replacements[i] = loom_op_results(new_op)[new_indices[i]];
      continue;
    }
    loom_scf_layout_candidate_t* candidate = &plan->candidates[candidate_index];
    for (uint8_t j = 0; j < candidate->dynamic_count; ++j) {
      const uint8_t axis = candidate->dynamic_axes[j];
      candidate->axis_values[axis] =
          loom_op_results(new_op)[new_indices[i] + j];
    }
    IREE_RETURN_IF_ERROR(loom_scf_layout_build_result_reconstruction(
        plan, candidate, new_op, &replacements[i]));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      plan->rewriter, old_op, replacements, old_op->result_count,
      value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      plan->rewriter, old_op, replacements, old_op->result_count);
}

static iree_status_t loom_scf_layout_ensure_operation(
    loom_scf_layout_transport_plan_t* plan, iree_host_size_t candidate_index);

static iree_status_t loom_scf_layout_ensure_projection_dependencies(
    loom_scf_layout_transport_plan_t* plan, iree_host_size_t operation_begin,
    iree_host_size_t operation_end, iree_host_size_t source_begin,
    iree_host_size_t source_end) {
  for (iree_host_size_t i = operation_begin; i < operation_end; ++i) {
    const loom_scf_layout_candidate_t* candidate = &plan->candidates[i];
    if (!candidate->selected) {
      continue;
    }
    IREE_ASSERT_LE(source_end, candidate->source_count);
    for (iree_host_size_t source_index = source_begin;
         source_index < source_end; ++source_index) {
      for (uint8_t axis_index = 0; axis_index < candidate->dynamic_count;
           ++axis_index) {
        const loom_scf_layout_projection_t* projection =
            &candidate->projections[source_index * candidate->dynamic_count +
                                    axis_index];
        if (projection->kind != LOOM_SCF_LAYOUT_PROJECTION_CANDIDATE_AXIS) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_scf_layout_ensure_operation(
            plan, projection->value.candidate_axis.candidate));
      }
    }
  }
  return iree_ok_status();
}

static loom_scf_for_build_flags_t loom_scf_layout_for_build_flags(
    const loom_op_t* op) {
  loom_scf_for_build_flags_t flags = 0;
  if (loom_scf_for_pipeline_depth_is_present(op)) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_PIPELINE_DEPTH;
  }
  if (loom_scf_for_unroll_factor_is_present(op)) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
  }
  if (loom_scf_for_has_unroll_policy(op)) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
  }
  if (loom_scf_for_has_unroll_schedule(op)) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_SCHEDULE;
  }
  return flags;
}

static iree_status_t loom_scf_layout_rebuild_for(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* old_op,
    iree_host_size_t operation_begin, iree_host_size_t operation_end) {
  uint16_t* new_indices = NULL;
  loom_type_t* result_types = NULL;
  uint16_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_layout_compute_repacking(
      plan, old_op, &new_indices, &result_types, &result_count));
  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_layout_adjust_tied_results(
      plan, old_op, new_indices, result_count, LOOM_SCF_LAYOUT_TIED_OPERAND_FOR,
      &tied_results));

  loom_region_t* old_body = loom_scf_for_body(old_op);
  loom_block_t* old_block = loom_region_entry_block(old_body);
  loom_op_t* old_yield = loom_scf_region_terminator(old_body);
  IREE_ASSERT(old_block && old_yield);

  loom_value_id_t* initial_values = NULL;
  loom_value_id_t* yielded_values = NULL;
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, result_count,
                                                 sizeof(*initial_values),
                                                 (void**)&initial_values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, result_count,
                                                 sizeof(*yielded_values),
                                                 (void**)&yielded_values));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, old_op->result_count,
                                sizeof(*replacements), (void**)&replacements));

  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(plan->rewriter);
  IREE_RETURN_IF_ERROR(loom_scf_layout_ensure_projection_dependencies(
      plan, operation_begin, operation_end, 0, 1));
  IREE_RETURN_IF_ERROR(loom_scf_layout_repack_source_values(
      plan, old_op, 0, loom_scf_for_iter_args(old_op), old_op, result_count,
      initial_values));

  const loom_scf_for_build_flags_t build_flags =
      loom_scf_layout_for_build_flags(old_op);
  const loom_value_id_t pipeline_depth =
      loom_scf_for_pipeline_depth_is_present(old_op)
          ? loom_scf_for_pipeline_depth(old_op)
          : LOOM_VALUE_ID_INVALID;
  const loom_value_id_t unroll_factor =
      loom_scf_for_unroll_factor_is_present(old_op)
          ? loom_scf_for_unroll_factor(old_op)
          : LOOM_VALUE_ID_INVALID;
  const loom_scf_for_unroll_policy_t unroll_policy =
      loom_scf_for_has_unroll_policy(old_op)
          ? loom_scf_for_unroll_policy(old_op)
          : 0;
  const loom_scf_for_unroll_schedule_t unroll_schedule =
      loom_scf_for_has_unroll_schedule(old_op)
          ? loom_scf_for_unroll_schedule(old_op)
          : 0;

  loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter->builder);
  loom_builder_set_before(&plan->rewriter->builder, old_op);
  loom_op_t* new_loop = NULL;
  iree_status_t status = loom_scf_for_build(
      &plan->rewriter->builder, build_flags, loom_scf_for_lower_bound(old_op),
      loom_scf_for_upper_bound(old_op), loom_scf_for_step(old_op),
      initial_values, result_count, result_types, tied_results,
      old_op->tied_result_count, pipeline_depth, unroll_factor, unroll_policy,
      unroll_schedule, old_op->location, &new_loop);
  if (iree_status_is_ok(status)) {
    new_loop->instance_flags = old_op->instance_flags;
    status = loom_scf_layout_copy_comments(plan->module, old_op, new_loop);
  }
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_block_t* new_block =
      loom_region_entry_block(loom_scf_for_body(new_loop));
  for (uint16_t i = 0; i < old_op->result_count; ++i) {
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(old_op)[i]);
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !plan->candidates[candidate_index].selected) {
      continue;
    }
    loom_scf_layout_candidate_t* candidate = &plan->candidates[candidate_index];
    loom_scf_layout_bind_endpoint(candidate, LOOM_SCF_LAYOUT_ENDPOINT_ENTRY,
                                  new_block->arg_ids + 1 + new_indices[i]);
  }

  IREE_RETURN_IF_ERROR(loom_scf_layout_ensure_projection_dependencies(
      plan, operation_begin, operation_end, 1, 2));
  IREE_RETURN_IF_ERROR(loom_scf_layout_repack_source_values(
      plan, old_op, 1, loom_scf_yield_values(old_yield), old_yield,
      result_count, yielded_values));
  loom_op_t* new_yield = NULL;
  saved_ip = loom_builder_enter_region(&plan->rewriter->builder, new_loop,
                                       loom_scf_for_body(new_loop));
  status = loom_scf_yield_build(&plan->rewriter->builder, yielded_values,
                                result_count, old_yield->location, &new_yield);
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  const loom_value_id_t old_iv = loom_block_arg_id(old_block, 0);
  const loom_value_id_t new_iv = loom_block_arg_id(new_block, 0);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_copy_value_name(plan->rewriter, old_iv, new_iv));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_with(plan->rewriter, old_iv, new_iv));
  IREE_RETURN_IF_ERROR(loom_scf_layout_reconstruct_block_arguments(
      plan, old_op, old_block, new_block, 1, LOOM_SCF_LAYOUT_ENDPOINT_ENTRY,
      new_indices, new_yield));

  loom_op_t* child_op = old_block->first_op;
  while (child_op && child_op != old_yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(plan->rewriter, child_op, new_yield));
    child_op = next_child_op;
  }

  IREE_RETURN_IF_ERROR(loom_scf_layout_reconstruct_results(
      plan, old_op, new_loop, new_indices, replacements));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      plan->rewriter, old_op, replacements, old_op->result_count,
      value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      plan->rewriter, old_op, replacements, old_op->result_count);
}

static iree_status_t loom_scf_layout_rebuild_while(
    loom_scf_layout_transport_plan_t* plan, loom_op_t* old_op,
    iree_host_size_t operation_begin, iree_host_size_t operation_end) {
  uint16_t* new_indices = NULL;
  loom_type_t* result_types = NULL;
  uint16_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_layout_compute_repacking(
      plan, old_op, &new_indices, &result_types, &result_count));
  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_layout_adjust_tied_results(
      plan, old_op, new_indices, result_count,
      LOOM_SCF_LAYOUT_TIED_OPERAND_WHILE, &tied_results));

  loom_block_t* old_before =
      loom_region_entry_block(loom_scf_while_before(old_op));
  loom_block_t* old_after =
      loom_region_entry_block(loom_scf_while_after(old_op));
  loom_op_t* old_condition = old_before ? old_before->last_op : NULL;
  loom_op_t* old_yield =
      loom_scf_region_terminator(loom_scf_while_after(old_op));
  IREE_ASSERT(old_condition && loom_scf_condition_isa(old_condition));
  IREE_ASSERT(old_after && old_yield);

  loom_value_id_t* initial_values = NULL;
  loom_value_id_t* forwarded_values = NULL;
  loom_value_id_t* yielded_values = NULL;
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, result_count,
                                                 sizeof(*initial_values),
                                                 (void**)&initial_values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, result_count,
                                                 sizeof(*forwarded_values),
                                                 (void**)&forwarded_values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, result_count,
                                                 sizeof(*yielded_values),
                                                 (void**)&yielded_values));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, old_op->result_count,
                                sizeof(*replacements), (void**)&replacements));

  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(plan->rewriter);
  IREE_RETURN_IF_ERROR(loom_scf_layout_ensure_projection_dependencies(
      plan, operation_begin, operation_end, 0, 1));
  IREE_RETURN_IF_ERROR(loom_scf_layout_repack_source_values(
      plan, old_op, 0, loom_scf_while_iter_args(old_op), old_op, result_count,
      initial_values));

  loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter->builder);
  loom_builder_set_before(&plan->rewriter->builder, old_op);
  loom_op_t* new_loop = NULL;
  iree_status_t status = loom_scf_while_build(
      &plan->rewriter->builder, initial_values, result_count,
      /*iter_args_types=*/result_types, result_types, result_count,
      tied_results, old_op->tied_result_count, old_op->location, &new_loop);
  if (iree_status_is_ok(status)) {
    new_loop->instance_flags = old_op->instance_flags;
    status = loom_scf_layout_copy_comments(plan->module, old_op, new_loop);
  }
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_block_t* new_before =
      loom_region_entry_block(loom_scf_while_before(new_loop));
  loom_block_t* new_after =
      loom_region_entry_block(loom_scf_while_after(new_loop));
  for (uint16_t i = 0; i < old_op->result_count; ++i) {
    const iree_host_size_t candidate_index =
        loom_scf_layout_candidate_index(plan, loom_op_const_results(old_op)[i]);
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !plan->candidates[candidate_index].selected) {
      continue;
    }
    loom_scf_layout_candidate_t* candidate = &plan->candidates[candidate_index];
    loom_scf_layout_bind_endpoint(candidate, LOOM_SCF_LAYOUT_ENDPOINT_ENTRY,
                                  new_before->arg_ids + new_indices[i]);
    loom_scf_layout_bind_endpoint(candidate, LOOM_SCF_LAYOUT_ENDPOINT_BODY,
                                  new_after->arg_ids + new_indices[i]);
  }

  IREE_RETURN_IF_ERROR(loom_scf_layout_ensure_projection_dependencies(
      plan, operation_begin, operation_end, 1, 3));
  IREE_RETURN_IF_ERROR(loom_scf_layout_repack_source_values(
      plan, old_op, 1, loom_scf_condition_forwarded(old_condition),
      old_condition, result_count, forwarded_values));
  IREE_RETURN_IF_ERROR(loom_scf_layout_repack_source_values(
      plan, old_op, 2, loom_scf_yield_values(old_yield), old_yield,
      result_count, yielded_values));

  loom_op_t* new_condition = NULL;
  saved_ip = loom_builder_enter_region(&plan->rewriter->builder, new_loop,
                                       loom_scf_while_before(new_loop));
  status = loom_scf_condition_build(
      &plan->rewriter->builder, loom_scf_condition_condition(old_condition),
      forwarded_values, result_count, old_condition->location, &new_condition);
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  loom_op_t* new_yield = NULL;
  saved_ip = loom_builder_enter_region(&plan->rewriter->builder, new_loop,
                                       loom_scf_while_after(new_loop));
  status = loom_scf_yield_build(&plan->rewriter->builder, yielded_values,
                                result_count, old_yield->location, &new_yield);
  loom_builder_restore(&plan->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_scf_layout_reconstruct_block_arguments(
      plan, old_op, old_before, new_before, 0, LOOM_SCF_LAYOUT_ENDPOINT_ENTRY,
      new_indices, new_condition));
  IREE_RETURN_IF_ERROR(loom_scf_layout_reconstruct_block_arguments(
      plan, old_op, old_after, new_after, 0, LOOM_SCF_LAYOUT_ENDPOINT_BODY,
      new_indices, new_yield));

  loom_op_t* child_op = old_before->first_op;
  while (child_op && child_op != old_condition) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(plan->rewriter, child_op, new_condition));
    child_op = next_child_op;
  }
  child_op = old_after->first_op;
  while (child_op && child_op != old_yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(plan->rewriter, child_op, new_yield));
    child_op = next_child_op;
  }

  IREE_RETURN_IF_ERROR(loom_scf_layout_reconstruct_results(
      plan, old_op, new_loop, new_indices, replacements));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      plan->rewriter, old_op, replacements, old_op->result_count,
      value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      plan->rewriter, old_op, replacements, old_op->result_count);
}

static iree_status_t loom_scf_layout_ensure_operation(
    loom_scf_layout_transport_plan_t* plan, iree_host_size_t candidate_index) {
  loom_scf_layout_candidate_t* indexed_candidate =
      &plan->candidates[candidate_index];
  const iree_host_size_t begin = indexed_candidate->operation_begin;
  const iree_host_size_t end = indexed_candidate->operation_end;
  const loom_scf_layout_rewrite_state_t state =
      plan->candidates[begin].rewrite_state;
  if (state == LOOM_SCF_LAYOUT_REWRITE_DONE ||
      state == LOOM_SCF_LAYOUT_REWRITE_ACTIVE) {
    return iree_ok_status();
  }

  bool selected = false;
  for (iree_host_size_t i = begin; i < end; ++i) {
    selected |= plan->candidates[i].selected;
    plan->candidates[i].rewrite_state = LOOM_SCF_LAYOUT_REWRITE_ACTIVE;
  }
  if (!selected) {
    for (iree_host_size_t i = begin; i < end; ++i) {
      plan->candidates[i].rewrite_state = LOOM_SCF_LAYOUT_REWRITE_DONE;
    }
    return iree_ok_status();
  }

  loom_op_t* op = plan->candidates[begin].op;
  if (!loom_scf_for_isa(op) && !loom_scf_while_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_scf_layout_ensure_projection_dependencies(
        plan, begin, end, 0, plan->candidates[begin].source_count));
  }
  if (loom_scf_if_isa(op) || loom_scf_switch_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_scf_layout_rebuild_region_selection(plan, op));
  } else if (loom_scf_select_isa(op)) {
    IREE_ASSERT_EQ(end - begin, 1u);
    IREE_RETURN_IF_ERROR(
        loom_scf_layout_rebuild_select(plan, &plan->candidates[begin]));
  } else if (loom_scf_lookup_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_scf_layout_rebuild_lookup(plan, op));
  } else if (loom_scf_for_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_scf_layout_rebuild_for(plan, op, begin, end));
  } else {
    IREE_ASSERT(loom_scf_while_isa(op));
    IREE_RETURN_IF_ERROR(loom_scf_layout_rebuild_while(plan, op, begin, end));
  }

  loom_scf_layout_transport_statistics_t* statistics =
      loom_scf_layout_transport_statistics(plan->pass);
  ++statistics->operations_rewritten;
  for (iree_host_size_t i = begin; i < end; ++i) {
    if (plan->candidates[i].selected) {
      ++statistics->layouts_decomposed;
      statistics->stride_values_inserted += plan->candidates[i].dynamic_count;
    }
    plan->candidates[i].rewrite_state = LOOM_SCF_LAYOUT_REWRITE_DONE;
  }
  plan->changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_layout_apply(
    loom_scf_layout_transport_plan_t* plan, bool* out_changed) {
  *out_changed = false;
  for (iree_host_size_t end = plan->candidate_count; end != 0;) {
    const iree_host_size_t begin = plan->candidates[end - 1].operation_begin;
    IREE_RETURN_IF_ERROR(loom_scf_layout_ensure_operation(plan, begin));
    end = begin;
  }
  *out_changed = plan->changed;
  return iree_ok_status();
}

iree_status_t loom_decompose_scf_layout_transports_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_arena_allocator_t arena;
  iree_arena_initialize(pass->arena->block_pool, &arena);
  loom_scf_layout_transport_plan_t plan = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
      .arena = &arena,
  };

  loom_walk_result_t walk_result;
  iree_status_t status = loom_walk_function(
      module, function, LOOM_WALK_POST_ORDER,
      (loom_walk_callback_t){.fn = loom_scf_layout_collect, .user_data = &plan},
      &walk_result);
  if (iree_status_is_ok(status) && plan.candidate_count != 0) {
    status = loom_local_value_domain_acquire_for_region_tree(
        module, body, &arena, &plan.domain);
  }
  if (iree_status_is_ok(status) && plan.candidate_count != 0) {
    status = loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(
            function,
            loom_target_function_version_target_facts(pass->function_version)),
        &plan.facts);
  }
  if (iree_status_is_ok(status) && plan.candidate_count != 0) {
    loom_scf_layout_describe_candidates(&plan);
  }
  if (iree_status_is_ok(status) && plan.candidate_count != 0) {
    status = loom_scf_layout_index_candidates(&plan);
  }
  if (iree_status_is_ok(status) && plan.candidate_count != 0) {
    status = loom_scf_layout_select_candidates(&plan);
  }
  bool changed = false;
  if (iree_status_is_ok(status) && plan.candidate_count != 0) {
    status = loom_scf_layout_apply(&plan, &changed);
  }
  if (changed) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    if (iree_status_is_ok(status)) {
      loom_pass_mark_changed(pass);
    }
  }

  loom_rewriter_deinitialize(&rewriter);
  loom_local_value_domain_release(&plan.domain);
  iree_arena_deinitialize(&arena);
  return status;
}
