// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/preamble.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/context.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/lower/abi.h"
#include "loom/target/arch/amdgpu/lower/cluster_preamble.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_repack.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race.h"
#include "loom/target/arch/amdgpu/lower/source_alloca_layout.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"
#include "loom/target/arch/amdgpu/lower/workgroup_storage.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#define LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_BITS 10u
#define LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_MASK 0x3FFu
#define LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_X_OFFSET 12u
#define LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_Y_OFFSET 16u
#define LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_Z_OFFSET 20u
#define LOOM_AMDGPU_DISPATCH_PACKET_GROUP_SEGMENT_SIZE_OFFSET 28u

typedef enum loom_amdgpu_preamble_query_kind_e {
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_NONE = 0,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKITEM_ID = 1,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_ID = 2,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_SIZE = 3,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_COUNT = 4,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKITEM_DISPATCH_ID = 5,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_SUBGROUP_SIZE = 6,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_SUBGROUP_COUNT = 7,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_SUBGROUP_ID = 8,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_SUBGROUP_LANE_ID = 9,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_ID = 10,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_WORKGROUP_ID = 11,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_WORKGROUP_FLAT_ID = 12,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_SIZE = 13,
  LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_COUNT = 14,
} loom_amdgpu_preamble_query_kind_t;

typedef uint32_t loom_amdgpu_preamble_query_flags_t;

enum {
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NONE = 0u,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL = 1u << 0,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_WORKITEM_ID = 1u << 1,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_WORKGROUP_ID = 1u << 2,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_BINDS_WORKITEM_ID = 1u << 3,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_BINDS_WORKGROUP_ID = 1u << 4,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_LINEAR_WORKITEM_ID = 1u << 5,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_ID = 1u << 6,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_WORKGROUP_ID = 1u << 7,
  LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_WORKGROUP_FLAT_ID = 1u << 8,
};

typedef struct loom_amdgpu_preamble_query_row_t {
  // Query family selected by the source op kind.
  loom_amdgpu_preamble_query_kind_t kind;
  // Query properties used by selection and preamble live-in discovery.
  loom_amdgpu_preamble_query_flags_t flags;
} loom_amdgpu_preamble_query_row_t;

#define LOOM_AMDGPU_KERNEL_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_AMDGPU_PREAMBLE_QUERY_ROW(op, kind_, flags_) \
  [LOOM_AMDGPU_KERNEL_OP_INDEX(LOOM_OP_KERNEL_##op)] = {  \
      .kind = LOOM_AMDGPU_PREAMBLE_QUERY_KIND_##kind_,    \
      .flags = flags_,                                    \
  }

static const loom_amdgpu_preamble_query_row_t
    kAmdgpuPreambleQueryRows[LOOM_OP_KERNEL_COUNT_] = {
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            WORKITEM_ID, WORKITEM_ID,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL |
                LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_WORKITEM_ID |
                LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_BINDS_WORKITEM_ID),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            WORKGROUP_ID, WORKGROUP_ID,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL |
                LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_WORKGROUP_ID |
                LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_BINDS_WORKGROUP_ID),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            WORKGROUP_SIZE, WORKGROUP_SIZE,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            WORKGROUP_COUNT, WORKGROUP_COUNT,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            WORKITEM_DISPATCH_ID, WORKITEM_DISPATCH_ID,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL |
                LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_WORKITEM_ID |
                LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_WORKGROUP_ID),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(SUBGROUP_SIZE, SUBGROUP_SIZE,
                                       LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NONE),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(SUBGROUP_COUNT, SUBGROUP_COUNT,
                                       LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NONE),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            SUBGROUP_ID, SUBGROUP_ID,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_LINEAR_WORKITEM_ID),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            SUBGROUP_LANE_ID, SUBGROUP_LANE_ID,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_LINEAR_WORKITEM_ID),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            CLUSTER_ID, CLUSTER_ID,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL |
                LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_ID),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            CLUSTER_WORKGROUP_ID, CLUSTER_WORKGROUP_ID,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL |
                LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_WORKGROUP_ID),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            CLUSTER_WORKGROUP_FLAT_ID, CLUSTER_WORKGROUP_FLAT_ID,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_WORKGROUP_FLAT_ID),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            CLUSTER_SIZE, CLUSTER_SIZE,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL),
        LOOM_AMDGPU_PREAMBLE_QUERY_ROW(
            CLUSTER_COUNT, CLUSTER_COUNT,
            LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL),
};

#undef LOOM_AMDGPU_PREAMBLE_QUERY_ROW
#undef LOOM_AMDGPU_KERNEL_OP_INDEX

static const loom_amdgpu_preamble_query_row_t* loom_amdgpu_preamble_query_row(
    loom_op_kind_t op_kind) {
  if (loom_op_dialect_id(op_kind) != LOOM_DIALECT_KERNEL) {
    return NULL;
  }
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuPreambleQueryRows)) {
    return NULL;
  }
  const loom_amdgpu_preamble_query_row_t* row =
      &kAmdgpuPreambleQueryRows[op_index];
  return row->kind == LOOM_AMDGPU_PREAMBLE_QUERY_KIND_NONE ? NULL : row;
}

static bool loom_amdgpu_preamble_query_row_has_flag(
    const loom_amdgpu_preamble_query_row_t* row,
    loom_amdgpu_preamble_query_flags_t flag) {
  return row != NULL && iree_any_bit_set(row->flags, flag);
}

static loom_value_id_t loom_amdgpu_preamble_query_result(
    const loom_op_t* op, const loom_amdgpu_preamble_query_row_t* row) {
  (void)row;
  IREE_ASSERT_EQ(op->result_count, 1u);
  return loom_op_const_results(op)[0];
}

static loom_kernel_dimension_t loom_amdgpu_preamble_query_dimension(
    const loom_op_t* op, const loom_amdgpu_preamble_query_row_t* row) {
  IREE_ASSERT(loom_amdgpu_preamble_query_row_has_flag(
      row, LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL));
  switch (row->kind) {
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKITEM_ID:
      return loom_kernel_workitem_id_dimension(op);
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_ID:
      return loom_kernel_workgroup_id_dimension(op);
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_SIZE:
      return loom_kernel_workgroup_size_dimension(op);
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_COUNT:
      return loom_kernel_workgroup_count_dimension(op);
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKITEM_DISPATCH_ID:
      return loom_kernel_workitem_dispatch_id_dimension(op);
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_ID:
      return loom_kernel_cluster_id_dimension(op);
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_WORKGROUP_ID:
      return loom_kernel_cluster_workgroup_id_dimension(op);
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_SIZE:
      return loom_kernel_cluster_size_dimension(op);
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_COUNT:
      return loom_kernel_cluster_count_dimension(op);
    default:
      IREE_ASSERT_UNREACHABLE("query has no dimension attribute");
      IREE_BUILTIN_UNREACHABLE();
  }
}

typedef struct loom_amdgpu_preamble_query_facts_t {
  // Module that owns source values and facts.
  const loom_module_t* module;
  // Source function being lowered or checked.
  loom_func_like_t function;
  // Selected target bundle for target and launch facts.
  const loom_target_bundle_t* bundle;
  // Typed AMDGPU target facts selected for this lowering contract.
  const loom_amdgpu_target_facts_t* target_facts;
  // Optional value-fact table used for specialization-aware proofs.
  const loom_value_fact_table_t* fact_table;
} loom_amdgpu_preamble_query_facts_t;

enum loom_amdgpu_preamble_value_flag_bits_e {
  LOOM_AMDGPU_PREAMBLE_VALUE_EXACT = 1u << 0,
  LOOM_AMDGPU_PREAMBLE_VALUE_REFERENCED = 1u << 1,
};

typedef struct loom_amdgpu_preamble_query_plan_t {
  // Exact-value and source-demand decisions retained by selection.
  uint8_t flags;
  // Constant result for exact queries, or workgroup scale for dispatch IDs.
  uint32_t value;
} loom_amdgpu_preamble_query_plan_t;

typedef struct loom_amdgpu_preamble_count_source_t {
  // First referenced dynamic query for one launch-count dimension.
  const loom_op_t* source_op;
  // The query's retained carrier and value decisions.
  const loom_amdgpu_preamble_query_plan_t* plan;
} loom_amdgpu_preamble_count_source_t;

iree_status_t loom_amdgpu_preamble_bind_query_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_value_id_t low_result) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_scalar_carrier(
      context, source_op, low_result,
      loom_low_lower_value_binding_type(context, source_result), &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static uint32_t loom_amdgpu_workgroup_size_dim(
    const loom_target_workgroup_size_t* size,
    loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return size->x;
    case LOOM_KERNEL_DIMENSION_Y:
      return size->y;
    case LOOM_KERNEL_DIMENSION_Z:
      return size->z;
    default:
      return 0;
  }
}

static bool loom_amdgpu_required_workgroup_size_dim(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, loom_kernel_dimension_t dimension,
    const loom_value_fact_table_t* fact_table, uint32_t* out_value) {
  *out_value = 0;
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  loom_target_workgroup_size_t size = {0};
  if (!loom_amdgpu_required_workgroup_size_from_facts(module, function, bundle,
                                                      fact_table, &size)) {
    return false;
  }
  *out_value = loom_amdgpu_workgroup_size_dim(&size, dimension);
  return *out_value != 0;
}

static uint32_t loom_amdgpu_ceil_div_u32(uint32_t numerator,
                                         uint32_t denominator) {
  IREE_ASSERT_NE(denominator, 0u);
  return numerator == 0 ? 0 : 1u + (numerator - 1u) / denominator;
}

static uint32_t loom_amdgpu_u32_log2(uint32_t value) {
  IREE_ASSERT(loom_amdgpu_u32_is_power_of_two(value));
  uint32_t log2 = 0;
  while (value > 1u) {
    value >>= 1u;
    ++log2;
  }
  return log2;
}

static bool loom_amdgpu_value_facts_exact_u32(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source_value,
    uint32_t* out_value) {
  *out_value = 0;
  if (fact_table == NULL) {
    return false;
  }
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, source_value), &value) ||
      value < 0 || value > UINT32_MAX) {
    return false;
  }
  *out_value = (uint32_t)value;
  return true;
}

static bool loom_amdgpu_preamble_query_launch_facts_satisfied(
    const loom_amdgpu_preamble_query_facts_t* facts, const loom_op_t* source_op,
    const loom_amdgpu_preamble_query_row_t* row,
    loom_amdgpu_preamble_query_plan_t* out_plan,
    iree_string_view_t* out_reason) {
  *out_plan = (loom_amdgpu_preamble_query_plan_t){0};
  *out_reason = IREE_SV("launch.query_supported");
  const loom_value_id_t source_result =
      loom_amdgpu_preamble_query_result(source_op, row);
  loom_kernel_dimension_t dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  if (loom_amdgpu_preamble_query_row_has_flag(
          row, LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_DIMENSIONAL)) {
    dimension = loom_amdgpu_preamble_query_dimension(source_op, row);
    if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
      *out_reason = IREE_SV("launch.query_dimension");
      return false;
    }
  }

  loom_target_workgroup_cluster_size_t cluster_size = {0};
  const bool has_nontrivial_cluster =
      loom_amdgpu_cluster_preamble_required_nontrivial_size(
          facts->module, facts->function.op, facts->fact_table, &cluster_size);
  const bool supports_cluster_launch_state =
      loom_amdgpu_cluster_preamble_target_supports_cluster_launch_state(
          facts->target_facts);
  const bool uses_cluster_launch_state =
      has_nontrivial_cluster && supports_cluster_launch_state;

  switch (row->kind) {
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKITEM_ID:
      return true;
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_ID:
      if (has_nontrivial_cluster && !supports_cluster_launch_state) {
        *out_reason = IREE_SV("launch.cluster_state_unsupported");
        return false;
      }
      return true;
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_SIZE: {
      out_plan->flags = LOOM_AMDGPU_PREAMBLE_VALUE_EXACT;
      const bool satisfied = loom_amdgpu_required_workgroup_size_dim(
          facts->module, facts->function, facts->bundle, dimension,
          facts->fact_table, &out_plan->value);
      if (!satisfied) {
        *out_reason = IREE_SV("launch.workgroup_size_fixed");
      }
      return satisfied;
    }
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_COUNT: {
      if (loom_amdgpu_value_facts_exact_u32(facts->fact_table, source_result,
                                            &out_plan->value)) {
        out_plan->flags = LOOM_AMDGPU_PREAMBLE_VALUE_EXACT;
        return true;
      }
      if (has_nontrivial_cluster) {
        if (!supports_cluster_launch_state) {
          *out_reason = IREE_SV("launch.cluster_state_unsupported");
        }
        return supports_cluster_launch_state;
      }
      uint32_t workgroup_size = 0;
      if (!loom_amdgpu_required_workgroup_size_dim(
              facts->module, facts->function, facts->bundle, dimension,
              facts->fact_table, &workgroup_size)) {
        *out_reason = IREE_SV("launch.workgroup_count_fixed_workgroup_size");
        return false;
      }
      return true;
    }
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKITEM_DISPATCH_ID: {
      if (has_nontrivial_cluster && !supports_cluster_launch_state) {
        *out_reason = IREE_SV("launch.cluster_state_unsupported");
        return false;
      }
      const bool satisfied = loom_amdgpu_required_workgroup_size_dim(
          facts->module, facts->function, facts->bundle, dimension,
          facts->fact_table, &out_plan->value);
      if (!satisfied) {
        *out_reason = IREE_SV("launch.workitem_dispatch_fixed_workgroup_size");
      }
      return satisfied;
    }
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_SUBGROUP_SIZE: {
      out_plan->flags = LOOM_AMDGPU_PREAMBLE_VALUE_EXACT;
      out_plan->value = loom_amdgpu_target_wavefront_size(facts->bundle);
      return true;
    }
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_SUBGROUP_COUNT: {
      const uint32_t wavefront_size =
          loom_amdgpu_target_wavefront_size(facts->bundle);
      uint32_t flat_workgroup_size = 0;
      const bool satisfied =
          loom_amdgpu_required_flat_workgroup_size_from_facts(
              facts->module, facts->function, facts->bundle, facts->fact_table,
              &flat_workgroup_size);
      if (!satisfied) {
        *out_reason = IREE_SV("launch.subgroup_count_fixed_workgroup_size");
      } else {
        out_plan->flags = LOOM_AMDGPU_PREAMBLE_VALUE_EXACT;
        out_plan->value =
            loom_amdgpu_ceil_div_u32(flat_workgroup_size, wavefront_size);
      }
      return satisfied;
    }
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_SUBGROUP_ID:
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_SUBGROUP_LANE_ID: {
      const uint32_t wavefront_size =
          loom_amdgpu_target_wavefront_size(facts->bundle);
      if (!loom_amdgpu_u32_is_power_of_two(wavefront_size)) {
        *out_reason = IREE_SV("launch.subgroup_size_power_of_two");
        return false;
      }
      uint32_t unused_flat_workgroup_size = 0;
      const bool satisfied =
          loom_amdgpu_required_flat_workgroup_size_from_facts(
              facts->module, facts->function, facts->bundle, facts->fact_table,
              &unused_flat_workgroup_size);
      if (!satisfied) {
        *out_reason = IREE_SV("launch.subgroup_index_fixed_workgroup_size");
      } else if (loom_amdgpu_value_facts_exact_u32(
                     facts->fact_table, source_result, &out_plan->value)) {
        out_plan->flags = LOOM_AMDGPU_PREAMBLE_VALUE_EXACT;
      }
      return satisfied;
    }
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_ID:
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_WORKGROUP_ID:
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_WORKGROUP_FLAT_ID:
      if (!has_nontrivial_cluster) {
        *out_reason = IREE_SV("launch.cluster_size_fixed_nontrivial");
        return false;
      }
      if (!supports_cluster_launch_state) {
        *out_reason = IREE_SV("launch.cluster_state_unsupported");
        return false;
      }
      return true;
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_SIZE:
      if (!uses_cluster_launch_state) {
        *out_reason = has_nontrivial_cluster
                          ? IREE_SV("launch.cluster_state_unsupported")
                          : IREE_SV("launch.cluster_size_fixed_nontrivial");
        return false;
      }
      out_plan->flags = LOOM_AMDGPU_PREAMBLE_VALUE_EXACT;
      out_plan->value =
          loom_amdgpu_cluster_preamble_size_dimension(&cluster_size, dimension);
      return out_plan->value != 0;
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_COUNT: {
      if (!uses_cluster_launch_state) {
        *out_reason = has_nontrivial_cluster
                          ? IREE_SV("launch.cluster_state_unsupported")
                          : IREE_SV("launch.cluster_size_fixed_nontrivial");
        return false;
      }
      if (loom_amdgpu_value_facts_exact_u32(facts->fact_table, source_result,
                                            &out_plan->value)) {
        out_plan->flags = LOOM_AMDGPU_PREAMBLE_VALUE_EXACT;
      }
      return true;
    }
    case LOOM_AMDGPU_PREAMBLE_QUERY_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU preamble query kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_select_preamble_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const loom_amdgpu_preamble_query_row_t* row =
      loom_amdgpu_preamble_query_row(source_op->kind);
  if (row == NULL) {
    return iree_ok_status();
  }
  const loom_amdgpu_preamble_query_facts_t facts = {
      .module = loom_low_lower_context_module(context),
      .function = loom_low_lower_context_source_function(context),
      .bundle = loom_low_lower_context_bundle(context),
      .target_facts = loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context)),
      .fact_table = loom_low_lower_context_fact_table(context),
  };
  iree_string_view_t unused_reason = iree_string_view_empty();
  loom_amdgpu_preamble_query_plan_t query = {0};
  bool selected = loom_amdgpu_preamble_query_launch_facts_satisfied(
      &facts, source_op, row, &query, &unused_reason);
  if (selected &&
      !loom_amdgpu_value_is_address_scalar(
          context, loom_amdgpu_preamble_query_result(source_op, row))) {
    selected = false;
  }
  if (selected) {
    const loom_value_id_t result =
        loom_amdgpu_preamble_query_result(source_op, row);
    if (!loom_value_has_no_uses(loom_module_value(facts.module, result)) ||
        loom_module_value_has_type_uses(facts.module, result)) {
      query.flags |= LOOM_AMDGPU_PREAMBLE_VALUE_REFERENCED;
    }
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_result_type(context, source_op, result, &result_type));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_plan_value_type(context, result, result_type));
    loom_amdgpu_preamble_query_plan_t* retained_query = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_query), (void**)&retained_query));
    *retained_query = query;
    *out_plan = loom_low_lower_plan_make(source_op->kind, retained_query);
  }
  return iree_ok_status();
}

typedef struct loom_amdgpu_dimension_source_kind_t {
  // ABI source kind for the workgroup-id coordinate.
  loom_amdgpu_hal_kernel_abi_source_kind_t workgroup;
  // ABI source kind for the unpacked workitem-id coordinate.
  loom_amdgpu_hal_kernel_abi_source_kind_t workitem;
} loom_amdgpu_dimension_source_kind_t;

#define LOOM_AMDGPU_DIMENSION_SOURCE_KIND(dimension, axis)                \
  [LOOM_KERNEL_DIMENSION_##dimension] = {                                 \
      .workgroup = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_##axis, \
      .workitem = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_##axis,   \
  }

static const loom_amdgpu_dimension_source_kind_t
    kLoomAmdgpuDimensionSourceKinds[LOOM_KERNEL_DIMENSION_COUNT_] = {
        LOOM_AMDGPU_DIMENSION_SOURCE_KIND(X, X),
        LOOM_AMDGPU_DIMENSION_SOURCE_KIND(Y, Y),
        LOOM_AMDGPU_DIMENSION_SOURCE_KIND(Z, Z),
};

#undef LOOM_AMDGPU_DIMENSION_SOURCE_KIND

static iree_string_view_t loom_amdgpu_workitem_id_source_name(
    loom_kernel_dimension_t dimension) {
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    IREE_ASSERT_UNREACHABLE("unknown kernel dimension");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_amdgpu_hal_kernel_abi_source_name(
      kLoomAmdgpuDimensionSourceKinds[dimension].workitem);
}

static iree_string_view_t loom_amdgpu_packed_workitem_id_source_name(
    uint32_t dimension_count) {
  static const loom_amdgpu_hal_kernel_abi_source_kind_t
      kPackedWorkitemIdSourceKinds[] = {
          [2] = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY,
          [3] = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ,
      };
  if (dimension_count >= IREE_ARRAYSIZE(kPackedWorkitemIdSourceKinds) ||
      kPackedWorkitemIdSourceKinds[dimension_count] ==
          LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN) {
    IREE_ASSERT_UNREACHABLE("packed workitem id requires two or three axes");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_amdgpu_hal_kernel_abi_source_name(
      kPackedWorkitemIdSourceKinds[dimension_count]);
}

static bool loom_amdgpu_uses_packed_workitem_id(
    loom_low_lower_context_t* context) {
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context));
  IREE_ASSERT(target_facts != NULL,
              "AMDGPU preamble requires AMDGPU target facts");
  return loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
      target_facts->properties.processor,
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID);
}

static iree_string_view_t loom_amdgpu_workgroup_id_source_name(
    loom_kernel_dimension_t dimension) {
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    IREE_ASSERT_UNREACHABLE("unknown kernel dimension");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_amdgpu_hal_kernel_abi_source_name(
      kLoomAmdgpuDimensionSourceKinds[dimension].workgroup);
}

static iree_status_t loom_amdgpu_emit_workitem_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  iree_string_view_t source = loom_amdgpu_workitem_id_source_name(dimension);
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, source, &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), 0,
                             source_id, loom_make_named_attr_slice(NULL, 0),
                             vgpr_type, source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_packed_workitem_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t dimension_count, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  iree_string_view_t source =
      loom_amdgpu_packed_workitem_id_source_name(dimension_count);
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, source, &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), 0,
                             source_id, loom_make_named_attr_slice(NULL, 0),
                             vgpr_type, source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("packed_tid"), &value_name_id));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_name(loom_low_lower_context_module(context),
                                 *out_low_value_id, value_name_id));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_packed_workitem_id_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t packed_id, loom_kernel_dimension_t dimension,
    bool requires_masked_x, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t shifted_id = packed_id;
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      if (!requires_masked_x) {
        *out_low_value_id = packed_id;
        return iree_ok_status();
      }
      break;
    case LOOM_KERNEL_DIMENSION_Y: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_BITS, packed_id, vgpr_type,
          &shifted_id));
      break;
    }
    case LOOM_KERNEL_DIMENSION_Z: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_BITS * 2u, packed_id,
          vgpr_type, &shifted_id));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU workitem-id dimension");
      IREE_BUILTIN_UNREACHABLE();
  }

  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, shifted_id,
      LOOM_AMDGPU_PACKED_WORKITEM_ID_DIMENSION_MASK, vgpr_type,
      out_low_value_id);
}

static iree_string_view_t loom_amdgpu_module_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, string_id);
}

static loom_value_id_t loom_amdgpu_lookup_live_in_by_source(
    loom_low_lower_context_t* context, iree_string_view_t expected_source) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  loom_op_t* low_function = loom_low_lower_context_low_function(context);
  loom_region_t* body =
      low_function ? loom_low_function_body(low_function) : NULL;
  IREE_ASSERT(body != NULL);
  IREE_ASSERT_GT(body->block_count, 0);

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_live_in_isa(op)) {
      break;
    }
    const iree_string_view_t source =
        loom_amdgpu_module_string_or_empty(module, loom_low_live_in_source(op));
    if (!iree_string_view_equal(source, expected_source)) {
      continue;
    }
    if (value_id != LOOM_VALUE_ID_INVALID) {
      IREE_ASSERT_UNREACHABLE("unique AMDGPU preamble live-in source");
      IREE_BUILTIN_UNREACHABLE();
    }
    value_id = loom_low_live_in_result(op);
  }
  return value_id;
}

static loom_value_id_t loom_amdgpu_lookup_packed_workitem_id_live_in(
    loom_low_lower_context_t* context, uint32_t* out_dimension_count) {
  *out_dimension_count = 0;

  const loom_value_id_t xy_value_id = loom_amdgpu_lookup_live_in_by_source(
      context, loom_amdgpu_hal_kernel_abi_source_name(
                   LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY));
  const loom_value_id_t xyz_value_id = loom_amdgpu_lookup_live_in_by_source(
      context, loom_amdgpu_hal_kernel_abi_source_name(
                   LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ));

  if (xy_value_id != LOOM_VALUE_ID_INVALID &&
      xyz_value_id != LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE("unique AMDGPU packed workitem-id live-in");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (xy_value_id != LOOM_VALUE_ID_INVALID) {
    *out_dimension_count = 2;
    return xy_value_id;
  } else if (xyz_value_id != LOOM_VALUE_ID_INVALID) {
    *out_dimension_count = 3;
    return xyz_value_id;
  }
  return LOOM_VALUE_ID_INVALID;
}

static loom_value_id_t loom_amdgpu_lookup_workitem_id_live_in(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension) {
  const loom_value_id_t value_id = loom_amdgpu_lookup_live_in_by_source(
      context, loom_amdgpu_workitem_id_source_name(dimension));
  if (value_id == LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU workitem-id live-in");
    IREE_BUILTIN_UNREACHABLE();
  }
  return value_id;
}

static iree_status_t loom_amdgpu_lookup_workitem_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t packed_dimension_count, loom_value_id_t packed_workitem_id,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  if (packed_workitem_id != LOOM_VALUE_ID_INVALID) {
    if ((uint32_t)dimension >= packed_dimension_count) {
      IREE_ASSERT_UNREACHABLE("selected AMDGPU packed workitem-id dimension");
      IREE_BUILTIN_UNREACHABLE();
    }
    return loom_amdgpu_emit_packed_workitem_id_extract(
        context, source_op, packed_workitem_id, dimension,
        packed_dimension_count > 1, out_low_value_id);
  }
  *out_low_value_id =
      loom_amdgpu_lookup_workitem_id_live_in(context, dimension);
  return iree_ok_status();
}

static loom_value_id_t loom_amdgpu_lookup_workgroup_id_live_in(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension) {
  const loom_value_id_t value_id = loom_amdgpu_lookup_live_in_by_source(
      context, loom_amdgpu_workgroup_id_source_name(dimension));
  if (value_id == LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU workgroup-id live-in");
    IREE_BUILTIN_UNREACHABLE();
  }
  return value_id;
}

static loom_value_id_t loom_amdgpu_lookup_dispatch_ptr_live_in(
    loom_low_lower_context_t* context) {
  const loom_value_id_t value_id = loom_amdgpu_lookup_live_in_by_source(
      context, loom_amdgpu_hal_kernel_abi_source_name(
                   LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_PTR));
  if (value_id == LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU dispatch pointer live-in");
    IREE_BUILTIN_UNREACHABLE();
  }
  return value_id;
}

static loom_value_id_t loom_amdgpu_lookup_dispatch_id_live_in(
    loom_low_lower_context_t* context) {
  const loom_value_id_t value_id = loom_amdgpu_lookup_live_in_by_source(
      context, loom_amdgpu_hal_kernel_abi_source_name(
                   LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_ID));
  if (value_id == LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU dispatch ID live-in");
    IREE_BUILTIN_UNREACHABLE();
  }
  return value_id;
}

iree_status_t loom_amdgpu_lookup_current_dispatch_ptr(
    loom_low_lower_context_t* context, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = loom_amdgpu_lookup_dispatch_ptr_live_in(context);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lookup_current_dispatch_id(
    loom_low_lower_context_t* context, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = loom_amdgpu_lookup_dispatch_id_live_in(context);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lookup_current_workitem_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  uint32_t packed_dimension_count = 0;
  const loom_value_id_t packed_workitem_id =
      loom_amdgpu_lookup_packed_workitem_id_live_in(context,
                                                    &packed_dimension_count);
  return loom_amdgpu_lookup_workitem_id(
      context, source_op, packed_dimension_count, packed_workitem_id, dimension,
      out_low_value_id);
}

static void loom_amdgpu_mark_lane_query_workitem_id_live_ins(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t** first_workitem_id_ops) {
  const loom_target_workgroup_size_t* workgroup_size =
      loom_low_lower_context_workgroup_size(context);

  if (first_workitem_id_ops[LOOM_KERNEL_DIMENSION_X] == NULL) {
    first_workitem_id_ops[LOOM_KERNEL_DIMENSION_X] = source_op;
  }
  if (workgroup_size->y > 1 &&
      first_workitem_id_ops[LOOM_KERNEL_DIMENSION_Y] == NULL) {
    first_workitem_id_ops[LOOM_KERNEL_DIMENSION_Y] = source_op;
  }
  if (workgroup_size->z > 1 &&
      first_workitem_id_ops[LOOM_KERNEL_DIMENSION_Z] == NULL) {
    first_workitem_id_ops[LOOM_KERNEL_DIMENSION_Z] = source_op;
  }
}

static iree_status_t loom_amdgpu_emit_dispatch_ptr_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t sgprx2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgprx2_type));
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context,
                         loom_amdgpu_hal_kernel_abi_source_name(
                             LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_PTR),
                         &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), 0,
                             source_id, loom_make_named_attr_slice(NULL, 0),
                             sgprx2_type, source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_dispatch_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t sgprx2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgprx2_type));
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context,
                         loom_amdgpu_hal_kernel_abi_source_name(
                             LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_ID),
                         &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), 0,
                             source_id, loom_make_named_attr_slice(NULL, 0),
                             sgprx2_type, source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_sgpr_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  const iree_string_view_t source =
      loom_amdgpu_hal_kernel_abi_source_name(source_kind);
  if (iree_string_view_is_empty(source)) {
    IREE_ASSERT_UNREACHABLE("known AMDGPU SGPR live-in source");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, source, &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), 0,
                             source_id, loom_make_named_attr_slice(NULL, 0),
                             sgpr_type, source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  return loom_amdgpu_emit_sgpr_live_in(
      context, source_op, kLoomAmdgpuDimensionSourceKinds[dimension].workgroup,
      out_low_value_id);
}

iree_status_t loom_amdgpu_lookup_current_workgroup_id(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  bool uses_architected_workgroup_ids = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_cluster_preamble_uses_architected_workgroup_ids(
          context, &uses_architected_workgroup_ids));
  if (uses_architected_workgroup_ids) {
    return loom_amdgpu_cluster_preamble_lookup_workgroup_id(context, dimension,
                                                            out_low_value_id);
  }
  *out_low_value_id =
      loom_amdgpu_lookup_workgroup_id_live_in(context, dimension);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vgpr_scaled_add(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t accumulator, loom_value_id_t value, uint32_t scale,
    loom_type_t result_type, loom_value_id_t* out_sum) {
  *out_sum = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
      context, source_op, value, scale, LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
      result_type, &scaled_value));
  return loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, accumulator,
      scaled_value, result_type, out_sum);
}

static iree_status_t loom_amdgpu_emit_workitem_dispatch_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_preamble_query_plan_t* plan,
    loom_value_id_t low_workgroup_id, loom_value_id_t low_workitem_id) {
  const loom_value_id_t source_result =
      loom_kernel_workitem_dispatch_id_result(source_op);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &result_type));
  const uint32_t workgroup_size = plan->value;
  loom_value_id_t low_scaled_workgroup_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
      context, source_op, low_workgroup_id, &low_scaled_workgroup_id));
  if (workgroup_size != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, low_scaled_workgroup_id, workgroup_size,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, result_type,
        &low_scaled_workgroup_id));
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      low_scaled_workgroup_id, low_workitem_id, result_type, &low_result));
  return loom_amdgpu_preamble_bind_query_result(context, source_op,
                                                source_result, low_result);
}

static void loom_amdgpu_find_first_dynamic_count_ops(
    loom_low_lower_context_t* context,
    loom_amdgpu_preamble_query_kind_t query_kind,
    loom_amdgpu_preamble_count_source_t* out_first_counts) {
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    out_first_counts[i] = (loom_amdgpu_preamble_count_source_t){0};
  }
  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    const loom_amdgpu_preamble_query_row_t* row =
        loom_amdgpu_preamble_query_row(selected_plan.plan.id);
    if (row == NULL || row->kind != query_kind) {
      continue;
    }
    const loom_amdgpu_preamble_query_plan_t* query =
        selected_plan.plan.target_data;
    if (!iree_any_bit_set(query->flags,
                          LOOM_AMDGPU_PREAMBLE_VALUE_REFERENCED) ||
        iree_any_bit_set(query->flags, LOOM_AMDGPU_PREAMBLE_VALUE_EXACT)) {
      continue;
    }
    const loom_kernel_dimension_t dimension =
        loom_amdgpu_preamble_query_dimension(selected_plan.source_op, row);
    if (out_first_counts[dimension].source_op == NULL) {
      out_first_counts[dimension] = (loom_amdgpu_preamble_count_source_t){
          .source_op = selected_plan.source_op, .plan = query};
    }
  }
}

iree_status_t loom_amdgpu_emit_preamble(void* user_data,
                                        loom_low_lower_context_t* context) {
  (void)user_data;
  const bool needs_sanitizer_report_coordinates =
      loom_low_lower_context_sanitizer_reporting_mode(context) !=
      LOOM_SANITIZER_REPORTING_MODE_TRAP;
  const loom_op_t* first_workitem_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t* first_workgroup_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t* first_cluster_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t*
      first_cluster_workgroup_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t* first_cluster_workgroup_flat_id_op = NULL;
  const loom_op_t* first_dispatch_ptr_op = NULL;
  const loom_op_t* first_dispatch_id_op = NULL;
  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    const loom_op_t* source_op = selected_plan.source_op;
    const loom_low_lower_plan_t plan = selected_plan.plan;
    const loom_amdgpu_preamble_query_row_t* row =
        loom_amdgpu_preamble_query_row(plan.id);
    if (row != NULL) {
      if (loom_amdgpu_preamble_query_row_has_flag(
              row, LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_LINEAR_WORKITEM_ID)) {
        const loom_amdgpu_preamble_query_plan_t* query = plan.target_data;
        if (!iree_any_bit_set(query->flags, LOOM_AMDGPU_PREAMBLE_VALUE_EXACT)) {
          loom_amdgpu_mark_lane_query_workitem_id_live_ins(
              context, source_op, first_workitem_id_ops);
        }
        continue;
      }
      if (loom_amdgpu_preamble_query_row_has_flag(
              row, LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_WORKITEM_ID)) {
        const loom_kernel_dimension_t dimension =
            loom_amdgpu_preamble_query_dimension(source_op, row);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_workitem_id_ops[dimension] == NULL) {
          first_workitem_id_ops[dimension] = source_op;
        }
      }
      if (loom_amdgpu_preamble_query_row_has_flag(
              row, LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_WORKGROUP_ID)) {
        const loom_kernel_dimension_t dimension =
            loom_amdgpu_preamble_query_dimension(source_op, row);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_workgroup_id_ops[dimension] == NULL) {
          first_workgroup_id_ops[dimension] = source_op;
        }
      }
      if (loom_amdgpu_preamble_query_row_has_flag(
              row, LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_ID)) {
        const loom_kernel_dimension_t dimension =
            loom_amdgpu_preamble_query_dimension(source_op, row);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_cluster_id_ops[dimension] == NULL) {
          first_cluster_id_ops[dimension] = source_op;
        }
      }
      if (loom_amdgpu_preamble_query_row_has_flag(
              row,
              LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_WORKGROUP_ID)) {
        const loom_kernel_dimension_t dimension =
            loom_amdgpu_preamble_query_dimension(source_op, row);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_cluster_workgroup_id_ops[dimension] == NULL) {
          first_cluster_workgroup_id_ops[dimension] = source_op;
        }
      }
      if (loom_amdgpu_preamble_query_row_has_flag(
              row,
              LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_NEEDS_CLUSTER_WORKGROUP_FLAT_ID) &&
          first_cluster_workgroup_flat_id_op == NULL) {
        first_cluster_workgroup_flat_id_op = source_op;
      }
      continue;
    }
    switch (plan.id) {
      case LOOM_OP_VECTOR_FRAGMENT_LOAD:
      case LOOM_OP_VECTOR_FRAGMENT_STORE: {
        loom_amdgpu_mark_lane_query_workitem_id_live_ins(context, source_op,
                                                         first_workitem_id_ops);
        break;
      }
      case LOOM_OP_VECTOR_FRAGMENT_REPACK: {
        const loom_amdgpu_fragment_repack_plan_t* repack_plan =
            (const loom_amdgpu_fragment_repack_plan_t*)plan.target_data;
        if (loom_amdgpu_fragment_repack_plan_requires_lane_id(repack_plan)) {
          loom_amdgpu_mark_lane_query_workitem_id_live_ins(
              context, source_op, first_workitem_id_ops);
        }
        break;
      }
      case LOOM_OP_KERNEL_SUBGROUP_SHUFFLE:
      case LOOM_OP_KERNEL_SUBGROUP_REDUCE:
      case LOOM_OP_KERNEL_SUBGROUP_SCAN:
      case LOOM_OP_KERNEL_WORKGROUP_REDUCE: {
        loom_amdgpu_mark_lane_query_workitem_id_live_ins(context, source_op,
                                                         first_workitem_id_ops);
        break;
      }
      case LOOM_OP_KERNEL_ASSERT:
      case LOOM_OP_SANITIZER_ASSERT_ACCESS:
      case LOOM_OP_SANITIZER_ASSERT_ACCESSES: {
        if (!needs_sanitizer_report_coordinates) {
          break;
        }
        if (first_dispatch_ptr_op == NULL) {
          first_dispatch_ptr_op = source_op;
        }
        if (first_workitem_id_ops[LOOM_KERNEL_DIMENSION_X] == NULL) {
          first_workitem_id_ops[LOOM_KERNEL_DIMENSION_X] = source_op;
        }
        if (first_workgroup_id_ops[LOOM_KERNEL_DIMENSION_X] == NULL) {
          first_workgroup_id_ops[LOOM_KERNEL_DIMENSION_X] = source_op;
        }
        break;
      }
      case LOOM_OP_SANITIZER_RACE_ACCESS:
      case LOOM_OP_SANITIZER_RACE_FRAGMENT_ACCESS: {
        if (plan.id == LOOM_OP_SANITIZER_RACE_FRAGMENT_ACCESS) {
          loom_amdgpu_mark_lane_query_workitem_id_live_ins(
              context, source_op, first_workitem_id_ops);
        }
        if (first_dispatch_ptr_op == NULL) {
          first_dispatch_ptr_op = source_op;
        }
        if (first_dispatch_id_op == NULL) {
          first_dispatch_id_op = source_op;
        }
        for (uint32_t j = 0; j < LOOM_KERNEL_DIMENSION_COUNT_; ++j) {
          if (first_workitem_id_ops[j] == NULL) {
            first_workitem_id_ops[j] = source_op;
          }
          if (first_workgroup_id_ops[j] == NULL) {
            first_workgroup_id_ops[j] = source_op;
          }
        }
        break;
      }
      case LOOM_OP_SANITIZER_RACE_SYNC: {
        if (first_dispatch_ptr_op == NULL) {
          first_dispatch_ptr_op = source_op;
        }
        if (first_dispatch_id_op == NULL) {
          first_dispatch_id_op = source_op;
        }
        for (uint32_t j = 0; j < LOOM_KERNEL_DIMENSION_COUNT_; ++j) {
          if (first_workitem_id_ops[j] == NULL) {
            first_workitem_id_ops[j] = source_op;
          }
          if (first_workgroup_id_ops[j] == NULL) {
            first_workgroup_id_ops[j] = source_op;
          }
        }
        break;
      }
      default:
        break;
    }
  }

  loom_value_id_t low_workitem_ids[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t low_workgroup_ids[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t workitem_id_dimension_count = 0;
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    if (first_workitem_id_ops[i] != NULL) {
      workitem_id_dimension_count = i + 1;
    }
  }
  bool uses_packed_workitem_id = false;
  if (workitem_id_dimension_count > 1) {
    uses_packed_workitem_id = loom_amdgpu_uses_packed_workitem_id(context);
  }

  loom_amdgpu_cluster_preamble_demands_t cluster_demands = {0};
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    cluster_demands.workgroup_id_ops[i] = first_workgroup_id_ops[i];
    cluster_demands.cluster_id_ops[i] = first_cluster_id_ops[i];
    cluster_demands.cluster_workgroup_id_ops[i] =
        first_cluster_workgroup_id_ops[i];
  }
  cluster_demands.cluster_workgroup_flat_id_op =
      first_cluster_workgroup_flat_id_op;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_cluster_preamble_emit_live_ins(context, &cluster_demands));

  bool uses_architected_workgroup_ids = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_cluster_preamble_uses_architected_workgroup_ids(
          context, &uses_architected_workgroup_ids));
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    if (!uses_architected_workgroup_ids && first_workgroup_id_ops[i] != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_id_live_in(
          context, first_workgroup_id_ops[i], (loom_kernel_dimension_t)i,
          &low_workgroup_ids[i]));
    }
  }

  loom_amdgpu_preamble_count_source_t
      first_workgroup_counts[LOOM_KERNEL_DIMENSION_COUNT_];
  loom_amdgpu_find_first_dynamic_count_ops(
      context, LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_COUNT,
      first_workgroup_counts);
  loom_amdgpu_preamble_count_source_t
      first_cluster_counts[LOOM_KERNEL_DIMENSION_COUNT_];
  loom_amdgpu_find_first_dynamic_count_ops(
      context, LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_COUNT,
      first_cluster_counts);
  bool emitted_dispatch_ptr = false;
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    const loom_op_t* first_count_op =
        first_workgroup_counts[i].source_op != NULL
            ? first_workgroup_counts[i].source_op
            : first_cluster_counts[i].source_op;
    if (first_count_op != NULL) {
      loom_value_id_t unused_low_dispatch_ptr = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_dispatch_ptr_live_in(
          context, first_count_op, &unused_low_dispatch_ptr));
      emitted_dispatch_ptr = true;
      break;
    }
  }
  if (!emitted_dispatch_ptr && first_dispatch_ptr_op != NULL) {
    loom_value_id_t unused_low_dispatch_ptr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_dispatch_ptr_live_in(
        context, first_dispatch_ptr_op, &unused_low_dispatch_ptr));
  }
  if (first_dispatch_id_op != NULL) {
    loom_value_id_t unused_low_dispatch_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_dispatch_id_live_in(
        context, first_dispatch_id_op, &unused_low_dispatch_id));
  }

  loom_value_id_t packed_workitem_id = LOOM_VALUE_ID_INVALID;
  if (uses_packed_workitem_id) {
    const loom_op_t* source_op = NULL;
    for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
      if (first_workitem_id_ops[i] != NULL) {
        source_op = first_workitem_id_ops[i];
        break;
      }
    }
    IREE_ASSERT(source_op != NULL);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_workitem_id_live_in(
        context, source_op, workitem_id_dimension_count, &packed_workitem_id));
  } else {
    for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
      if (first_workitem_id_ops[i] == NULL) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workitem_id_live_in(
          context, first_workitem_id_ops[i], (loom_kernel_dimension_t)i,
          &low_workitem_ids[i]));
    }
  }

  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    const loom_op_t* source_op = selected_plan.source_op;
    const loom_low_lower_plan_t plan = selected_plan.plan;
    const loom_amdgpu_preamble_query_row_t* row =
        loom_amdgpu_preamble_query_row(plan.id);
    if (row == NULL) {
      continue;
    }
    if (loom_amdgpu_preamble_query_row_has_flag(
            row, LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_BINDS_WORKITEM_ID)) {
      if (uses_packed_workitem_id) {
        continue;
      }
      const loom_kernel_dimension_t dimension =
          loom_amdgpu_preamble_query_dimension(source_op, row);
      IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
      IREE_ASSERT_NE(low_workitem_ids[dimension], LOOM_VALUE_ID_INVALID);
      IREE_RETURN_IF_ERROR(loom_amdgpu_preamble_bind_query_result(
          context, source_op, loom_amdgpu_preamble_query_result(source_op, row),
          low_workitem_ids[dimension]));
    }
    if (loom_amdgpu_preamble_query_row_has_flag(
            row, LOOM_AMDGPU_PREAMBLE_QUERY_FLAG_BINDS_WORKGROUP_ID)) {
      if (uses_architected_workgroup_ids) {
        continue;
      }
      const loom_kernel_dimension_t dimension =
          loom_amdgpu_preamble_query_dimension(source_op, row);
      IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
      IREE_ASSERT_NE(low_workgroup_ids[dimension], LOOM_VALUE_ID_INVALID);
      IREE_RETURN_IF_ERROR(loom_amdgpu_preamble_bind_query_result(
          context, source_op, loom_amdgpu_preamble_query_result(source_op, row),
          low_workgroup_ids[dimension]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_query_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t value) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      result_type, &low_result));
  return loom_amdgpu_preamble_bind_query_result(context, source_op,
                                                source_result, low_result);
}

static iree_status_t loom_amdgpu_emit_local_linear_workitem_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t packed_dimension_count, loom_value_id_t packed_workitem_id,
    const loom_target_workgroup_size_t* workgroup_size, loom_type_t result_type,
    loom_value_id_t* out_linear_id) {
  *out_linear_id = LOOM_VALUE_ID_INVALID;

  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workitem_id(
      context, source_op, packed_dimension_count, packed_workitem_id,
      LOOM_KERNEL_DIMENSION_X, &linear_id));

  if (workgroup_size->y > 1) {
    loom_value_id_t workitem_y = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workitem_id(
        context, source_op, packed_dimension_count, packed_workitem_id,
        LOOM_KERNEL_DIMENSION_Y, &workitem_y));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scaled_add(
        context, source_op, linear_id, workitem_y, workgroup_size->x,
        result_type, &linear_id));
  }

  if (workgroup_size->z > 1) {
    const uint64_t z_scale = (uint64_t)workgroup_size->x * workgroup_size->y;
    IREE_ASSERT_LE(z_scale, UINT32_MAX);
    loom_value_id_t workitem_z = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workitem_id(
        context, source_op, packed_dimension_count, packed_workitem_id,
        LOOM_KERNEL_DIMENSION_Z, &workitem_z));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scaled_add(
        context, source_op, linear_id, workitem_z, (uint32_t)z_scale,
        result_type, &linear_id));
  }

  *out_linear_id = linear_id;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_query_linear_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, uint32_t* out_wavefront_size,
    uint32_t* out_flat_workgroup_size, loom_value_id_t* out_linear_id) {
  *out_wavefront_size = 0;
  *out_flat_workgroup_size = 0;
  *out_linear_id = LOOM_VALUE_ID_INVALID;
  const loom_target_workgroup_size_t* workgroup_size =
      loom_low_lower_context_workgroup_size(context);
  const uint32_t flat_workgroup_size =
      workgroup_size->x * workgroup_size->y * workgroup_size->z;

  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  if (!loom_amdgpu_u32_is_power_of_two(wavefront_size)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU power-of-two subgroup size");
    IREE_BUILTIN_UNREACHABLE();
  }

  const bool result_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (!result_is_vgpr) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU subgroup query VGPR result");
    IREE_BUILTIN_UNREACHABLE();
  }

  uint32_t packed_dimension_count = 0;
  const loom_value_id_t packed_workitem_id =
      loom_amdgpu_lookup_packed_workitem_id_live_in(context,
                                                    &packed_dimension_count);
  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_local_linear_workitem_id(
      context, source_op, packed_dimension_count, packed_workitem_id,
      workgroup_size, result_type, &linear_id));
  *out_wavefront_size = wavefront_size;
  *out_flat_workgroup_size = flat_workgroup_size;
  *out_linear_id = linear_id;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_current_subgroup_lane_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_lane_id) {
  *out_lane_id = LOOM_VALUE_ID_INVALID;

  uint32_t wavefront_size = 0;
  uint32_t flat_workgroup_size = 0;
  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_query_linear_id(
      context, source_op, result_type, &wavefront_size, &flat_workgroup_size,
      &linear_id));
  if (flat_workgroup_size <= wavefront_size) {
    *out_lane_id = linear_id;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, linear_id,
        wavefront_size - 1, result_type, out_lane_id));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_current_workitem_linear_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_linear_id) {
  uint32_t unused_wavefront_size = 0;
  uint32_t unused_flat_workgroup_size = 0;
  return loom_amdgpu_emit_subgroup_query_linear_id(
      context, source_op, result_type, &unused_wavefront_size,
      &unused_flat_workgroup_size, out_linear_id);
}

static iree_status_t loom_amdgpu_emit_subgroup_linear_query(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_preamble_query_plan_t* plan,
    loom_value_id_t source_result, bool is_lane_id) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &result_type));
  if (iree_any_bit_set(plan->flags, LOOM_AMDGPU_PREAMBLE_VALUE_EXACT)) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, plan->value,
        result_type, &low_result));
    return loom_amdgpu_preamble_bind_query_result(context, source_op,
                                                  source_result, low_result);
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  if (is_lane_id) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
        context, source_op, result_type, &low_result));
  } else {
    uint32_t wavefront_size = 0;
    uint32_t unused_flat_workgroup_size = 0;
    loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_query_linear_id(
        context, source_op, result_type, &wavefront_size,
        &unused_flat_workgroup_size, &linear_id));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        loom_amdgpu_u32_log2(wavefront_size), linear_id, result_type,
        &low_result));
  }
  return loom_amdgpu_preamble_bind_query_result(context, source_op,
                                                source_result, low_result);
}

static uint32_t loom_amdgpu_dispatch_packet_grid_size_offset(
    loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_X_OFFSET;
    case LOOM_KERNEL_DIMENSION_Y:
      return LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_Y_OFFSET;
    case LOOM_KERNEL_DIMENSION_Z:
      return LOOM_AMDGPU_DISPATCH_PACKET_GRID_SIZE_Z_OFFSET;
    default:
      IREE_ASSERT_UNREACHABLE("unknown kernel dimension");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_emit_dispatch_packet_dword(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, uint32_t offset, loom_type_t result_type,
    loom_value_id_t* out_grid_size) {
  *out_grid_size = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), offset, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  loom_value_id_t operands[] = {
      dispatch_ptr,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
      operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, &low_op));
  *out_grid_size = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_dispatch_packet_grid_size(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, loom_kernel_dimension_t dimension,
    loom_type_t result_type, loom_value_id_t* out_grid_size) {
  return loom_amdgpu_emit_dispatch_packet_dword(
      context, source_op, dispatch_ptr,
      loom_amdgpu_dispatch_packet_grid_size_offset(dimension), result_type,
      out_grid_size);
}

static iree_status_t loom_amdgpu_emit_workgroup_count(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, loom_kernel_dimension_t dimension,
    loom_type_t result_type, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  bool uses_clustered_dispatch = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_uses_clustered_dispatch(
      context, &uses_clustered_dispatch));
  if (uses_clustered_dispatch) {
    return loom_amdgpu_cluster_preamble_emit_workgroup_count(
        context, source_op, dispatch_ptr, dimension, result_type,
        out_low_value_id);
  }
  const uint32_t workgroup_size = loom_amdgpu_workgroup_size_dim(
      loom_low_lower_context_workgroup_size(context), dimension);

  loom_value_id_t grid_size = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_dispatch_packet_grid_size(
      context, source_op, dispatch_ptr, dimension, result_type, &grid_size));
  return loom_amdgpu_emit_sgpr_divide_u32_constant(
      context, source_op, grid_size, workgroup_size, result_type,
      out_low_value_id);
}

iree_status_t loom_amdgpu_emit_current_workgroup_linear_id_u64_checked(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_linear_id, loom_value_id_t* out_overflow) {
  *out_linear_id = LOOM_VALUE_ID_INVALID;
  *out_overflow = LOOM_VALUE_ID_INVALID;

  loom_value_id_t workgroup_x = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_current_workgroup_id(
      context, LOOM_KERNEL_DIMENSION_X, &workgroup_x));
  loom_value_id_t workgroup_y = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_current_workgroup_id(
      context, LOOM_KERNEL_DIMENSION_Y, &workgroup_y));
  loom_value_id_t workgroup_z = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_current_workgroup_id(
      context, LOOM_KERNEL_DIMENSION_Z, &workgroup_z));

  const loom_value_id_t dispatch_ptr =
      loom_amdgpu_lookup_dispatch_ptr_live_in(context);
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t workgroup_count_x = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_count(
      context, source_op, dispatch_ptr, LOOM_KERNEL_DIMENSION_X, sgpr_type,
      &workgroup_count_x));
  loom_value_id_t workgroup_count_y = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_count(
      context, source_op, dispatch_ptr, LOOM_KERNEL_DIMENSION_Y, sgpr_type,
      &workgroup_count_y));

  loom_value_id_t workgroup_z_count_y = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_i64_mul_lo(context, source_op, workgroup_z,
                                  workgroup_count_y, &workgroup_z_count_y));
  loom_value_id_t workgroup_y_u64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_from_u32(
      context, source_op, workgroup_y, &workgroup_y_u64));
  loom_value_id_t workgroup_zy = LOOM_VALUE_ID_INVALID;
  loom_value_id_t row_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add_checked(
      context, source_op, workgroup_z_count_y, workgroup_y_u64, &workgroup_zy,
      &row_overflow));

  loom_value_id_t workgroup_count_x_u64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_from_u32(
      context, source_op, workgroup_count_x, &workgroup_count_x_u64));
  loom_value_id_t scaled_zy = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scale_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_mul_checked(
      context, source_op, workgroup_zy, workgroup_count_x_u64, &scaled_zy,
      &scale_overflow));
  loom_value_id_t workgroup_x_u64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_from_u32(
      context, source_op, workgroup_x, &workgroup_x_u64));
  loom_value_id_t add_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add_checked(
      context, source_op, scaled_zy, workgroup_x_u64, out_linear_id,
      &add_overflow));
  loom_value_id_t overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32, row_overflow,
      scale_overflow, sgpr_type, &overflow));
  return loom_amdgpu_emit_binary(context, source_op,
                                 LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32, overflow,
                                 add_overflow, sgpr_type, out_overflow);
}

iree_status_t loom_amdgpu_emit_current_group_segment_size(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_group_segment_size) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_dispatch_packet_dword(
      context, source_op, loom_amdgpu_lookup_dispatch_ptr_live_in(context),
      LOOM_AMDGPU_DISPATCH_PACKET_GROUP_SEGMENT_SIZE_OFFSET, sgpr_type,
      out_group_segment_size);
}

iree_status_t loom_amdgpu_plan_entry_setup(void* user_data,
                                           loom_low_lower_context_t* context) {
  (void)user_data;
  return loom_amdgpu_validate_workgroup_storage(context);
}

iree_status_t loom_amdgpu_emit_entry_setup(void* user_data,
                                           loom_low_lower_context_t* context) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_emit_entry_setup(context));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_alloca_layout_emit_low_storage_roots(context));

  loom_amdgpu_preamble_count_source_t
      first_workgroup_counts[LOOM_KERNEL_DIMENSION_COUNT_];
  loom_amdgpu_find_first_dynamic_count_ops(
      context, LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_COUNT,
      first_workgroup_counts);
  loom_amdgpu_preamble_count_source_t
      first_cluster_counts[LOOM_KERNEL_DIMENSION_COUNT_];
  loom_amdgpu_find_first_dynamic_count_ops(
      context, LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_COUNT,
      first_cluster_counts);

  bool uses_dynamic_count = false;
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    uses_dynamic_count = uses_dynamic_count ||
                         first_workgroup_counts[i].source_op != NULL ||
                         first_cluster_counts[i].source_op != NULL;
  }

  loom_value_id_t low_workgroup_counts[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t low_cluster_counts[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  if (uses_dynamic_count) {
    const loom_value_id_t dispatch_ptr =
        loom_amdgpu_lookup_dispatch_ptr_live_in(context);
    loom_type_t count_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &count_type));

    for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
      if (first_workgroup_counts[i].source_op != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_count(
            context, first_workgroup_counts[i].source_op, dispatch_ptr,
            (loom_kernel_dimension_t)i, count_type, &low_workgroup_counts[i]));
      }
      if (first_cluster_counts[i].source_op != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_emit_cluster_count(
            context, first_cluster_counts[i].source_op, dispatch_ptr,
            (loom_kernel_dimension_t)i, count_type, &low_cluster_counts[i]));
      }
    }
  }

  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    const loom_amdgpu_preamble_query_row_t* row =
        loom_amdgpu_preamble_query_row(selected_plan.plan.id);
    if (row == NULL ||
        (row->kind != LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_COUNT &&
         row->kind != LOOM_AMDGPU_PREAMBLE_QUERY_KIND_CLUSTER_COUNT)) {
      continue;
    }
    const loom_op_t* source_op = selected_plan.source_op;
    const loom_kernel_dimension_t dimension =
        loom_amdgpu_preamble_query_dimension(source_op, row);
    IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
    const loom_value_id_t source_result =
        loom_amdgpu_preamble_query_result(source_op, row);
    const loom_amdgpu_preamble_query_plan_t* query =
        selected_plan.plan.target_data;
    if (!iree_any_bit_set(query->flags,
                          LOOM_AMDGPU_PREAMBLE_VALUE_REFERENCED)) {
      continue;
    }
    if (iree_any_bit_set(query->flags, LOOM_AMDGPU_PREAMBLE_VALUE_EXACT)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_query_constant(
          context, source_op, source_result, query->value));
    } else {
      const loom_value_id_t low_count =
          row->kind == LOOM_AMDGPU_PREAMBLE_QUERY_KIND_WORKGROUP_COUNT
              ? low_workgroup_counts[dimension]
              : low_cluster_counts[dimension];
      IREE_ASSERT_NE(low_count, LOOM_VALUE_ID_INVALID);
      IREE_RETURN_IF_ERROR(loom_amdgpu_preamble_bind_query_result(
          context, source_op, source_result, low_count));
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_emit_entry_setup(context));
  return loom_amdgpu_emit_hal_kernel_predicate_argument_setup(context);
}

iree_status_t loom_amdgpu_lower_preamble_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t selected_plan) {
  const loom_amdgpu_preamble_query_plan_t* plan = selected_plan.target_data;
  switch (source_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID: {
      uint32_t packed_dimension_count = 0;
      const loom_value_id_t packed_workitem_id =
          loom_amdgpu_lookup_packed_workitem_id_live_in(
              context, &packed_dimension_count);
      if (packed_workitem_id != LOOM_VALUE_ID_INVALID) {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_id_dimension(source_op);
        if ((uint32_t)dimension >= packed_dimension_count) {
          IREE_ASSERT_UNREACHABLE(
              "selected AMDGPU packed workitem-id dimension");
          IREE_BUILTIN_UNREACHABLE();
        }
        loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_workitem_id_extract(
            context, source_op, packed_workitem_id, dimension,
            packed_dimension_count > 1, &low_result));
        return loom_amdgpu_preamble_bind_query_result(
            context, source_op, loom_kernel_workitem_id_result(source_op),
            low_result);
      }
      // Entry setup already bound the unpacked workitem identity.
      return iree_ok_status();
    }
    case LOOM_OP_KERNEL_WORKGROUP_SIZE:
    case LOOM_OP_KERNEL_SUBGROUP_SIZE:
    case LOOM_OP_KERNEL_SUBGROUP_COUNT:
    case LOOM_OP_KERNEL_CLUSTER_SIZE:
      return loom_amdgpu_emit_query_constant(
          context, source_op, loom_op_const_results(source_op)[0], plan->value);
    case LOOM_OP_KERNEL_WORKGROUP_ID:
    case LOOM_OP_KERNEL_WORKGROUP_COUNT:
    case LOOM_OP_KERNEL_CLUSTER_ID:
    case LOOM_OP_KERNEL_CLUSTER_WORKGROUP_ID:
    case LOOM_OP_KERNEL_CLUSTER_WORKGROUP_FLAT_ID:
    case LOOM_OP_KERNEL_CLUSTER_COUNT:
      // Entry setup already bound these dispatch values.
      return iree_ok_status();
    case LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID: {
      const loom_kernel_dimension_t dimension =
          loom_kernel_workitem_dispatch_id_dimension(source_op);
      IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
      loom_value_id_t low_workgroup_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_current_workgroup_id(
          context, dimension, &low_workgroup_id));

      uint32_t packed_dimension_count = 0;
      const loom_value_id_t packed_workitem_id =
          loom_amdgpu_lookup_packed_workitem_id_live_in(
              context, &packed_dimension_count);
      loom_value_id_t low_workitem_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_workitem_id(
          context, source_op, packed_dimension_count, packed_workitem_id,
          dimension, &low_workitem_id));
      return loom_amdgpu_emit_workitem_dispatch_id(
          context, source_op, plan, low_workgroup_id, low_workitem_id);
    }
    case LOOM_OP_KERNEL_SUBGROUP_ID:
      return loom_amdgpu_emit_subgroup_linear_query(
          context, source_op, plan, loom_kernel_subgroup_id_result(source_op),
          /*is_lane_id=*/false);
    case LOOM_OP_KERNEL_SUBGROUP_LANE_ID:
      return loom_amdgpu_emit_subgroup_linear_query(
          context, source_op, plan,
          loom_kernel_subgroup_lane_id_result(source_op),
          /*is_lane_id=*/true);
    default:
      IREE_ASSERT_UNREACHABLE("AMDGPU preamble plan selected unknown op kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_preamble(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  const loom_amdgpu_preamble_query_row_t* row =
      loom_amdgpu_preamble_query_row(op->kind);
  if (row == NULL) {
    return iree_ok_status();
  }
  *out_handled = true;
  const loom_amdgpu_preamble_query_facts_t facts = {
      .module = loom_target_low_legality_module(context),
      .function = loom_target_low_legality_function(context),
      .bundle = bundle,
      .target_facts = loom_amdgpu_target_facts_cast(
          loom_target_low_legality_target_facts(context)),
      .fact_table = loom_target_low_legality_fact_table(context),
  };
  iree_string_view_t reason = iree_string_view_empty();
  loom_amdgpu_preamble_query_plan_t query = {0};
  const bool satisfied = loom_amdgpu_preamble_query_launch_facts_satisfied(
      &facts, op, row, &query, &reason);
  if (!satisfied) {
    return loom_amdgpu_low_legality_reject(context, op, reason);
  }
  return iree_ok_status();
}
