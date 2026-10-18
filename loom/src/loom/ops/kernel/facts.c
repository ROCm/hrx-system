// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the kernel dialect.

#include "loom/target/facts.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ir/float_facts.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#define LOOM_KERNEL_DEFAULT_MAX_SUBGROUP_SIZE 128u

static const loom_value_fact_topology_axis_t
    kKernelDimensionTopologyAxes[LOOM_KERNEL_DIMENSION_COUNT_] = {
        [LOOM_KERNEL_DIMENSION_X] = LOOM_VALUE_FACT_TOPOLOGY_AXIS_X,
        [LOOM_KERNEL_DIMENSION_Y] = LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y,
        [LOOM_KERNEL_DIMENSION_Z] = LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z,
};

static loom_value_fact_topology_axis_t loom_kernel_dimension_topology_axis(
    loom_kernel_dimension_t dimension) {
  IREE_ASSERT_LT((uint32_t)dimension,
                 IREE_ARRAYSIZE(kKernelDimensionTopologyAxes));
  return kKernelDimensionTopologyAxes[dimension];
}

static loom_value_facts_t loom_kernel_hal_coordinate_facts(void) {
  return loom_value_facts_make(0, (int64_t)UINT32_MAX, 1);
}

static loom_value_facts_t loom_kernel_hal_positive_u32_facts(void) {
  return loom_value_facts_make(1, (int64_t)UINT32_MAX, 1);
}

static bool loom_kernel_value_integer_domain_facts(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  if (!loom_value_facts_scalar_type_domain(loom_type_element_type(type),
                                           &lower_bound, &upper_bound)) {
    return false;
  }
  *out_facts = loom_value_facts_make(lower_bound, upper_bound, 1);
  return true;
}

static loom_value_facts_t loom_kernel_subgroup_lane_mask_facts(
    const loom_module_t* module, loom_value_id_t mask) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (loom_kernel_value_integer_domain_facts(module, mask, &facts)) {
    loom_value_facts_mark_subgroup_lane_mask(&facts);
  }
  return facts;
}

static loom_value_facts_t loom_kernel_positive_u32_extent_facts(
    loom_value_facts_t facts, uint32_t maximum_extent) {
  const int64_t maximum =
      maximum_extent == 0 ? (int64_t)UINT32_MAX : (int64_t)maximum_extent;
  if (loom_value_facts_is_float(facts)) {
    return loom_value_facts_make(1, maximum, 1);
  }
  const int64_t lower_bound = iree_max(facts.range_lo, 1);
  const int64_t upper_bound = iree_min(facts.range_hi, maximum);
  if (lower_bound > upper_bound) {
    return loom_value_facts_make(1, maximum, 1);
  }
  loom_value_facts_t extent =
      loom_value_facts_make(lower_bound, upper_bound, facts.known_divisor);
  extent.flags |= facts.flags & LOOM_VALUE_FACT_POWER_OF_TWO;
  return extent;
}

static loom_value_facts_t loom_kernel_intersect_integer_facts(
    loom_value_facts_t lhs, loom_value_facts_t rhs) {
  if (loom_value_facts_is_float(lhs)) {
    return rhs;
  }
  if (loom_value_facts_is_float(rhs)) {
    return lhs;
  }
  const int64_t lower_bound = iree_max(lhs.range_lo, rhs.range_lo);
  const int64_t upper_bound = iree_min(lhs.range_hi, rhs.range_hi);
  if (lower_bound > upper_bound) {
    return loom_value_facts_unknown();
  }
  int64_t divisor = 1;
  if (!iree_math_checked_lcm_i64(lhs.known_divisor, rhs.known_divisor,
                                 &divisor)) {
    divisor = iree_math_gcd_i64(lhs.known_divisor, rhs.known_divisor);
  }
  loom_value_facts_t facts =
      loom_value_facts_make(lower_bound, upper_bound, divisor);
  if (!loom_value_facts_is_exact(facts)) {
    facts.flags |= (lhs.flags | rhs.flags) & (LOOM_VALUE_FACT_POWER_OF_TWO);
  }
  return facts;
}

static loom_value_facts_t loom_kernel_coordinate_from_extent_facts(
    loom_value_facts_t extent_facts) {
  if (loom_value_facts_is_float(extent_facts) || extent_facts.range_hi < 1 ||
      extent_facts.range_hi > UINT32_MAX) {
    return loom_kernel_hal_coordinate_facts();
  }
  return loom_value_facts_make(0, extent_facts.range_hi - 1, 1);
}

static uint32_t loom_kernel_workgroup_size_dim(
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

static uint32_t loom_kernel_workgroup_cluster_size_dim(
    const loom_target_workgroup_cluster_size_t* size,
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

static uint32_t loom_kernel_grid_size_dim(const loom_target_grid_size_t* size,
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

static uint32_t loom_kernel_workgroup_count_dim(
    const loom_target_workgroup_count_limit_t* size,
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

static bool loom_kernel_workgroup_size_flat_product(
    const loom_target_workgroup_size_t* size, uint32_t* out_flat_size) {
  if (size->x == 0 || size->y == 0 || size->z == 0) {
    return false;
  }
  const uint64_t flat_size = (uint64_t)size->x * size->y * size->z;
  if (flat_size > UINT32_MAX) {
    return false;
  }
  *out_flat_size = (uint32_t)flat_size;
  return true;
}

static bool loom_kernel_workgroup_cluster_size_flat_product(
    const loom_target_workgroup_cluster_size_t* size, uint32_t* out_flat_size) {
  if (size->x == 0 || size->y == 0 || size->z == 0) {
    return false;
  }
  const uint64_t flat_size = (uint64_t)size->x * size->y * size->z;
  if (flat_size > UINT32_MAX) {
    return false;
  }
  *out_flat_size = (uint32_t)flat_size;
  return true;
}

static uint32_t loom_kernel_ceil_div_u32(uint32_t numerator,
                                         uint32_t denominator) {
  IREE_ASSERT_NE(denominator, 0u);
  return numerator == 0 ? 0 : 1u + (numerator - 1u) / denominator;
}

static uint32_t loom_kernel_max_workgroup_count(
    const loom_target_snapshot_t* snapshot,
    const loom_target_workgroup_size_t* required_workgroup_size,
    loom_kernel_dimension_t dimension) {
  uint32_t max_count = loom_kernel_workgroup_count_dim(
      &snapshot->max_workgroup_count, dimension);
  const uint32_t max_grid_size =
      loom_kernel_grid_size_dim(&snapshot->max_grid_size, dimension);
  if (max_grid_size != 0) {
    uint32_t workgroup_size =
        loom_kernel_workgroup_size_dim(required_workgroup_size, dimension);
    if (workgroup_size == 0) {
      workgroup_size = 1;
    }
    const uint32_t grid_limited_count = max_grid_size / workgroup_size;
    max_count = max_count == 0 ? grid_limited_count
                               : iree_min(max_count, grid_limited_count);
  }
  return max_count;
}

static void loom_kernel_mark_workitem_topology_domain(
    loom_kernel_dimension_t dimension, loom_value_facts_t* facts) {
  loom_value_facts_mark_topology_domain(
      facts, LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID,
      loom_kernel_dimension_topology_axis(dimension));
}

static void loom_kernel_mark_workgroup_topology_domain(
    loom_kernel_dimension_t dimension, loom_value_facts_t* facts) {
  loom_value_facts_mark_topology_domain(
      facts, LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKGROUP_ID,
      loom_kernel_dimension_topology_axis(dimension));
}

static void loom_kernel_mark_cluster_topology_domain(
    loom_kernel_dimension_t dimension, loom_value_facts_t* facts) {
  loom_value_facts_mark_topology_domain(
      facts, LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_ID,
      loom_kernel_dimension_topology_axis(dimension));
}

static void loom_kernel_mark_cluster_workgroup_topology_domain(
    loom_kernel_dimension_t dimension, loom_value_facts_t* facts) {
  loom_value_facts_mark_topology_domain(
      facts, LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_WORKGROUP_ID,
      loom_kernel_dimension_topology_axis(dimension));
}

static bool loom_kernel_launch_config_operand_facts(
    const loom_fact_context_t* context, loom_kernel_dimension_t dimension,
    loom_value_id_t (*operand_lookup)(const loom_op_t* launch_config,
                                      loom_kernel_dimension_t dimension),
    loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (!context || !context->table ||
      !loom_kernel_def_isa(context->function.op)) {
    return false;
  }
  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(context->function.op);
  if (!launch_config) {
    return false;
  }
  const loom_value_id_t operand = operand_lookup(launch_config, dimension);
  if (operand == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  *out_facts = loom_value_fact_table_lookup(context->table, operand);
  return true;
}

static const loom_target_bundle_t* loom_kernel_target_bundle(
    const loom_fact_context_t* context) {
  const loom_target_bundle_t* bundle =
      loom_target_facts_bundle(context->target_facts);
  if (!bundle) {
    return NULL;
  }
  IREE_ASSERT(bundle->snapshot != NULL);
  IREE_ASSERT(bundle->export_plan != NULL);
  return bundle;
}

static uint32_t loom_kernel_context_fixed_subgroup_size(
    const loom_fact_context_t* context) {
  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return 0;
  }
  return bundle->snapshot->subgroup_size;
}

static bool loom_kernel_context_required_workgroup_size(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_target_workgroup_size_t* out_size) {
  *out_size = (loom_target_workgroup_size_t){0};

  if (module && loom_kernel_def_isa(context->function.op) &&
      loom_kernel_def_static_workgroup_size_from_facts(
          module, context->function.op, context->table, out_size)) {
    return true;
  }

  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return false;
  }
  *out_size = bundle->export_plan->hal_kernel.required_workgroup_size;
  return out_size->x != 0 || out_size->y != 0 || out_size->z != 0;
}

static bool loom_kernel_context_workgroup_cluster_size(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_target_workgroup_cluster_size_t* out_size) {
  *out_size = (loom_target_workgroup_cluster_size_t){0};
  if (!context || !loom_kernel_def_isa(context->function.op)) {
    return false;
  }

  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(context->function.op);
  if (!launch_config) {
    return false;
  }
  if (!loom_kernel_launch_config_has_workgroup_cluster_size(launch_config)) {
    *out_size = (loom_target_workgroup_cluster_size_t){1, 1, 1};
    return true;
  }
  if (!module) {
    return false;
  }
  return loom_kernel_def_static_workgroup_cluster_size_from_facts(
      module, context->function.op, context->table, out_size);
}

static bool loom_kernel_context_fixed_flat_workgroup_size(
    const loom_fact_context_t* context, const loom_module_t* module,
    uint32_t* out_flat_size) {
  *out_flat_size = 0;
  loom_target_workgroup_size_t required_workgroup_size = {0};
  if (!loom_kernel_context_required_workgroup_size(context, module,
                                                   &required_workgroup_size)) {
    return false;
  }
  return loom_kernel_workgroup_size_flat_product(&required_workgroup_size,
                                                 out_flat_size);
}

static uint32_t loom_kernel_context_max_flat_workgroup_size(
    const loom_fact_context_t* context, const loom_module_t* module) {
  uint32_t flat_size = 0;
  if (loom_kernel_context_fixed_flat_workgroup_size(context, module,
                                                    &flat_size)) {
    return flat_size;
  }

  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return 0;
  }
  if (bundle->snapshot->max_flat_workgroup_size != 0) {
    return bundle->snapshot->max_flat_workgroup_size;
  }
  if (loom_kernel_workgroup_size_flat_product(
          &bundle->snapshot->max_workgroup_size, &flat_size)) {
    return flat_size;
  }
  return 0;
}

static loom_value_facts_t loom_kernel_workgroup_size_target_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_kernel_dimension_t dimension) {
  loom_target_workgroup_size_t required_workgroup_size = {0};
  if (loom_kernel_context_required_workgroup_size(context, module,
                                                  &required_workgroup_size)) {
    const uint32_t fixed_workgroup_size =
        loom_kernel_workgroup_size_dim(&required_workgroup_size, dimension);
    if (fixed_workgroup_size != 0) {
      return loom_value_facts_exact_i64((int64_t)fixed_workgroup_size);
    }
  }

  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_kernel_hal_positive_u32_facts();
  }
  const uint32_t max_workgroup_size = loom_kernel_workgroup_size_dim(
      &bundle->snapshot->max_workgroup_size, dimension);
  if (max_workgroup_size != 0) {
    return loom_value_facts_make(1, (int64_t)max_workgroup_size, 1);
  }
  return loom_kernel_hal_positive_u32_facts();
}

static loom_value_facts_t loom_kernel_workgroup_count_target_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_kernel_dimension_t dimension) {
  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_kernel_hal_positive_u32_facts();
  }
  loom_target_workgroup_size_t required_workgroup_size = {0};
  if (!loom_kernel_context_required_workgroup_size(context, module,
                                                   &required_workgroup_size)) {
    required_workgroup_size =
        bundle->export_plan->hal_kernel.required_workgroup_size;
  }
  const uint32_t max_workgroup_count = loom_kernel_max_workgroup_count(
      bundle->snapshot, &required_workgroup_size, dimension);
  if (max_workgroup_count == 0) {
    return loom_kernel_hal_positive_u32_facts();
  }
  return loom_value_facts_make(1, (int64_t)max_workgroup_count, 1);
}

static loom_value_facts_t loom_kernel_launch_workgroup_size_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_kernel_dimension_t dimension) {
  loom_value_facts_t target_facts =
      loom_kernel_workgroup_size_target_facts(context, module, dimension);
  loom_value_facts_t launch_facts = {0};
  if (!loom_kernel_launch_config_operand_facts(
          context, dimension, loom_kernel_launch_config_workgroup_size_operand,
          &launch_facts)) {
    return target_facts;
  }

  const uint32_t maximum_extent =
      target_facts.range_hi > 0 && target_facts.range_hi <= UINT32_MAX
          ? (uint32_t)target_facts.range_hi
          : 0;
  launch_facts =
      loom_kernel_positive_u32_extent_facts(launch_facts, maximum_extent);
  return loom_kernel_intersect_integer_facts(launch_facts, target_facts);
}

static bool loom_kernel_context_has_dense_subgroup_partition(
    const loom_fact_context_t* context, const loom_module_t* module) {
  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle) {
    return false;
  }
  if (bundle->snapshot->subgroup_partition ==
      LOOM_TARGET_SUBGROUP_PARTITION_DENSE) {
    return true;
  }
  const uint32_t subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (subgroup_size == 1) {
    return true;
  }
  if (bundle->snapshot->subgroup_partition !=
          LOOM_TARGET_SUBGROUP_PARTITION_WORKGROUP_X_ALIGNED ||
      subgroup_size == 0) {
    return false;
  }
  const loom_value_facts_t workgroup_x =
      loom_kernel_launch_workgroup_size_facts(context, module,
                                              LOOM_KERNEL_DIMENSION_X);
  return loom_value_facts_divisible_by(workgroup_x, subgroup_size);
}

static loom_value_facts_t loom_kernel_launch_workgroup_count_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_kernel_dimension_t dimension) {
  loom_value_facts_t target_facts =
      loom_kernel_workgroup_count_target_facts(context, module, dimension);
  loom_value_facts_t launch_facts = {0};
  if (!loom_kernel_launch_config_operand_facts(
          context, dimension, loom_kernel_launch_config_workgroup_count_operand,
          &launch_facts)) {
    return target_facts;
  }

  const uint32_t maximum_extent =
      target_facts.range_hi > 0 && target_facts.range_hi <= UINT32_MAX
          ? (uint32_t)target_facts.range_hi
          : 0;
  launch_facts =
      loom_kernel_positive_u32_extent_facts(launch_facts, maximum_extent);
  return loom_kernel_intersect_integer_facts(launch_facts, target_facts);
}

static loom_value_facts_t loom_kernel_launch_workgroup_cluster_size_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_kernel_dimension_t dimension) {
  loom_target_workgroup_cluster_size_t cluster_size = {0};
  if (!loom_kernel_context_workgroup_cluster_size(context, module,
                                                  &cluster_size)) {
    return loom_kernel_hal_positive_u32_facts();
  }
  const uint32_t extent =
      loom_kernel_workgroup_cluster_size_dim(&cluster_size, dimension);
  return extent == 0 ? loom_kernel_hal_positive_u32_facts()
                     : loom_value_facts_exact_i64((int64_t)extent);
}

static loom_value_facts_t loom_kernel_positive_extent_quotient_facts(
    loom_value_facts_t numerator_facts, uint32_t denominator) {
  if (denominator == 0 || loom_value_facts_is_float(numerator_facts) ||
      numerator_facts.range_lo < 1 || numerator_facts.range_hi > UINT32_MAX) {
    return loom_kernel_hal_positive_u32_facts();
  }
  const int64_t divisor = (int64_t)denominator;
  const int64_t lower_bound = numerator_facts.range_lo / divisor +
                              (numerator_facts.range_lo % divisor != 0);
  const int64_t upper_bound = numerator_facts.range_hi / divisor;
  if (lower_bound < 1 || lower_bound > upper_bound) {
    return loom_kernel_hal_positive_u32_facts();
  }
  return loom_value_facts_make(lower_bound, upper_bound, 1);
}

static loom_value_facts_t loom_kernel_launch_workgroup_cluster_count_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_kernel_dimension_t dimension) {
  const loom_value_facts_t workgroup_count_facts =
      loom_kernel_launch_workgroup_count_facts(context, module, dimension);
  const loom_value_facts_t cluster_size_facts =
      loom_kernel_launch_workgroup_cluster_size_facts(context, module,
                                                      dimension);
  if (!loom_value_facts_is_exact(cluster_size_facts) ||
      cluster_size_facts.range_lo < 1 ||
      cluster_size_facts.range_lo > UINT32_MAX) {
    return loom_kernel_hal_positive_u32_facts();
  }
  return loom_kernel_positive_extent_quotient_facts(
      workgroup_count_facts, (uint32_t)cluster_size_facts.range_lo);
}

static loom_value_facts_t loom_kernel_workitem_dispatch_id_target_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_kernel_dimension_t dimension) {
  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_kernel_hal_coordinate_facts();
  }
  const uint32_t max_grid_size =
      loom_kernel_grid_size_dim(&bundle->snapshot->max_grid_size, dimension);
  if (max_grid_size != 0) {
    return loom_value_facts_make(0, (int64_t)max_grid_size - 1, 1);
  }

  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_kernel_context_required_workgroup_size(context, module,
                                                   &workgroup_size)) {
    workgroup_size = bundle->snapshot->max_workgroup_size;
  }
  const uint32_t size =
      loom_kernel_workgroup_size_dim(&workgroup_size, dimension);
  const uint32_t count = loom_kernel_workgroup_count_dim(
      &bundle->snapshot->max_workgroup_count, dimension);
  if (size == 0 || count == 0) {
    return loom_kernel_hal_coordinate_facts();
  }
  const uint64_t max_dispatch_size = (uint64_t)size * count;
  if (max_dispatch_size == 0 || max_dispatch_size > UINT32_MAX) {
    return loom_kernel_hal_coordinate_facts();
  }
  return loom_value_facts_make(0, (int64_t)max_dispatch_size - 1, 1);
}

static loom_value_facts_t loom_kernel_launch_workitem_dispatch_id_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_kernel_dimension_t dimension) {
  loom_value_facts_t target_facts =
      loom_kernel_workitem_dispatch_id_target_facts(context, module, dimension);
  const loom_value_facts_t count_facts =
      loom_kernel_launch_workgroup_count_facts(context, module, dimension);
  const loom_value_facts_t size_facts =
      loom_kernel_launch_workgroup_size_facts(context, module, dimension);

  int64_t upper_bound = 0;
  if (iree_checked_mul_i64(count_facts.range_hi, size_facts.range_hi,
                           &upper_bound) &&
      upper_bound >= 1) {
    const loom_value_facts_t launch_facts =
        loom_value_facts_make(0, upper_bound - 1, 1);
    target_facts =
        loom_kernel_intersect_integer_facts(launch_facts, target_facts);
  }
  // A unit-size axis makes the dispatch coordinate exactly the workgroup
  // coordinate, even when the other axes contain several invocations.
  int64_t size = 0;
  if (loom_value_facts_as_exact_i64(size_facts, &size) && size == 1) {
    loom_value_facts_mark_workgroup_uniform(&target_facts);
    loom_kernel_mark_workgroup_topology_domain(dimension, &target_facts);
  } else {
    loom_value_facts_mark_lane_varying(&target_facts);
  }
  return target_facts;
}

static uint32_t loom_kernel_max_subgroup_size(
    const loom_fact_context_t* context) {
  const uint32_t fixed_subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (fixed_subgroup_size != 0) {
    return fixed_subgroup_size;
  }
  // A partial workgroup limits active lanes, not the subgroup execution width.
  return LOOM_KERNEL_DEFAULT_MAX_SUBGROUP_SIZE;
}

static uint32_t loom_kernel_max_subgroup_lane_count(
    const loom_fact_context_t* context, const loom_module_t* module) {
  uint32_t max_lane_count = loom_kernel_max_subgroup_size(context);
  if (!loom_kernel_context_has_dense_subgroup_partition(context, module)) {
    return max_lane_count;
  }
  const uint32_t max_flat_workgroup_size =
      loom_kernel_context_max_flat_workgroup_size(context, module);
  if (max_flat_workgroup_size != 0) {
    max_lane_count = iree_min(max_lane_count, max_flat_workgroup_size);
  }
  return max_lane_count;
}

static uint32_t loom_kernel_max_subgroup_count(
    const loom_fact_context_t* context, const loom_module_t* module) {
  const uint32_t max_flat_workgroup_size =
      loom_kernel_context_max_flat_workgroup_size(context, module);
  if (max_flat_workgroup_size == 0) {
    return 0;
  }
  const uint32_t fixed_subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (fixed_subgroup_size != 0 &&
      loom_kernel_context_has_dense_subgroup_partition(context, module)) {
    return loom_kernel_ceil_div_u32(max_flat_workgroup_size,
                                    fixed_subgroup_size);
  }
  return max_flat_workgroup_size;
}

static bool loom_kernel_exact_subgroup_count(const loom_fact_context_t* context,
                                             const loom_module_t* module,
                                             uint32_t* out_count) {
  *out_count = 0;
  const uint32_t fixed_subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (fixed_subgroup_size == 0 ||
      !loom_kernel_context_has_dense_subgroup_partition(context, module)) {
    return false;
  }
  uint32_t flat_workgroup_size = 0;
  if (!loom_kernel_context_fixed_flat_workgroup_size(context, module,
                                                     &flat_workgroup_size)) {
    return false;
  }
  *out_count =
      loom_kernel_ceil_div_u32(flat_workgroup_size, fixed_subgroup_size);
  return *out_count != 0;
}

static uint32_t loom_kernel_min_subgroup_count(
    const loom_fact_context_t* context, const loom_module_t* module) {
  uint32_t flat_workgroup_size = 0;
  if (!loom_kernel_context_fixed_flat_workgroup_size(context, module,
                                                     &flat_workgroup_size)) {
    return 1;
  }
  const uint32_t max_subgroup_size = loom_kernel_max_subgroup_size(context);
  if (max_subgroup_size == 0) {
    return 1;
  }
  const uint32_t min_count =
      loom_kernel_ceil_div_u32(flat_workgroup_size, max_subgroup_size);
  return min_count == 0 ? 1 : min_count;
}

iree_status_t loom_kernel_workitem_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  result_facts[0] = loom_kernel_coordinate_from_extent_facts(
      loom_kernel_launch_workgroup_size_facts(
          context, module, loom_kernel_workitem_id_dimension(op)));
  loom_value_facts_mark_lane_varying(&result_facts[0]);
  loom_kernel_mark_workitem_topology_domain(
      loom_kernel_workitem_id_dimension(op), &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_workgroup_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  result_facts[0] = loom_kernel_coordinate_from_extent_facts(
      loom_kernel_launch_workgroup_count_facts(
          context, module, loom_kernel_workgroup_id_dimension(op)));
  loom_value_facts_mark_workgroup_uniform(&result_facts[0]);
  loom_kernel_mark_workgroup_topology_domain(
      loom_kernel_workgroup_id_dimension(op), &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_cluster_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  const loom_kernel_dimension_t dimension =
      loom_kernel_cluster_id_dimension(op);
  result_facts[0] = loom_kernel_coordinate_from_extent_facts(
      loom_kernel_launch_workgroup_cluster_count_facts(context, module,
                                                       dimension));
  loom_value_facts_mark_cluster_uniform(&result_facts[0]);
  loom_kernel_mark_cluster_topology_domain(dimension, &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_cluster_workgroup_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  const loom_kernel_dimension_t dimension =
      loom_kernel_cluster_workgroup_id_dimension(op);
  result_facts[0] = loom_kernel_coordinate_from_extent_facts(
      loom_kernel_launch_workgroup_cluster_size_facts(context, module,
                                                      dimension));
  if (loom_value_facts_is_zero(result_facts[0])) {
    loom_value_facts_mark_cluster_uniform(&result_facts[0]);
  } else {
    loom_value_facts_mark_workgroup_uniform(&result_facts[0]);
  }
  loom_kernel_mark_cluster_workgroup_topology_domain(dimension,
                                                     &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_cluster_workgroup_flat_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)op;
  (void)operand_facts;
  loom_target_workgroup_cluster_size_t cluster_size = {0};
  uint32_t flat_size = 0;
  if (loom_kernel_context_workgroup_cluster_size(context, module,
                                                 &cluster_size) &&
      loom_kernel_workgroup_cluster_size_flat_product(&cluster_size,
                                                      &flat_size)) {
    result_facts[0] = loom_value_facts_make(0, (int64_t)flat_size - 1, 1);
  } else {
    result_facts[0] = loom_kernel_hal_coordinate_facts();
  }
  if (loom_value_facts_is_zero(result_facts[0])) {
    loom_value_facts_mark_cluster_uniform(&result_facts[0]);
  } else {
    loom_value_facts_mark_workgroup_uniform(&result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_cluster_size_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  result_facts[0] = loom_kernel_launch_workgroup_cluster_size_facts(
      context, module, loom_kernel_cluster_size_dimension(op));
  loom_value_facts_mark_cluster_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_cluster_count_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  result_facts[0] = loom_kernel_launch_workgroup_cluster_count_facts(
      context, module, loom_kernel_cluster_count_dimension(op));
  loom_value_facts_mark_cluster_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_workgroup_size_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  result_facts[0] = loom_kernel_launch_workgroup_size_facts(
      context, module, loom_kernel_workgroup_size_dimension(op));
  loom_value_facts_mark_cluster_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_workgroup_count_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  result_facts[0] = loom_kernel_launch_workgroup_count_facts(
      context, module, loom_kernel_workgroup_count_dimension(op));
  loom_value_facts_mark_cluster_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_workitem_dispatch_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  result_facts[0] = loom_kernel_launch_workitem_dispatch_id_facts(
      context, module, loom_kernel_workitem_dispatch_id_dimension(op));
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_lane_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)op;
  (void)operand_facts;
  const uint32_t max_lane_count =
      loom_kernel_max_subgroup_lane_count(context, module);
  result_facts[0] = loom_value_facts_make(0, (int64_t)max_lane_count - 1, 1);
  loom_value_facts_mark_lane_varying(&result_facts[0]);
  loom_value_facts_mark_topology_domain(
      &result_facts[0], LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID,
      LOOM_VALUE_FACT_TOPOLOGY_AXIS_LANE);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_size_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)op;
  (void)operand_facts;
  const uint32_t fixed_subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (fixed_subgroup_size != 0) {
    result_facts[0] = loom_value_facts_exact_i64((int64_t)fixed_subgroup_size);
    return iree_ok_status();
  }
  const uint32_t max_subgroup_size = loom_kernel_max_subgroup_size(context);
  result_facts[0] = loom_value_facts_make(1, (int64_t)max_subgroup_size, 1);
  loom_value_facts_mark_workgroup_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)op;
  (void)operand_facts;
  uint32_t exact_count = 0;
  if (loom_kernel_exact_subgroup_count(context, module, &exact_count)) {
    if (exact_count == 1) {
      result_facts[0] = loom_value_facts_exact_i64(0);
    } else {
      result_facts[0] = loom_value_facts_make(0, (int64_t)exact_count - 1, 1);
      loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
    }
    return iree_ok_status();
  }
  const uint32_t max_subgroup_count =
      loom_kernel_max_subgroup_count(context, module);
  if (max_subgroup_count == 0) {
    result_facts[0] = loom_kernel_hal_coordinate_facts();
  } else {
    result_facts[0] =
        loom_value_facts_make(0, (int64_t)max_subgroup_count - 1, 1);
  }
  loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_count_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)op;
  (void)operand_facts;
  uint32_t exact_count = 0;
  if (loom_kernel_exact_subgroup_count(context, module, &exact_count)) {
    result_facts[0] = loom_value_facts_exact_i64((int64_t)exact_count);
    return iree_ok_status();
  }
  const uint32_t max_subgroup_count =
      loom_kernel_max_subgroup_count(context, module);
  if (max_subgroup_count == 0) {
    result_facts[0] = loom_kernel_hal_positive_u32_facts();
  } else {
    uint32_t min_subgroup_count =
        loom_kernel_min_subgroup_count(context, module);
    if (min_subgroup_count > max_subgroup_count) {
      min_subgroup_count = max_subgroup_count;
    }
    result_facts[0] = loom_value_facts_make((int64_t)min_subgroup_count,
                                            (int64_t)max_subgroup_count, 1);
  }
  loom_value_facts_mark_workgroup_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_vote_any_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  (void)op;
  (void)operand_facts;
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_vote_all_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  (void)op;
  (void)operand_facts;
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_vote_ballot_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)operand_facts;
  result_facts[0] = loom_kernel_subgroup_lane_mask_facts(
      module, loom_kernel_subgroup_vote_ballot_mask(op));
  if (loom_value_facts_is_subgroup_lane_mask(result_facts[0])) {
    loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_active_mask_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)operand_facts;
  result_facts[0] = loom_kernel_subgroup_lane_mask_facts(
      module, loom_kernel_subgroup_active_mask_mask(op));
  if (loom_value_facts_is_subgroup_lane_mask(result_facts[0])) {
    loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_match_any_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  result_facts[0] = loom_kernel_subgroup_lane_mask_facts(
      module, loom_kernel_subgroup_match_any_mask(op));
  if (!loom_value_facts_is_subgroup_lane_mask(result_facts[0])) {
    return iree_ok_status();
  }
  if (loom_value_facts_is_subgroup_uniform(operand_facts[0])) {
    loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  } else {
    loom_value_facts_mark_lane_varying(&result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_match_all_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)operand_facts;
  result_facts[0] = loom_kernel_subgroup_lane_mask_facts(
      module, loom_kernel_subgroup_match_all_mask(op));
  if (loom_value_facts_is_subgroup_lane_mask(result_facts[0])) {
    loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  }
  result_facts[1] = loom_value_facts_make(0, 1, 1);
  loom_value_facts_mark_subgroup_uniform(&result_facts[1]);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_broadcast_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  (void)op;
  result_facts[0] = operand_facts[0];
  const loom_value_fact_uniform_scope_t value_scope =
      loom_value_facts_uniform_scope(operand_facts[0]);
  if (value_scope != LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE) {
    loom_value_facts_mark_uniform_at_scope(&result_facts[0], value_scope);
  } else if (loom_value_facts_is_subgroup_uniform(operand_facts[1])) {
    loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  } else {
    loom_value_facts_propagate_binary_distribution(
        operand_facts[0], operand_facts[1], &result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_broadcast_first_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  (void)op;
  result_facts[0] = operand_facts[0];
  loom_value_facts_mark_uniform_at_scope(
      &result_facts[0],
      iree_max(LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP,
               loom_value_facts_uniform_scope(operand_facts[0])));
  return iree_ok_status();
}

// Bounds a sum of independently varying inputs, not a product of equal values.
// Fixed-width overflow invalidates the interval and its divisibility proof.
static loom_value_facts_t loom_kernel_integer_sum_facts(
    const loom_module_t* module, loom_value_id_t result,
    loom_value_facts_t input, uint32_t minimum_count, uint32_t maximum_count) {
  loom_value_facts_t domain = {0};
  if (!loom_kernel_value_integer_domain_facts(module, result, &domain)) {
    return loom_value_facts_unknown();
  }
  if (maximum_count == 0) {
    return loom_value_facts_exact_i64(0);
  }
  input =
      loom_value_facts_clamp_domain(input, domain.range_lo, domain.range_hi);
  int64_t lower_minimum = 0;
  int64_t lower_maximum = 0;
  int64_t upper_minimum = 0;
  int64_t upper_maximum = 0;
  if (!iree_checked_mul_i64(input.range_lo, minimum_count, &lower_minimum) ||
      !iree_checked_mul_i64(input.range_lo, maximum_count, &lower_maximum) ||
      !iree_checked_mul_i64(input.range_hi, minimum_count, &upper_minimum) ||
      !iree_checked_mul_i64(input.range_hi, maximum_count, &upper_maximum)) {
    return loom_value_facts_unknown();
  }
  const int64_t lower_bound = iree_min(lower_minimum, lower_maximum);
  const int64_t upper_bound = iree_max(upper_minimum, upper_maximum);
  if (lower_bound < domain.range_lo || upper_bound > domain.range_hi) {
    return loom_value_facts_unknown();
  }
  return loom_value_facts_make(lower_bound, upper_bound, input.known_divisor);
}

// Active population is bounded by both execution width and workgroup size,
// even when lane IDs are sparse and a dense subgroup partition is not promised.
static uint32_t loom_kernel_max_active_subgroup_size(
    const loom_fact_context_t* context, const loom_module_t* module,
    int64_t cluster_size) {
  uint32_t count = loom_kernel_max_subgroup_size(context);
  const uint32_t workgroup_count =
      loom_kernel_context_max_flat_workgroup_size(context, module);
  if (workgroup_count != 0) {
    count = iree_min(count, workgroup_count);
  }
  if (cluster_size > 0 && (uint64_t)cluster_size < count) {
    count = (uint32_t)cluster_size;
  }
  return count;
}

// Extrema select an input value under both native and preserve contracts.
// Idempotence alone is insufficient: bitwise combines need not select either
// operand, and arithmetic combines may introduce new floating-point values.
static bool loom_kernel_combining_is_float_extremum(
    loom_combining_kind_t kind) {
  return kind == LOOM_COMBINING_KIND_MINIMUMF ||
         kind == LOOM_COMBINING_KIND_MAXIMUMF ||
         kind == LOOM_COMBINING_KIND_MINNUMF ||
         kind == LOOM_COMBINING_KIND_MAXNUMF;
}

static loom_value_facts_t loom_kernel_empty_prefix_element_facts(
    loom_value_facts_t input, loom_value_facts_t seed) {
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_meet_float(&input, &seed, &result);
  // Even uniform inputs can differ from the empty-prefix seed. Element
  // extraction consumes this distribution independently of the outer vector.
  loom_value_facts_mark_lane_varying(&result);
  return result;
}

// An exclusive prefix is either empty or an extremum of authored inputs. Join
// the typed seed without combining it into the nonempty result. Reuse the
// existing bounded summaries rather than allocating facts per vector element.
static iree_status_t loom_kernel_float_exclusive_scan_facts(
    loom_fact_context_t* context, loom_type_t type, loom_combining_kind_t kind,
    loom_value_facts_t input, loom_value_facts_t* out) {
  double seed_value = 0.0;
  (void)loom_combining_kind_float_scan_seed(kind, &seed_value);
  const loom_value_facts_t seed =
      loom_value_facts_exact_float(loom_type_element_type(type), seed_value);
  *out = loom_kernel_empty_prefix_element_facts(input, seed);
  if (loom_type_is_scalar(type)) {
    return iree_ok_status();
  }

  loom_value_facts_t extension = loom_value_facts_unknown();
  loom_value_fact_uniform_element_t uniform = {0};
  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_uniform_element(context, input, &uniform)) {
    IREE_RETURN_IF_ERROR(loom_value_facts_make_uniform_element(
        context, loom_kernel_empty_prefix_element_facts(uniform.element, seed),
        &extension));
  } else if (loom_value_facts_query_small_static_lanes(context, input,
                                                       &lanes)) {
    loom_value_facts_t joined[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT];
    for (iree_host_size_t i = 0; i < lanes.count; ++i) {
      joined[i] = loom_kernel_empty_prefix_element_facts(lanes.lanes[i], seed);
    }
    IREE_RETURN_IF_ERROR(loom_value_facts_make_small_static_lanes(
        context,
        (loom_value_fact_small_static_lanes_t){.lanes = joined,
                                               .count = lanes.count},
        &extension));
  }
  out->extension_id = extension.extension_id;
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_reduce_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  const bool has_cluster_size =
      loom_kernel_subgroup_reduce_has_cluster_size(op);
  const bool has_cluster_stride =
      loom_kernel_subgroup_reduce_has_cluster_stride(op);
  const loom_combining_kind_t kind = loom_kernel_subgroup_reduce_kind(op);
  result_facts[0] = loom_value_facts_unknown();
  if (kind == LOOM_COMBINING_KIND_ADDI) {
    const uint32_t maximum_count = loom_kernel_max_active_subgroup_size(
        context, module,
        has_cluster_size ? loom_kernel_subgroup_reduce_cluster_size(op) : 0);
    result_facts[0] = loom_kernel_integer_sum_facts(
        module, loom_kernel_subgroup_reduce_result(op), operand_facts[0], 1,
        maximum_count);
  } else if (loom_kernel_combining_is_float_extremum(kind)) {
    result_facts[0] = operand_facts[0];
  }
  if (!has_cluster_size && !has_cluster_stride) {
    loom_value_facts_mark_subgroup_uniform(&result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_workgroup_reduce_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  const loom_combining_kind_t kind = loom_kernel_workgroup_reduce_kind(op);
  const bool is_float_extremum = loom_kernel_combining_is_float_extremum(kind);
  const uint32_t maximum_count =
      kind == LOOM_COMBINING_KIND_ADDI
          ? loom_kernel_context_max_flat_workgroup_size(context, module)
          : 0;
  uint32_t exact_count = 0;
  const uint32_t minimum_count =
      maximum_count != 0 && loom_kernel_context_fixed_flat_workgroup_size(
                                context, module, &exact_count)
          ? exact_count
          : 1;
  const loom_value_slice_t results = loom_kernel_workgroup_reduce_results(op);
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    if (is_float_extremum) {
      result_facts[i] = operand_facts[i];
    } else if (maximum_count != 0) {
      result_facts[i] = loom_kernel_integer_sum_facts(
          module, results.values[i], operand_facts[i], minimum_count,
          maximum_count);
    } else {
      result_facts[i] = loom_value_facts_unknown();
    }
    loom_value_facts_mark_workgroup_uniform(&result_facts[i]);
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_workgroup_vote_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)operand_facts;
  const loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  if (loom_type_is_scalar(result_type) &&
      loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1) {
    result_facts[0] = loom_value_facts_make(0, 1, 1);
  } else {
    result_facts[0] = loom_value_facts_make(0, INT64_MAX, 1);
  }
  loom_value_facts_mark_workgroup_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_kernel_scan_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
  const bool subgroup = loom_kernel_subgroup_scan_isa(op);
  const loom_combining_kind_t kind = subgroup
                                         ? loom_kernel_subgroup_scan_kind(op)
                                         : loom_kernel_workgroup_scan_kind(op);
  const bool exclusive = subgroup
                             ? loom_kernel_subgroup_scan_mode(op) ==
                                   LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE
                             : loom_kernel_workgroup_scan_mode(op) ==
                                   LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE;
  const loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  if (kind == LOOM_COMBINING_KIND_ADDI) {
    uint32_t maximum_count = 0;
    if (subgroup) {
      maximum_count = loom_kernel_max_active_subgroup_size(
          context, module,
          loom_kernel_subgroup_scan_has_cluster_size(op)
              ? loom_kernel_subgroup_scan_cluster_size(op)
              : 0);
    } else {
      maximum_count =
          loom_kernel_context_max_flat_workgroup_size(context, module);
    }
    if (maximum_count != 0) {
      result_facts[0] = loom_kernel_integer_sum_facts(
          module, loom_op_const_results(op)[0], operand_facts[0],
          exclusive ? 0 : 1, maximum_count - (exclusive ? 1 : 0));
    }
  } else if (loom_kernel_combining_is_float_extremum(kind)) {
    if (exclusive) {
      IREE_RETURN_IF_ERROR(loom_kernel_float_exclusive_scan_facts(
          context, result_type, kind, operand_facts[0], &result_facts[0]));
    } else {
      result_facts[0] = operand_facts[0];
    }
  }
  if (loom_type_is_scalar(result_type) &&
      loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1) {
    loom_value_facts_mark_lane_predicate(&result_facts[0]);
  } else {
    loom_value_facts_mark_lane_varying(&result_facts[0]);
  }
  return iree_ok_status();
}
