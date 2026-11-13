// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/workgroup_collective_legalization.h"

#include <stdint.h>

#include "loom/ops/atomic.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/combining.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/combining.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/launch.h"

typedef struct loom_kernel_workgroup_topology_t {
  // Static launch dimensions used to flatten logical workitem coordinates.
  loom_target_workgroup_size_t workgroup_size;
  // Target-owned relationship between physical subgroups and logical IDs.
  loom_target_subgroup_partition_t partition;
  // Active invocations in the flattened workgroup.
  uint32_t workitem_count;
  // Physical invocations per subgroup.
  uint32_t subgroup_size;
  // Subgroups covering the workgroup, including a partial final subgroup.
  uint32_t subgroup_count;
} loom_kernel_workgroup_topology_t;

typedef struct loom_kernel_workgroup_payload_t {
  // Scalar or vector type carried by the source collective.
  loom_type_t logical_type;
  // Scratch representation, with predicates widened to byte storage.
  loom_type_t storage_type;
  // Element type of the flattened scratch view.
  loom_scalar_type_t storage_element_type;
  // Components per payload slot; one for a scalar.
  int64_t lane_count;
  // Scratch element width and minimum allocation alignment in bytes.
  int64_t element_byte_count;
  // Elements in the partial and publication slots combined.
  int64_t scratch_element_count;
  // Byte extent of the complete scratch allocation.
  int64_t scratch_byte_count;
  // Subgroup reduction value published to its partial slot.
  loom_value_id_t local_total;
  // Typed view of the compiler-owned workgroup scratch allocation.
  loom_value_id_t scratch_view;
} loom_kernel_workgroup_payload_t;

typedef struct loom_kernel_workgroup_collective_state_t {
  // Borrowed owner of the current target-legalization transaction.
  loom_target_legalization_context_t* context;
  // Borrowed insertion builder for replacement operations.
  loom_builder_t* builder;
  // Source collective location inherited by generated operations.
  loom_location_id_t location;
  // Shared combining operation for all payloads.
  loom_combining_kind_t kind;
  // Source arithmetic permissions preserved by each component reduction.
  uint8_t math_flags;
  // Qualified physical partitioning for this workgroup.
  loom_kernel_workgroup_topology_t topology;
  // Index type used by scratch slot addressing.
  loom_type_t index_type;
  // Reusable zero index in the replacement region.
  loom_value_id_t zero_index;
  // Current subgroup ordinal, bounded by the selected topology.
  loom_value_id_t subgroup_id;
  // Current physical lane within the subgroup.
  loom_value_id_t lane_id;
  // Predicate selecting each subgroup's publication leader.
  loom_value_id_t lane_is_zero;
} loom_kernel_workgroup_collective_state_t;

static loom_kernel_subgroup_scan_mode_t
loom_kernel_workgroup_subgroup_scan_mode(
    loom_kernel_workgroup_scan_mode_t mode) {
  switch (mode) {
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE:
      return LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE;
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE:
      return LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE;
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_COUNT_:
      IREE_ASSERT_UNREACHABLE("verified workgroup scan mode");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_BUILTIN_UNREACHABLE();
}

static loom_kernel_subgroup_scan_direction_t
loom_kernel_workgroup_subgroup_scan_direction(
    loom_kernel_workgroup_scan_direction_t direction) {
  switch (direction) {
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD:
      return LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD;
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_REVERSE:
      return LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE;
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_COUNT_:
      IREE_ASSERT_UNREACHABLE("verified workgroup scan direction");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_kernel_workgroup_collective_topology(
    const loom_target_legalization_context_t* context,
    loom_kernel_workgroup_topology_t* out_topology) {
  *out_topology = (loom_kernel_workgroup_topology_t){0};
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  if (!loom_kernel_def_isa(context->function.op) || bundle == NULL ||
      bundle->snapshot == NULL) {
    return false;
  }

  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_kernel_def_static_workgroup_size_from_facts(
          context->module, context->function.op, context->fact_table,
          &workgroup_size)) {
    if (bundle->export_plan == NULL ||
        bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
      return false;
    }
    workgroup_size = bundle->export_plan->hal_kernel.required_workgroup_size;
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_target_workgroup_size_flat_product_u32(&workgroup_size,
                                                   &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return false;
  }
  const uint32_t subgroup_size = bundle->snapshot->subgroup_size;
  *out_topology = (loom_kernel_workgroup_topology_t){
      .workgroup_size = workgroup_size,
      .partition = bundle->snapshot->subgroup_partition,
      .workitem_count = flat_workgroup_size,
      .subgroup_size = subgroup_size,
      .subgroup_count = subgroup_size == 0
                            ? 0
                            : flat_workgroup_size / subgroup_size +
                                  (flat_workgroup_size % subgroup_size != 0),
  };
  return true;
}

static bool loom_kernel_workgroup_collective_math_is_supported(
    loom_combining_kind_t kind, uint8_t math_flags) {
  // Ordinary scalar/vector arithmetic does not establish source-format
  // preservation for sums and products. Extrema retain their fixed semantics
  // independently of the collective arithmetic mode.
  return math_flags == 0 ||
         (kind != LOOM_COMBINING_KIND_ADDF && kind != LOOM_COMBINING_KIND_MULF);
}

static bool loom_kernel_workgroup_payload_initialize(
    const loom_target_legalization_context_t* context, loom_type_t logical_type,
    uint32_t scratch_slot_count, loom_kernel_workgroup_payload_t* out_payload) {
  *out_payload = (loom_kernel_workgroup_payload_t){0};
  int64_t lane_count = 1;
  if (loom_type_is_vector(logical_type)) {
    uint64_t static_element_count = 0;
    if (loom_type_rank(logical_type) != 1 ||
        !loom_type_static_element_count(logical_type, &static_element_count) ||
        static_element_count == 0 ||
        static_element_count > LOOM_DIM_MAX_STATIC_SIZE) {
      return false;
    }
    lane_count = (int64_t)static_element_count;
  } else if (!loom_type_is_scalar(logical_type)) {
    return false;
  }

  const loom_scalar_type_t logical_element_type =
      loom_type_element_type(logical_type);
  const int32_t logical_element_bit_count =
      loom_scalar_type_bitwidth(logical_element_type);
  loom_scalar_type_t storage_element_type = logical_element_type;
  int64_t element_byte_count = 0;
  if (logical_element_type == LOOM_SCALAR_TYPE_I1) {
    storage_element_type =
        context->predicate_storage_type
            ? context->predicate_storage_type(context->target_facts)
            : LOOM_SCALAR_TYPE_I8;
    element_byte_count = loom_scalar_type_bitwidth(storage_element_type) / 8;
  } else if (logical_element_bit_count > 0 &&
             (logical_element_bit_count % 8) == 0) {
    element_byte_count = logical_element_bit_count / 8;
  } else {
    return false;
  }

  int64_t scratch_element_count = 0;
  int64_t scratch_byte_count = 0;
  if (!iree_checked_mul_i64(lane_count, scratch_slot_count,
                            &scratch_element_count) ||
      scratch_element_count > LOOM_DIM_MAX_STATIC_SIZE ||
      !iree_checked_mul_i64(scratch_element_count, element_byte_count,
                            &scratch_byte_count)) {
    return false;
  }

  loom_type_t storage_type = logical_type;
  if (storage_element_type != logical_element_type) {
    storage_type.header = loom_type_make_header(
        loom_type_kind(logical_type), storage_element_type,
        loom_type_rank(logical_type), loom_type_flags(logical_type));
  }
  *out_payload = (loom_kernel_workgroup_payload_t){
      .logical_type = logical_type,
      .storage_type = storage_type,
      .storage_element_type = storage_element_type,
      .lane_count = lane_count,
      .element_byte_count = element_byte_count,
      .scratch_element_count = scratch_element_count,
      .scratch_byte_count = scratch_byte_count,
      .local_total = LOOM_VALUE_ID_INVALID,
      .scratch_view = LOOM_VALUE_ID_INVALID,
  };
  return true;
}

static iree_status_t loom_kernel_workgroup_build_index_constant(
    loom_kernel_workgroup_collective_state_t* state, int64_t value,
    loom_type_t type, loom_value_id_t* out_value) {
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(state->builder, loom_attr_i64(value), type,
                                state->location, &constant_op));
  *out_value = loom_index_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_collective_state_initialize_base(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_combining_kind_t kind, uint8_t math_flags,
    loom_kernel_workgroup_topology_t topology,
    loom_kernel_workgroup_collective_state_t* out_state) {
  *out_state = (loom_kernel_workgroup_collective_state_t){
      .context = context,
      .builder = &context->rewriter->builder,
      .location = op->location,
      .kind = kind,
      .math_flags = math_flags,
      .topology = topology,
      .index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      .zero_index = LOOM_VALUE_ID_INVALID,
      .subgroup_id = LOOM_VALUE_ID_INVALID,
      .lane_id = LOOM_VALUE_ID_INVALID,
      .lane_is_zero = LOOM_VALUE_ID_INVALID,
  };
  return loom_kernel_workgroup_build_index_constant(
      out_state, 0, out_state->index_type, &out_state->zero_index);
}

static iree_status_t loom_kernel_workgroup_collective_state_initialize(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_combining_kind_t kind, uint8_t math_flags,
    loom_kernel_workgroup_topology_t topology,
    loom_kernel_workgroup_collective_state_t* out_state) {
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_collective_state_initialize_base(
      context, op, kind, math_flags, topology, out_state));

  loom_op_t* subgroup_id_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_kernel_subgroup_id_build(out_state->builder, out_state->index_type,
                                    out_state->location, &subgroup_id_op));
  out_state->subgroup_id = loom_kernel_subgroup_id_result(subgroup_id_op);
  loom_op_t* lane_id_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_lane_id_build(
      out_state->builder, out_state->index_type, out_state->location,
      &lane_id_op));
  out_state->lane_id = loom_kernel_subgroup_lane_id_result(lane_id_op);
  loom_op_t* lane_is_zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      out_state->builder, LOOM_INDEX_CMP_PREDICATE_EQ, out_state->lane_id,
      out_state->zero_index, out_state->location, &lane_is_zero_op));
  out_state->lane_is_zero = loom_index_cmp_result(lane_is_zero_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_subgroup_reduce(
    loom_kernel_workgroup_collective_state_t* state, loom_value_id_t value,
    loom_type_t result_type, uint32_t cluster_size,
    loom_value_id_t* out_result) {
  loom_op_t* reduce_op = NULL;
  const loom_kernel_subgroup_reduce_build_flags_t build_flags =
      cluster_size != 0
          ? LOOM_KERNEL_SUBGROUP_REDUCE_BUILD_FLAG_HAS_CLUSTER_SIZE
          : 0;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_reduce_build(
      state->builder, build_flags, state->kind, state->math_flags, value,
      cluster_size, /*cluster_stride=*/0, result_type, state->location,
      &reduce_op));
  *out_result = loom_kernel_subgroup_reduce_result(reduce_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_subgroup_scan(
    loom_kernel_workgroup_collective_state_t* state, loom_value_id_t value,
    loom_type_t result_type, loom_kernel_subgroup_scan_mode_t mode,
    loom_kernel_subgroup_scan_direction_t direction, uint32_t cluster_size,
    loom_value_id_t* out_result) {
  loom_op_t* scan_op = NULL;
  const loom_kernel_subgroup_scan_build_flags_t build_flags =
      cluster_size != 0 ? LOOM_KERNEL_SUBGROUP_SCAN_BUILD_FLAG_HAS_CLUSTER_SIZE
                        : 0;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_scan_build(
      state->builder, build_flags, state->kind, state->math_flags, value,
      cluster_size, /*cluster_stride=*/0, mode, direction, result_type,
      state->location, &scan_op));
  *out_result = loom_kernel_subgroup_scan_result(scan_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_combine(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_payload_t* payload, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  if (loom_type_is_scalar(payload->logical_type)) {
    return loom_scalar_combining_build(
        state->builder, state->kind, /*fastmath_flags=*/0, lhs, rhs,
        payload->logical_type, state->location, out_result);
  }
  return loom_vector_combining_build(
      state->builder, state->kind, /*fastmath_flags=*/0, lhs, rhs,
      payload->logical_type, state->location, out_result);
}

static iree_status_t loom_kernel_workgroup_build_storage_value(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_payload_t* payload, loom_value_id_t value,
    loom_value_id_t* out_storage_value) {
  if (loom_type_equal(payload->logical_type, payload->storage_type)) {
    *out_storage_value = value;
    return iree_ok_status();
  }
  loom_op_t* extend_op = NULL;
  if (loom_type_is_scalar(payload->logical_type)) {
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        state->builder, value, payload->logical_type, payload->storage_type,
        state->location, &extend_op));
    *out_storage_value = loom_scalar_extui_result(extend_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_extui_build(
        state->builder, value, payload->logical_type, payload->storage_type,
        state->location, &extend_op));
    *out_storage_value = loom_vector_extui_result(extend_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_logical_value(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_payload_t* payload,
    loom_value_id_t storage_value, loom_value_id_t* out_logical_value) {
  if (loom_type_equal(payload->logical_type, payload->storage_type)) {
    *out_logical_value = storage_value;
    return iree_ok_status();
  }
  loom_op_t* truncate_op = NULL;
  if (loom_type_is_scalar(payload->logical_type)) {
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
        state->builder, storage_value, payload->storage_type,
        payload->logical_type, state->location, &truncate_op));
    *out_logical_value = loom_scalar_trunci_result(truncate_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_trunci_build(
        state->builder, storage_value, payload->storage_type,
        payload->logical_type, state->location, &truncate_op));
    *out_logical_value = loom_vector_trunci_result(truncate_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_scratch(
    loom_kernel_workgroup_collective_state_t* state,
    loom_kernel_workgroup_payload_t* payload) {
  loom_value_id_t byte_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, payload->scratch_byte_count,
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), &byte_count));
  loom_op_t* buffer_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      state->builder, /*build_flags=*/0, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
      LOOM_VALUE_ID_INVALID, payload->element_byte_count, byte_count,
      loom_type_buffer(), state->location, &buffer_op));
  loom_value_id_t zero_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, 0, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), &zero_offset));
  const loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, payload->storage_element_type,
      loom_dim_pack_static(payload->scratch_element_count), /*encoding_id=*/0);
  loom_op_t* view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      state->builder, loom_buffer_alloca_result(buffer_op), zero_offset,
      view_type, state->location, &view_op));
  payload->scratch_view = loom_buffer_view_result(view_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_element_index(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_payload_t* payload, loom_value_id_t slot,
    loom_value_id_t* out_element_index) {
  if (payload->lane_count == 1) {
    *out_element_index = slot;
    return iree_ok_status();
  }
  loom_value_id_t lane_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, payload->lane_count, state->index_type, &lane_count));
  loom_op_t* multiply_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_mul_build(state->builder, slot, lane_count,
                                            state->location, &multiply_op));
  *out_element_index = loom_index_mul_result(multiply_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_store(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_payload_t* payload, loom_value_id_t value,
    loom_value_id_t slot) {
  loom_value_id_t storage_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_storage_value(
      state, payload, value, &storage_value));
  loom_value_id_t element_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_element_index(
      state, payload, slot, &element_index));
  const int64_t static_indices[] = {INT64_MIN};
  loom_op_t* store_op = NULL;
  if (loom_type_is_scalar(payload->storage_type)) {
    return loom_view_store_build(
        state->builder, /*build_flags=*/0, /*instance_flags=*/0, storage_value,
        payload->scratch_view, &element_index, 1, static_indices,
        IREE_ARRAYSIZE(static_indices),
        /*cache_scope=*/0, /*cache_temporal=*/0, state->location, &store_op);
  }
  return loom_vector_store_build(
      state->builder, /*build_flags=*/0, /*instance_flags=*/0, storage_value,
      payload->scratch_view, &element_index, 1, static_indices,
      IREE_ARRAYSIZE(static_indices),
      /*cache_scope=*/0, /*cache_temporal=*/0, state->location, &store_op);
}

static iree_status_t loom_kernel_workgroup_build_load(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_payload_t* payload, loom_value_id_t slot,
    loom_value_id_t* out_value) {
  loom_value_id_t element_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_element_index(
      state, payload, slot, &element_index));
  const int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = NULL;
  loom_value_id_t storage_value = LOOM_VALUE_ID_INVALID;
  if (loom_type_is_scalar(payload->storage_type)) {
    IREE_RETURN_IF_ERROR(loom_view_load_build(
        state->builder, /*build_flags=*/0, /*instance_flags=*/0,
        payload->scratch_view, &element_index, 1, static_indices,
        IREE_ARRAYSIZE(static_indices),
        /*cache_scope=*/0, /*cache_temporal=*/0, payload->storage_type,
        state->location, &load_op));
    storage_value = loom_view_load_result(load_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_load_build(
        state->builder, /*build_flags=*/0, /*instance_flags=*/0,
        payload->scratch_view, &element_index, 1, static_indices,
        IREE_ARRAYSIZE(static_indices),
        /*cache_scope=*/0, /*cache_temporal=*/0, payload->storage_type,
        state->location, &load_op));
    storage_value = loom_vector_load_result(load_op);
  }
  return loom_kernel_workgroup_build_logical_value(state, payload,
                                                   storage_value, out_value);
}

static iree_status_t loom_kernel_workgroup_build_barrier(
    loom_kernel_workgroup_collective_state_t* state) {
  loom_op_t* barrier_op = NULL;
  return loom_kernel_barrier_build(
      state->builder, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
      LOOM_ATOMIC_SCOPE_WORKGROUP, LOOM_ATOMIC_ORDERING_ACQ_REL,
      state->location, &barrier_op);
}

static iree_status_t loom_kernel_workgroup_publish_local_totals(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_payload_t* payloads,
    iree_host_size_t payload_count) {
  loom_op_t* leader_if = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(state->builder, /*build_flags=*/0, state->lane_is_zero,
                        /*result_types=*/NULL, 0, /*tied_results=*/NULL, 0,
                        state->location, &leader_if));
  const loom_builder_ip_t saved_ip = loom_builder_enter_region(
      state->builder, leader_if, loom_scf_if_then_region(leader_if));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < payload_count && iree_status_is_ok(status);
       ++i) {
    status = loom_kernel_workgroup_build_store(
        state, &payloads[i], payloads[i].local_total, state->subgroup_id);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* yield_op = NULL;
    status = loom_scf_yield_build(state->builder, /*values=*/NULL, 0,
                                  state->location, &yield_op);
  }
  loom_builder_restore(state->builder, saved_ip);
  return status;
}

static iree_status_t loom_kernel_workgroup_build_hierarchical_lane_state(
    loom_kernel_workgroup_collective_state_t* state,
    loom_value_id_t* out_first_subgroup, loom_value_id_t* out_lane_is_used,
    loom_value_id_t* out_safe_slot) {
  loom_op_t* subgroup_is_zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      state->builder, LOOM_INDEX_CMP_PREDICATE_EQ, state->subgroup_id,
      state->zero_index, state->location, &subgroup_is_zero_op));
  loom_value_id_t subgroup_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, state->topology.subgroup_count, state->index_type,
      &subgroup_count));
  loom_op_t* lane_is_used_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      state->builder, LOOM_INDEX_CMP_PREDICATE_ULT, state->lane_id,
      subgroup_count, state->location, &lane_is_used_op));
  loom_op_t* safe_slot_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      state->builder, loom_index_cmp_result(lane_is_used_op), state->lane_id,
      state->zero_index, state->index_type, state->location, &safe_slot_op));
  *out_first_subgroup = loom_index_cmp_result(subgroup_is_zero_op);
  *out_lane_is_used = loom_index_cmp_result(lane_is_used_op);
  *out_safe_slot = loom_scf_select_result(safe_slot_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_hierarchical_reduce(
    loom_kernel_workgroup_collective_state_t* state,
    loom_kernel_workgroup_payload_t* payloads, iree_host_size_t payload_count,
    loom_value_id_t* out_results) {
  loom_value_id_t first_subgroup_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t lane_is_used = LOOM_VALUE_ID_INVALID;
  loom_value_id_t safe_slot = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_hierarchical_lane_state(
      state, &first_subgroup_condition, &lane_is_used, &safe_slot));
  (void)lane_is_used;
  // Partial slots remain immutable until the result publication barrier. A
  // separate result slot requires no implicit subgroup memory ordering between
  // the partial loads and the leader's store.
  loom_value_id_t result_slot = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, state->topology.subgroup_count, state->index_type, &result_slot));
  loom_op_t* fold_if = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      state->builder, /*build_flags=*/0, first_subgroup_condition,
      /*result_types=*/NULL, 0, /*tied_results=*/NULL, 0, state->location,
      &fold_if));
  const loom_builder_ip_t saved_ip = loom_builder_enter_region(
      state->builder, fold_if, loom_scf_if_then_region(fold_if));

  loom_value_id_t* reductions = NULL;
  iree_status_t status =
      iree_arena_allocate_array(state->context->arena, payload_count,
                                sizeof(*reductions), (void**)&reductions);
  for (iree_host_size_t i = 0; i < payload_count && iree_status_is_ok(status);
       ++i) {
    loom_value_id_t partial = LOOM_VALUE_ID_INVALID;
    status = loom_kernel_workgroup_build_load(state, &payloads[i], safe_slot,
                                              &partial);
    if (iree_status_is_ok(status)) {
      status = loom_kernel_workgroup_build_subgroup_reduce(
          state, partial, payloads[i].logical_type,
          state->topology.subgroup_count, &reductions[i]);
    }
  }

  loom_op_t* leader_if = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_scf_if_build(state->builder, /*build_flags=*/0,
                               state->lane_is_zero,
                               /*result_types=*/NULL, 0, /*tied_results=*/NULL,
                               0, state->location, &leader_if);
  }
  loom_builder_ip_t leader_saved_ip = {0};
  if (iree_status_is_ok(status)) {
    leader_saved_ip = loom_builder_enter_region(
        state->builder, leader_if, loom_scf_if_then_region(leader_if));
    for (iree_host_size_t i = 0; i < payload_count && iree_status_is_ok(status);
         ++i) {
      status = loom_kernel_workgroup_build_store(state, &payloads[i],
                                                 reductions[i], result_slot);
    }
    if (iree_status_is_ok(status)) {
      loom_op_t* leader_yield = NULL;
      status = loom_scf_yield_build(state->builder, /*values=*/NULL, 0,
                                    state->location, &leader_yield);
    }
    loom_builder_restore(state->builder, leader_saved_ip);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* fold_yield = NULL;
    status = loom_scf_yield_build(state->builder, /*values=*/NULL, 0,
                                  state->location, &fold_yield);
  }
  loom_builder_restore(state->builder, saved_ip);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_barrier(state));
  for (iree_host_size_t i = 0; i < payload_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_load(
        state, &payloads[i], result_slot, &out_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_linear_reduce(
    loom_kernel_workgroup_collective_state_t* state,
    loom_kernel_workgroup_payload_t* payloads, iree_host_size_t payload_count,
    loom_value_id_t* out_results) {
  for (iree_host_size_t i = 0; i < payload_count; ++i) {
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_load(
        state, &payloads[i], state->zero_index, &accumulator));
    for (uint32_t slot = 1; slot < state->topology.subgroup_count; ++slot) {
      loom_value_id_t slot_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
          state, slot, state->index_type, &slot_value));
      loom_value_id_t partial = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_load(
          state, &payloads[i], slot_value, &partial));
      IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_combine(
          state, &payloads[i], accumulator, partial, &accumulator));
    }
    out_results[i] = accumulator;
  }
  return iree_ok_status();
}

// Floating extrema have an empty-prefix seed, not an algebraic identity. A
// nonempty prefix must never absorb that seed from an empty adjacent prefix.
static bool loom_kernel_workgroup_scan_needs_presence(
    const loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_payload_t* payload) {
  double identity = 0.0;
  return loom_scalar_type_is_float(
             loom_type_element_type(payload->logical_type)) &&
         !loom_combining_kind_float_identity(state->kind, &identity);
}

static iree_status_t loom_kernel_workgroup_build_scan_presence(
    loom_kernel_workgroup_collective_state_t* state,
    loom_kernel_subgroup_scan_mode_t mode,
    loom_kernel_subgroup_scan_direction_t direction,
    loom_value_id_t* out_outer_nonempty, loom_value_id_t* out_local_nonempty) {
  *out_local_nonempty = LOOM_VALUE_ID_INVALID;
  loom_value_id_t edge_subgroup = state->zero_index;
  if (direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE) {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, state->topology.subgroup_count - 1, state->index_type,
        &edge_subgroup));
  }
  loom_op_t* outer_nonempty_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      state->builder, LOOM_INDEX_CMP_PREDICATE_NE, state->subgroup_id,
      edge_subgroup, state->location, &outer_nonempty_op));
  *out_outer_nonempty = loom_index_cmp_result(outer_nonempty_op);
  if (mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE) {
    return iree_ok_status();
  }

  loom_value_id_t edge_lane = state->zero_index;
  if (direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE) {
    loom_value_id_t full_last_lane = LOOM_VALUE_ID_INVALID;
    loom_value_id_t tail_last_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, state->topology.subgroup_size - 1, state->index_type,
        &full_last_lane));
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state,
        (state->topology.workitem_count - 1) % state->topology.subgroup_size,
        state->index_type, &tail_last_lane));
    loom_op_t* edge_lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_select_build(
        state->builder, *out_outer_nonempty, full_last_lane, tail_last_lane,
        state->index_type, state->location, &edge_lane_op));
    edge_lane = loom_scf_select_result(edge_lane_op);
  }
  loom_op_t* local_nonempty_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      state->builder, LOOM_INDEX_CMP_PREDICATE_NE, state->lane_id, edge_lane,
      state->location, &local_nonempty_op));
  *out_local_nonempty = loom_index_cmp_result(local_nonempty_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_hierarchical_scan(
    loom_kernel_workgroup_collective_state_t* state,
    loom_kernel_workgroup_payload_t* payload,
    loom_kernel_subgroup_scan_mode_t mode,
    loom_kernel_subgroup_scan_direction_t direction,
    loom_value_id_t local_prefix, loom_value_id_t* out_result) {
  loom_value_id_t first_subgroup_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t lane_is_used = LOOM_VALUE_ID_INVALID;
  loom_value_id_t safe_slot = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_hierarchical_lane_state(
      state, &first_subgroup_condition, &lane_is_used, &safe_slot));
  // Adjustments occupy the second half of scratch, leaving every subgroup's
  // partial immutable while the first subgroup reads and scans them.
  loom_value_id_t result_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, state->topology.subgroup_count, state->index_type, &result_base));
  loom_op_t* fold_if = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      state->builder, /*build_flags=*/0, first_subgroup_condition,
      /*result_types=*/NULL, 0, /*tied_results=*/NULL, 0, state->location,
      &fold_if));
  const loom_builder_ip_t saved_ip = loom_builder_enter_region(
      state->builder, fold_if, loom_scf_if_then_region(fold_if));
  loom_value_id_t partial = LOOM_VALUE_ID_INVALID;
  iree_status_t status =
      loom_kernel_workgroup_build_load(state, payload, safe_slot, &partial);
  loom_value_id_t adjustment = LOOM_VALUE_ID_INVALID;
  if (iree_status_is_ok(status)) {
    status = loom_kernel_workgroup_build_subgroup_scan(
        state, partial, payload->logical_type,
        LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE, direction,
        state->topology.subgroup_count, &adjustment);
  }
  loom_op_t* store_if = NULL;
  if (iree_status_is_ok(status)) {
    status =
        loom_scf_if_build(state->builder, /*build_flags=*/0, lane_is_used,
                          /*result_types=*/NULL, 0,
                          /*tied_results=*/NULL, 0, state->location, &store_if);
  }
  loom_builder_ip_t store_saved_ip = {0};
  if (iree_status_is_ok(status)) {
    store_saved_ip = loom_builder_enter_region(
        state->builder, store_if, loom_scf_if_then_region(store_if));
    loom_op_t* result_slot_op = NULL;
    status = loom_index_add_build(state->builder, result_base, state->lane_id,
                                  state->index_type, state->location,
                                  &result_slot_op);
    if (iree_status_is_ok(status)) {
      status = loom_kernel_workgroup_build_store(
          state, payload, adjustment, loom_index_add_result(result_slot_op));
    }
    if (iree_status_is_ok(status)) {
      loom_op_t* store_yield = NULL;
      status = loom_scf_yield_build(state->builder, /*values=*/NULL, 0,
                                    state->location, &store_yield);
    }
    loom_builder_restore(state->builder, store_saved_ip);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* yield_op = NULL;
    status = loom_scf_yield_build(state->builder, /*values=*/NULL, 0,
                                  state->location, &yield_op);
  }
  loom_builder_restore(state->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_barrier(state));
  loom_op_t* result_slot_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_add_build(
      state->builder, result_base, state->subgroup_id, state->index_type,
      state->location, &result_slot_op));
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_load(
      state, payload, loom_index_add_result(result_slot_op), &adjustment));
  const bool forward = direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_combine(
      state, payload, forward ? adjustment : local_prefix,
      forward ? local_prefix : adjustment, out_result));
  if (loom_kernel_workgroup_scan_needs_presence(state, payload)) {
    loom_value_id_t outer_nonempty = LOOM_VALUE_ID_INVALID;
    loom_value_id_t local_nonempty = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_scan_presence(
        state, mode, direction, &outer_nonempty, &local_nonempty));
    if (local_nonempty != LOOM_VALUE_ID_INVALID) {
      loom_op_t* local_select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(
          state->builder, local_nonempty, *out_result, adjustment,
          payload->logical_type, state->location, &local_select_op));
      *out_result = loom_scf_select_result(local_select_op);
    }
    loom_op_t* outer_select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_select_build(
        state->builder, outer_nonempty, *out_result, local_prefix,
        payload->logical_type, state->location, &outer_select_op));
    *out_result = loom_scf_select_result(outer_select_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_linear_scan(
    loom_kernel_workgroup_collective_state_t* state,
    loom_kernel_workgroup_payload_t* payload,
    loom_kernel_subgroup_scan_mode_t mode,
    loom_kernel_subgroup_scan_direction_t direction,
    loom_value_id_t local_prefix, loom_value_id_t* out_result) {
  loom_value_id_t accumulator = local_prefix;
  loom_value_id_t nonempty = LOOM_VALUE_ID_INVALID;
  if (mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE &&
      loom_kernel_workgroup_scan_needs_presence(state, payload)) {
    loom_value_id_t outer_nonempty = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_scan_presence(
        state, mode, direction, &outer_nonempty, &nonempty));
  }
  for (uint32_t slot = 0; slot < state->topology.subgroup_count; ++slot) {
    loom_value_id_t slot_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, slot, state->index_type, &slot_value));
    loom_op_t* include_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        state->builder,
        direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD
            ? LOOM_INDEX_CMP_PREDICATE_ULT
            : LOOM_INDEX_CMP_PREDICATE_UGT,
        slot_value, state->subgroup_id, state->location, &include_op));
    loom_value_id_t partial = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_kernel_workgroup_build_load(state, payload, slot_value, &partial));
    loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
    if (direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD) {
      IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_combine(
          state, payload, partial, accumulator, &combined));
    } else {
      IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_combine(
          state, payload, accumulator, partial, &combined));
    }
    if (nonempty != LOOM_VALUE_ID_INVALID) {
      loom_op_t* combine_select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(
          state->builder, nonempty, combined, partial, payload->logical_type,
          state->location, &combine_select_op));
      combined = loom_scf_select_result(combine_select_op);
      loom_op_t* nonempty_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_ori_build(
          state->builder, nonempty, loom_index_cmp_result(include_op),
          loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location,
          &nonempty_op));
      nonempty = loom_scalar_ori_result(nonempty_op);
    }
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_select_build(
        state->builder, loom_index_cmp_result(include_op), combined,
        accumulator, payload->logical_type, state->location, &select_op));
    accumulator = loom_scf_select_result(select_op);
  }
  *out_result = accumulator;
  return iree_ok_status();
}

// Logical groups use one slot with a single reader/writer in each relay phase.
// At most four components share the relay, independently of vector length.
enum { LOOM_KERNEL_WORKGROUP_LOGICAL_GROUP_SIZE = 4 };

typedef struct loom_kernel_workgroup_logical_geometry_t {
  // Number of logical publication groups, including the partial final group.
  uint32_t group_count;
  // Publication group containing this workitem.
  loom_value_id_t group;
  // Workitem's position within its publication group.
  loom_value_id_t position;
  // First workitem in each group initializes its scratch slot.
  loom_value_id_t first;
  // Last live workitem in each group owns the group total.
  loom_value_id_t leader;
  // Last valid scratch slot, used for statically bounded tree reads.
  loom_value_id_t last_group;
} loom_kernel_workgroup_logical_geometry_t;

static iree_status_t loom_kernel_workgroup_build_empty_prefix(
    loom_kernel_workgroup_collective_state_t* state, loom_type_t type,
    loom_value_id_t* out_value) {
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  loom_attribute_t seed = loom_attr_absent();
  if (loom_scalar_type_is_float(element_type)) {
    double value = 0.0;
    loom_combining_kind_float_scan_seed(state->kind, &value);
    seed = loom_attr_f64(value);
  } else {
    int64_t value = 0;
    loom_combining_kind_integer_identity(state->kind, element_type, &value);
    seed = loom_attr_i64(value);
  }
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      state->builder, seed, loom_type_scalar(element_type), state->location,
      &constant));
  *out_value = loom_scalar_constant_result(constant);
  if (loom_type_is_vector(type)) {
    loom_op_t* splat = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splat_build(
        state->builder, *out_value, type, state->location, &splat));
    *out_value = loom_vector_splat_result(splat);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_logical_geometry(
    loom_kernel_workgroup_collective_state_t* state,
    loom_kernel_workgroup_scan_direction_t direction,
    loom_kernel_workgroup_logical_geometry_t* out_geometry) {
  const uint32_t dimensions[] = {state->topology.workgroup_size.x,
                                 state->topology.workgroup_size.y,
                                 state->topology.workgroup_size.z};
  loom_value_id_t rank = state->zero_index;
  for (int dimension = 2; dimension >= 0; --dimension) {
    if (dimensions[dimension] == 1) {
      continue;
    }
    loom_value_id_t extent = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, dimensions[dimension], state->index_type, &extent));
    loom_op_t* scale = NULL;
    IREE_RETURN_IF_ERROR(loom_index_mul_build(state->builder, rank, extent,
                                              state->location, &scale));
    loom_op_t* coordinate = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_workitem_id_build(
        state->builder, (loom_kernel_dimension_t)dimension, state->index_type,
        state->location, &coordinate));
    loom_op_t* sum = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_add_build(state->builder, loom_index_mul_result(scale),
                             loom_kernel_workitem_id_result(coordinate),
                             state->index_type, state->location, &sum));
    rank = loom_index_add_result(sum);
  }
  loom_value_id_t last_workitem = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, state->topology.workitem_count - 1, state->index_type,
      &last_workitem));
  if (direction == LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_REVERSE) {
    loom_op_t* reverse = NULL;
    IREE_RETURN_IF_ERROR(loom_index_sub_build(state->builder, last_workitem,
                                              rank, state->index_type,
                                              state->location, &reverse));
    rank = loom_index_sub_result(reverse);
  }
  loom_value_id_t group_size = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, LOOM_KERNEL_WORKGROUP_LOGICAL_GROUP_SIZE, state->index_type,
      &group_size));
  loom_op_t* group = NULL;
  IREE_RETURN_IF_ERROR(loom_index_div_build(state->builder, rank, group_size,
                                            state->location, &group));
  loom_op_t* position = NULL;
  IREE_RETURN_IF_ERROR(loom_index_rem_build(state->builder, rank, group_size,
                                            state->location, &position));
  loom_op_t* first = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cmp_build(state->builder, LOOM_INDEX_CMP_PREDICATE_EQ,
                           loom_index_rem_result(position), state->zero_index,
                           state->location, &first));
  loom_value_id_t last_position = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, LOOM_KERNEL_WORKGROUP_LOGICAL_GROUP_SIZE - 1, state->index_type,
      &last_position));
  loom_op_t* full_leader = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cmp_build(state->builder, LOOM_INDEX_CMP_PREDICATE_EQ,
                           loom_index_rem_result(position), last_position,
                           state->location, &full_leader));
  loom_op_t* tail_leader = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cmp_build(state->builder, LOOM_INDEX_CMP_PREDICATE_EQ, rank,
                           last_workitem, state->location, &tail_leader));
  loom_op_t* leader = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_ori_build(
      state->builder, loom_index_cmp_result(full_leader),
      loom_index_cmp_result(tail_leader), loom_type_scalar(LOOM_SCALAR_TYPE_I1),
      state->location, &leader));
  const uint32_t group_count = (state->topology.workitem_count - 1) /
                                   LOOM_KERNEL_WORKGROUP_LOGICAL_GROUP_SIZE +
                               1;
  loom_value_id_t last_group = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, group_count - 1, state->index_type, &last_group));
  *out_geometry = (loom_kernel_workgroup_logical_geometry_t){
      .group_count = group_count,
      .group = loom_index_div_result(group),
      .position = loom_index_rem_result(position),
      .first = loom_index_cmp_result(first),
      .leader = loom_scalar_ori_result(leader),
      .last_group = last_group,
  };
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_build_logical_component(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_logical_geometry_t* geometry,
    const loom_kernel_workgroup_payload_t* payload, loom_value_id_t value,
    bool is_reduction, loom_kernel_workgroup_scan_mode_t mode,
    loom_value_id_t* out_result) {
  loom_builder_t* builder = state->builder;
  const loom_type_t type = payload->logical_type;
  loom_value_id_t local_prefix = value;
  if (mode == LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_workgroup_build_empty_prefix(state, type, &local_prefix));
  }
  loom_op_t* initial_publish = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(builder, /*build_flags=*/0,
                                         geometry->first, NULL, 0, NULL, 0,
                                         state->location, &initial_publish));
  loom_builder_ip_t saved = loom_builder_enter_region(
      builder, initial_publish, loom_scf_if_then_region(initial_publish));
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_store(state, payload, value,
                                                         geometry->group));
  loom_op_t* yield = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, NULL, 0, state->location, &yield));
  loom_builder_restore(builder, saved);
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_barrier(state));

  loom_value_id_t total = value;
  const loom_type_t relay_types[] = {type, type};
  for (uint32_t phase = 1; phase < LOOM_KERNEL_WORKGROUP_LOGICAL_GROUP_SIZE &&
                           phase < state->topology.workitem_count;
       ++phase) {
    loom_value_id_t phase_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, phase, state->index_type, &phase_index));
    loom_op_t* active = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_EQ, geometry->position, phase_index,
        state->location, &active));
    loom_op_t* relay = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_if_build(
        builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
        loom_index_cmp_result(active), relay_types, IREE_ARRAYSIZE(relay_types),
        NULL, 0, state->location, &relay));
    saved = loom_builder_enter_region(builder, relay,
                                      loom_scf_if_then_region(relay));
    // This phase has one reader/writer per slot. Its read precedes the write
    // in program order; the barrier publishes that new prefix to the next
    // owner.
    loom_value_id_t prior = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_load(
        state, payload, geometry->group, &prior));
    loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_combine(
        state, payload, prior, value, &combined));
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_store(
        state, payload, combined, geometry->group));
    const loom_value_id_t active_results[] = {
        combined,
        mode == LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE ? combined : prior};
    IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, active_results,
                                              IREE_ARRAYSIZE(active_results),
                                              state->location, &yield));
    loom_builder_enter_region(builder, relay, loom_scf_if_else_region(relay));
    const loom_value_id_t inactive_results[] = {total, local_prefix};
    IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, inactive_results,
                                              IREE_ARRAYSIZE(inactive_results),
                                              state->location, &yield));
    loom_builder_restore(builder, saved);
    total = loom_scf_if_results(relay).values[0];
    local_prefix = loom_scf_if_results(relay).values[1];
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_barrier(state));
  }

  uint32_t group_capacity = 1;
  while (group_capacity < geometry->group_count) {
    group_capacity <<= 1;
  }
  for (uint32_t stride = 1; stride < geometry->group_count; stride <<= 1) {
    loom_value_id_t stride_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, stride, state->index_type, &stride_index));
    loom_op_t* bit = NULL;
    IREE_RETURN_IF_ERROR(loom_index_andi_build(
        builder, geometry->group, stride_index, state->location, &bit));
    loom_op_t* right = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_NE, loom_index_andi_result(bit),
        state->zero_index, state->location, &right));
    loom_op_t* active = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
        builder, loom_index_cmp_result(right), geometry->leader,
        loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &active));
    loom_op_t* tree = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                          loom_scalar_andi_result(active), &type, 1, NULL, 0,
                          state->location, &tree));
    saved =
        loom_builder_enter_region(builder, tree, loom_scf_if_then_region(tree));
    loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, (group_capacity - 1) ^ (stride * 2 - 1), state->index_type,
        &mask));
    loom_op_t* begin = NULL;
    IREE_RETURN_IF_ERROR(loom_index_andi_build(builder, geometry->group, mask,
                                               state->location, &begin));
    loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, stride - 1, state->index_type, &offset));
    loom_op_t* endpoint = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_add_build(builder, loom_index_andi_result(begin), offset,
                             state->index_type, state->location, &endpoint));
    loom_op_t* bounded_endpoint = NULL;
    IREE_RETURN_IF_ERROR(loom_index_min_build(
        builder, loom_index_add_result(endpoint), geometry->last_group,
        state->location, &bounded_endpoint));
    // Left endpoints are outside the right half of every group interval.
    // Readers and writers therefore use disjoint slots throughout this stage.
    loom_value_id_t partial = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_load(
        state, payload, loom_index_min_result(bounded_endpoint), &partial));
    loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_combine(
        state, payload, partial, total, &combined));
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_store(
        state, payload, combined, geometry->group));
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(builder, &combined, 1, state->location, &yield));
    loom_builder_enter_region(builder, tree, loom_scf_if_else_region(tree));
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(builder, &total, 1, state->location, &yield));
    loom_builder_restore(builder, saved);
    total = loom_scf_if_results(tree).values[0];
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_barrier(state));
  }

  if (is_reduction) {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_load(
        state, payload, geometry->last_group, out_result));
  } else if (geometry->group_count == 1) {
    // The relay already produced every prefix. Its last barrier also ends
    // scratch reads, so no adjustment or additional reuse barrier is needed.
    *out_result = local_prefix;
    return iree_ok_status();
  } else {
    loom_op_t* has_preceding = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_NE, geometry->group,
        state->zero_index, state->location, &has_preceding));
    loom_op_t* adjustment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                          loom_index_cmp_result(has_preceding), &type, 1, NULL,
                          0, state->location, &adjustment));
    saved = loom_builder_enter_region(builder, adjustment,
                                      loom_scf_if_then_region(adjustment));
    loom_value_id_t unit = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, 1, state->index_type, &unit));
    loom_op_t* previous_group = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_sub_build(builder, geometry->group, unit, state->index_type,
                             state->location, &previous_group));
    loom_value_id_t prior = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_load(
        state, payload, loom_index_sub_result(previous_group), &prior));
    loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_combine(
        state, payload, prior, local_prefix, &combined));
    if (mode == LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE) {
      loom_op_t* nonempty = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(builder, geometry->first,
                                                 prior, combined, type,
                                                 state->location, &nonempty));
      combined = loom_scf_select_result(nonempty);
    }
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(builder, &combined, 1, state->location, &yield));
    loom_builder_enter_region(builder, adjustment,
                              loom_scf_if_else_region(adjustment));
    IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &local_prefix, 1,
                                              state->location, &yield));
    loom_builder_restore(builder, saved);
    *out_result = loom_scf_if_results(adjustment).values[0];
  }
  // All final reads finish before the next component or collective can reuse
  // this scratch allocation.
  return loom_kernel_workgroup_build_barrier(state);
}

static iree_status_t loom_kernel_workgroup_build_logical_payload(
    loom_kernel_workgroup_collective_state_t* state,
    const loom_kernel_workgroup_logical_geometry_t* geometry,
    loom_value_id_t source, bool is_reduction,
    loom_kernel_workgroup_scan_mode_t mode,
    loom_kernel_workgroup_payload_t* payload, loom_value_id_t* out_result) {
  loom_builder_t* builder = state->builder;
  const loom_type_t source_type =
      loom_module_value_type(state->context->module, source);
  if (state->topology.workitem_count == 1) {
    *out_result = source;
    return mode == LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE
               ? loom_kernel_workgroup_build_empty_prefix(state, source_type,
                                                          out_result)
               : iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_scratch(state, payload));
  if (loom_type_equal(source_type, payload->logical_type)) {
    return loom_kernel_workgroup_build_logical_component(
        state, geometry, payload, source, is_reduction, mode, out_result);
  }

  loom_value_id_t end = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, loom_type_dim_static_size_at(source_type, 0), state->index_type,
      &end));
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
      state, payload->lane_count, state->index_type, &step));
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, state->zero_index, end, step, &source, 1,
      /*result_types=*/NULL, NULL, 0, LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, /*unroll_schedule=*/0,
      state->location, &loop));
  const loom_builder_ip_t saved =
      loom_builder_enter_region(builder, loop, loom_scf_for_body(loop));
  const loom_value_id_t component =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t assembled =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  const int64_t dynamic_offset = INT64_MIN;
  loom_op_t* extract = NULL;
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  if (loom_type_is_scalar(payload->logical_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        builder, source, &component, 1, &dynamic_offset, 1,
        payload->logical_type, state->location, &extract));
    value = loom_vector_extract_result(extract);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        builder, source, &component, 1, &dynamic_offset, 1,
        payload->logical_type, state->location, &extract));
    value = loom_vector_slice_result(extract);
  }
  loom_value_id_t prefix = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_logical_component(
      state, geometry, payload, value, is_reduction, mode, &prefix));
  const loom_type_t element_type =
      loom_type_scalar(loom_type_element_type(source_type));
  for (int64_t lane = 0; lane < payload->lane_count; ++lane) {
    loom_value_id_t element = prefix;
    if (loom_type_is_vector(payload->logical_type)) {
      IREE_RETURN_IF_ERROR(
          loom_vector_extract_build(builder, prefix, NULL, 0, &lane, 1,
                                    element_type, state->location, &extract));
      element = loom_vector_extract_result(extract);
    }
    loom_value_id_t lane_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_index_constant(
        state, lane, state->index_type, &lane_index));
    loom_op_t* position = NULL;
    IREE_RETURN_IF_ERROR(loom_index_add_build(builder, component, lane_index,
                                              state->index_type,
                                              state->location, &position));
    const loom_value_id_t index = loom_index_add_result(position);
    loom_op_t* insert = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        builder, element, assembled, &index, 1, &dynamic_offset, 1, source_type,
        state->location, &insert));
    assembled = loom_vector_insert_result(insert);
  }
  loom_op_t* yield = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, &assembled, 1, state->location, &yield));
  loom_builder_restore(builder, saved);
  *out_result = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_replace_logical_collective(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_kernel_workgroup_topology_t topology,
    loom_target_legalizer_result_t* out_result) {
  const bool is_reduction = loom_kernel_workgroup_reduce_isa(op);
  loom_value_id_t scan_source = is_reduction
                                    ? LOOM_VALUE_ID_INVALID
                                    : loom_kernel_workgroup_scan_value(op);
  const loom_value_slice_t sources =
      is_reduction ? loom_kernel_workgroup_reduce_values(op)
                   : (loom_value_slice_t){.values = &scan_source, .count = 1};
  const loom_combining_kind_t kind = is_reduction
                                         ? loom_kernel_workgroup_reduce_kind(op)
                                         : loom_kernel_workgroup_scan_kind(op);
  const uint8_t math_flags = is_reduction
                                 ? loom_kernel_workgroup_reduce_math(op)
                                 : loom_kernel_workgroup_scan_math(op);
  const loom_kernel_workgroup_scan_mode_t mode =
      is_reduction ? LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE
                   : loom_kernel_workgroup_scan_mode(op);
  const loom_kernel_workgroup_scan_direction_t direction =
      is_reduction ? LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD
                   : loom_kernel_workgroup_scan_direction(op);
  const uint32_t group_count =
      (topology.workitem_count - 1) / LOOM_KERNEL_WORKGROUP_LOGICAL_GROUP_SIZE +
      1;
  loom_kernel_workgroup_payload_t* payloads = NULL;
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, sources.count, sizeof(*payloads), (void**)&payloads));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, sources.count,
                                                 sizeof(*replacements),
                                                 (void**)&replacements));
  for (iree_host_size_t i = 0; i < sources.count; ++i) {
    loom_type_t type =
        loom_module_value_type(context->module, sources.values[i]);
    if (!loom_kernel_workgroup_payload_initialize(context, type, 1,
                                                  &payloads[i])) {
      return iree_ok_status();
    }
    if (loom_type_is_vector(type)) {
      int64_t lane_count = 4;
      while (payloads[i].lane_count % lane_count != 0) {
        lane_count /= 2;
      }
      type = lane_count == 1
                 ? loom_type_scalar(loom_type_element_type(type))
                 : loom_type_shaped_1d(LOOM_TYPE_VECTOR,
                                       loom_type_element_type(type),
                                       loom_dim_pack_static(lane_count), 0);
    }
    if (!loom_kernel_workgroup_payload_initialize(context, type, group_count,
                                                  &payloads[i])) {
      return iree_ok_status();
    }
  }
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_kernel_workgroup_collective_state_t state;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_collective_state_initialize_base(
      context, op, kind, math_flags, topology, &state));
  loom_kernel_workgroup_logical_geometry_t geometry;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_logical_geometry(
      &state, direction, &geometry));
  for (iree_host_size_t i = 0; i < sources.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_logical_payload(
        &state, &geometry, sources.values[i], is_reduction, mode, &payloads[i],
        &replacements[i]));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, sources.count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, sources.count));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_replace_single_subgroup_reduce(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  const loom_value_slice_t values = loom_kernel_workgroup_reduce_values(op);
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, values.count,
                                                 sizeof(*replacements),
                                                 (void**)&replacements));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    loom_op_t* reduce_op = NULL;
    const loom_type_t type =
        loom_module_value_type(context->module, values.values[i]);
    IREE_RETURN_IF_ERROR(loom_kernel_subgroup_reduce_build(
        &rewriter->builder, /*build_flags=*/0,
        loom_kernel_workgroup_reduce_kind(op),
        loom_kernel_workgroup_reduce_math(op), values.values[i],
        /*cluster_size=*/0, /*cluster_stride=*/0, type, op->location,
        &reduce_op));
    replacements[i] = loom_kernel_subgroup_reduce_result(reduce_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, values.count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, values.count));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_kernel_workgroup_replace_single_subgroup_scan(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t source = loom_kernel_workgroup_scan_value(op);
  const loom_type_t type = loom_module_value_type(context->module, source);
  loom_op_t* scan_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_scan_build(
      &rewriter->builder, /*build_flags=*/0,
      loom_kernel_workgroup_scan_kind(op), loom_kernel_workgroup_scan_math(op),
      source, /*cluster_size=*/0, /*cluster_stride=*/0,
      loom_kernel_workgroup_subgroup_scan_mode(
          loom_kernel_workgroup_scan_mode(op)),
      loom_kernel_workgroup_subgroup_scan_direction(
          loom_kernel_workgroup_scan_direction(op)),
      type, op->location, &scan_op));
  const loom_value_id_t replacement = loom_kernel_subgroup_scan_result(scan_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

iree_status_t loom_kernel_target_legalize_workgroup_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
      context->policy != LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    return iree_ok_status();
  }
  loom_kernel_workgroup_topology_t topology = {0};
  if (!loom_kernel_workgroup_collective_topology(context, &topology)) {
    return iree_ok_status();
  }
  const loom_combining_kind_t kind = loom_kernel_workgroup_reduce_kind(op);
  const uint8_t math_flags = loom_kernel_workgroup_reduce_math(op);
  if (!loom_kernel_workgroup_collective_math_is_supported(kind, math_flags)) {
    return iree_ok_status();
  }
  if (topology.subgroup_size == 0 ||
      topology.partition < LOOM_TARGET_SUBGROUP_PARTITION_DENSE) {
    return loom_kernel_workgroup_replace_logical_collective(
        context, op, topology, out_result);
  }
  if (topology.subgroup_count == 1) {
    return loom_kernel_workgroup_replace_single_subgroup_reduce(context, op,
                                                                out_result);
  }
  const loom_value_slice_t values = loom_kernel_workgroup_reduce_values(op);
  loom_kernel_workgroup_payload_t* payloads = NULL;
  loom_value_id_t* replacements = NULL;
  const uint32_t scratch_slot_count =
      topology.subgroup_count +
      (topology.subgroup_count <= topology.subgroup_size ? 1u : 0u);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, values.count, sizeof(*payloads), (void**)&payloads));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, values.count,
                                                 sizeof(*replacements),
                                                 (void**)&replacements));
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    if (!loom_kernel_workgroup_payload_initialize(
            context, loom_module_value_type(context->module, values.values[i]),
            scratch_slot_count, &payloads[i])) {
      return iree_ok_status();
    }
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_kernel_workgroup_collective_state_t state;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_collective_state_initialize(
      context, op, kind, math_flags, topology, &state));
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_subgroup_reduce(
        &state, values.values[i], payloads[i].logical_type,
        /*cluster_size=*/0, &payloads[i].local_total));
    IREE_RETURN_IF_ERROR(
        loom_kernel_workgroup_build_scratch(&state, &payloads[i]));
  }
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_publish_local_totals(
      &state, payloads, values.count));
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_barrier(&state));
  if (topology.subgroup_count <= topology.subgroup_size) {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_hierarchical_reduce(
        &state, payloads, values.count, replacements));
  } else {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_linear_reduce(
        &state, payloads, values.count, replacements));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, values.count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, values.count));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

iree_status_t loom_kernel_target_legalize_workgroup_scan(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
      context->policy != LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    return iree_ok_status();
  }
  loom_kernel_workgroup_topology_t topology = {0};
  if (!loom_kernel_workgroup_collective_topology(context, &topology)) {
    return iree_ok_status();
  }
  const loom_combining_kind_t kind = loom_kernel_workgroup_scan_kind(op);
  const uint8_t math_flags = loom_kernel_workgroup_scan_math(op);
  if (!loom_kernel_workgroup_collective_math_is_supported(kind, math_flags)) {
    return iree_ok_status();
  }
  if (topology.subgroup_size == 0 ||
      topology.partition < LOOM_TARGET_SUBGROUP_PARTITION_LINEAR) {
    return loom_kernel_workgroup_replace_logical_collective(
        context, op, topology, out_result);
  }
  if (topology.subgroup_count == 1) {
    return loom_kernel_workgroup_replace_single_subgroup_scan(context, op,
                                                              out_result);
  }
  const loom_value_id_t source = loom_kernel_workgroup_scan_value(op);
  loom_kernel_workgroup_payload_t payload;
  const uint32_t scratch_slot_count =
      topology.subgroup_count *
      (topology.subgroup_count <= topology.subgroup_size ? 2u : 1u);
  if (!loom_kernel_workgroup_payload_initialize(
          context, loom_module_value_type(context->module, source),
          scratch_slot_count, &payload)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_kernel_workgroup_collective_state_t state;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_collective_state_initialize(
      context, op, kind, math_flags, topology, &state));
  loom_value_id_t local_prefix = LOOM_VALUE_ID_INVALID;
  const loom_kernel_subgroup_scan_mode_t mode =
      loom_kernel_workgroup_subgroup_scan_mode(
          loom_kernel_workgroup_scan_mode(op));
  const loom_kernel_subgroup_scan_direction_t direction =
      loom_kernel_workgroup_subgroup_scan_direction(
          loom_kernel_workgroup_scan_direction(op));
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_subgroup_scan(
      &state, source, payload.logical_type, mode, direction, /*cluster_size=*/0,
      &local_prefix));
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_subgroup_reduce(
      &state, source, payload.logical_type, /*cluster_size=*/0,
      &payload.local_total));
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_scratch(&state, &payload));
  IREE_RETURN_IF_ERROR(
      loom_kernel_workgroup_publish_local_totals(&state, &payload, 1));
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_build_barrier(&state));

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (topology.subgroup_count <= topology.subgroup_size) {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_hierarchical_scan(
        &state, &payload, mode, direction, local_prefix, &replacement));
  } else {
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_linear_scan(
        &state, &payload, mode, direction, local_prefix, &replacement));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}
