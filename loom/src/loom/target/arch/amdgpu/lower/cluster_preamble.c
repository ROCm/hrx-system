// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/cluster_preamble.h"

#include "loom/ops/kernel/launch_config.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#define LOOM_AMDGPU_EXTENDED_DISPATCH_PACKET_CLUSTER_COUNT_X_OFFSET 12u
#define LOOM_AMDGPU_EXTENDED_DISPATCH_PACKET_CLUSTER_COUNT_YZ_OFFSET 16u

#define LOOM_AMDGPU_CLUSTER_WORKGROUP_COORDINATE_BITS 4u
#define LOOM_AMDGPU_CLUSTER_WORKGROUP_COORDINATE_MASK 0xFu
#define LOOM_AMDGPU_CLUSTER_ID_Y_MASK 0xFFFFu
#define LOOM_AMDGPU_CLUSTER_ID_Z_SHIFT 16u

typedef struct loom_amdgpu_cluster_preamble_state_t {
  // True after target and launch facts have initialized this state.
  bool initialized;
  // True when workgroup identity arrives in architected TTMP launch state.
  bool uses_architected_workgroup_ids;
  // True when launch counts arrive in an extended clustered-dispatch packet.
  bool uses_clustered_dispatch;
  // Static number of workgroups in each cluster dimension.
  loom_target_workgroup_cluster_size_t cluster_size;
  // Raw packed cluster-local workgroup state imported from TTMP6.
  loom_value_id_t cluster_workgroup_info;
  // Raw packed y/z architected coordinates imported from TTMP7.
  loom_value_id_t cluster_id_yz;
  // Raw x architected coordinate imported from TTMP9.
  loom_value_id_t cluster_id_x;
  // Materialized global workgroup coordinates.
  loom_value_id_t workgroup_ids[LOOM_KERNEL_DIMENSION_COUNT_];
  // Materialized cluster coordinates.
  loom_value_id_t cluster_ids[LOOM_KERNEL_DIMENSION_COUNT_];
  // Materialized workgroup coordinates within the cluster.
  loom_value_id_t cluster_workgroup_ids[LOOM_KERNEL_DIMENSION_COUNT_];
  // Materialized flat workgroup coordinate within the cluster.
  loom_value_id_t cluster_workgroup_flat_id;
  // Packed dynamic y/z cluster counts loaded from the dispatch packet.
  loom_value_id_t packed_cluster_count_yz;
  // Materialized cluster counts loaded from the dispatch packet.
  loom_value_id_t cluster_counts[LOOM_KERNEL_DIMENSION_COUNT_];
  // Source operations whose results require cluster identity materialization.
  loom_amdgpu_cluster_preamble_demands_t demands;
} loom_amdgpu_cluster_preamble_state_t;

static const uint8_t loom_amdgpu_cluster_preamble_state_key = 0;

uint32_t loom_amdgpu_cluster_preamble_size_dimension(
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

bool loom_amdgpu_cluster_preamble_required_nontrivial_size(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_value_fact_table_t* fact_table,
    loom_target_workgroup_cluster_size_t* out_size) {
  *out_size = (loom_target_workgroup_cluster_size_t){0};
  return loom_kernel_def_static_workgroup_cluster_size_from_facts(
             module, function_op, fact_table, out_size) &&
         (out_size->x != 1 || out_size->y != 1 || out_size->z != 1);
}

bool loom_amdgpu_cluster_preamble_target_uses_architected_workgroup_ids(
    const loom_amdgpu_target_facts_t* target_facts) {
  IREE_ASSERT(target_facts != NULL,
              "AMDGPU cluster preamble requires AMDGPU target facts");
  return loom_amdgpu_processor_properties_have_flags(
      target_facts->properties.processor,
      LOOM_AMDGPU_PROCESSOR_INFO_FLAG_ARCHITECTED_WORKGROUP_IDS);
}

bool loom_amdgpu_cluster_preamble_target_supports_cluster_launch_state(
    const loom_amdgpu_target_facts_t* target_facts) {
  IREE_ASSERT(target_facts != NULL,
              "AMDGPU cluster preamble requires AMDGPU target facts");
  return loom_amdgpu_processor_properties_have_flags(
      target_facts->properties.processor,
      LOOM_AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE);
}

static iree_status_t loom_amdgpu_cluster_preamble_state(
    loom_low_lower_context_t* context,
    loom_amdgpu_cluster_preamble_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_cluster_preamble_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_cluster_preamble_state_key, sizeof(*state),
      (void**)&state));
  if (!state->initialized) {
    const bool has_nontrivial_cluster =
        loom_amdgpu_cluster_preamble_required_nontrivial_size(
            loom_low_lower_context_module(context),
            loom_low_lower_context_source_function(context).op,
            loom_low_lower_context_fact_table(context), &state->cluster_size);
    if (!has_nontrivial_cluster) {
      state->cluster_size = (loom_target_workgroup_cluster_size_t){
          .x = 1,
          .y = 1,
          .z = 1,
      };
    }
    // Architected workgroup identity uses TTMP9/TTMP7 even for ordinary
    // dispatches with a trivial 1x1x1 source cluster.
    const loom_amdgpu_target_facts_t* target_facts =
        loom_amdgpu_target_facts_cast(
            loom_low_lower_context_target_facts(context));
    state->uses_architected_workgroup_ids =
        loom_amdgpu_cluster_preamble_target_uses_architected_workgroup_ids(
            target_facts);
    const bool supports_cluster_launch_state =
        loom_amdgpu_cluster_preamble_target_supports_cluster_launch_state(
            target_facts);
    state->uses_clustered_dispatch =
        has_nontrivial_cluster && supports_cluster_launch_state;
    state->cluster_workgroup_info = LOOM_VALUE_ID_INVALID;
    state->cluster_id_yz = LOOM_VALUE_ID_INVALID;
    state->cluster_id_x = LOOM_VALUE_ID_INVALID;
    state->cluster_workgroup_flat_id = LOOM_VALUE_ID_INVALID;
    state->packed_cluster_count_yz = LOOM_VALUE_ID_INVALID;
    for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
      state->workgroup_ids[i] = LOOM_VALUE_ID_INVALID;
      state->cluster_ids[i] = LOOM_VALUE_ID_INVALID;
      state->cluster_workgroup_ids[i] = LOOM_VALUE_ID_INVALID;
      state->cluster_counts[i] = LOOM_VALUE_ID_INVALID;
    }
    state->initialized = true;
  }
  *out_state = state;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_cluster_preamble_uses_architected_workgroup_ids(
    loom_low_lower_context_t* context,
    bool* out_uses_architected_workgroup_ids) {
  *out_uses_architected_workgroup_ids = false;
  loom_amdgpu_cluster_preamble_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_state(context, &state));
  *out_uses_architected_workgroup_ids = state->uses_architected_workgroup_ids;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_cluster_preamble_uses_clustered_dispatch(
    loom_low_lower_context_t* context, bool* out_uses_clustered_dispatch) {
  *out_uses_clustered_dispatch = false;
  loom_amdgpu_cluster_preamble_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_state(context, &state));
  *out_uses_clustered_dispatch = state->uses_clustered_dispatch;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_cluster_preamble_emit_sgpr_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  const iree_string_view_t source =
      loom_amdgpu_hal_kernel_abi_source_name(source_kind);
  if (iree_string_view_is_empty(source)) {
    IREE_ASSERT_UNREACHABLE(
        "known AMDGPU architected launch-state SGPR source");
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

iree_status_t loom_amdgpu_cluster_preamble_emit_live_ins(
    loom_low_lower_context_t* context,
    const loom_amdgpu_cluster_preamble_demands_t* demands) {
  loom_amdgpu_cluster_preamble_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_state(context, &state));
  if (!state->uses_architected_workgroup_ids) {
    return iree_ok_status();
  }
  state->demands = *demands;

  const loom_op_t* cluster_workgroup_info_op = NULL;
  const loom_op_t* cluster_id_yz_op = NULL;
  const loom_op_t* cluster_id_x_op = NULL;
  loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flags_t
      cluster_workgroup_info_launch_mask =
          LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X;
  loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flags_t
      cluster_id_yz_launch_mask =
          LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X;
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    const uint32_t cluster_size = loom_amdgpu_cluster_preamble_size_dimension(
        &state->cluster_size, (loom_kernel_dimension_t)i);
    if (cluster_workgroup_info_op == NULL &&
        (demands->cluster_workgroup_id_ops[i] != NULL ||
         (demands->workgroup_id_ops[i] != NULL && cluster_size > 1))) {
      cluster_workgroup_info_op = demands->cluster_workgroup_id_ops[i] != NULL
                                      ? demands->cluster_workgroup_id_ops[i]
                                      : demands->workgroup_id_ops[i];
    }
    if (demands->cluster_workgroup_id_ops[i] != NULL ||
        (demands->workgroup_id_ops[i] != NULL && cluster_size > 1)) {
      cluster_workgroup_info_launch_mask |=
          LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X << i;
    }
    const loom_op_t* cluster_coordinate_op = demands->cluster_id_ops[i] != NULL
                                                 ? demands->cluster_id_ops[i]
                                                 : demands->workgroup_id_ops[i];
    if (i == LOOM_KERNEL_DIMENSION_X && cluster_id_x_op == NULL) {
      cluster_id_x_op = cluster_coordinate_op;
    } else if (i != LOOM_KERNEL_DIMENSION_X && cluster_coordinate_op != NULL) {
      if (cluster_id_yz_op == NULL) {
        cluster_id_yz_op = cluster_coordinate_op;
      }
      cluster_id_yz_launch_mask |=
          LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X << i;
    }
  }

  if (cluster_workgroup_info_op != NULL) {
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN;
    switch (cluster_workgroup_info_launch_mask) {
      case LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X:
        source_kind =
            LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_WORKGROUP_INFO_X;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
          LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Y:
        source_kind =
            LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_WORKGROUP_INFO_XY;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
          LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Z:
        source_kind =
            LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_WORKGROUP_INFO_XZ;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_KNOWN_FLAGS:
        source_kind = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_WORKGROUP_INFO;
        break;
      default:
        IREE_ASSERT_UNREACHABLE(
            "cluster workgroup-info must serve at least one coordinate");
        IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_emit_sgpr_live_in(
        context, cluster_workgroup_info_op, source_kind,
        &state->cluster_workgroup_info));
  }
  if (cluster_id_yz_op != NULL) {
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN;
    switch (cluster_id_yz_launch_mask) {
      case LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
          LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Y:
        source_kind = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_ID_Y;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
          LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Z:
        source_kind = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_ID_Z;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_KNOWN_FLAGS:
        source_kind = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_ID_YZ;
        break;
      default:
        IREE_ASSERT_UNREACHABLE(
            "cluster y/z live-in must serve at least one coordinate");
        IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_emit_sgpr_live_in(
        context, cluster_id_yz_op, source_kind, &state->cluster_id_yz));
  }
  if (cluster_id_x_op != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_emit_sgpr_live_in(
        context, cluster_id_x_op,
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_ID_X, &state->cluster_id_x));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_cluster_preamble_emit_bfe_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t packed_value, uint32_t control, loom_type_t result_type,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), control, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {packed_value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_BFE_U32_LIT, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &result_type, 1, &low_op));
  *out_low_value_id = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_cluster_preamble_materialize_workgroup_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension,
    loom_amdgpu_cluster_preamble_state_t* state) {
  IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  IREE_ASSERT(state->uses_architected_workgroup_ids);
  if (state->cluster_workgroup_ids[dimension] != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (state->cluster_workgroup_info == LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE("emitted cluster workgroup-info live-in");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return loom_amdgpu_emit_sgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
          state->cluster_workgroup_info,
          LOOM_AMDGPU_CLUSTER_WORKGROUP_COORDINATE_MASK, sgpr_type,
          &state->cluster_workgroup_ids[dimension]);
    case LOOM_KERNEL_DIMENSION_Y:
    case LOOM_KERNEL_DIMENSION_Z: {
      // The selected GFX12 BFE literal layout places the four-bit width in bits
      // 16..22 and the offset in bits 0..4.
      const uint32_t offset =
          (uint32_t)dimension * LOOM_AMDGPU_CLUSTER_WORKGROUP_COORDINATE_BITS;
      const uint32_t control =
          (LOOM_AMDGPU_CLUSTER_WORKGROUP_COORDINATE_BITS << 16u) | offset;
      return loom_amdgpu_cluster_preamble_emit_bfe_u32(
          context, source_op, state->cluster_workgroup_info, control, sgpr_type,
          &state->cluster_workgroup_ids[dimension]);
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown cluster workgroup-id dimension");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_cluster_preamble_materialize_cluster_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension,
    loom_amdgpu_cluster_preamble_state_t* state) {
  IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  IREE_ASSERT(state->uses_architected_workgroup_ids);
  if (state->cluster_ids[dimension] != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (dimension == LOOM_KERNEL_DIMENSION_X) {
    if (state->cluster_id_x == LOOM_VALUE_ID_INVALID) {
      IREE_ASSERT_UNREACHABLE("emitted cluster-id x live-in");
      IREE_BUILTIN_UNREACHABLE();
    }
    state->cluster_ids[dimension] = state->cluster_id_x;
    return iree_ok_status();
  }
  if (state->cluster_id_yz == LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE("emitted cluster-id y/z live-in");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  if (dimension == LOOM_KERNEL_DIMENSION_Y) {
    return loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
        state->cluster_id_yz, LOOM_AMDGPU_CLUSTER_ID_Y_MASK, sgpr_type,
        &state->cluster_ids[dimension]);
  }
  return loom_amdgpu_emit_sgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B32,
      state->cluster_id_yz, LOOM_AMDGPU_CLUSTER_ID_Z_SHIFT, sgpr_type,
      &state->cluster_ids[dimension]);
}

static iree_status_t
loom_amdgpu_cluster_preamble_materialize_global_workgroup_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension,
    loom_amdgpu_cluster_preamble_state_t* state) {
  IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  IREE_ASSERT(state->uses_architected_workgroup_ids);
  if (state->workgroup_ids[dimension] != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_materialize_cluster_id(
      context, source_op, dimension, state));
  const uint32_t cluster_size = loom_amdgpu_cluster_preamble_size_dimension(
      &state->cluster_size, dimension);
  IREE_ASSERT_NE(cluster_size, 0u);
  if (cluster_size == 1) {
    state->workgroup_ids[dimension] = state->cluster_ids[dimension];
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_materialize_workgroup_id(
      context, source_op, dimension, state));

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_sgpr_scaled_add_u32(
      context, source_op, state->cluster_ids[dimension], cluster_size,
      state->cluster_workgroup_ids[dimension], sgpr_type,
      &state->workgroup_ids[dimension]);
}

static iree_status_t loom_amdgpu_cluster_preamble_materialize_flat_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_cluster_preamble_state_t* state) {
  IREE_ASSERT(state->uses_architected_workgroup_ids);
  if (state->cluster_workgroup_flat_id != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_GETREG_B32_CLUSTER_WORKGROUP_FLAT_ID,
      /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(NULL, 0), &sgpr_type, 1, &low_op));
  state->cluster_workgroup_flat_id =
      loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_cluster_preamble_bind_identity_queries(
    loom_low_lower_context_t* context,
    const loom_amdgpu_cluster_preamble_state_t* state) {
  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    const loom_op_t* source_op = selected_plan.source_op;
    loom_value_id_t source_result = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    switch (source_op->kind) {
      case LOOM_OP_KERNEL_WORKGROUP_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workgroup_id_dimension(source_op);
        source_result = loom_kernel_workgroup_id_result(source_op);
        low_result = state->workgroup_ids[dimension];
        break;
      }
      case LOOM_OP_KERNEL_CLUSTER_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_cluster_id_dimension(source_op);
        source_result = loom_kernel_cluster_id_result(source_op);
        low_result = state->cluster_ids[dimension];
        break;
      }
      case LOOM_OP_KERNEL_CLUSTER_WORKGROUP_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_cluster_workgroup_id_dimension(source_op);
        source_result = loom_kernel_cluster_workgroup_id_result(source_op);
        low_result = state->cluster_workgroup_ids[dimension];
        break;
      }
      case LOOM_OP_KERNEL_CLUSTER_WORKGROUP_FLAT_ID:
        source_result = loom_kernel_cluster_workgroup_flat_id_result(source_op);
        low_result = state->cluster_workgroup_flat_id;
        break;
      default:
        continue;
    }
    if (low_result == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, source_result, low_result));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_cluster_preamble_emit_entry_setup(
    loom_low_lower_context_t* context) {
  loom_amdgpu_cluster_preamble_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_state(context, &state));
  if (!state->uses_architected_workgroup_ids) {
    return iree_ok_status();
  }

  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    const loom_kernel_dimension_t dimension = (loom_kernel_dimension_t)i;
    if (state->demands.workgroup_id_ops[i] != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_cluster_preamble_materialize_global_workgroup_id(
              context, state->demands.workgroup_id_ops[i], dimension, state));
    }
    if (state->demands.cluster_id_ops[i] != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_materialize_cluster_id(
          context, state->demands.cluster_id_ops[i], dimension, state));
    }
    if (state->demands.cluster_workgroup_id_ops[i] != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_cluster_preamble_materialize_workgroup_id(
              context, state->demands.cluster_workgroup_id_ops[i], dimension,
              state));
    }
  }
  if (state->demands.cluster_workgroup_flat_id_op != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_materialize_flat_id(
        context, state->demands.cluster_workgroup_flat_id_op, state));
  }
  return loom_amdgpu_cluster_preamble_bind_identity_queries(context, state);
}

iree_status_t loom_amdgpu_cluster_preamble_lookup_workgroup_id(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  loom_amdgpu_cluster_preamble_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_state(context, &state));
  IREE_ASSERT(state->uses_architected_workgroup_ids);
  if (state->workgroup_ids[dimension] == LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_UNREACHABLE("materialized architected workgroup coordinate");
    IREE_BUILTIN_UNREACHABLE();
  }
  *out_low_value_id = state->workgroup_ids[dimension];
  return iree_ok_status();
}

iree_status_t loom_amdgpu_cluster_preamble_lookup_size(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension,
    uint32_t* out_size) {
  *out_size = 0;
  IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  loom_amdgpu_cluster_preamble_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_state(context, &state));
  IREE_ASSERT(state->uses_architected_workgroup_ids);
  *out_size = loom_amdgpu_cluster_preamble_size_dimension(&state->cluster_size,
                                                          dimension);
  IREE_ASSERT_NE(*out_size, 0u);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_cluster_preamble_emit_dispatch_dword(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, uint32_t offset, loom_type_t result_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), offset, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {dispatch_ptr};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
      operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_cluster_preamble_emit_cluster_count(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, loom_kernel_dimension_t dimension,
    loom_type_t result_type, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  loom_amdgpu_cluster_preamble_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_state(context, &state));
  IREE_ASSERT(state->uses_clustered_dispatch);
  if (state->cluster_counts[dimension] != LOOM_VALUE_ID_INVALID) {
    *out_low_value_id = state->cluster_counts[dimension];
    return iree_ok_status();
  }

  if (dimension == LOOM_KERNEL_DIMENSION_X) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_emit_dispatch_dword(
        context, source_op, dispatch_ptr,
        LOOM_AMDGPU_EXTENDED_DISPATCH_PACKET_CLUSTER_COUNT_X_OFFSET,
        result_type, &state->cluster_counts[dimension]));
  } else {
    if (state->packed_cluster_count_yz == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_emit_dispatch_dword(
          context, source_op, dispatch_ptr,
          LOOM_AMDGPU_EXTENDED_DISPATCH_PACKET_CLUSTER_COUNT_YZ_OFFSET,
          result_type, &state->packed_cluster_count_yz));
    }
    if (dimension == LOOM_KERNEL_DIMENSION_Y) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
          state->packed_cluster_count_yz, UINT16_MAX, result_type,
          &state->cluster_counts[dimension]));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B32,
          state->packed_cluster_count_yz, 16u, result_type,
          &state->cluster_counts[dimension]));
    }
  }
  *out_low_value_id = state->cluster_counts[dimension];
  return iree_ok_status();
}

iree_status_t loom_amdgpu_cluster_preamble_emit_workgroup_count(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, loom_kernel_dimension_t dimension,
    loom_type_t result_type, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cluster_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_emit_cluster_count(
      context, source_op, dispatch_ptr, dimension, result_type,
      &cluster_count));
  uint32_t cluster_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cluster_preamble_lookup_size(
      context, dimension, &cluster_size));
  return loom_amdgpu_emit_sgpr_scale_u32(context, source_op, cluster_count,
                                         cluster_size, result_type,
                                         out_low_value_id);
}
