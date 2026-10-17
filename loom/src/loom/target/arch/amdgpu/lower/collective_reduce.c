// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/codegen/low/lower/participation.h"
#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/collective_combine.h"
#include "loom/target/arch/amdgpu/lower/collective_payload.h"
#include "loom/target/arch/amdgpu/lower/collective_tree.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/workgroup.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static uint32_t loom_amdgpu_subgroup_u32_log2(uint32_t value) {
  uint32_t log2 = 0;
  while (value > 1) {
    value >>= 1;
    ++log2;
  }
  return log2;
}

static bool loom_amdgpu_subgroup_reduce_has_cluster_attrs(const loom_op_t* op) {
  return loom_kernel_subgroup_reduce_has_cluster_size(op) ||
         loom_kernel_subgroup_reduce_has_cluster_stride(op);
}

static bool loom_amdgpu_subgroup_cluster_size_is_supported(
    int64_t cluster_size, uint32_t wavefront_size) {
  return cluster_size > 0 && cluster_size <= (int64_t)wavefront_size &&
         loom_amdgpu_u32_is_power_of_two((uint32_t)cluster_size);
}

typedef uint8_t loom_amdgpu_subgroup_reduce_shape_flags_t;

enum loom_amdgpu_subgroup_reduce_shape_flag_bits_e {
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_CLUSTERED = 1u << 0,
};

typedef enum loom_amdgpu_subgroup_reduce_shape_failure_e {
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE = 0,
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_STRIDE = 1,
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_SIZE = 2,
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_CLUSTER_MULTIPLE = 3,
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE = 4,
} loom_amdgpu_subgroup_reduce_shape_failure_t;

typedef struct loom_amdgpu_subgroup_reduce_shape_t {
  // Semantic lanes participating in the selected subgroup reduce.
  uint32_t active_lane_count;
  // Shape properties derived from subgroup-reduce attributes and launch facts.
  loom_amdgpu_subgroup_reduce_shape_flags_t flags;
} loom_amdgpu_subgroup_reduce_shape_t;

static bool loom_amdgpu_subgroup_reduce_resolve_shape(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, const loom_op_t* op,
    uint32_t wavefront_size, loom_amdgpu_subgroup_reduce_shape_t* out_shape,
    loom_amdgpu_subgroup_reduce_shape_failure_t* out_failure) {
  *out_shape = (loom_amdgpu_subgroup_reduce_shape_t){0};
  *out_failure = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE;

  const bool has_cluster_attrs =
      loom_amdgpu_subgroup_reduce_has_cluster_attrs(op);
  int64_t cluster_size = 0;
  if (has_cluster_attrs) {
    if (loom_kernel_subgroup_reduce_has_cluster_stride(op)) {
      *out_failure = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_STRIDE;
      return false;
    }
    if (!loom_kernel_subgroup_reduce_has_cluster_size(op)) {
      *out_failure = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_SIZE;
      return false;
    }

    cluster_size = loom_kernel_subgroup_reduce_cluster_size(op);
    if (!loom_amdgpu_subgroup_cluster_size_is_supported(cluster_size,
                                                        wavefront_size)) {
      *out_failure = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_SIZE;
      return false;
    }
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    *out_failure =
        has_cluster_attrs
            ? LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_CLUSTER_MULTIPLE
            : LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE;
    return false;
  }

  if (!has_cluster_attrs) {
    if (flat_workgroup_size <= wavefront_size) {
      *out_shape = (loom_amdgpu_subgroup_reduce_shape_t){
          .active_lane_count = flat_workgroup_size,
      };
      return true;
    }
    if ((flat_workgroup_size % wavefront_size) != 0) {
      *out_failure =
          LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE;
      return false;
    }
    *out_shape = (loom_amdgpu_subgroup_reduce_shape_t){
        .active_lane_count = wavefront_size,
    };
    return true;
  }

  if (flat_workgroup_size <= wavefront_size) {
    if ((flat_workgroup_size % (uint32_t)cluster_size) != 0) {
      *out_failure =
          LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_CLUSTER_MULTIPLE;
      return false;
    }
  } else if ((flat_workgroup_size % wavefront_size) != 0) {
    *out_failure =
        LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE;
    return false;
  }

  *out_shape = (loom_amdgpu_subgroup_reduce_shape_t){
      .active_lane_count = (uint32_t)cluster_size,
      .flags = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_CLUSTERED,
  };
  return true;
}

static iree_string_view_t loom_amdgpu_subgroup_reduce_shape_failure_key(
    loom_amdgpu_subgroup_reduce_shape_failure_t failure) {
  switch (failure) {
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_STRIDE:
      return IREE_SV("subgroup_reduce.cluster_stride");
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_SIZE:
      return IREE_SV("subgroup_reduce.power_of_two_cluster_size");
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_CLUSTER_MULTIPLE:
      return IREE_SV("subgroup_reduce.fixed_workgroup_cluster_multiple");
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE:
      return IREE_SV("subgroup_reduce.fixed_workgroup_wave_multiple");
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU subgroup reduce shape failure requires reason");
  return IREE_SV("subgroup_reduce.fixed_workgroup_wave_multiple");
}

typedef enum loom_amdgpu_collective_result_demand_e {
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS = 0,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_LEADER_WORKITEM = 1,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_SUBGROUP_LEADER_LANE = 2,
} loom_amdgpu_collective_result_demand_t;

typedef uint8_t loom_amdgpu_collective_result_demand_bits_t;

enum loom_amdgpu_collective_result_demand_bits_e {
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ALL_WORKITEMS =
      1u << LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_LEADER_WORKITEM =
      1u << LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_LEADER_WORKITEM,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_SUBGROUP_LEADER_LANE =
      1u << LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_SUBGROUP_LEADER_LANE,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ANY =
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ALL_WORKITEMS |
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_LEADER_WORKITEM |
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_SUBGROUP_LEADER_LANE,
};

typedef uint8_t loom_amdgpu_workgroup_reduce_publication_rule_flags_t;

enum loom_amdgpu_workgroup_reduce_publication_rule_flags_e {
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE = 1u << 0,
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_SINGLE_REGISTER = 1u << 1,
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_TAIL_CAPACITY = 1u << 2,
};

typedef struct loom_amdgpu_workgroup_reduce_publication_rule_t {
  // Publication strategy selected when this rule matches.
  loom_amdgpu_workgroup_reduce_publication_kind_t publication_kind;
  // Result-demand kinds accepted by this rule.
  loom_amdgpu_collective_result_demand_bits_t demand_bits;
  // Workgroup/payload facts required by this rule.
  loom_amdgpu_workgroup_reduce_publication_rule_flags_t required_flags;
  // Compile-report key describing the selected publication strategy.
  iree_string_view_t report_key;
} loom_amdgpu_workgroup_reduce_publication_rule_t;

static const loom_amdgpu_workgroup_reduce_publication_rule_t
    kLoomAmdgpuWorkgroupReducePublicationRules[] = {
        {
            .publication_kind =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM,
            .demand_bits =
                LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_LEADER_WORKITEM,
            .required_flags =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE,
            .report_key = IREE_SVL("amdgpu.workgroup_reduce.publication."
                                   "leader_workitem"),
        },
        {
            .publication_kind =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE,
            .demand_bits =
                LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_SUBGROUP_LEADER_LANE,
            .required_flags =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE |
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_SINGLE_REGISTER |
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_TAIL_CAPACITY,
            .report_key = IREE_SVL("amdgpu.workgroup_reduce.publication."
                                   "redundant_subgroup_leader_lane"),
        },
        {
            .publication_kind =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP,
            .demand_bits =
                LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ALL_WORKITEMS,
            .required_flags =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE |
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_SINGLE_REGISTER |
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_TAIL_CAPACITY,
            .report_key = IREE_SVL("amdgpu.workgroup_reduce.publication."
                                   "redundant_subgroup"),
        },
        {
            .publication_kind = LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS,
            .demand_bits = LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ANY,
            .required_flags = 0,
            .report_key = IREE_SVL("amdgpu.workgroup_reduce.publication.lds"),
        },
};

static loom_amdgpu_collective_result_demand_t
loom_amdgpu_collective_result_demand(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_slice_t results =
      loom_kernel_workgroup_reduce_results(source_op);

  loom_target_workgroup_size_t workgroup_size = {0};
  const bool has_one_dimensional_workgroup =
      loom_amdgpu_required_workgroup_size(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &workgroup_size) &&
      workgroup_size.x != 0 && workgroup_size.y == 1 && workgroup_size.z == 1;

  loom_amdgpu_collective_result_demand_t result_demand =
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_LEADER_WORKITEM;
  bool has_uses = false;
  for (iree_host_size_t i = 0;
       i < results.count &&
       result_demand != LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS;
       ++i) {
    const loom_value_t* result_value =
        loom_module_value(module, results.values[i]);
    const loom_use_t* uses = loom_value_uses(result_value);
    has_uses |= result_value->use_count != 0;
    for (uint32_t j = 0; j < result_value->use_count; ++j) {
      const loom_op_t* user_op = loom_use_user_op(uses[j]);
      const loom_control_participant_zeros_t participant_zero_bits =
          loom_low_lower_source_participant_zeros(context, user_op);
      if (iree_any_bit_set(participant_zero_bits,
                           LOOM_CONTROL_PARTICIPANT_ZERO_WORKITEM_X) &&
          has_one_dimensional_workgroup) {
        continue;
      }
      if (!iree_any_bit_set(participant_zero_bits,
                            LOOM_CONTROL_PARTICIPANT_ZERO_SUBGROUP_LANE)) {
        result_demand = LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS;
        break;
      }
      result_demand = LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_SUBGROUP_LEADER_LANE;
    }
  }
  return has_uses ? result_demand
                  : LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS;
}

static loom_amdgpu_workgroup_reduce_publication_kind_t
loom_amdgpu_select_workgroup_reduce_publication_kind(
    const loom_amdgpu_workgroup_collective_shape_t* shape,
    uint32_t wavefront_size, uint32_t register_count,
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!iree_any_bit_set(shape->flags,
                        LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE)) {
    return LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS;
  }

  const loom_amdgpu_collective_result_demand_t result_demand =
      loom_amdgpu_collective_result_demand(context, source_op);

  loom_amdgpu_workgroup_reduce_publication_rule_flags_t available_flags =
      LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE;
  if (register_count == 1) {
    available_flags |=
        LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_SINGLE_REGISTER;
  }
  // Replicated publication assigns one LDS partial to each live lane. Even
  // the shortest wave must have room for every partial in its reduction tree.
  const uint32_t tail_lane_count = shape->flat_workgroup_size % wavefront_size;
  if (tail_lane_count == 0 || tail_lane_count >= shape->wave_count) {
    available_flags |=
        LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_TAIL_CAPACITY;
  }

  const loom_amdgpu_collective_result_demand_bits_t demand_bits =
      (uint8_t)(1u << (uint32_t)result_demand);
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuWorkgroupReducePublicationRules); ++i) {
    const loom_amdgpu_workgroup_reduce_publication_rule_t* rule =
        &kLoomAmdgpuWorkgroupReducePublicationRules[i];
    if ((rule->demand_bits & demand_bits) == 0 ||
        (rule->required_flags & available_flags) != rule->required_flags) {
      continue;
    }
    return rule->publication_kind;
  }
  return LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS;
}

iree_string_view_t loom_amdgpu_workgroup_reduce_publication_report_key(
    loom_amdgpu_workgroup_reduce_publication_kind_t publication_kind) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuWorkgroupReducePublicationRules); ++i) {
    const loom_amdgpu_workgroup_reduce_publication_rule_t* rule =
        &kLoomAmdgpuWorkgroupReducePublicationRules[i];
    if (rule->publication_kind == publication_kind) {
      return rule->report_key;
    }
  }
  return iree_string_view_empty();
}

static bool loom_amdgpu_subgroup_reduce_scalar_broadcast_is_supported(
    uint32_t register_count, uint32_t wavefront_size,
    uint32_t active_lane_count, bool has_cluster_attrs) {
  return register_count > 0 && wavefront_size == 64 &&
         active_lane_count == 64 && !has_cluster_attrs;
}

static iree_string_view_t loom_amdgpu_workgroup_reduce_shape_failure_key(
    loom_amdgpu_workgroup_collective_shape_failure_t failure) {
  switch (failure) {
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WORKGROUP_SIZE:
      return IREE_SV("workgroup_reduce.fixed_workgroup_size");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WAVE_COUNT:
      return IREE_SV("workgroup_reduce.wave_count");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_SCRATCH_BYTE_LENGTH:
      return IREE_SV("workgroup_reduce.scratch_byte_length");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU workgroup reduce shape failure requires reason");
  return IREE_SV("workgroup_reduce.fixed_workgroup_size");
}

iree_status_t loom_amdgpu_select_kernel_subgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_reduce_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_reduce_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_reduce_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_reduce_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_collective_arithmetic_payload_is_supported(
          module, value, &payload_kind, &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind =
      loom_kernel_subgroup_reduce_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_subgroup_wavefront_size(context, &wavefront_size)) {
    return iree_ok_status();
  }
  loom_amdgpu_subgroup_reduce_shape_t shape = {0};
  loom_amdgpu_subgroup_reduce_shape_failure_t shape_failure =
      LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_subgroup_reduce_resolve_shape(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), source_op, wavefront_size,
          &shape, &shape_failure)) {
    return iree_ok_status();
  }
  const uint32_t active_lane_count = shape.active_lane_count;
  const bool is_clustered = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_CLUSTERED);
  const bool direct_width_selected = loom_amdgpu_select_direct_subgroup_width(
      context, wavefront_size, active_lane_count);
  loom_amdgpu_subgroup_reduce_publication_kind_t publication_kind =
      LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_ALL_LANES;
  if (!direct_width_selected) {
    if (!loom_amdgpu_subgroup_reduce_scalar_broadcast_is_supported(
            register_count, wavefront_size, active_lane_count, is_clustered)) {
      return iree_ok_status();
    }
    publication_kind = LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_SCALAR_BROADCAST;
  }

  const loom_amdgpu_descriptor_resolution_t resolutions[] = {{
      .descriptor_ref = combine_descriptor_ref,
      .out_descriptor = &out_plan->combine_descriptor,
  }};
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, IREE_ARRAYSIZE(resolutions), &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }
  if (publication_kind == LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_ALL_LANES) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
        &out_plan->bpermute_descriptor, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_READLANE_B32_SRC1_INLINE,
        &out_plan->readlane_descriptor, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
    loom_low_lower_resolved_descriptor_t unused_copy_descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        &unused_copy_descriptor, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_reduce_crosslane_kind(
      context, active_lane_count, kind, payload_kind, &out_plan->dpp_descriptor,
      &out_plan->dpp_combine_descriptor, &out_plan->permlanex16_descriptor,
      &out_plan->crosslane_kind));
  if (publication_kind ==
          LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_SCALAR_BROADCAST &&
      out_plan->crosslane_kind !=
          LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16) {
    return iree_ok_status();
  }

  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    const loom_amdgpu_descriptor_resolution_t guard_resolutions[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
            .out_descriptor = &out_plan->guard_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
            .out_descriptor = &out_plan->select_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, guard_resolutions, IREE_ARRAYSIZE(guard_resolutions),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_reduce_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->wavefront_size = wavefront_size;
  out_plan->active_lane_count = active_lane_count;
  out_plan->publication_kind = publication_kind;
  *out_selected = true;
  return loom_low_lower_plan_value_type(
      context, out_plan->result,
      loom_low_register_type(
          loom_low_lower_context_descriptor_set(context)->stable_id,
          LOOM_AMDGPU_REG_CLASS_ID_VGPR, register_count));
}

iree_status_t loom_amdgpu_select_kernel_workgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_workgroup_reduce_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_workgroup_reduce_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_workgroup_reduce_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_slice_t values =
      loom_kernel_workgroup_reduce_values(source_op);
  const loom_value_slice_t results =
      loom_kernel_workgroup_reduce_results(source_op);
  const loom_combining_kind_t kind =
      loom_kernel_workgroup_reduce_kind(source_op);
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, values.count, sizeof(*out_plan->payloads),
      (void**)&out_plan->payloads));
  out_plan->payload_count = (uint32_t)values.count;
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  uint32_t register_count = 0;
  bool payloads_supported = true;
  // The combine row is keyed by kind: all accepted scalar/vector payloads
  // use the same 32-bit component instruction and cross-lane tree.
  for (uint32_t i = 0; i < out_plan->payload_count && payloads_supported; ++i) {
    loom_amdgpu_workgroup_reduce_payload_t* payload = &out_plan->payloads[i];
    *payload = (loom_amdgpu_workgroup_reduce_payload_t){
        .value = values.values[i], .result = results.values[i]};
    payloads_supported =
        loom_amdgpu_collective_arithmetic_payload_is_supported(
            module, payload->value, &payload->kind, &payload->register_count) &&
        loom_amdgpu_collective_combine_descriptor_ref(kind, payload->kind,
                                                      &combine_descriptor_ref);
    register_count += payload->register_count;
  }
  if (!payloads_supported) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_subgroup_wavefront_size(context, &wavefront_size)) {
    return iree_ok_status();
  }
  const uint32_t partition_wavefront_size =
      loom_amdgpu_target_native_subgroup_width(
          loom_amdgpu_target_facts_cast(
              loom_low_lower_context_target_facts(context)),
          wavefront_size);

  loom_amdgpu_workgroup_collective_shape_t shape = {0};
  loom_amdgpu_workgroup_collective_shape_failure_t shape_failure =
      LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_collective_resolve_workgroup_shape(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), partition_wavefront_size,
          register_count, &shape, &shape_failure)) {
    return iree_ok_status();
  }
  const bool is_multi_wave = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE);
  const bool has_partial_tail = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_PARTIAL_TAIL);
  const bool needs_cross_wave_guard =
      is_multi_wave &&
      (has_partial_tail || !loom_amdgpu_u32_is_power_of_two(shape.wave_count));
  const bool needs_subgroup_guard =
      shape.flat_workgroup_size < partition_wavefront_size &&
      !loom_amdgpu_u32_is_power_of_two(shape.flat_workgroup_size);
  const bool needs_peer_guard = needs_subgroup_guard || needs_cross_wave_guard;

  const loom_amdgpu_descriptor_resolution_t resolutions[] = {
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
          .out_descriptor = &out_plan->bpermute_descriptor,
      },
      {
          .descriptor_ref = combine_descriptor_ref,
          .out_descriptor = &out_plan->combine_descriptor,
      },
  };
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, IREE_ARRAYSIZE(resolutions), &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  const uint32_t per_wave_active_lane_count =
      is_multi_wave ? partition_wavefront_size : shape.flat_workgroup_size;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_reduce_crosslane_kind(
      context, per_wave_active_lane_count, kind, out_plan->payloads[0].kind,
      &out_plan->dpp_descriptor, &out_plan->dpp_combine_descriptor,
      &out_plan->permlanex16_descriptor, &out_plan->crosslane.per_wave));
  out_plan->crosslane.cross_wave =
      loom_amdgpu_subgroup_reduce_choose_crosslane_kind(
          shape.wave_count, &out_plan->dpp_descriptor,
          &out_plan->dpp_combine_descriptor, &out_plan->permlanex16_descriptor);

  if (needs_peer_guard || is_multi_wave) {
    const loom_amdgpu_descriptor_resolution_t guard_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
            .out_descriptor = &out_plan->guard_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, guard_resolution, IREE_ARRAYSIZE(guard_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }
  if (is_multi_wave) {
    loom_low_lower_resolved_descriptor_t unused_lane_mask_descriptor = {0};
    const loom_amdgpu_descriptor_resolution_t lane_mask_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
            .out_descriptor = &unused_lane_mask_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, lane_mask_resolution, IREE_ARRAYSIZE(lane_mask_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }
  if (has_partial_tail) {
    const loom_amdgpu_descriptor_resolution_t lane_ge_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
            .out_descriptor = &out_plan->lane_ge_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, lane_ge_resolution, IREE_ARRAYSIZE(lane_ge_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  if (needs_peer_guard) {
    const loom_amdgpu_descriptor_resolution_t select_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
            .out_descriptor = &out_plan->select_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, select_resolution, IREE_ARRAYSIZE(select_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  if (is_multi_wave) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_resolve_cross_wave_descriptors(
        context, &out_plan->cross_wave, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  out_plan->register_count = register_count;
  out_plan->wavefront_size = wavefront_size;
  out_plan->partition_wavefront_size = partition_wavefront_size;
  out_plan->flat_workgroup_size = shape.flat_workgroup_size;
  out_plan->scratch_byte_length = shape.scratch_byte_length;
  out_plan->publication_kind =
      loom_amdgpu_select_workgroup_reduce_publication_kind(
          &shape, partition_wavefront_size, register_count, context, source_op);
  *out_selected = true;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < out_plan->payload_count && iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_workgroup_reduce_payload_t* payload =
        &out_plan->payloads[i];
    status = loom_low_lower_plan_value_type(
        context, payload->result,
        loom_low_register_type(
            loom_low_lower_context_descriptor_set(context)->stable_id,
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, payload->register_count));
  }
  return status;
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_partition_lane_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t linear_id, uint32_t flat_workgroup_size,
    uint32_t partition_wavefront_size, loom_type_t lane_type,
    loom_value_id_t* out_lane_id) {
  *out_lane_id = LOOM_VALUE_ID_INVALID;
  if (flat_workgroup_size <= partition_wavefront_size) {
    *out_lane_id = linear_id;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, linear_id,
      partition_wavefront_size - 1, lane_type, out_lane_id);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind,
      plan->payload_materialization, &low_value));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &result_registers[i]));
  }
  if (plan->publication_kind ==
      LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_SCALAR_BROADCAST) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_scalar_broadcast(
        context, source_op, plan, lane_id, lane_type, result_registers));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, plan, lane_id, lane_type, LOOM_VALUE_ID_INVALID,
        result_registers));
  }

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t scratch_base, loom_value_id_t dynamic_byte_offset,
    uint32_t static_byte_offset, loom_type_t lane_type,
    loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, scratch_base,
      dynamic_byte_offset, lane_type, out_address));
  if (static_byte_offset == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      *out_address, static_byte_offset, lane_type, out_address);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t scratch_base, uint32_t static_byte_offset,
    loom_type_t lane_type, loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  if (static_byte_offset == 0) {
    *out_address = scratch_base;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      scratch_base, static_byte_offset, lane_type, out_address);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_write(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t address,
    loom_value_id_t value) {
  const loom_value_id_t operands[] = {
      address,
      value,
  };
  loom_op_t* write_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.lds_write_descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &write_op);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_read(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t address,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {address};
  loom_op_t* read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.lds_read_descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &lane_type,
      1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &read_op));
  *out_value = loom_value_slice_get(loom_low_op_results(read_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  return loom_amdgpu_collective_emit_cross_wave_barrier(context, source_op,
                                                        &plan->cross_wave);
}

static iree_status_t loom_amdgpu_bind_workgroup_reduce_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    const loom_value_id_t* result_registers) {
  iree_status_t status = iree_ok_status();
  uint32_t register_offset = 0;
  for (uint32_t i = 0; i < plan->payload_count && iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_workgroup_reduce_payload_t* payload = &plan->payloads[i];
    status = loom_amdgpu_collective_bind_payload_result(
        context, source_op, payload->result, payload->register_count,
        result_registers + register_offset);
    register_offset += payload->register_count;
  }
  return status;
}

static iree_status_t loom_amdgpu_complete_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    const loom_value_id_t* result_registers) {
  // Every workitem must finish reading shared partials before another dynamic
  // execution can reuse the reduction's workgroup storage.
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));
  return loom_amdgpu_bind_workgroup_reduce_results(context, source_op, plan,
                                                   result_registers);
}

static iree_status_t loom_amdgpu_import_workgroup_reduce_payloads(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_type_t lane_type,
    loom_value_id_t* out_registers) {
  iree_status_t status = iree_ok_status();
  uint32_t register_offset = 0;
  for (uint32_t i = 0; i < plan->payload_count && iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_workgroup_reduce_payload_t* payload = &plan->payloads[i];
    loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
    status = loom_amdgpu_collective_lookup_payload(
        context, source_op, payload->value, payload->kind,
        payload->payload_materialization, &low_value);
    for (uint32_t j = 0;
         j < payload->register_count && iree_status_is_ok(status); ++j) {
      status = loom_amdgpu_collective_payload_register(
          context, source_op, payload->register_count, low_value, j, lane_type,
          &out_registers[register_offset++]);
    }
  }
  return status;
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_saveexec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t guard,
    loom_type_t mask_type, loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {mask_type, active_type};
  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.saveexec_descriptor, &guard, 1,
      loom_make_named_attr_slice(NULL, 0), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &saveexec_op));
  *out_saved_exec = loom_op_const_results(saveexec_op)[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_restore_exec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    loom_value_id_t saved_exec) {
  loom_op_t* restore_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.restore_exec_descriptor, &saved_exec, 1,
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &restore_op);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_linear_leader_guard(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t linear_id,
    loom_type_t lane_type, loom_type_t mask_type,
    loom_value_id_t* out_leader_guard) {
  *out_leader_guard = LOOM_VALUE_ID_INVALID;
  loom_value_id_t leader_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 1, lane_type,
      &leader_count));
  return loom_amdgpu_emit_subgroup_lane_compare(
      context, source_op, &plan->guard_descriptor, linear_id, leader_count,
      mask_type, out_leader_guard);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_serial_scratch_fold(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    loom_value_id_t scratch_base, loom_value_id_t linear_id,
    uint32_t wave_count, loom_type_t lane_type, loom_type_t mask_type,
    loom_value_id_t* inout_registers) {
  loom_value_id_t leader_guard = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_linear_leader_guard(
      context, source_op, plan, linear_id, lane_type, mask_type,
      &leader_guard));
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
      context, source_op, plan, leader_guard, mask_type, &saved_exec));

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
            context, source_op, scratch_base, register_byte_offset, lane_type,
            &address));
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
        context, source_op, plan, address, lane_type, &accumulator));
    for (uint32_t wave_ordinal = 1; wave_ordinal < wave_count; ++wave_ordinal) {
      const uint32_t scratch_byte_offset =
          register_byte_offset + wave_ordinal * 4u;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
              context, source_op, scratch_base, scratch_byte_offset, lane_type,
              &address));
      loom_value_id_t partial = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &partial));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, partial,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return loom_amdgpu_emit_workgroup_reduce_restore_exec(context, source_op,
                                                        plan, saved_exec);
}

static iree_status_t
loom_amdgpu_emit_workgroup_reduce_serial_scratch_publication(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    loom_value_id_t scratch_base, loom_value_id_t linear_id,
    uint32_t wave_count, loom_type_t lane_type, loom_type_t mask_type,
    loom_value_id_t* inout_registers) {
  loom_value_id_t leader_guard = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_linear_leader_guard(
      context, source_op, plan, linear_id, lane_type, mask_type,
      &leader_guard));
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
      context, source_op, plan, leader_guard, mask_type, &saved_exec));
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
            context, source_op, scratch_base, register_byte_offset, lane_type,
            &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
        context, source_op, plan, address, inout_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
      context, source_op, plan, saved_exec));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
            context, source_op, scratch_base, register_byte_offset, lane_type,
            &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
        context, source_op, plan, address, lane_type, &inout_registers[i]));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t* result_registers = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, plan->register_count, sizeof(*result_registers),
      (void**)&result_registers));
  if (plan->flat_workgroup_size <= plan->partition_wavefront_size) {
    const loom_amdgpu_subgroup_reduce_plan_t subgroup_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .dpp_descriptor = plan->dpp_descriptor,
        .dpp_combine_descriptor = plan->dpp_combine_descriptor,
        .permlanex16_descriptor = plan->permlanex16_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .register_count = plan->register_count,
        .wavefront_size = plan->partition_wavefront_size,
        .active_lane_count = plan->flat_workgroup_size,
        .crosslane_kind = plan->crosslane.per_wave,
    };
    loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
        context, source_op, lane_type, &lane_id));
    IREE_RETURN_IF_ERROR(loom_amdgpu_import_workgroup_reduce_payloads(
        context, source_op, plan, lane_type, result_registers));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, &subgroup_plan, lane_id, lane_type,
        LOOM_VALUE_ID_INVALID, result_registers));
    return loom_amdgpu_bind_workgroup_reduce_results(context, source_op, plan,
                                                     result_registers);
  }

  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_workitem_linear_id(
      context, source_op, lane_type, &linear_id));
  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_partition_lane_id(
      context, source_op, linear_id, plan->flat_workgroup_size,
      plan->partition_wavefront_size, lane_type, &lane_id));

  const uint32_t register_count = plan->register_count;
  const uint32_t flat_workgroup_size = plan->flat_workgroup_size;
  const uint32_t partition_wavefront_size = plan->partition_wavefront_size;
  const bool has_partial_tail =
      (flat_workgroup_size % partition_wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + partition_wavefront_size - 1) /
      partition_wavefront_size;
  const uint32_t tail_lane_count =
      flat_workgroup_size % partition_wavefront_size;

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      builder, (int64_t)plan->scratch_byte_length, /*byte_alignment=*/4,
      loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP), source_op->location,
      &storage_op));
  loom_op_t* storage_address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, loom_low_storage_reserve_storage(storage_op), /*offset=*/0,
      lane_type, source_op->location, &storage_address_op));
  const loom_value_id_t scratch_base =
      loom_low_storage_address_result(storage_address_op);

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t dynamic_active_lane_count = LOOM_VALUE_ID_INVALID;
  if (has_partial_tail) {
    loom_value_id_t tail_wave_first_linear_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        (wave_count - 1) * partition_wavefront_size, lane_type,
        &tail_wave_first_linear_id));
    loom_value_id_t tail_wave_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->lane_ge_descriptor, linear_id,
        tail_wave_first_linear_id, mask_type, &tail_wave_guard));

    loom_value_id_t full_lane_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        partition_wavefront_size, lane_type, &full_lane_count_value));
    loom_value_id_t tail_lane_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        tail_lane_count, lane_type, &tail_lane_count_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_select(
        context, source_op, &plan->select_descriptor, full_lane_count_value,
        tail_lane_count_value, tail_wave_guard, lane_type,
        &dynamic_active_lane_count));
  }

  {
    IREE_RETURN_IF_ERROR(loom_amdgpu_import_workgroup_reduce_payloads(
        context, source_op, plan, lane_type, result_registers));

    const loom_amdgpu_subgroup_reduce_plan_t per_wave_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .dpp_descriptor = plan->dpp_descriptor,
        .dpp_combine_descriptor = plan->dpp_combine_descriptor,
        .permlanex16_descriptor = plan->permlanex16_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .register_count = register_count,
        .wavefront_size = partition_wavefront_size,
        .active_lane_count = partition_wavefront_size,
        .crosslane_kind = plan->crosslane.per_wave,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, &per_wave_plan, lane_id, lane_type,
        dynamic_active_lane_count, result_registers));

    loom_value_id_t subgroup_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        loom_amdgpu_subgroup_u32_log2(partition_wavefront_size), linear_id,
        lane_type, &subgroup_id));
    loom_value_id_t subgroup_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_vgpr_scale_u32(context, source_op, subgroup_id, 4,
                                        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
                                        lane_type, &subgroup_byte_offset));

    loom_value_id_t lane_zero_count = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 1, lane_type,
        &lane_zero_count));
    loom_value_id_t lane_zero_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->guard_descriptor, lane_id, lane_zero_count,
        mask_type, &lane_zero_guard));
    loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, lane_zero_guard, mask_type, &saved_exec));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, subgroup_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
          context, source_op, plan, address, result_registers[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, saved_exec));

    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

    // RDNA wave64 workgroups execute as native 32-lane partitions. Fold the
    // partition partials through LDS under linear workitem 0 instead of forming
    // a synthetic cross-lane subgroup from partition leaders.
    if (plan->wavefront_size > partition_wavefront_size) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_workgroup_reduce_serial_scratch_fold(
              context, source_op, plan, scratch_base, linear_id, wave_count,
              lane_type, mask_type, result_registers));
      if (plan->publication_kind ==
          LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM) {
        return loom_amdgpu_complete_workgroup_reduce(context, source_op, plan,
                                                     result_registers);
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_workgroup_reduce_serial_scratch_publication(
              context, source_op, plan, scratch_base, linear_id, wave_count,
              lane_type, mask_type, result_registers));
      return loom_amdgpu_complete_workgroup_reduce(context, source_op, plan,
                                                   result_registers);
    }

    if (plan->publication_kind ==
            LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP ||
        plan->publication_kind ==
            LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE) {
      loom_value_id_t wave_count_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, wave_count,
          lane_type, &wave_count_value));
      loom_value_id_t wave_participant_guard = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, lane_id,
          wave_count_value, mask_type, &wave_participant_guard));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
          context, source_op, plan, wave_participant_guard, mask_type,
          &saved_exec));

      loom_value_id_t lane_byte_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, lane_id, lane_type, &lane_byte_offset));
      for (uint32_t i = 0; i < register_count; ++i) {
        const uint32_t register_byte_offset = i * wave_count * 4u;
        loom_value_id_t address = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
            context, source_op, scratch_base, lane_byte_offset,
            register_byte_offset, lane_type, &address));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
            context, source_op, plan, address, lane_type,
            &result_registers[i]));
      }

      const loom_amdgpu_subgroup_reduce_plan_t cross_wave_plan = {
          .bpermute_descriptor = plan->bpermute_descriptor,
          .dpp_descriptor = plan->dpp_descriptor,
          .dpp_combine_descriptor = plan->dpp_combine_descriptor,
          .permlanex16_descriptor = plan->permlanex16_descriptor,
          .combine_descriptor = plan->combine_descriptor,
          .guard_descriptor = plan->guard_descriptor,
          .select_descriptor = plan->select_descriptor,
          .register_count = register_count,
          .wavefront_size = partition_wavefront_size,
          .active_lane_count = wave_count,
          .crosslane_kind = plan->crosslane.cross_wave,
      };
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
          context, source_op, &cross_wave_plan, lane_id, lane_type,
          LOOM_VALUE_ID_INVALID, result_registers));

      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
          context, source_op, plan, saved_exec));

      if (plan->publication_kind ==
          LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE) {
        return loom_amdgpu_complete_workgroup_reduce(context, source_op, plan,
                                                     result_registers);
      }

      loom_value_id_t first_lane_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
          lane_type, &first_lane_offset));
      for (uint32_t i = 0; i < register_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->bpermute_descriptor, first_lane_offset,
            /*static_byte_offset=*/0, result_registers[i], lane_type,
            &result_registers[i]));
      }

      return loom_amdgpu_complete_workgroup_reduce(context, source_op, plan,
                                                   result_registers);
    }

    loom_value_id_t wave_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, wave_count,
        lane_type, &wave_count_value));
    loom_value_id_t first_wave_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->guard_descriptor, linear_id,
        wave_count_value, mask_type, &first_wave_guard));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, first_wave_guard, mask_type, &saved_exec));

    loom_value_id_t lane_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
        context, source_op, lane_id, lane_type, &lane_byte_offset));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, lane_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &result_registers[i]));
    }

    const loom_amdgpu_subgroup_reduce_plan_t cross_wave_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .dpp_descriptor = plan->dpp_descriptor,
        .dpp_combine_descriptor = plan->dpp_combine_descriptor,
        .permlanex16_descriptor = plan->permlanex16_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .register_count = register_count,
        .wavefront_size = partition_wavefront_size,
        .active_lane_count = wave_count,
        .crosslane_kind = plan->crosslane.cross_wave,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, &cross_wave_plan, lane_id, lane_type,
        LOOM_VALUE_ID_INVALID, result_registers));

    if (plan->publication_kind ==
        LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
          context, source_op, plan, saved_exec));
      return loom_amdgpu_complete_workgroup_reduce(context, source_op, plan,
                                                   result_registers);
    }

    loom_value_id_t publish_saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, lane_zero_guard, mask_type,
        &publish_saved_exec));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, lane_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
          context, source_op, plan, address, result_registers[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, publish_saved_exec));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, saved_exec));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
              context, source_op, scratch_base, register_byte_offset, lane_type,
              &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &result_registers[i]));
    }

    return loom_amdgpu_complete_workgroup_reduce(context, source_op, plan,
                                                 result_registers);
  }
}
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);

  const loom_value_id_t value = loom_kernel_subgroup_reduce_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_collective_arithmetic_payload_is_supported(
          module, value, &payload_kind, &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_reduce.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_subgroup_reduce_kind(op);
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.combining_kind"));
  }

  const uint32_t wavefront_size = bundle->snapshot->subgroup_size;
  if (!loom_amdgpu_wavefront_size_is_valid(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.wavefront_size"));
  }
  loom_amdgpu_subgroup_reduce_shape_t shape = {0};
  loom_amdgpu_subgroup_reduce_shape_failure_t shape_failure =
      LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_subgroup_reduce_resolve_shape(
          module, loom_target_low_legality_function(context), bundle, op,
          wavefront_size, &shape, &shape_failure)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_subgroup_reduce_shape_failure_key(shape_failure));
  }
  const uint32_t active_lane_count = shape.active_lane_count;
  const bool is_clustered = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_CLUSTERED);
  const bool direct_width_supported =
      loom_amdgpu_target_supports_direct_subgroup_width(
          loom_amdgpu_target_facts_cast(
              loom_target_low_legality_target_facts(context)),
          wavefront_size, active_lane_count);
  bool scalar_broadcast_supported = false;
  if (!direct_width_supported) {
    scalar_broadcast_supported =
        loom_amdgpu_subgroup_reduce_scalar_broadcast_is_supported(
            unused_register_count, wavefront_size, active_lane_count,
            is_clustered);
    if (!scalar_broadcast_supported) {
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("subgroup_reduce.native_width.scalar_broadcast"));
    }
  }
  if (scalar_broadcast_supported) {
    const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.reduce_combine"),
            .descriptor_ref = combine_descriptor_ref,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_readlane_b32"),
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_V_READLANE_B32_SRC1_INLINE,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_permlanex16_b32"),
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_V_PERMLANEX16_B32_SRC12_INLINE,
        },
    };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_legality_verify_descriptor_requirements(
            context, op, requirements, IREE_ARRAYSIZE(requirements)));
    if (!loom_amdgpu_subgroup_reduce_dpp_row_descriptor_is_present(
            loom_target_low_legality_descriptor_set(context), kind,
            payload_kind)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.reduce_dpp_row"));
    }
  } else {
    const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.ds_bpermute_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.reduce_combine"),
            .descriptor_ref = combine_descriptor_ref,
        },
    };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_legality_verify_descriptor_requirements(
            context, op, requirements, IREE_ARRAYSIZE(requirements)));
  }
  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    static const loom_amdgpu_low_legality_descriptor_requirement_t
        active_lane_requirements[] = {
            {
                .constraint_key = IREE_SVL("descriptor.v_cmp_ult_u32"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
            },
            {
                .constraint_key = IREE_SVL("descriptor.v_cndmask_b32"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
            },
        };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_legality_verify_descriptor_requirements(
            context, op, active_lane_requirements,
            IREE_ARRAYSIZE(active_lane_requirements)));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_workgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_slice_t values = loom_kernel_workgroup_reduce_values(op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_workgroup_reduce_kind(op);
  uint32_t register_count = 0;
  iree_string_view_t failure_key = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    loom_amdgpu_subgroup_payload_kind_t payload_kind =
        LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
    uint32_t payload_register_count = 0;
    if (!loom_amdgpu_collective_arithmetic_payload_is_supported(
            module, values.values[i], &payload_kind, &payload_register_count)) {
      failure_key = IREE_SV("workgroup_reduce.payload");
      break;
    }
    if (!loom_amdgpu_collective_combine_descriptor_ref(
            kind, payload_kind, &combine_descriptor_ref)) {
      failure_key = IREE_SV("workgroup_reduce.combining_kind");
      break;
    }
    register_count += payload_register_count;
  }
  if (!iree_string_view_is_empty(failure_key)) {
    return loom_amdgpu_low_legality_reject(context, op, failure_key);
  }

  const uint32_t wavefront_size = bundle->snapshot->subgroup_size;
  if (!loom_amdgpu_wavefront_size_is_valid(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.wavefront_size"));
  }
  const uint32_t partition_wavefront_size =
      loom_amdgpu_target_native_subgroup_width(
          loom_amdgpu_target_facts_cast(
              loom_target_low_legality_target_facts(context)),
          wavefront_size);

  loom_amdgpu_workgroup_collective_shape_t shape = {0};
  loom_amdgpu_workgroup_collective_shape_failure_t shape_failure =
      LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_collective_resolve_workgroup_shape(
          module, loom_target_low_legality_function(context), bundle,
          partition_wavefront_size, register_count, &shape, &shape_failure)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_workgroup_reduce_shape_failure_key(shape_failure));
  }
  const bool is_multi_wave = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE);
  const bool has_partial_tail = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_PARTIAL_TAIL);
  const bool needs_cross_wave_guard =
      is_multi_wave &&
      (has_partial_tail || !loom_amdgpu_u32_is_power_of_two(shape.wave_count));
  const bool needs_subgroup_guard =
      shape.flat_workgroup_size < partition_wavefront_size &&
      !loom_amdgpu_u32_is_power_of_two(shape.flat_workgroup_size);
  const bool needs_peer_guard = needs_subgroup_guard || needs_cross_wave_guard;

  const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
      {
          .constraint_key = IREE_SVL("descriptor.ds_bpermute_b32"),
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      },
      {
          .constraint_key = IREE_SVL("descriptor.combine"),
          .descriptor_ref = combine_descriptor_ref,
      },
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirements(
      context, op, requirements, IREE_ARRAYSIZE(requirements)));
  if (needs_peer_guard || is_multi_wave) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
        IREE_SV("descriptor.v_cmp_ult_u32")));
  }
  if (is_multi_wave) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        IREE_SV("descriptor.v_and_b32_lit")));
  }
  if (has_partial_tail) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
        IREE_SV("descriptor.v_cmp_uge_u32")));
  }
  if (needs_peer_guard) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
        IREE_SV("descriptor.v_cndmask_b32")));
  }
  if (is_multi_wave) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_collective_verify_cross_wave_descriptor_requirements(
            context, op));
  }

  return iree_ok_status();
}

#undef LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS
