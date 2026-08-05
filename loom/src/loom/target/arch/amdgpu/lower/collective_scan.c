// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/collective_combine.h"
#include "loom/target/arch/amdgpu/lower/collective_payload.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/workgroup.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#define LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS 6u

static uint32_t loom_amdgpu_subgroup_u32_log2(uint32_t value) {
  uint32_t log2 = 0;
  while (value > 1) {
    value >>= 1;
    ++log2;
  }
  return log2;
}

static bool loom_amdgpu_subgroup_optional_attr_is_present(const loom_op_t* op,
                                                          uint16_t attr_index) {
  return attr_index < op->attribute_count &&
         !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static bool loom_amdgpu_subgroup_scan_has_cluster_attrs(const loom_op_t* op) {
  return loom_amdgpu_subgroup_optional_attr_is_present(
             op, loom_kernel_subgroup_scan_cluster_size_ATTR_INDEX) ||
         loom_amdgpu_subgroup_optional_attr_is_present(
             op, loom_kernel_subgroup_scan_cluster_stride_ATTR_INDEX);
}

static bool loom_amdgpu_subgroup_full_wave_workgroups(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t wavefront_size) {
  uint32_t flat_workgroup_size = 0;
  return loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                  &flat_workgroup_size) &&
         flat_workgroup_size >= wavefront_size &&
         (flat_workgroup_size % wavefront_size) == 0;
}

typedef enum loom_amdgpu_scan_mode_flag_bits_t {
  LOOM_AMDGPU_SCAN_MODE_REQUIRES_IDENTITY = 1u << 0,
} loom_amdgpu_scan_mode_flag_bits_t;

typedef uint32_t loom_amdgpu_scan_mode_flags_t;

typedef struct loom_amdgpu_scan_mode_rule_t {
  // Subgroup scan mode used by lowering plans.
  loom_kernel_subgroup_scan_mode_t subgroup_mode;
  // Mode properties consumed by selection, legality, and lowering setup.
  loom_amdgpu_scan_mode_flags_t flags;
} loom_amdgpu_scan_mode_rule_t;

static const loom_amdgpu_scan_mode_rule_t
    kLoomAmdgpuSubgroupScanModeRules[LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_] = {
        [LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE] =
            {LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE, 0},
        [LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE] =
            {LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE,
             LOOM_AMDGPU_SCAN_MODE_REQUIRES_IDENTITY},
};

static const loom_amdgpu_scan_mode_rule_t
    kLoomAmdgpuWorkgroupScanModeRules[LOOM_KERNEL_WORKGROUP_SCAN_MODE_COUNT_] =
        {
            [LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE] =
                {LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE, 0},
            [LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE] =
                {LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE,
                 LOOM_AMDGPU_SCAN_MODE_REQUIRES_IDENTITY},
};

static const loom_amdgpu_scan_mode_rule_t* loom_amdgpu_subgroup_scan_mode_rule(
    loom_kernel_subgroup_scan_mode_t mode) {
  if ((uint32_t)mode >= LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_) {
    return NULL;
  }
  return &kLoomAmdgpuSubgroupScanModeRules[mode];
}

static const loom_amdgpu_scan_mode_rule_t* loom_amdgpu_workgroup_scan_mode_rule(
    loom_kernel_workgroup_scan_mode_t mode) {
  if ((uint32_t)mode >= LOOM_KERNEL_WORKGROUP_SCAN_MODE_COUNT_) {
    return NULL;
  }
  return &kLoomAmdgpuWorkgroupScanModeRules[mode];
}

static bool loom_amdgpu_scan_mode_requires_identity(
    const loom_amdgpu_scan_mode_rule_t* rule) {
  return iree_all_bits_set(rule->flags,
                           LOOM_AMDGPU_SCAN_MODE_REQUIRES_IDENTITY);
}

typedef struct loom_amdgpu_scan_direction_rule_t {
  // Subgroup scan direction used by lowering plans.
  loom_kernel_subgroup_scan_direction_t subgroup_direction;
  // Guard descriptor used to select active lanes for the direction.
  loom_amdgpu_descriptor_ref_t guard_descriptor_ref;
} loom_amdgpu_scan_direction_rule_t;

static const loom_amdgpu_scan_direction_rule_t
    kLoomAmdgpuSubgroupScanDirectionRules
        [LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_] = {
            [LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD] =
                {LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32},
            [LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE] =
                {LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32},
};

static const loom_amdgpu_scan_direction_rule_t
    kLoomAmdgpuWorkgroupScanDirectionRules
        [LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_COUNT_] = {
            [LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD] =
                {LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32},
            [LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_REVERSE] =
                {LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32},
};

static const loom_amdgpu_scan_direction_rule_t*
loom_amdgpu_subgroup_scan_direction_rule(
    loom_kernel_subgroup_scan_direction_t direction) {
  if ((uint32_t)direction >= LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_) {
    return NULL;
  }
  const loom_amdgpu_scan_direction_rule_t* rule =
      &kLoomAmdgpuSubgroupScanDirectionRules[direction];
  return rule->guard_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE ? NULL
                                                                       : rule;
}

static const loom_amdgpu_scan_direction_rule_t*
loom_amdgpu_workgroup_scan_direction_rule(
    loom_kernel_workgroup_scan_direction_t direction) {
  if ((uint32_t)direction >= LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_COUNT_) {
    return NULL;
  }
  const loom_amdgpu_scan_direction_rule_t* rule =
      &kLoomAmdgpuWorkgroupScanDirectionRules[direction];
  return rule->guard_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE ? NULL
                                                                       : rule;
}

static iree_string_view_t loom_amdgpu_workgroup_scan_shape_failure_key(
    loom_amdgpu_workgroup_collective_shape_failure_t failure) {
  switch (failure) {
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WORKGROUP_SIZE:
      return IREE_SV("workgroup_scan.fixed_workgroup_size");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WAVE_COUNT:
      return IREE_SV("workgroup_scan.wave_count");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_SCRATCH_BYTE_LENGTH:
      return IREE_SV("workgroup_scan.scratch_byte_length");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU workgroup scan shape failure requires reason");
  return IREE_SV("workgroup_scan.fixed_workgroup_size");
}

iree_status_t loom_amdgpu_select_kernel_subgroup_scan_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_scan_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_scan_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_scan_isa(source_op)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_subgroup_scan_has_cluster_attrs(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_scan_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind = loom_kernel_subgroup_scan_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return iree_ok_status();
  }
  const loom_amdgpu_scan_mode_rule_t* mode_rule =
      loom_amdgpu_subgroup_scan_mode_rule(
          loom_kernel_subgroup_scan_mode(source_op));
  if (mode_rule == NULL) {
    return iree_ok_status();
  }
  uint32_t identity_bits = 0;
  if (loom_amdgpu_scan_mode_requires_identity(mode_rule)) {
    if (!loom_amdgpu_collective_combine_identity_bits(kind, &identity_bits)) {
      return iree_ok_status();
    }
  }

  const loom_amdgpu_scan_direction_rule_t* direction_rule =
      loom_amdgpu_subgroup_scan_direction_rule(
          loom_kernel_subgroup_scan_direction(source_op));
  if (direction_rule == NULL) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_full_wave_direct_subgroup_width(context,
                                                          &wavefront_size) ||
      !loom_amdgpu_subgroup_full_wave_workgroups(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), wavefront_size)) {
    return iree_ok_status();
  }

  const loom_amdgpu_descriptor_resolution_t resolutions[] = {
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
          .out_descriptor = &out_plan->bpermute_descriptor,
      },
      {
          .descriptor_ref = combine_descriptor_ref,
          .out_descriptor = &out_plan->combine_descriptor,
      },
      {
          .descriptor_ref = direction_rule->guard_descriptor_ref,
          .out_descriptor = &out_plan->guard_descriptor,
      },
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
          .out_descriptor = &out_plan->select_descriptor,
      },
  };
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, IREE_ARRAYSIZE(resolutions), &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_scan_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->mode = mode_rule->subgroup_mode;
  out_plan->direction = direction_rule->subgroup_direction;
  out_plan->identity_bits = identity_bits;
  out_plan->wavefront_size = wavefront_size;
  out_plan->active_lane_count = wavefront_size;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_workgroup_scan_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_workgroup_scan_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_workgroup_scan_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_workgroup_scan_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_workgroup_scan_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind = loom_kernel_workgroup_scan_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return iree_ok_status();
  }

  const loom_amdgpu_scan_mode_rule_t* mode_rule =
      loom_amdgpu_workgroup_scan_mode_rule(
          loom_kernel_workgroup_scan_mode(source_op));
  if (mode_rule == NULL) {
    return iree_ok_status();
  }
  uint32_t identity_bits = 0;
  if (loom_amdgpu_scan_mode_requires_identity(mode_rule)) {
    if (!loom_amdgpu_collective_combine_identity_bits(kind, &identity_bits)) {
      return iree_ok_status();
    }
  }

  const loom_amdgpu_scan_direction_rule_t* direction_rule =
      loom_amdgpu_workgroup_scan_direction_rule(
          loom_kernel_workgroup_scan_direction(source_op));
  if (direction_rule == NULL) {
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
  const bool is_multi_partition = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE);
  const bool has_partial_partition = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_PARTIAL_TAIL);

  const loom_amdgpu_descriptor_resolution_t resolutions[] = {
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
          .out_descriptor = &out_plan->bpermute_descriptor,
      },
      {
          .descriptor_ref = combine_descriptor_ref,
          .out_descriptor = &out_plan->combine_descriptor,
      },
      {
          .descriptor_ref = direction_rule->guard_descriptor_ref,
          .out_descriptor = &out_plan->guard_descriptor,
      },
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
          .out_descriptor = &out_plan->select_descriptor,
      },
  };
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, IREE_ARRAYSIZE(resolutions), &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  if (is_multi_partition &&
      !loom_amdgpu_scan_mode_requires_identity(mode_rule)) {
    if (!loom_amdgpu_collective_combine_identity_bits(kind, &identity_bits)) {
      return iree_ok_status();
    }
  }

  if (is_multi_partition) {
    const loom_amdgpu_descriptor_resolution_t lane_lt_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
            .out_descriptor = &out_plan->lane_lt_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, lane_lt_resolution, IREE_ARRAYSIZE(lane_lt_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }

    if (has_partial_partition) {
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

    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_resolve_cross_wave_descriptors(
        context, &out_plan->cross_wave, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_workgroup_scan_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->mode = mode_rule->subgroup_mode;
  out_plan->direction = direction_rule->subgroup_direction;
  out_plan->identity_bits = identity_bits;
  out_plan->partition_wavefront_size = partition_wavefront_size;
  out_plan->flat_workgroup_size = shape.flat_workgroup_size;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_combine(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_lane_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_guard) {
  *out_guard = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_guard = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_scan_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t guard, loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      false_value,
      true_value,
      guard,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_scan_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan, loom_value_id_t lane_id,
    loom_value_id_t active_lane_count, uint32_t offset, loom_type_t lane_type,
    loom_type_t mask_type, loom_value_id_t* out_source_byte_offset,
    loom_value_id_t* out_guard) {
  *out_source_byte_offset = LOOM_VALUE_ID_INVALID;
  *out_guard = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, offset,
      lane_type, &low_offset));

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  switch (plan->direction) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32, lane_id,
          low_offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, lane_id, low_offset,
          mask_type, out_guard));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, lane_id,
          low_offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, source_lane,
          active_lane_count, mask_type, out_guard));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU subgroup scan lowering requires a supported direction");
      IREE_BUILTIN_UNREACHABLE();
  }

  return loom_amdgpu_emit_subgroup_lane_byte_offset(
      context, source_op, source_lane, lane_type, out_source_byte_offset);
}

static iree_status_t loom_amdgpu_emit_workgroup_scan_scratch_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t scratch_base, loom_value_id_t dynamic_byte_offset,
    uint32_t static_byte_offset, loom_type_t lane_type,
    loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, scratch_base,
      dynamic_byte_offset, lane_type, out_address));
  if (static_byte_offset == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      *out_address, static_byte_offset, lane_type, out_address);
}

static iree_status_t loom_amdgpu_emit_workgroup_scan_scratch_write(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_scan_plan_t* plan, loom_value_id_t address,
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

static iree_status_t loom_amdgpu_emit_workgroup_scan_scratch_read(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_scan_plan_t* plan, loom_value_id_t address,
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

static iree_status_t loom_amdgpu_emit_workgroup_scan_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_scan_plan_t* plan) {
  return loom_amdgpu_collective_emit_cross_wave_barrier(context, source_op,
                                                        &plan->cross_wave);
}

static iree_status_t loom_amdgpu_emit_workgroup_scan_saveexec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_scan_plan_t* plan, loom_value_id_t guard,
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

static iree_status_t loom_amdgpu_emit_workgroup_scan_restore_exec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_scan_plan_t* plan, loom_value_id_t saved_exec) {
  loom_op_t* restore_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.restore_exec_descriptor, &saved_exec, 1,
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &restore_op);
}

static iree_status_t loom_amdgpu_emit_subgroup_scan_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_type_t mask_type,
    loom_value_id_t dynamic_active_lane_count,
    loom_value_id_t* inout_registers) {
  if (plan->active_lane_count == 0 ||
      plan->active_lane_count > plan->wavefront_size) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU subgroup scan lowering requires a valid active lane count");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t active_lane_count = dynamic_active_lane_count;
  if (plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE &&
      active_lane_count == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->active_lane_count, lane_type, &active_lane_count));
  }

  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  loom_value_id_t guards[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {0};
  uint32_t step_count = 0;
  for (uint32_t offset = 1; offset < plan->active_lane_count; offset <<= 1) {
    IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_source(
        context, source_op, plan, lane_id, active_lane_count, offset, lane_type,
        mask_type, &source_byte_offsets[step_count], &guards[step_count]));
    ++step_count;
  }

  loom_value_id_t exclusive_byte_offset = LOOM_VALUE_ID_INVALID;
  loom_value_id_t exclusive_guard = LOOM_VALUE_ID_INVALID;
  const bool is_exclusive =
      plan->mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE;
  if (is_exclusive) {
    exclusive_byte_offset = source_byte_offsets[0];
    exclusive_guard = guards[0];
  } else if (plan->mode != LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU subgroup scan lowering requires a supported mode");
    IREE_BUILTIN_UNREACHABLE();
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];

    for (uint32_t step_index = 0; step_index < step_count; ++step_index) {
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          source_byte_offsets[step_index], /*static_byte_offset=*/0,
          accumulator, lane_type, &peer));
      loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &combined));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
          context, source_op, &plan->select_descriptor, accumulator, combined,
          guards[step_index], lane_type, &accumulator));
    }

    if (is_exclusive) {
      if (step_count == 0) {
        loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            plan->identity_bits, lane_type, &identity));
        accumulator = identity;
      } else {
        loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->bpermute_descriptor,
            exclusive_byte_offset, /*static_byte_offset=*/0, accumulator,
            lane_type, &shifted));
        loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            plan->identity_bits, lane_type, &identity));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
            context, source_op, &plan->select_descriptor, identity, shifted,
            exclusive_guard, lane_type, &accumulator));
      }
    }

    inout_registers[i] = accumulator;
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &result_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_tree(
      context, source_op, plan, lane_id, lane_type, mask_type,
      LOOM_VALUE_ID_INVALID, result_registers));

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

iree_status_t loom_amdgpu_lower_kernel_workgroup_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_scan_plan_t* plan) {
  const uint32_t partition_wavefront_size = plan->partition_wavefront_size;
  const loom_amdgpu_subgroup_scan_plan_t subgroup_plan = {
      .bpermute_descriptor = plan->bpermute_descriptor,
      .combine_descriptor = plan->combine_descriptor,
      .guard_descriptor = plan->guard_descriptor,
      .select_descriptor = plan->select_descriptor,
      .value = plan->value,
      .result = plan->result,
      .payload_kind = plan->payload_kind,
      .register_count = plan->register_count,
      .mode = plan->mode,
      .direction = plan->direction,
      .identity_bits = plan->identity_bits,
      .wavefront_size = partition_wavefront_size,
      .active_lane_count = plan->flat_workgroup_size,
  };
  if (plan->flat_workgroup_size <= partition_wavefront_size) {
    return loom_amdgpu_lower_kernel_subgroup_scan(context, source_op,
                                                  &subgroup_plan);
  }

  const bool has_partial_partition =
      (plan->flat_workgroup_size % partition_wavefront_size) != 0;
  const uint32_t partition_count =
      (plan->flat_workgroup_size + partition_wavefront_size - 1) /
      partition_wavefront_size;
  const uint32_t tail_lane_count =
      has_partial_partition
          ? plan->flat_workgroup_size % partition_wavefront_size
          : partition_wavefront_size;
  if (partition_count > partition_wavefront_size) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU workgroup scan lowering requires a valid partition count");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_workitem_linear_id(
      context, source_op, lane_type, &linear_id));
  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, linear_id,
      partition_wavefront_size - 1, lane_type, &lane_id));

  const uint32_t register_count = plan->register_count;
  const uint32_t scratch_byte_length = partition_count * register_count * 4u;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      builder, scratch_byte_length, /*byte_alignment=*/4,
      loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP), source_op->location,
      &storage_op));
  loom_op_t* storage_address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, loom_low_storage_reserve_storage(storage_op), /*offset=*/0,
      lane_type, source_op->location, &storage_address_op));
  const loom_value_id_t scratch_base =
      loom_low_storage_address_result(storage_address_op);

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, register_count, low_value, i, lane_type,
        &source_registers[i]));
    result_registers[i] = source_registers[i];
  }

  loom_value_id_t tail_partition_guard = LOOM_VALUE_ID_INVALID;
  loom_value_id_t dynamic_active_lane_count = LOOM_VALUE_ID_INVALID;
  if (has_partial_partition) {
    loom_value_id_t tail_partition_first_linear_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        (partition_count - 1) * partition_wavefront_size, lane_type,
        &tail_partition_first_linear_id));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->lane_ge_descriptor, linear_id,
        tail_partition_first_linear_id, mask_type, &tail_partition_guard));

    if (plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE) {
      loom_value_id_t full_lane_count_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          partition_wavefront_size, lane_type, &full_lane_count_value));
      loom_value_id_t tail_lane_count_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          tail_lane_count, lane_type, &tail_lane_count_value));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
          context, source_op, &plan->select_descriptor, full_lane_count_value,
          tail_lane_count_value, tail_partition_guard, lane_type,
          &dynamic_active_lane_count));
    }
  }

  const loom_amdgpu_subgroup_scan_plan_t intra_partition_plan = {
      .bpermute_descriptor = plan->bpermute_descriptor,
      .combine_descriptor = plan->combine_descriptor,
      .guard_descriptor = plan->guard_descriptor,
      .select_descriptor = plan->select_descriptor,
      .value = plan->value,
      .result = plan->result,
      .payload_kind = plan->payload_kind,
      .register_count = register_count,
      .mode = plan->mode,
      .direction = plan->direction,
      .identity_bits = plan->identity_bits,
      .wavefront_size = partition_wavefront_size,
      .active_lane_count = partition_wavefront_size,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_tree(
      context, source_op, &intra_partition_plan, lane_id, lane_type, mask_type,
      dynamic_active_lane_count, result_registers));

  loom_value_id_t
      partition_total_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < register_count; ++i) {
    partition_total_registers[i] = result_registers[i];
    if (plan->mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor,
          partition_total_registers[i], source_registers[i], lane_type,
          &partition_total_registers[i]));
    }
  }

  loom_value_id_t partition_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      loom_amdgpu_subgroup_u32_log2(partition_wavefront_size), linear_id,
      lane_type, &partition_id));
  loom_value_id_t partition_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
      context, source_op, partition_id, 4, LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
      lane_type, &partition_byte_offset));

  loom_value_id_t producer_lane_value = LOOM_VALUE_ID_INVALID;
  if (has_partial_partition &&
      plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD) {
    loom_value_id_t full_producer_lane_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        partition_wavefront_size - 1, lane_type, &full_producer_lane_value));
    loom_value_id_t tail_producer_lane_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        tail_lane_count - 1, lane_type, &tail_producer_lane_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
        context, source_op, &plan->select_descriptor, full_producer_lane_value,
        tail_producer_lane_value, tail_partition_guard, lane_type,
        &producer_lane_value));
  } else {
    const uint32_t producer_lane =
        plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD
            ? partition_wavefront_size - 1
            : 1;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, producer_lane,
        lane_type, &producer_lane_value));
  }
  loom_value_id_t producer_lane_guard = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
      context, source_op, &plan->guard_descriptor, lane_id, producer_lane_value,
      mask_type, &producer_lane_guard));

  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_saveexec(
      context, source_op, plan, producer_lane_guard, mask_type, &saved_exec));
  for (uint32_t i = 0; i < register_count; ++i) {
    const uint32_t register_byte_offset = i * partition_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
        context, source_op, scratch_base, partition_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_write(
        context, source_op, plan, address, partition_total_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_restore_exec(
      context, source_op, plan, saved_exec));

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_workgroup_scan_barrier(context, source_op, plan));

  if (partition_count == 2) {
    loom_value_id_t prefix_partition_guard = tail_partition_guard;
    const bool prefix_guard_selects_second_partition = has_partial_partition;
    if (!prefix_guard_selects_second_partition) {
      loom_value_id_t partition_size_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          partition_wavefront_size, lane_type, &partition_size_value));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->lane_lt_descriptor, linear_id,
          partition_size_value, mask_type, &prefix_partition_guard));
    }

    loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->identity_bits, lane_type, &identity));

    loom_value_id_t zero_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &zero_byte_offset));
    const uint32_t prefix_slot =
        plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD ? 0 : 1;
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset =
          (i * partition_count + prefix_slot) * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
          context, source_op, scratch_base, zero_byte_offset,
          register_byte_offset, lane_type, &address));
      loom_value_id_t peer_partition_prefix = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_read(
          context, source_op, plan, address, lane_type,
          &peer_partition_prefix));
      loom_value_id_t partition_prefix = LOOM_VALUE_ID_INVALID;
      if (plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD) {
        if (prefix_guard_selects_second_partition) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
              context, source_op, &plan->select_descriptor, identity,
              peer_partition_prefix, prefix_partition_guard, lane_type,
              &partition_prefix));
        } else {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
              context, source_op, &plan->select_descriptor,
              peer_partition_prefix, identity, prefix_partition_guard,
              lane_type, &partition_prefix));
        }
      } else {
        if (prefix_guard_selects_second_partition) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
              context, source_op, &plan->select_descriptor,
              peer_partition_prefix, identity, prefix_partition_guard,
              lane_type, &partition_prefix));
        } else {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
              context, source_op, &plan->select_descriptor, identity,
              peer_partition_prefix, prefix_partition_guard, lane_type,
              &partition_prefix));
        }
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, result_registers[i],
          partition_prefix, lane_type, &result_registers[i]));
    }
    return loom_amdgpu_collective_bind_payload_result(
        context, source_op, plan->result, register_count, result_registers);
  }

  loom_value_id_t partition_count_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, partition_count,
      lane_type, &partition_count_value));
  loom_value_id_t first_partition_guard = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
      context, source_op, &plan->lane_lt_descriptor, linear_id,
      partition_count_value, mask_type, &first_partition_guard));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_saveexec(
      context, source_op, plan, first_partition_guard, mask_type, &saved_exec));

  loom_value_id_t lane_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
      context, source_op, lane_id, lane_type, &lane_byte_offset));
  loom_value_id_t
      partition_prefix_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < register_count; ++i) {
    const uint32_t register_byte_offset = i * partition_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
        context, source_op, scratch_base, lane_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_read(
        context, source_op, plan, address, lane_type,
        &partition_prefix_registers[i]));
  }

  const loom_amdgpu_subgroup_scan_plan_t cross_partition_plan = {
      .bpermute_descriptor = plan->bpermute_descriptor,
      .combine_descriptor = plan->combine_descriptor,
      .guard_descriptor = plan->guard_descriptor,
      .select_descriptor = plan->select_descriptor,
      .value = plan->value,
      .result = plan->result,
      .payload_kind = plan->payload_kind,
      .register_count = register_count,
      .mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE,
      .direction = plan->direction,
      .identity_bits = plan->identity_bits,
      .wavefront_size = partition_wavefront_size,
      .active_lane_count = partition_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_tree(
      context, source_op, &cross_partition_plan, lane_id, lane_type, mask_type,
      LOOM_VALUE_ID_INVALID, partition_prefix_registers));

  for (uint32_t i = 0; i < register_count; ++i) {
    const uint32_t register_byte_offset = i * partition_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
        context, source_op, scratch_base, lane_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_write(
        context, source_op, plan, address, partition_prefix_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_restore_exec(
      context, source_op, plan, saved_exec));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_workgroup_scan_barrier(context, source_op, plan));

  for (uint32_t i = 0; i < register_count; ++i) {
    const uint32_t register_byte_offset = i * partition_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
        context, source_op, scratch_base, partition_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_read(
        context, source_op, plan, address, lane_type,
        &partition_prefix_registers[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
        context, source_op, &plan->combine_descriptor, result_registers[i],
        partition_prefix_registers[i], lane_type, &result_registers[i]));
  }

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, register_count, result_registers);
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_workgroup_scan(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_workgroup_scan_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_scan.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_workgroup_scan_kind(op);
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.combining_kind"));
  }

  const loom_amdgpu_scan_mode_rule_t* mode_rule =
      loom_amdgpu_workgroup_scan_mode_rule(loom_kernel_workgroup_scan_mode(op));
  if (mode_rule == NULL) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_scan.mode"));
  }
  if (loom_amdgpu_scan_mode_requires_identity(mode_rule)) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("workgroup_scan.identity"));
    }
  }

  const loom_amdgpu_scan_direction_rule_t* direction_rule =
      loom_amdgpu_workgroup_scan_direction_rule(
          loom_kernel_workgroup_scan_direction(op));
  if (direction_rule == NULL) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_scan.direction"));
  }

  const uint32_t wavefront_size = loom_amdgpu_target_wavefront_size(bundle);
  if (!loom_amdgpu_wavefront_size_is_valid(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.wavefront_size"));
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
          partition_wavefront_size, unused_register_count, &shape,
          &shape_failure)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_workgroup_scan_shape_failure_key(shape_failure));
  }
  const bool is_multi_partition = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE);
  const bool has_partial_partition = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_PARTIAL_TAIL);
  if (!loom_amdgpu_scan_mode_requires_identity(mode_rule) &&
      is_multi_partition) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("workgroup_scan.identity"));
    }
  }

  const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
      {
          .constraint_key = IREE_SVL("descriptor.ds_bpermute_b32"),
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      },
      {
          .constraint_key = IREE_SVL("descriptor.scan_combine"),
          .descriptor_ref = combine_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.scan_guard"),
          .descriptor_ref = direction_rule->guard_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.v_cndmask_b32"),
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      },
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirements(
      context, op, requirements, IREE_ARRAYSIZE(requirements)));
  if (is_multi_partition) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
        IREE_SV("descriptor.v_cmp_ult_u32")));
    if (has_partial_partition) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_low_legality_verify_descriptor_requirement(
              context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
              IREE_SV("descriptor.v_cmp_uge_u32")));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_collective_verify_cross_wave_descriptor_requirements(
            context, op));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_scan(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (loom_amdgpu_subgroup_scan_has_cluster_attrs(op)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.full_subgroup"));
  }

  const loom_value_id_t value = loom_kernel_subgroup_scan_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_scan.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_subgroup_scan_kind(op);
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.combining_kind"));
  }

  const loom_amdgpu_scan_mode_rule_t* mode_rule =
      loom_amdgpu_subgroup_scan_mode_rule(loom_kernel_subgroup_scan_mode(op));
  if (mode_rule == NULL) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_scan.mode"));
  }
  if (loom_amdgpu_scan_mode_requires_identity(mode_rule)) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("subgroup_scan.identity"));
    }
  }

  const loom_amdgpu_scan_direction_rule_t* direction_rule =
      loom_amdgpu_subgroup_scan_direction_rule(
          loom_kernel_subgroup_scan_direction(op));
  if (direction_rule == NULL) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_scan.direction"));
  }

  const uint32_t wavefront_size = loom_amdgpu_target_wavefront_size(bundle);
  if (!loom_amdgpu_wavefront_size_is_valid(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.wavefront_size"));
  }
  if (!loom_amdgpu_subgroup_full_wave_workgroups(
          module, loom_target_low_legality_function(context), bundle,
          wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.fixed_workgroup_wave_multiple"));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_direct_subgroup_width(
      context, op, wavefront_size, wavefront_size,
      IREE_SV("subgroup_scan.native_width")));

  const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
      {
          .constraint_key = IREE_SVL("descriptor.ds_bpermute_b32"),
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      },
      {
          .constraint_key = IREE_SVL("descriptor.scan_combine"),
          .descriptor_ref = combine_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.scan_guard"),
          .descriptor_ref = direction_rule->guard_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.v_cndmask_b32"),
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      },
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirements(
      context, op, requirements, IREE_ARRAYSIZE(requirements)));

  return iree_ok_status();
}

#undef LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS
