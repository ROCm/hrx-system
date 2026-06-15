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

static bool loom_amdgpu_subgroup_wavefront_size_is_supported(
    uint32_t wavefront_size) {
  return wavefront_size == 32 || wavefront_size == 64;
}

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
  const loom_kernel_subgroup_scan_mode_t mode =
      loom_kernel_subgroup_scan_mode(source_op);
  switch (mode) {
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE:
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE: {
      uint32_t unused_identity_bits = 0;
      if (!loom_amdgpu_collective_combine_identity_bits(
              kind, &unused_identity_bits)) {
        return iree_ok_status();
      }
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_:
      return iree_ok_status();
  }

  loom_amdgpu_descriptor_ref_t guard_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_kernel_subgroup_scan_direction_t direction =
      loom_kernel_subgroup_scan_direction(source_op);
  switch (direction) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size) ||
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
          .descriptor_ref = guard_descriptor_ref,
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
  out_plan->kind = kind;
  out_plan->mode = mode;
  out_plan->direction = direction;
  out_plan->wavefront_size = wavefront_size;
  out_plan->active_lane_count = wavefront_size;
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_map_workgroup_scan_mode(
    loom_kernel_workgroup_scan_mode_t source_mode,
    loom_kernel_subgroup_scan_mode_t* out_mode) {
  switch (source_mode) {
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE:
      *out_mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE;
      return true;
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE:
      *out_mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE;
      return true;
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_amdgpu_map_workgroup_scan_direction(
    loom_kernel_workgroup_scan_direction_t source_direction,
    loom_kernel_subgroup_scan_direction_t* out_direction) {
  switch (source_direction) {
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD:
      *out_direction = LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD;
      return true;
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_REVERSE:
      *out_direction = LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE;
      return true;
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_COUNT_:
      return false;
  }
  return false;
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

  loom_kernel_subgroup_scan_mode_t mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_;
  if (!loom_amdgpu_map_workgroup_scan_mode(
          loom_kernel_workgroup_scan_mode(source_op), &mode)) {
    return iree_ok_status();
  }
  if (mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return iree_ok_status();
    }
  }

  loom_kernel_subgroup_scan_direction_t direction =
      LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_;
  if (!loom_amdgpu_map_workgroup_scan_direction(
          loom_kernel_workgroup_scan_direction(source_op), &direction)) {
    return iree_ok_status();
  }
  loom_amdgpu_descriptor_ref_t guard_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (direction) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return iree_ok_status();
  }
  const bool has_partial_tail = flat_workgroup_size > wavefront_size &&
                                (flat_workgroup_size % wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + wavefront_size - 1) / wavefront_size;
  if (flat_workgroup_size > wavefront_size && wave_count > wavefront_size) {
    return iree_ok_status();
  }
  const uint64_t scratch_byte_length =
      (uint64_t)wave_count * register_count * 4u;
  if (scratch_byte_length > UINT32_MAX) {
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
          .descriptor_ref = guard_descriptor_ref,
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

  if (mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE ||
      flat_workgroup_size > wavefront_size) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return iree_ok_status();
    }
  }

  if (flat_workgroup_size > wavefront_size) {
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

    const loom_amdgpu_descriptor_resolution_t scratch_resolutions[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
            .out_descriptor = &out_plan->lds_read_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32,
            .out_descriptor = &out_plan->lds_write_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER,
            .out_descriptor = &out_plan->barrier_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
            .out_descriptor = &out_plan->saveexec_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
            .out_descriptor = &out_plan->restore_exec_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, scratch_resolutions, IREE_ARRAYSIZE(scratch_resolutions),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_workgroup_scan_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->kind = kind;
  out_plan->mode = mode;
  out_plan->direction = direction;
  out_plan->wavefront_size = wavefront_size;
  out_plan->flat_workgroup_size = flat_workgroup_size;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_bpermute_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_byte_offset, loom_value_id_t low_source_value,
    loom_type_t lane_type, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_source_byte_offset,
      low_source_value,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_lane_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane, loom_type_t lane_type,
    loom_value_id_t* out_byte_offset) {
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 2, lane,
      lane_type, out_byte_offset);
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
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU subgroup scan has invalid direction");
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
      context, &plan->lds_write_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
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
      context, &plan->lds_read_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &read_op));
  *out_value = loom_value_slice_get(loom_low_op_results(read_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_scan_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_scan_plan_t* plan) {
  loom_op_t* barrier_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->barrier_descriptor, /*operands=*/NULL,
      /*operand_count=*/0, loom_make_named_attr_slice(NULL, 0),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &barrier_op);
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
      context, &plan->saveexec_descriptor, &guard, 1,
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
      context, &plan->restore_exec_descriptor, &saved_exec, 1,
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
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU subgroup scan has invalid active lane count");
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
  uint32_t identity_bits = 0;
  const bool is_exclusive =
      plan->mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE;
  if (is_exclusive) {
    if (!loom_amdgpu_collective_combine_identity_bits(plan->kind,
                                                      &identity_bits)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU subgroup scan has no identity for combining kind");
    }
    exclusive_byte_offset = source_byte_offsets[0];
    exclusive_guard = guards[0];
  } else if (plan->mode != LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU subgroup scan has invalid mode");
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];

    for (uint32_t step_index = 0; step_index < step_count; ++step_index) {
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          source_byte_offsets[step_index], accumulator, lane_type, &peer));
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
            identity_bits, lane_type, &identity));
        accumulator = identity;
      } else {
        loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->bpermute_descriptor,
            exclusive_byte_offset, accumulator, lane_type, &shifted));
        loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            identity_bits, lane_type, &identity));
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
  const loom_amdgpu_subgroup_scan_plan_t subgroup_plan = {
      .bpermute_descriptor = plan->bpermute_descriptor,
      .combine_descriptor = plan->combine_descriptor,
      .guard_descriptor = plan->guard_descriptor,
      .select_descriptor = plan->select_descriptor,
      .value = plan->value,
      .result = plan->result,
      .payload_kind = plan->payload_kind,
      .register_count = plan->register_count,
      .kind = plan->kind,
      .mode = plan->mode,
      .direction = plan->direction,
      .wavefront_size = plan->wavefront_size,
      .active_lane_count = plan->flat_workgroup_size,
  };
  if (plan->flat_workgroup_size <= plan->wavefront_size) {
    return loom_amdgpu_lower_kernel_subgroup_scan(context, source_op,
                                                  &subgroup_plan);
  }

  const bool has_partial_tail =
      (plan->flat_workgroup_size % plan->wavefront_size) != 0;
  const uint32_t wave_count =
      (plan->flat_workgroup_size + plan->wavefront_size - 1) /
      plan->wavefront_size;
  const uint32_t tail_lane_count =
      has_partial_tail ? plan->flat_workgroup_size % plan->wavefront_size
                       : plan->wavefront_size;
  if (wave_count > plan->wavefront_size) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU workgroup scan wave count exceeds one wave");
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  const uint32_t register_count = plan->register_count;
  const uint32_t scratch_byte_length = wave_count * register_count * 4u;
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

  loom_value_id_t tail_wave_guard = LOOM_VALUE_ID_INVALID;
  loom_value_id_t dynamic_active_lane_count = LOOM_VALUE_ID_INVALID;
  if (has_partial_tail) {
    loom_value_id_t tail_wave_first_linear_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        (wave_count - 1) * plan->wavefront_size, lane_type,
        &tail_wave_first_linear_id));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->lane_ge_descriptor, linear_id,
        tail_wave_first_linear_id, mask_type, &tail_wave_guard));

    if (plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE) {
      loom_value_id_t full_lane_count_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          plan->wavefront_size, lane_type, &full_lane_count_value));
      loom_value_id_t tail_lane_count_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          tail_lane_count, lane_type, &tail_lane_count_value));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
          context, source_op, &plan->select_descriptor, full_lane_count_value,
          tail_lane_count_value, tail_wave_guard, lane_type,
          &dynamic_active_lane_count));
    }
  }

  const loom_amdgpu_subgroup_scan_plan_t intra_wave_plan = {
      .bpermute_descriptor = plan->bpermute_descriptor,
      .combine_descriptor = plan->combine_descriptor,
      .guard_descriptor = plan->guard_descriptor,
      .select_descriptor = plan->select_descriptor,
      .value = plan->value,
      .result = plan->result,
      .payload_kind = plan->payload_kind,
      .register_count = register_count,
      .kind = plan->kind,
      .mode = plan->mode,
      .direction = plan->direction,
      .wavefront_size = plan->wavefront_size,
      .active_lane_count = plan->wavefront_size,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_tree(
      context, source_op, &intra_wave_plan, lane_id, lane_type, mask_type,
      dynamic_active_lane_count, result_registers));

  loom_value_id_t wave_total_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < register_count; ++i) {
    wave_total_registers[i] = result_registers[i];
    if (plan->mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor,
          wave_total_registers[i], source_registers[i], lane_type,
          &wave_total_registers[i]));
    }
  }

  loom_value_id_t subgroup_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      loom_amdgpu_subgroup_u32_log2(plan->wavefront_size), linear_id, lane_type,
      &subgroup_id));
  loom_value_id_t subgroup_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
      context, source_op, subgroup_id, 4, LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
      lane_type, &subgroup_byte_offset));

  loom_value_id_t producer_lane_value = LOOM_VALUE_ID_INVALID;
  if (has_partial_tail &&
      plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD) {
    loom_value_id_t full_producer_lane_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->wavefront_size - 1, lane_type, &full_producer_lane_value));
    loom_value_id_t tail_producer_lane_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        tail_lane_count - 1, lane_type, &tail_producer_lane_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
        context, source_op, &plan->select_descriptor, full_producer_lane_value,
        tail_producer_lane_value, tail_wave_guard, lane_type,
        &producer_lane_value));
  } else {
    const uint32_t producer_lane =
        plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD
            ? plan->wavefront_size - 1
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
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
        context, source_op, scratch_base, subgroup_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_write(
        context, source_op, plan, address, wave_total_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_restore_exec(
      context, source_op, plan, saved_exec));

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_workgroup_scan_barrier(context, source_op, plan));

  if (wave_count == 2) {
    loom_value_id_t prefix_wave_guard = tail_wave_guard;
    const bool prefix_guard_selects_second_wave = has_partial_tail;
    if (!prefix_guard_selects_second_wave) {
      loom_value_id_t wavefront_size_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          plan->wavefront_size, lane_type, &wavefront_size_value));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->lane_lt_descriptor, linear_id,
          wavefront_size_value, mask_type, &prefix_wave_guard));
    }

    uint32_t identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(plan->kind,
                                                      &identity_bits)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU workgroup scan has no identity for combining kind");
    }
    loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, identity_bits,
        lane_type, &identity));

    loom_value_id_t zero_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &zero_byte_offset));
    const uint32_t prefix_slot =
        plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD ? 0 : 1;
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = (i * wave_count + prefix_slot) * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
          context, source_op, scratch_base, zero_byte_offset,
          register_byte_offset, lane_type, &address));
      loom_value_id_t peer_wave_prefix = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_read(
          context, source_op, plan, address, lane_type, &peer_wave_prefix));
      loom_value_id_t wave_prefix = LOOM_VALUE_ID_INVALID;
      if (plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD) {
        if (prefix_guard_selects_second_wave) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
              context, source_op, &plan->select_descriptor, identity,
              peer_wave_prefix, prefix_wave_guard, lane_type, &wave_prefix));
        } else {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
              context, source_op, &plan->select_descriptor, peer_wave_prefix,
              identity, prefix_wave_guard, lane_type, &wave_prefix));
        }
      } else {
        if (prefix_guard_selects_second_wave) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
              context, source_op, &plan->select_descriptor, peer_wave_prefix,
              identity, prefix_wave_guard, lane_type, &wave_prefix));
        } else {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
              context, source_op, &plan->select_descriptor, identity,
              peer_wave_prefix, prefix_wave_guard, lane_type, &wave_prefix));
        }
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, result_registers[i],
          wave_prefix, lane_type, &result_registers[i]));
    }
    return loom_amdgpu_collective_bind_payload_result(
        context, source_op, plan->result, register_count, result_registers);
  }

  loom_value_id_t wave_count_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, wave_count,
      lane_type, &wave_count_value));
  loom_value_id_t first_wave_guard = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
      context, source_op, &plan->lane_lt_descriptor, linear_id,
      wave_count_value, mask_type, &first_wave_guard));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_saveexec(
      context, source_op, plan, first_wave_guard, mask_type, &saved_exec));

  loom_value_id_t lane_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
      context, source_op, lane_id, lane_type, &lane_byte_offset));
  loom_value_id_t wave_prefix_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
        context, source_op, scratch_base, lane_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_read(
        context, source_op, plan, address, lane_type,
        &wave_prefix_registers[i]));
  }

  const loom_amdgpu_subgroup_scan_plan_t cross_wave_plan = {
      .bpermute_descriptor = plan->bpermute_descriptor,
      .combine_descriptor = plan->combine_descriptor,
      .guard_descriptor = plan->guard_descriptor,
      .select_descriptor = plan->select_descriptor,
      .value = plan->value,
      .result = plan->result,
      .payload_kind = plan->payload_kind,
      .register_count = register_count,
      .kind = plan->kind,
      .mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE,
      .direction = plan->direction,
      .wavefront_size = plan->wavefront_size,
      .active_lane_count = wave_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_tree(
      context, source_op, &cross_wave_plan, lane_id, lane_type, mask_type,
      LOOM_VALUE_ID_INVALID, wave_prefix_registers));

  for (uint32_t i = 0; i < register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
        context, source_op, scratch_base, lane_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_write(
        context, source_op, plan, address, wave_prefix_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_restore_exec(
      context, source_op, plan, saved_exec));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_workgroup_scan_barrier(context, source_op, plan));

  for (uint32_t i = 0; i < register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_address(
        context, source_op, scratch_base, subgroup_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_scan_scratch_read(
        context, source_op, plan, address, lane_type,
        &wave_prefix_registers[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
        context, source_op, &plan->combine_descriptor, result_registers[i],
        wave_prefix_registers[i], lane_type, &result_registers[i]));
  }

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, register_count, result_registers);
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_wavefront(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key, uint32_t* out_wavefront_size) {
  *out_wavefront_size = 0;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, out_wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(*out_wavefront_size)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
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

  loom_kernel_subgroup_scan_mode_t mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_;
  if (!loom_amdgpu_map_workgroup_scan_mode(loom_kernel_workgroup_scan_mode(op),
                                           &mode)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_scan.mode"));
  }
  if (mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("workgroup_scan.identity"));
    }
  }

  loom_kernel_subgroup_scan_direction_t direction =
      LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_;
  if (!loom_amdgpu_map_workgroup_scan_direction(
          loom_kernel_workgroup_scan_direction(op), &direction)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_scan.direction"));
  }
  loom_amdgpu_descriptor_ref_t guard_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (direction) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("workgroup_scan.direction"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.wavefront_size"));
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          module, loom_target_low_legality_function(context), bundle,
          &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.fixed_workgroup_size"));
  }
  const bool has_partial_tail = flat_workgroup_size > wavefront_size &&
                                (flat_workgroup_size % wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + wavefront_size - 1) / wavefront_size;
  if (flat_workgroup_size > wavefront_size && wave_count > wavefront_size) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.wave_count"));
  }
  const uint64_t scratch_byte_length =
      (uint64_t)wave_count * unused_register_count * 4u;
  if (scratch_byte_length > UINT32_MAX) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.scratch_byte_length"));
  }
  if (mode != LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE &&
      flat_workgroup_size > wavefront_size) {
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
          .descriptor_ref = guard_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.v_cndmask_b32"),
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      },
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirements(
      context, op, requirements, IREE_ARRAYSIZE(requirements)));
  if (flat_workgroup_size > wavefront_size) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
        IREE_SV("descriptor.v_cmp_ult_u32")));
    if (has_partial_tail) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_low_legality_verify_descriptor_requirement(
              context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
              IREE_SV("descriptor.v_cmp_uge_u32")));
    }
    static const loom_amdgpu_low_legality_descriptor_requirement_t
        scratch_requirements[] = {
            {
                .constraint_key = IREE_SVL("descriptor.ds_read_b32"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
            },
            {
                .constraint_key = IREE_SVL("descriptor.ds_write_b32"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32,
            },
            {
                .constraint_key = IREE_SVL("descriptor.s_barrier"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER,
            },
            {
                .constraint_key = IREE_SVL("descriptor.s_and_saveexec_b64"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
            },
            {
                .constraint_key = IREE_SVL("descriptor.s_mov_b64_exec"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
            },
        };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_legality_verify_descriptor_requirements(
            context, op, scratch_requirements,
            IREE_ARRAYSIZE(scratch_requirements)));
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

  switch (loom_kernel_subgroup_scan_mode(op)) {
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE:
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE: {
      uint32_t unused_identity_bits = 0;
      if (!loom_amdgpu_collective_combine_identity_bits(
              kind, &unused_identity_bits)) {
        return loom_amdgpu_low_legality_reject(
            context, op, IREE_SV("subgroup_scan.identity"));
      }
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_:
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("subgroup_scan.mode"));
  }

  loom_amdgpu_descriptor_ref_t guard_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (loom_kernel_subgroup_scan_direction(op)) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("subgroup_scan.direction"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.wavefront_size"));
  }
  if (!loom_amdgpu_subgroup_full_wave_workgroups(
          module, loom_target_low_legality_function(context), bundle,
          wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.fixed_workgroup_wave_multiple"));
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
          .descriptor_ref = guard_descriptor_ref,
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
