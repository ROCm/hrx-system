// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/collective_payload.h"

#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/sync.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"

uint32_t loom_amdgpu_collective_transport_register_count(loom_type_t type) {
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_set_contains(
          LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
          element_type)) {
    return 0;
  }
  if (loom_type_is_scalar(type)) {
    return (loom_scalar_type_bitwidth(element_type) + 31u) / 32u;
  }
  loom_amdgpu_vector_storage_t storage;
  return loom_amdgpu_type_vector_storage(type, &storage)
             ? storage.register_count
             : 0;
}

static loom_amdgpu_subgroup_payload_kind_t loom_amdgpu_collective_payload_kind(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  if (loom_amdgpu_type_is_i32(type)) {
    *out_register_count = 1;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR;
  }
  if (loom_amdgpu_type_is_f32(type)) {
    *out_register_count = 1;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR;
  }
  const uint32_t i32_lane_count = loom_amdgpu_vector_i32_lane_count(type);
  if (i32_lane_count != 0) {
    *out_register_count = i32_lane_count;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR;
  }
  const uint32_t f32_lane_count = loom_amdgpu_vector_f32_lane_count(type);
  if (f32_lane_count != 0) {
    *out_register_count = f32_lane_count;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR;
  }
  return LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
}

bool loom_amdgpu_collective_arithmetic_payload_is_supported(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_amdgpu_subgroup_payload_kind_t* out_kind,
    uint32_t* out_register_count) {
  *out_kind = LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  const loom_type_t type = loom_module_value_type(module, value_id);
  *out_kind = loom_amdgpu_collective_payload_kind(type, out_register_count);
  return *out_kind != LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
}

bool loom_amdgpu_collective_payload_is_integer(
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  return payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR ||
         payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR;
}

bool loom_amdgpu_collective_payload_is_float(
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  return payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR ||
         payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR;
}

bool loom_amdgpu_collective_resolve_workgroup_shape(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t partition_lane_count,
    uint32_t register_count,
    loom_amdgpu_workgroup_collective_shape_t* out_shape,
    loom_amdgpu_workgroup_collective_shape_failure_t* out_failure) {
  *out_shape = (loom_amdgpu_workgroup_collective_shape_t){0};
  *out_failure = LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE;
  if (partition_lane_count == 0 || register_count == 0) {
    *out_failure =
        LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WORKGROUP_SIZE;
    return false;
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    *out_failure =
        LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WORKGROUP_SIZE;
    return false;
  }

  const uint32_t wave_count =
      (flat_workgroup_size + partition_lane_count - 1) / partition_lane_count;
  if (flat_workgroup_size > partition_lane_count &&
      wave_count > partition_lane_count) {
    *out_failure = LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WAVE_COUNT;
    return false;
  }

  loom_amdgpu_workgroup_collective_shape_flags_t flags = 0;
  if (flat_workgroup_size > partition_lane_count) {
    flags |= LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE;
    if ((flat_workgroup_size % partition_lane_count) != 0) {
      flags |= LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_PARTIAL_TAIL;
    }
  }
  const uint64_t scratch_byte_length =
      iree_all_bits_set(flags,
                        LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE)
          ? (uint64_t)wave_count * register_count * 4u
          : 0;
  if (scratch_byte_length > UINT32_MAX) {
    *out_failure =
        LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_SCRATCH_BYTE_LENGTH;
    return false;
  }

  *out_shape = (loom_amdgpu_workgroup_collective_shape_t){
      .flat_workgroup_size = flat_workgroup_size,
      .wave_count = wave_count,
      .scratch_byte_length = (uint32_t)scratch_byte_length,
      .flags = flags,
  };
  return true;
}

iree_status_t loom_amdgpu_finalize_collective_payload(
    loom_low_lower_context_t* context, loom_low_lower_plan_t plan) {
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  const void** materialization = NULL;
  switch (plan.id) {
    case LOOM_OP_KERNEL_SUBGROUP_REDUCE: {
      loom_amdgpu_subgroup_reduce_plan_t* reduce =
          (loom_amdgpu_subgroup_reduce_plan_t*)plan.target_data;
      value = reduce->value;
      payload_kind = reduce->payload_kind;
      materialization = &reduce->payload_materialization;
      break;
    }
    case LOOM_OP_KERNEL_WORKGROUP_REDUCE: {
      loom_amdgpu_workgroup_reduce_plan_t* reduce =
          (loom_amdgpu_workgroup_reduce_plan_t*)plan.target_data;
      iree_status_t status = iree_ok_status();
      for (uint32_t i = 0;
           i < reduce->payload_count && iree_status_is_ok(status); ++i) {
        loom_amdgpu_workgroup_reduce_payload_t* payload = &reduce->payloads[i];
        if (payload->kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR) {
          status = loom_amdgpu_prepare_vgpr_literal(
              context, payload->value, &payload->payload_materialization);
        }
      }
      return status;
    }
    case LOOM_OP_KERNEL_SUBGROUP_SCAN: {
      loom_amdgpu_subgroup_scan_plan_t* scan =
          (loom_amdgpu_subgroup_scan_plan_t*)plan.target_data;
      value = scan->value;
      payload_kind = scan->payload_kind;
      materialization = &scan->payload_materialization;
      break;
    }
    case LOOM_OP_KERNEL_WORKGROUP_SCAN: {
      loom_amdgpu_workgroup_scan_plan_t* scan =
          (loom_amdgpu_workgroup_scan_plan_t*)plan.target_data;
      value = scan->value;
      payload_kind = scan->payload_kind;
      materialization = &scan->payload_materialization;
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("expected an arithmetic collective plan");
      IREE_BUILTIN_UNREACHABLE();
  }
  return payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR
             ? loom_amdgpu_prepare_vgpr_literal(context, value, materialization)
             : iree_ok_status();
}

iree_status_t loom_amdgpu_collective_lookup_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, loom_amdgpu_subgroup_payload_kind_t payload_kind,
    const void* materialization, loom_value_id_t* out_low_value) {
  switch (payload_kind) {
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR:
      return loom_amdgpu_emit_prepared_vgpr_value(
          context, source_op, value, materialization, out_low_value);
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR:
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR:
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR:
      *out_low_value = loom_low_lower_lookup_value(context, value);
      return iree_ok_status();
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("AMDGPU collective lowering requires a payload kind");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_amdgpu_collective_payload_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t register_count, loom_value_id_t low_value, uint32_t register_index,
    loom_type_t lane_type, loom_value_id_t* out_register) {
  loom_value_id_t low_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_value, register_count, register_index, lane_type,
      &low_register));
  return loom_amdgpu_materialize_full_low_vgpr_b32(context, source_op,
                                                   low_register, out_register);
}

iree_status_t loom_amdgpu_collective_bind_payload_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t register_count,
    const loom_value_id_t* result_registers) {
  return loom_amdgpu_bind_low_register_range(context, source_op, source_result,
                                             result_registers, register_count);
}

iree_status_t loom_amdgpu_collective_resolve_cross_wave_descriptors(
    loom_low_lower_context_t* context,
    loom_amdgpu_workgroup_collective_cross_wave_descriptors_t* out_descriptors,
    bool* out_present) {
  const loom_amdgpu_descriptor_resolution_t resolutions[] = {
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
          .out_descriptor = &out_descriptors->lds_read_descriptor,
      },
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32,
          .out_descriptor = &out_descriptors->lds_write_descriptor,
      },
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
          .out_descriptor = &out_descriptors->saveexec_descriptor,
      },
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
          .out_descriptor = &out_descriptors->restore_exec_descriptor,
      },
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, IREE_ARRAYSIZE(resolutions), out_present));
  if (!*out_present) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_workgroup_barrier_plan(
      context, &out_descriptors->barrier));
  *out_present = out_descriptors->barrier.kind !=
                 LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_collective_emit_cross_wave_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_collective_cross_wave_descriptors_t*
        descriptors) {
  return loom_amdgpu_lower_workgroup_barrier_plan(context, source_op,
                                                  &descriptors->barrier);
}

iree_status_t loom_amdgpu_collective_verify_cross_wave_descriptor_requirements(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  static const loom_amdgpu_low_legality_descriptor_requirement_t
      requirements[] = {
          {
              .constraint_key = IREE_SVL("descriptor.ds_read_b32"),
              .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
          },
          {
              .constraint_key = IREE_SVL("descriptor.ds_write_b32"),
              .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32,
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirements(
      context, op, requirements, IREE_ARRAYSIZE(requirements)));
  if (!loom_amdgpu_workgroup_barrier_lowering_available(
          loom_target_low_legality_descriptor_set(context))) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.workgroup_barrier"));
  }
  return iree_ok_status();
}
