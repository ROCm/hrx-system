// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"

#include <stdint.h>

#include "loom/ir/attribute.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_AMDGPU_FRAGMENT_VIEW_RANK = 2,
  LOOM_AMDGPU_FRAGMENT_LANE_MODULUS = 16,
  LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT = 4,
};

typedef struct loom_amdgpu_fragment_memory_environment_t {
  // Source module being checked or lowered.
  const loom_module_t* module;
  // Source facts available for shape, view, and address reasoning.
  const loom_value_fact_table_t* fact_table;
  // Target bundle selected for this source-to-low attempt.
  const loom_target_bundle_t* bundle;
  // Low descriptor set selected by the target bundle.
  const loom_low_descriptor_set_t* descriptor_set;
  // Source function owning the fragment movement op.
  loom_func_like_t source_function;
} loom_amdgpu_fragment_memory_environment_t;

typedef struct loom_amdgpu_fragment_memory_source_t {
  // Source vector fragment role.
  loom_vector_role_t vector_role;
  // View value read or written by the fragment movement op.
  loom_value_id_t view;
  // Vector payload result for loads or stored payload for stores.
  loom_value_id_t payload;
  // Source fragment row count value.
  loom_value_id_t rows;
  // Source fragment column count value.
  loom_value_id_t columns;
  // Static index array spelling the base view indices.
  loom_attribute_t static_indices;
  // Dynamic index operands referenced by the static index sentinel slots.
  loom_value_slice_t dynamic_indices;
  // Optional cache scope attr on the source op.
  loom_attribute_t cache_scope;
  // Optional cache temporal attr on the source op.
  loom_attribute_t cache_temporal;
} loom_amdgpu_fragment_memory_source_t;

typedef struct loom_amdgpu_fragment_memory_diagnostic_t {
  // Stable constraint key identifying the first failed representation contract.
  iree_string_view_t constraint_key;
} loom_amdgpu_fragment_memory_diagnostic_t;

typedef struct loom_amdgpu_fragment_lane_ids_t {
  // Full subgroup lane id.
  loom_value_id_t lane;
  // Lane id modulo the fragment row/column tile.
  loom_value_id_t lane_mod;
  // Lane id divided by the fragment row/column tile.
  loom_value_id_t lane_div;
} loom_amdgpu_fragment_lane_ids_t;

typedef struct loom_amdgpu_fragment_memory_address_t {
  // Low packet address operand after static offset immediates are split out.
  loom_value_id_t low_vaddr;
  // Encoded descriptor offset immediate value.
  int64_t immediate_offset;
} loom_amdgpu_fragment_memory_address_t;

typedef enum loom_amdgpu_fragment_memory_address_register_kind_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR,
} loom_amdgpu_fragment_memory_address_register_kind_t;

typedef struct loom_amdgpu_fragment_memory_address_accumulator_t {
  // Low scalar register containing the accumulated byte address.
  loom_value_id_t value;
  // Register class of value.
  loom_amdgpu_fragment_memory_address_register_kind_t register_kind;
} loom_amdgpu_fragment_memory_address_accumulator_t;

static bool loom_amdgpu_fragment_memory_reject(
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic,
    iree_string_view_t constraint_key) {
  if (diagnostic != NULL &&
      iree_string_view_is_empty(diagnostic->constraint_key)) {
    diagnostic->constraint_key = constraint_key;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_role_from_vector_role(
    loom_vector_role_t role, loom_contract_operand_role_t* out_role) {
  *out_role = LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
  switch (role) {
    case LOOM_VECTOR_ROLE_LHS:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
      return true;
    case LOOM_VECTOR_ROLE_RHS:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
      return true;
    case LOOM_VECTOR_ROLE_INIT:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
      return true;
    case LOOM_VECTOR_ROLE_RESULT:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_RESULT;
      return true;
    case LOOM_VECTOR_ROLE_COUNT_:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_exact_nonnegative_i64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  return loom_amdgpu_value_facts_as_exact_non_negative_i64(
      loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

static bool loom_amdgpu_fragment_memory_shape_matches(
    const loom_value_fact_table_t* fact_table,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, loom_value_id_t rows,
    loom_value_id_t columns) {
  int64_t row_count = 0;
  int64_t column_count = 0;
  if (!loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, rows,
                                                         &row_count) ||
      !loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, columns,
                                                         &column_count)) {
    return false;
  }

  const loom_amdgpu_matrix_tile_shape_t shape = layout->tile_shape;
  switch (role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
      return row_count == shape.result_row_count &&
             column_count == shape.reduction_count;
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      return row_count == shape.reduction_count &&
             column_count == shape.result_column_count;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      return row_count == shape.result_row_count &&
             column_count == shape.result_column_count;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_can_narrow_result_store(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t storage_element_type) {
  return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE &&
         (role == LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
          role == LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR) &&
         expected_element_type == LOOM_SCALAR_TYPE_F32 &&
         storage_element_type == LOOM_SCALAR_TYPE_BF16;
}

static bool loom_amdgpu_fragment_memory_payload_matches(
    loom_type_t payload_type,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (role_layout == NULL || !loom_type_is_vector(payload_type) ||
      loom_type_rank(payload_type) != 1 ||
      !loom_type_is_all_static(payload_type)) {
    return false;
  }

  switch (role_layout->role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
    case LOOM_CONTRACT_OPERAND_ROLE_RHS: {
      if (loom_type_element_type(payload_type) != expected_element_type ||
          !loom_scalar_type_is_float(expected_element_type)) {
        return false;
      }
      const int32_t element_bit_count =
          loom_scalar_type_bitwidth(expected_element_type);
      if (element_bit_count != role_layout->element_bit_count) {
        return false;
      }
      if (element_bit_count == 32) {
        return loom_amdgpu_vector_f32_lane_count(payload_type) ==
               role_layout->register_count;
      }
      if (element_bit_count != 16) {
        return false;
      }
      uint32_t payload_bit_count = 0;
      uint32_t register_count = 0;
      return loom_amdgpu_type_packed_16bit_float_storage(
                 payload_type, &payload_bit_count, &register_count) &&
             register_count == role_layout->register_count &&
             payload_bit_count == (uint32_t)role_layout->register_count * 32u;
    }
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT: {
      if (expected_element_type != LOOM_SCALAR_TYPE_F32) {
        return false;
      }
      const loom_scalar_type_t payload_element_type =
          loom_type_element_type(payload_type);
      if (payload_element_type == expected_element_type) {
        return loom_amdgpu_vector_f32_lane_count(payload_type) ==
               role_layout->register_count;
      }
      if (!loom_amdgpu_fragment_memory_can_narrow_result_store(
              operation_kind, role_layout->role, expected_element_type,
              payload_element_type)) {
        return false;
      }
      uint32_t payload_bit_count = 0;
      uint32_t register_count = 0;
      return loom_amdgpu_type_packed_16bit_float_storage(
                 payload_type, &payload_bit_count, &register_count) &&
             payload_bit_count == (uint32_t)role_layout->register_count * 16u &&
             register_count ==
                 ((uint32_t)role_layout->register_count + 1u) / 2u;
    }
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_scalar_type_from_numeric(
    loom_amdgpu_matrix_numeric_type_t numeric_type,
    loom_scalar_type_t* out_element_type) {
  *out_element_type = LOOM_SCALAR_TYPE_COUNT_;
  switch (numeric_type) {
    case LOOM_AMDGPU_MATRIX_NUMERIC_F16:
      *out_element_type = LOOM_SCALAR_TYPE_F16;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_BF16:
      *out_element_type = LOOM_SCALAR_TYPE_BF16;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_F32:
      *out_element_type = LOOM_SCALAR_TYPE_F32;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_descriptor_payload(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role,
    loom_amdgpu_matrix_payload_shape_t* out_payload) {
  *out_payload = (loom_amdgpu_matrix_payload_shape_t){0};
  switch (role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
      *out_payload = descriptor->lhs_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      *out_payload = descriptor->rhs_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
      *out_payload = descriptor->accumulator_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      *out_payload = descriptor->result_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_descriptor_role_element_type(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role, loom_scalar_type_t* out_element_type) {
  loom_amdgpu_matrix_payload_shape_t payload = {0};
  return loom_amdgpu_fragment_memory_descriptor_payload(descriptor, role,
                                                        &payload) &&
         loom_amdgpu_fragment_memory_scalar_type_from_numeric(
             payload.numeric_type, out_element_type);
}

static bool loom_amdgpu_fragment_memory_payload_element_matches(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_type_t payload_type,
    loom_scalar_type_t expected_element_type) {
  if (!loom_type_is_vector(payload_type)) {
    return true;
  }
  const loom_scalar_type_t payload_element_type =
      loom_type_element_type(payload_type);
  return payload_element_type == expected_element_type ||
         loom_amdgpu_fragment_memory_can_narrow_result_store(
             operation_kind, role, expected_element_type, payload_element_type);
}

static bool loom_amdgpu_fragment_memory_target_layout(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    loom_contract_operand_role_t role,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_type_t payload_type,
    const loom_amdgpu_matrix_fragment_layout_t** out_layout,
    loom_scalar_type_t* out_expected_element_type,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  *out_layout = NULL;
  *out_expected_element_type = LOOM_SCALAR_TYPE_COUNT_;
  if (environment->bundle == NULL || environment->bundle->snapshot == NULL ||
      environment->descriptor_set == NULL) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.target_layout"));
  }

  const iree_host_size_t descriptor_count =
      loom_amdgpu_matrix_contract_descriptor_count();
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    if (layout == NULL ||
        environment->bundle->snapshot->subgroup_size != layout->wave_size ||
        !loom_amdgpu_descriptor_set_has_ref(environment->descriptor_set,
                                            descriptor->low_descriptor_ref)) {
      continue;
    }
    loom_scalar_type_t expected_element_type = LOOM_SCALAR_TYPE_COUNT_;
    if (!loom_amdgpu_fragment_memory_descriptor_role_element_type(
            descriptor, role, &expected_element_type)) {
      continue;
    }
    if (!loom_amdgpu_fragment_memory_payload_element_matches(
            operation_kind, role, payload_type, expected_element_type)) {
      continue;
    }
    *out_layout = layout;
    *out_expected_element_type = expected_element_type;
    return true;
  }

  return loom_amdgpu_fragment_memory_reject(
      diagnostic, IREE_SV("fragment_memory.target_layout"));
}

static bool loom_amdgpu_fragment_memory_view_matches(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_type_t view_type, loom_type_t payload_type,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    loom_vector_memory_access_t* out_access) {
  *out_access = (loom_vector_memory_access_t){0};
  if (!loom_type_is_view(view_type) ||
      loom_type_rank(view_type) != LOOM_AMDGPU_FRAGMENT_VIEW_RANK ||
      !loom_type_is_all_static(view_type)) {
    return false;
  }
  const loom_scalar_type_t view_element_type =
      loom_type_element_type(view_type);
  const loom_scalar_type_t payload_element_type =
      loom_type_element_type(payload_type);
  const bool view_is_narrowed =
      loom_amdgpu_fragment_memory_can_narrow_result_store(
          operation_kind, role_layout->role, expected_element_type,
          view_element_type);
  if (loom_type_is_vector(payload_type) &&
      payload_element_type != view_element_type &&
      !(view_is_narrowed && payload_element_type == expected_element_type)) {
    return false;
  }

  switch (role_layout->role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      if (!loom_scalar_type_is_float(expected_element_type) ||
          loom_scalar_type_bitwidth(expected_element_type) !=
              role_layout->element_bit_count) {
        return false;
      }
      break;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      if (expected_element_type != LOOM_SCALAR_TYPE_F32) {
        return false;
      }
      if (view_element_type != expected_element_type && !view_is_narrowed) {
        return false;
      }
      break;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
  if (view_element_type != expected_element_type && !view_is_narrowed) {
    return false;
  }

  loom_type_t scalar_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, view_element_type,
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  const loom_fact_context_t* fact_context =
      fact_table ? &fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, module, view_type,
                                          scalar_vector_type, out_access)) {
    return false;
  }
  return out_access->static_element_byte_count > 0 &&
         out_access->static_element_byte_count <= UINT16_MAX &&
         (out_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_DENSE ||
          out_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);
}

static bool loom_amdgpu_fragment_memory_fill_view_strides(
    const loom_vector_memory_access_t* access,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    uint32_t* out_axis_byte_strides,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  for (uint8_t axis = 0; axis < access->view_rank; ++axis) {
    int64_t element_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(access, axis,
                                                      &element_stride) ||
        element_stride <= 0) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.view_stride"));
    }
    int64_t byte_stride = 0;
    if (!iree_checked_mul_i64(element_stride, access->static_element_byte_count,
                              &byte_stride) ||
        byte_stride <= 0 || byte_stride > UINT32_MAX) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.view_stride"));
    }
    out_axis_byte_strides[axis] = (uint32_t)byte_stride;
  }

  if (role_layout->elements_per_register > 1) {
    uint8_t packed_axis = UINT8_MAX;
    switch (role_layout->map_kind) {
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
        packed_axis = 1;
        break;
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
        packed_axis = 0;
        break;
      default:
        return loom_amdgpu_fragment_memory_reject(
            diagnostic, IREE_SV("fragment_memory.layout_map"));
    }
    if (packed_axis >= access->view_rank ||
        out_axis_byte_strides[packed_axis] !=
            (uint32_t)access->static_element_byte_count) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.packed_axis_stride"));
    }
  }

  return true;
}

static bool loom_amdgpu_fragment_memory_fill_origins(
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    loom_amdgpu_fragment_origin_plan_t* out_origins) {
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != LOOM_AMDGPU_FRAGMENT_VIEW_RANK) {
    return false;
  }

  uint16_t dynamic_ordinal = 0;
  for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
    out_origins[axis] = (loom_amdgpu_fragment_origin_plan_t){
        .dynamic_index = LOOM_VALUE_ID_INVALID,
        .static_index = 0,
    };
    const int64_t static_index = static_indices.i64_array[axis];
    if (static_index == INT64_MIN) {
      if (dynamic_ordinal >= dynamic_indices.count) {
        return false;
      }
      out_origins[axis].dynamic_index =
          loom_value_slice_get(dynamic_indices, dynamic_ordinal++);
    } else if (static_index < 0) {
      return false;
    } else {
      out_origins[axis].static_index = static_index;
    }
  }
  return dynamic_ordinal == dynamic_indices.count;
}

static bool loom_amdgpu_fragment_memory_view_base_value(
    const loom_module_t* module, loom_value_id_t view_value_id,
    loom_value_id_t* out_byte_offset) {
  *out_byte_offset = LOOM_VALUE_ID_INVALID;
  while (view_value_id < module->values.count) {
    const loom_value_t* view_value = loom_module_value(module, view_value_id);
    if (loom_value_is_block_arg(view_value)) {
      return false;
    }
    const loom_op_t* defining_op = loom_value_def_op(view_value);
    if (defining_op == NULL) {
      return false;
    }
    if (loom_view_refine_isa(defining_op)) {
      view_value_id = loom_view_refine_source(defining_op);
      continue;
    }
    if (loom_buffer_view_isa(defining_op)) {
      *out_byte_offset = loom_buffer_view_byte_offset(defining_op);
      return true;
    }
    return false;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_descriptor_ref(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space, uint16_t packet_register_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      if (operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD) {
        switch (packet_register_count) {
          case 1:
            *out_descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR;
            return true;
          case 2:
            *out_descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR;
            return true;
          case 4:
            *out_descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B128_SADDR;
            return true;
          default:
            return false;
        }
      }
      switch (packet_register_count) {
        case 1:
          *out_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR;
          return true;
        case 2:
          *out_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR;
          return true;
        case 4:
          *out_descriptor_ref =
              LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B128_SADDR;
          return true;
        default:
          return false;
      }
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      if (operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD) {
        switch (packet_register_count) {
          case 1:
            *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32;
            return true;
          case 2:
            *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64;
            return true;
          case 3:
            *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B96;
            return true;
          case 4:
            *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128;
            return true;
          default:
            return false;
        }
      }
      switch (packet_register_count) {
        case 1:
          *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32;
          return true;
        case 2:
          *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B64;
          return true;
        case 3:
          *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B96;
          return true;
        case 4:
          *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B128;
          return true;
        default:
          return false;
      }
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_16bit_store_descriptor_ref(
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B16_SADDR;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return false;
  }
}

static void loom_amdgpu_fragment_memory_source_from_op(
    const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_source_t* out_source) {
  *out_source = (loom_amdgpu_fragment_memory_source_t){
      .vector_role = LOOM_VECTOR_ROLE_COUNT_,
      .view = LOOM_VALUE_ID_INVALID,
      .payload = LOOM_VALUE_ID_INVALID,
      .rows = LOOM_VALUE_ID_INVALID,
      .columns = LOOM_VALUE_ID_INVALID,
      .static_indices = loom_attr_absent(),
      .dynamic_indices = {0},
      .cache_scope = loom_attr_absent(),
      .cache_temporal = loom_attr_absent(),
  };
  if (operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD) {
    out_source->vector_role = loom_vector_fragment_load_role(source_op);
    out_source->view = loom_vector_fragment_load_view(source_op);
    out_source->payload = loom_vector_fragment_load_result(source_op);
    out_source->rows = loom_vector_fragment_load_rows(source_op);
    out_source->columns = loom_vector_fragment_load_columns(source_op);
    out_source->static_indices =
        loom_vector_fragment_load_static_indices(source_op);
    out_source->dynamic_indices = loom_vector_fragment_load_indices(source_op);
    out_source->cache_scope = loom_op_attrs(
        source_op)[loom_vector_fragment_load_cache_scope_ATTR_INDEX];
    out_source->cache_temporal = loom_op_attrs(
        source_op)[loom_vector_fragment_load_cache_temporal_ATTR_INDEX];
    return;
  }

  out_source->vector_role = loom_vector_fragment_store_role(source_op);
  out_source->view = loom_vector_fragment_store_view(source_op);
  out_source->payload = loom_vector_fragment_store_value(source_op);
  out_source->rows = loom_vector_fragment_store_rows(source_op);
  out_source->columns = loom_vector_fragment_store_columns(source_op);
  out_source->static_indices =
      loom_vector_fragment_store_static_indices(source_op);
  out_source->dynamic_indices = loom_vector_fragment_store_indices(source_op);
  out_source->cache_scope = loom_op_attrs(
      source_op)[loom_vector_fragment_store_cache_scope_ATTR_INDEX];
  out_source->cache_temporal = loom_op_attrs(
      source_op)[loom_vector_fragment_store_cache_temporal_ATTR_INDEX];
}

static loom_value_id_t loom_amdgpu_fragment_memory_narrowed_result_round_source(
    const loom_module_t* module, loom_value_id_t payload,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (payload >= module->values.count || role_layout == NULL) {
    return LOOM_VALUE_ID_INVALID;
  }
  const loom_type_t payload_type = loom_module_value_type(module, payload);
  if (loom_type_element_type(payload_type) == LOOM_SCALAR_TYPE_F32 &&
      loom_amdgpu_vector_f32_lane_count(payload_type) ==
          role_layout->register_count) {
    return payload;
  }

  const loom_value_t* payload_value = loom_module_value(module, payload);
  const loom_op_t* defining_op = loom_value_def_op(payload_value);
  if (defining_op == NULL || !loom_vector_fptrunc_isa(defining_op) ||
      loom_vector_fptrunc_result(defining_op) != payload) {
    return LOOM_VALUE_ID_INVALID;
  }

  const loom_value_id_t input = loom_vector_fptrunc_input(defining_op);
  if (input >= module->values.count) {
    return LOOM_VALUE_ID_INVALID;
  }
  const loom_type_t input_type = loom_module_value_type(module, input);
  if (loom_type_element_type(input_type) != LOOM_SCALAR_TYPE_F32 ||
      loom_amdgpu_vector_f32_lane_count(input_type) !=
          role_layout->register_count) {
    return LOOM_VALUE_ID_INVALID;
  }
  return input;
}

static bool loom_amdgpu_fragment_memory_analyze(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    const loom_amdgpu_fragment_memory_source_t* source,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  if (out_plan != NULL) {
    *out_plan = (loom_amdgpu_fragment_memory_plan_t){0};
  }
  if (!loom_kernel_def_isa(environment->source_function.op)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.kernel_entry"));
  }
  if (!loom_attr_is_absent(source->cache_scope) ||
      !loom_attr_is_absent(source->cache_temporal)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.cache_policy"));
  }

  loom_contract_operand_role_t role = LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
  if (!loom_amdgpu_fragment_memory_role_from_vector_role(source->vector_role,
                                                         &role)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.role"));
  }
  const loom_type_t payload_type =
      loom_module_value_type(environment->module, source->payload);
  const loom_amdgpu_matrix_fragment_layout_t* layout = NULL;
  loom_scalar_type_t expected_element_type = LOOM_SCALAR_TYPE_COUNT_;
  if (!loom_amdgpu_fragment_memory_target_layout(
          environment, role, operation_kind, payload_type, &layout,
          &expected_element_type, diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_fragment_memory_shape_matches(environment->fact_table,
                                                 layout, role, source->rows,
                                                 source->columns)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.shape"));
  }

  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, role);
  if (role_layout == NULL ||
      role_layout->register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.role_layout"));
  }
  if (!loom_amdgpu_fragment_memory_payload_matches(
          payload_type, operation_kind, expected_element_type, role_layout)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.payload"));
  }

  loom_vector_memory_access_t access = {0};
  const loom_type_t view_type =
      loom_module_value_type(environment->module, source->view);
  if (!loom_amdgpu_fragment_memory_view_matches(
          environment->module, environment->fact_table, view_type, payload_type,
          operation_kind, expected_element_type, role_layout, &access)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.view"));
  }
  uint32_t axis_byte_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  if (!loom_amdgpu_fragment_memory_fill_view_strides(
          &access, role_layout, axis_byte_strides, diagnostic)) {
    return false;
  }
  const bool narrowed_result_store =
      loom_amdgpu_fragment_memory_can_narrow_result_store(
          operation_kind, role, expected_element_type,
          loom_type_element_type(view_type));
  const loom_value_id_t narrowed_result_round_source =
      narrowed_result_store
          ? loom_amdgpu_fragment_memory_narrowed_result_round_source(
                environment->module, source->payload, role_layout)
          : LOOM_VALUE_ID_INVALID;

  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &environment->fact_table->context,
          loom_value_fact_table_lookup(environment->fact_table, source->view),
          &view_reference)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.view_reference"));
  }
  int64_t base_byte_offset = 0;
  loom_value_id_t dynamic_base_byte_offset = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_value_facts_as_exact_non_negative_i64(
          view_reference.base_byte_offset, &base_byte_offset)) {
    if (!loom_amdgpu_fragment_memory_view_base_value(
            environment->module, source->view, &dynamic_base_byte_offset)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.base_offset"));
    }
  }
  if (view_reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    uint64_t root_byte_offset = 0;
    if (!loom_amdgpu_source_lds_layout_lookup_root(
            environment->fact_table, environment->source_function,
            view_reference.root_value_id, &root_byte_offset) ||
        root_byte_offset > INT64_MAX) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.workgroup_root"));
    }
    if (!iree_checked_add_i64(base_byte_offset, (int64_t)root_byte_offset,
                              &base_byte_offset)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.base_offset"));
    }
  }

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (narrowed_result_store) {
    if (!loom_amdgpu_fragment_memory_16bit_store_descriptor_ref(
            view_reference.memory_space, &descriptor_ref)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.memory_space"));
    }
  } else if (!loom_amdgpu_fragment_memory_descriptor_ref(
                 operation_kind, view_reference.memory_space,
                 /*packet_register_count=*/1, &descriptor_ref)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.memory_space"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(environment->descriptor_set,
                                          descriptor_ref)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.packet"));
  }

  loom_amdgpu_fragment_origin_plan_t
      origins[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  if (!loom_amdgpu_fragment_memory_fill_origins(
          source->static_indices, source->dynamic_indices, origins)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.indices"));
  }

  if (out_plan != NULL) {
    *out_plan = (loom_amdgpu_fragment_memory_plan_t){
        .operation_kind = operation_kind,
        .role = role,
        .layout_kind = layout->kind,
        .view = source->view,
        .payload = source->payload,
        .memory_space = view_reference.memory_space,
        .root_value_id = view_reference.root_value_id,
        .alias_scope_id = view_reference.alias_scope_id,
        .base_byte_offset = (uint64_t)base_byte_offset,
        .dynamic_base_byte_offset = dynamic_base_byte_offset,
        .view_rank = access.view_rank,
        .register_count = role_layout->register_count,
        .elements_per_register = role_layout->elements_per_register,
        .element_byte_count = (uint16_t)access.static_element_byte_count,
        .narrowed_result_store = narrowed_result_store,
        .narrowed_result_round_source = narrowed_result_round_source,
    };
    for (uint8_t axis = 0; axis < access.view_rank; ++axis) {
      out_plan->origins[axis] = origins[axis];
      out_plan->axis_byte_strides[axis] = axis_byte_strides[axis];
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_fragment_memory_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_fragment_memory_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_amdgpu_fragment_memory_environment_t environment = {
      .module = module,
      .fact_table = loom_low_lower_context_fact_table(context),
      .bundle = loom_low_lower_context_bundle(context),
      .descriptor_set = loom_low_lower_context_descriptor_set(context),
      .source_function = loom_low_lower_context_source_function(context),
  };
  loom_amdgpu_fragment_memory_source_t source = {0};
  loom_amdgpu_fragment_memory_source_from_op(source_op, operation_kind,
                                             &source);
  if (!loom_amdgpu_fragment_memory_analyze(&environment, &source,
                                           operation_kind, out_plan,
                                           /*diagnostic=*/NULL)) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_fragment_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_fragment_memory_select(context, source_op,
                                            LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
                                            out_plan, out_selected);
}

iree_status_t loom_amdgpu_select_vector_fragment_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_fragment_memory_select(context, source_op,
                                            LOOM_AMDGPU_MEMORY_OPERATION_STORE,
                                            out_plan, out_selected);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_fragment_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_memory_operation_kind_t operation_kind =
      LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
  if (op->kind == LOOM_OP_VECTOR_FRAGMENT_STORE) {
    operation_kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE;
  } else if (op->kind != LOOM_OP_VECTOR_FRAGMENT_LOAD) {
    *out_handled = false;
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_amdgpu_fragment_memory_environment_t environment = {
      .module = module,
      .fact_table = loom_target_low_legality_fact_table(context),
      .bundle = bundle,
      .descriptor_set = loom_target_low_legality_descriptor_set(context),
      .source_function = loom_target_low_legality_function(context),
  };
  loom_amdgpu_fragment_memory_source_t source = {0};
  loom_amdgpu_fragment_memory_source_from_op(op, operation_kind, &source);
  loom_amdgpu_fragment_memory_diagnostic_t diagnostic = {0};
  if (loom_amdgpu_fragment_memory_analyze(&environment, &source, operation_kind,
                                          /*out_plan=*/NULL, &diagnostic)) {
    return iree_ok_status();
  }
  iree_string_view_t constraint_key = diagnostic.constraint_key;
  if (iree_string_view_is_empty(constraint_key)) {
    constraint_key = IREE_SV("fragment_memory");
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static iree_status_t loom_amdgpu_fragment_memory_low_register_kind(
    loom_low_lower_context_t* context, loom_value_id_t low_value,
    loom_amdgpu_fragment_memory_address_register_kind_t* out_register_kind) {
  *out_register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE;
  if (low_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 1) {
    *out_register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR;
    return iree_ok_status();
  }

  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr && loom_low_register_type_unit_count(low_type) == 1) {
    *out_register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "AMDGPU fragment memory address selected a non-scalar register");
}

static iree_status_t loom_amdgpu_emit_fragment_memory_add_address_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_term,
    loom_amdgpu_fragment_memory_address_register_kind_t term_register_kind,
    loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  if (term_register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    return iree_ok_status();
  }
  if (inout_accumulator->register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    *inout_accumulator = (loom_amdgpu_fragment_memory_address_accumulator_t){
        .value = low_term,
        .register_kind = term_register_kind,
    };
    return iree_ok_status();
  }
  if (inout_accumulator->register_kind ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR &&
      term_register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        inout_accumulator->value, low_term, sgpr_type,
        &inout_accumulator->value));
    return iree_ok_status();
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  if (inout_accumulator->register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR) {
    if (term_register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      low_lhs = low_term;
      low_rhs = inout_accumulator->value;
    } else {
      low_lhs = inout_accumulator->value;
      low_rhs = low_term;
    }
  } else {
    low_lhs = inout_accumulator->value;
    low_rhs = low_term;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, low_lhs,
      low_rhs, vgpr_type, &inout_accumulator->value));
  inout_accumulator->register_kind =
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_dynamic_base_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_type_t sgpr_type,
    loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  if (plan->dynamic_base_byte_offset == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  loom_value_id_t low_base_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, plan->dynamic_base_byte_offset, &low_base_offset));
  loom_amdgpu_fragment_memory_address_register_kind_t register_kind =
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_low_register_kind(
      context, low_base_offset, &register_kind));
  return loom_amdgpu_emit_fragment_memory_add_address_term(
      context, source_op, low_base_offset, register_kind, sgpr_type, vgpr_type,
      inout_accumulator);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_dynamic_origin_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_type_t sgpr_type,
    loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  for (uint8_t axis = 0; axis < plan->view_rank; ++axis) {
    const loom_amdgpu_fragment_origin_plan_t* origin = &plan->origins[axis];
    if (origin->dynamic_index == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, origin->dynamic_index, &low_index));
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    loom_amdgpu_fragment_memory_address_register_kind_t register_kind =
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_low_register_kind(
        context, low_index, &register_kind));
    if (register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_scale_u32(
          context, source_op, low_index, plan->axis_byte_strides[axis],
          sgpr_type, &low_term));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
          context, source_op, low_index, plan->axis_byte_strides[axis],
          LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &low_term));
      register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term, register_kind, sgpr_type, vgpr_type,
        inout_accumulator));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_memory_static_origin_offset(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    uint64_t* inout_static_byte_offset) {
  if (*inout_static_byte_offset > INT64_MAX) {
    return false;
  }
  int64_t static_byte_offset = (int64_t)*inout_static_byte_offset;
  for (uint8_t axis = 0; axis < plan->view_rank; ++axis) {
    const loom_amdgpu_fragment_origin_plan_t* origin = &plan->origins[axis];
    if (origin->dynamic_index != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    int64_t axis_offset = 0;
    if (!iree_checked_mul_i64(origin->static_index,
                              plan->axis_byte_strides[axis], &axis_offset) ||
        !iree_checked_add_i64(static_byte_offset, axis_offset,
                              &static_byte_offset)) {
      return false;
    }
  }
  *inout_static_byte_offset = (uint64_t)static_byte_offset;
  return true;
}

static bool loom_amdgpu_fragment_memory_scale_stride_u32(
    uint32_t factor, uint32_t byte_stride, uint32_t* out_byte_stride) {
  const uint64_t scaled_byte_stride = (uint64_t)factor * byte_stride;
  if (scaled_byte_stride > UINT32_MAX) {
    return false;
  }
  *out_byte_stride = (uint32_t)scaled_byte_stride;
  return true;
}

static bool loom_amdgpu_fragment_memory_register_terms(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint32_t* out_lane_mod_stride, uint32_t* out_lane_div_stride,
    uint64_t* out_static_byte_offset) {
  const loom_amdgpu_matrix_tile_shape_t shape = layout->tile_shape;
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  *out_lane_mod_stride = 0;
  *out_lane_div_stride = 0;
  *out_static_byte_offset = 0;
  if (role_layout == NULL) {
    return false;
  }
  switch (role_layout->map_kind) {
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[0];
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[1];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      *out_lane_div_stride = plan->axis_byte_strides[0];
      *out_static_byte_offset =
          (uint64_t)register_index *
          (shape.result_row_count / plan->register_count) *
          plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[0];
      if (!loom_amdgpu_fragment_memory_scale_stride_u32(
              (uint32_t)plan->register_count * plan->elements_per_register,
              plan->axis_byte_strides[1], out_lane_div_stride)) {
        return false;
      }
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[1];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      if (!loom_amdgpu_fragment_memory_scale_stride_u32(
              (uint32_t)plan->register_count * plan->elements_per_register,
              plan->axis_byte_strides[0], out_lane_div_stride)) {
        return false;
      }
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      if (!loom_amdgpu_fragment_memory_scale_stride_u32(
              plan->register_count, plan->axis_byte_strides[0],
              out_lane_div_stride)) {
        return false;
      }
      *out_static_byte_offset =
          (uint64_t)register_index * plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_descriptor_offset_info(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_descriptor_offset_immediate_info_t* out_info) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const uint32_t descriptor_ordinal =
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return false;
  }
  if (loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          out_info)) {
    return true;
  }
  return loom_amdgpu_descriptor_offset_immediate_info(
      descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED,
      out_info);
}

static bool loom_amdgpu_fragment_memory_register_group_is_contiguous(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t register_count, uint32_t register_byte_count) {
  if (register_count == 0 ||
      register_index + register_count > plan->register_count) {
    return false;
  }

  uint32_t base_lane_mod_stride = 0;
  uint32_t base_lane_div_stride = 0;
  uint64_t base_static_byte_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, register_index, &base_lane_mod_stride,
          &base_lane_div_stride, &base_static_byte_offset)) {
    return false;
  }
  for (uint16_t i = 1; i < register_count; ++i) {
    uint32_t lane_mod_stride = 0;
    uint32_t lane_div_stride = 0;
    uint64_t static_byte_offset = 0;
    if (!loom_amdgpu_fragment_memory_register_terms(
            layout, plan, register_index + i, &lane_mod_stride,
            &lane_div_stride, &static_byte_offset)) {
      return false;
    }
    if (lane_mod_stride != base_lane_mod_stride ||
        lane_div_stride != base_lane_div_stride ||
        static_byte_offset !=
            base_static_byte_offset + (uint64_t)i * register_byte_count) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_select_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t* out_packet_register_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_packet_register_count = 0;
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  static const uint16_t kCandidates[] = {4, 3, 2, 1};
  const uint16_t remaining = plan->register_count - register_index;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCandidates); ++i) {
    const uint16_t candidate = kCandidates[i];
    if (candidate > remaining) {
      continue;
    }
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_descriptor_ref(
            plan->operation_kind, plan->memory_space, candidate,
            &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) ||
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            layout, plan, register_index, candidate,
            LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT)) {
      continue;
    }
    *out_packet_register_count = candidate;
    *out_descriptor_ref = descriptor_ref;
    return true;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
    loom_value_fact_memory_space_t memory_space, uint16_t result_register_count,
    uint16_t* out_packet_register_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_packet_register_count = 0;
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (result_register_count == 1) {
    *out_packet_register_count = 1;
    return loom_amdgpu_fragment_memory_16bit_store_descriptor_ref(
        memory_space, out_descriptor_ref);
  }
  if ((result_register_count & 1u) != 0) {
    return false;
  }
  *out_packet_register_count = result_register_count / 2u;
  return loom_amdgpu_fragment_memory_descriptor_ref(
      LOOM_AMDGPU_MEMORY_OPERATION_STORE, memory_space,
      *out_packet_register_count, out_descriptor_ref);
}

static bool loom_amdgpu_fragment_memory_can_use_packed_payload_slice(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t result_register_count) {
  return plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID ||
         result_register_count == 1 ||
         (((register_index | result_register_count) & 1u) == 0);
}

static bool loom_amdgpu_fragment_memory_select_narrowed_store_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t* out_result_register_count, uint16_t* out_packet_register_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_result_register_count = 0;
  *out_packet_register_count = 0;
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  static const uint16_t kCandidates[] = {8, 6, 4, 2, 1};
  const uint16_t remaining = plan->register_count - register_index;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCandidates); ++i) {
    const uint16_t candidate = kCandidates[i];
    if (candidate > remaining ||
        !loom_amdgpu_fragment_memory_can_use_packed_payload_slice(
            plan, register_index, candidate)) {
      continue;
    }
    uint16_t packet_register_count = 0;
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
            plan->memory_space, candidate, &packet_register_count,
            &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) ||
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            layout, plan, register_index, candidate,
            plan->element_byte_count)) {
      continue;
    }
    *out_result_register_count = candidate;
    *out_packet_register_count = packet_register_count;
    *out_descriptor_ref = descriptor_ref;
    return true;
  }
  return false;
}

static iree_status_t loom_amdgpu_fragment_memory_packet_type(
    loom_low_lower_context_t* context, uint16_t packet_register_count,
    loom_type_t vgpr_type, loom_type_t* out_type) {
  if (packet_register_count == 1) {
    *out_type = vgpr_type;
    return iree_ok_status();
  }
  return loom_amdgpu_make_vgpr_range_type(context, packet_register_count,
                                          out_type);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_16bit_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_payload, uint16_t payload_register_count,
    uint16_t lane_index, loom_type_t vgpr_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint16_t source_register_index = lane_index / 2u;
  if (source_register_index >= payload_register_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU narrowed fragment store lane exceeds payload register count");
  }

  loom_value_id_t source_register = low_payload;
  if (payload_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_payload, source_register_index, vgpr_type,
        &source_register));
  }
  if ((lane_index & 1u) == 0) {
    *out_lane = source_register;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      source_register, vgpr_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f32_to_16bit_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t source_register_count,
    uint16_t register_index, loom_type_t vgpr_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_register = low_source;
  if (source_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, register_index, vgpr_type,
        &source_register));
  }
  return loom_amdgpu_emit_f32_to_bf16_lane(context, source_op, source_register,
                                           vgpr_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f32_pair_to_packed_16bit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t register_index, loom_type_t vgpr_type,
    loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_low_slice(context, source_op, low_source, register_index,
                                 vgpr_type, &low_source_register));
  loom_value_id_t high_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source, register_index + 1u, vgpr_type,
      &high_source_register));
  return loom_amdgpu_emit_f32_pair_to_packed_bf16(
      context, source_op, low_source_register, high_source_register, vgpr_type,
      out_packed);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_payload,
    uint16_t register_index, uint16_t result_register_count,
    uint16_t packet_register_count, loom_type_t vgpr_type,
    loom_value_id_t* out_packet) {
  *out_packet = LOOM_VALUE_ID_INVALID;
  if (result_register_count == 1) {
    if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_emit_fragment_memory_f32_to_16bit_lane(
          context, source_op, low_payload, plan->register_count, register_index,
          vgpr_type, out_packet);
    }
    const uint16_t payload_register_count = (plan->register_count + 1u) / 2u;
    return loom_amdgpu_emit_fragment_memory_packed_16bit_lane(
        context, source_op, low_payload, payload_register_count, register_index,
        vgpr_type, out_packet);
  }

  if (plan->narrowed_result_round_source == LOOM_VALUE_ID_INVALID) {
    loom_type_t packet_type = vgpr_type;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
        context, packet_register_count, vgpr_type, &packet_type));
    const uint16_t payload_register_count = (plan->register_count + 1u) / 2u;
    if (register_index == 0 &&
        packet_register_count == payload_register_count) {
      *out_packet = low_payload;
      return iree_ok_status();
    }
    return loom_amdgpu_emit_low_slice(context, source_op, low_payload,
                                      register_index / 2u, packet_type,
                                      out_packet);
  }

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
      0};
  for (uint16_t i = 0; i < packet_register_count; ++i) {
    const uint16_t source_register_index = register_index + i * 2u;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_f32_pair_to_packed_16bit(
            context, source_op, low_payload, source_register_index, vgpr_type,
            &packed_registers[i]));
  }
  if (packet_register_count == 1) {
    *out_packet = packed_registers[0];
    return iree_ok_status();
  }

  loom_type_t packet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
      context, packet_register_count, vgpr_type, &packet_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), packed_registers,
      packet_register_count, packet_type, source_op->location, &concat_op));
  *out_packet = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static void loom_amdgpu_fragment_memory_split_static_offset(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_fact_memory_space_t memory_space, uint64_t static_byte_offset,
    uint64_t* out_vaddr_static_byte_offset, int64_t* out_immediate_offset) {
  *out_vaddr_static_byte_offset = static_byte_offset;
  *out_immediate_offset = 0;
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info = {0};
  if (!loom_amdgpu_fragment_memory_descriptor_offset_info(
          context, descriptor_ref, &offset_info) ||
      offset_info.unit_byte_count == 0) {
    return;
  }

  if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if ((static_byte_offset % offset_info.unit_byte_count) != 0) {
      return;
    }
    const uint64_t encoded_offset =
        static_byte_offset / offset_info.unit_byte_count;
    if (encoded_offset > offset_info.unsigned_max ||
        encoded_offset > INT64_MAX) {
      return;
    }
    *out_vaddr_static_byte_offset = 0;
    *out_immediate_offset = (int64_t)encoded_offset;
    return;
  }

  if (offset_info.unsigned_max == UINT64_MAX) {
    return;
  }
  const uint64_t window_unit_count = offset_info.unsigned_max + 1;
  if (offset_info.unit_byte_count > UINT64_MAX / window_unit_count) {
    return;
  }
  const uint64_t window_byte_count =
      window_unit_count * offset_info.unit_byte_count;
  if (window_byte_count == 0) {
    return;
  }
  const uint64_t window_base =
      (static_byte_offset / window_byte_count) * window_byte_count;
  const uint64_t window_offset = static_byte_offset - window_base;
  const uint64_t encoded_offset = window_offset / offset_info.unit_byte_count;
  if (encoded_offset > offset_info.unsigned_max || encoded_offset > INT64_MAX) {
    return;
  }
  const uint64_t immediate_byte_offset =
      encoded_offset * offset_info.unit_byte_count;
  *out_vaddr_static_byte_offset =
      window_base + (window_offset - immediate_byte_offset);
  *out_immediate_offset = (int64_t)encoded_offset;
}

static iree_status_t loom_amdgpu_emit_fragment_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_fragment_lane_ids_t* lane_ids,
    loom_value_id_t low_resource, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_t* out_address) {
  *out_address = (loom_amdgpu_fragment_memory_address_t){
      .low_vaddr = LOOM_VALUE_ID_INVALID,
      .immediate_offset = 0,
  };
  uint32_t lane_mod_stride = 0;
  uint32_t lane_div_stride = 0;
  if (plan->base_byte_offset > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU fragment memory base exceeds i64");
  }
  int64_t static_byte_offset_i64 = (int64_t)plan->base_byte_offset;
  uint64_t register_static_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, register_index, &lane_mod_stride, &lane_div_stride,
          &register_static_offset)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU fragment memory register map is invalid");
  }
  if (register_static_offset > INT64_MAX ||
      !iree_checked_add_i64(static_byte_offset_i64,
                            (int64_t)register_static_offset,
                            &static_byte_offset_i64)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU fragment memory byte offset exceeds i64");
  }
  uint64_t static_byte_offset = (uint64_t)static_byte_offset_i64;
  if (!loom_amdgpu_fragment_memory_static_origin_offset(plan,
                                                        &static_byte_offset) ||
      static_byte_offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU fragment memory byte offset exceeds u32");
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_amdgpu_fragment_memory_address_accumulator_t accumulator = {
      .value = LOOM_VALUE_ID_INVALID,
      .register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_dynamic_base_term(
      context, source_op, plan, sgpr_type, vgpr_type, &accumulator));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_dynamic_origin_terms(
      context, source_op, plan, sgpr_type, vgpr_type, &accumulator));

  if (lane_mod_stride != 0 && lane_div_stride != 0 &&
      (uint64_t)lane_mod_stride * LOOM_AMDGPU_FRAGMENT_LANE_MODULUS ==
          lane_div_stride) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, lane_ids->lane, lane_mod_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        &accumulator));
    lane_mod_stride = 0;
    lane_div_stride = 0;
  }
  if (lane_div_stride != 0) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, lane_ids->lane_div, lane_div_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        &accumulator));
  }
  if (lane_mod_stride != 0) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, lane_ids->lane_mod, lane_mod_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        &accumulator));
  }

  loom_amdgpu_fragment_memory_split_static_offset(
      context, descriptor_ref, plan->memory_space, static_byte_offset,
      &static_byte_offset, &out_address->immediate_offset);

  if (static_byte_offset != 0) {
    if (accumulator.register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
      return loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          (uint32_t)static_byte_offset, vgpr_type, &out_address->low_vaddr);
    }
    if (accumulator.register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          (uint32_t)static_byte_offset, sgpr_type, &low_static_offset));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
          accumulator.value, low_static_offset, sgpr_type, &accumulator.value));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
          accumulator.value, (uint32_t)static_byte_offset, vgpr_type,
          &accumulator.value));
    }
  }

  if (accumulator.register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, &out_address->low_vaddr);
  }
  return loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, accumulator.value, &out_address->low_vaddr);
}

static loom_low_memory_space_t loom_amdgpu_fragment_memory_low_space(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return LOOM_LOW_MEMORY_SPACE_GLOBAL;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return LOOM_LOW_MEMORY_SPACE_GENERIC;
  }
}

static iree_status_t loom_amdgpu_record_fragment_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  loom_low_memory_access_summary_t summary = {
      .memory_space = loom_amdgpu_fragment_memory_low_space(plan->memory_space),
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
  };
  if (summary.memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC) {
    summary.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE;
  }
  if (plan->alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    summary.alias_root_id = plan->alias_scope_id;
    summary.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  }
  return loom_low_lower_record_memory_access_summary(context, low_op, &summary);
}

static iree_status_t loom_amdgpu_make_fragment_memory_attrs(
    loom_low_lower_context_t* context, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, int64_t immediate_offset,
    iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  return loom_amdgpu_append_i64_attr(context, IREE_SV("offset"),
                                     immediate_offset, attrs, attr_capacity,
                                     out_attr_count);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_lane_ids(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t vgpr_type, loom_amdgpu_fragment_lane_ids_t* out_lane_ids) {
  *out_lane_ids = (loom_amdgpu_fragment_lane_ids_t){
      .lane = LOOM_VALUE_ID_INVALID,
      .lane_mod = LOOM_VALUE_ID_INVALID,
      .lane_div = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, vgpr_type, &out_lane_ids->lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      out_lane_ids->lane, LOOM_AMDGPU_FRAGMENT_LANE_MODULUS - 1, vgpr_type,
      &out_lane_ids->lane_mod));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 4,
      out_lane_ids->lane, vgpr_type, &out_lane_ids->lane_div);
}

static iree_status_t loom_amdgpu_emit_fragment_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_type_t result_type,
    const loom_amdgpu_fragment_memory_address_t* address,
    loom_value_id_t low_resource, loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_fragment_memory_attrs(
      context, attrs, IREE_ARRAYSIZE(attrs), address->immediate_offset,
      &attr_count));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_descriptor_has_implicit_resource_operand(
          loom_low_lower_context_descriptor_set(context),
          descriptor.descriptor)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_m0_u32(context, source_op, &descriptor, 0, &low_m0));
  }

  loom_value_id_t operands[3] = {0};
  iree_host_size_t operand_count = 0;
  operands[operand_count++] = address->low_vaddr;
  if (plan->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    operands[operand_count++] = low_resource;
  }
  if (low_m0 != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = low_m0;
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_record_fragment_memory_packet(context, low_op, plan));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_store_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_fragment_memory_address_t* address,
    loom_value_id_t low_payload_register, loom_value_id_t low_resource) {
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_fragment_memory_attrs(
      context, attrs, IREE_ARRAYSIZE(attrs), address->immediate_offset,
      &attr_count));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_descriptor_has_implicit_resource_operand(
          loom_low_lower_context_descriptor_set(context),
          descriptor.descriptor)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_m0_u32(context, source_op, &descriptor, 0, &low_m0));
  }

  loom_value_id_t operands[4] = {0};
  iree_host_size_t operand_count = 0;
  operands[operand_count++] = address->low_vaddr;
  operands[operand_count++] = low_payload_register;
  if (plan->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    operands[operand_count++] = low_resource;
  }
  if (low_m0 != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = low_m0;
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  return loom_amdgpu_record_fragment_memory_packet(context, low_op, plan);
}

iree_status_t loom_amdgpu_lower_vector_fragment_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  if (layout == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU fragment memory plan has no layout");
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_amdgpu_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, vgpr_type, &lane_ids));

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  if (plan->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, plan->view, &low_resource));
  }

  loom_value_id_t low_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  uint16_t packet_count = 0;
  uint16_t register_index = 0;
  while (register_index < plan->register_count) {
    uint16_t packet_register_count = 0;
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_select_packet(
            loom_low_lower_context_descriptor_set(context), layout, plan,
            register_index, &packet_register_count, &descriptor_ref)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU fragment memory has no packet");
    }
    loom_type_t packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
        context, packet_register_count, vgpr_type, &packet_type));
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, register_index, descriptor_ref,
        &lane_ids, low_resource, vgpr_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, plan, descriptor_ref, packet_type, &address,
        low_resource, &low_packets[packet_count++]));
    register_index += packet_register_count;
  }

  if (packet_count == 1) {
    return loom_low_lower_bind_value(context, plan->payload, low_packets[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, plan->register_count, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_packets, packet_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->payload,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_vector_fragment_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  if (layout == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU fragment memory plan has no layout");
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_amdgpu_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, vgpr_type, &lane_ids));

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  if (plan->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, plan->view, &low_resource));
  }
  if (plan->narrowed_result_store) {
    loom_value_id_t low_payload = LOOM_VALUE_ID_INVALID;
    if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->narrowed_result_round_source, &low_payload));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->payload, &low_payload));
    }

    uint16_t register_index = 0;
    while (register_index < plan->register_count) {
      uint16_t result_register_count = 0;
      uint16_t packet_register_count = 0;
      loom_amdgpu_descriptor_ref_t descriptor_ref =
          LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
      if (!loom_amdgpu_fragment_memory_select_narrowed_store_packet(
              loom_low_lower_context_descriptor_set(context), layout, plan,
              register_index, &result_register_count, &packet_register_count,
              &descriptor_ref)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "AMDGPU narrowed fragment store has no packet");
      }
      loom_value_id_t low_payload_packet = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
          context, source_op, plan, low_payload, register_index,
          result_register_count, packet_register_count, vgpr_type,
          &low_payload_packet));
      loom_amdgpu_fragment_memory_address_t address;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
          context, source_op, layout, plan, register_index, descriptor_ref,
          &lane_ids, low_resource, vgpr_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
          context, source_op, plan, descriptor_ref, &address,
          low_payload_packet, low_resource));
      register_index += result_register_count;
    }
    return iree_ok_status();
  }

  loom_value_id_t low_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->payload, &low_payload));

  uint16_t register_index = 0;
  while (register_index < plan->register_count) {
    uint16_t packet_register_count = 0;
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_select_packet(
            loom_low_lower_context_descriptor_set(context), layout, plan,
            register_index, &packet_register_count, &descriptor_ref)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU fragment memory has no packet");
    }
    loom_type_t packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
        context, packet_register_count, vgpr_type, &packet_type));
    loom_value_id_t low_payload_packet = low_payload;
    if (packet_register_count != plan->register_count) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_payload, register_index, packet_type,
          &low_payload_packet));
    }
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, register_index, descriptor_ref,
        &lane_ids, low_resource, vgpr_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
        context, source_op, plan, descriptor_ref, &address, low_payload_packet,
        low_resource));
    register_index += packet_register_count;
  }
  return iree_ok_status();
}
