// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/target/arch/spirv/contracts/logical_core.h"
#include "loom/target/arch/spirv/contracts/logical_core_lower_rules.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/error_catalog.h"
#include "loom/target/arch/spirv/lower/collective_arithmetic.h"
#include "loom/target/arch/spirv/lower/lower.h"
#include "loom/target/arch/spirv/lower/matrix.h"
#include "loom/target/arch/spirv/lower/workgroup.h"
#include "loom/target/arch/spirv/lower/workgroup_layout.h"
#include "loom/target/arch/spirv/lower/workgroup_reduce.h"
#include "loom/target/arch/spirv/ops/types.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/target/registers.h"

static iree_status_t loom_spirv_make_hal_buffer_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_string_id_t hal_buffer_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("hal.buffer"), &hal_buffer_id));
  *out_type = loom_type_dialect_opaque(hal_buffer_id);
  return iree_ok_status();
}

static iree_status_t loom_spirv_make_register_type(
    loom_low_lower_context_t* context, uint16_t register_class_id,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, register_class_id, 1,
                                           out_type);
}

static iree_status_t loom_spirv_make_typed_register_type(
    loom_low_lower_context_t* context, uint16_t register_class_id,
    loom_type_t value_type, loom_type_t* out_type) {
  return loom_low_lower_make_typed_register_type(context, register_class_id, 1,
                                                 value_type, out_type);
}

static bool loom_spirv_source_type_is_id(loom_type_t type) {
  loom_spirv_value_type_t value_type = {0};
  return loom_spirv_value_type_from_loom_type(type, &value_type);
}

static bool loom_spirv_source_type_is_offset64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_spirv_source_type_is_fp8(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_F8E4M3 ||
         scalar_type == LOOM_SCALAR_TYPE_F8E5M2;
}

static bool loom_spirv_source_type_is_bfloat16(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_BF16;
}

static bool loom_spirv_source_type_supported(void* user_data,
                                             const loom_module_t* module,
                                             loom_type_t source_type) {
  (void)user_data;
  (void)module;
  return loom_spirv_source_type_is_fp8(source_type);
}

static bool loom_spirv_bitstream_packet_carrier_supported(
    void* user_data, const loom_target_facts_t* target_facts,
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_type_t source_type) {
  (void)user_data;
  (void)module;
  (void)fact_table;
  (void)value_id;
  if (!loom_type_is_vector(source_type) || loom_type_rank(source_type) != 1 ||
      loom_type_dim_is_dynamic_at(source_type, 0)) {
    return false;
  }
  if (loom_type_dim_static_size_at(source_type, 0) == 1) {
    const loom_scalar_type_t element_type = loom_type_element_type(source_type);
    if (element_type == LOOM_SCALAR_TYPE_OFFSET) {
      return true;
    }
    source_type = loom_type_scalar(element_type);
  } else if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_BF16 &&
             !iree_all_bits_set(loom_target_facts_bundle(target_facts)
                                    ->config->contract_feature_bits,
                                LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR)) {
    // Software BF16 uses scalar i32 payloads. Only native BF16 has a vector
    // carrier; shared boundary projection splits the software form into
    // scalars.
    return false;
  }
  loom_spirv_value_type_t value_type = {0};
  return loom_spirv_value_type_from_loom_type(source_type, &value_type);
}

static const loom_target_bitstream_policy_t kSpirvBitstreamPolicy = {
    .word_bit_count = 32u,
    .packet_carrier_supported =
        {
            .fn = loom_spirv_bitstream_packet_carrier_supported,
            .user_data = NULL,
        },
};

static const uint16_t kSpirvVectorPacketBitCounts[] = {
    256u, 128u, 64u, 32u, 16u, 8u,
};
static const uint16_t kSpirvVectorPacketLaneCounts[] = {4u, 3u, 2u, 1u};
static const loom_target_vector_packet_lane_limit_t
    kSpirvVectorPacketStructuralLaneLimits[] = {
        {LOOM_SCALAR_TYPE_I1, 4u},     {LOOM_SCALAR_TYPE_I8, 4u},
        {LOOM_SCALAR_TYPE_F8E4M3, 1u}, {LOOM_SCALAR_TYPE_F8E5M2, 1u},
        {LOOM_SCALAR_TYPE_I16, 4u},    {LOOM_SCALAR_TYPE_I32, 4u},
        {LOOM_SCALAR_TYPE_INDEX, 4u},  {LOOM_SCALAR_TYPE_I64, 4u},
        {LOOM_SCALAR_TYPE_OFFSET, 4u}, {LOOM_SCALAR_TYPE_F16, 4u},
        {LOOM_SCALAR_TYPE_BF16, 4u},   {LOOM_SCALAR_TYPE_F32, 4u},
        {LOOM_SCALAR_TYPE_F64, 4u},
};
static_assert(IREE_ARRAYSIZE(kSpirvVectorPacketLaneCounts) <=
                  LOOM_TARGET_VECTOR_PACKET_LANE_COUNT_LIMIT,
              "packet lane candidates exceed the shared planner capacity");

static const loom_target_vector_packet_policy_t kSpirvVectorPacketPolicy = {
    .native_bit_counts = kSpirvVectorPacketBitCounts,
    .native_lane_counts = kSpirvVectorPacketLaneCounts,
    .index_bit_count = 32u,
    .offset_bit_count = 64u,
    .flags = LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_DYNAMIC_WINDOW_REPLAY |
             LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_MEMORY_ROOT_COMPONENTS |
             LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_DYNAMIC_SNAPSHOT_SELECTION |
             LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_BOUNDARY_PROJECTION,
    .structural_lane_limits = kSpirvVectorPacketStructuralLaneLimits,
    .maximum_unpacketized_bit_count = 256u,
    .native_bit_count_count = IREE_ARRAYSIZE(kSpirvVectorPacketBitCounts),
    .native_lane_count_count = IREE_ARRAYSIZE(kSpirvVectorPacketLaneCounts),
    .structural_lane_limit_count =
        IREE_ARRAYSIZE(kSpirvVectorPacketStructuralLaneLimits),
};

static bool loom_spirv_source_value_fragment(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source_value_id,
    loom_vector_fragment_fact_t* out_fragment) {
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_value_id);
  return loom_vector_fragment_fact_query_value_facts(&fact_table->context,
                                                     facts, out_fragment) &&
         loom_vector_fragment_fact_has_matrix_shape(*out_fragment);
}

static bool loom_spirv_source_vector_carrier_supported(
    void* user_data, const loom_target_facts_t* target_facts,
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_type_t source_type) {
  loom_vector_fragment_fact_t fragment = {0};
  return (fact_table != NULL && value_id != LOOM_VALUE_ID_INVALID &&
          loom_spirv_source_value_fragment(fact_table, value_id, &fragment)) ||
         loom_spirv_bitstream_packet_carrier_supported(
             user_data, target_facts, module, NULL, LOOM_VALUE_ID_INVALID,
             source_type);
}

static bool loom_spirv_source_function_vector_carrier_supported(
    void* user_data, const loom_target_facts_t* target_facts,
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_type_t source_type) {
  (void)user_data;
  (void)target_facts;
  (void)module;
  (void)fact_table;
  (void)value_id;
  (void)source_type;
  // Callable imports and exits have no vector ABI, including matrix fragments.
  return false;
}

static bool loom_spirv_fragment_dimension(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    uint16_t* out_dimension) {
  int64_t dimension = 0;
  if (value_id == LOOM_VALUE_ID_INVALID ||
      !loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &dimension) ||
      dimension <= 0 || dimension > UINT16_MAX) {
    return false;
  }
  *out_dimension = (uint16_t)dimension;
  return true;
}

static bool loom_spirv_fragment_role(
    loom_vector_fragment_fact_t fragment,
    loom_contract_operand_role_t* out_contract_role,
    loom_spirv_cooperative_matrix_use_t* out_use) {
  if (fragment.role_flags == LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS) {
    *out_contract_role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
    *out_use = LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR;
    return true;
  }
  if (fragment.role_flags == LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS) {
    *out_contract_role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
    *out_use = LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_BKHR;
    return true;
  }
  if (loom_vector_fragment_fact_is_accumulator_like(fragment)) {
    *out_contract_role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
    *out_use = LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_ACCUMULATOR_KHR;
    return true;
  }
  return false;
}

static bool loom_spirv_fragment_matrix_payload(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id, loom_vector_fragment_fact_t fragment,
    uint16_t* out_rows, uint16_t* out_columns,
    loom_spirv_scalar_type_t* out_component_type,
    loom_spirv_cooperative_matrix_use_t* out_use) {
  if (!loom_spirv_fragment_dimension(
          fact_table, loom_vector_fragment_fact_row_value(fragment),
          out_rows) ||
      !loom_spirv_fragment_dimension(
          fact_table, loom_vector_fragment_fact_column_value(fragment),
          out_columns)) {
    return false;
  }
  if (fragment.shape_rank == 3) {
    uint16_t block_count = 0;
    if (!loom_spirv_fragment_dimension(
            fact_table, loom_vector_fragment_fact_block_value(fragment),
            &block_count) ||
        block_count != 1) {
      return false;
    }
  }

  loom_contract_operand_role_t contract_role =
      LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
  if (!loom_spirv_fragment_role(fragment, &contract_role, out_use)) {
    return false;
  }
  loom_contract_operand_t contract_operand = {0};
  loom_contract_rejection_bits_t rejection_bits = LOOM_CONTRACT_REJECTION_NONE;
  if (!loom_contract_vector_operand_from_fragment(
          module, source_value_id, fragment, contract_role, &contract_operand,
          &rejection_bits) ||
      !loom_spirv_matrix_component_type(contract_operand.numeric_type,
                                        out_component_type)) {
    return false;
  }
  return true;
}

static iree_status_t loom_spirv_map_type(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_type_t source_type,
                                         loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_spirv_source_type_is_bfloat16(source_type) &&
      !iree_all_bits_set(
          loom_low_lower_context_bundle(context)->config->contract_feature_bits,
          LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR)) {
    return loom_spirv_make_typed_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID,
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), out_low_type);
  }
  if (loom_spirv_source_type_is_fp8(source_type)) {
    return loom_spirv_make_typed_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID,
        loom_type_scalar(LOOM_SCALAR_TYPE_I8), out_low_type);
  }
  if (loom_spirv_source_type_is_id(source_type)) {
    return loom_spirv_make_typed_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, source_type, out_low_type);
  }
  if (loom_type_is_vector(source_type) && loom_type_rank(source_type) == 1 &&
      !loom_type_dim_is_dynamic_at(source_type, 0) &&
      loom_type_dim_static_size_at(source_type, 0) == 1) {
    return loom_spirv_map_type(
        user_data, context, source_op,
        loom_type_scalar(loom_type_element_type(source_type)), out_low_type);
  }
  if (loom_spirv_source_type_is_offset64(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64, out_low_type);
  }
  if (loom_type_is_buffer(source_type) || loom_type_is_view(source_type)) {
    return loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER,
        out_low_type);
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_map_value(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_value_id,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  if (loom_type_is_view(source_type)) {
    bool is_workgroup = false;
    uint16_t register_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_spirv_resolve_workgroup_view_reg_class(
        context, source_value_id, &is_workgroup, &register_class_id));
    if (is_workgroup) {
      if (register_class_id == LOOM_LOW_REG_CLASS_NONE) {
        return iree_ok_status();
      }
      return loom_spirv_make_register_type(context, register_class_id,
                                           out_low_type);
    }
  }
  if (loom_type_is_vector(source_type)) {
    if (loom_low_lower_source_op_is_callable_exit(context, source_op)) {
      return iree_ok_status();
    }
    loom_vector_fragment_fact_t fragment = {0};
    const loom_value_fact_table_t* fact_table =
        loom_low_lower_context_fact_table(context);
    if (loom_spirv_source_value_fragment(fact_table, source_value_id,
                                         &fragment)) {
      uint16_t rows = 0;
      uint16_t columns = 0;
      loom_spirv_scalar_type_t component_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
      loom_spirv_cooperative_matrix_use_t use =
          LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MAX;
      if (loom_spirv_fragment_matrix_payload(
              loom_low_lower_context_module(context), fact_table,
              source_value_id, fragment, &rows, &columns, &component_type,
              &use)) {
        loom_type_t matrix_type = loom_type_none();
        IREE_RETURN_IF_ERROR(loom_spirv_cooperative_matrix_type_make(
            loom_low_lower_context_module(context), rows, columns,
            component_type, loom_spirv_matrix_scope(), use, &matrix_type));
        return loom_spirv_make_typed_register_type(
            context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, matrix_type,
            out_low_type);
      }
      return loom_spirv_make_register_type(
          context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, out_low_type);
    }
  }
  return loom_spirv_map_type(user_data, context, source_op, source_type,
                             out_low_type);
}

static iree_status_t loom_spirv_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  const loom_type_t source_type =
      loom_module_value_type(environment->module, source_value_id);
  if (!loom_type_is_view(source_type)) {
    return iree_ok_status();
  }

  bool is_workgroup = false;
  uint16_t register_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_spirv_resolve_workgroup_contract_view_reg_class(
      environment, source_value_id, &is_workgroup, &register_class_id));
  if (is_workgroup && register_class_id != LOOM_LOW_REG_CLASS_NONE) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        register_class_id, LOOM_LOW_REPRESENTATION_ID_NONE, 1);
  }
  return iree_ok_status();
}

static uint32_t loom_spirv_hal_binding_index(loom_low_lower_context_t* context,
                                             uint16_t source_argument_index) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &argument_count);
  uint32_t resource_index = 0;
  for (uint16_t i = 0; i < source_argument_index && i < argument_count; ++i) {
    loom_type_t type = loom_module_value_type(
        loom_low_lower_context_module(context), argument_ids[i]);
    if (loom_type_is_buffer(type)) {
      ++resource_index;
    }
  }
  return resource_index;
}

static iree_status_t loom_spirv_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)user_data;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_type_is_buffer(source_type)) {
    loom_type_t binding_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_spirv_make_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER,
        &binding_type));
    loom_type_t resource_source_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_spirv_make_hal_buffer_type(context, &resource_source_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = binding_type,
        .resource_import_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING,
        .resource_index =
            loom_spirv_hal_binding_index(context, source_argument_index),
        .resource_source_type = resource_source_type,
    };
    return iree_ok_status();
  }
  if (loom_type_is_vector(source_type)) {
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .body_type = loom_type_none(),
  };
  return loom_spirv_map_type(user_data, context, source_function_op,
                             source_type, &out_argument->abi_type);
}

static iree_status_t loom_spirv_map_abi_layout(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count, iree_arena_allocator_t* scratch_arena,
    loom_named_attr_slice_t* out_abi_layout) {
  (void)user_data;
  (void)arg_types;
  (void)result_types;
  *out_abi_layout = loom_named_attr_slice_empty();
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (layout_kind != LOOM_LOW_LOWER_ABI_LAYOUT_KIND_KERNEL ||
      bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(result_count, 0,
                 "SPIR-V HAL ABI layout reached a kernel with results");

  uint16_t parameter_count = 0;
  const loom_low_lower_abi_argument_t* argument_map =
      loom_low_lower_context_argument_map(context, &parameter_count);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &source_argument_count);
  IREE_ASSERT_EQ(source_argument_count, parameter_count,
                 "SPIR-V HAL ABI layout reached an inconsistent argument map");

  loom_type_t* parameter_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, parameter_count,
                                                 sizeof(*parameter_types),
                                                 (void**)&parameter_types));
  iree_host_size_t direct_argument_count = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < parameter_count; ++i) {
    if (argument_map[i].kind == LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE) {
      // The HAL-visible interface receives bindings as buffers even though the
      // target-low carrier is a physical storage-buffer address.
      parameter_types[i] = loom_type_buffer();
    } else {
      parameter_types[i] = loom_module_value_type(module, source_arguments[i]);
      ++direct_argument_count;
    }
  }
  IREE_ASSERT_EQ(arg_count, direct_argument_count,
                 "SPIR-V HAL ABI layout reached an inconsistent low signature");

  loom_type_t signature = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_module_intern_function_type(
      module, parameter_types, parameter_count, NULL, 0, &signature));
  loom_string_id_t signature_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("signature"), &signature_key));
  loom_named_attr_t* field = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(scratch_arena, sizeof(*field), (void**)&field));
  *field = (loom_named_attr_t){
      .name_id = signature_key,
      .value = loom_attr_type(loom_module_lookup_type_id(module, signature)),
  };
  *out_abi_layout = loom_make_named_attr_slice(field, 1);
  return iree_ok_status();
}

#include "loom/target/arch/spirv/contracts/tables.inl"

static iree_status_t loom_spirv_preselect_op(void* user_data,
                                             loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  if (loom_kernel_workgroup_reduce_isa(source_op)) {
    return loom_spirv_select_workgroup_reduce(context, source_op, out_plan);
  }
  if (loom_kernel_subgroup_reduce_isa(source_op) ||
      loom_kernel_subgroup_scan_isa(source_op)) {
    return loom_spirv_select_subgroup_extrema(context, source_op, out_plan);
  }
  return loom_spirv_select_workgroup_plan(context, source_op, out_plan);
}

static iree_status_t loom_spirv_emit_op(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_kernel_workgroup_reduce_isa(source_op)) {
    return loom_spirv_emit_workgroup_reduce(context, source_op, plan);
  }
  if (loom_kernel_subgroup_reduce_isa(source_op) ||
      loom_kernel_subgroup_scan_isa(source_op)) {
    return loom_spirv_emit_subgroup_extrema(context, source_op, plan);
  }
  return loom_spirv_lower_workgroup_op(context, source_op, plan);
}

static iree_status_t loom_spirv_describe_plan(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan, bool is_elided,
    uint64_t execution_count_plus_one,
    loom_low_lower_plan_report_t* out_report) {
  (void)user_data;
  (void)context;
  (void)plan;
  (void)is_elided;
  (void)execution_count_plus_one;
  if (loom_kernel_workgroup_reduce_isa(source_op)) {
    out_report->plan_key = IREE_SV("workgroup_subgroup_tree");
  }
  if (loom_kernel_subgroup_reduce_isa(source_op) ||
      loom_kernel_subgroup_scan_isa(source_op)) {
    out_report->plan_key = IREE_SV("integer_float_extrema");
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_entry_setup(
    void* user_data, loom_low_lower_context_t* context) {
  (void)user_data;
  return loom_spirv_workgroup_layout_emit_storage_roots(context);
}

static iree_status_t loom_spirv_plan_entry_setup(
    void* user_data, loom_low_lower_context_t* context) {
  (void)user_data;
  return loom_spirv_plan_workgroup_entry_setup(context);
}

static void loom_spirv_mark_plan_storage_demands(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_kernel_workgroup_reduce_isa(source_op)) {
    for (uint16_t i = 0; i < source_op->operand_count; ++i) {
      loom_low_lower_require_source_value_storage(
          context, loom_op_const_operands(source_op)[i]);
    }
    return;
  }
  if (loom_kernel_subgroup_reduce_isa(source_op)) {
    loom_low_lower_require_source_value_storage(
        context, loom_kernel_subgroup_reduce_value(source_op));
    return;
  }
  if (loom_kernel_subgroup_scan_isa(source_op)) {
    loom_low_lower_require_source_value_storage(
        context, loom_kernel_subgroup_scan_value(source_op));
    return;
  }
  loom_spirv_mark_workgroup_plan_storage_demands(context, source_op, plan);
}

static const loom_low_lower_policy_t kSpirvLowLowerPolicy = {
    .name = IREE_SVL("spirv-logical-lower"),
    .error_catalog = &loom_spirv_error_catalog,
    .vector_packet_policy = &kSpirvVectorPacketPolicy,
    .bitstream_policy = &kSpirvBitstreamPolicy,
    .source_vector_carrier_supported =
        {.fn = loom_spirv_source_vector_carrier_supported, .user_data = NULL},
    .source_function_vector_carrier_supported =
        {.fn = loom_spirv_source_function_vector_carrier_supported,
         .user_data = NULL},
    .map_abi_layout = {.fn = loom_spirv_map_abi_layout, .user_data = NULL},
    .map_type = {.fn = loom_spirv_map_type, .user_data = NULL},
    .map_value = {.fn = loom_spirv_map_value, .user_data = NULL},
    .map_contract_value = {.fn = loom_spirv_map_contract_value,
                           .user_data = NULL},
    .map_argument = {.fn = loom_spirv_map_argument, .user_data = NULL},
    .source_type_supported = {.fn = loom_spirv_source_type_supported,
                              .user_data = NULL},
    .entry_setup = {.plan = loom_spirv_plan_entry_setup,
                    .emit = loom_spirv_emit_entry_setup,
                    .user_data = NULL},
    .contract = LOOM_SPIRV_LOGICAL_CONTRACT,
    .source_memory_root_byte_offset =
        {
            .fn = loom_spirv_workgroup_layout_source_memory_root_byte_offset,
            .user_data = NULL,
        },
    .descriptor_matrix =
        {
            .options = loom_spirv_descriptor_matrix_options,
            .query = loom_spirv_descriptor_matrix_query,
            .attrs = NULL,
            .user_data = NULL,
        },
    .preselect_op = {.fn = loom_spirv_preselect_op, .user_data = NULL},
    .describe_plan = {.fn = loom_spirv_describe_plan, .user_data = NULL},
    .mark_plan_storage_demands =
        {
            .fn = loom_spirv_mark_plan_storage_demands,
            .user_data = NULL,
        },
    .emit_op = {.fn = loom_spirv_emit_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_spirv_low_lower_policy(void) {
  return &kSpirvLowLowerPolicy;
}

void loom_spirv_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("spirv.logical.core"),
          .policy = &kSpirvLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
