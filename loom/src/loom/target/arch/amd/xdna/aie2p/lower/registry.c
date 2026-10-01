// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/vector/fragment.h"
#include "loom/target/arch/amd/xdna/aie2p/contracts/core.h"
#include "loom/target/arch/amd/xdna/aie2p/contracts/core_lower_rules.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/gather.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/gather.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/matrix.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/rodata.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/storage.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"

static bool loom_aie2p_source_type_supported(void* user_data,
                                             const loom_module_t* module,
                                             loom_type_t source_type) {
  (void)user_data;
  (void)module;
  if (!loom_type_is_scalar(source_type) && !loom_type_is_vector(source_type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  return element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
         element_type == LOOM_SCALAR_TYPE_F8E5M2;
}

static iree_status_t loom_aie2p_map_type(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_type_t source_type,
                                         loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_type_is_buffer(source_type)) {
    return loom_low_lower_make_register_type(
        context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1, out_low_type);
  }
  if (loom_type_is_scalar(source_type)) {
    switch (loom_type_element_type(source_type)) {
      case LOOM_SCALAR_TYPE_INDEX:
      case LOOM_SCALAR_TYPE_OFFSET:
      case LOOM_SCALAR_TYPE_I1:
      case LOOM_SCALAR_TYPE_I8:
      case LOOM_SCALAR_TYPE_I16:
      case LOOM_SCALAR_TYPE_I32:
      case LOOM_SCALAR_TYPE_F8E4M3:
      case LOOM_SCALAR_TYPE_F8E5M2:
      case LOOM_SCALAR_TYPE_F16:
      case LOOM_SCALAR_TYPE_BF16:
      case LOOM_SCALAR_TYPE_F32:
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1, out_low_type);
      case LOOM_SCALAR_TYPE_I64:
      case LOOM_SCALAR_TYPE_F64:
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 2, out_low_type);
      default:
        break;
    }
  }
  if (loom_type_is_vector(source_type) &&
      loom_type_is_all_static(source_type)) {
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(source_type, &element_count) ||
        element_count == 0) {
      return loom_low_lower_emit_source_type_unsupported(
          context, source_op, IREE_SV("source"), source_type);
    }
    const loom_scalar_type_t element_type = loom_type_element_type(source_type);
    const bool is_rank_one = loom_type_rank(source_type) == 1;
    const int32_t element_bits = loom_scalar_type_bitwidth(element_type);
    if (is_rank_one && element_count == 32 &&
        element_type == LOOM_SCALAR_TYPE_F32) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 2, out_low_type);
    }
    // Rank-one accumulator values above 1024 bits retain the containing
    // four-unit physical view. AIE2P has no allocatable three-unit MBMS view;
    // units beyond the source vector's logical extent remain unobservable.
    const bool has_accumulator_element_type =
        element_type == LOOM_SCALAR_TYPE_I32 ||
        element_type == LOOM_SCALAR_TYPE_I64 ||
        element_type == LOOM_SCALAR_TYPE_F32;
    if (is_rank_one && has_accumulator_element_type && element_bits > 0 &&
        element_count > 1024 / (uint32_t)element_bits &&
        element_count <= 2048 / (uint32_t)element_bits) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 4, out_low_type);
    }
    if (element_count <= 128 && element_type == LOOM_SCALAR_TYPE_I1) {
      const uint32_t predicate_register_count =
          (uint32_t)((element_count + 63u) / 64u);
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ELPREDICATE,
          predicate_register_count, out_low_type);
    }
    if (element_bits > 0 && element_count > 512 / (uint32_t)element_bits &&
        element_count <= 1024 / (uint32_t)element_bits) {
      // Ordinary wide vectors retain the same ordered W-register payload
      // across logical shape changes, just like single-X values below.
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 4, out_low_type);
    }
    if (element_bits > 0 && element_count <= 512 / (uint32_t)element_bits) {
      // Ordinary vectors retain a full X-register carrier. Narrow vector
      // memory forms address W subregisters of that carrier; choosing a W
      // carrier from the logical type alone makes the same SSA value unusable
      // by the 512-bit vector ALU.
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 2, out_low_type);
    }
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

// Native BFP operands carry 64 signed mantissa bytes followed by eight shared
// exponent bytes in EX. Other schemas retain their ordinary vector carrier.
static bool loom_aie2p_value_has_bfp_storage(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value) {
  const loom_value_fact_encoded_operand_schema_t native = {
      .element_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_BFP16EBS8,
      .payload_packing = LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT,
      .rounding_policy = LOOM_VALUE_FACT_ROUNDING_POLICY_FLUSH_SUBNORMAL,
      .payload_register_count = 18,
      .payload_element_count = 64,
  };
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, value);
  loom_value_fact_encoding_summary_t summary = {0};
  if (loom_value_facts_query_encoding_summary(&fact_table->context, facts,
                                              &summary)) {
    return loom_value_fact_encoded_operand_schema_equal(
        summary.storage_schema.encoded_operand, native);
  }
  loom_vector_fragment_fact_t fragment;
  return loom_vector_fragment_fact_query_value_facts(&fact_table->context,
                                                     facts, &fragment) &&
         loom_value_fact_encoded_operand_schema_equal(fragment.encoded_operand,
                                                      native);
}

static iree_status_t loom_aie2p_map_value(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_value_id,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  if (loom_type_is_vector(source_type) && loom_type_rank(source_type) == 1 &&
      loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_I8 &&
      loom_type_dim_static_size_at(source_type, 0) == 72 &&
      loom_aie2p_value_has_bfp_storage(
          loom_low_lower_context_fact_table(context), source_value_id)) {
    return loom_low_lower_make_register_type(
        context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MEXA, 1, out_low_type);
  }
  return loom_aie2p_map_type(user_data, context, source_op, source_type,
                             out_low_type);
}

static iree_status_t loom_aie2p_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t resource_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1, &resource_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = resource_type,
        .resource_import_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER,
        .resource_index = source_argument_index,
        .resource_source_type = source_type,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_aie2p_map_type(user_data, context, source_function_op,
                             source_type, &out_argument->abi_type);
}

#include "loom/target/arch/amd/xdna/aie2p/contracts/core_index.inl"

static iree_status_t loom_aie2p_preselect_op(void* user_data,
                                             loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_select_matrix_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_select_gather_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_select_rodata_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
  return loom_aie2p_select_storage_plan(context, source_op, out_plan);
}

static void loom_aie2p_mark_plan_storage_demands(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_aie2p_matrix_plan_isa(plan)) {
    loom_aie2p_mark_matrix_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_gather_plan_isa(plan)) {
    loom_aie2p_mark_gather_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_rodata_plan_isa(plan)) {
    loom_aie2p_mark_rodata_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_storage_plan_isa(plan)) {
    loom_aie2p_mark_storage_plan_demands(context, source_op, plan);
  } else {
    IREE_ASSERT_UNREACHABLE("AIE2P storage demand has unknown plan kind");
  }
}

static void loom_aie2p_describe_plan(void* user_data,
                                     loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     loom_low_lower_plan_report_t* out_report) {
  (void)user_data;
  if (loom_aie2p_matrix_plan_isa(plan)) {
    loom_aie2p_describe_matrix_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_gather_plan_isa(plan)) {
    loom_aie2p_describe_gather_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_rodata_plan_isa(plan)) {
    loom_aie2p_describe_rodata_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_storage_plan_isa(plan)) {
    loom_aie2p_describe_storage_plan(context, source_op, plan, out_report);
  } else {
    IREE_ASSERT_UNREACHABLE("AIE2P report has unknown plan kind");
  }
}

static iree_status_t loom_aie2p_emit_op(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_aie2p_matrix_plan_isa(plan)) {
    return loom_aie2p_emit_matrix_plan(context, source_op, plan);
  }
  if (loom_aie2p_gather_plan_isa(plan)) {
    return loom_aie2p_emit_gather_plan(context, source_op, plan);
  }
  if (loom_aie2p_rodata_plan_isa(plan)) {
    return loom_aie2p_emit_rodata_plan(context, source_op, plan);
  }
  if (loom_aie2p_storage_plan_isa(plan)) {
    return loom_aie2p_emit_storage_plan(context, source_op, plan);
  }
  IREE_ASSERT_UNREACHABLE("AIE2P emission has unknown plan kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_aie2p_finalize_module(
    void* user_data, loom_module_t* module,
    loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena) {
  (void)user_data;
  return loom_aie2p_finalize_gather_module(module, module_state, scratch_arena);
}

static const loom_low_lower_policy_t kAie2pCoreLowLowerPolicy = {
    .name = IREE_SVL("amd-xdna-aie2p-core-low-lower"),
    .error_catalog = &loom_xdna_error_catalog,
    .source_type_supported =
        {
            .fn = loom_aie2p_source_type_supported,
            .user_data = NULL,
        },
    .map_type = {.fn = loom_aie2p_map_type, .user_data = NULL},
    .map_value = {.fn = loom_aie2p_map_value, .user_data = NULL},
    .map_argument = {.fn = loom_aie2p_map_argument, .user_data = NULL},
    .contract = LOOM_AIE2P_CORE_CONTRACT,
    .query_op_contract =
        {
            .fn = loom_aie2p_query_gather_contract,
            .user_data = NULL,
        },
    .descriptor_matrix =
        {
            .options = loom_aie2p_descriptor_matrix_options,
            .query = loom_aie2p_descriptor_matrix_query,
            .attrs = NULL,
            .user_data = NULL,
        },
    .preselect_op = {.fn = loom_aie2p_preselect_op, .user_data = NULL},
    .mark_plan_storage_demands =
        {
            .fn = loom_aie2p_mark_plan_storage_demands,
            .user_data = NULL,
        },
    .describe_plan = {.fn = loom_aie2p_describe_plan, .user_data = NULL},
    .emit_op = {.fn = loom_aie2p_emit_op, .user_data = NULL},
    .finalize_module = {.fn = loom_aie2p_finalize_module, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_aie2p_core_low_lower_policy(void) {
  return &kAie2pCoreLowLowerPolicy;
}

void loom_aie2p_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("amd.xdna.aie2p.core"),
          .policy = &kAie2pCoreLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
