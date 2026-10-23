// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/fragment.h"
#include "loom/target/arch/amd/xdna/aie2p/contracts/core.h"
#include "loom/target/arch/amd/xdna/aie2p/contracts/core_lower_rules.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/gather.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/f32_representation.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/fifo_state.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/gather.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/interleave.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/matrix.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/rodata.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/shuffle.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/transpose.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/vector_construct.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/vector_load.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/fifo.h"
#include "loom/target/arch/amd/xdna/aie2p/vector_carrier.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"
#include "loom/target/contract.h"

static iree_status_t loom_aie2p_query_op_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_aie2p_query_fifo_state_contract(
      user_data, environment, source_op, out_result));
  if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_query_vector_load_contract(
      user_data, environment, source_op, out_result));
  if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_query_vector_construct_contract(
      user_data, environment, source_op, out_result));
  if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_query_interleave_contract(
      user_data, environment, source_op, out_result));
  if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_query_transpose_contract(
      user_data, environment, source_op, out_result));
  if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  return loom_aie2p_query_gather_contract(user_data, environment, source_op,
                                          out_result);
}

static bool loom_aie2p_source_type_supported(void* user_data,
                                             const loom_module_t* module,
                                             loom_type_t source_type) {
  (void)user_data;
  if (loom_aie2p_load_fifo_type_isa(module, source_type)) {
    return true;
  }
  if (!loom_type_is_scalar(source_type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  return element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
         element_type == LOOM_SCALAR_TYPE_F8E5M2;
}

static bool loom_aie2p_source_vector_carrier_supported(
    void* user_data, const loom_target_facts_t* target_facts,
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_type_t source_type) {
  (void)user_data;
  (void)target_facts;
  (void)module;
  (void)fact_table;
  (void)value_id;
  return loom_aie2p_vector_carrier_for_type(source_type).kind !=
         LOOM_AIE2P_VECTOR_CARRIER_NONE;
}

static iree_status_t loom_aie2p_map_type(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_type_t source_type,
                                         loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_aie2p_load_fifo_type_isa(loom_low_lower_context_module(context),
                                    source_type)) {
    return loom_low_lower_make_register_type(
        context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ELDFIFOREG, 1, out_low_type);
  }
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
  if (loom_type_is_vector(source_type)) {
    const loom_aie2p_vector_carrier_t carrier =
        loom_aie2p_vector_carrier_for_type(source_type);
    switch (carrier.kind) {
      case LOOM_AIE2P_VECTOR_CARRIER_ORDINARY:
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, carrier.unit_count,
            out_low_type);
      case LOOM_AIE2P_VECTOR_CARRIER_PREDICATE:
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ELPREDICATE,
            carrier.unit_count, out_low_type);
      case LOOM_AIE2P_VECTOR_CARRIER_ACCUMULATOR:
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, carrier.unit_count,
            out_low_type);
      case LOOM_AIE2P_VECTOR_CARRIER_NONE:
      default:
        break;
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
  if (loom_type_is_view(source_type)) {
    return loom_low_lower_make_register_type(
        context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1, out_low_type);
  }
  if (loom_aie2p_f32_packet_type(source_type)) {
    loom_low_representation_id_t representation =
        LOOM_LOW_REPRESENTATION_ID_NONE;
    loom_low_lower_representation_lookup_if_ready(context, source_value_id,
                                                  &representation);
    if (representation == LOOM_AIE2P_F32_REPRESENTATION_ACCUMULATOR) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EBMLL, 1, out_low_type);
    }
    IREE_ASSERT(representation == LOOM_LOW_REPRESENTATION_ID_NONE ||
                representation == LOOM_AIE2P_F32_REPRESENTATION_VECTOR);
  }
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
  const loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  const bool is_direct_helper = loom_func_like_inline_policy(source_function) ==
                                LOOM_INLINE_POLICY_NOINLINE;
  if (loom_type_is_view(source_type) && is_direct_helper) {
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = loom_type_none(),
        .body_type = loom_type_none(),
    };
    return loom_low_lower_make_register_type(
        context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1, &out_argument->abi_type);
  }
  if (loom_type_is_buffer(source_type) && !is_direct_helper) {
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
      .body_type = loom_type_none(),
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
      loom_aie2p_select_fifo_state_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_select_vector_construct_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_select_interleave_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_select_transpose_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_select_shuffle_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
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
  return loom_aie2p_select_rodata_plan(context, source_op, out_plan);
}

static void loom_aie2p_mark_plan_storage_demands(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_aie2p_fifo_state_plan_isa(plan)) {
    return;
  } else if (loom_aie2p_vector_load_plan_isa(plan)) {
    loom_aie2p_mark_vector_load_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_vector_construct_plan_isa(plan)) {
    loom_low_lower_require_source_operands_storage(context, source_op);
  } else if (loom_aie2p_matrix_plan_isa(plan)) {
    loom_aie2p_mark_matrix_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_interleave_plan_isa(plan)) {
    loom_aie2p_mark_interleave_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_transpose_plan_isa(plan)) {
    loom_aie2p_mark_transpose_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_shuffle_plan_isa(plan)) {
    loom_aie2p_mark_shuffle_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_gather_plan_isa(plan)) {
    loom_aie2p_mark_gather_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_rodata_plan_isa(plan)) {
    loom_aie2p_mark_rodata_plan_demands(context, source_op, plan);
  } else {
    IREE_ASSERT_UNREACHABLE("AIE2P storage demand has unknown plan kind");
  }
}

static iree_status_t loom_aie2p_describe_plan(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan, bool is_elided,
    uint64_t execution_count_plus_one,
    loom_low_lower_plan_report_t* out_report) {
  (void)user_data;
  (void)is_elided;
  (void)execution_count_plus_one;
  if (loom_aie2p_fifo_state_plan_isa(plan)) {
    loom_aie2p_describe_fifo_state_plan(out_report);
  } else if (loom_aie2p_vector_load_plan_isa(plan)) {
    loom_aie2p_describe_vector_load_plan(out_report);
  } else if (loom_aie2p_vector_construct_plan_isa(plan)) {
    loom_aie2p_describe_vector_construct_plan(out_report);
  } else if (loom_aie2p_matrix_plan_isa(plan)) {
    loom_aie2p_describe_matrix_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_interleave_plan_isa(plan)) {
    loom_aie2p_describe_interleave_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_transpose_plan_isa(plan)) {
    loom_aie2p_describe_transpose_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_shuffle_plan_isa(plan)) {
    loom_aie2p_describe_shuffle_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_gather_plan_isa(plan)) {
    loom_aie2p_describe_gather_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_rodata_plan_isa(plan)) {
    loom_aie2p_describe_rodata_plan(context, source_op, plan, out_report);
  } else {
    IREE_ASSERT_UNREACHABLE("AIE2P report has unknown plan kind");
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_finalize_plan(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_aie2p_shuffle_plan_isa(plan)) {
    return loom_aie2p_finalize_shuffle_plan(context, source_op, plan);
  }
  if (loom_aie2p_transpose_plan_isa(plan)) {
    return loom_aie2p_finalize_transpose_plan(context, source_op, plan);
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_emit_op(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_aie2p_fifo_state_plan_isa(plan)) {
    return loom_aie2p_emit_fifo_state_plan(context, source_op, plan);
  }
  if (loom_aie2p_vector_load_plan_isa(plan)) {
    return loom_aie2p_emit_vector_load_plan(context, source_op, plan);
  }
  if (loom_aie2p_vector_construct_plan_isa(plan)) {
    return loom_aie2p_emit_vector_construct_plan(context, source_op, plan);
  }
  if (loom_aie2p_matrix_plan_isa(plan)) {
    return loom_aie2p_emit_matrix_plan(context, source_op, plan);
  }
  if (loom_aie2p_interleave_plan_isa(plan)) {
    return loom_aie2p_emit_interleave_plan(context, source_op, plan);
  }
  if (loom_aie2p_transpose_plan_isa(plan)) {
    return loom_aie2p_emit_transpose_plan(context, source_op, plan);
  }
  if (loom_aie2p_shuffle_plan_isa(plan)) {
    return loom_aie2p_emit_shuffle_plan(context, source_op, plan);
  }
  if (loom_aie2p_gather_plan_isa(plan)) {
    return loom_aie2p_emit_gather_plan(context, source_op, plan);
  }
  if (loom_aie2p_rodata_plan_isa(plan)) {
    return loom_aie2p_emit_rodata_plan(context, source_op, plan);
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

static const uint16_t kAie2pVectorPacketBitCounts[] = {128u, 256u, 512u};
static const uint16_t kAie2pVectorPacketLaneCounts[] = {64u, 32u, 16u, 8u};
static const loom_target_vector_packet_lane_limit_t
    kAie2pVectorPacketStructuralLaneLimits[] = {
        {
            .element_type = LOOM_SCALAR_TYPE_I1,
            .maximum_lane_count = 64u,
        },
};
static_assert(IREE_ARRAYSIZE(kAie2pVectorPacketLaneCounts) <=
                  LOOM_TARGET_VECTOR_PACKET_LANE_COUNT_LIMIT,
              "packet lane candidates exceed the shared planner capacity");

static const loom_target_vector_packet_policy_t kAie2pVectorPacketPolicy = {
    .native_bit_counts = kAie2pVectorPacketBitCounts,
    .native_lane_counts = kAie2pVectorPacketLaneCounts,
    .index_bit_count = LOOM_AIE2P_INDEX_CARRIER_BIT_COUNT,
    .offset_bit_count = LOOM_AIE2P_OFFSET_CARRIER_BIT_COUNT,
    .structural_lane_limits = kAie2pVectorPacketStructuralLaneLimits,
    .maximum_unpacketized_bit_count = 0,
    .native_bit_count_count = IREE_ARRAYSIZE(kAie2pVectorPacketBitCounts),
    .native_lane_count_count = IREE_ARRAYSIZE(kAie2pVectorPacketLaneCounts),
    .structural_lane_limit_count =
        IREE_ARRAYSIZE(kAie2pVectorPacketStructuralLaneLimits),
};

static const loom_target_bitstream_policy_t kAie2pBitstreamPolicy = {
    .word_bit_count = 32u,
    .packet_carrier_supported =
        {
            .fn = loom_aie2p_source_vector_carrier_supported,
            .user_data = NULL,
        },
};

static const loom_low_lower_function_storage_mapping_t kAie2pFunctionStorage[] =
    {
        {LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, LOOM_STORAGE_SPACE_PRIVATE,
         AIE2P_CORE_REG_CLASS_ID_AIE2P_EP},
        {LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, LOOM_STORAGE_SPACE_WORKGROUP,
         AIE2P_CORE_REG_CLASS_ID_AIE2P_EP},
};

static const loom_low_lower_policy_t kAie2pCoreLowLowerPolicy = {
    .name = IREE_SVL("amd-xdna-aie2p-core-low-lower"),
    .error_catalog = &loom_xdna_error_catalog,
    .vector_packet_policy = &kAie2pVectorPacketPolicy,
    .bitstream_policy = &kAie2pBitstreamPolicy,
    .function_storage = {kAie2pFunctionStorage,
                         IREE_ARRAYSIZE(kAie2pFunctionStorage)},
    .source_type_supported =
        {
            .fn = loom_aie2p_source_type_supported,
            .user_data = NULL,
        },
    .source_vector_carrier_supported =
        {
            .fn = loom_aie2p_source_vector_carrier_supported,
            .user_data = NULL,
        },
    .map_type = {.fn = loom_aie2p_map_type, .user_data = NULL},
    .map_value = {.fn = loom_aie2p_map_value, .user_data = NULL},
    .map_argument = {.fn = loom_aie2p_map_argument, .user_data = NULL},
    .emit_preamble = {.fn = loom_aie2p_emit_fifo_state_preamble,
                      .user_data = NULL},
    .contract = LOOM_AIE2P_CORE_CONTRACT,
    .query_op_contract =
        {
            .fn = loom_aie2p_query_op_contract,
            .user_data = NULL,
        },
    .descriptor_matrix =
        {
            .options = loom_aie2p_descriptor_matrix_options,
            .query = loom_aie2p_descriptor_matrix_query,
            .attrs = NULL,
            .user_data = NULL,
        },
    .source_plan_observer = &loom_aie2p_f32_representation_observer,
    .prepare_source_memory =
        {
            .fn = loom_aie2p_prepare_vector_load_plan,
            .user_data = NULL,
        },
    .preselect_op = {.fn = loom_aie2p_preselect_op, .user_data = NULL},
    .mark_plan_storage_demands =
        {
            .fn = loom_aie2p_mark_plan_storage_demands,
            .user_data = NULL,
        },
    .describe_plan = {.fn = loom_aie2p_describe_plan, .user_data = NULL},
    .finalize_plan = {.fn = loom_aie2p_finalize_plan},
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
          .contract_set_stable_id = UINT64_C(0x25ed3c7d25a1452a),
          .policy = &kAie2pCoreLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
