// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/emit/wasm/contracts/core_simd128.h"
#include "loom/target/emit/wasm/contracts/core_simd128_lower_rules.h"
#include "loom/target/emit/wasm/error_catalog.h"
#include "loom/target/emit/wasm/lower/lower.h"
#include "loom/target/emit/wasm/lower/predicate_representation.h"
#include "loom/target/emit/wasm/lower/vector_carrier.h"
#include "loom/target/emit/wasm/lower/vector_structural.h"

static bool loom_wasm_type_is_i32_register(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      return true;
    default:
      return false;
  }
}

static bool loom_wasm_type_is_scalar_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_wasm_type_is_scalar_i64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I64;
}

static bool loom_wasm_type_is_scalar_f64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F64;
}

static bool loom_wasm_source_type_supported(void* user_data,
                                            const loom_module_t* module,
                                            loom_type_t source_type) {
  (void)user_data;
  (void)module;
  if (!loom_type_is_scalar(source_type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(source_type);
  return scalar_type == LOOM_SCALAR_TYPE_F8E4M3 ||
         scalar_type == LOOM_SCALAR_TYPE_F8E5M2;
}

static bool loom_wasm_source_function_vector_carrier_supported(
    void* user_data, const loom_module_t* module, loom_type_t source_type) {
  (void)user_data;
  (void)module;
  return loom_wasm_vector_type_is_callable(source_type);
}

static bool loom_wasm_source_vector_carrier_supported(
    void* user_data, const loom_module_t* module, loom_type_t source_type) {
  (void)user_data;
  (void)module;
  return loom_wasm_vector_carrier_for_type(source_type).packet_count != 0;
}

static iree_status_t loom_wasm_make_i32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_I32, 1, out_type);
}

static iree_status_t loom_wasm_make_f32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_F32, 1, out_type);
}

static iree_status_t loom_wasm_make_i64_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_I64, 1, out_type);
}

static iree_status_t loom_wasm_make_f64_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_F64, 1, out_type);
}

static iree_status_t loom_wasm_make_v128_register_type(
    loom_low_lower_context_t* context, uint32_t packet_count,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_V128, packet_count, out_type);
}

static iree_status_t loom_wasm_map_type(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_type_t source_type,
                                        loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_type_is_buffer(source_type) || loom_type_is_view(source_type) ||
      loom_wasm_type_is_i32_register(source_type)) {
    return loom_wasm_make_i32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_scalar_f32(source_type)) {
    return loom_wasm_make_f32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_scalar_i64(source_type)) {
    return loom_wasm_make_i64_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_scalar_f64(source_type)) {
    return loom_wasm_make_f64_register_type(context, out_low_type);
  }
  const loom_wasm_vector_carrier_t carrier =
      loom_wasm_vector_carrier_for_type(source_type);
  if (carrier.packet_count != 0) {
    return loom_wasm_make_v128_register_type(context, carrier.packet_count,
                                             out_low_type);
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_map_value(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t source_value_id,
                                         loom_type_t source_type,
                                         loom_type_t* out_low_type) {
  if (!loom_wasm_predicate_type(source_type, NULL)) {
    return loom_wasm_map_type(user_data, context, source_op, source_type,
                              out_low_type);
  }
  loom_low_representation_id_t representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_lookup_if_ready(context, source_value_id,
                                                &representation);
  const loom_wasm_vector_carrier_t carrier =
      loom_wasm_predicate_carrier(source_type, representation);
  if (carrier.packet_count != 0) {
    return loom_wasm_make_v128_register_type(context, carrier.packet_count,
                                             out_low_type);
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  const loom_type_t source_type =
      loom_module_value_type(environment->module, source_value_id);
  if (!loom_wasm_predicate_type(source_type, NULL)) {
    return iree_ok_status();
  }
  loom_low_representation_id_t representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  IREE_RETURN_IF_ERROR(loom_low_lower_representation_query_lookup(
      environment, source_value_id, &representation));
  const loom_wasm_vector_carrier_t carrier =
      loom_wasm_predicate_carrier(source_type, representation);
  if (carrier.packet_count != 0) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        WASM_CORE_SIMD128_REG_CLASS_ID_V128, representation,
        carrier.packet_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  if (loom_type_is_vector(source_type) &&
      !loom_wasm_vector_type_is_callable(source_type)) {
    return iree_ok_status();
  }
  return loom_wasm_map_type(user_data, context, source_function_op, source_type,
                            &out_argument->abi_type);
}

static iree_status_t loom_wasm_preselect_op(void* user_data,
                                            loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  return loom_wasm_select_vector_structural_plan(context, source_op, out_plan);
}

static void loom_wasm_mark_plan_storage_demands(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_wasm_vector_structural_plan_isa(plan)) {
    loom_wasm_mark_vector_structural_plan_demands(context, source_op, plan);
    return;
  }
  IREE_ASSERT_UNREACHABLE("Wasm storage demand has unknown plan kind");
}

static iree_status_t loom_wasm_describe_plan(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan, bool is_elided,
    uint64_t execution_count_plus_one,
    loom_low_lower_plan_report_t* out_report) {
  (void)user_data;
  (void)is_elided;
  (void)execution_count_plus_one;
  if (loom_wasm_vector_structural_plan_isa(plan)) {
    loom_wasm_describe_vector_structural_plan(context, source_op, plan,
                                              out_report);
  } else {
    IREE_ASSERT_UNREACHABLE("Wasm report has unknown plan kind");
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_op(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_wasm_vector_structural_plan_isa(plan)) {
    return loom_wasm_emit_vector_structural_plan(context, source_op, plan);
  }
  IREE_ASSERT_UNREACHABLE("Wasm emission has unknown plan kind");
  IREE_BUILTIN_UNREACHABLE();
}

#include "loom/target/emit/wasm/contracts/tables.inl"

static const uint16_t kWasmVectorPacketBitCounts[] = {128u};
static const uint16_t kWasmVectorPacketLaneCounts[] = {16u, 8u, 4u, 2u};
static const loom_target_vector_packet_lane_limit_t
    kWasmVectorPacketStructuralLaneLimits[] = {
        {
            .element_type = LOOM_SCALAR_TYPE_I1,
            .maximum_lane_count = 16u,
        },
};
static_assert(IREE_ARRAYSIZE(kWasmVectorPacketLaneCounts) <=
                  LOOM_TARGET_VECTOR_PACKET_LANE_COUNT_LIMIT,
              "packet lane candidates exceed the shared planner capacity");

static const loom_target_vector_packet_policy_t kWasmVectorPacketPolicy = {
    .native_bit_counts = kWasmVectorPacketBitCounts,
    .native_lane_counts = kWasmVectorPacketLaneCounts,
    .index_bit_count = 32u,
    .offset_bit_count = 32u,
    .structural_lane_limits = kWasmVectorPacketStructuralLaneLimits,
    .maximum_unpacketized_bit_count = 128u,
    .native_bit_count_count = IREE_ARRAYSIZE(kWasmVectorPacketBitCounts),
    .native_lane_count_count = IREE_ARRAYSIZE(kWasmVectorPacketLaneCounts),
    .structural_lane_limit_count =
        IREE_ARRAYSIZE(kWasmVectorPacketStructuralLaneLimits),
};

static const loom_low_lower_policy_t kWasmLowLowerPolicy = {
    .name = IREE_SVL("wasm-lower"),
    .error_catalog = &loom_wasm_error_catalog,
    .vector_packet_policy = &kWasmVectorPacketPolicy,
    .map_type = {.fn = loom_wasm_map_type, .user_data = NULL},
    .map_value = {.fn = loom_wasm_map_value, .user_data = NULL},
    .map_contract_value = {.fn = loom_wasm_map_contract_value,
                           .user_data = NULL},
    .map_argument = {.fn = loom_wasm_map_argument, .user_data = NULL},
    .source_type_supported = {.fn = loom_wasm_source_type_supported,
                              .user_data = NULL},
    .source_vector_carrier_supported =
        {
            .fn = loom_wasm_source_vector_carrier_supported,
            .user_data = NULL,
        },
    .source_function_vector_carrier_supported =
        {
            .fn = loom_wasm_source_function_vector_carrier_supported,
            .user_data = NULL,
        },
    .source_plan_observer = &loom_wasm_predicate_representation_observer,
    .contract = LOOM_WASM_CONTRACT,
    .query_op_contract =
        {
            .fn = loom_wasm_query_vector_structural_contract,
            .user_data = NULL,
        },
    .preselect_op = {.fn = loom_wasm_preselect_op, .user_data = NULL},
    .mark_plan_storage_demands =
        {
            .fn = loom_wasm_mark_plan_storage_demands,
            .user_data = NULL,
        },
    .describe_plan = {.fn = loom_wasm_describe_plan, .user_data = NULL},
    .emit_op = {.fn = loom_wasm_emit_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_wasm_low_lower_policy(void) {
  return &kWasmLowLowerPolicy;
}

void loom_wasm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("wasm.core.simd128"),
          .policy = &kWasmLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
