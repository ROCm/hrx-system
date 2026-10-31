// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/provider.h"

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/arch/wasm/descriptors/low_registry.h"
#include "loom/target/arch/wasm/legalization_float.h"
#include "loom/target/arch/wasm/legalization_memory.h"
#include "loom/target/arch/wasm/low_verify.h"
#include "loom/target/arch/wasm/lower/lower.h"
#include "loom/target/arch/wasm/math_policy.h"
#include "loom/target/arch/wasm/ops/ops.h"
#include "loom/target/arch/wasm/ops/registry.h"
#include "loom/target/arch/wasm/records/target_records.h"
#include "loom/transforms/scalar/target_legalization.h"
#include "loom/transforms/vector/packet_insert_legalization.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/shape_legalization.h"
#include "loom/transforms/vector/target_legalization.h"
#include "loom/transforms/vector/to_scalar.h"

static iree_status_t loom_wasm_profile_project_facts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  out_facts->selector = LOOM_WASM_TARGET_KIND_SIMD128;
  return iree_ok_status();
}

static const loom_target_profile_type_t kProfileType = {
    .name = IREE_SVL("wasm"),
    .fact_type = &loom_wasm_target_fact_type,
    .project_facts = loom_wasm_profile_project_facts,
    .equal = loom_target_profile_equal_common,
};

static const loom_target_profile_t kSimd128Profile = {
    .type = &kProfileType,
    .target_bundle = &loom_wasm_low_target_bundle_core_simd128,
};

static iree_status_t loom_wasm_select_profile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  *out_profile = NULL;
  if (!iree_string_view_equal(selector, IREE_SV("simd128"))) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown Wasm target profile '%.*s'",
                            (int)selector.size, selector.data);
  }
  *out_profile = &kSimd128Profile;
  return iree_ok_status();
}

static iree_status_t loom_wasm_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location) {
  const loom_target_facts_t* facts = resolved_target->facts;
  static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ == 30,
                "wasm target flags reserve the first 30 bits for common "
                "target facts");
  static_assert(LOOM_WASM_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT ==
                    (1u << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT),
                "wasm target flags must follow target fact ordinals");
  static_assert(LOOM_WASM_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS ==
                    (1u << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
                "wasm target flags must follow target fact ordinals");

  const loom_wasm_target_build_flags_t build_flags =
      (loom_wasm_target_build_flags_t)facts->explicit_fields;
  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_WASM_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->storage.export_plan.export_symbol, &export_symbol));
  }
  loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_WASM_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->storage.config.contract_set.spelling,
        &contract_set_key));
  }

  const loom_target_snapshot_t* snapshot = &facts->storage.snapshot;
  const loom_target_export_plan_t* export_plan = &facts->storage.export_plan;
  const loom_target_config_t* config = &facts->storage.config;
  loom_op_t* target_op = NULL;
  return loom_wasm_target_build(
      builder, build_flags, (loom_wasm_target_kind_t)facts->selector, symbol,
      snapshot->codegen_format, snapshot->artifact_format,
      snapshot->default_pointer_bitwidth, snapshot->index_bitwidth,
      snapshot->offset_bitwidth, snapshot->max_workgroup_size.x,
      snapshot->max_workgroup_size.y, snapshot->max_workgroup_size.z,
      snapshot->max_flat_workgroup_size, snapshot->max_workgroup_storage_bytes,
      snapshot->subgroup_size, snapshot->max_grid_size.x,
      snapshot->max_grid_size.y, snapshot->max_grid_size.z,
      snapshot->max_flat_grid_size, snapshot->max_workgroup_count.x,
      snapshot->max_workgroup_count.y, snapshot->max_workgroup_count.z,
      snapshot->memory_spaces.generic, snapshot->memory_spaces.global,
      snapshot->memory_spaces.workgroup, snapshot->memory_spaces.constant,
      snapshot->memory_spaces.private_memory, snapshot->memory_spaces.host,
      snapshot->memory_spaces.descriptor, export_plan->abi_kind, export_symbol,
      export_plan->linkage, contract_set_key, config->contract_feature_bits,
      location, &target_op);
}

static const loom_low_verify_provider_t* const kLoomWasmLowVerifyProviders[] = {
    &loom_wasm_low_verify_provider,
};

static bool loom_wasm_legalizer_descriptor_set_is_simd128(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id ==
             loom_wasm_core_simd128_descriptor_set()->target_stable_id;
}

static uint64_t loom_wasm_vector_type_payload_bit_count(loom_type_t type) {
  uint64_t lane_count = 0;
  if (!loom_type_is_vector(type) || !loom_type_is_all_static(type) ||
      !loom_type_static_element_count(type, &lane_count)) {
    return 0;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  uint64_t payload_bit_count = 0;
  if (element_bit_count <= 0) {
    return 0;
  }
  if (!iree_checked_mul_u64(lane_count, (uint32_t)element_bit_count,
                            &payload_bit_count)) {
    return UINT64_MAX;
  }
  return payload_bit_count;
}

static iree_status_t loom_wasm_legalize_float8_to_bfloat_extension(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_legalizer_descriptor_set_is_simd128(context->descriptor_set)) {
    return iree_ok_status();
  }
  (void)entry;
  IREE_RETURN_IF_ERROR(loom_scalar_rewrite_float8_extension(context, op));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_wasm_legalize_vector_load(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_legalizer_descriptor_set_is_simd128(context->descriptor_set)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_static_shape_memory_rewrite_op(context, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
    return iree_ok_status();
  }
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_load_result(op));
  const uint64_t payload_bit_count =
      loom_wasm_vector_type_payload_bit_count(result_type);
  if (payload_bit_count != 0 && payload_bit_count < 128u) {
    IREE_RETURN_IF_ERROR(
        loom_wasm_partial_vector_memory_rewrite_op(context, op, &rewritten));
    if (!rewritten) {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
          context->pass, context->rewriter, op, &rewritten));
    }
  } else if (payload_bit_count <= 128u) {
    return iree_ok_status();
  } else if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    return iree_ok_status();
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_load(
        context, op, context->vector_packet_policy, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_store_value_can_remain_packed(
    loom_target_legalization_context_t* context, const loom_op_t* store_op,
    bool* out_can_remain_packed) {
  *out_can_remain_packed = false;
  const loom_value_t* value =
      loom_module_value(context->module, loom_vector_store_value(store_op));
  if (loom_value_is_block_arg(value)) {
    *out_can_remain_packed = true;
    return iree_ok_status();
  }
  const loom_op_t* producer = loom_value_def_op(value);
  loom_target_contract_query_result_t producer_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
      context, producer, &producer_result));
  *out_can_remain_packed =
      producer_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  return iree_ok_status();
}

static iree_status_t loom_wasm_legalize_vector_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_legalizer_descriptor_set_is_simd128(context->descriptor_set)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_static_shape_memory_rewrite_op(context, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
    return iree_ok_status();
  }
  const loom_type_t value_type =
      loom_module_value_type(context->module, loom_vector_store_value(op));
  const uint64_t payload_bit_count =
      loom_wasm_vector_type_payload_bit_count(value_type);
  if (payload_bit_count != 0 && payload_bit_count < 128u) {
    bool can_remain_packed = false;
    IREE_RETURN_IF_ERROR(loom_wasm_store_value_can_remain_packed(
        context, op, &can_remain_packed));
    if (can_remain_packed) {
      IREE_RETURN_IF_ERROR(
          loom_wasm_partial_vector_memory_rewrite_op(context, op, &rewritten));
      if (!rewritten) {
        IREE_RETURN_IF_ERROR(loom_vector_store_captured_to_scalar_rewrite_op(
            context->pass, context->rewriter, op, &rewritten));
      }
    } else if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
      out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
      return iree_ok_status();
    } else {
      IREE_RETURN_IF_ERROR(
          loom_wasm_partial_vector_memory_rewrite_op(context, op, &rewritten));
      if (!rewritten) {
        IREE_RETURN_IF_ERROR(loom_vector_store_to_scalar_rewrite_op(
            context->pass, context->rewriter, op, &rewritten));
      }
    }
  } else if (payload_bit_count <= 128u) {
    return iree_ok_status();
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_store(
        context, op, context->vector_packet_policy, &rewritten));
  }
  if (!rewritten && context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    return iree_ok_status();
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_legalize_vector_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_legalizer_descriptor_set_is_simd128(context->descriptor_set)) {
    return iree_ok_status();
  }
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_extract_result(op));
  if (!loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_legalize_vector_insert(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_legalizer_descriptor_set_is_simd128(context->descriptor_set)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_insert(
      context, op, context->vector_packet_policy,
      /*insert_materializer=*/NULL, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_legalize_vector_slice(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_legalizer_descriptor_set_is_simd128(context->descriptor_set) ||
      loom_vector_slice_offsets(op).count == 0) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_dynamic_slice(
      context, op, context->vector_packet_policy,
      /*consumer_materializer=*/NULL, &rewritten));
  if (!rewritten && context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  } else if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_legalize_vector_transpose(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_legalizer_descriptor_set_is_simd128(context->descriptor_set)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_transpose_to_shuffle_rewrite_op(
      context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_legalize_vector_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_legalizer_descriptor_set_is_simd128(context->descriptor_set)) {
    return iree_ok_status();
  }

  loom_vector_packet_reduce_result_t packet_result =
      LOOM_VECTOR_PACKET_REDUCE_RESULT_NONE;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_reduce(
      context, op, context->vector_packet_policy, &packet_result));
  bool rewritten = packet_result == LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
  if (packet_result == LOOM_VECTOR_PACKET_REDUCE_RESULT_CAPTURE_INPUT) {
    IREE_RETURN_IF_ERROR(loom_vector_reduce_captured_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kLoomWasmLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_SCALAR_EXTF,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2,
        .match = loom_scalar_match_float8_to_bfloat_extension,
        .legalize = loom_wasm_legalize_float8_to_bfloat_extension,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MINIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_wasm_legalize_scalar_float_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MAXIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_wasm_legalize_scalar_float_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MINNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_wasm_legalize_scalar_float_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MAXNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_wasm_legalize_scalar_float_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CLAMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_wasm_legalize_scalar_float_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
        .legalize = loom_wasm_legalize_vector_cmpf,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .legalize = loom_wasm_legalize_vector_float_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .legalize = loom_wasm_legalize_vector_float_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .legalize = loom_wasm_legalize_vector_float_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .legalize = loom_wasm_legalize_vector_float_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CLAMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .legalize = loom_wasm_legalize_vector_float_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_wasm_legalize_vector_load,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_wasm_legalize_vector_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTRACT,
        .legalize = loom_wasm_legalize_vector_extract,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INSERT,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_wasm_legalize_vector_insert,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SLICE,
        .legalize = loom_wasm_legalize_vector_slice,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSPOSE,
        .legalize = loom_wasm_legalize_vector_transpose,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_wasm_legalize_vector_reduce,
    },
};

static const loom_target_legalizer_provider_t kLoomWasmLegalizerProvider = {
    .name = IREE_SVL("wasm"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
    .rules = kLoomWasmLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kLoomWasmLegalizerRules),
};

static const loom_target_legalizer_provider_t* const
    kLoomWasmLegalizerProviders[] = {
        &kLoomWasmLegalizerProvider,
};

const loom_target_provider_t loom_wasm_target_provider = {
    .profile_type = &kProfileType,
    .select_profile = loom_wasm_select_profile,
    .materialize_definition = loom_wasm_materialize_definition,
    .select_call_policy = loom_target_select_call_policy_direct,
    .register_context = loom_wasm_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_wasm_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_wasm_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_wasm_math_policy_registry_initialize,
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomWasmLegalizerProviders),
            .values = kLoomWasmLegalizerProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomWasmLowVerifyProviders),
            .values = kLoomWasmLowVerifyProviders,
        },
    .target_fact_type = &loom_wasm_target_fact_type,
};

static const loom_target_provider_t* const kLoomWasmTargetProviders[] = {
    &loom_wasm_target_provider,
};

const loom_target_provider_set_t loom_wasm_target_provider_set = {
    .providers = kLoomWasmTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomWasmTargetProviders),
};
