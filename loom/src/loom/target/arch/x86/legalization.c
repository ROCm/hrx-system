// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/legalization.h"

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/records/target_records.h"
#include "loom/target/arch/x86/vector_insert.h"
#include "loom/transforms/scalar/target_legalization.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/to_scalar.h"

static bool loom_x86_legalizer_descriptor_set_is_x86(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id == LOOM_X86_TARGET_STABLE_ID;
}

static iree_status_t loom_x86_retain_native_vector_op(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  (void)op;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_x86_legalizer_descriptor_set_is_x86(context->descriptor_set)) {
    return iree_ok_status();
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
  };
  return iree_ok_status();
}

static bool loom_x86_match_vector_float8_extension(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_extf_result(op));
  return loom_scalar_type_set_contains(
      LOOM_SCALAR_TYPE_SET_16BIT_FLOAT | LOOM_SCALAR_TYPE_SET_F32,
      loom_type_element_type(result_type));
}

static bool loom_x86_match_vector_float8_truncation(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_fptrunc_result(op));
  return loom_scalar_type_set_contains(
      LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2,
      loom_type_element_type(result_type));
}

static iree_status_t loom_x86_legalize_float8_to_bfloat_extension(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_x86_legalizer_descriptor_set_is_x86(context->descriptor_set)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scalar_rewrite_float8_extension(context, op));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_x86_legalize_vector_float8_conversion(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_x86_legalizer_descriptor_set_is_x86(context->descriptor_set)) {
    return iree_ok_status();
  }

  // Capture each source lane at the authored conversion. Reconstructing a
  // producer at a later scalar consumer could move a read across an aliasing
  // write and change the converted value.
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_legalize_vector_divrem(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_x86_legalizer_descriptor_set_is_x86(context->descriptor_set)) {
    return iree_ok_status();
  }

  // Preserve the quotient/remainder tuple while converting unsupported vector
  // carriers to scalar lanes. Each lane can then select the paired native
  // divide instead of the shared fallback splitting the source operation.
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_legalize_vector_packets(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_x86_legalizer_descriptor_set_is_x86(context->descriptor_set) ||
      context->source_function_has_unsupported_vector_carrier ||
      context->vector_packet_policy == NULL) {
    return iree_ok_status();
  }

  bool rewritten = false;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD: {
      IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_load(
          context, op, context->vector_packet_policy, &rewritten));
      break;
    }
    case LOOM_OP_VECTOR_STORE: {
      IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_store(
          context, op, context->vector_packet_policy, &rewritten));
      break;
    }
    case LOOM_OP_VECTOR_INSERT: {
      IREE_RETURN_IF_ERROR(
          loom_x86_vector_insert_legalize(context, op, &rewritten));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE();
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kX86LegalizerRules[] = {
    {
        .root_kind = LOOM_OP_VECTOR_LOAD,
        .legalize = loom_x86_legalize_vector_packets,
    },
    {
        // A native-width store may terminate packets retained from an
        // oversized producer graph without materializing its logical result.
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_x86_legalize_vector_packets,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INSERT,
        .legalize = loom_x86_legalize_vector_packets,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTF,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2,
        .match = loom_scalar_match_float8_to_bfloat_extension,
        .legalize = loom_x86_legalize_float8_to_bfloat_extension,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTF,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2,
        .match = loom_x86_match_vector_float8_extension,
        .legalize = loom_x86_legalize_vector_float8_conversion,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FPTRUNC,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_16BIT_FLOAT | LOOM_SCALAR_TYPE_SET_F32,
        .match = loom_x86_match_vector_float8_truncation,
        .legalize = loom_x86_legalize_vector_float8_conversion,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DIVREMSI,
        .legalize = loom_x86_legalize_vector_divrem,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DIVREMUI,
        .legalize = loom_x86_legalize_vector_divrem,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT2F,
        .legalize = loom_x86_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4I,
        .legalize = loom_x86_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT8I4,
        .legalize = loom_x86_retain_native_vector_op,
    },
};

const loom_target_legalizer_provider_t
    loom_x86_target_legalizer_provider_storage = {
        .name = IREE_SVL("x86"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .rules = kX86LegalizerRules,
        .rule_count = IREE_ARRAYSIZE(kX86LegalizerRules),
};

const loom_target_legalizer_provider_t* loom_x86_target_legalizer_provider(
    void) {
  return &loom_x86_target_legalizer_provider_storage;
}
