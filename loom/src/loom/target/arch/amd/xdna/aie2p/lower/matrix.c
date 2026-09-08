// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/matrix.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/util/fact_table.h"

typedef enum loom_aie2p_matrix_plan_kind_e {
  LOOM_AIE2P_MATRIX_PLAN_MMA_I8_M8N8K8 = 0x100,
  LOOM_AIE2P_MATRIX_PLAN_STORE_I32_M8N8 = 0x101,
} loom_aie2p_matrix_plan_kind_t;

typedef enum loom_aie2p_matrix_operation_e {
  LOOM_AIE2P_MATRIX_OPERATION_MULTIPLY = 0,
  LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE = 1,
  LOOM_AIE2P_MATRIX_OPERATION_COUNT_ = 2,
} loom_aie2p_matrix_operation_t;

typedef enum loom_aie2p_matrix_numeric_mode_e {
  LOOM_AIE2P_MATRIX_NUMERIC_S8S8 = 0,
  LOOM_AIE2P_MATRIX_NUMERIC_U8S8 = 1,
  LOOM_AIE2P_MATRIX_NUMERIC_S8U8 = 2,
  LOOM_AIE2P_MATRIX_NUMERIC_U8U8 = 3,
  LOOM_AIE2P_MATRIX_NUMERIC_COUNT_ = 4,
} loom_aie2p_matrix_numeric_mode_t;

typedef enum loom_aie2p_matrix_rejection_bit_e {
  LOOM_AIE2P_MATRIX_REJECTION_SHAPE = 1u << 0,
  LOOM_AIE2P_MATRIX_REJECTION_PAYLOAD = 1u << 1,
  LOOM_AIE2P_MATRIX_REJECTION_FRAGMENT = 1u << 2,
} loom_aie2p_matrix_rejection_bit_t;

typedef struct loom_aie2p_matrix_mma_plan_t {
  // Descriptor materializing the AIE integer matrix control word.
  loom_low_lower_resolved_descriptor_t control_constant;
  // Native 8x8x8 integer matrix operation packet.
  loom_low_lower_resolved_descriptor_t operation_descriptor;
  // Whether the packet starts or extends an accumulator.
  loom_aie2p_matrix_operation_t operation;
  // Exact AIE integer matrix control word.
  uint16_t control;
} loom_aie2p_matrix_mma_plan_t;

typedef struct loom_aie2p_matrix_store_plan_t {
  // Native 512-bit accumulator-quarter store packet.
  loom_low_lower_resolved_descriptor_t store;
} loom_aie2p_matrix_store_plan_t;

typedef struct loom_aie2p_integer_matrix_mode_t {
  // Descriptor key for each matrix operation kind.
  iree_string_view_t descriptor_keys[LOOM_AIE2P_MATRIX_OPERATION_COUNT_];
  // Exact signedness and 8x8x8 B-mode control word.
  uint16_t control;
} loom_aie2p_integer_matrix_mode_t;

static const loom_aie2p_integer_matrix_mode_t
    loom_aie2p_integer_matrix_modes[LOOM_AIE2P_MATRIX_NUMERIC_COUNT_] = {
        [LOOM_AIE2P_MATRIX_NUMERIC_S8S8] =
            {
                .descriptor_keys =
                    {
                        [LOOM_AIE2P_MATRIX_OPERATION_MULTIPLY] = IREE_SVL(
                            "amd.xdna.aie2p.matrix.multiply.s8s8.m8n8k8."
                            "configured"),
                        [LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE] = IREE_SVL(
                            "amd.xdna.aie2p.matrix.accumulate.s8s8.m8n8k8."
                            "configured"),
                    },
                .control = 776,
            },
        [LOOM_AIE2P_MATRIX_NUMERIC_U8S8] =
            {
                .descriptor_keys =
                    {
                        [LOOM_AIE2P_MATRIX_OPERATION_MULTIPLY] = IREE_SVL(
                            "amd.xdna.aie2p.matrix.multiply.u8s8.m8n8k8."
                            "configured"),
                        [LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE] = IREE_SVL(
                            "amd.xdna.aie2p.matrix.accumulate.u8s8.m8n8k8."
                            "configured"),
                    },
                .control = 264,
            },
        [LOOM_AIE2P_MATRIX_NUMERIC_S8U8] =
            {
                .descriptor_keys =
                    {
                        [LOOM_AIE2P_MATRIX_OPERATION_MULTIPLY] = IREE_SVL(
                            "amd.xdna.aie2p.matrix.multiply.s8u8.m8n8k8."
                            "configured"),
                        [LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE] = IREE_SVL(
                            "amd.xdna.aie2p.matrix.accumulate.s8u8.m8n8k8."
                            "configured"),
                    },
                .control = 520,
            },
        [LOOM_AIE2P_MATRIX_NUMERIC_U8U8] =
            {
                .descriptor_keys =
                    {
                        [LOOM_AIE2P_MATRIX_OPERATION_MULTIPLY] = IREE_SVL(
                            "amd.xdna.aie2p.matrix.multiply.u8u8.m8n8k8."
                            "configured"),
                        [LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE] = IREE_SVL(
                            "amd.xdna.aie2p.matrix.accumulate.u8u8.m8n8k8."
                            "configured"),
                    },
                .control = 8,
            },
};

static loom_low_lower_resolved_descriptor_t loom_aie2p_matrix_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  const uint32_t ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key);
  return (loom_low_lower_resolved_descriptor_t){
      .descriptor =
          loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal),
  };
}

static bool loom_aie2p_matrix_numeric_mode(
    loom_contract_numeric_type_t lhs, loom_contract_numeric_type_t rhs,
    loom_aie2p_matrix_numeric_mode_t* out_mode) {
  if ((lhs != LOOM_CONTRACT_NUMERIC_I8 && lhs != LOOM_CONTRACT_NUMERIC_U8) ||
      (rhs != LOOM_CONTRACT_NUMERIC_I8 && rhs != LOOM_CONTRACT_NUMERIC_U8)) {
    return false;
  }
  *out_mode =
      (loom_aie2p_matrix_numeric_mode_t)((lhs == LOOM_CONTRACT_NUMERIC_U8
                                              ? 1u
                                              : 0u) |
                                         (rhs == LOOM_CONTRACT_NUMERIC_U8
                                              ? 2u
                                              : 0u));
  return true;
}

static bool loom_aie2p_matrix_encoded_operand_is_native_unscaled(
    const loom_contract_encoded_operand_t* encoded) {
  if (encoded->available_auxiliary_operands != 0 ||
      encoded->required_auxiliary_operands != 0) {
    return false;
  }
  const loom_value_fact_encoded_operand_schema_t source_schema =
      encoded->source_schema.encoded_operand;
  const loom_value_fact_encoded_operand_schema_t target_schema =
      encoded->target_schema.encoded_operand;
  if (loom_value_fact_encoded_operand_schema_is_unknown(source_schema) &&
      loom_value_fact_encoded_operand_schema_is_unknown(target_schema)) {
    return encoded->source_schema.static_spec_encoding_id == 0 &&
           encoded->target_schema.static_spec_encoding_id == 0;
  }
  if (!loom_value_fact_encoded_operand_schema_equal(source_schema,
                                                    target_schema)) {
    return false;
  }

  loom_value_fact_encoded_operand_schema_t interpretation = source_schema;
  interpretation.element_format = 0;
  interpretation.payload_packing = 0;
  interpretation.payload_register_count = 0;
  interpretation.payload_element_count = 0;
  return source_schema.element_format != 0 &&
         source_schema.payload_packing ==
             LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT &&
         source_schema.payload_register_count != 0 &&
         source_schema.payload_element_count != 0 &&
         loom_value_fact_encoded_operand_schema_is_unknown(interpretation);
}

static loom_aie2p_matrix_rejection_bit_t loom_aie2p_matrix_request_rejection(
    const loom_contract_request_t* request,
    loom_aie2p_matrix_numeric_mode_t* out_numeric_mode) {
  if (request->shape.m != 8 || request->shape.n != 8 || request->shape.k != 8 ||
      request->shape.block_count != 1 || request->k_group_size != 8) {
    return LOOM_AIE2P_MATRIX_REJECTION_SHAPE;
  }
  if (request->kind != LOOM_CONTRACT_KIND_MATRIX_MULTIPLY ||
      request->arithmetic != LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT ||
      request->capability_class != LOOM_CONTRACT_CAPABILITY_CLASS_MATRIX ||
      request->lhs.role != LOOM_CONTRACT_OPERAND_ROLE_LHS ||
      request->rhs.role != LOOM_CONTRACT_OPERAND_ROLE_RHS ||
      request->accumulator.role != LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR ||
      request->result.role != LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
      !loom_aie2p_matrix_numeric_mode(request->lhs.numeric_type,
                                      request->rhs.numeric_type,
                                      out_numeric_mode) ||
      request->accumulator.numeric_type != LOOM_CONTRACT_NUMERIC_I32 ||
      request->result.numeric_type != LOOM_CONTRACT_NUMERIC_I32 ||
      request->lhs.payload_register_count != 16 ||
      request->rhs.payload_register_count != 16 ||
      request->accumulator.payload_register_count != 64 ||
      request->result.payload_register_count != 64 ||
      request->lhs.payload_element_count != 64 ||
      request->rhs.payload_element_count != 64 ||
      request->accumulator.payload_element_count != 64 ||
      request->result.payload_element_count != 64 ||
      !loom_aie2p_matrix_encoded_operand_is_native_unscaled(
          &request->lhs.encoded) ||
      !loom_aie2p_matrix_encoded_operand_is_native_unscaled(
          &request->rhs.encoded) ||
      !loom_aie2p_matrix_encoded_operand_is_native_unscaled(
          &request->accumulator.encoded) ||
      !loom_aie2p_matrix_encoded_operand_is_native_unscaled(
          &request->result.encoded)) {
    return LOOM_AIE2P_MATRIX_REJECTION_PAYLOAD;
  }
  if (request->fragment.atom_bits != LOOM_CONTRACT_FRAGMENT_INTERNAL ||
      request->fragment.vector_bit_width != 0 ||
      request->fragment.source_lane_count != 64 ||
      request->fragment.result_lane_count != 64 ||
      request->fragment.subgroup_size != 0) {
    return LOOM_AIE2P_MATRIX_REJECTION_FRAGMENT;
  }
  return 0;
}

static iree_string_view_t loom_aie2p_matrix_rejection_constraint(
    loom_aie2p_matrix_rejection_bit_t rejection) {
  switch (rejection) {
    case LOOM_AIE2P_MATRIX_REJECTION_SHAPE:
      return IREE_SV("aie2p.matrix.tile_shape");
    case LOOM_AIE2P_MATRIX_REJECTION_PAYLOAD:
      return IREE_SV("aie2p.matrix.payload");
    case LOOM_AIE2P_MATRIX_REJECTION_FRAGMENT:
      return IREE_SV("aie2p.matrix.fragment");
  }
  return IREE_SV("aie2p.matrix.contract");
}

static bool loom_aie2p_matrix_init_is_zero(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op) {
  const loom_value_t* init_value =
      loom_module_value(module, loom_vector_mma_init(source_op));
  if (loom_value_is_block_arg(init_value)) {
    return false;
  }
  const loom_op_t* fragment_op = loom_value_def_op(init_value);
  if (fragment_op == NULL || !loom_vector_fragment_isa(fragment_op)) {
    return false;
  }
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  return loom_value_facts_query_all_equal_element(
             &fact_table->context,
             loom_value_fact_table_lookup(
                 fact_table, loom_vector_fragment_data(fragment_op)),
             &element_facts) &&
         loom_value_facts_is_zero(element_facts);
}

static loom_contract_vector_mma_options_t loom_aie2p_matrix_options(void) {
  return (loom_contract_vector_mma_options_t){
      .fragment_projection =
          LOOM_CONTRACT_VECTOR_MMA_FRAGMENT_PROJECTION_EXPLICIT,
      .k_group_size = 8,
      .fragment =
          (loom_contract_fragment_t){
              .atom_bits = LOOM_CONTRACT_FRAGMENT_INTERNAL,
              .source_lane_count = 64,
              .result_lane_count = 64,
          },
      .capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_MATRIX,
      .policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
  };
}

iree_status_t loom_aie2p_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options) {
  (void)user_data;
  (void)environment;
  (void)rule;
  *out_options = loom_aie2p_matrix_options();
  return iree_ok_status();
}

iree_status_t loom_aie2p_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* request,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();

  loom_aie2p_matrix_numeric_mode_t numeric_mode = 0;
  loom_aie2p_matrix_rejection_bit_t rejection =
      loom_aie2p_matrix_request_rejection(request, &numeric_mode);
  if (rejection != 0) {
    return loom_low_lower_descriptor_matrix_reject(
        environment, rule, source_op,
        loom_aie2p_matrix_rejection_constraint(rejection), 0, rejection, 0,
        out_result);
  }

  const loom_aie2p_matrix_operation_t operation =
      loom_aie2p_matrix_init_is_zero(environment->module,
                                     environment->fact_table, source_op)
          ? LOOM_AIE2P_MATRIX_OPERATION_MULTIPLY
          : LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE;
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  out_result->selected_descriptor =
      loom_aie2p_matrix_descriptor(environment->descriptor_set,
                                   loom_aie2p_integer_matrix_modes[numeric_mode]
                                       .descriptor_keys[operation])
          .descriptor;
  return iree_ok_status();
}

static bool loom_aie2p_matrix_source_type_is(const loom_module_t* module,
                                             loom_value_id_t value_id,
                                             loom_type_kind_t kind,
                                             loom_scalar_type_t element_type,
                                             uint8_t rank,
                                             const int64_t* shape) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (loom_type_kind(type) != kind ||
      loom_type_element_type(type) != element_type ||
      loom_type_rank(type) != rank || !loom_type_is_all_static(type)) {
    return false;
  }
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_static_size_at(type, i) != shape[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_aie2p_select_matrix_mma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  const loom_contract_vector_mma_options_t options =
      loom_aie2p_matrix_options();
  loom_contract_request_t request = {0};
  loom_aie2p_matrix_numeric_mode_t numeric_mode = 0;
  if (!loom_contract_request_from_vector_mma_op(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op, &options,
          &request, NULL) ||
      loom_aie2p_matrix_request_rejection(&request, &numeric_mode) != 0) {
    return iree_ok_status();
  }

  const loom_aie2p_matrix_operation_t operation =
      loom_aie2p_matrix_init_is_zero(loom_low_lower_context_module(context),
                                     loom_low_lower_context_fact_table(context),
                                     source_op)
          ? LOOM_AIE2P_MATRIX_OPERATION_MULTIPLY
          : LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE;
  loom_aie2p_matrix_mma_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  *plan = (loom_aie2p_matrix_mma_plan_t){
      .control_constant = loom_aie2p_matrix_descriptor(
          descriptor_set, IREE_SV("amd.xdna.aie2p.constant.i32.mova")),
      .operation_descriptor = loom_aie2p_matrix_descriptor(
          descriptor_set, loom_aie2p_integer_matrix_modes[numeric_mode]
                              .descriptor_keys[operation]),
      .operation = operation,
      .control = loom_aie2p_integer_matrix_modes[numeric_mode].control,
  };
  *out_plan =
      loom_low_lower_plan_make(LOOM_AIE2P_MATRIX_PLAN_MMA_I8_M8N8K8, plan);
  return iree_ok_status();
}

static bool loom_aie2p_matrix_fragment_store_matches(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op) {
  static const int64_t kValueShape[] = {64};
  static const int64_t kViewShape[] = {8, 8};
  if (loom_vector_fragment_store_role(source_op) != LOOM_VECTOR_ROLE_RESULT ||
      loom_vector_fragment_store_indices(source_op).count != 0 ||
      loom_vector_fragment_store_blocks_is_present(source_op) ||
      !loom_aie2p_matrix_source_type_is(
          module, loom_vector_fragment_store_value(source_op), LOOM_TYPE_VECTOR,
          LOOM_SCALAR_TYPE_I32, 1, kValueShape) ||
      !loom_aie2p_matrix_source_type_is(
          module, loom_vector_fragment_store_view(source_op), LOOM_TYPE_VIEW,
          LOOM_SCALAR_TYPE_I32, 2, kViewShape)) {
    return false;
  }
  const loom_i64_array_t static_indices = loom_attr_as_i64_array(
      loom_vector_fragment_store_static_indices(source_op));
  if (static_indices.count != 2 || static_indices.values[0] != 0 ||
      static_indices.values[1] != 0) {
    return false;
  }
  int64_t rows = 0;
  int64_t columns = 0;
  return loom_value_facts_as_exact_i64(
             loom_value_fact_table_lookup(
                 fact_table, loom_vector_fragment_store_rows(source_op)),
             &rows) &&
         loom_value_facts_as_exact_i64(
             loom_value_fact_table_lookup(
                 fact_table, loom_vector_fragment_store_columns(source_op)),
             &columns) &&
         rows == 8 && columns == 8;
}

static iree_status_t loom_aie2p_select_matrix_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  if (!loom_aie2p_matrix_fragment_store_matches(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op)) {
    return iree_ok_status();
  }

  loom_aie2p_matrix_store_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  plan->store = loom_aie2p_matrix_descriptor(
      loom_low_lower_context_descriptor_set(context),
      IREE_SV("amd.xdna.aie2p.store.accumulator.indexed.immediate"));
  *out_plan =
      loom_low_lower_plan_make(LOOM_AIE2P_MATRIX_PLAN_STORE_I32_M8N8, plan);
  return iree_ok_status();
}

bool loom_aie2p_matrix_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_AIE2P_MATRIX_PLAN_MMA_I8_M8N8K8 ||
         plan.id == LOOM_AIE2P_MATRIX_PLAN_STORE_I32_M8N8;
}

iree_status_t loom_aie2p_select_matrix_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (loom_vector_mma_isa(source_op)) {
    return loom_aie2p_select_matrix_mma(context, source_op, out_plan);
  }
  if (loom_vector_fragment_store_isa(source_op)) {
    return loom_aie2p_select_matrix_store(context, source_op, out_plan);
  }
  return iree_ok_status();
}

void loom_aie2p_mark_matrix_plan_demands(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  switch ((loom_aie2p_matrix_plan_kind_t)plan.id) {
    case LOOM_AIE2P_MATRIX_PLAN_MMA_I8_M8N8K8: {
      const loom_aie2p_matrix_mma_plan_t* matrix_plan =
          (const loom_aie2p_matrix_mma_plan_t*)plan.target_data;
      loom_low_lower_require_source_value_storage(
          context, loom_vector_mma_lhs(source_op));
      loom_low_lower_require_source_value_storage(
          context, loom_vector_mma_rhs(source_op));
      if (matrix_plan->operation == LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE) {
        loom_low_lower_require_source_value_storage(
            context, loom_vector_mma_init(source_op));
      }
      return;
    }
    case LOOM_AIE2P_MATRIX_PLAN_STORE_I32_M8N8:
      loom_low_lower_require_source_value_storage(
          context, loom_vector_fragment_store_value(source_op));
      loom_low_lower_require_source_value_storage(
          context, loom_vector_fragment_store_view(source_op));
      return;
  }
  IREE_ASSERT_UNREACHABLE("AIE2P matrix demand has unknown plan kind");
}

void loom_aie2p_describe_matrix_plan(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     loom_low_lower_plan_report_t* out_report) {
  (void)source_op;
  *out_report = (loom_low_lower_plan_report_t){0};
  switch ((loom_aie2p_matrix_plan_kind_t)plan.id) {
    case LOOM_AIE2P_MATRIX_PLAN_MMA_I8_M8N8K8: {
      const loom_aie2p_matrix_mma_plan_t* matrix_plan =
          (const loom_aie2p_matrix_mma_plan_t*)plan.target_data;
      out_report->plan_key = loom_low_descriptor_set_string(
          loom_low_lower_context_descriptor_set(context),
          matrix_plan->operation_descriptor.descriptor->mnemonic_string_offset);
      return;
    }
    case LOOM_AIE2P_MATRIX_PLAN_STORE_I32_M8N8:
      out_report->plan_key = IREE_SV("matrix.fragment-store.i32.m8n8");
      return;
  }
  IREE_ASSERT_UNREACHABLE("AIE2P matrix report has unknown plan kind");
}

static iree_status_t loom_aie2p_emit_matrix_mma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_aie2p_matrix_mma_plan_t* plan) {
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t init = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_lhs(source_op), &lhs));
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_rhs(source_op), &rhs));
  if (plan->operation == LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_vector_mma_init(source_op), &init));
  }

  loom_type_t control_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1, &control_type));
  loom_string_id_t immediate_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("i"), &immediate_name));
  const loom_named_attr_t control_attr = {
      .name_id = immediate_name,
      .value = loom_attr_i64(plan->control),
  };
  loom_op_t* control_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &plan->control_constant,
      loom_make_named_attr_slice(&control_attr, 1), control_type,
      source_op->location, &control_op));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 4, &result_type));
  loom_value_id_t operands[4];
  iree_host_size_t operand_count = 0;
  if (plan->operation == LOOM_AIE2P_MATRIX_OPERATION_ACCUMULATE) {
    operands[operand_count++] = init;
  }
  operands[operand_count++] = lhs;
  operands[operand_count++] = rhs;
  operands[operand_count++] = loom_low_const_result(control_op);
  loom_op_t* operation_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->operation_descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), &result_type, 1, NULL, 0,
      source_op->location, &operation_op));
  return loom_low_lower_bind_value(context, loom_vector_mma_result(source_op),
                                   loom_low_op_results(operation_op).values[0]);
}

static iree_status_t loom_aie2p_emit_matrix_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_aie2p_matrix_store_plan_t* plan) {
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_fragment_store_value(source_op), &value));
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_fragment_store_view(source_op), &view));

  loom_type_t quarter_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 1, &quarter_type));
  loom_string_id_t immediate_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("imm"), &immediate_name));

  for (uint32_t quarter = 0; quarter < 4; ++quarter) {
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slice_build(
        loom_low_lower_context_builder(context), value, quarter, quarter_type,
        source_op->location, &slice_op));
    const loom_value_id_t operands[] = {
        loom_low_slice_result(slice_op),
        view,
    };
    const loom_named_attr_t offset_attr = {
        .name_id = immediate_name,
        .value = loom_attr_i64((int64_t)quarter * 64),
    };
    loom_op_t* store_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->store, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(&offset_attr, 1), NULL, 0, NULL, 0,
        source_op->location, &store_op));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_emit_matrix_plan(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan) {
  switch ((loom_aie2p_matrix_plan_kind_t)plan.id) {
    case LOOM_AIE2P_MATRIX_PLAN_MMA_I8_M8N8K8:
      return loom_aie2p_emit_matrix_mma(
          context, source_op,
          (const loom_aie2p_matrix_mma_plan_t*)plan.target_data);
    case LOOM_AIE2P_MATRIX_PLAN_STORE_I32_M8N8:
      return loom_aie2p_emit_matrix_store(
          context, source_op,
          (const loom_aie2p_matrix_store_plan_t*)plan.target_data);
  }
  IREE_ASSERT_UNREACHABLE("AIE2P matrix emission has unknown plan kind");
  IREE_BUILTIN_UNREACHABLE();
}
