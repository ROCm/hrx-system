// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/util/fact_table.h"
#include "loom/util/math.h"

static iree_status_t loom_buffer_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_buffer_verify_strided_layout_rank(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_type_t result_type) {
  loom_value_facts_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_type_address_layout(
          /*context=*/NULL, module, result_type, static_strides,
          IREE_ARRAYSIZE(static_strides), &layout)) {
    return iree_ok_status();
  }
  if (layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    return iree_ok_status();
  }

  uint8_t result_rank = loom_type_rank(result_type);
  if (layout.rank == result_rank) return iree_ok_status();

  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("result type")),
      loom_param_i64(result_rank),
      loom_param_string(IREE_SV("layout stride list")),
      loom_param_i64(layout.rank),
  };
  return loom_buffer_emit(emitter, op, LOOM_ERR_SHAPE_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_buffer_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_buffer_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_buffer_verify_concrete_memory_space(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, loom_value_fact_memory_space_t value) {
  if (value != LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN) return iree_ok_status();
  return loom_buffer_emit_attribute_value_constraint(
      emitter, op, attr_name, value, IREE_SV("concrete memory space"));
}

static iree_status_t loom_buffer_verify_scratch_memory_space(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_value_fact_memory_space_t value) {
  if (value == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      value == LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    return iree_ok_status();
  }
  return loom_buffer_emit_attribute_value_constraint(
      emitter, op, IREE_SV("memory_space"), value,
      IREE_SV("workgroup or private memory space"));
}

static bool loom_buffer_try_get_local_memory_space_fact(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_fact_memory_space_t* out_memory_space) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;

  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) return false;

  if (loom_buffer_alloca_isa(defining_op)) {
    *out_memory_space = loom_buffer_alloca_memory_space(defining_op);
    return true;
  }
  if (loom_buffer_assume_memory_space_isa(defining_op)) {
    *out_memory_space =
        loom_buffer_assume_memory_space_memory_space(defining_op);
    return true;
  }
  return false;
}

static iree_status_t loom_buffer_verify_memory_space_refinement(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_fact_memory_space_t value) {
  loom_value_fact_memory_space_t existing_memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_buffer_try_get_local_memory_space_fact(
          module, loom_buffer_assume_memory_space_buffer(op),
          &existing_memory_space)) {
    return iree_ok_status();
  }
  if (existing_memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN) {
    return iree_ok_status();
  }
  if (existing_memory_space == value) return iree_ok_status();

  char expected[64] = {0};
  iree_snprintf(expected, sizeof(expected), "existing memory-space fact %u",
                (unsigned)existing_memory_space);
  return loom_buffer_emit_attribute_value_constraint(
      emitter, op, IREE_SV("memory_space"), value,
      iree_make_cstring_view(expected));
}

iree_status_t loom_buffer_alloca_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  int64_t base_alignment = loom_buffer_alloca_base_alignment(op);
  if (!loom_is_power_of_two_i64(base_alignment)) {
    return loom_buffer_emit_attribute_value_constraint(
        emitter, op, IREE_SV("base_alignment"), base_alignment,
        IREE_SV("positive power-of-two byte alignment"));
  }
  return loom_buffer_verify_scratch_memory_space(
      emitter, op, loom_buffer_alloca_memory_space(op));
}

iree_status_t loom_buffer_assume_memory_space_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  uint8_t memory_space = loom_buffer_assume_memory_space_memory_space(op);
  IREE_RETURN_IF_ERROR(loom_buffer_verify_concrete_memory_space(
      emitter, op, IREE_SV("memory_space"), memory_space));
  return loom_buffer_verify_memory_space_refinement(module, emitter, op,
                                                    memory_space);
}

iree_status_t loom_buffer_assume_alignment_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  int64_t minimum_alignment =
      loom_buffer_assume_alignment_minimum_alignment(op);
  if (loom_is_power_of_two_i64(minimum_alignment)) {
    return iree_ok_status();
  }
  return loom_buffer_emit_attribute_value_constraint(
      emitter, op, IREE_SV("minimum_alignment"), minimum_alignment,
      IREE_SV("positive power-of-two byte alignment"));
}

iree_status_t loom_buffer_view_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_buffer_view_result(op));
  if (!loom_type_is_view(result_type)) {
    return iree_ok_status();
  }

  if (!loom_type_has_encoding(result_type)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("result type layout")),
        loom_param_string(IREE_SV("view result type")),
    };
    return loom_buffer_emit(emitter, op, LOOM_ERR_ENCODING_001, params,
                            IREE_ARRAYSIZE(params));
  }

  return loom_buffer_verify_strided_layout_rank(module, op, emitter,
                                                result_type);
}
