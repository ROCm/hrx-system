// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/util/fact_table.h"

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
  if (layout.rank == result_rank) {
    return iree_ok_status();
  }

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

static iree_status_t loom_buffer_verify_restrict_reference_types(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_type_t operand_type =
        loom_module_value_type(module, operands[i]);
    char operand_name[32];
    iree_snprintf(operand_name, sizeof(operand_name), "reference operand %u",
                  i);
    if (!loom_type_is_buffer(operand_type) &&
        !loom_type_is_view(operand_type)) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(iree_make_cstring_view(operand_name)),
          loom_param_type(operand_type),
          loom_param_string(IREE_SV("buffer or view")),
      };
      return loom_buffer_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                              IREE_ARRAYSIZE(params));
    }
    const loom_type_t result_type = loom_module_value_type(module, results[i]);
    if (!loom_type_equal(operand_type, result_type)) {
      char result_name[32];
      iree_snprintf(result_name, sizeof(result_name), "result %u", i);
      const loom_diagnostic_param_t params[] = {
          loom_param_string(iree_make_cstring_view(operand_name)),
          loom_param_type(operand_type),
          loom_param_string(iree_make_cstring_view(result_name)),
          loom_param_type(result_type),
      };
      return loom_buffer_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                              IREE_ARRAYSIZE(params));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_buffer_verify_concrete_memory_space(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, loom_value_fact_memory_space_t value) {
  if (value != LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN) {
    return iree_ok_status();
  }
  return loom_buffer_emit_attribute_value_constraint(
      emitter, op, attr_name, value, IREE_SV("concrete memory space"));
}

static iree_status_t loom_buffer_verify_allocatable_memory_space(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_value_fact_memory_space_t value) {
  switch (value) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return iree_ok_status();
    default:
      return loom_buffer_emit_attribute_value_constraint(
          emitter, op, IREE_SV("memory_space"), value,
          IREE_SV("global, workgroup, private, host, or generic memory space"));
  }
}

static bool loom_buffer_try_get_local_memory_space_fact(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_fact_memory_space_t* out_memory_space) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }

  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return false;
  }

  if (loom_buffer_alloca_isa(defining_op)) {
    *out_memory_space = loom_buffer_alloca_memory_space(defining_op);
    return true;
  }
  if (loom_buffer_alloca_init_isa(defining_op)) {
    *out_memory_space = loom_buffer_alloca_init_memory_space(defining_op);
    return true;
  }
  if (loom_buffer_constant_isa(defining_op)) {
    *out_memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT;
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
  if (existing_memory_space == value) {
    return iree_ok_status();
  }

  char expected[64] = {0};
  iree_snprintf(expected, sizeof(expected), "existing memory-space fact %u",
                (unsigned)existing_memory_space);
  return loom_buffer_emit_attribute_value_constraint(
      emitter, op, IREE_SV("memory_space"), value,
      iree_make_cstring_view(expected));
}

static iree_status_t loom_buffer_verify_alignment(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t name, int64_t alignment) {
  if (!iree_math_is_power_of_two_i64(alignment)) {
    return loom_buffer_emit_attribute_value_constraint(
        emitter, op, name, alignment,
        IREE_SV("positive power-of-two byte alignment"));
  }
  return iree_ok_status();
}

iree_status_t loom_buffer_constant_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  if (!loom_buffer_constant_has_alignment(op)) {
    return iree_ok_status();
  }
  return loom_buffer_verify_alignment(emitter, op, IREE_SV("alignment"),
                                      loom_buffer_constant_alignment(op));
}

iree_status_t loom_buffer_alloca_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(
      loom_buffer_verify_alignment(emitter, op, IREE_SV("base_alignment"),
                                   loom_buffer_alloca_base_alignment(op)));
  return loom_buffer_verify_allocatable_memory_space(
      emitter, op, loom_buffer_alloca_memory_space(op));
}

loom_trait_flags_t loom_buffer_alloca_init_effective_traits(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_trait_flags_t collective_traits =
      LOOM_TRAIT_CONVERGENT | LOOM_TRAIT_MEMORY_FENCE;
  const loom_trait_flags_t traits = op->traits & ~collective_traits;
  return loom_buffer_alloca_init_memory_space(op) ==
                 LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
             ? traits | collective_traits
             : traits;
}

iree_status_t loom_buffer_alloca_init_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(
      loom_buffer_verify_alignment(emitter, op, IREE_SV("base_alignment"),
                                   loom_buffer_alloca_init_base_alignment(op)));
  return loom_buffer_verify_allocatable_memory_space(
      emitter, op, loom_buffer_alloca_init_memory_space(op));
}

iree_status_t loom_buffer_pack_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t byte_lengths = loom_buffer_pack_byte_lengths(op);
  if (byte_lengths.count == 0) {
    return loom_buffer_emit_attribute_value_constraint(
        emitter, op, IREE_SV("byte_lengths"), 0,
        IREE_SV("at least one physical byte range"));
  }

  loom_attribute_t minimum_alignments = loom_buffer_pack_minimum_alignments(op);
  if (minimum_alignments.count != byte_lengths.count) {
    return loom_buffer_emit_attribute_value_constraint(
        emitter, op, IREE_SV("minimum_alignments"), minimum_alignments.count,
        IREE_SV("one entry per byte length"));
  }
  for (uint16_t i = 0; i < minimum_alignments.count; ++i) {
    const int64_t minimum_alignment = minimum_alignments.i64_array[i];
    if (iree_math_is_power_of_two_i64(minimum_alignment)) {
      continue;
    }
    char attribute_name[64] = {0};
    iree_snprintf(attribute_name, sizeof(attribute_name),
                  "minimum_alignments[%u]", (unsigned)i);
    return loom_buffer_emit_attribute_value_constraint(
        emitter, op, iree_make_cstring_view(attribute_name), minimum_alignment,
        IREE_SV("positive power-of-two byte alignment"));
  }
  return iree_ok_status();
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
  if (iree_math_is_power_of_two_i64(minimum_alignment)) {
    return iree_ok_status();
  }
  return loom_buffer_emit_attribute_value_constraint(
      emitter, op, IREE_SV("minimum_alignment"), minimum_alignment,
      IREE_SV("positive power-of-two byte alignment"));
}

iree_status_t loom_buffer_restrict_begin_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_value_slice_t restricted =
      loom_buffer_restrict_begin_restricted(op);
  if (restricted.count == 0) {
    return loom_buffer_emit_attribute_value_constraint(
        emitter, op, IREE_SV("restricted operand count"), 0,
        IREE_SV("at least one restricted reference"));
  }
  if (op->operand_count == op->result_count) {
    return loom_buffer_verify_restrict_reference_types(module, op, emitter);
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("restricted and peer operands")),
      loom_param_u32(op->operand_count),
      loom_param_string(IREE_SV("results")),
      loom_param_u32(op->result_count),
  };
  return loom_buffer_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                          IREE_ARRAYSIZE(params));
}

iree_status_t loom_buffer_restrict_end_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_buffer_verify_restrict_reference_types(module, op, emitter);
}

iree_status_t loom_buffer_view_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_buffer_view_result(op));
  if (!loom_type_is_view(result_type)) {
    return iree_ok_status();
  }

  return loom_buffer_verify_strided_layout_rank(module, op, emitter,
                                                result_type);
}
