// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/ops/pipeline/ops.h"

static iree_status_t loom_pipeline_emit(iree_diagnostic_emitter_t emitter,
                                        const loom_op_t* op,
                                        const loom_error_def_t* error,
                                        const loom_diagnostic_param_t* params,
                                        iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_pipeline_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t actual_field, uint32_t actual_count,
    iree_string_view_t expected_field, uint32_t expected_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(actual_field),
      loom_param_u32(actual_count),
      loom_param_string(expected_field),
      loom_param_u32(expected_count),
  };
  return loom_pipeline_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                            IREE_ARRAYSIZE(params));
}

iree_status_t loom_pipeline_def_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  const loom_func_like_t pipeline = loom_func_like_const_cast(module, op);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(pipeline, &argument_count);
  const int64_t specialization_count =
      loom_func_like_specialization_count(pipeline);
  if (specialization_count < 0 || specialization_count > argument_count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("specialization_count")),
        loom_param_i64(specialization_count),
        loom_param_string(IREE_SV("between zero and the argument count")),
    };
    return loom_pipeline_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                              IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_compose_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  const loom_symbol_ref_t callee = loom_pipeline_compose_callee(op);
  if (callee.module_id == 0 && callee.symbol_id < module->symbols.count) {
    const loom_func_like_t definition = loom_func_like_const_cast(
        module, module->symbols.entries[callee.symbol_id].defining_op);
    if (loom_func_like_isa(definition)) {
      uint16_t argument_count = 0;
      loom_func_like_arg_ids(definition, &argument_count);
      const int64_t specialization_count =
          loom_func_like_specialization_count(definition);
      if (specialization_count < 0 || specialization_count > argument_count) {
        // The definition owns this diagnostic, even if a use is visited first.
        return iree_ok_status();
      }
      const loom_value_slice_t specializations =
          loom_pipeline_compose_specializations(op);
      if (specialization_count != specializations.count) {
        return loom_pipeline_emit_count_mismatch(
            emitter, op, IREE_SV("specialization"), specializations.count,
            IREE_SV("pipeline specialization argument"),
            (uint32_t)specialization_count);
      }
    }
  }
  const loom_call_like_t call = loom_call_like_const_cast(module, op);
  return loom_function_call_contract_verify(
      module, op, callee, loom_call_like_operands(call),
      loom_call_like_results(call), 0, emitter);
}

static iree_status_t loom_pipeline_verify_worker_axis_list(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t field, loom_attribute_t static_values,
    uint16_t dynamic_count, uint32_t rank, int64_t minimum) {
  if (static_values.count != rank) {
    return loom_pipeline_emit_count_mismatch(
        emitter, op, field, static_values.count, IREE_SV("worker rank"), rank);
  }
  uint32_t expected_dynamic_count = 0;
  for (uint32_t i = 0; i < static_values.count; ++i) {
    const int64_t value = static_values.i64_array[i];
    if (value == INT64_MIN) {
      ++expected_dynamic_count;
    } else if (value < minimum) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(field),
          loom_param_i64(value),
          loom_param_string(
              minimum == 0 ? IREE_SV("nonnegative worker coordinate or count")
                           : IREE_SV("positive worker stride")),
      };
      return loom_pipeline_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                                IREE_ARRAYSIZE(params));
    }
  }
  if (dynamic_count != expected_dynamic_count) {
    return loom_pipeline_emit_count_mismatch(emitter, op, field, dynamic_count,
                                             IREE_SV("dynamic axis"),
                                             expected_dynamic_count);
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_memory_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  const loom_op_t* pipeline = op->parent_op;
  while (pipeline && !loom_pipeline_def_isa(pipeline)) {
    pipeline = pipeline->parent_op;
  }
  if (!pipeline || !loom_pipeline_def_has_scope(pipeline) ||
      loom_pipeline_def_scope(pipeline) != LOOM_PIPELINE_DEF_SCOPE_KERNEL) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(module, op)),
        loom_param_string(IREE_SV("required")),
        loom_param_string(IREE_SV("pipeline.def<kernel>")),
        loom_param_string(pipeline
                              ? IREE_SV("pipeline.def without kernel scope")
                              : IREE_SV("none")),
    };
    return loom_pipeline_emit(emitter, op, LOOM_ERR_STRUCTURE_029, params,
                              IREE_ARRAYSIZE(params));
  }
  const loom_attribute_t coordinates =
      loom_pipeline_memory_static_coordinates(op);
  if (coordinates.count == 0) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("worker rank")),
        loom_param_i64(0),
        loom_param_string(IREE_SV("at least one worker dimension")),
    };
    return loom_pipeline_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                              IREE_ARRAYSIZE(params));
  }
  return loom_pipeline_verify_worker_axis_list(
      op, emitter, IREE_SV("coordinates"), coordinates,
      loom_pipeline_memory_coordinates(op).count, coordinates.count, 0);
}

iree_status_t loom_pipeline_strand_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  (void)module;
  const loom_attribute_t origins = loom_pipeline_strand_static_origins(op);
  if (origins.count == 0) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("worker rank")),
        loom_param_i64(0),
        loom_param_string(IREE_SV("at least one worker dimension")),
    };
    return loom_pipeline_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                              IREE_ARRAYSIZE(params));
  }
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_worker_axis_list(
      op, emitter, IREE_SV("origins"), origins,
      loom_pipeline_strand_origins(op).count, origins.count, 0));
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_worker_axis_list(
      op, emitter, IREE_SV("counts"), loom_pipeline_strand_static_counts(op),
      loom_pipeline_strand_counts(op).count, origins.count, 0));
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_worker_axis_list(
      op, emitter, IREE_SV("strides"), loom_pipeline_strand_static_strides(op),
      loom_pipeline_strand_strides(op).count, origins.count, 1));
  const loom_block_t* entry =
      loom_region_const_entry_block(loom_pipeline_strand_body(op));
  if (entry->arg_count != 0) {
    return loom_pipeline_emit_count_mismatch(
        emitter, op, IREE_SV("strand entry argument"), entry->arg_count,
        IREE_SV("implicit capture argument"), 0);
  }
  return iree_ok_status();
}
