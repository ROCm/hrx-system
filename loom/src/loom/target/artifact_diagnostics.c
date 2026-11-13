// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_diagnostics.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"

iree_status_t loom_target_artifact_emit_requirement(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t format, iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {loom_param_string(format),
                                            loom_param_string(constraint)};
  return iree_diagnostic_emit(
      emitter,
      &(loom_diagnostic_emission_t){.op = op,
                                    .error = LOOM_ERR_BACKEND_055,
                                    .params = params,
                                    .param_count = IREE_ARRAYSIZE(params)});
}

iree_status_t loom_target_artifact_emit_symbol_requirement(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t format, iree_string_view_t name,
    iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {loom_param_string(format),
                                            loom_param_string(name),
                                            loom_param_string(constraint)};
  return iree_diagnostic_emit(
      emitter,
      &(loom_diagnostic_emission_t){.op = op,
                                    .error = LOOM_ERR_BACKEND_057,
                                    .params = params,
                                    .param_count = IREE_ARRAYSIZE(params)});
}

iree_status_t loom_target_artifact_emit_duplicate_export(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_op_t* previous_op, iree_string_view_t format,
    iree_string_view_t name) {
  const loom_diagnostic_param_t params[] = {loom_param_string(format),
                                            loom_param_string(name)};
  const loom_diagnostic_related_op_t previous = {
      .op = previous_op, .label = IREE_SV("previous export with this name")};
  return iree_diagnostic_emit(
      emitter,
      &(loom_diagnostic_emission_t){.op = op,
                                    .error = LOOM_ERR_BACKEND_056,
                                    .params = params,
                                    .param_count = IREE_ARRAYSIZE(params),
                                    .related_ops = &previous,
                                    .related_op_count = 1});
}

iree_status_t loom_target_artifact_emit_limit(iree_diagnostic_emitter_t emitter,
                                              const loom_op_t* op,
                                              iree_string_view_t format,
                                              iree_string_view_t quantity,
                                              uint64_t actual,
                                              uint64_t maximum) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(format), loom_param_string(quantity),
      loom_param_u64(actual), loom_param_u64(maximum)};
  return iree_diagnostic_emit(
      emitter,
      &(loom_diagnostic_emission_t){.op = op,
                                    .error = LOOM_ERR_BACKEND_060,
                                    .params = params,
                                    .param_count = IREE_ARRAYSIZE(params)});
}

iree_status_t loom_target_artifact_emit_symbol_offset(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t format, iree_string_view_t name, uint64_t offset,
    uint64_t length) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(format), loom_param_string(name),
      loom_param_u64(offset), loom_param_u64(length)};
  return iree_diagnostic_emit(
      emitter,
      &(loom_diagnostic_emission_t){.op = op,
                                    .error = LOOM_ERR_BACKEND_061,
                                    .params = params,
                                    .param_count = IREE_ARRAYSIZE(params)});
}
