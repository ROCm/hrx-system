// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/diagnostics/wait_hazard.h"

#include "loom/target/arch/amdgpu/error_catalog.h"

iree_status_t loom_amdgpu_wait_hazard_emit(
    const loom_amdgpu_wait_hazard_t* hazard,
    const loom_diagnostic_sink_t* sink) {
  if (hazard == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait hazard is required");
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(hazard->diagnostic_code),
      loom_param_bool(hazard->has_kernel),
      loom_param_string(hazard->kernel_name),
      loom_param_u64(hazard->kernel_entry_offset),
      loom_param_string(hazard->counter_name),
      loom_param_string(hazard->access_kind),
      loom_param_string(hazard->register_class),
      loom_param_u32(hazard->register_index),
      loom_param_u32(hazard->register_width),
      loom_param_string(hazard->section_name),
      loom_param_u64(hazard->consumer_section_offset),
      loom_param_u64(hazard->consumer_file_offset),
      loom_param_string(hazard->consumer_instruction),
      loom_param_u64(hazard->producer_section_offset),
      loom_param_u64(hazard->producer_file_offset),
      loom_param_string(hazard->producer_instruction),
      loom_param_u32(hazard->required_count),
      loom_param_string(hazard->explanation),
  };
  const loom_source_range_t location = {
      .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
      .filename =
          hazard->has_kernel ? hazard->kernel_name : hazard->section_name,
  };
  const loom_diagnostic_t diagnostic = {
      .severity = LOOM_ERR_AMDGPU_045->severity,
      .error = LOOM_ERR_AMDGPU_045,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .emitter = LOOM_EMITTER_PASS,
      .origin = location,
      .source_location = location,
  };
  return loom_diagnostic_emit(sink, &diagnostic);
}
