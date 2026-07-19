// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured diagnostics for final-artifact AMDGPU wait hazards.

#ifndef LOOM_TARGET_ARCH_AMDGPU_DIAGNOSTICS_WAIT_HAZARD_H_
#define LOOM_TARGET_ARCH_AMDGPU_DIAGNOSTICS_WAIT_HAZARD_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"

#ifdef __cplusplus
extern "C" {
#endif

// One producer/consumer wait hazard decoded from a final AMDGPU artifact.
//
// String views are borrowed only for a synchronous diagnostic emission. Sinks
// must copy any diagnostic fields they retain, as required by
// loom_diagnostic_sink_t.
typedef struct loom_amdgpu_wait_hazard_t {
  // Stable machine-readable hazard classification.
  iree_string_view_t diagnostic_code;
  // True when the analyzer attributed this hazard to a kernel descriptor.
  bool has_kernel;
  // Kernel symbol name, or empty when no kernel identity is available.
  iree_string_view_t kernel_name;
  // Kernel entry-point byte offset in the artifact .text section.
  uint64_t kernel_entry_offset;
  // Stable hardware wait-counter name.
  iree_string_view_t counter_name;
  // Stable consumer access kind.
  iree_string_view_t access_kind;
  // Stable ISA register-file name.
  iree_string_view_t register_class;
  // First 32-bit register lane implicated in the hazard.
  uint32_t register_index;
  // Number of contiguous 32-bit register lanes implicated in the hazard.
  uint32_t register_width;
  // Artifact section containing the consumer instruction.
  iree_string_view_t section_name;
  // Consumer instruction byte offset within |section_name|.
  uint64_t consumer_section_offset;
  // Consumer instruction byte offset within the artifact file.
  uint64_t consumer_file_offset;
  // Decoded consumer instruction text.
  iree_string_view_t consumer_instruction;
  // Producer instruction byte offset within |section_name|.
  uint64_t producer_section_offset;
  // Producer instruction byte offset within the artifact file.
  uint64_t producer_file_offset;
  // Decoded producer instruction text.
  iree_string_view_t producer_instruction;
  // Wait count required before the consumer instruction.
  uint32_t required_count;
  // Analyzer-provided explanation of the hazard.
  iree_string_view_t explanation;
} loom_amdgpu_wait_hazard_t;

// Emits |hazard| through |sink| as the canonical AMDGPU wait diagnostic.
iree_status_t loom_amdgpu_wait_hazard_emit(
    const loom_amdgpu_wait_hazard_t* hazard,
    const loom_diagnostic_sink_t* sink);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_DIAGNOSTICS_WAIT_HAZARD_H_
