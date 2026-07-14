// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_LORA_H_
#define EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_LORA_H_

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/stages/ideogram4_dit_program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Authored dynamic LoRA update split around its producer dependency barrier.
typedef struct id4_ideogram4_dit_program_lora_application_t {
  // Static target being applied; NULL identifies an exact base projection.
  const id4_ideogram4_dit_lora_target_t* target;
  // BF16 low-rank activation produced by the down projection.
  id4_pipeline_program_tensor_t low_rank;
  // Adapter-major BF16 up-projection matrices assembled from all segments.
  id4_pipeline_program_tensor_t up;
  // I32 segment descriptors used to address |up| and issue-time strengths.
  id4_pipeline_program_tensor_t segment_metadata;
  // Issue-time F32 strength vector indexed by adapter ordinal.
  id4_pipeline_program_tensor_t strengths;
  // Initialized BF16 base projection updated in place.
  id4_pipeline_program_tensor_t output;
  // Number of initialized token rows processed by the update.
  uint32_t token_count;
  // Number of physically allocated token rows in |low_rank| and |output|.
  uint32_t token_capacity;
  // Number of output channels updated for each token.
  uint32_t output_size;
  // Number of ordered adapters represented by |strengths|.
  uint32_t adapter_count;
} id4_ideogram4_dit_program_lora_application_t;

// Validates a borrowed static LoRA topology before program authoring.
iree_status_t id4_ideogram4_dit_program_validate_lora_topology(
    id4_ideogram4_dit_lora_topology_t topology);

// Finds the target patching |base_parameter_key| or returns NULL.
const id4_ideogram4_dit_lora_target_t*
id4_ideogram4_dit_program_lookup_lora_target(
    id4_ideogram4_dit_lora_topology_t topology,
    iree_string_view_t base_parameter_key);

// Authors parameter assembly and the low-rank down projection.
//
// A NULL |target| authors no operations and returns an empty application. The
// caller may record the base projection before or after this call. Both must be
// separated from finish by a program barrier.
iree_status_t id4_ideogram4_dit_program_begin_lora(
    id4_pipeline_program_builder_t* builder, iree_string_view_t operation_name,
    const id4_ideogram4_dit_lora_target_t* target,
    iree_host_size_t adapter_count, id4_pipeline_program_tensor_t strengths,
    uint32_t token_count, uint32_t token_capacity, uint32_t input_size,
    uint32_t output_size, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t output, iree_host_size_t* target_use_count,
    id4_ideogram4_dit_program_lora_application_t* out_application);

// Authors the scaled up projection and adds it to the initialized base output.
//
// The caller must author a barrier after begin and before this call. Another
// barrier is required before any consumer of |application->output|.
iree_status_t id4_ideogram4_dit_program_finish_lora(
    id4_pipeline_program_builder_t* builder, iree_string_view_t operation_name,
    const id4_ideogram4_dit_program_lora_application_t* application);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_LORA_H_
