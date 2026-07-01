// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_STATE_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_STATE_H_

#include <stdbool.h>
#include <stdint.h>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/pipeline/binding.h"
#include "experimental/id4/stages/ideogram4_decode.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Ordinals for the coarse generation stages owned by one session.
typedef enum id4_ideogram4_generation_stage_ordinal_e {
  // Device-side initial latent noise stage.
  ID4_IDEOGRAM4_GENERATION_STAGE_NOISE = 0,
  // Qwen3-VL prompt-conditioning stage.
  ID4_IDEOGRAM4_GENERATION_STAGE_QWEN = 1,
  // Conditioned Ideogram 4 DiT branch stage.
  ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED = 2,
  // Unconditioned Ideogram 4 DiT branch stage.
  ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED = 3,
  // Device-side Euler denoise-step sampler stage.
  ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER = 4,
  // VAE-backed latent-to-image decode stage.
  ID4_IDEOGRAM4_GENERATION_STAGE_DECODE = 5,
  // Number of coarse generation stages.
  ID4_IDEOGRAM4_GENERATION_STAGE_COUNT = 6,
} id4_ideogram4_generation_stage_ordinal_t;

// Ideogram 4 model session owning coarse pipeline stages.
struct id4_ideogram4_session_t {
  // Reference count for shared session ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for session-owned storage.
  iree_allocator_t host_allocator;
  // Qwen3-VL model dimensions used by the text conditioning stage.
  id4_qwen3_vl_model_config_t qwen_model;
  // Ideogram 4 DiT model dimensions used by conditioned/unconditioned stages.
  id4_ideogram4_dit_model_config_t dit_model;
  // Ideogram 4 latent-to-image decode model contract.
  id4_ideogram4_decode_model_config_t decode_model;
  // Coarse Qwen3-VL forward stage owned by the session.
  id4_pipeline_stage_t* qwen_stage;
  // Conditioned Ideogram 4 DiT forward stage owned by the session.
  id4_pipeline_stage_t* dit_conditioned_stage;
  // Unconditioned Ideogram 4 DiT forward stage owned by the session.
  id4_pipeline_stage_t* dit_unconditioned_stage;
  // Device-side sampler initial-latent noise stage owned by the session.
  id4_pipeline_stage_t* sampler_noise_stage;
  // Device-side sampler denoise-step stage owned by the session.
  id4_pipeline_stage_t* sampler_denoise_stage;
  // Ideogram 4 VAE-backed latent decode stage owned by the session.
  id4_pipeline_stage_t* decode_stage;
  // Resident parameter slabs retained across compatible generation plans.
  id4_pipeline_parameter_slab_set_t*
      resident_stage_parameter_slabs[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
  // True after immutable session state has loaded.
  bool is_loaded;
};

// Asynchronous Qwen3-VL execution state owned by standalone Qwen issue.
struct id4_ideogram4_qwen_execution_t {
  // Allocator used for execution-owned storage.
  iree_allocator_t host_allocator;
  // Session retained for the lifetime of this asynchronous execution.
  id4_ideogram4_session_t* session;
  // Plan retained for diagnostics and boundary metadata.
  id4_pipeline_plan_t* plan;
  // Prepared Qwen bundle retained until execution release.
  id4_pipeline_bundle_t* bundle;
  // Owned boundary tensor buffers in plan order.
  id4_pipeline_buffer_binding_set_t boundary_bindings;
  // Owned diagnostic tap buffers in plan order.
  id4_pipeline_buffer_binding_set_t diagnostic_tap_bindings;
  // Semaphore signaled when parameter loading and preparation complete.
  iree_hal_semaphore_t* prepare_semaphore;
  // Semaphore chaining host-to-device request tensor uploads.
  iree_hal_semaphore_t* upload_semaphore;
  // Final payload value signaled on |upload_semaphore|.
  uint64_t upload_payload_value;
  // Number of token positions in the execution.
  uint32_t token_count;
  // Exported condition tensor binding owned by |boundary_bindings|.
  iree_hal_buffer_binding_t condition_binding;
};

// Retains |session| for package-private asynchronous ownership.
void id4_ideogram4_session_retain(id4_ideogram4_session_t* session);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_STATE_H_
