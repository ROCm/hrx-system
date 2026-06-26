// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_H_

#include "experimental/id4/ideogram4/request.h"
#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/ideogram4_dit_program.h"
#include "experimental/id4/stages/vae_program.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque Ideogram 4 model session owning coarse pipeline stages.
typedef struct id4_ideogram4_session_t id4_ideogram4_session_t;

// Opaque asynchronous Qwen3-VL execution handle.
typedef struct id4_ideogram4_qwen_execution_t id4_ideogram4_qwen_execution_t;

// Opaque planned generation assembled from session-owned coarse stages.
typedef struct id4_ideogram4_generation_plan_t id4_ideogram4_generation_plan_t;

// Opaque prepared generation bundle assembled from one generation plan.
typedef struct id4_ideogram4_generation_bundle_t
    id4_ideogram4_generation_bundle_t;

// Opaque asynchronous full-generation execution handle.
typedef struct id4_ideogram4_generation_execution_t
    id4_ideogram4_generation_execution_t;

// Parameter scopes used by concrete Ideogram 4 stages.
typedef struct id4_ideogram4_session_parameter_scopes_t {
  // Scope containing Qwen3-VL text encoder weights.
  iree_string_view_t qwen;
  // Scope containing conditioned Ideogram 4 DiT weights.
  iree_string_view_t dit_conditioned;
  // Scope containing conditioned native-FP8 Ideogram 4 DiT weights.
  iree_string_view_t dit_conditioned_fp8;
  // Scope containing unconditioned Ideogram 4 DiT weights.
  iree_string_view_t dit_unconditioned;
  // Scope containing unconditioned native-FP8 Ideogram 4 DiT weights.
  iree_string_view_t dit_unconditioned_fp8;
  // Scope containing VAE weights.
  iree_string_view_t vae;
} id4_ideogram4_session_parameter_scopes_t;

// DiT parameter storage policy selected for session-owned DiT stages.
typedef enum id4_ideogram4_session_dit_parameter_format_e {
  // Invalid DiT parameter storage policy.
  ID4_IDEOGRAM4_SESSION_DIT_PARAMETER_FORMAT_INVALID = 0,
  // All DiT parameters are sourced from BF16-expanded parameter scopes.
  ID4_IDEOGRAM4_SESSION_DIT_PARAMETER_FORMAT_BF16 = 1,
  // Supported DiT weights are sourced from native scaled FP8 scopes while
  // other DiT parameters remain sourced from BF16-expanded scopes.
  ID4_IDEOGRAM4_SESSION_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3 = 2,
} id4_ideogram4_session_dit_parameter_format_t;

// Options for creating an Ideogram 4 model session.
typedef struct id4_ideogram4_session_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Shared pipeline services used by all session stages.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used by session stages during preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Parameter scopes selected by session-owned stages.
  id4_ideogram4_session_parameter_scopes_t parameter_scopes;
  // DiT parameter storage policy selected by session-owned DiT stages.
  id4_ideogram4_session_dit_parameter_format_t dit_parameter_format;
  // Activation storage format selected for the session-owned VAE decode stage.
  id4_vae_activation_format_t vae_activation_format;
} id4_ideogram4_session_create_options_t;

// Options for loading immutable model session state.
typedef struct id4_ideogram4_session_load_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Diagnostics sink for load events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_session_load_options_t;

// Target and representation policy selected when planning a generation.
typedef struct id4_ideogram4_generation_plan_policy_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Activation storage format selected for DiT intermediates.
  id4_ideogram4_dit_activation_format_t dit_activation_format;
  // VAE tiling policy used by the final latent decode stage.
  id4_vae_tiling_config_t vae_tiling;
} id4_ideogram4_generation_plan_policy_t;

// Options for creating a session-owned generation plan from one prompt.
typedef struct id4_ideogram4_generation_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Parsed request supplying structured prompt JSON.
  const id4_ideogram4_request_t* request;
  // Tokenizer used to derive Qwen prompt length from |request|.
  const iree_tokenizer_t* tokenizer;
  // Tokenizer flags used while encoding the Qwen prompt text.
  iree_tokenizer_encode_flags_t tokenizer_flags;
  // Target and representation policy for this generation plan.
  id4_ideogram4_generation_plan_policy_t policy;
  // Device index within the session device group used by every stage plan.
  iree_host_size_t device_index;
  // Queue affinity used by every stage plan.
  iree_hal_queue_affinity_t queue_affinity;
  // Diagnostics sink for plan events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_generation_plan_options_t;

// Stable summary of a session-owned generation plan.
typedef struct id4_ideogram4_generation_plan_summary_t {
  // Qwen prompt token positions used by conditioned DiT planning.
  uint32_t qwen_token_count;
  // Number of denoise steps requested for the generation.
  uint32_t denoise_step_count;
  // Diffusion latent tensor shape shared by DiT, sampler, and decode stages.
  id4_pipeline_program_shape_t diffusion_latent_shape;
  // Decoded image tensor shape produced by the final decode stage.
  id4_pipeline_program_shape_t decoded_image_shape;
  // DiT activation storage format selected by the plan.
  id4_ideogram4_dit_activation_format_t dit_activation_format;
  // VAE tiling policy selected by the plan.
  id4_vae_tiling_config_t vae_tiling;
} id4_ideogram4_generation_plan_summary_t;

// Parameter providers used when preparing one generation plan.
typedef struct id4_ideogram4_generation_parameter_providers_t {
  // Provider containing Qwen3-VL text encoder weights.
  iree_io_parameter_provider_t* qwen;
  // Provider containing conditioned Ideogram 4 DiT weights.
  iree_io_parameter_provider_t* dit_conditioned;
  // Provider containing unconditioned Ideogram 4 DiT weights.
  iree_io_parameter_provider_t* dit_unconditioned;
  // Provider containing VAE decode weights.
  iree_io_parameter_provider_t* vae;
} id4_ideogram4_generation_parameter_providers_t;

// Options for preparing reusable generation state from one generation plan.
typedef struct id4_ideogram4_generation_prepare_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Parameter providers selected for each session-owned model component.
  id4_ideogram4_generation_parameter_providers_t parameter_providers;
  // Kernel library used to resolve planned Loom module paths.
  id4_pipeline_kernel_library_t* kernel_library;
  // HAL command-buffer mode used when preparing reusable regions.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Semaphores that generation preparation waits on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled after generation state is ready for issue.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for prepare events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_generation_prepare_options_t;

// Options for issuing one prepared full-generation execution.
typedef struct id4_ideogram4_generation_issue_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Parsed request matching the prepared generation bundle shape.
  const id4_ideogram4_request_t* request;
  // Tokenizer used to lower the request's Qwen prompt into Qwen inputs.
  const iree_tokenizer_t* tokenizer;
  // Tokenizer flags used while encoding the Qwen prompt text.
  iree_tokenizer_encode_flags_t tokenizer_flags;
  // Semaphores that request uploads and generation execution wait on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled when the final decoded image is ready.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for issue events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_generation_issue_options_t;

// Options for issuing one asynchronous Qwen3-VL conditioning execution.
typedef struct id4_ideogram4_qwen_issue_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Parsed request supplying structured prompt JSON.
  const id4_ideogram4_request_t* request;
  // Tokenizer used to lower the request's Qwen prompt into Qwen inputs.
  const iree_tokenizer_t* tokenizer;
  // Tokenizer flags used while encoding the Qwen prompt text.
  iree_tokenizer_encode_flags_t tokenizer_flags;
  // Parameter provider containing Qwen weights in the session parameter scope.
  iree_io_parameter_provider_t* parameter_provider;
  // Kernel library used to resolve planned Loom module paths.
  id4_pipeline_kernel_library_t* kernel_library;
  // Device index within the session device group used for this Qwen plan.
  iree_host_size_t device_index;
  // Queue affinity used by Qwen preparation, uploads, and execution.
  iree_hal_queue_affinity_t queue_affinity;
  // HAL command-buffer mode used when preparing reusable regions.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Caller-owned diagnostic tap names to capture.
  iree_string_view_list_t diagnostic_tap_names;
  // Semaphores that preparation, input upload, and execution wait on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled when Qwen execution completes.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for plan, prepare, and issue events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_qwen_issue_options_t;

// Result bindings produced by one Qwen3-VL execution.
typedef struct id4_ideogram4_qwen_result_t {
  // Number of prompt token positions in the execution.
  uint32_t token_count;
  // Exported condition tensor binding owned by the execution handle.
  iree_hal_buffer_binding_t condition_binding;
} id4_ideogram4_qwen_result_t;

// Result bindings produced by one full-generation execution.
typedef struct id4_ideogram4_generation_result_t {
  // Final decoded RGB image binding owned by the execution handle.
  iree_hal_buffer_binding_t decoded_image_binding;
  // Final diffusion latent binding retained by the execution handle.
  iree_hal_buffer_binding_t final_latent_binding;
} id4_ideogram4_generation_result_t;

// Creates an Ideogram 4 model session.
iree_status_t id4_ideogram4_session_create(
    const id4_ideogram4_session_create_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_session_t** out_session);

// Releases |session| from the caller.
void id4_ideogram4_session_release(id4_ideogram4_session_t* session);

// Loads immutable stage state for |session|.
iree_status_t id4_ideogram4_session_load(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_session_load_options_t* options);

// Creates a generation plan for one parsed full-generation request.
iree_status_t id4_ideogram4_session_plan_generation(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_generation_plan_t** out_plan);

// Releases |plan| from the caller.
void id4_ideogram4_generation_plan_release(
    id4_ideogram4_generation_plan_t* plan);

// Returns a stable summary for |plan|.
iree_status_t id4_ideogram4_generation_plan_summary(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_plan_summary_t* out_summary);

// Appends an inspectable generation-plan JSON object to |builder|.
iree_status_t id4_ideogram4_generation_plan_format_json(
    const id4_ideogram4_generation_plan_t* plan,
    iree_string_builder_t* builder);

// Prepares reusable generation boundary state for |plan|.
//
// Heavy stage bundles and their parameter slabs are phase-resident: they are
// prepared during issue when enough memory is available, submitted to the HAL,
// and then released from host ownership after the queue captures their
// resources.
iree_status_t id4_ideogram4_session_prepare_generation(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_prepare_options_t* options,
    id4_ideogram4_generation_bundle_t** out_bundle);

// Releases |bundle| after all queued work using it has completed.
void id4_ideogram4_generation_bundle_release(
    id4_ideogram4_generation_bundle_t* bundle);

// Issues one asynchronous full-generation execution from |bundle|.
//
// Generation bundles have mutable boundary binding tables and must not be
// issued concurrently. The returned execution handle retains |bundle| until it
// is released.
iree_status_t id4_ideogram4_session_issue_generation(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_issue_options_t* options,
    id4_ideogram4_generation_execution_t** out_execution);

// Releases |execution| from the caller after its completion signal is reached.
void id4_ideogram4_generation_execution_release(
    id4_ideogram4_generation_execution_t* execution);

// Returns the result bindings retained by |execution|.
iree_status_t id4_ideogram4_generation_execution_result(
    const id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_result_t* out_result);

// Issues one asynchronous Qwen3-VL conditioning execution.
iree_status_t id4_ideogram4_session_issue_qwen(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t** out_execution);

// Releases |execution| from the caller after its completion signal is reached.
void id4_ideogram4_qwen_execution_release(
    id4_ideogram4_qwen_execution_t* execution);

// Returns the plan retained by |execution|.
const id4_pipeline_plan_t* id4_ideogram4_qwen_execution_plan(
    const id4_ideogram4_qwen_execution_t* execution);

// Returns the result bindings retained by |execution|.
iree_status_t id4_ideogram4_qwen_execution_result(
    const id4_ideogram4_qwen_execution_t* execution,
    id4_ideogram4_qwen_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_H_
