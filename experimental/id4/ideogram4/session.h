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
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/stages/ideogram4_dit_program.h"
#include "experimental/id4/stages/qwen3_vl_program.h"
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

// Opaque prepared generation phase bundle assembled from one generation bundle.
typedef struct id4_ideogram4_generation_phase_bundle_t
    id4_ideogram4_generation_phase_bundle_t;

// Opaque asynchronous full-generation execution handle.
typedef struct id4_ideogram4_generation_execution_t
    id4_ideogram4_generation_execution_t;

// Parameter scopes used by concrete Ideogram 4 stages.
typedef struct id4_ideogram4_session_parameter_scopes_t {
  // Scope containing Qwen3-VL text encoder weights.
  iree_string_view_t qwen;
  // Scope containing conditioned Ideogram 4 DiT weights.
  iree_string_view_t dit_conditioned;
  // Scope containing conditioned FP8 e4m3 Ideogram 4 DiT weights.
  iree_string_view_t dit_conditioned_fp8;
  // Scope containing unconditioned Ideogram 4 DiT weights.
  iree_string_view_t dit_unconditioned;
  // Scope containing unconditioned FP8 e4m3 Ideogram 4 DiT weights.
  iree_string_view_t dit_unconditioned_fp8;
  // Scope containing VAE weights.
  iree_string_view_t vae;
} id4_ideogram4_session_parameter_scopes_t;

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
  id4_ideogram4_dit_parameter_format_t dit_parameter_format;
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
  // Execution storage strategy selected for DiT linear weights.
  id4_ideogram4_dit_weight_execution_format_t dit_weight_execution_format;
  // Execution storage strategy selected for Qwen3-VL linear weights.
  id4_qwen3_vl_weight_execution_strategy_t qwen_weight_execution_strategy;
  // Attention implementation selected for DiT transformer blocks.
  id4_ideogram4_dit_attention_implementation_t dit_attention_implementation;
  // Feed-forward implementation selected for DiT transformer blocks.
  id4_ideogram4_dit_feed_forward_implementation_t
      dit_feed_forward_implementation;
  // VAE tiling policy used by the final latent decode stage.
  id4_vae_tiling_config_t vae_tiling;
} id4_ideogram4_generation_plan_policy_t;

// Diagnostic taps requested from one coarse generation stage.
typedef struct id4_ideogram4_generation_stage_diagnostic_tap_list_t {
  // Stable stage key from generation-plan JSON.
  iree_string_view_t stage_key;
  // Caller-owned diagnostic tap names requested from the stage.
  iree_string_view_list_t tap_names;
} id4_ideogram4_generation_stage_diagnostic_tap_list_t;

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
  // Number of stage-qualified diagnostic tap lists.
  iree_host_size_t stage_diagnostic_tap_list_count;
  // Caller-owned diagnostic tap selections grouped by generation stage key.
  const id4_ideogram4_generation_stage_diagnostic_tap_list_t*
      stage_diagnostic_tap_lists;
  // Diagnostics sink for plan events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_generation_plan_options_t;

// Stable summary of a session-owned generation plan.
typedef struct id4_ideogram4_generation_plan_summary_t {
  // Qwen prompt token positions used by conditioned DiT planning.
  uint32_t qwen_token_count;
  // BF16 packed token capacity used by Qwen linear kernels.
  uint32_t qwen_token_capacity;
  // Diffusion image token positions derived from the latent image grid.
  uint32_t image_token_count;
  // Combined text and image token positions used by conditioned DiT planning.
  uint32_t conditioned_dit_token_count;
  // BF16 packed token capacity used by conditioned DiT transformer blocks.
  uint32_t conditioned_dit_token_capacity;
  // Image-only token positions used by unconditioned DiT planning.
  uint32_t unconditioned_dit_token_count;
  // BF16 packed token capacity used by unconditioned DiT transformer blocks.
  uint32_t unconditioned_dit_token_capacity;
  // Number of denoise steps requested for the generation.
  uint32_t denoise_step_count;
  // Diffusion latent tensor shape shared by DiT, sampler, and decode stages.
  id4_pipeline_program_shape_t diffusion_latent_shape;
  // Decoded image tensor shape produced by the final decode stage.
  id4_pipeline_program_shape_t decoded_image_shape;
  // DiT activation storage format selected by the plan.
  id4_ideogram4_dit_activation_format_t dit_activation_format;
  // DiT linear weight execution strategy selected by the plan.
  id4_ideogram4_dit_weight_execution_format_t dit_weight_execution_format;
  // Qwen3-VL linear weight execution strategy selected by the plan.
  id4_qwen3_vl_weight_execution_strategy_t qwen_weight_execution_strategy;
  // DiT attention implementation selected by the plan.
  id4_ideogram4_dit_attention_implementation_t dit_attention_implementation;
  // DiT feed-forward implementation selected by the plan.
  id4_ideogram4_dit_feed_forward_implementation_t
      dit_feed_forward_implementation;
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

// Generation-stage bundle residency policy.
typedef uint32_t id4_ideogram4_generation_residency_mode_t;

// Generation-stage bundle residency policy values.
typedef enum id4_ideogram4_generation_residency_mode_e {
  // Invalid residency mode.
  ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_INVALID = 0,
  // Prepare heavy stage bundles at issue-time phase boundaries.
  ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES = 1,
  // Retain selected coarse stage bundles after first preparation.
  ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES = 2,
  // Prepare and retain every coarse stage bundle in the generation bundle.
  ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES = 3,
} id4_ideogram4_generation_residency_mode_e;

// Bitmask selecting high-level generation phases for phase issue.
typedef uint32_t id4_ideogram4_generation_phase_mask_t;

// High-level generation phases used by phase prepare and issue.
typedef enum id4_ideogram4_generation_phase_bit_e {
  // No generation phase selected.
  ID4_IDEOGRAM4_GENERATION_PHASE_NONE = 0u,
  // Prompt conditioning and initial latent-noise generation phase.
  ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING = 1u << 0,
  // Repeated DiT branch and sampler denoise-step phase.
  ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE = 1u << 1,
  // Final latent-to-image decode phase.
  ID4_IDEOGRAM4_GENERATION_PHASE_DECODE = 1u << 2,
  // Every high-level generation phase.
  ID4_IDEOGRAM4_GENERATION_PHASE_ALL =
      ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING |
      ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE |
      ID4_IDEOGRAM4_GENERATION_PHASE_DECODE,
} id4_ideogram4_generation_phase_bit_t;

// Full-generation issue scheduling policy.
typedef enum id4_ideogram4_generation_issue_policy_e {
  // Prepares and issues high-level generation phases, preserving branch
  // concurrency inside each phase.
  ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_PHASE_CONCURRENT = 0,
  // Prepares and issues heavyweight stages serially, waiting at semantic
  // boundaries so stage-owned parameter slabs can be released before the next
  // heavyweight stage is prepared.
  ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL = 1,
} id4_ideogram4_generation_issue_policy_t;

// Bitmask selecting coarse stage bundles retained across generation issues.
typedef uint32_t id4_ideogram4_generation_resident_stage_mask_t;

// Coarse generation stage bundles that may be retained.
typedef enum id4_ideogram4_generation_resident_stage_bit_e {
  // No coarse stage bundle selected for residency.
  ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE = 0u,
  // Qwen3-VL prompt-conditioning stage bundle.
  ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_QWEN = 1u << 0,
  // Initial latent-noise sampler stage bundle.
  ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_SAMPLER_NOISE = 1u << 1,
  // Conditioned DiT branch stage bundle.
  ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_CONDITIONED = 1u << 2,
  // Unconditioned DiT branch stage bundle.
  ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_UNCONDITIONED = 1u << 3,
  // Denoise-step sampler stage bundle.
  ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_SAMPLER_DENOISE = 1u << 4,
  // Final latent-to-image decode stage bundle.
  ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DECODE = 1u << 5,
  // Every coarse generation stage bundle.
  ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_ALL =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_QWEN |
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_SAMPLER_NOISE |
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_CONDITIONED |
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_UNCONDITIONED |
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_SAMPLER_DENOISE |
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DECODE,
} id4_ideogram4_generation_resident_stage_bit_t;

// Options for estimating generation resource lifetimes.
typedef struct id4_ideogram4_generation_resource_statistics_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stage-bundle residency policy being evaluated.
  id4_ideogram4_generation_residency_mode_t residency_mode;
  // Stage-bundle mask used when |residency_mode| selects explicit stages.
  id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask;
} id4_ideogram4_generation_resource_statistics_options_t;

// Logical resource lifetime statistics derived from a generation plan.
//
// These counters describe ID4-owned resources whose lifetimes are explicit in
// the generation schedule: boundary buffers, diagnostic tap buffers, prepared
// stage parameter/constant slabs, and queued local slab high-water marks. They
// intentionally exclude HAL allocator pool overhead and external parameter
// storage.
typedef struct id4_ideogram4_generation_resource_statistics_t {
  // Generation boundary buffers retained after planned aliases are applied.
  iree_device_size_t boundary_buffer_byte_length;
  // Diagnostic tap buffers retained by the generation bundle.
  iree_device_size_t diagnostic_tap_buffer_byte_length;
  // Resident prepared-stage parameter slab bytes from the selected policy.
  iree_device_size_t resident_stage_parameter_byte_length;
  // Resident prepared-stage constant slab bytes from the selected policy.
  iree_device_size_t resident_stage_constant_byte_length;
  // Resident prepared-stage parameter plus constant slab bytes.
  iree_device_size_t resident_stage_bundle_byte_length;
  // Largest phase-concurrent parameter slab live set.
  iree_device_size_t phase_concurrent_parameter_peak_byte_length;
  // Largest phase-concurrent constant slab live set.
  iree_device_size_t phase_concurrent_constant_peak_byte_length;
  // Largest phase-concurrent local slab high-water live set.
  iree_device_size_t phase_concurrent_local_peak_byte_length;
  // Largest phase-concurrent total live set excluding external parameter
  // storage.
  iree_device_size_t phase_concurrent_total_peak_byte_length;
  // Largest stage-serial parameter slab live set.
  iree_device_size_t stage_serial_parameter_peak_byte_length;
  // Largest stage-serial constant slab live set.
  iree_device_size_t stage_serial_constant_peak_byte_length;
  // Largest stage-serial local slab high-water live set.
  iree_device_size_t stage_serial_local_peak_byte_length;
  // Largest stage-serial total live set excluding external parameter storage.
  iree_device_size_t stage_serial_total_peak_byte_length;
} id4_ideogram4_generation_resource_statistics_t;

// Options for selecting a resident-stage policy from a generation plan.
typedef struct id4_ideogram4_generation_residency_select_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Issue policy whose live-memory peak must fit the budget.
  id4_ideogram4_generation_issue_policy_t issue_policy;
  // Candidate coarse stage bundles the selector may retain.
  id4_ideogram4_generation_resident_stage_mask_t candidate_stage_mask;
  // Maximum logical live bytes allowed by the selected policy.
  iree_device_size_t memory_budget_byte_length;
} id4_ideogram4_generation_residency_select_options_t;

// Selected resident-stage policy and its resource estimate.
typedef struct id4_ideogram4_generation_residency_selection_t {
  // Residency mode to pass to generation preparation.
  id4_ideogram4_generation_residency_mode_t residency_mode;
  // Resident stage mask to pass to generation preparation.
  id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask;
  // Logical live bytes for the selected issue policy.
  iree_device_size_t selected_peak_byte_length;
  // Resource statistics for the selected residency policy.
  id4_ideogram4_generation_resource_statistics_t resource_statistics;
} id4_ideogram4_generation_residency_selection_t;

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
  // Stage-bundle residency policy selected for this prepared generation.
  id4_ideogram4_generation_residency_mode_t residency_mode;
  // Coarse stage bundles retained when |residency_mode| selects residency.
  id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask;
  // HAL command-buffer mode used when preparing reusable regions.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Semaphores that generation preparation waits on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled after generation state is ready for issue.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Kernel diagnostic artifacts requested while preparing stage bundles.
  id4_pipeline_kernel_diagnostic_artifact_flags_t
      kernel_diagnostic_artifact_flags;
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
  // Policy controlling how the full generation is submitted.
  id4_ideogram4_generation_issue_policy_t issue_policy;
  // Kernel diagnostic artifacts requested for issue-time stage preparation.
  id4_pipeline_kernel_diagnostic_artifact_flags_t
      kernel_diagnostic_artifact_flags;
  // Semaphores that request uploads and generation execution wait on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled when the final decoded image is ready.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for issue events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_generation_issue_options_t;

// Options for beginning one phase-driven full-generation execution.
typedef struct id4_ideogram4_generation_begin_options_t {
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
  // Semaphores that generation begin waits on before request uploads.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled after request uploads are queued.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for begin events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_generation_begin_options_t;

// Options for preparing one high-level generation phase bundle.
typedef struct id4_ideogram4_generation_phase_prepare_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Single high-level generation phase to prepare.
  id4_ideogram4_generation_phase_mask_t phase_mask;
  // Semaphores that phase preparation waits on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled after phase preparation is ready for issue.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Kernel diagnostic artifacts requested while preparing stage bundles.
  id4_pipeline_kernel_diagnostic_artifact_flags_t
      kernel_diagnostic_artifact_flags;
  // Diagnostics sink for prepare events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_generation_phase_prepare_options_t;

// Options for issuing one prepared high-level generation phase.
typedef struct id4_ideogram4_generation_phase_issue_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Semaphores that phase issue waits on before queueing phase work.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled after the phase completes.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for issue events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_ideogram4_generation_phase_issue_options_t;

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
  // Kernel diagnostic artifacts requested while preparing this execution.
  id4_pipeline_kernel_diagnostic_artifact_flags_t
      kernel_diagnostic_artifact_flags;
  // Linear weight execution strategy selected for Qwen3-VL.
  id4_qwen3_vl_weight_execution_strategy_t qwen_weight_execution_strategy;
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
  // Conditioned DiT velocity binding retained by the execution handle.
  iree_hal_buffer_binding_t conditioned_velocity_binding;
  // Unconditioned DiT velocity binding retained by the execution handle.
  iree_hal_buffer_binding_t unconditioned_velocity_binding;
  // CFG denoised latent binding retained by the execution handle.
  iree_hal_buffer_binding_t denoised_latent_binding;
  // Final Euler latent binding retained by the execution handle.
  iree_hal_buffer_binding_t final_latent_binding;
  // Final decoded RGB image binding owned by the execution handle.
  iree_hal_buffer_binding_t decoded_image_binding;
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

// Returns the number of coarse stage plans retained by |plan|.
iree_host_size_t id4_ideogram4_generation_plan_stage_count(
    const id4_ideogram4_generation_plan_t* plan);

// Returns borrowed metadata for coarse stage plan |index|.
//
// |out_stage_key| receives the same stable key used by generation-plan JSON.
// |out_stage_plan| receives a borrowed plan valid until |plan| is released.
iree_status_t id4_ideogram4_generation_plan_stage_at(
    const id4_ideogram4_generation_plan_t* plan, iree_host_size_t index,
    iree_string_view_t* out_stage_key,
    const id4_pipeline_plan_t** out_stage_plan);

// Estimates logical resource lifetime peaks for |plan| and |options|.
iree_status_t id4_ideogram4_generation_plan_resource_statistics(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_resource_statistics_options_t* options,
    id4_ideogram4_generation_resource_statistics_t* out_statistics);

// Selects the highest-value resident-stage policy that fits a memory budget.
iree_status_t id4_ideogram4_generation_plan_select_residency(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_residency_select_options_t* options,
    id4_ideogram4_generation_residency_selection_t* out_selection);

// Appends an inspectable generation-plan JSON object to |builder|.
iree_status_t id4_ideogram4_generation_plan_format_json(
    const id4_ideogram4_generation_plan_t* plan,
    iree_string_builder_t* builder);

// Prepares reusable generation state for |plan|.
//
// |options->residency_mode| selects whether heavy stage bundles are prepared at
// issue-time phase boundaries, lazily retained by selected coarse stage
// bundles, or all retained by the prepared generation bundle. Selected resident
// stage bundles remove repeated parameter loading after their first preparation
// while allowing unrelated stages to release memory first.
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

// Begins one asynchronous phase-driven full-generation execution.
//
// Generation bundles have mutable boundary binding tables and must not be
// issued concurrently. The returned execution handle retains |bundle| until it
// is released.
iree_status_t id4_ideogram4_session_begin_generation(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_begin_options_t* options,
    id4_ideogram4_generation_execution_t** out_execution);

// Prepares one high-level phase bundle from |bundle|.
iree_status_t id4_ideogram4_generation_bundle_prepare_phase(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_phase_prepare_options_t* options,
    id4_ideogram4_generation_phase_bundle_t** out_phase_bundle);

// Issues one prepared high-level phase against |execution|.
iree_status_t id4_ideogram4_generation_execution_issue_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    const id4_ideogram4_generation_phase_issue_options_t* options);

// Releases |phase_bundle| after all queued work using it has completed.
iree_status_t id4_ideogram4_generation_phase_bundle_release(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle);

// Releases |execution| from the caller after its completion signal is reached.
void id4_ideogram4_generation_execution_release(
    id4_ideogram4_generation_execution_t* execution);

// Returns the result bindings retained by |execution|.
iree_status_t id4_ideogram4_generation_execution_result(
    const id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_result_t* out_result);

// Finds a captured diagnostic tap retained by |execution|.
//
// |out_layout| receives a borrowed tensor layout valid until |execution| is
// released. |out_binding| receives the issue-time buffer binding retained by
// the prepared generation bundle.
iree_status_t id4_ideogram4_generation_execution_find_diagnostic_tap(
    const id4_ideogram4_generation_execution_t* execution,
    iree_string_view_t stage_key, iree_string_view_t tap_name,
    const id4_pipeline_tensor_layout_t** out_layout,
    iree_hal_buffer_binding_t* out_binding);

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
