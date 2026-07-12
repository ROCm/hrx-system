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

// Parameter materialization strategy selected for one stage bundle.
typedef enum id4_ideogram4_generation_stage_prepare_mode_e {
  // Retains stage parameter loading work for issue-time submission.
  ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_DEFER_PARAMETERS = 0,
  // Submits stage parameter loading during stage-bundle preparation and leaves
  // the materialized slabs owned by the stage bundle.
  ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_MATERIALIZE_PARAMETERS = 1,
  // Submits stage parameter loading during stage-bundle preparation and retains
  // compatible materialized slabs in the session cache.
  ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_RETAIN_PARAMETERS = 2,
} id4_ideogram4_generation_stage_prepare_mode_t;

// Static metadata for one coarse generation stage.
typedef struct id4_ideogram4_generation_stage_descriptor_t {
  // Stage ordinal used by generation scheduling tables.
  id4_ideogram4_generation_stage_ordinal_t ordinal;
  // Stable generation-plan JSON key for the stage.
  const char* key;
  // Stage-bundle residency bit associated with this stage.
  id4_ideogram4_generation_resident_stage_mask_t resident_stage_bit;
} id4_ideogram4_generation_stage_descriptor_t;

// Prepared storage slot for one coarse generation stage.
typedef struct id4_ideogram4_generation_stage_slot_t {
  // Session-owned stage implementation selected for this slot.
  id4_pipeline_stage_t* stage;
  // Retained plan for this coarse stage.
  id4_pipeline_plan_t* plan;
  // Owned boundary tensor buffers in plan order.
  id4_pipeline_buffer_binding_set_t boundary_bindings;
  // Owned diagnostic tap buffers in plan order.
  id4_pipeline_buffer_binding_set_t diagnostic_tap_bindings;
} id4_ideogram4_generation_stage_slot_t;

// Prepared generation residency policy with explicit resource lifetimes.
typedef struct id4_ideogram4_generation_residency_policy_t {
  // Public residency mode used to construct this policy.
  id4_ideogram4_generation_residency_mode_t mode;
  // Stage bundles retained for the whole prepared generation bundle.
  id4_ideogram4_generation_resident_stage_mask_t request_stage_mask;
  // Stage bundles materialized for each phase and released after that phase.
  id4_ideogram4_generation_resident_stage_mask_t
      phase_stage_masks[ID4_IDEOGRAM4_GENERATION_PHASE_COUNT];
} id4_ideogram4_generation_residency_policy_t;

// Stage bundle reference retained by one generation issue path.
typedef struct id4_ideogram4_generation_stage_bundle_ref_t {
  // Prepared stage bundle used by the current generation issue path.
  id4_pipeline_bundle_t* bundle;
  // True when |bundle| is owned by this reference and must be released.
  bool owns_bundle;
  // Whether |bundle| has been submitted to HAL queue execution.
  bool was_issued;
} id4_ideogram4_generation_stage_bundle_ref_t;

// Boundary alias connecting producer and consumer stage bindings.
typedef struct id4_ideogram4_generation_boundary_alias_t {
  // Stage slot producing the replacement binding.
  id4_ideogram4_generation_stage_ordinal_t source_stage;
  // Boundary tensor exported by the producer stage.
  iree_string_view_t source_name;
  // Stage slot consuming the replacement binding.
  id4_ideogram4_generation_stage_ordinal_t target_stage;
  // Boundary tensor imported by the consumer stage.
  iree_string_view_t target_name;
} id4_ideogram4_generation_boundary_alias_t;

// Bitmask of private phase descriptor flags.
typedef uint32_t id4_ideogram4_generation_phase_flags_t;

// Private phase descriptor flag bits.
typedef enum id4_ideogram4_generation_phase_flag_bits_e {
  // Phase is issued once for every denoise step.
  ID4_IDEOGRAM4_GENERATION_PHASE_REPEATED_PER_DENOISE_STEP = 1u << 0,
} id4_ideogram4_generation_phase_flag_bits_t;

// Static metadata for one public generation issue phase.
typedef struct id4_ideogram4_generation_phase_descriptor_t {
  // Human-readable phase name for diagnostics and plan dumps.
  iree_string_view_t name;
  // Public phase mask bit represented by this descriptor.
  id4_ideogram4_generation_phase_mask_t phase_mask;
  // Phase behavior flags.
  id4_ideogram4_generation_phase_flags_t flags;
  // Number of stage ordinals in |stage_ordinals|.
  iree_host_size_t stage_count;
  // Stage ordinals issued by this phase.
  id4_ideogram4_generation_stage_ordinal_t stage_ordinals[3];
} id4_ideogram4_generation_phase_descriptor_t;

// Residency statistics accumulated while formatting generation plans.
typedef struct id4_ideogram4_generation_residency_statistics_t {
  // Sum of final parameter slab bytes across included stages.
  iree_device_size_t parameter_byte_length;
  // Largest final parameter slab byte length in any one included stage.
  iree_device_size_t largest_stage_parameter_byte_length;
  // Sum of embedded constant slab bytes across included stages.
  iree_device_size_t constant_byte_length;
  // Sum of local transient slab allocation bytes across included stages.
  iree_device_size_t local_slab_byte_length;
  // Sum of local transient high-water marks across included stages.
  iree_device_size_t local_high_water_mark;
  // Sum of planned stage boundary tensor bytes across included stages.
  iree_device_size_t stage_boundary_byte_length;
} id4_ideogram4_generation_residency_statistics_t;

// Session-owned resident slabs keyed by their immutable parameter source.
typedef struct id4_ideogram4_resident_parameter_cache_entry_t {
  // Resident execution-layout slabs reusable by compatible stage plans.
  id4_pipeline_parameter_slab_set_t* slabs;
  // Retained source identity used to populate |slabs|.
  id4_pipeline_parameter_source_t source;
  // Owned storage backing an execution-layout source scope string.
  char* execution_layout_scope_storage;
} id4_ideogram4_resident_parameter_cache_entry_t;

// Ideogram 4 model session owning coarse pipeline stages.
struct id4_ideogram4_session_t {
  // Reference count for shared session ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for session-owned storage.
  iree_allocator_t host_allocator;
  // Qwen3-VL model dimensions used by the text conditioning stage.
  id4_qwen3_vl_model_config_t qwen_model;
  // Qwen3-VL provider parameter format used by input lowering and stage
  // planning.
  id4_qwen3_vl_parameter_format_t qwen_parameter_format;
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
  id4_ideogram4_resident_parameter_cache_entry_t
      resident_stage_parameters[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
  // True after immutable session state has loaded.
  bool is_loaded;
};

// Planned generation assembled from session-owned coarse stages.
struct id4_ideogram4_generation_plan_t {
  // Allocator used for plan-owned storage.
  iree_allocator_t host_allocator;
  // Session retained for the lifetime of the generation plan.
  id4_ideogram4_session_t* session;
  // Stable generation-level shape and scheduling summary.
  id4_ideogram4_generation_plan_summary_t summary;
  // Planned Qwen3-VL prompt conditioning stage.
  id4_pipeline_plan_t* qwen_plan;
  // Planned conditioned Ideogram 4 DiT stage.
  id4_pipeline_plan_t* dit_conditioned_plan;
  // Planned unconditioned Ideogram 4 DiT stage.
  id4_pipeline_plan_t* dit_unconditioned_plan;
  // Planned device-side sampler initial-latent noise stage.
  id4_pipeline_plan_t* sampler_noise_plan;
  // Planned device-side sampler denoise-step stage.
  id4_pipeline_plan_t* sampler_denoise_plan;
  // Planned VAE-backed latent decode stage.
  id4_pipeline_plan_t* decode_plan;
};

// Prepared generation bundle assembled from one generation plan.
struct id4_ideogram4_generation_bundle_t {
  // Reference count for shared generation-bundle ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for bundle-owned storage.
  iree_allocator_t host_allocator;
  // Session retained for the lifetime of this prepared generation.
  id4_ideogram4_session_t* session;
  // Stable generation-level shape and scheduling summary.
  id4_ideogram4_generation_plan_summary_t summary;
  // Device shared by every prepared stage in the current single-device plan.
  iree_hal_device_t* device;
  // Queue affinity shared by every prepared stage in the current plan.
  iree_hal_queue_affinity_t queue_affinity;
  // Parameter sources retained for stage-bundle preparation.
  id4_ideogram4_generation_parameter_sources_t parameter_sources;
  // Owned packed storage backing execution-layout source scope strings.
  char* parameter_source_scope_storage;
  // Kernel library retained for stage-bundle preparation.
  id4_pipeline_kernel_library_t* kernel_library;
  // HAL command-buffer mode used when preparing stage bundles.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Stage-bundle residency policy selected during generation preparation.
  id4_ideogram4_generation_residency_policy_t residency_policy;
  // Maximum compact target bytes retained while issuing a deferred stage.
  iree_device_size_t maximum_parameter_window_byte_length;
  // Prepared coarse stage bundles retained by selected stage-bundle residency.
  id4_pipeline_bundle_t*
      resident_stage_bundles[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
  // Coarse stage slots owned by this generation bundle.
  id4_ideogram4_generation_stage_slot_t
      stages[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
};

// Prepared generation phase bundle assembled from one generation bundle.
struct id4_ideogram4_generation_phase_bundle_t {
  // Allocator used for heap-allocated public phase bundles.
  iree_allocator_t host_allocator;
  // Prepared generation bundle retained by public phase bundles.
  id4_ideogram4_generation_bundle_t* generation_bundle;
  // High-level generation phase represented by this bundle.
  id4_ideogram4_generation_phase_mask_t phase_mask;
  // Stage bundle references required while this phase is issued.
  id4_ideogram4_generation_stage_bundle_ref_t
      stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
};

// Asynchronous full-generation execution state.
struct id4_ideogram4_generation_execution_t {
  // Allocator used for execution-owned storage.
  iree_allocator_t host_allocator;
  // Prepared generation bundle retained while queued work may use it.
  id4_ideogram4_generation_bundle_t* bundle;
  // Deferred parameter load lookahead selected for stage issues.
  iree_host_size_t parameter_load_prefetch_segment_distance;
  // Stage issue flags forwarded to every coarse stage submission.
  id4_pipeline_stage_issue_flags_t stage_issue_flags;
  // Lowered Qwen prompt inputs used by the conditioning phase.
  id4_ideogram4_qwen_inputs_t qwen_inputs;
  // Lowered DiT conditioning and guidance inputs used by the denoise phase.
  id4_ideogram4_dit_inputs_t dit_inputs;
  // Lowered denoise schedule uploaded before each sampler step.
  id4_ideogram4_denoise_schedule_t denoise_schedule;
  // Semaphore chaining host-to-device request tensor uploads.
  iree_hal_semaphore_t* upload_semaphore;
  // Semaphore signaled when the Qwen stage completes.
  iree_hal_semaphore_t* qwen_done_semaphore;
  // Semaphore signaled when initial latent noise generation completes.
  iree_hal_semaphore_t* noise_done_semaphore;
  // Semaphore signaled when the conditioned DiT stage completes.
  iree_hal_semaphore_t* dit_conditioned_done_semaphore;
  // Semaphore signaled when the unconditioned DiT stage completes.
  iree_hal_semaphore_t* dit_unconditioned_done_semaphore;
  // Semaphore signaled when each sampler denoise step completes.
  iree_hal_semaphore_t* sampler_done_semaphore;
  // Semaphore signaled when the decode stage completes.
  iree_hal_semaphore_t* decode_done_semaphore;
  // Final payload value signaled on |upload_semaphore|.
  uint64_t upload_payload_value;
  // Upload payload value for Qwen conditioning inputs.
  uint64_t qwen_upload_payload_value;
  // Upload payload value for sampler seed input.
  uint64_t seed_upload_payload_value;
  // Conditioned DiT velocity binding retained from the conditioned DiT stage.
  iree_hal_buffer_binding_t conditioned_velocity_binding;
  // Unconditioned DiT velocity binding retained from the unconditioned DiT
  // stage.
  iree_hal_buffer_binding_t unconditioned_velocity_binding;
  // CFG denoised latent binding retained from the sampler stage.
  iree_hal_buffer_binding_t denoised_latent_binding;
  // Final diffusion latent binding retained from the sampler stage.
  iree_hal_buffer_binding_t final_latent_binding;
  // Final decoded image binding retained from the decode stage.
  iree_hal_buffer_binding_t decoded_image_binding;
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
