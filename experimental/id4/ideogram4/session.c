// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/session.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/binding.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/stages/sampler.h"

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
  // True after immutable session state has loaded.
  bool is_loaded;
};

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

typedef enum id4_ideogram4_generation_stage_ordinal_e {
  ID4_IDEOGRAM4_GENERATION_STAGE_NOISE = 0,
  ID4_IDEOGRAM4_GENERATION_STAGE_QWEN = 1,
  ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED = 2,
  ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED = 3,
  ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER = 4,
  ID4_IDEOGRAM4_GENERATION_STAGE_DECODE = 5,
  ID4_IDEOGRAM4_GENERATION_STAGE_COUNT = 6,
} id4_ideogram4_generation_stage_ordinal_t;

typedef struct id4_ideogram4_generation_stage_descriptor_t {
  // Stage ordinal used by generation scheduling tables.
  id4_ideogram4_generation_stage_ordinal_t ordinal;
  // Stable generation-plan JSON key for the stage.
  const char* key;
  // Stage-bundle residency bit associated with this stage.
  id4_ideogram4_generation_resident_stage_mask_t resident_stage_bit;
} id4_ideogram4_generation_stage_descriptor_t;

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

typedef struct id4_ideogram4_generation_stage_bundle_ref_t {
  // Prepared stage bundle used by the current generation issue path.
  id4_pipeline_bundle_t* bundle;
  // True when |bundle| is owned by this reference and must be released.
  bool owns_bundle;
  // Whether |bundle| has been submitted to HAL queue execution.
  bool was_issued;
} id4_ideogram4_generation_stage_bundle_ref_t;

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

typedef uint32_t id4_ideogram4_generation_phase_flags_t;

typedef enum id4_ideogram4_generation_phase_flag_bits_e {
  // Phase is issued once for every denoise step.
  ID4_IDEOGRAM4_GENERATION_PHASE_REPEATED_PER_DENOISE_STEP = 1u << 0,
} id4_ideogram4_generation_phase_flag_bits_t;

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

typedef struct id4_ideogram4_boundary_upload_context_t {
  // Device receiving the queued upload operations.
  iree_hal_device_t* device;
  // Queue affinity used for every upload in this context.
  iree_hal_queue_affinity_t queue_affinity;
  // Plan whose boundary tensor table is searched by name.
  const id4_pipeline_plan_t* plan;
  // Binding set whose entries are updated by name.
  const id4_pipeline_buffer_binding_set_t* boundary_bindings;
  // Semaphore chaining the upload queue operations.
  iree_hal_semaphore_t* semaphore;
  // Payload value mutated after every queued update.
  uint64_t* payload_value;
} id4_ideogram4_boundary_upload_context_t;

static const id4_ideogram4_generation_stage_descriptor_t
    id4_ideogram4_generation_stage_descriptors[] = {
        {
            // Initial latent noise stage.
            .ordinal = ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
            .key = "sampler_noise",
            .resident_stage_bit =
                ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_SAMPLER_NOISE,
        },
        {
            // Qwen text conditioning stage.
            .ordinal = ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
            .key = "qwen",
            .resident_stage_bit = ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_QWEN,
        },
        {
            // Conditioned DiT denoise branch stage.
            .ordinal = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
            .key = "dit_conditioned",
            .resident_stage_bit =
                ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_CONDITIONED,
        },
        {
            // Unconditioned DiT denoise branch stage.
            .ordinal = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
            .key = "dit_unconditioned",
            .resident_stage_bit =
                ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_UNCONDITIONED,
        },
        {
            // CFG sampler denoise-step stage.
            .ordinal = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .key = "sampler_denoise",
            .resident_stage_bit =
                ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_SAMPLER_DENOISE,
        },
        {
            // VAE-backed latent decode stage.
            .ordinal = ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
            .key = "decode",
            .resident_stage_bit =
                ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DECODE,
        },
};

static const id4_ideogram4_generation_stage_descriptor_t*
id4_ideogram4_generation_stage_descriptor(
    id4_ideogram4_generation_stage_ordinal_t ordinal) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (descriptor->ordinal == ordinal) return descriptor;
  }
  return NULL;
}

static const id4_ideogram4_generation_stage_descriptor_t*
id4_ideogram4_generation_stage_descriptor_for_key(
    iree_string_view_t stage_key) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (iree_string_view_equal(iree_make_cstring_view(descriptor->key),
                               stage_key)) {
      return descriptor;
    }
  }
  return NULL;
}

static const id4_ideogram4_generation_phase_descriptor_t
    id4_ideogram4_generation_phase_descriptors[] = {
        {
            // Phase that produces prompt conditioning and initial latent noise.
            .name = IREE_SVL("conditioning"),
            .phase_mask = ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING,
            .flags = 0,
            .stage_count = 2,
            .stage_ordinals =
                {
                    ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
                    ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
                },
        },
        {
            // Phase that repeats the conditioned/unconditioned DiT branches.
            .name = IREE_SVL("denoise"),
            .phase_mask = ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE,
            .flags = ID4_IDEOGRAM4_GENERATION_PHASE_REPEATED_PER_DENOISE_STEP,
            .stage_count = 3,
            .stage_ordinals =
                {
                    ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
                    ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
                    ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
                },
        },
        {
            // Phase that decodes the final latent into an image.
            .name = IREE_SVL("decode"),
            .phase_mask = ID4_IDEOGRAM4_GENERATION_PHASE_DECODE,
            .flags = 0,
            .stage_count = 1,
            .stage_ordinals =
                {
                    ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
                },
        },
};

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
  // Parameter providers retained for stage-bundle preparation.
  id4_ideogram4_generation_parameter_providers_t parameter_providers;
  // Kernel library retained for stage-bundle preparation.
  id4_pipeline_kernel_library_t* kernel_library;
  // HAL command-buffer mode used when preparing stage bundles.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Stage-bundle residency policy selected during generation preparation.
  id4_ideogram4_generation_residency_mode_t residency_mode;
  // Coarse stage bundles retained by this generation bundle.
  id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask;
  // Prepared coarse stage bundles retained by selected stage-bundle residency.
  id4_pipeline_bundle_t*
      resident_stage_bundles[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
  // Coarse stage slots owned by this generation bundle.
  id4_ideogram4_generation_stage_slot_t
      stages[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
};

struct id4_ideogram4_generation_execution_t {
  // Allocator used for execution-owned storage.
  iree_allocator_t host_allocator;
  // Prepared generation bundle retained while queued work may use it.
  id4_ideogram4_generation_bundle_t* bundle;
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

static const id4_ideogram4_generation_boundary_alias_t
    id4_ideogram4_generation_boundary_aliases[] = {
        {
            // Initial latent noise consumed by the conditioned DiT branch.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
            .source_name = IREE_SVL("x"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
            .target_name = IREE_SVL("x"),
        },
        {
            // Initial latent noise consumed by the unconditioned DiT branch.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
            .source_name = IREE_SVL("x"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
            .target_name = IREE_SVL("x"),
        },
        {
            // Initial latent noise consumed as the sampler input state.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
            .source_name = IREE_SVL("x"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .target_name = IREE_SVL("x_t"),
        },
        {
            // Initial latent noise updated in place by the sampler output.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
            .source_name = IREE_SVL("x"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .target_name = IREE_SVL("x_next"),
        },
        {
            // Qwen text conditioning consumed by the conditioned DiT branch.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
            .source_name = IREE_SVL("condition"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
            .target_name = IREE_SVL("condition"),
        },
        {
            // Conditioned DiT velocity consumed by CFG sampling.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
            .source_name = IREE_SVL("velocity"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .target_name = IREE_SVL("cond_out"),
        },
        {
            // Unconditioned DiT velocity consumed by CFG sampling.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
            .source_name = IREE_SVL("velocity"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .target_name = IREE_SVL("uncond_out"),
        },
        {
            // Final Euler latent consumed by VAE-backed decode.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .source_name = IREE_SVL("x_next"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
            .target_name = IREE_SVL("media.latent.diffusion"),
        },
};

static iree_status_t id4_ideogram4_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_ideogram4_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (!semaphore_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore array is required",
                            (int)list_name.size, list_name.data);
  }
  if (!semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s payload value array is required",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!semaphore_list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_validate_diagnostic_tap_names(
    iree_string_view_list_t names) {
  if (names.count == 0) {
    if (names.values) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "diagnostic tap name array requires at least one tap name");
    }
    return iree_ok_status();
  }
  if (!names.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap name array is required");
  }
  for (iree_host_size_t i = 0; i < names.count; ++i) {
    if (iree_string_view_is_empty(names.values[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "diagnostic tap name %" PRIhsz " is empty", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t
id4_ideogram4_validate_generation_stage_diagnostic_tap_lists(
    iree_host_size_t list_count,
    const id4_ideogram4_generation_stage_diagnostic_tap_list_t* lists) {
  if (list_count == 0) {
    if (lists) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "generation diagnostic tap list array requires at least one entry");
    }
    return iree_ok_status();
  }
  if (!lists) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generation diagnostic tap list array is required");
  }
  for (iree_host_size_t i = 0; i < list_count; ++i) {
    if (iree_string_view_is_empty(lists[i].stage_key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generation diagnostic tap list %" PRIhsz
                              " has an empty stage key",
                              i);
    }
    if (!id4_ideogram4_generation_stage_descriptor_for_key(
            lists[i].stage_key)) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "generation diagnostic tap list %" PRIhsz
                              " references unknown stage `%.*s`",
                              i, (int)lists[i].stage_key.size,
                              lists[i].stage_key.data);
    }
    if (lists[i].tap_names.count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "generation diagnostic tap list for stage `%.*s` is empty",
          (int)lists[i].stage_key.size, lists[i].stage_key.data);
    }
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_validate_diagnostic_tap_names(lists[i].tap_names));
    for (iree_host_size_t j = i + 1; j < list_count; ++j) {
      if (iree_string_view_equal(lists[i].stage_key, lists[j].stage_key)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "generation diagnostic tap list contains duplicate stage `%.*s`",
            (int)lists[i].stage_key.size, lists[i].stage_key.data);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_validate_generation_phase_mask(
    id4_ideogram4_generation_phase_mask_t phase_mask) {
  const id4_ideogram4_generation_phase_mask_t unknown_phase_mask =
      phase_mask & ~ID4_IDEOGRAM4_GENERATION_PHASE_ALL;
  if (unknown_phase_mask != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation phase mask 0x%x "
                            "contains unknown phase bits 0x%x",
                            phase_mask, unknown_phase_mask);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_validate_generation_resident_stage_mask(
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask) {
  const id4_ideogram4_generation_resident_stage_mask_t unknown_stage_mask =
      resident_stage_mask & ~ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_ALL;
  if (unknown_stage_mask != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation resident stage mask 0x%x "
                            "contains unknown stage bits 0x%x",
                            resident_stage_mask, unknown_stage_mask);
  }
  return iree_ok_status();
}

static iree_status_t
id4_ideogram4_validate_generation_prepare_residency_options(
    id4_ideogram4_generation_residency_mode_t mode,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_resident_stage_mask(
      resident_stage_mask));
  switch (mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES:
      if (resident_stage_mask != ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 issue-phase residency mode must not select resident "
            "stage bundles");
      }
      return iree_ok_status();
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES:
      if (resident_stage_mask == ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 selected-stage-bundle residency mode requires at "
            "least one resident stage bundle");
      }
      return iree_ok_status();
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
      if (resident_stage_mask != ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 all-stage-bundle residency mode derives resident "
            "stage bundles and must not receive an explicit stage mask");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation residency mode %" PRIu32
                              " is invalid",
                              (uint32_t)mode);
  }
}

static iree_status_t id4_ideogram4_validate_generation_bundle_residency(
    id4_ideogram4_generation_residency_mode_t mode,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_resident_stage_mask(
      resident_stage_mask));
  switch (mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES:
      if (resident_stage_mask != ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 issue-phase generation bundle must not retain "
            "resident stage bundles");
      }
      return iree_ok_status();
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES:
      if (resident_stage_mask == ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 selected-stage-bundle generation bundle requires at "
            "least one resident stage bundle");
      }
      return iree_ok_status();
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
      if (resident_stage_mask != ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_ALL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 all-stage-bundle generation bundle must retain every "
            "stage bundle");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation residency mode %" PRIu32
                              " is invalid",
                              (uint32_t)mode);
  }
}

static iree_status_t id4_ideogram4_validate_dit_parameter_format(
    const id4_ideogram4_session_create_options_t* options) {
  switch (options->dit_parameter_format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      if (!iree_string_view_is_empty(
              options->parameter_scopes.dit_conditioned_fp8) ||
          !iree_string_view_is_empty(
              options->parameter_scopes.dit_unconditioned_fp8)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 BF16 DiT parameter format must not provide FP8 "
            "parameter scopes");
      }
      return iree_ok_status();
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      if (iree_string_view_is_empty(
              options->parameter_scopes.dit_conditioned_fp8)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 FP8 e4m3 DiT parameter format requires a "
            "conditioned FP8 parameter scope");
      }
      if (iree_string_view_is_empty(
              options->parameter_scopes.dit_unconditioned_fp8)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 FP8 e4m3 DiT parameter format requires an "
            "unconditioned FP8 parameter scope");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 DiT parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)options->dit_parameter_format);
  }
}

static iree_status_t id4_ideogram4_upload_boundary_tensor(
    const id4_ideogram4_boundary_upload_context_t* context,
    iree_string_view_t binding_name, const void* source_data,
    iree_host_size_t source_length,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary upload context is required");
  }
  iree_hal_buffer_binding_t binding;
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      context->plan, context->boundary_bindings, binding_name, &binding));
  return id4_pipeline_queue_update_binding(
      context->device, context->queue_affinity, &binding, source_data,
      source_length, initial_wait_semaphore_list, context->semaphore,
      context->payload_value);
}

static iree_hal_semaphore_list_t id4_ideogram4_single_semaphore_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  *semaphore_storage = semaphore;
  *payload_storage = payload_value;
  iree_hal_semaphore_list_t list = {
      // Number of semaphore edges in this stack-backed list.
      .count = 1,
      // Stack-backed semaphore handle.
      .semaphores = semaphore_storage,
      // Stack-backed payload value.
      .payload_values = payload_storage,
  };
  return list;
}

static iree_hal_semaphore_list_t id4_ideogram4_two_semaphore_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* first_semaphore, uint64_t first_payload_value,
    iree_hal_semaphore_t* second_semaphore, uint64_t second_payload_value) {
  semaphore_storage[0] = first_semaphore;
  semaphore_storage[1] = second_semaphore;
  payload_storage[0] = first_payload_value;
  payload_storage[1] = second_payload_value;
  iree_hal_semaphore_list_t list = {
      // Number of semaphore edges in this stack-backed list.
      .count = 2,
      // Stack-backed semaphore handles.
      .semaphores = semaphore_storage,
      // Stack-backed payload values.
      .payload_values = payload_storage,
  };
  return list;
}

static iree_status_t id4_ideogram4_validate_session_create_options(
    const id4_ideogram4_session_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 session create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 session create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session device group is required");
  }
  if (!options->services.executable_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session executable cache is required");
  }
  if (iree_allocator_is_null(options->services.host_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session host allocator is required");
  }
  if (!options->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session kernel cache is required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_dit_parameter_format(options));
  if (options->vae_activation_format == ID4_VAE_ACTIVATION_FORMAT_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 session VAE activation format is required");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_create_qwen_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_qwen3_vl_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  stage_options.parameter_scope = options->parameter_scopes.qwen;
  stage_options.model = session->qwen_model;
  return id4_qwen3_vl_stage_create(&stage_options, session->host_allocator,
                                   &session->qwen_stage);
}

static iree_status_t id4_ideogram4_create_dit_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session, iree_string_view_t parameter_scope,
    iree_string_view_t fp8_parameter_scope, id4_pipeline_stage_t** out_stage) {
  id4_ideogram4_dit_parameter_source_rule_list_t source_rules;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      options->dit_parameter_format, session->dit_model, fp8_parameter_scope,
      options->services.host_allocator, &source_rules));

  id4_ideogram4_dit_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  stage_options.parameter_scope = parameter_scope;
  stage_options.parameter_source_rule_count = source_rules.count;
  stage_options.parameter_source_rules = source_rules.values;
  stage_options.model = session->dit_model;
  iree_status_t status = id4_ideogram4_dit_stage_create(
      &stage_options, session->host_allocator, out_stage);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &source_rules, options->services.host_allocator);
  return status;
}

static iree_status_t id4_ideogram4_create_sampler_noise_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_sampler_noise_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  return id4_sampler_noise_stage_create(&stage_options, session->host_allocator,
                                        &session->sampler_noise_stage);
}

static iree_status_t id4_ideogram4_create_sampler_denoise_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_sampler_denoise_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  return id4_sampler_denoise_stage_create(
      &stage_options, session->host_allocator, &session->sampler_denoise_stage);
}

static iree_status_t id4_ideogram4_create_decode_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_ideogram4_decode_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  stage_options.parameter_scope = options->parameter_scopes.vae;
  stage_options.model = session->decode_model;
  stage_options.vae_activation_format = options->vae_activation_format;
  return id4_ideogram4_decode_stage_create(
      &stage_options, session->host_allocator, &session->decode_stage);
}

static iree_status_t id4_ideogram4_create_stages(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  session->qwen_model = *id4_qwen3_vl_program_ideogram4_model_config();
  session->dit_model = *id4_ideogram4_dit_program_ideogram4_model_config();
  session->decode_model =
      *id4_ideogram4_decode_program_ideogram4_model_config();

  iree_status_t status = id4_ideogram4_create_qwen_stage(options, session);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_dit_stage(
        options, session, options->parameter_scopes.dit_conditioned,
        options->parameter_scopes.dit_conditioned_fp8,
        &session->dit_conditioned_stage);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_dit_stage(
        options, session, options->parameter_scopes.dit_unconditioned,
        options->parameter_scopes.dit_unconditioned_fp8,
        &session->dit_unconditioned_stage);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_sampler_noise_stage(options, session);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_sampler_denoise_stage(options, session);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_decode_stage(options, session);
  }
  return status;
}

iree_status_t id4_ideogram4_session_create(
    const id4_ideogram4_session_create_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_session_create_options(options));

  id4_ideogram4_session_t* session = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*session), (void**)&session);
  if (iree_status_is_ok(status)) {
    memset(session, 0, sizeof(*session));
    iree_atomic_ref_count_init(&session->ref_count);
    session->host_allocator = host_allocator;
    status = id4_ideogram4_create_stages(options, session);
  }
  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    id4_ideogram4_session_release(session);
  }
  return status;
}

static void id4_ideogram4_session_retain(id4_ideogram4_session_t* session) {
  if (!session) return;
  iree_atomic_ref_count_inc(&session->ref_count);
}

void id4_ideogram4_session_release(id4_ideogram4_session_t* session) {
  if (session && iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    iree_allocator_t host_allocator = session->host_allocator;
    id4_pipeline_stage_release(session->decode_stage);
    id4_pipeline_stage_release(session->sampler_denoise_stage);
    id4_pipeline_stage_release(session->sampler_noise_stage);
    id4_pipeline_stage_release(session->dit_unconditioned_stage);
    id4_pipeline_stage_release(session->dit_conditioned_stage);
    id4_pipeline_stage_release(session->qwen_stage);
    iree_allocator_free(host_allocator, session);
  }
}

static iree_status_t id4_ideogram4_validate_session_load_options(
    const id4_ideogram4_session_load_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session load options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 session load")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 session load extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 session load")));
  return iree_ok_status();
}

iree_status_t id4_ideogram4_session_load(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_session_load_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_session_load_options(options));
  if (session->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Ideogram 4 session is already loaded");
  }
  id4_pipeline_stage_load_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.diagnostics_sink = options->diagnostics_sink;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->qwen_stage, &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->dit_conditioned_stage, &stage_options));
  IREE_RETURN_IF_ERROR(id4_pipeline_stage_load(session->dit_unconditioned_stage,
                                               &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->sampler_noise_stage, &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->sampler_denoise_stage, &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->decode_stage, &stage_options));
  session->is_loaded = true;
  return iree_ok_status();
}

static id4_pipeline_program_shape_t
id4_ideogram4_generation_request_diffusion_latent_shape(
    const id4_ideogram4_request_t* request) {
  return id4_pipeline_program_make_shape_rank4(
      request->generation.latent_width, request->generation.latent_height, 128,
      1);
}

static id4_pipeline_program_shape_t
id4_ideogram4_generation_decoded_image_shape(
    id4_ideogram4_decode_model_config_t model,
    id4_pipeline_program_shape_t diffusion_latent_shape) {
  return id4_pipeline_program_make_shape_rank4(
      diffusion_latent_shape.dims[0] * model.vae.scale_x,
      diffusion_latent_shape.dims[1] * model.vae.scale_y,
      model.vae.decoded_channel_count, diffusion_latent_shape.dims[3]);
}

static iree_status_t id4_ideogram4_validate_generation_request(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_request_t* request) {
  if (!request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation request is required");
  }
  if (!iree_all_bits_set(request->flags,
                         ID4_IDEOGRAM4_REQUEST_FLAG_HAS_GENERATION)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation request metadata is required");
  }
  if (request->generation.denoise_step_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation step count is zero");
  }
  if (request->generation.latent_width == 0 ||
      request->generation.latent_height == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation latent dimensions must be "
                            "nonzero");
  }
  if (!isfinite(request->generation.guidance_scale) ||
      request->generation.guidance_scale <= 0.0f ||
      request->generation.guidance_scale > FLT_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation guidance scale is invalid");
  }
  id4_pipeline_program_shape_t latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(request);
  if (latent_shape.dims[2] != session->dit_model.input_channel_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation latent channel count %" PRIu64
        " does not match DiT channel count %" PRIu32,
        latent_shape.dims[2], session->dit_model.input_channel_count);
  }
  return id4_ideogram4_decode_program_validate_diffusion_latent_shape(
      session->decode_model, latent_shape);
}

static iree_status_t id4_ideogram4_validate_generation_policy(
    id4_ideogram4_generation_plan_policy_t policy) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      policy.structure_size, sizeof(policy),
      IREE_SV("Ideogram 4 generation plan policy")));
  if (policy.next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation plan policy extension structures are not "
        "supported");
  }
  switch (policy.dit_activation_format) {
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL:
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT activation format %" PRIu32 " is invalid",
          (uint32_t)policy.dit_activation_format);
  }
  switch (policy.dit_weight_execution_format) {
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT:
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT:
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_DIRECT_FEED_FORWARD_BF16_RESIDENT:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT weight execution format %" PRIu32
          " is invalid",
          (uint32_t)policy.dit_weight_execution_format);
  }
  switch (policy.qwen_weight_execution_strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation Qwen weight execution strategy %" PRIu32
          " is invalid",
          (uint32_t)policy.qwen_weight_execution_strategy);
  }
  switch (policy.dit_attention_implementation) {
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT attention implementation %" PRIu32
          " is invalid",
          (uint32_t)policy.dit_attention_implementation);
  }
  switch (policy.dit_feed_forward_implementation) {
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT:
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT feed-forward implementation %" PRIu32
          " is invalid",
          (uint32_t)policy.dit_feed_forward_implementation);
  }
  switch (policy.vae_tiling.mode) {
    case ID4_VAE_TILING_MODE_DISABLED:
    case ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE:
    case ID4_VAE_TILING_MODE_RELATIVE_TILE_SIZE:
    case ID4_VAE_TILING_MODE_MEMORY_BUDGET:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation VAE tiling mode %" PRIu32
                              " is invalid",
                              (uint32_t)policy.vae_tiling.mode);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_validate_generation_plan_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Ideogram 4 session must be loaded before "
                            "generation planning");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation plan options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation plan")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation plan extension structures are not supported");
  }
  if (!options->tokenizer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation tokenizer is required");
  }
  const iree_host_size_t device_count = iree_hal_device_group_device_count(
      id4_pipeline_stage_services(session->qwen_stage)->device_group);
  if (options->device_index >= device_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 generation device index %" PRIhsz
                            " exceeds device count %" PRIhsz,
                            options->device_index, device_count);
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_request(session, options->request));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_policy(options->policy));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_stage_diagnostic_tap_lists(
          options->stage_diagnostic_tap_list_count,
          options->stage_diagnostic_tap_lists));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation plan")));
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_plan_allocate(
    id4_ideogram4_session_t* session, iree_allocator_t host_allocator,
    id4_ideogram4_generation_plan_t** out_plan) {
  *out_plan = NULL;
  id4_ideogram4_generation_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*plan), (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->host_allocator = host_allocator;
  plan->session = session;
  id4_ideogram4_session_retain(session);
  *out_plan = plan;
  return iree_ok_status();
}

static iree_string_view_list_t
id4_ideogram4_generation_stage_diagnostic_tap_names(
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  if (!descriptor) return iree_string_view_list_empty();
  iree_string_view_t stage_key = iree_make_cstring_view(descriptor->key);
  for (iree_host_size_t i = 0; i < options->stage_diagnostic_tap_list_count;
       ++i) {
    const id4_ideogram4_generation_stage_diagnostic_tap_list_t* list =
        &options->stage_diagnostic_tap_lists[i];
    if (iree_string_view_equal(list->stage_key, stage_key)) {
      return list->tap_names;
    }
  }
  return iree_string_view_list_empty();
}

static iree_status_t id4_ideogram4_plan_stage(
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_pipeline_stage_t* stage, const void* stage_options,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  iree_string_view_list_t diagnostic_tap_names =
      id4_ideogram4_generation_stage_diagnostic_tap_names(options,
                                                          stage_ordinal);
  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = stage_options;
  plan_options.flags =
      diagnostic_tap_names.count == 0
          ? 0
          : ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.device_index = options->device_index;
  plan_options.queue_affinity = options->queue_affinity;
  plan_options.diagnostic_tap_names = diagnostic_tap_names;
  plan_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_qwen(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    uint32_t token_count, id4_pipeline_plan_t** out_plan) {
  id4_qwen3_vl_stage_plan_options_t qwen_options;
  memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = token_count;
  qwen_options.weight_execution_strategy =
      options->policy.qwen_weight_execution_strategy;
  return id4_ideogram4_plan_stage(ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
                                  session->qwen_stage, &qwen_options, options,
                                  out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_dit(
    id4_pipeline_stage_t* stage,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_dit_conditioning_mode_t conditioning_mode,
    uint32_t text_token_count, id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_dit_stage_plan_options_t dit_options;
  memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request.latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  dit_options.request.conditioning_mode = conditioning_mode;
  dit_options.request.text_token_count = text_token_count;
  dit_options.activation_format = options->policy.dit_activation_format;
  dit_options.weight_execution_format =
      options->policy.dit_weight_execution_format;
  dit_options.attention_implementation =
      options->policy.dit_attention_implementation;
  dit_options.feed_forward_implementation =
      options->policy.dit_feed_forward_implementation;
  return id4_ideogram4_plan_stage(
      conditioning_mode == ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED
          ? ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED
          : ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
      stage, &dit_options, options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_sampler_noise(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_noise_stage_plan_options_t sampler_options;
  memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  return id4_ideogram4_plan_stage(ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
                                  session->sampler_noise_stage,
                                  &sampler_options, options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_sampler_denoise(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_denoise_stage_plan_options_t sampler_options;
  memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  return id4_ideogram4_plan_stage(ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
                                  session->sampler_denoise_stage,
                                  &sampler_options, options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_decode(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_decode_stage_plan_options_t decode_options;
  memset(&decode_options, 0, sizeof(decode_options));
  decode_options.structure_size = sizeof(decode_options);
  decode_options.request.diffusion_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  decode_options.request.vae_tiling = options->policy.vae_tiling;
  return id4_ideogram4_plan_stage(ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
                                  session->decode_stage, &decode_options,
                                  options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_stages(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_generation_plan_t* plan) {
  uint32_t token_count = 0;
  id4_ideogram4_qwen_lowering_options_t qwen_lowering_options;
  memset(&qwen_lowering_options, 0, sizeof(qwen_lowering_options));
  qwen_lowering_options.structure_size = sizeof(qwen_lowering_options);
  qwen_lowering_options.tokenizer = options->tokenizer;
  qwen_lowering_options.request = options->request;
  qwen_lowering_options.tokenizer_flags = options->tokenizer_flags;
  qwen_lowering_options.max_token_count = session->qwen_model.max_token_count;
  qwen_lowering_options.vocab_size = session->qwen_model.vocab_size;
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_count_qwen_tokens(
      &qwen_lowering_options, session->host_allocator, &token_count));

  plan->summary.diffusion_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  plan->summary.qwen_token_count = token_count;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_program_calculate_bf16_token_capacity(
      token_count, &plan->summary.qwen_token_capacity));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_image_token_count(
      session->dit_model, plan->summary.diffusion_latent_shape,
      &plan->summary.image_token_count));
  if (plan->summary.image_token_count > UINT32_MAX - token_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 generation conditioned token count "
                            "overflow");
  }
  plan->summary.conditioned_dit_token_count =
      token_count + plan->summary.image_token_count;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_calculate_bf16_token_capacity(
      plan->summary.conditioned_dit_token_count,
      &plan->summary.conditioned_dit_token_capacity));
  plan->summary.unconditioned_dit_token_count = plan->summary.image_token_count;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_calculate_bf16_token_capacity(
      plan->summary.unconditioned_dit_token_count,
      &plan->summary.unconditioned_dit_token_capacity));
  plan->summary.denoise_step_count =
      options->request->generation.denoise_step_count;
  plan->summary.decoded_image_shape =
      id4_ideogram4_generation_decoded_image_shape(
          session->decode_model, plan->summary.diffusion_latent_shape);
  plan->summary.dit_activation_format = options->policy.dit_activation_format;
  plan->summary.dit_weight_execution_format =
      options->policy.dit_weight_execution_format;
  plan->summary.qwen_weight_execution_strategy =
      options->policy.qwen_weight_execution_strategy;
  plan->summary.dit_attention_implementation =
      options->policy.dit_attention_implementation;
  plan->summary.dit_feed_forward_implementation =
      options->policy.dit_feed_forward_implementation;
  plan->summary.vae_tiling = options->policy.vae_tiling;

  iree_status_t status = id4_ideogram4_plan_generation_qwen(
      session, options, token_count, &plan->qwen_plan);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_dit(
        session->dit_conditioned_stage, options,
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED, token_count,
        &plan->dit_conditioned_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_dit(
        session->dit_unconditioned_stage, options,
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED, 0,
        &plan->dit_unconditioned_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_sampler_noise(
        session, options, &plan->sampler_noise_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_sampler_denoise(
        session, options, &plan->sampler_denoise_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_decode(session, options,
                                                  &plan->decode_plan);
  }
  return status;
}

iree_status_t id4_ideogram4_session_plan_generation(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_generation_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_plan_options(session, options));

  id4_ideogram4_generation_plan_t* plan = NULL;
  iree_status_t status = id4_ideogram4_generation_plan_allocate(
      session, session->host_allocator, &plan);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_stages(session, options, plan);
  }
  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    id4_ideogram4_generation_plan_release(plan);
  }
  return status;
}

void id4_ideogram4_generation_plan_release(
    id4_ideogram4_generation_plan_t* plan) {
  if (!plan) return;
  id4_pipeline_plan_release(plan->decode_plan);
  id4_pipeline_plan_release(plan->sampler_denoise_plan);
  id4_pipeline_plan_release(plan->sampler_noise_plan);
  id4_pipeline_plan_release(plan->dit_unconditioned_plan);
  id4_pipeline_plan_release(plan->dit_conditioned_plan);
  id4_pipeline_plan_release(plan->qwen_plan);
  id4_ideogram4_session_release(plan->session);
  iree_allocator_t host_allocator = plan->host_allocator;
  iree_allocator_free(host_allocator, plan);
}

iree_status_t id4_ideogram4_generation_plan_summary(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_plan_summary_t* out_summary) {
  if (!plan || !out_summary) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan and summary output are required");
  }
  *out_summary = plan->summary;
  return iree_ok_status();
}

iree_host_size_t id4_ideogram4_generation_plan_stage_count(
    const id4_ideogram4_generation_plan_t* plan) {
  return plan ? IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors) : 0;
}

static iree_status_t id4_ideogram4_generation_plan_append_shape_json(
    iree_string_builder_t* builder, id4_pipeline_program_shape_t shape) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"rank\":%u,\"dims\":[", shape.rank));
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIu64, shape.dims[i]));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

static iree_status_t id4_ideogram4_generation_plan_append_tiling_json(
    iree_string_builder_t* builder, id4_vae_tiling_config_t tiling) {
  return iree_string_builder_append_format(
      builder,
      "{\"mode\":%u,\"tile_size_x\":%" PRIu32 ",\"tile_size_y\":%" PRIu32
      ",\"relative_size_x\":%g"
      ",\"relative_size_y\":%g,\"overlap\":%g,\"memory_budget\":%" PRIu64 "}",
      (uint32_t)tiling.mode, tiling.tile_size_x, tiling.tile_size_y,
      (double)tiling.relative_size_x, (double)tiling.relative_size_y,
      (double)tiling.overlap, (uint64_t)tiling.memory_budget);
}

static const id4_pipeline_plan_t* id4_ideogram4_generation_stage_plan(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t ordinal) {
  switch (ordinal) {
    case ID4_IDEOGRAM4_GENERATION_STAGE_NOISE:
      return plan->sampler_noise_plan;
    case ID4_IDEOGRAM4_GENERATION_STAGE_QWEN:
      return plan->qwen_plan;
    case ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED:
      return plan->dit_conditioned_plan;
    case ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED:
      return plan->dit_unconditioned_plan;
    case ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER:
      return plan->sampler_denoise_plan;
    case ID4_IDEOGRAM4_GENERATION_STAGE_DECODE:
      return plan->decode_plan;
    default:
      return NULL;
  }
}

iree_status_t id4_ideogram4_generation_plan_stage_at(
    const id4_ideogram4_generation_plan_t* plan, iree_host_size_t index,
    iree_string_view_t* out_stage_key,
    const id4_pipeline_plan_t** out_stage_plan) {
  if (!plan || !out_stage_key || !out_stage_plan) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan, stage key output, and stage plan output "
        "are required");
  }
  *out_stage_key = iree_string_view_empty();
  *out_stage_plan = NULL;
  const iree_host_size_t stage_count =
      id4_ideogram4_generation_plan_stage_count(plan);
  if (index >= stage_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 generation plan stage index %" PRIhsz
                            " exceeds stage count %" PRIhsz,
                            index, stage_count);
  }

  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      &id4_ideogram4_generation_stage_descriptors[index];
  const id4_pipeline_plan_t* stage_plan =
      id4_ideogram4_generation_stage_plan(plan, descriptor->ordinal);
  if (!stage_plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation plan stage %s is missing",
                            descriptor->key);
  }
  *out_stage_key = iree_make_cstring_view(descriptor->key);
  *out_stage_plan = stage_plan;
  return iree_ok_status();
}

typedef struct id4_ideogram4_generation_stage_resource_statistics_t {
  // Parameter slab bytes retained by one prepared stage bundle.
  iree_device_size_t parameter_byte_length;
  // Constant slab bytes retained by one prepared stage bundle.
  iree_device_size_t constant_byte_length;
  // Local transient high-water bytes used while issuing one stage.
  iree_device_size_t local_high_water_mark;
} id4_ideogram4_generation_stage_resource_statistics_t;

static iree_status_t id4_ideogram4_generation_add_device_size(
    iree_device_size_t* inout_value, iree_device_size_t addend,
    iree_string_view_t field_name) {
  if (UINT64_MAX - *inout_value < addend) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram 4 generation resource statistic %.*s overflows",
        (int)field_name.size, field_name.data);
  }
  *inout_value += addend;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_stage_resource_statistics(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_ideogram4_generation_stage_resource_statistics_t* out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  const id4_pipeline_plan_t* stage_plan =
      id4_ideogram4_generation_stage_plan(plan, stage_ordinal);
  if (!stage_plan) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        id4_ideogram4_generation_stage_descriptor(stage_ordinal);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation resource statistics reference missing "
        "stage %s",
        descriptor ? descriptor->key : "<unknown>");
  }
  id4_pipeline_plan_statistics_t stage_statistics =
      id4_pipeline_plan_statistics(stage_plan);
  out_statistics->parameter_byte_length =
      stage_statistics.parameter_slab_byte_length;
  out_statistics->constant_byte_length =
      stage_statistics.constant_slab_byte_length;
  out_statistics->local_high_water_mark =
      stage_statistics.memory_slab_high_water_mark;
  return iree_ok_status();
}

static bool id4_ideogram4_generation_resident_mask_contains_stage(
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  return descriptor &&
         iree_any_bit_set(resident_stage_mask, descriptor->resident_stage_bit);
}

static id4_ideogram4_generation_resident_stage_mask_t
id4_ideogram4_generation_normalized_resource_resident_stage_mask(
    id4_ideogram4_generation_residency_mode_t residency_mode,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask) {
  switch (residency_mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
      return ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_ALL;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES:
      return resident_stage_mask;
    default:
      return ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  }
}

static iree_status_t id4_ideogram4_generation_resource_add_stage(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_ideogram4_generation_stage_resource_statistics_t* io_statistics) {
  id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
      plan, stage_ordinal, &stage_statistics));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &io_statistics->parameter_byte_length,
      stage_statistics.parameter_byte_length, IREE_SV("parameter")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &io_statistics->constant_byte_length,
      stage_statistics.constant_byte_length, IREE_SV("constant")));
  return id4_ideogram4_generation_add_device_size(
      &io_statistics->local_high_water_mark,
      stage_statistics.local_high_water_mark, IREE_SV("local"));
}

static iree_status_t id4_ideogram4_generation_resource_stage_group(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    iree_host_size_t stage_count,
    const id4_ideogram4_generation_stage_ordinal_t* stage_ordinals,
    id4_ideogram4_generation_stage_resource_statistics_t* out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  for (iree_host_size_t i = 0; i < stage_count; ++i) {
    if (id4_ideogram4_generation_resident_mask_contains_stage(
            resident_stage_mask, stage_ordinals[i])) {
      id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
          plan, stage_ordinals[i], &stage_statistics));
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
          &out_statistics->local_high_water_mark,
          stage_statistics.local_high_water_mark, IREE_SV("local")));
      continue;
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_add_stage(
        plan, stage_ordinals[i], out_statistics));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_resource_resident_stages(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    id4_ideogram4_generation_stage_resource_statistics_t* out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (!iree_any_bit_set(resident_stage_mask,
                          descriptor->resident_stage_bit)) {
      continue;
    }
    id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
        plan, descriptor->ordinal, &stage_statistics));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &out_statistics->parameter_byte_length,
        stage_statistics.parameter_byte_length, IREE_SV("resident.parameter")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &out_statistics->constant_byte_length,
        stage_statistics.constant_byte_length, IREE_SV("resident.constant")));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_plan_find_boundary_layout(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_string_view_t name, const id4_pipeline_tensor_layout_t** out_layout) {
  *out_layout = NULL;
  const id4_pipeline_plan_t* stage_plan =
      id4_ideogram4_generation_stage_plan(plan, stage_ordinal);
  if (!stage_plan) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        id4_ideogram4_generation_stage_descriptor(stage_ordinal);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation resource statistics reference missing "
        "stage %s",
        descriptor ? descriptor->key : "<unknown>");
  }
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(stage_plan);
  for (iree_host_size_t i = 0; i < boundary_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(stage_plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_layout = &boundary->layout;
      return iree_ok_status();
    }
  }
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "Ideogram 4 generation resource statistics stage %s has no boundary "
      "tensor `%.*s`",
      descriptor ? descriptor->key : "<unknown>", (int)name.size, name.data);
}

static iree_status_t
id4_ideogram4_generation_resource_retained_boundary_buffers(
    const id4_ideogram4_generation_plan_t* plan,
    iree_device_size_t* out_boundary_byte_length,
    iree_device_size_t* out_diagnostic_tap_byte_length) {
  *out_boundary_byte_length = 0;
  *out_diagnostic_tap_byte_length = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_pipeline_plan_t* stage_plan = id4_ideogram4_generation_stage_plan(
        plan, id4_ideogram4_generation_stage_descriptors[i].ordinal);
    if (!stage_plan) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation resource statistics reference missing "
          "stage %s",
          id4_ideogram4_generation_stage_descriptors[i].key);
    }
    id4_pipeline_plan_statistics_t stage_statistics =
        id4_pipeline_plan_statistics(stage_plan);
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        out_boundary_byte_length, stage_statistics.boundary_tensor_byte_length,
        IREE_SV("boundary")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        out_diagnostic_tap_byte_length,
        stage_statistics.diagnostic_tap_byte_length, IREE_SV("tap")));
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_boundary_aliases); ++i) {
    const id4_ideogram4_generation_boundary_alias_t* alias =
        &id4_ideogram4_generation_boundary_aliases[i];
    const id4_pipeline_tensor_layout_t* source_layout = NULL;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_find_boundary_layout(
        plan, alias->source_stage, alias->source_name, &source_layout));
    const id4_pipeline_tensor_layout_t* target_layout = NULL;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_find_boundary_layout(
        plan, alias->target_stage, alias->target_name, &target_layout));
    if (source_layout->byte_length != target_layout->byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation boundary alias %.*s to %.*s has byte "
          "length mismatch",
          (int)alias->source_name.size, alias->source_name.data,
          (int)alias->target_name.size, alias->target_name.data);
    }
    if (*out_boundary_byte_length < target_layout->byte_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Ideogram 4 generation boundary alias %.*s to %.*s underflows "
          "retained boundary accounting",
          (int)alias->source_name.size, alias->source_name.data,
          (int)alias->target_name.size, alias->target_name.data);
    }
    *out_boundary_byte_length -= target_layout->byte_length;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_resource_total_peak(
    const id4_ideogram4_generation_resource_statistics_t* statistics,
    iree_device_size_t parameter_byte_length,
    iree_device_size_t constant_byte_length,
    iree_device_size_t local_byte_length,
    iree_device_size_t* out_total_byte_length) {
  iree_device_size_t total_byte_length =
      statistics->boundary_buffer_byte_length;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &total_byte_length, statistics->diagnostic_tap_buffer_byte_length,
      IREE_SV("peak.tap")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &total_byte_length, parameter_byte_length, IREE_SV("peak.parameter")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &total_byte_length, constant_byte_length, IREE_SV("peak.constant")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &total_byte_length, local_byte_length, IREE_SV("peak.local")));
  *out_total_byte_length = total_byte_length;
  return iree_ok_status();
}

iree_status_t id4_ideogram4_generation_plan_resource_statistics(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_resource_statistics_options_t* options,
    id4_ideogram4_generation_resource_statistics_t* out_statistics) {
  if (!plan || !options || !out_statistics) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan, resource statistics options, and output "
        "are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation resource statistics")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation resource statistics extension structures are "
        "not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_prepare_residency_options(
          options->residency_mode, options->resident_stage_mask));

  memset(out_statistics, 0, sizeof(*out_statistics));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_generation_resource_retained_boundary_buffers(
          plan, &out_statistics->boundary_buffer_byte_length,
          &out_statistics->diagnostic_tap_buffer_byte_length));

  const id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask =
      id4_ideogram4_generation_normalized_resource_resident_stage_mask(
          options->residency_mode, options->resident_stage_mask);
  id4_ideogram4_generation_stage_resource_statistics_t resident_statistics;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_resident_stages(
      plan, resident_stage_mask, &resident_statistics));
  out_statistics->resident_stage_parameter_byte_length =
      resident_statistics.parameter_byte_length;
  out_statistics->resident_stage_constant_byte_length =
      resident_statistics.constant_byte_length;
  out_statistics->resident_stage_bundle_byte_length =
      resident_statistics.parameter_byte_length;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &out_statistics->resident_stage_bundle_byte_length,
      resident_statistics.constant_byte_length, IREE_SV("resident.bundle")));

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    const id4_ideogram4_generation_phase_descriptor_t* phase =
        &id4_ideogram4_generation_phase_descriptors[i];
    id4_ideogram4_generation_stage_resource_statistics_t phase_statistics;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_stage_group(
        plan, resident_stage_mask, phase->stage_count, phase->stage_ordinals,
        &phase_statistics));
    iree_device_size_t parameter_byte_length =
        resident_statistics.parameter_byte_length;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &parameter_byte_length, phase_statistics.parameter_byte_length,
        IREE_SV("phase.parameter")));
    iree_device_size_t constant_byte_length =
        resident_statistics.constant_byte_length;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &constant_byte_length, phase_statistics.constant_byte_length,
        IREE_SV("phase.constant")));
    if (parameter_byte_length >
        out_statistics->phase_concurrent_parameter_peak_byte_length) {
      out_statistics->phase_concurrent_parameter_peak_byte_length =
          parameter_byte_length;
    }
    if (constant_byte_length >
        out_statistics->phase_concurrent_constant_peak_byte_length) {
      out_statistics->phase_concurrent_constant_peak_byte_length =
          constant_byte_length;
    }
    if (phase_statistics.local_high_water_mark >
        out_statistics->phase_concurrent_local_peak_byte_length) {
      out_statistics->phase_concurrent_local_peak_byte_length =
          phase_statistics.local_high_water_mark;
    }
    iree_device_size_t total_byte_length = 0;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_total_peak(
        out_statistics, parameter_byte_length, constant_byte_length,
        phase_statistics.local_high_water_mark, &total_byte_length));
    if (total_byte_length >
        out_statistics->phase_concurrent_total_peak_byte_length) {
      out_statistics->phase_concurrent_total_peak_byte_length =
          total_byte_length;
    }
  }

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_stage_group(
        plan, resident_stage_mask, 1, &descriptor->ordinal, &stage_statistics));
    iree_device_size_t parameter_byte_length =
        resident_statistics.parameter_byte_length;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &parameter_byte_length, stage_statistics.parameter_byte_length,
        IREE_SV("stage.parameter")));
    iree_device_size_t constant_byte_length =
        resident_statistics.constant_byte_length;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &constant_byte_length, stage_statistics.constant_byte_length,
        IREE_SV("stage.constant")));
    if (parameter_byte_length >
        out_statistics->stage_serial_parameter_peak_byte_length) {
      out_statistics->stage_serial_parameter_peak_byte_length =
          parameter_byte_length;
    }
    if (constant_byte_length >
        out_statistics->stage_serial_constant_peak_byte_length) {
      out_statistics->stage_serial_constant_peak_byte_length =
          constant_byte_length;
    }
    if (stage_statistics.local_high_water_mark >
        out_statistics->stage_serial_local_peak_byte_length) {
      out_statistics->stage_serial_local_peak_byte_length =
          stage_statistics.local_high_water_mark;
    }
    iree_device_size_t total_byte_length = 0;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_total_peak(
        out_statistics, parameter_byte_length, constant_byte_length,
        stage_statistics.local_high_water_mark, &total_byte_length));
    if (total_byte_length >
        out_statistics->stage_serial_total_peak_byte_length) {
      out_statistics->stage_serial_total_peak_byte_length = total_byte_length;
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_residency_statistics_accumulate(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_phase_descriptor_t* phase,
    id4_ideogram4_generation_residency_statistics_t* io_statistics) {
  for (iree_host_size_t i = 0; i < phase->stage_count; ++i) {
    const id4_pipeline_plan_t* stage_plan =
        id4_ideogram4_generation_stage_plan(plan, phase->stage_ordinals[i]);
    if (!stage_plan) {
      const id4_ideogram4_generation_stage_descriptor_t* descriptor =
          id4_ideogram4_generation_stage_descriptor(phase->stage_ordinals[i]);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation phase %.*s references missing "
          "stage %s",
          (int)phase->name.size, phase->name.data,
          descriptor ? descriptor->key : "<unknown>");
    }
    id4_pipeline_plan_statistics_t stage_statistics =
        id4_pipeline_plan_statistics(stage_plan);
    io_statistics->parameter_byte_length +=
        stage_statistics.parameter_slab_byte_length;
    if (stage_statistics.largest_parameter_slab_byte_length >
        io_statistics->largest_stage_parameter_byte_length) {
      io_statistics->largest_stage_parameter_byte_length =
          stage_statistics.largest_parameter_slab_byte_length;
    }
    io_statistics->constant_byte_length +=
        stage_statistics.constant_slab_byte_length;
    io_statistics->local_slab_byte_length +=
        stage_statistics.memory_slab_byte_length;
    io_statistics->local_high_water_mark +=
        stage_statistics.memory_slab_high_water_mark;
    io_statistics->stage_boundary_byte_length +=
        stage_statistics.boundary_tensor_byte_length;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_plan_append_stage_key_json(
    iree_string_builder_t* builder,
    id4_ideogram4_generation_stage_ordinal_t ordinal) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(ordinal);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown Ideogram 4 generation stage ordinal %u",
                            (uint32_t)ordinal);
  }
  return iree_string_builder_append_format(builder, "\"%s\"", descriptor->key);
}

static iree_status_t id4_ideogram4_generation_plan_append_phase_json(
    const id4_ideogram4_generation_plan_t* plan, iree_string_builder_t* builder,
    const id4_ideogram4_generation_phase_descriptor_t* phase) {
  id4_ideogram4_generation_residency_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_residency_statistics_accumulate(
      plan, phase, &statistics));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"name\":\"%.*s\"", (int)phase->name.size, phase->name.data));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"stage_keys\":["));
  for (iree_host_size_t i = 0; i < phase->stage_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_key_json(
        builder, phase->stage_ordinals[i]));
  }
  const bool repeated_per_denoise_step = iree_any_bit_set(
      phase->flags, ID4_IDEOGRAM4_GENERATION_PHASE_REPEATED_PER_DENOISE_STEP);
  return iree_string_builder_append_format(
      builder,
      "],\"repeated_per_denoise_step\":%s"
      ",\"parameter_byte_length\":%" PRIu64
      ",\"largest_stage_parameter_byte_length\":%" PRIu64
      ",\"constant_byte_length\":%" PRIu64
      ",\"local_slab_byte_length\":%" PRIu64
      ",\"local_high_water_mark\":%" PRIu64
      ",\"stage_boundary_byte_length\":%" PRIu64 "}",
      repeated_per_denoise_step ? "true" : "false",
      (uint64_t)statistics.parameter_byte_length,
      (uint64_t)statistics.largest_stage_parameter_byte_length,
      (uint64_t)statistics.constant_byte_length,
      (uint64_t)statistics.local_slab_byte_length,
      (uint64_t)statistics.local_high_water_mark,
      (uint64_t)statistics.stage_boundary_byte_length);
}

static iree_status_t id4_ideogram4_generation_plan_append_residency_json(
    const id4_ideogram4_generation_plan_t* plan,
    iree_string_builder_t* builder) {
  id4_ideogram4_generation_residency_statistics_t total_statistics;
  memset(&total_statistics, 0, sizeof(total_statistics));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_generation_residency_statistics_accumulate(
            plan, &id4_ideogram4_generation_phase_descriptors[i],
            &total_statistics));
  }

  iree_device_size_t parameter_high_water_mark = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    id4_ideogram4_generation_residency_statistics_t phase_statistics;
    memset(&phase_statistics, 0, sizeof(phase_statistics));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_generation_residency_statistics_accumulate(
            plan, &id4_ideogram4_generation_phase_descriptors[i],
            &phase_statistics));
    if (phase_statistics.parameter_byte_length > parameter_high_water_mark) {
      parameter_high_water_mark = phase_statistics.parameter_byte_length;
    }
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "{\"total_stage_parameter_byte_length\":%" PRIu64
      ",\"phase_parameter_high_water_mark\":%" PRIu64
      ",\"largest_stage_parameter_byte_length\":%" PRIu64
      ",\"total_stage_boundary_byte_length\":%" PRIu64 ",\"phases\":[",
      (uint64_t)total_statistics.parameter_byte_length,
      (uint64_t)parameter_high_water_mark,
      (uint64_t)total_statistics.largest_stage_parameter_byte_length,
      (uint64_t)total_statistics.stage_boundary_byte_length));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_phase_json(
        plan, builder, &id4_ideogram4_generation_phase_descriptors[i]));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

static iree_status_t id4_ideogram4_generation_plan_append_stage_json(
    iree_string_builder_t* builder,
    id4_ideogram4_generation_stage_ordinal_t ordinal,
    const id4_pipeline_plan_t* stage_plan) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(ordinal);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown Ideogram 4 generation stage ordinal %u",
                            (uint32_t)ordinal);
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "\"%s\":", descriptor->key));
  return id4_pipeline_plan_format_json(stage_plan, builder);
}

iree_status_t id4_ideogram4_generation_plan_format_json(
    const id4_ideogram4_generation_plan_t* plan,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "{\"kind\":\"ideogram4_generation\",\"summary\":{\"qwen_token_count\":"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "%" PRIu32 ",\"qwen_token_capacity\":%" PRIu32
      ",\"image_token_count\":%" PRIu32
      ",\"conditioned_dit_token_count\":%" PRIu32
      ",\"conditioned_dit_token_capacity\":%" PRIu32
      ",\"unconditioned_dit_token_count\":%" PRIu32
      ",\"unconditioned_dit_token_capacity\":%" PRIu32
      ",\"denoise_step_count\":%" PRIu32 ",\"diffusion_latent_shape\":",
      plan->summary.qwen_token_count, plan->summary.qwen_token_capacity,
      plan->summary.image_token_count,
      plan->summary.conditioned_dit_token_count,
      plan->summary.conditioned_dit_token_capacity,
      plan->summary.unconditioned_dit_token_count,
      plan->summary.unconditioned_dit_token_capacity,
      plan->summary.denoise_step_count));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_shape_json(
      builder, plan->summary.diffusion_latent_shape));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, ",\"decoded_image_shape\":"));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_shape_json(
      builder, plan->summary.decoded_image_shape));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"dit_activation_format\":%u,\"dit_weight_execution_format\":%u,"
      "\"qwen_weight_execution_strategy\":%u,"
      "\"dit_attention_implementation\":%u,"
      "\"dit_feed_forward_implementation\":%u,\"vae_tiling\":",
      (uint32_t)plan->summary.dit_activation_format,
      (uint32_t)plan->summary.dit_weight_execution_format,
      (uint32_t)plan->summary.qwen_weight_execution_strategy,
      (uint32_t)plan->summary.dit_attention_implementation,
      (uint32_t)plan->summary.dit_feed_forward_implementation));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_tiling_json(
      builder, plan->summary.vae_tiling));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "},\"residency\":"));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_generation_plan_append_residency_json(plan, builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"stages\":{"));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, plan->qwen_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
      plan->dit_conditioned_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
      plan->dit_unconditioned_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE, plan->sampler_noise_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
      plan->sampler_denoise_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE, plan->decode_plan));
  return iree_string_builder_append_cstring(builder, "}}");
}

static iree_status_t id4_ideogram4_validate_generation_prepare_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_prepare_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation preparation");
  }
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation plan is required");
  }
  if (plan->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan was created by a different session");
  }
  if (!plan->qwen_plan || !plan->dit_conditioned_plan ||
      !plan->dit_unconditioned_plan || !plan->sampler_noise_plan ||
      !plan->sampler_denoise_plan || !plan->decode_plan) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan must contain every coarse stage plan");
  }
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation prepare")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram 4 generation prepare extension "
                            "structures are not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_prepare_residency_options(
          options->residency_mode, options->resident_stage_mask));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation prepare "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation prepare signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare final signal is required");
  }
  if (!options->parameter_providers.qwen) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation Qwen parameter provider is required");
  }
  if (!options->parameter_providers.dit_conditioned) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation conditioned DiT parameter provider is required");
  }
  if (!options->parameter_providers.dit_unconditioned) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation unconditioned DiT "
                            "parameter provider is required");
  }
  if (!options->parameter_providers.vae) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation VAE parameter provider is required");
  }
  if (!options->kernel_library) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare kernel library is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation prepare")));
  return iree_ok_status();
}

static void id4_ideogram4_generation_stage_slot_deinitialize(
    id4_ideogram4_generation_stage_slot_t* slot) {
  id4_pipeline_buffer_binding_set_deinitialize(&slot->diagnostic_tap_bindings);
  id4_pipeline_buffer_binding_set_deinitialize(&slot->boundary_bindings);
  id4_pipeline_plan_release(slot->plan);
  memset(slot, 0, sizeof(*slot));
}

static void id4_ideogram4_generation_parameter_providers_retain(
    const id4_ideogram4_generation_parameter_providers_t* providers) {
  iree_io_parameter_provider_retain(providers->qwen);
  iree_io_parameter_provider_retain(providers->dit_conditioned);
  iree_io_parameter_provider_retain(providers->dit_unconditioned);
  iree_io_parameter_provider_retain(providers->vae);
}

static void id4_ideogram4_generation_parameter_providers_release(
    id4_ideogram4_generation_parameter_providers_t* providers) {
  iree_io_parameter_provider_release(providers->vae);
  iree_io_parameter_provider_release(providers->dit_unconditioned);
  iree_io_parameter_provider_release(providers->dit_conditioned);
  iree_io_parameter_provider_release(providers->qwen);
  memset(providers, 0, sizeof(*providers));
}

static void id4_ideogram4_generation_bundle_capture_prepare_resources(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  bundle->parameter_providers = options->parameter_providers;
  id4_ideogram4_generation_parameter_providers_retain(
      &bundle->parameter_providers);
  bundle->kernel_library = options->kernel_library;
  id4_pipeline_kernel_library_retain(bundle->kernel_library);
  bundle->command_buffer_mode = options->command_buffer_mode;
  bundle->residency_mode = options->residency_mode;
  switch (options->residency_mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
      bundle->resident_stage_mask = ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_ALL;
      break;
    default:
      bundle->resident_stage_mask = options->resident_stage_mask;
      break;
  }
}

static void id4_ideogram4_generation_bundle_assign_slot(
    id4_ideogram4_generation_stage_slot_t* slot, id4_pipeline_stage_t* stage,
    id4_pipeline_plan_t* plan) {
  slot->stage = stage;
  slot->plan = plan;
  id4_pipeline_plan_retain(plan);
}

static iree_status_t id4_ideogram4_generation_bundle_create(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    iree_allocator_t host_allocator,
    id4_ideogram4_generation_bundle_t** out_bundle) {
  *out_bundle = NULL;
  id4_ideogram4_generation_bundle_t* bundle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*bundle), (void**)&bundle));
  memset(bundle, 0, sizeof(*bundle));
  iree_atomic_ref_count_init(&bundle->ref_count);
  bundle->host_allocator = host_allocator;
  bundle->session = session;
  bundle->summary = plan->summary;
  id4_ideogram4_session_retain(session);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_NOISE],
      session->sampler_noise_stage, plan->sampler_noise_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_QWEN], session->qwen_stage,
      plan->qwen_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED],
      session->dit_conditioned_stage, plan->dit_conditioned_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED],
      session->dit_unconditioned_stage, plan->dit_unconditioned_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER],
      session->sampler_denoise_stage, plan->sampler_denoise_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE],
      session->decode_stage, plan->decode_plan);
  *out_bundle = bundle;
  return iree_ok_status();
}

static void id4_ideogram4_generation_bundle_retain(
    id4_ideogram4_generation_bundle_t* bundle) {
  if (!bundle) return;
  iree_atomic_ref_count_inc(&bundle->ref_count);
}

static iree_status_t id4_ideogram4_generation_stage_plan_placement(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_device_index,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  const iree_host_size_t placement_count =
      id4_pipeline_plan_placement_count(plan);
  if (placement_count != 1) {
    iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation stage %.*s has %" PRIhsz
        " placements; generation preparation currently requires one "
        "placement per coarse stage",
        (int)stage_name.size, stage_name.data, placement_count);
  }
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(plan, 0);
  if (!placement) {
    iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation stage %.*s placement is missing",
        (int)stage_name.size, stage_name.data);
  }
  *out_device_index = placement->device_index;
  *out_queue_affinity = placement->queue_affinity;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_select_placement(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_host_size_t device_index = 0;
  iree_hal_queue_affinity_t queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_plan_placement(
      bundle->stages[0].plan, &device_index, &queue_affinity));
  iree_hal_device_group_t* device_group =
      id4_pipeline_plan_device_group(bundle->stages[0].plan);
  for (iree_host_size_t i = 1; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    iree_host_size_t stage_device_index = 0;
    iree_hal_queue_affinity_t stage_queue_affinity =
        IREE_HAL_QUEUE_AFFINITY_ANY;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_plan_placement(
        bundle->stages[i].plan, &stage_device_index, &stage_queue_affinity));
    if (id4_pipeline_plan_device_group(bundle->stages[i].plan) !=
        device_group) {
      iree_string_view_t stage_name =
          id4_pipeline_plan_stage_name(bundle->stages[i].plan);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation stage %.*s was planned for a different "
          "device group",
          (int)stage_name.size, stage_name.data);
    }
    if (stage_device_index != device_index ||
        stage_queue_affinity != queue_affinity) {
      iree_string_view_t stage_name =
          id4_pipeline_plan_stage_name(bundle->stages[i].plan);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Ideogram 4 generation stage %.*s placement differs from the first "
          "stage; multi-placement generation preparation is not implemented",
          (int)stage_name.size, stage_name.data);
    }
  }
  bundle->device = iree_hal_device_group_device_at(device_group, device_index);
  if (!bundle->device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation selected device %" PRIhsz
                            " is missing",
                            device_index);
  }
  bundle->queue_affinity = queue_affinity;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_allocate_bindings(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT && iree_status_is_ok(status);
       ++i) {
    id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[i];
    status = id4_pipeline_allocate_boundary_bindings(
        bundle->device, bundle->queue_affinity, slot->plan,
        bundle->host_allocator, &slot->boundary_bindings);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_allocate_diagnostic_tap_bindings(
          bundle->device, bundle->queue_affinity, slot->plan,
          bundle->host_allocator, &slot->diagnostic_tap_bindings);
    }
  }
  return status;
}

static bool id4_ideogram4_generation_shapes_equal(
    id4_pipeline_tensor_shape_t lhs, id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static iree_status_t id4_ideogram4_generation_find_boundary_tensor(
    const id4_pipeline_plan_t* plan, iree_string_view_t name,
    const id4_pipeline_boundary_tensor_plan_t** out_boundary) {
  *out_boundary = NULL;
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  for (iree_host_size_t i = 0; i < boundary_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_boundary = boundary;
      return iree_ok_status();
    }
  }
  iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "Ideogram 4 generation stage %.*s has no boundary tensor `%.*s`",
      (int)stage_name.size, stage_name.data, (int)name.size, name.data);
}

static iree_status_t id4_ideogram4_generation_validate_boundary_alias(
    const id4_ideogram4_generation_stage_slot_t* source_slot,
    iree_string_view_t source_name,
    const id4_ideogram4_generation_stage_slot_t* target_slot,
    iree_string_view_t target_name) {
  const id4_pipeline_boundary_tensor_plan_t* source_boundary = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_boundary_tensor(
      source_slot->plan, source_name, &source_boundary));
  const id4_pipeline_boundary_tensor_plan_t* target_boundary = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_boundary_tensor(
      target_slot->plan, target_name, &target_boundary));

  iree_string_view_t source_stage_name =
      id4_pipeline_plan_stage_name(source_slot->plan);
  iree_string_view_t target_stage_name =
      id4_pipeline_plan_stage_name(target_slot->plan);
  if (!iree_all_bits_set(source_boundary->flags,
                         ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation alias source %.*s/%.*s is not exported",
        (int)source_stage_name.size, source_stage_name.data,
        (int)source_name.size, source_name.data);
  }
  if (!iree_all_bits_set(target_boundary->flags,
                         ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation alias target %.*s/%.*s is not imported",
        (int)target_stage_name.size, target_stage_name.data,
        (int)target_name.size, target_name.data);
  }
  const id4_pipeline_tensor_layout_t* source_layout = &source_boundary->layout;
  const id4_pipeline_tensor_layout_t* target_layout = &target_boundary->layout;
  if (source_layout->dtype != target_layout->dtype ||
      !id4_ideogram4_generation_shapes_equal(source_layout->shape,
                                             target_layout->shape) ||
      source_layout->byte_length != target_layout->byte_length ||
      source_layout->alignment != target_layout->alignment) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation boundary alias %.*s/%.*s to %.*s/%.*s has "
        "incompatible tensor layouts",
        (int)source_stage_name.size, source_stage_name.data,
        (int)source_name.size, source_name.data, (int)target_stage_name.size,
        target_stage_name.data, (int)target_name.size, target_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_apply_boundary_alias(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_boundary_alias_t* alias) {
  id4_ideogram4_generation_stage_slot_t* source_slot =
      &bundle->stages[alias->source_stage];
  id4_ideogram4_generation_stage_slot_t* target_slot =
      &bundle->stages[alias->target_stage];
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_validate_boundary_alias(
      source_slot, alias->source_name, target_slot, alias->target_name));

  iree_hal_buffer_binding_t replacement;
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      source_slot->plan, &source_slot->boundary_bindings, alias->source_name,
      &replacement));
  return id4_pipeline_replace_boundary_binding(target_slot->plan,
                                               &target_slot->boundary_bindings,
                                               alias->target_name, replacement);
}

static iree_status_t id4_ideogram4_generation_bundle_apply_boundary_aliases(
    id4_ideogram4_generation_bundle_t* bundle) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_boundary_aliases); ++i) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_bundle_apply_boundary_alias(
        bundle, &id4_ideogram4_generation_boundary_aliases[i]));
  }
  return iree_ok_status();
}

static iree_io_parameter_provider_t*
id4_ideogram4_generation_prepare_stage_parameter_provider(
    const id4_ideogram4_generation_parameter_providers_t* providers,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  iree_io_parameter_provider_t*
      stage_providers[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT] = {
          NULL,
          providers->qwen,
          providers->dit_conditioned,
          providers->dit_unconditioned,
          NULL,
          providers->vae,
      };
  if (stage_ordinal >= ID4_IDEOGRAM4_GENERATION_STAGE_COUNT) return NULL;
  return stage_providers[stage_ordinal];
}

void id4_ideogram4_generation_bundle_release(
    id4_ideogram4_generation_bundle_t* bundle) {
  if (IREE_LIKELY(bundle) &&
      iree_atomic_ref_count_dec(&bundle->ref_count) == 1) {
    for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT;
         ++i) {
      id4_pipeline_bundle_release(bundle->resident_stage_bundles[i]);
    }
    for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT;
         ++i) {
      id4_ideogram4_generation_stage_slot_deinitialize(&bundle->stages[i]);
    }
    id4_pipeline_kernel_library_release(bundle->kernel_library);
    id4_ideogram4_generation_parameter_providers_release(
        &bundle->parameter_providers);
    id4_ideogram4_session_release(bundle->session);
    iree_allocator_t host_allocator = bundle->host_allocator;
    iree_allocator_free(host_allocator, bundle);
  }
}

static iree_status_t id4_ideogram4_validate_generation_issue_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_issue_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation issue");
  }
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation bundle is required");
  }
  if (bundle->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation bundle was prepared by a different session");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_bundle_residency(
      bundle->residency_mode, bundle->resident_stage_mask));
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation issue")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation issue extension structures are not supported");
  }
  switch (options->issue_policy) {
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_PHASE_CONCURRENT:
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation issue policy %" PRIu32
                              " is invalid",
                              (uint32_t)options->issue_policy);
  }
  if (!options->tokenizer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue tokenizer is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_request(session, options->request));
  if (options->request->generation.denoise_step_count !=
      bundle->summary.denoise_step_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue step count does not match prepared "
        "bundle");
  }
  id4_pipeline_program_shape_t request_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  if (request_latent_shape.rank !=
      bundle->summary.diffusion_latent_shape.rank) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue latent shape does not match prepared "
        "bundle");
  }
  for (uint32_t i = 0; i < request_latent_shape.rank; ++i) {
    if (request_latent_shape.dims[i] !=
        bundle->summary.diffusion_latent_shape.dims[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation issue latent shape does not match prepared "
          "bundle");
    }
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation issue "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue final signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation issue"));
}

static iree_status_t id4_ideogram4_validate_generation_begin_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_begin_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation begin");
  }
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation bundle is required");
  }
  if (bundle->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation bundle was prepared by a different session");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_bundle_residency(
      bundle->residency_mode, bundle->resident_stage_mask));
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation begin options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation begin")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation begin extension structures are not supported");
  }
  if (!options->tokenizer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin tokenizer is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_request(session, options->request));
  if (options->request->generation.denoise_step_count !=
      bundle->summary.denoise_step_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin step count does not match prepared "
        "bundle");
  }
  id4_pipeline_program_shape_t request_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  if (request_latent_shape.rank !=
      bundle->summary.diffusion_latent_shape.rank) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin latent shape does not match prepared "
        "bundle");
  }
  for (uint32_t i = 0; i < request_latent_shape.rank; ++i) {
    if (request_latent_shape.dims[i] !=
        bundle->summary.diffusion_latent_shape.dims[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation begin latent shape does not match prepared "
          "bundle");
    }
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation begin "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation begin signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation begin signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation begin"));
}

static iree_status_t id4_ideogram4_validate_single_generation_phase_mask(
    id4_ideogram4_generation_phase_mask_t phase_mask) {
  switch (phase_mask) {
    case ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING:
    case ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE:
    case ID4_IDEOGRAM4_GENERATION_PHASE_DECODE:
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation phase mask 0x%x does not identify one "
          "generation phase",
          phase_mask);
  }
}

static iree_status_t id4_ideogram4_validate_generation_phase_prepare_options(
    const id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_phase_prepare_options_t* options) {
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation bundle is required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_bundle_residency(
      bundle->residency_mode, bundle->resident_stage_mask));
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation phase prepare")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram 4 generation phase prepare extension "
                            "structures are not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_single_generation_phase_mask(options->phase_mask));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation phase "
                                            "prepare wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation phase prepare signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase prepare signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink,
      IREE_SV("Ideogram 4 generation phase prepare"));
}

static iree_status_t id4_ideogram4_validate_generation_phase_issue_options(
    const id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    const id4_ideogram4_generation_phase_issue_options_t* options) {
  if (!execution) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation execution is required");
  }
  if (!phase_bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation phase bundle is required");
  }
  if (phase_bundle->generation_bundle &&
      phase_bundle->generation_bundle != execution->bundle) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase bundle does not belong to execution "
        "bundle");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_single_generation_phase_mask(
      phase_bundle->phase_mask));
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation phase issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram 4 generation phase issue extension "
                            "structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list,
      IREE_SV("Ideogram 4 generation phase issue wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation phase issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase issue signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation phase issue"));
}

static iree_status_t id4_ideogram4_generation_execution_allocate(
    id4_ideogram4_generation_bundle_t* bundle, iree_allocator_t host_allocator,
    id4_ideogram4_generation_execution_t** out_execution) {
  *out_execution = NULL;
  id4_ideogram4_generation_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  memset(execution, 0, sizeof(*execution));
  execution->host_allocator = host_allocator;
  execution->bundle = bundle;
  id4_ideogram4_generation_bundle_retain(bundle);
  *out_execution = execution;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_execution_create_semaphores(
    id4_ideogram4_generation_execution_t* execution) {
  id4_ideogram4_generation_bundle_t* bundle = execution->bundle;
  iree_status_t status = iree_hal_semaphore_create(
      bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &execution->upload_semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->qwen_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->noise_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &execution->dit_conditioned_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &execution->dit_unconditioned_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->sampler_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->decode_done_semaphore);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_upload_boundary_tensor(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_string_view_t binding_name, const void* source_data,
    iree_host_size_t source_length,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  id4_ideogram4_generation_bundle_t* bundle = execution->bundle;
  id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[stage_ordinal];
  id4_ideogram4_boundary_upload_context_t upload_context = {
      .device = bundle->device,
      .queue_affinity = bundle->queue_affinity,
      .plan = slot->plan,
      .boundary_bindings = &slot->boundary_bindings,
      .semaphore = execution->upload_semaphore,
      .payload_value = &execution->upload_payload_value,
  };
  return id4_ideogram4_upload_boundary_tensor(&upload_context, binding_name,
                                              source_data, source_length,
                                              initial_wait_semaphore_list);
}

static iree_status_t id4_ideogram4_generation_upload_noise_seed(
    id4_ideogram4_generation_execution_t* execution, uint64_t seed,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  int32_t seed_words[2] = {
      // Low 32 bits of the request seed.
      (int32_t)(seed & 0xFFFFFFFFull),
      // High 32 bits of the request seed.
      (int32_t)(seed >> 32),
  };
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE, IREE_SV("seed"),
      seed_words, sizeof(seed_words), initial_wait_semaphore_list);
}

static iree_status_t id4_ideogram4_generation_upload_qwen_inputs(
    id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_qwen_inputs_t* inputs,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  const iree_host_size_t token_ids_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_ids[0]);
  const iree_host_size_t attention_mask_length =
      inputs->token_count * (iree_host_size_t)inputs->token_count *
      sizeof(inputs->attention_mask[0]);
  const iree_host_size_t token_weights_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_weights[0]);
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SV("token_ids"),
      inputs->token_ids, token_ids_length, initial_wait_semaphore_list));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SV("attention_mask"),
      inputs->attention_mask, attention_mask_length,
      iree_hal_semaphore_list_empty()));
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SV("token_weights"),
      inputs->token_weights, token_weights_length,
      iree_hal_semaphore_list_empty());
}

static iree_status_t id4_ideogram4_generation_upload_dit_branch_inputs(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    const id4_ideogram4_dit_branch_inputs_t* inputs) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, stage_ordinal, IREE_SV("image_indicator"),
      inputs->image_indicator, inputs->image_indicator_byte_length,
      iree_hal_semaphore_list_empty()));
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, stage_ordinal, IREE_SV("position_embedding"),
      inputs->position_embedding, inputs->position_embedding_byte_length,
      iree_hal_semaphore_list_empty());
}

static iree_status_t id4_ideogram4_generation_upload_denoise_step(
    id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_denoise_step_t* step) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
      IREE_SV("timestep"), &step->timestep, sizeof(step->timestep),
      iree_hal_semaphore_list_empty()));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
      IREE_SV("timestep"), &step->timestep, sizeof(step->timestep),
      iree_hal_semaphore_list_empty()));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, IREE_SV("scalings"),
      step->scalings, sizeof(step->scalings), iree_hal_semaphore_list_empty()));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, IREE_SV("sigmas"),
      step->sigmas, sizeof(step->sigmas), iree_hal_semaphore_list_empty()));
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, IREE_SV("guidance"),
      step->guidance, sizeof(step->guidance), iree_hal_semaphore_list_empty());
}

static iree_hal_semaphore_list_t id4_ideogram4_generation_make_wait_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_host_size_t count) {
  iree_hal_semaphore_list_t list = {
      // Number of semaphore edges in this stack-backed list.
      .count = count,
      // Stack-backed semaphore handles.
      .semaphores = semaphore_storage,
      // Stack-backed payload values.
      .payload_values = payload_storage,
  };
  return list;
}

static void id4_ideogram4_generation_push_wait(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_host_size_t* inout_count, iree_hal_semaphore_t* semaphore,
    uint64_t payload_value) {
  if (!semaphore) return;
  semaphore_storage[*inout_count] = semaphore;
  payload_storage[*inout_count] = payload_value;
  ++*inout_count;
}

static iree_status_t id4_ideogram4_generation_wait_stage_bundle_readiness(
    id4_pipeline_bundle_t* stage_bundle) {
  if (!stage_bundle) return iree_ok_status();
  return iree_hal_semaphore_list_wait(
      id4_pipeline_bundle_readiness_semaphore_list(stage_bundle),
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_release_stage_bundle_ref(
    id4_ideogram4_generation_stage_bundle_ref_t* stage_bundle_ref) {
  if (!stage_bundle_ref->bundle || !stage_bundle_ref->owns_bundle) {
    stage_bundle_ref->bundle = NULL;
    stage_bundle_ref->owns_bundle = false;
    stage_bundle_ref->was_issued = false;
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (!stage_bundle_ref->was_issued) {
    status = id4_ideogram4_generation_wait_stage_bundle_readiness(
        stage_bundle_ref->bundle);
  }
  id4_pipeline_bundle_release(stage_bundle_ref->bundle);
  stage_bundle_ref->bundle = NULL;
  stage_bundle_ref->owns_bundle = false;
  stage_bundle_ref->was_issued = false;
  return status;
}

static iree_status_t id4_ideogram4_generation_release_stage_bundle_refs(
    id4_ideogram4_generation_stage_bundle_ref_t* stage_bundle_refs) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    status = iree_status_join(status,
                              id4_ideogram4_generation_release_stage_bundle_ref(
                                  &stage_bundle_refs[i]));
  }
  return status;
}

static void id4_ideogram4_generation_phase_bundle_initialize(
    id4_ideogram4_generation_phase_mask_t phase_mask,
    id4_ideogram4_generation_phase_bundle_t* out_phase_bundle) {
  memset(out_phase_bundle->stage_bundle_refs, 0,
         sizeof(out_phase_bundle->stage_bundle_refs));
  out_phase_bundle->phase_mask = phase_mask;
}

static iree_status_t id4_ideogram4_generation_phase_bundle_deinitialize(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle) {
  iree_status_t status = id4_ideogram4_generation_release_stage_bundle_refs(
      phase_bundle->stage_bundle_refs);
  phase_bundle->phase_mask = ID4_IDEOGRAM4_GENERATION_PHASE_NONE;
  return status;
}

static const id4_ideogram4_generation_phase_descriptor_t*
id4_ideogram4_generation_phase_descriptor_for_mask(
    id4_ideogram4_generation_phase_mask_t phase_mask) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    const id4_ideogram4_generation_phase_descriptor_t* phase =
        &id4_ideogram4_generation_phase_descriptors[i];
    if (phase->phase_mask == phase_mask) return phase;
  }
  return NULL;
}

static iree_status_t id4_ideogram4_generation_prepare_stage_bundle(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_bundle_t** out_stage_bundle) {
  IREE_ASSERT_ARGUMENT(out_stage_bundle);
  *out_stage_bundle = NULL;
  id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[stage_ordinal];
  const bool has_parameter_slabs =
      id4_pipeline_plan_parameter_slab_count(slot->plan) != 0;

  iree_hal_semaphore_t* prepare_semaphore = NULL;
  iree_hal_semaphore_t* prepare_semaphore_storage = NULL;
  uint64_t prepare_payload_storage = 1;
  iree_hal_semaphore_list_t signal_list = iree_hal_semaphore_list_empty();
  iree_status_t status = iree_ok_status();
  if (has_parameter_slabs) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &prepare_semaphore);
  }
  if (iree_status_is_ok(status) && has_parameter_slabs) {
    signal_list = id4_ideogram4_single_semaphore_list(
        &prepare_semaphore_storage, &prepare_payload_storage, prepare_semaphore,
        prepare_payload_storage);
  }

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider =
      id4_ideogram4_generation_prepare_stage_parameter_provider(
          &bundle->parameter_providers, stage_ordinal);
  prepare_options.kernel_library = bundle->kernel_library;
  prepare_options.wait_semaphore_list = has_parameter_slabs
                                            ? wait_semaphore_list
                                            : iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.command_buffer_mode = bundle->command_buffer_mode;
  prepare_options.diagnostics_sink = diagnostics_sink;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_stage_prepare(slot->stage, slot->plan,
                                        &prepare_options, out_stage_bundle);
  }

  iree_hal_semaphore_release(prepare_semaphore);
  return status;
}

static iree_status_t id4_ideogram4_generation_bundle_signal_prepared(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  return iree_hal_device_queue_barrier(
      bundle->device, bundle->queue_affinity, options->wait_semaphore_list,
      options->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_bundle_wait_resident_bundles(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    status = iree_status_join(
        status, id4_ideogram4_generation_wait_stage_bundle_readiness(
                    bundle->resident_stage_bundles[i]));
  }
  return status;
}

static iree_host_size_t
id4_ideogram4_generation_bundle_resident_readiness_count(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    if (!bundle->resident_stage_bundles[i]) continue;
    iree_hal_semaphore_list_t readiness_list =
        id4_pipeline_bundle_readiness_semaphore_list(
            bundle->resident_stage_bundles[i]);
    count += readiness_list.count;
  }
  return count;
}

static iree_hal_semaphore_list_t
id4_ideogram4_generation_bundle_resident_readiness_list(
    id4_ideogram4_generation_bundle_t* bundle,
    iree_hal_semaphore_t** semaphores, uint64_t* payload_values) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    if (!bundle->resident_stage_bundles[i]) continue;
    iree_hal_semaphore_list_t readiness_list =
        id4_pipeline_bundle_readiness_semaphore_list(
            bundle->resident_stage_bundles[i]);
    for (iree_host_size_t j = 0; j < readiness_list.count; ++j) {
      semaphores[count] = readiness_list.semaphores[j];
      payload_values[count] = readiness_list.payload_values[j];
      ++count;
    }
  }
  return (iree_hal_semaphore_list_t){
      // Number of readiness edges retained by resident stage bundles.
      .count = count,
      // Stack-backed readiness semaphore handles.
      .semaphores = count == 0 ? NULL : semaphores,
      // Stack-backed readiness payload values.
      .payload_values = count == 0 ? NULL : payload_values,
  };
}

static iree_status_t
id4_ideogram4_generation_bundle_signal_resident_bundles_prepared(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  const iree_host_size_t readiness_count =
      id4_ideogram4_generation_bundle_resident_readiness_count(bundle);
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(options->wait_semaphore_list.count,
                                  readiness_count, &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 generation resident prepare wait "
                            "list count overflow");
  }
  iree_hal_semaphore_t** readiness_semaphores = NULL;
  uint64_t* readiness_payload_values = NULL;
  if (wait_count != 0) {
    readiness_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        wait_count * sizeof(readiness_semaphores[0]));
    readiness_payload_values = (uint64_t*)iree_alloca(
        wait_count * sizeof(readiness_payload_values[0]));
  }
  for (iree_host_size_t i = 0; i < options->wait_semaphore_list.count; ++i) {
    readiness_semaphores[i] = options->wait_semaphore_list.semaphores[i];
    readiness_payload_values[i] =
        options->wait_semaphore_list.payload_values[i];
  }
  iree_hal_semaphore_t** resident_readiness_semaphores =
      wait_count == 0
          ? NULL
          : readiness_semaphores + options->wait_semaphore_list.count;
  uint64_t* resident_readiness_payload_values =
      wait_count == 0
          ? NULL
          : readiness_payload_values + options->wait_semaphore_list.count;
  iree_hal_semaphore_list_t readiness_list =
      id4_ideogram4_generation_bundle_resident_readiness_list(
          bundle, resident_readiness_semaphores,
          resident_readiness_payload_values);
  readiness_list.count = wait_count;
  readiness_list.semaphores = wait_count == 0 ? NULL : readiness_semaphores;
  readiness_list.payload_values =
      wait_count == 0 ? NULL : readiness_payload_values;
  return iree_hal_device_queue_barrier(
      bundle->device, bundle->queue_affinity, readiness_list,
      options->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static bool id4_ideogram4_generation_bundle_stage_is_resident(
    const id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  return descriptor && iree_any_bit_set(bundle->resident_stage_mask,
                                        descriptor->resident_stage_bit);
}

static iree_status_t id4_ideogram4_generation_bundle_prepare_resident_stages(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors) &&
       iree_status_is_ok(status);
       ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (!iree_any_bit_set(bundle->resident_stage_mask,
                          descriptor->resident_stage_bit)) {
      continue;
    }
    const id4_ideogram4_generation_stage_ordinal_t stage_ordinal =
        descriptor->ordinal;
    if (bundle->resident_stage_bundles[stage_ordinal]) continue;
    status = id4_ideogram4_generation_prepare_stage_bundle(
        bundle, stage_ordinal, options->wait_semaphore_list,
        options->diagnostics_sink,
        &bundle->resident_stage_bundles[stage_ordinal]);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_signal_resident_bundles_prepared(
        bundle, options);
  }
  return status;
}

iree_status_t id4_ideogram4_session_prepare_generation(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_prepare_options_t* options,
    id4_ideogram4_generation_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_prepare_options(
      session, plan, options));

  id4_ideogram4_generation_bundle_t* bundle = NULL;
  iree_status_t status = id4_ideogram4_generation_bundle_create(
      session, plan, session->host_allocator, &bundle);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_select_placement(bundle);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_allocate_bindings(bundle);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_apply_boundary_aliases(bundle);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_bundle_capture_prepare_resources(bundle, options);
  }
  if (iree_status_is_ok(status)) {
    switch (options->residency_mode) {
      case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES:
        status =
            id4_ideogram4_generation_bundle_signal_prepared(bundle, options);
        break;
      case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES:
        status =
            id4_ideogram4_generation_bundle_signal_prepared(bundle, options);
        break;
      case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
        status = id4_ideogram4_generation_bundle_prepare_resident_stages(
            bundle, options);
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 generation residency mode %" PRIu32 " is invalid",
            (uint32_t)options->residency_mode);
        break;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    status = iree_status_join(
        status, id4_ideogram4_generation_bundle_wait_resident_bundles(bundle));
    id4_ideogram4_generation_bundle_release(bundle);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_acquire_stage_bundle_ref(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_stage_bundle_ref_t* out_stage_bundle_ref) {
  memset(out_stage_bundle_ref, 0, sizeof(*out_stage_bundle_ref));
  if (id4_ideogram4_generation_bundle_stage_is_resident(bundle,
                                                        stage_ordinal)) {
    if (!bundle->resident_stage_bundles[stage_ordinal]) {
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_prepare_stage_bundle(
          bundle, stage_ordinal, wait_semaphore_list, diagnostics_sink,
          &bundle->resident_stage_bundles[stage_ordinal]));
    }
    out_stage_bundle_ref->bundle =
        bundle->resident_stage_bundles[stage_ordinal];
    out_stage_bundle_ref->owns_bundle = false;
    return iree_ok_status();
  }
  iree_status_t status = id4_ideogram4_generation_prepare_stage_bundle(
      bundle, stage_ordinal, wait_semaphore_list, diagnostics_sink,
      &out_stage_bundle_ref->bundle);
  out_stage_bundle_ref->owns_bundle = iree_status_is_ok(status);
  return status;
}

static iree_status_t id4_ideogram4_generation_prepare_phase_bundle(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_phase_bundle_t* out_phase_bundle) {
  id4_ideogram4_generation_phase_bundle_initialize(phase_mask,
                                                   out_phase_bundle);
  const id4_ideogram4_generation_phase_descriptor_t* phase =
      id4_ideogram4_generation_phase_descriptor_for_mask(phase_mask);
  if (!phase) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation phase mask 0x%x does not "
                            "identify one generation phase",
                            phase_mask);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < phase->stage_count && iree_status_is_ok(status); ++i) {
    const id4_ideogram4_generation_stage_ordinal_t stage_ordinal =
        phase->stage_ordinals[i];
    status = id4_ideogram4_generation_acquire_stage_bundle_ref(
        bundle, stage_ordinal, wait_semaphore_list, diagnostics_sink,
        &out_phase_bundle->stage_bundle_refs[stage_ordinal]);
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status,
        id4_ideogram4_generation_phase_bundle_deinitialize(out_phase_bundle));
  }
  return status;
}

static id4_pipeline_bundle_t* id4_ideogram4_generation_phase_stage_bundle(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  return phase_bundle->stage_bundle_refs[stage_ordinal].bundle;
}

static iree_host_size_t id4_ideogram4_generation_phase_bundle_readiness_count(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    id4_pipeline_bundle_t* stage_bundle =
        phase_bundle->stage_bundle_refs[i].bundle;
    if (!stage_bundle) continue;
    count += id4_pipeline_bundle_readiness_semaphore_list(stage_bundle).count;
  }
  return count;
}

static iree_hal_semaphore_list_t
id4_ideogram4_generation_phase_bundle_readiness_list(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_t** semaphores, uint64_t* payload_values) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    id4_pipeline_bundle_t* stage_bundle =
        phase_bundle->stage_bundle_refs[i].bundle;
    if (!stage_bundle) continue;
    iree_hal_semaphore_list_t readiness_list =
        id4_pipeline_bundle_readiness_semaphore_list(stage_bundle);
    for (iree_host_size_t j = 0; j < readiness_list.count; ++j) {
      semaphores[count] = readiness_list.semaphores[j];
      payload_values[count] = readiness_list.payload_values[j];
      ++count;
    }
  }
  return (iree_hal_semaphore_list_t){
      // Number of readiness edges retained by the phase bundle.
      .count = count,
      // Stack-backed readiness semaphore handles.
      .semaphores = count == 0 ? NULL : semaphores,
      // Stack-backed readiness payload values.
      .payload_values = count == 0 ? NULL : payload_values,
  };
}

static iree_status_t id4_ideogram4_generation_phase_bundle_signal_prepared(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  const iree_host_size_t readiness_count =
      id4_ideogram4_generation_phase_bundle_readiness_count(phase_bundle);
  iree_hal_semaphore_t** readiness_semaphores =
      readiness_count == 0
          ? NULL
          : (iree_hal_semaphore_t**)iree_alloca(
                readiness_count * sizeof(readiness_semaphores[0]));
  uint64_t* readiness_payload_values =
      readiness_count == 0
          ? NULL
          : (uint64_t*)iree_alloca(readiness_count *
                                   sizeof(readiness_payload_values[0]));
  iree_hal_semaphore_list_t readiness_list =
      id4_ideogram4_generation_phase_bundle_readiness_list(
          phase_bundle, readiness_semaphores, readiness_payload_values);
  return iree_hal_device_queue_barrier(bundle->device, bundle->queue_affinity,
                                       readiness_list, signal_semaphore_list,
                                       IREE_HAL_EXECUTE_FLAG_NONE);
}

iree_status_t id4_ideogram4_generation_phase_bundle_release(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle) {
  if (!phase_bundle) return iree_ok_status();
  iree_status_t status =
      id4_ideogram4_generation_phase_bundle_deinitialize(phase_bundle);
  id4_ideogram4_generation_bundle_release(phase_bundle->generation_bundle);
  iree_allocator_t host_allocator = phase_bundle->host_allocator;
  if (!iree_allocator_is_null(host_allocator)) {
    iree_allocator_free(host_allocator, phase_bundle);
  }
  return status;
}

iree_status_t id4_ideogram4_generation_bundle_prepare_phase(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_phase_prepare_options_t* options,
    id4_ideogram4_generation_phase_bundle_t** out_phase_bundle) {
  IREE_ASSERT_ARGUMENT(out_phase_bundle);
  *out_phase_bundle = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_phase_prepare_options(bundle, options));

  id4_ideogram4_generation_phase_bundle_t* phase_bundle = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      bundle->host_allocator, sizeof(*phase_bundle), (void**)&phase_bundle));
  memset(phase_bundle, 0, sizeof(*phase_bundle));
  phase_bundle->host_allocator = bundle->host_allocator;
  phase_bundle->generation_bundle = bundle;
  id4_ideogram4_generation_bundle_retain(bundle);

  iree_status_t status = id4_ideogram4_generation_prepare_phase_bundle(
      bundle, options->phase_mask, options->wait_semaphore_list,
      options->diagnostics_sink, phase_bundle);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_phase_bundle_signal_prepared(
        bundle, phase_bundle, options->signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    *out_phase_bundle = phase_bundle;
  } else {
    status = iree_status_join(
        status, id4_ideogram4_generation_phase_bundle_release(phase_bundle));
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_chain_upload_after_sampler(
    id4_ideogram4_generation_execution_t* execution,
    uint64_t sampler_payload_value) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->sampler_done_semaphore, sampler_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = execution->upload_payload_value + 1;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore, &signal_payload_value, execution->upload_semaphore,
      signal_payload_value);
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE));
  execution->upload_payload_value = signal_payload_value;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_chain_phase_issue_wait(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_hal_semaphore_list_t phase_wait_list) {
  if (phase_wait_list.count == 0) return iree_ok_status();
  iree_host_size_t wait_count_capacity = 0;
  if (!iree_host_size_checked_add(phase_wait_list.count, 1,
                                  &wait_count_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram 4 generation phase issue wait list count overflow");
  }
  iree_hal_semaphore_t** wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
      wait_count_capacity * sizeof(wait_semaphores[0]));
  uint64_t* wait_payload_values = (uint64_t*)iree_alloca(
      wait_count_capacity * sizeof(wait_payload_values[0]));
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  for (iree_host_size_t i = 0; i < phase_wait_list.count; ++i) {
    id4_ideogram4_generation_push_wait(
        wait_semaphores, wait_payload_values, &wait_count,
        phase_wait_list.semaphores[i], phase_wait_list.payload_values[i]);
  }
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = execution->upload_payload_value + 1;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore, &signal_payload_value, execution->upload_semaphore,
      signal_payload_value);
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE));
  execution->upload_payload_value = signal_payload_value;
  if (phase_mask == ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING) {
    execution->qwen_upload_payload_value = signal_payload_value;
    execution->seed_upload_payload_value = signal_payload_value;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_issue_stage(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_pipeline_bundle_t* stage_bundle,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_slot_t* slot =
      &execution->bundle->stages[stage_ordinal];
  iree_hal_semaphore_t* signal_semaphore_storage = NULL;
  uint64_t signal_payload_storage = 0;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore_storage, &signal_payload_storage, signal_semaphore,
      signal_payload_value);

  id4_pipeline_stage_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = slot->boundary_bindings.count;
  issue_options.boundary_bindings = slot->boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count =
      slot->diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings =
      slot->diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = wait_semaphore_list;
  issue_options.signal_semaphore_list = signal_list;
  issue_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_issue(slot->stage, stage_bundle, &issue_options);
}

static iree_status_t id4_ideogram4_generation_issue_qwen(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* qwen_bundle, uint64_t qwen_upload_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     qwen_upload_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, qwen_bundle, wait_list,
      execution->qwen_done_semaphore, 1, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_noise(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* noise_bundle, uint64_t seed_upload_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     seed_upload_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE, noise_bundle, wait_list,
      execution->noise_done_semaphore, 1, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_dit_branch(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_pipeline_bundle_t* branch_bundle,
    iree_hal_semaphore_t* latent_semaphore, uint64_t latent_payload_value,
    iree_hal_semaphore_t* condition_semaphore, uint64_t condition_payload_value,
    iree_hal_semaphore_t* branch_done_semaphore, uint64_t branch_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[4];
  uint64_t wait_payload_values[4];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, latent_semaphore,
                                     latent_payload_value);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, condition_semaphore,
                                     condition_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, stage_ordinal, branch_bundle, wait_list, branch_done_semaphore,
      branch_payload_value, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_sampler(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* sampler_bundle, uint64_t payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[3];
  uint64_t wait_payload_values[3];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->dit_conditioned_done_semaphore, payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->dit_unconditioned_done_semaphore, payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, sampler_bundle,
      wait_list, execution->sampler_done_semaphore, payload_value,
      diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_decode(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* decode_bundle, uint64_t sampler_payload,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[3];
  uint64_t wait_payload_values[3];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->sampler_done_semaphore, sampler_payload);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE, decode_bundle,
      wait_list, execution->decode_done_semaphore, 1, diagnostics_sink));
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = 1;
  wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, execution->decode_done_semaphore,
      wait_payload_value);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      final_signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_find_outputs(
    id4_ideogram4_generation_execution_t* execution) {
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
          .plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
           .boundary_bindings,
      IREE_SV("velocity"), &execution->conditioned_velocity_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle
          ->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
          .plan,
      &execution->bundle
           ->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
           .boundary_bindings,
      IREE_SV("velocity"), &execution->unconditioned_velocity_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER].plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
           .boundary_bindings,
      IREE_SV("denoised"), &execution->denoised_latent_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER].plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
           .boundary_bindings,
      IREE_SV("x_next"), &execution->final_latent_binding));
  return id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE].plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE]
           .boundary_bindings,
      IREE_SV("media.image.decoded"), &execution->decoded_image_binding);
}

static iree_status_t id4_ideogram4_generation_begin_execution(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_request_t* request, const iree_tokenizer_t* tokenizer,
    iree_tokenizer_encode_flags_t tokenizer_flags,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    id4_ideogram4_generation_execution_t** out_execution) {
  id4_ideogram4_generation_execution_t* execution = NULL;
  iree_status_t status = id4_ideogram4_generation_execution_allocate(
      bundle, session->host_allocator, &execution);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_qwen_lowering_options_t qwen_lowering_options;
    memset(&qwen_lowering_options, 0, sizeof(qwen_lowering_options));
    qwen_lowering_options.structure_size = sizeof(qwen_lowering_options);
    qwen_lowering_options.tokenizer = tokenizer;
    qwen_lowering_options.request = request;
    qwen_lowering_options.tokenizer_flags = tokenizer_flags;
    qwen_lowering_options.max_token_count = session->qwen_model.max_token_count;
    qwen_lowering_options.vocab_size = session->qwen_model.vocab_size;
    status = id4_ideogram4_request_lower_qwen_inputs(&qwen_lowering_options,
                                                     execution->host_allocator,
                                                     &execution->qwen_inputs);
  }
  if (iree_status_is_ok(status) &&
      execution->qwen_inputs.token_count != bundle->summary.qwen_token_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue Qwen token count %" PRIu32
        " does not match prepared bundle count %" PRIu32,
        execution->qwen_inputs.token_count, bundle->summary.qwen_token_count);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_dit_lowering_options_t dit_lowering_options;
    memset(&dit_lowering_options, 0, sizeof(dit_lowering_options));
    dit_lowering_options.structure_size = sizeof(dit_lowering_options);
    dit_lowering_options.generation = &request->generation;
    dit_lowering_options.text_token_count = execution->qwen_inputs.token_count;
    dit_lowering_options.attention_head_size =
        session->dit_model.hidden_size /
        session->dit_model.attention_head_count;
    status = id4_ideogram4_request_lower_dit_inputs(&dit_lowering_options,
                                                    execution->host_allocator,
                                                    &execution->dit_inputs);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_generation_lower_denoise_schedule(
        &request->generation, execution->host_allocator,
        &execution->denoise_schedule);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_execution_create_semaphores(execution);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(wait_semaphore_list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_qwen_inputs(
        execution, &execution->qwen_inputs, iree_hal_semaphore_list_empty());
  }
  if (iree_status_is_ok(status)) {
    execution->qwen_upload_payload_value = execution->upload_payload_value;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_noise_seed(
        execution, request->generation.seed, iree_hal_semaphore_list_empty());
  }
  if (iree_status_is_ok(status)) {
    execution->seed_upload_payload_value = execution->upload_payload_value;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_dit_branch_inputs(
        execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
        &execution->dit_inputs.conditioned);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_dit_branch_inputs(
        execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
        &execution->dit_inputs.unconditioned);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_denoise_step(
        execution, &execution->denoise_schedule.steps[0]);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_find_outputs(execution);
  }
  if (iree_status_is_ok(status) && signal_semaphore_list.count != 0) {
    iree_hal_semaphore_t* wait_semaphore = NULL;
    uint64_t wait_payload_value = execution->upload_payload_value;
    iree_hal_semaphore_list_t upload_wait_list =
        id4_ideogram4_single_semaphore_list(
            &wait_semaphore, &wait_payload_value, execution->upload_semaphore,
            wait_payload_value);
    status = iree_hal_device_queue_barrier(
        bundle->device, bundle->queue_affinity, upload_wait_list,
        signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_generation_execution_release(execution);
  }
  return status;
}

iree_status_t id4_ideogram4_session_begin_generation(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_begin_options_t* options,
    id4_ideogram4_generation_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_begin_options(
      session, bundle, options));
  return id4_ideogram4_generation_begin_execution(
      session, bundle, options->request, options->tokenizer,
      options->tokenizer_flags, options->wait_semaphore_list,
      options->signal_semaphore_list, out_execution);
}

static iree_status_t id4_ideogram4_generation_issue_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = id4_ideogram4_generation_issue_qwen(
      execution,
      id4_ideogram4_generation_phase_stage_bundle(
          phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN),
      execution->qwen_upload_payload_value, diagnostics_sink);
  phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_QWEN]
      .was_issued = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_noise(
        execution,
        id4_ideogram4_generation_phase_stage_bundle(
            phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE),
        execution->seed_upload_payload_value, diagnostics_sink);
    phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_NOISE]
        .was_issued = iree_status_is_ok(status);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_wait_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution) {
  iree_hal_semaphore_t* qwen_done_semaphore = execution->qwen_done_semaphore;
  uint64_t qwen_done_payload_value = 1;
  iree_hal_semaphore_list_t qwen_done_list = {
      // One Qwen completion edge required before allocating DiT weights.
      .count = 1,
      // Stack-backed Qwen completion semaphore.
      .semaphores = &qwen_done_semaphore,
      // Stack-backed Qwen completion payload value.
      .payload_values = &qwen_done_payload_value,
  };
  return iree_hal_semaphore_list_wait(qwen_done_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_wait_one(
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = payload_value;
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, semaphore, payload_value);
  return iree_hal_semaphore_list_wait(wait_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_issue_qwen_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t qwen_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
      iree_hal_semaphore_list_empty(), diagnostics_sink, &qwen_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_qwen(
        execution, qwen_ref.bundle, execution->qwen_upload_payload_value,
        diagnostics_sink);
    qwen_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_ideogram4_generation_wait_one(execution->qwen_done_semaphore, 1);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&qwen_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_noise_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t noise_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
      iree_hal_semaphore_list_empty(), diagnostics_sink, &noise_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_noise(
        execution, noise_ref.bundle, execution->seed_upload_payload_value,
        diagnostics_sink);
    noise_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_ideogram4_generation_wait_one(execution->noise_done_semaphore, 1);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&noise_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_dit_branch_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_hal_semaphore_t* latent_semaphore, uint64_t latent_payload_value,
    iree_hal_semaphore_t* condition_semaphore, uint64_t condition_payload_value,
    iree_hal_semaphore_t* branch_done_semaphore, uint64_t branch_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t branch_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, stage_ordinal, iree_hal_semaphore_list_empty(),
      diagnostics_sink, &branch_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_dit_branch(
        execution, stage_ordinal, branch_ref.bundle, latent_semaphore,
        latent_payload_value, condition_semaphore, condition_payload_value,
        branch_done_semaphore, branch_payload_value, diagnostics_sink);
    branch_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_one(branch_done_semaphore,
                                               branch_payload_value);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&branch_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_sampler_stage_serial(
    id4_ideogram4_generation_execution_t* execution, uint64_t payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t sampler_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
      iree_hal_semaphore_list_empty(), diagnostics_sink, &sampler_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_sampler(
        execution, sampler_ref.bundle, payload_value, diagnostics_sink);
    sampler_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_one(
        execution->sampler_done_semaphore, payload_value);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&sampler_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_decode_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t decode_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
      iree_hal_semaphore_list_empty(), diagnostics_sink, &decode_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_decode(
        execution, decode_ref.bundle, execution->denoise_schedule.step_count,
        final_signal_list, diagnostics_sink);
    decode_ref.was_issued = iree_status_is_ok(status);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&decode_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_conditioning_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_qwen_stage_serial(
      execution, diagnostics_sink));
  return id4_ideogram4_generation_issue_noise_stage_serial(execution,
                                                           diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_denoise_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < execution->denoise_schedule.step_count && iree_status_is_ok(status);
       ++i) {
    const uint64_t payload_value = (uint64_t)i + 1;
    iree_hal_semaphore_t* latent_semaphore =
        i == 0 ? execution->noise_done_semaphore
               : execution->sampler_done_semaphore;
    const uint64_t latent_payload_value = i == 0 ? 1 : (uint64_t)i;
    if (i > 0) {
      status = id4_ideogram4_generation_chain_upload_after_sampler(execution,
                                                                   (uint64_t)i);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_upload_denoise_step(
            execution, &execution->denoise_schedule.steps[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch_stage_serial(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
          latent_semaphore, latent_payload_value,
          execution->qwen_done_semaphore, 1,
          execution->dit_conditioned_done_semaphore, payload_value,
          diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch_stage_serial(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
          latent_semaphore, latent_payload_value, NULL, 0,
          execution->dit_unconditioned_done_semaphore, payload_value,
          diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_sampler_stage_serial(
          execution, payload_value, diagnostics_sink);
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_conditioning_stage_serial(
      execution, diagnostics_sink));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_denoise_stage_serial(
      execution, diagnostics_sink));
  return id4_ideogram4_generation_issue_decode_stage_serial(
      execution, final_signal_list, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_denoise_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < execution->denoise_schedule.step_count && iree_status_is_ok(status);
       ++i) {
    const uint64_t payload_value = (uint64_t)i + 1;
    iree_hal_semaphore_t* latent_semaphore =
        i == 0 ? execution->noise_done_semaphore
               : execution->sampler_done_semaphore;
    const uint64_t latent_payload_value = i == 0 ? 1 : (uint64_t)i;
    if (i > 0) {
      status = id4_ideogram4_generation_chain_upload_after_sampler(execution,
                                                                   (uint64_t)i);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_upload_denoise_step(
            execution, &execution->denoise_schedule.steps[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED),
          latent_semaphore, latent_payload_value,
          execution->qwen_done_semaphore, 1,
          execution->dit_conditioned_done_semaphore, payload_value,
          diagnostics_sink);
      phase_bundle
          ->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
          .was_issued |= iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED),
          latent_semaphore, latent_payload_value, NULL, 0,
          execution->dit_unconditioned_done_semaphore, payload_value,
          diagnostics_sink);
      phase_bundle
          ->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
          .was_issued |= iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_sampler(
          execution,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER),
          payload_value, diagnostics_sink);
      phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
          .was_issued |= iree_status_is_ok(status);
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_decode_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = id4_ideogram4_generation_issue_decode(
      execution,
      id4_ideogram4_generation_phase_stage_bundle(
          phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE),
      execution->denoise_schedule.step_count, final_signal_list,
      diagnostics_sink);
  phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE]
      .was_issued = iree_status_is_ok(status);
  return status;
}

static iree_status_t id4_ideogram4_generation_signal_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count,
                                     execution->qwen_done_semaphore, 1);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count,
                                     execution->noise_done_semaphore, 1);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_signal_denoise_phase(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = execution->denoise_schedule.step_count;
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, execution->sampler_done_semaphore,
      wait_payload_value);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

iree_status_t id4_ideogram4_generation_execution_issue_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    const id4_ideogram4_generation_phase_issue_options_t* options) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_phase_issue_options(
      execution, phase_bundle, options));
  iree_status_t status = id4_ideogram4_generation_chain_phase_issue_wait(
      execution, phase_bundle->phase_mask, options->wait_semaphore_list);
  if (!iree_status_is_ok(status)) return status;
  switch (phase_bundle->phase_mask) {
    case ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING:
      status = id4_ideogram4_generation_issue_conditioning_phase(
          execution, phase_bundle, options->diagnostics_sink);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_signal_conditioning_phase(
            execution, options->signal_semaphore_list);
      }
      return status;
    case ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE:
      status = id4_ideogram4_generation_issue_denoise_phase(
          execution, phase_bundle, options->diagnostics_sink);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_signal_denoise_phase(
            execution, options->signal_semaphore_list);
      }
      return status;
    case ID4_IDEOGRAM4_GENERATION_PHASE_DECODE:
      return id4_ideogram4_generation_issue_decode_phase(
          execution, phase_bundle, options->signal_semaphore_list,
          options->diagnostics_sink);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation phase mask 0x%x does not identify one "
          "generation phase",
          phase_bundle->phase_mask);
  }
}

iree_status_t id4_ideogram4_session_issue_generation(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_issue_options_t* options,
    id4_ideogram4_generation_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_issue_options(
      session, bundle, options));

  id4_ideogram4_generation_execution_t* execution = NULL;
  if (options->issue_policy ==
      ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL) {
    iree_status_t status = id4_ideogram4_generation_begin_execution(
        session, bundle, options->request, options->tokenizer,
        options->tokenizer_flags, options->wait_semaphore_list,
        iree_hal_semaphore_list_empty(), &execution);
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_stage_serial(
          execution, options->signal_semaphore_list, options->diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      *out_execution = execution;
    } else {
      id4_ideogram4_generation_execution_release(execution);
    }
    return status;
  }

  id4_ideogram4_generation_phase_bundle_t conditioning_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING, &conditioning_phase);
  id4_ideogram4_generation_phase_bundle_t denoise_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE, &denoise_phase);
  id4_ideogram4_generation_phase_bundle_t decode_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_DECODE, &decode_phase);

  iree_status_t status = id4_ideogram4_generation_begin_execution(
      session, bundle, options->request, options->tokenizer,
      options->tokenizer_flags, options->wait_semaphore_list,
      iree_hal_semaphore_list_empty(), &execution);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING,
        iree_hal_semaphore_list_empty(), options->diagnostics_sink,
        &conditioning_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_conditioning_phase(
        execution, &conditioning_phase, options->diagnostics_sink);
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&conditioning_phase));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_conditioning_phase(execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE,
        iree_hal_semaphore_list_empty(), options->diagnostics_sink,
        &denoise_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_denoise_phase(
        execution, &denoise_phase, options->diagnostics_sink);
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&denoise_phase));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_DECODE,
        iree_hal_semaphore_list_empty(), options->diagnostics_sink,
        &decode_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_decode_phase(
        execution, &decode_phase, options->signal_semaphore_list,
        options->diagnostics_sink);
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&decode_phase));
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&conditioning_phase));
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&denoise_phase));
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_generation_execution_release(execution);
  }
  return status;
}

void id4_ideogram4_generation_execution_release(
    id4_ideogram4_generation_execution_t* execution) {
  if (!execution) return;
  id4_ideogram4_denoise_schedule_deinitialize(&execution->denoise_schedule,
                                              execution->host_allocator);
  id4_ideogram4_dit_inputs_deinitialize(&execution->dit_inputs,
                                        execution->host_allocator);
  id4_ideogram4_qwen_inputs_deinitialize(&execution->qwen_inputs,
                                         execution->host_allocator);
  iree_hal_semaphore_release(execution->decode_done_semaphore);
  iree_hal_semaphore_release(execution->sampler_done_semaphore);
  iree_hal_semaphore_release(execution->dit_unconditioned_done_semaphore);
  iree_hal_semaphore_release(execution->dit_conditioned_done_semaphore);
  iree_hal_semaphore_release(execution->noise_done_semaphore);
  iree_hal_semaphore_release(execution->qwen_done_semaphore);
  iree_hal_semaphore_release(execution->upload_semaphore);
  id4_ideogram4_generation_bundle_release(execution->bundle);
  iree_allocator_t host_allocator = execution->host_allocator;
  iree_allocator_free(host_allocator, execution);
}

iree_status_t id4_ideogram4_generation_execution_result(
    const id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_result_t* out_result) {
  if (!execution || !out_result) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation execution and result output are required");
  }
  out_result->conditioned_velocity_binding =
      execution->conditioned_velocity_binding;
  out_result->unconditioned_velocity_binding =
      execution->unconditioned_velocity_binding;
  out_result->denoised_latent_binding = execution->denoised_latent_binding;
  out_result->final_latent_binding = execution->final_latent_binding;
  out_result->decoded_image_binding = execution->decoded_image_binding;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_find_diagnostic_tap_plan(
    const id4_pipeline_plan_t* plan, iree_string_view_t tap_name,
    const id4_pipeline_diagnostic_tap_plan_t** out_tap) {
  *out_tap = NULL;
  const iree_host_size_t tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  for (iree_host_size_t i = 0; i < tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (tap && iree_string_view_equal(tap->name, tap_name)) {
      *out_tap = tap;
      return iree_ok_status();
    }
  }
  iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "Ideogram 4 generation stage %.*s has no diagnostic tap `%.*s`",
      (int)stage_name.size, stage_name.data, (int)tap_name.size, tap_name.data);
}

iree_status_t id4_ideogram4_generation_execution_find_diagnostic_tap(
    const id4_ideogram4_generation_execution_t* execution,
    iree_string_view_t stage_key, iree_string_view_t tap_name,
    const id4_pipeline_tensor_layout_t** out_layout,
    iree_hal_buffer_binding_t* out_binding) {
  if (!execution || iree_string_view_is_empty(stage_key) ||
      iree_string_view_is_empty(tap_name) || !out_layout || !out_binding) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation diagnostic tap lookup requires execution, "
        "stage key, tap name, layout output, and binding output");
  }
  *out_layout = NULL;
  memset(out_binding, 0, sizeof(*out_binding));
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor_for_key(stage_key);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Ideogram 4 generation has no stage `%.*s`",
                            (int)stage_key.size, stage_key.data);
  }
  const id4_ideogram4_generation_stage_slot_t* slot =
      &execution->bundle->stages[descriptor->ordinal];
  const id4_pipeline_diagnostic_tap_plan_t* tap = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_diagnostic_tap_plan(
      slot->plan, tap_name, &tap));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_diagnostic_tap_binding(
      slot->plan, &slot->diagnostic_tap_bindings, tap_name, out_binding));
  *out_layout = &tap->layout;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_validate_qwen_issue_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_qwen_issue_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Ideogram 4 session must be loaded before issue");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("Qwen issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Qwen issue extension structures are not "
                            "supported");
  }
  if (!options->request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue request is required");
  }
  if (!options->tokenizer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue tokenizer is required");
  }
  if (!options->parameter_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue parameter provider is required");
  }
  if (!options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue kernel library is required");
  }
  switch (options->qwen_weight_execution_strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Qwen issue weight execution strategy %" PRIu32 " is invalid",
          (uint32_t)options->qwen_weight_execution_strategy);
  }
  const iree_host_size_t device_count = iree_hal_device_group_device_count(
      id4_pipeline_stage_services(session->qwen_stage)->device_group);
  if (options->device_index >= device_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen issue device index %" PRIhsz
                            " exceeds device count %" PRIhsz,
                            options->device_index, device_count);
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_diagnostic_tap_names(
      options->diagnostic_tap_names));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Qwen issue wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("Qwen issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue final signal is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Qwen issue")));
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_qwen_execution_allocate(
    id4_ideogram4_session_t* session, iree_allocator_t host_allocator,
    id4_ideogram4_qwen_execution_t** out_execution) {
  *out_execution = NULL;
  id4_ideogram4_qwen_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  memset(execution, 0, sizeof(*execution));
  execution->host_allocator = host_allocator;
  execution->session = session;
  id4_ideogram4_session_retain(session);
  *out_execution = execution;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_qwen_create_plan(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_qwen_issue_options_t* options,
    const id4_ideogram4_qwen_inputs_t* inputs,
    id4_ideogram4_qwen_execution_t* execution) {
  id4_qwen3_vl_stage_plan_options_t qwen_options;
  memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = inputs->token_count;
  qwen_options.weight_execution_strategy =
      options->qwen_weight_execution_strategy;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &qwen_options;
  plan_options.flags =
      options->diagnostic_tap_names.count == 0
          ? 0
          : ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.device_index = options->device_index;
  plan_options.queue_affinity = options->queue_affinity;
  plan_options.diagnostic_tap_names = options->diagnostic_tap_names;
  plan_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_plan(session->qwen_stage, &plan_options,
                                 &execution->plan);
}

static iree_status_t id4_ideogram4_qwen_create_semaphores(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    id4_ideogram4_qwen_execution_t* execution) {
  iree_status_t status = iree_hal_semaphore_create(
      device, queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &execution->prepare_semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, queue_affinity, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->upload_semaphore);
  }
  return status;
}

static iree_status_t id4_ideogram4_qwen_prepare_bundle(
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t* execution) {
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = 1;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore, &signal_payload_value, execution->prepare_semaphore,
      signal_payload_value);

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = options->parameter_provider;
  prepare_options.kernel_library = options->kernel_library;
  prepare_options.wait_semaphore_list = options->wait_semaphore_list;
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.command_buffer_mode = options->command_buffer_mode;
  prepare_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_prepare(execution->session->qwen_stage,
                                    execution->plan, &prepare_options,
                                    &execution->bundle);
}

static iree_status_t id4_ideogram4_qwen_allocate_bindings(
    iree_hal_device_t* device,
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t* execution) {
  iree_status_t status = id4_pipeline_allocate_boundary_bindings(
      device, options->queue_affinity, execution->plan,
      execution->host_allocator, &execution->boundary_bindings);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_allocate_diagnostic_tap_bindings(
        device, options->queue_affinity, execution->plan,
        execution->host_allocator, &execution->diagnostic_tap_bindings);
  }
  return status;
}

static iree_status_t id4_ideogram4_qwen_upload_inputs(
    iree_hal_device_t* device,
    const id4_ideogram4_qwen_issue_options_t* options,
    const id4_ideogram4_qwen_inputs_t* inputs,
    id4_ideogram4_qwen_execution_t* execution) {
  const iree_host_size_t token_ids_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_ids[0]);
  const iree_host_size_t attention_mask_length =
      inputs->token_count * (iree_host_size_t)inputs->token_count *
      sizeof(inputs->attention_mask[0]);
  const iree_host_size_t token_weights_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_weights[0]);

  id4_ideogram4_boundary_upload_context_t upload_context = {
      .device = device,
      .queue_affinity = options->queue_affinity,
      .plan = execution->plan,
      .boundary_bindings = &execution->boundary_bindings,
      .semaphore = execution->upload_semaphore,
      .payload_value = &execution->upload_payload_value,
  };

  IREE_RETURN_IF_ERROR(id4_ideogram4_upload_boundary_tensor(
      &upload_context, IREE_SV("token_ids"), inputs->token_ids,
      token_ids_length, options->wait_semaphore_list));
  IREE_RETURN_IF_ERROR(id4_ideogram4_upload_boundary_tensor(
      &upload_context, IREE_SV("attention_mask"), inputs->attention_mask,
      attention_mask_length, iree_hal_semaphore_list_empty()));
  return id4_ideogram4_upload_boundary_tensor(
      &upload_context, IREE_SV("token_weights"), inputs->token_weights,
      token_weights_length, iree_hal_semaphore_list_empty());
}

static iree_status_t id4_ideogram4_qwen_find_outputs(
    id4_ideogram4_qwen_execution_t* execution) {
  return id4_pipeline_find_boundary_binding(
      execution->plan, &execution->boundary_bindings, IREE_SV("condition"),
      &execution->condition_binding);
}

static iree_status_t id4_ideogram4_qwen_issue_bundle(
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t* execution) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_two_semaphore_list(
      wait_semaphores, wait_payload_values, execution->prepare_semaphore, 1,
      execution->upload_semaphore, execution->upload_payload_value);

  id4_pipeline_stage_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = execution->boundary_bindings.count;
  issue_options.boundary_bindings = execution->boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count =
      execution->diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings =
      execution->diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = wait_list;
  issue_options.signal_semaphore_list = options->signal_semaphore_list;
  issue_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_issue(execution->session->qwen_stage,
                                  execution->bundle, &issue_options);
}

iree_status_t id4_ideogram4_session_issue_qwen(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_qwen_issue_options(session, options));

  id4_ideogram4_qwen_inputs_t inputs;
  memset(&inputs, 0, sizeof(inputs));
  id4_ideogram4_qwen_execution_t* execution = NULL;

  iree_status_t status = id4_ideogram4_qwen_execution_allocate(
      session, session->host_allocator, &execution);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_qwen_lowering_options_t lowering_options;
    memset(&lowering_options, 0, sizeof(lowering_options));
    lowering_options.structure_size = sizeof(lowering_options);
    lowering_options.tokenizer = options->tokenizer;
    lowering_options.request = options->request;
    lowering_options.tokenizer_flags = options->tokenizer_flags;
    lowering_options.max_token_count = session->qwen_model.max_token_count;
    lowering_options.vocab_size = session->qwen_model.vocab_size;
    status = id4_ideogram4_request_lower_qwen_inputs(
        &lowering_options, session->host_allocator, &inputs);
  }
  if (iree_status_is_ok(status)) {
    execution->token_count = inputs.token_count;
    status =
        id4_ideogram4_qwen_create_plan(session, options, &inputs, execution);
  }
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    device = iree_hal_device_group_device_at(
        id4_pipeline_stage_services(session->qwen_stage)->device_group,
        options->device_index);
    status = id4_ideogram4_qwen_create_semaphores(
        device, options->queue_affinity, execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_qwen_prepare_bundle(options, execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_qwen_allocate_bindings(device, options, execution);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_ideogram4_qwen_upload_inputs(device, options, &inputs, execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_qwen_find_outputs(execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_qwen_issue_bundle(options, execution);
  }
  id4_ideogram4_qwen_inputs_deinitialize(&inputs, session->host_allocator);
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_qwen_execution_release(execution);
  }
  return status;
}

void id4_ideogram4_qwen_execution_release(
    id4_ideogram4_qwen_execution_t* execution) {
  if (!execution) return;
  id4_pipeline_buffer_binding_set_deinitialize(
      &execution->diagnostic_tap_bindings);
  id4_pipeline_buffer_binding_set_deinitialize(&execution->boundary_bindings);
  id4_pipeline_bundle_release(execution->bundle);
  id4_pipeline_plan_release(execution->plan);
  iree_hal_semaphore_release(execution->upload_semaphore);
  iree_hal_semaphore_release(execution->prepare_semaphore);
  id4_ideogram4_session_release(execution->session);
  iree_allocator_t host_allocator = execution->host_allocator;
  iree_allocator_free(host_allocator, execution);
}

const id4_pipeline_plan_t* id4_ideogram4_qwen_execution_plan(
    const id4_ideogram4_qwen_execution_t* execution) {
  IREE_ASSERT_ARGUMENT(execution);
  return execution->plan;
}

iree_status_t id4_ideogram4_qwen_execution_result(
    const id4_ideogram4_qwen_execution_t* execution,
    id4_ideogram4_qwen_result_t* out_result) {
  if (!execution || !out_result) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen execution and result output are required");
  }
  out_result->token_count = execution->token_count;
  out_result->condition_binding = execution->condition_binding;
  return iree_ok_status();
}
