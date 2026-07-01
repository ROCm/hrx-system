// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/session_generation.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>

#include "experimental/id4/stages/ideogram4_decode.h"

const id4_ideogram4_generation_stage_descriptor_t
    id4_ideogram4_generation_stage_descriptors
        [ID4_IDEOGRAM4_GENERATION_STAGE_COUNT] = {
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
                .resident_stage_bit =
                    ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_QWEN,
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

const id4_ideogram4_generation_phase_descriptor_t
    id4_ideogram4_generation_phase_descriptors
        [ID4_IDEOGRAM4_GENERATION_PHASE_DESCRIPTOR_COUNT] = {
            {
                // Phase that produces prompt conditioning and initial latent
                // noise.
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
                // Phase that repeats the conditioned/unconditioned DiT
                // branches.
                .name = IREE_SVL("denoise"),
                .phase_mask = ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE,
                .flags =
                    ID4_IDEOGRAM4_GENERATION_PHASE_REPEATED_PER_DENOISE_STEP,
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

const id4_ideogram4_generation_boundary_alias_t
    id4_ideogram4_generation_boundary_aliases
        [ID4_IDEOGRAM4_GENERATION_BOUNDARY_ALIAS_COUNT] = {
            {
                // Initial latent noise consumed by the conditioned DiT branch.
                .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
                .source_name = IREE_SVL("x"),
                .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
                .target_name = IREE_SVL("x"),
            },
            {
                // Initial latent noise consumed by the unconditioned DiT
                // branch.
                .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
                .source_name = IREE_SVL("x"),
                .target_stage =
                    ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
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
                // Qwen text conditioning consumed by the conditioned DiT
                // branch.
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
                .source_stage =
                    ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
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

const id4_ideogram4_generation_stage_descriptor_t*
id4_ideogram4_generation_stage_descriptor(
    id4_ideogram4_generation_stage_ordinal_t ordinal) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (descriptor->ordinal == ordinal) {
      return descriptor;
    }
  }
  return NULL;
}

const id4_ideogram4_generation_stage_descriptor_t*
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

const id4_pipeline_plan_t* id4_ideogram4_generation_stage_plan(
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

id4_pipeline_program_shape_t
id4_ideogram4_generation_request_diffusion_latent_shape(
    const id4_ideogram4_request_t* request) {
  return id4_pipeline_program_make_shape_rank4(
      request->generation.latent_width, request->generation.latent_height, 128,
      1);
}

iree_status_t id4_ideogram4_validate_generation_request(
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

iree_status_t id4_ideogram4_validate_generation_phase_mask(
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

iree_status_t id4_ideogram4_validate_single_generation_phase_mask(
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

iree_status_t id4_ideogram4_validate_generation_resident_stage_mask(
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

iree_status_t id4_ideogram4_validate_generation_prepare_residency_options(
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
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES:
      if (resident_stage_mask != ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 phase-stage-bundle residency mode must not select "
            "resident stage bundles");
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

iree_status_t id4_ideogram4_validate_generation_bundle_residency(
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
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES:
      if (resident_stage_mask != ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 phase-stage-bundle generation bundle must not retain "
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
