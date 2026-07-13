// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_SAMPLING_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_SAMPLING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Advertised Ideogram 4 sampler presets.
typedef enum id4_ideogram4_sampler_preset_e {
  // Invalid or unspecified sampler preset.
  ID4_IDEOGRAM4_SAMPLER_PRESET_INVALID = 0,
  // Highest-quality 48-step sampler.
  ID4_IDEOGRAM4_SAMPLER_PRESET_V4_QUALITY_48 = 1,
  // Default 20-step sampler.
  ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20 = 2,
  // Low-latency 12-step sampler.
  ID4_IDEOGRAM4_SAMPLER_PRESET_V4_TURBO_12 = 3,
} id4_ideogram4_sampler_preset_t;

// Host-side scalar state consumed by one device denoise step.
typedef struct id4_ideogram4_denoise_step_t {
  // Current flow time consumed by the DiT timestep embedding.
  float flow_time;
  // Next flow time used by the Euler update.
  float next_flow_time;
  // Asymmetric classifier-free guidance scale.
  float guidance_scale;
} id4_ideogram4_denoise_step_t;

// Resolution-specialized denoise schedule for one request.
typedef struct id4_ideogram4_denoise_schedule_t {
  // Number of denoise steps in |steps|.
  uint32_t step_count;
  // Allocator-owned step table with |step_count| entries.
  id4_ideogram4_denoise_step_t* steps;
} id4_ideogram4_denoise_schedule_t;

// Parses an advertised sampler preset name.
iree_status_t id4_ideogram4_sampler_preset_parse(
    iree_string_view_t value, id4_ideogram4_sampler_preset_t* out_preset);

// Returns the stable advertised name for |preset| or an empty string when the
// preset value is invalid.
iree_string_view_t id4_ideogram4_sampler_preset_name(
    id4_ideogram4_sampler_preset_t preset);

// Returns the denoise step count for |preset| or zero when the preset value is
// invalid.
uint32_t id4_ideogram4_sampler_preset_step_count(
    id4_ideogram4_sampler_preset_t preset);

// Lowers |preset| into the official resolution-adjusted logit-normal schedule.
iree_status_t id4_ideogram4_sampler_preset_lower_schedule(
    id4_ideogram4_sampler_preset_t preset, uint32_t image_width,
    uint32_t image_height, iree_allocator_t host_allocator,
    id4_ideogram4_denoise_schedule_t* out_schedule);

// Releases storage owned by |schedule|.
void id4_ideogram4_denoise_schedule_deinitialize(
    id4_ideogram4_denoise_schedule_t* schedule,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_SAMPLING_H_
