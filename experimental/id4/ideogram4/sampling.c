// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/sampling.h"

#include <math.h>
#include <string.h>

typedef struct id4_ideogram4_sampler_preset_config_t {
  // Stable name accepted by request parsing and command-line bindings.
  iree_string_view_t name;
  // Number of denoise steps in the preset.
  uint32_t step_count;
  // Resolution-independent logit-normal mean.
  double base_mean;
  // Logit-normal standard deviation.
  double standard_deviation;
  // Number of final polish steps using |cleanup_guidance_scale|.
  uint32_t cleanup_step_count;
  // Guidance scale used by the main denoise steps.
  float main_guidance_scale;
  // Guidance scale used by the final polish steps.
  float cleanup_guidance_scale;
  // Normal quantiles for the preset's float32 linear step intervals.
  const double* interval_quantiles;
} id4_ideogram4_sampler_preset_config_t;

// torch.special.ndtri(torch.linspace(0, 1, 13, dtype=torch.float32)).
static const double id4_ideogram4_quantiles_12[] = {
    -INFINITY,
    -1.3829941109015564,
    -0.96742154622163301,
    -0.67448975019608171,
    -0.43072727197391308,
    -0.21042834333077498,
    0.0,
    0.21042834333077498,
    0.43072719000928167,
    0.67448975019608171,
    0.96742148658143123,
    1.3829942566933073,
    INFINITY,
};

// torch.special.ndtri(torch.linspace(0, 1, 21, dtype=torch.float32)).
static const double id4_ideogram4_quantiles_20[] = {
    -INFINITY,
    -1.6448536197274128,
    -1.2815515570538296,
    -1.0364333639298176,
    -0.84162122292777997,
    -0.67448975019608171,
    -0.52440047842221371,
    -0.38532048249957768,
    -0.25334708770787262,
    -0.12566130167778,
    0.0,
    0.12566137697327023,
    0.25334716484750908,
    0.3853204020395285,
    0.52440047842221371,
    0.67448975019608171,
    0.84162127615345206,
    1.036433491749684,
    1.281551429692279,
    1.6448535113665226,
    INFINITY,
};

// torch.special.ndtri(torch.linspace(0, 1, 49, dtype=torch.float32)).
static const double id4_ideogram4_quantiles_48[] = {
    -INFINITY,
    -2.0368341193141046,
    -1.7316643821816959,
    -1.5341205443525463,
    -1.3829941109015564,
    -1.2581615335888077,
    -1.1503493803760079,
    -1.0544724083522907,
    -0.96742154622163301,
    -0.88714655901887607,
    -0.81221776686849056,
    -0.74159402747028946,
    -0.67448975019608171,
    -0.61029458018798466,
    -0.54852222481079438,
    -0.48877641111466952,
    -0.43072727197391308,
    -0.37409535678583111,
    -0.31863936396437514,
    -0.26414695104075725,
    -0.21042834333077498,
    -0.1573106846101707,
    -0.10463343057627626,
    -0.052245130505695701,
    0.0,
    0.052245130505695701,
    0.10463335546287943,
    0.1573106846101707,
    0.21042834333077498,
    0.26414687368525941,
    0.31863936396437514,
    0.37409535678583111,
    0.43072719000928167,
    0.48877641111466952,
    0.54852222481079438,
    0.6102946701830303,
    0.67448975019608171,
    0.74159397829661022,
    0.81221787076276075,
    0.88714655901887607,
    0.96742148658143123,
    1.0544725386055829,
    1.1503493803760079,
    1.258161451165946,
    1.3829942566933073,
    1.5341205443525463,
    1.7316641730734978,
    2.0368345280946456,
    INFINITY,
};

static const id4_ideogram4_sampler_preset_config_t
    id4_ideogram4_sampler_presets[] = {
        {
            .name = IREE_SVL("V4_QUALITY_48"),
            .step_count = 48,
            .base_mean = 0.0,
            .standard_deviation = 1.5,
            .cleanup_step_count = 3,
            .main_guidance_scale = 7.0f,
            .cleanup_guidance_scale = 3.0f,
            .interval_quantiles = id4_ideogram4_quantiles_48,
        },
        {
            .name = IREE_SVL("V4_DEFAULT_20"),
            .step_count = 20,
            .base_mean = 0.0,
            .standard_deviation = 1.75,
            .cleanup_step_count = 2,
            .main_guidance_scale = 7.0f,
            .cleanup_guidance_scale = 3.0f,
            .interval_quantiles = id4_ideogram4_quantiles_20,
        },
        {
            .name = IREE_SVL("V4_TURBO_12"),
            .step_count = 12,
            .base_mean = 0.5,
            .standard_deviation = 1.75,
            .cleanup_step_count = 1,
            .main_guidance_scale = 7.0f,
            .cleanup_guidance_scale = 3.0f,
            .interval_quantiles = id4_ideogram4_quantiles_12,
        },
};

static const id4_ideogram4_sampler_preset_config_t*
id4_ideogram4_sampler_preset_config(id4_ideogram4_sampler_preset_t preset) {
  const uint32_t preset_ordinal = (uint32_t)preset;
  if (preset_ordinal == 0 ||
      preset_ordinal > IREE_ARRAYSIZE(id4_ideogram4_sampler_presets)) {
    return NULL;
  }
  return &id4_ideogram4_sampler_presets[preset_ordinal - 1];
}

iree_status_t id4_ideogram4_sampler_preset_parse(
    iree_string_view_t value, id4_ideogram4_sampler_preset_t* out_preset) {
  IREE_ASSERT_ARGUMENT(out_preset);
  *out_preset = ID4_IDEOGRAM4_SAMPLER_PRESET_INVALID;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(id4_ideogram4_sampler_presets); ++i) {
    if (iree_string_view_equal(value, id4_ideogram4_sampler_presets[i].name)) {
      *out_preset = (id4_ideogram4_sampler_preset_t)(i + 1);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unrecognized Ideogram 4 sampler preset `%.*s`",
                          (int)value.size, value.data);
}

iree_string_view_t id4_ideogram4_sampler_preset_name(
    id4_ideogram4_sampler_preset_t preset) {
  const id4_ideogram4_sampler_preset_config_t* config =
      id4_ideogram4_sampler_preset_config(preset);
  return config ? config->name : iree_string_view_empty();
}

uint32_t id4_ideogram4_sampler_preset_step_count(
    id4_ideogram4_sampler_preset_t preset) {
  const id4_ideogram4_sampler_preset_config_t* config =
      id4_ideogram4_sampler_preset_config(preset);
  return config ? config->step_count : 0;
}

static float id4_ideogram4_sampler_interval_flow_time(
    const id4_ideogram4_sampler_preset_config_t* config, double mean,
    uint32_t interval_ordinal) {
  const double y = mean + config->standard_deviation *
                              config->interval_quantiles[interval_ordinal];
  double flow_time = 1.0 / (1.0 + exp(y));
  const double minimum_flow_time = 1.0 / (1.0 + exp(9.0));
  const double maximum_flow_time = 1.0 / (1.0 + exp(-7.5));
  flow_time = fmin(fmax(flow_time, minimum_flow_time), maximum_flow_time);
  return (float)flow_time;
}

iree_status_t id4_ideogram4_sampler_preset_lower_schedule(
    id4_ideogram4_sampler_preset_t preset, uint32_t image_width,
    uint32_t image_height, iree_allocator_t host_allocator,
    id4_ideogram4_denoise_schedule_t* out_schedule) {
  IREE_ASSERT_ARGUMENT(out_schedule);
  memset(out_schedule, 0, sizeof(*out_schedule));
  const id4_ideogram4_sampler_preset_config_t* config =
      id4_ideogram4_sampler_preset_config(preset);
  if (!config) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 sampler preset is invalid");
  }
  if (image_width == 0 || image_height == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 sampler image dimensions are zero");
  }
  const double pixel_count = (double)image_width * (double)image_height;
  const double reference_pixel_count = 512.0 * 512.0;
  const double mean =
      config->base_mean + 0.5 * log(pixel_count / reference_pixel_count);

  id4_ideogram4_denoise_step_t* steps = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, config->step_count, sizeof(steps[0]), (void**)&steps);
  if (iree_status_is_ok(status)) {
    for (uint32_t step_ordinal = 0; step_ordinal < config->step_count;
         ++step_ordinal) {
      const uint32_t current_interval = config->step_count - step_ordinal;
      const uint32_t next_interval = current_interval - 1;
      steps[step_ordinal].flow_time = id4_ideogram4_sampler_interval_flow_time(
          config, mean, current_interval);
      steps[step_ordinal].next_flow_time =
          id4_ideogram4_sampler_interval_flow_time(config, mean, next_interval);
      const uint32_t main_step_count =
          config->step_count - config->cleanup_step_count;
      steps[step_ordinal].guidance_scale = step_ordinal < main_step_count
                                               ? config->main_guidance_scale
                                               : config->cleanup_guidance_scale;
    }
    out_schedule->step_count = config->step_count;
    out_schedule->steps = steps;
  }
  return status;
}

void id4_ideogram4_denoise_schedule_deinitialize(
    id4_ideogram4_denoise_schedule_t* schedule,
    iree_allocator_t host_allocator) {
  if (!schedule) return;
  iree_allocator_free(host_allocator, schedule->steps);
  memset(schedule, 0, sizeof(*schedule));
}
