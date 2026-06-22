// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/vae_program.h"

#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/kernel_library.h"
#include "iree/hal/command_buffer.h"

enum {
  ID4_VAE_DECODE_WORKGROUP_SIZE_X = 256,
  ID4_VAE_CONFIG_VALUE_BUFFER_CAPACITY = 24,
  ID4_VAE_DECODE_CONFIG_CAPACITY = 15,
};

typedef struct id4_vae_program_config_list_t {
  // Number of config bindings stored in bindings.
  iree_host_size_t count;
  // Fixed-capacity kernel config binding storage.
  id4_pipeline_kernel_config_binding_t bindings[ID4_VAE_DECODE_CONFIG_CAPACITY];
  // Fixed-capacity string storage backing binding values.
  char value_storage[ID4_VAE_DECODE_CONFIG_CAPACITY]
                    [ID4_VAE_CONFIG_VALUE_BUFFER_CAPACITY];
} id4_vae_program_config_list_t;

static iree_status_t id4_vae_program_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_vae_program_format_u64(
    uint64_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%" PRIu64, value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format VAE program config value");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_vae_program_config_list_add_u64(
    id4_vae_program_config_list_t* config_list, iree_string_view_t key,
    uint64_t value) {
  if (config_list->count >= IREE_ARRAYSIZE(config_list->bindings)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VAE decode config binding capacity exceeded");
  }
  iree_string_view_t value_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_vae_program_format_u64(
      value, config_list->value_storage[config_list->count],
      ID4_VAE_CONFIG_VALUE_BUFFER_CAPACITY, &value_string));
  config_list->bindings[config_list->count] =
      id4_pipeline_make_kernel_config_binding(key, value_string);
  ++config_list->count;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_validate_model_config(
    id4_vae_model_config_t model) {
  if (!iree_all_bits_set(model.capabilities, ID4_VAE_CAPABILITY_DECODE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE model config must support decode");
  }
  if (model.scale_x == 0 || model.scale_y == 0 || model.scale_t == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE model scale factors must be greater than zero");
  }
  if (model.latent_channel_count == 0 || model.decoded_channel_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE model channel counts must be greater than zero");
  }
  if (model.min_tile_size_x == 0 || model.min_tile_size_y == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE model minimum tile dimensions must be greater than zero");
  }
  if (model.default_tile_size_x < model.min_tile_size_x ||
      model.default_tile_size_y < model.min_tile_size_y) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE model default tile dimensions must be at least the minimum");
  }
  if (!(model.max_overlap >= 0.0f && model.max_overlap <= 0.5f)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE model maximum overlap must be in the range [0, 0.5]");
  }
  return iree_ok_status();
}

static iree_status_t id4_vae_program_validate_request_shape(
    id4_vae_model_config_t model, id4_pipeline_program_shape_t latent_shape) {
  if (latent_shape.rank != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE decode latent shape must be rank-4 NHWC, got rank %u",
        latent_shape.rank);
  }
  for (uint32_t i = 0; i < latent_shape.rank; ++i) {
    if (latent_shape.dims[i] == 0 || latent_shape.dims[i] > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VAE decode latent shape dimension %u is out of range", i);
    }
  }
  if (latent_shape.dims[3] != model.latent_channel_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE decode latent channel count %" PRIu64
                            " does not match model channel count %u",
                            latent_shape.dims[3], model.latent_channel_count);
  }
  return iree_ok_status();
}

static uint32_t id4_vae_program_ceil_div_u32(uint32_t dividend,
                                             uint32_t divisor) {
  return dividend / divisor + (dividend % divisor != 0 ? 1 : 0);
}

static iree_status_t id4_vae_program_mul_u64(uint64_t lhs, uint64_t rhs,
                                             uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VAE decode dimension multiplication overflowed");
  }
  *out_result = lhs * rhs;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_nhwc_element_count(
    uint32_t batch_count, uint32_t height, uint32_t width,
    uint32_t channel_count, uint64_t* out_element_count) {
  uint64_t element_count = batch_count;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_mul_u64(element_count, height, &element_count));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_mul_u64(element_count, width, &element_count));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_mul_u64(element_count, channel_count, &element_count));
  *out_element_count = element_count;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_validate_overlap(
    id4_vae_model_config_t model, id4_vae_tiling_config_t tiling) {
  if (tiling.mode == ID4_VAE_TILING_MODE_DISABLED) return iree_ok_status();
  if (!(tiling.overlap >= 0.0f && tiling.overlap <= model.max_overlap)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE tiling overlap must be in the range [0, %.3f]",
                            model.max_overlap);
  }
  return iree_ok_status();
}

static uint32_t id4_vae_program_round_positive_to_u32(float value) {
  return (uint32_t)(value + 0.5f);
}

static iree_status_t id4_vae_program_resolve_relative_tile_size(
    uint32_t latent_size, uint32_t min_tile_size, float relative_size,
    float overlap, uint32_t* out_tile_size) {
  if (!(relative_size > 0.0f)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE relative tile size must be greater than zero");
  }
  float factor = relative_size;
  if (factor > 1.0f) {
    factor = 1.0f / (factor - factor * overlap + overlap);
  }
  uint32_t tile_size =
      id4_vae_program_round_positive_to_u32((float)latent_size * factor);
  if (tile_size < min_tile_size || tile_size > latent_size) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "resolved VAE relative tile size %u is outside [%u, %u]", tile_size,
        min_tile_size, latent_size);
  }
  *out_tile_size = tile_size;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_tile_scratch_bytes(
    id4_vae_model_config_t model, uint32_t batch_count, uint32_t tile_size_x,
    uint32_t tile_size_y, iree_device_size_t* out_byte_length) {
  uint64_t tile_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_nhwc_element_count(
      batch_count, tile_size_y * model.scale_y, tile_size_x * model.scale_x,
      model.decoded_channel_count, &tile_element_count));
  if (tile_element_count > UINT64_MAX / sizeof(float)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VAE tile scratch byte length overflowed");
  }
  *out_byte_length = tile_element_count * sizeof(float);
  return iree_ok_status();
}

static iree_status_t id4_vae_program_resolve_budget_tile_size(
    id4_vae_model_config_t model, uint32_t batch_count, uint32_t latent_width,
    uint32_t latent_height, iree_device_size_t memory_budget,
    uint32_t* out_tile_size_x, uint32_t* out_tile_size_y) {
  if (memory_budget == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE memory-budget tiling requires a budget");
  }

  uint32_t tile_size_x = latent_width;
  uint32_t tile_size_y = latent_height;
  iree_device_size_t scratch_bytes = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_tile_scratch_bytes(
      model, batch_count, tile_size_x, tile_size_y, &scratch_bytes));
  while (scratch_bytes > memory_budget) {
    if (tile_size_x == model.min_tile_size_x &&
        tile_size_y == model.min_tile_size_y) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VAE minimum tile scratch size %" PRIu64
                              " exceeds memory budget %" PRIu64,
                              (uint64_t)scratch_bytes, (uint64_t)memory_budget);
    }
    if (tile_size_x >= tile_size_y && tile_size_x > model.min_tile_size_x) {
      --tile_size_x;
    } else if (tile_size_y > model.min_tile_size_y) {
      --tile_size_y;
    } else {
      --tile_size_x;
    }
    IREE_RETURN_IF_ERROR(id4_vae_program_tile_scratch_bytes(
        model, batch_count, tile_size_x, tile_size_y, &scratch_bytes));
  }

  *out_tile_size_x = tile_size_x;
  *out_tile_size_y = tile_size_y;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_resolve_tile_size(
    id4_vae_model_config_t model, id4_vae_decode_request_config_t request,
    uint32_t* out_tile_size_x, uint32_t* out_tile_size_y) {
  const uint32_t batch_count = (uint32_t)request.latent_shape.dims[0];
  const uint32_t latent_height = (uint32_t)request.latent_shape.dims[1];
  const uint32_t latent_width = (uint32_t)request.latent_shape.dims[2];
  switch (request.tiling.mode) {
    case ID4_VAE_TILING_MODE_DISABLED:
      *out_tile_size_x = latent_width;
      *out_tile_size_y = latent_height;
      return iree_ok_status();
    case ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE: {
      const uint32_t tile_size_x = request.tiling.tile_size_x;
      const uint32_t tile_size_y = request.tiling.tile_size_y;
      if (tile_size_x < model.min_tile_size_x ||
          tile_size_y < model.min_tile_size_y || tile_size_x > latent_width ||
          tile_size_y > latent_height) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "explicit VAE tile size %ux%u is outside [%u..%u]x[%u..%u]",
            tile_size_x, tile_size_y, model.min_tile_size_x, latent_width,
            model.min_tile_size_y, latent_height);
      }
      *out_tile_size_x = tile_size_x;
      *out_tile_size_y = tile_size_y;
      return iree_ok_status();
    }
    case ID4_VAE_TILING_MODE_RELATIVE_TILE_SIZE: {
      IREE_RETURN_IF_ERROR(id4_vae_program_resolve_relative_tile_size(
          latent_width, model.min_tile_size_x, request.tiling.relative_size_x,
          request.tiling.overlap, out_tile_size_x));
      return id4_vae_program_resolve_relative_tile_size(
          latent_height, model.min_tile_size_y, request.tiling.relative_size_y,
          request.tiling.overlap, out_tile_size_y);
    }
    case ID4_VAE_TILING_MODE_MEMORY_BUDGET:
      return id4_vae_program_resolve_budget_tile_size(
          model, batch_count, latent_width, latent_height,
          request.tiling.memory_budget, out_tile_size_x, out_tile_size_y);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VAE tiling mode %d is invalid",
                              (int)request.tiling.mode);
  }
}

static uint32_t id4_vae_program_tile_count(uint32_t latent_size,
                                           uint32_t tile_size, float overlap) {
  if (tile_size >= latent_size) return 1;
  uint32_t overlap_pixels =
      id4_vae_program_round_positive_to_u32((float)tile_size * overlap);
  if (overlap_pixels >= tile_size) overlap_pixels = tile_size - 1;
  const uint32_t step = tile_size - overlap_pixels;
  return id4_vae_program_ceil_div_u32(latent_size - tile_size, step) + 1;
}

static iree_status_t id4_vae_program_validate_decode_options(
    const id4_vae_program_options_t* options,
    const id4_pipeline_program_builder_t* builder,
    id4_vae_decode_tiling_plan_t* out_tiling_plan) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE program options are required");
  }
  IREE_RETURN_IF_ERROR(id4_vae_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("VAE program")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VAE program extension structures are not supported");
  }
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE program builder is required");
  }
  return id4_vae_program_resolve_decode_tiling(options->model, options->request,
                                               out_tiling_plan);
}

static iree_status_t id4_vae_program_import_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_import_tensor_flags_t flags,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_import_tensor_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.flags = flags;
  options.name = name;
  options.dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32;
  options.shape = shape;
  return id4_pipeline_program_import_tensor(builder, &options, out_tensor);
}

static iree_status_t id4_vae_program_acquire_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_acquire_tensor_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.name = name;
  options.dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32;
  options.shape = shape;
  return id4_pipeline_program_acquire_tensor(builder, &options, out_tensor);
}

static iree_hal_dispatch_config_t id4_vae_program_make_dispatch_config(
    uint32_t element_count) {
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(
          id4_vae_program_ceil_div_u32(element_count,
                                       ID4_VAE_DECODE_WORKGROUP_SIZE_X),
          1, 1);
  dispatch_config.workgroup_size[0] = ID4_VAE_DECODE_WORKGROUP_SIZE_X;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

static iree_status_t id4_vae_program_build_decode_configs(
    id4_vae_model_config_t model, id4_vae_decode_tiling_plan_t tiling_plan,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.output_element_count"),
      tiling_plan.decoded_element_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.latent_element_count"),
      tiling_plan.latent_element_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.latent_height"),
      tiling_plan.latent_height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.latent_width"),
      tiling_plan.latent_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.latent_channel_count"),
      tiling_plan.latent_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.decoded_height"),
      tiling_plan.decoded_height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.decoded_width"),
      tiling_plan.decoded_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.decoded_channel_count"),
      tiling_plan.decoded_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.scale_x"), model.scale_x));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.scale_y"), model.scale_y));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.tile_width"),
      tiling_plan.tile_size_x));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.tile_height"),
      tiling_plan.tile_size_y));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.tile_count_x"),
      tiling_plan.tile_count_x));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.tile_count_y"),
      tiling_plan.tile_count_y));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.overlap_milli"),
      tiling_plan.overlap_milli);
}

iree_status_t id4_vae_program_resolve_decode_tiling(
    id4_vae_model_config_t model, id4_vae_decode_request_config_t request,
    id4_vae_decode_tiling_plan_t* out_tiling_plan) {
  IREE_ASSERT_ARGUMENT(out_tiling_plan);
  memset(out_tiling_plan, 0, sizeof(*out_tiling_plan));
  IREE_RETURN_IF_ERROR(id4_vae_program_validate_model_config(model));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_validate_request_shape(model, request.latent_shape));
  if (request.tiling.mode != ID4_VAE_TILING_MODE_DISABLED &&
      !iree_all_bits_set(model.capabilities,
                         ID4_VAE_CAPABILITY_SPATIAL_TILING)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE model config does not support spatial tiling");
  }
  IREE_RETURN_IF_ERROR(id4_vae_program_validate_overlap(model, request.tiling));

  const uint32_t batch_count = (uint32_t)request.latent_shape.dims[0];
  const uint32_t latent_height = (uint32_t)request.latent_shape.dims[1];
  const uint32_t latent_width = (uint32_t)request.latent_shape.dims[2];
  const uint32_t latent_channel_count = (uint32_t)request.latent_shape.dims[3];
  uint32_t tile_size_x = 0;
  uint32_t tile_size_y = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_resolve_tile_size(
      model, request, &tile_size_x, &tile_size_y));

  const float overlap = request.tiling.mode == ID4_VAE_TILING_MODE_DISABLED
                            ? 0.0f
                            : request.tiling.overlap;
  const uint32_t decoded_height = latent_height * model.scale_y;
  const uint32_t decoded_width = latent_width * model.scale_x;
  const uint32_t tile_count_x =
      id4_vae_program_tile_count(latent_width, tile_size_x, overlap);
  const uint32_t tile_count_y =
      id4_vae_program_tile_count(latent_height, tile_size_y, overlap);

  uint64_t decoded_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_nhwc_element_count(
      batch_count, decoded_height, decoded_width, model.decoded_channel_count,
      &decoded_element_count));
  if (decoded_element_count == 0 ||
      decoded_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE decoded element count %" PRIu64 " exceeds max count %u",
        decoded_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  uint64_t latent_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_nhwc_element_count(
      batch_count, latent_height, latent_width, latent_channel_count,
      &latent_element_count));
  uint64_t tile_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_nhwc_element_count(
      batch_count, tile_size_y * model.scale_y, tile_size_x * model.scale_x,
      model.decoded_channel_count, &tile_element_count));

  out_tiling_plan->batch_count = batch_count;
  out_tiling_plan->latent_height = latent_height;
  out_tiling_plan->latent_width = latent_width;
  out_tiling_plan->latent_channel_count = latent_channel_count;
  out_tiling_plan->decoded_height = decoded_height;
  out_tiling_plan->decoded_width = decoded_width;
  out_tiling_plan->decoded_channel_count = model.decoded_channel_count;
  out_tiling_plan->tile_size_x = tile_size_x;
  out_tiling_plan->tile_size_y = tile_size_y;
  out_tiling_plan->tile_count_x = tile_count_x;
  out_tiling_plan->tile_count_y = tile_count_y;
  out_tiling_plan->overlap_x = overlap;
  out_tiling_plan->overlap_y = overlap;
  out_tiling_plan->overlap_milli =
      id4_vae_program_round_positive_to_u32(overlap * 1000.0f);
  out_tiling_plan->decoded_element_count = decoded_element_count;
  out_tiling_plan->latent_element_count = latent_element_count;
  out_tiling_plan->tile_element_count = tile_element_count;
  out_tiling_plan->estimated_transient_peak =
      tile_element_count * sizeof(float);
  return iree_ok_status();
}

iree_status_t id4_vae_program_author_decode(
    const id4_vae_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  id4_vae_decode_tiling_plan_t tiling_plan;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_validate_decode_options(options, builder, &tiling_plan));

  id4_pipeline_program_shape_t latent_shape = options->request.latent_shape;
  id4_pipeline_program_shape_t decoded_shape =
      id4_pipeline_program_make_shape_rank4(
          tiling_plan.batch_count, tiling_plan.decoded_height,
          tiling_plan.decoded_width, tiling_plan.decoded_channel_count);
  id4_pipeline_program_shape_t tile_shape =
      id4_pipeline_program_make_shape_rank4(
          tiling_plan.batch_count,
          tiling_plan.tile_size_y * options->model.scale_y,
          tiling_plan.tile_size_x * options->model.scale_x,
          tiling_plan.decoded_channel_count);

  id4_pipeline_program_tensor_t latent = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_import_tensor(
      builder, IREE_SV("media.latent.input"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED, latent_shape,
      &latent));
  id4_pipeline_program_tensor_t decoded_tile =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, IREE_SV("vae.decode.tile"), tile_shape, &decoded_tile));
  id4_pipeline_program_tensor_t decoded_image =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(
      id4_vae_program_import_tensor(builder, IREE_SV("media.image.decoded"), 0,
                                    decoded_shape, &decoded_image));

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_decode_configs(
      options->model, tiling_plan, &config_list));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(latent),
      id4_pipeline_program_write(decoded_tile),
      id4_pipeline_program_write(decoded_image),
  };

  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = IREE_SV("vae.decode.nearest");
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("vae/decode_nearest_f32"), IREE_SV("id4_vae_decode_nearest_f32"));
  dispatch_options.dispatch_config = id4_vae_program_make_dispatch_config(
      (uint32_t)tiling_plan.decoded_element_count);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_tap_options_t tap_options;
  memset(&tap_options, 0, sizeof(tap_options));
  tap_options.structure_size = sizeof(tap_options);
  tap_options.name = IREE_SV("vae.decode.tile");
  tap_options.tensor = decoded_tile;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_tap(builder, &tap_options));

  id4_pipeline_program_export_options_t export_options;
  memset(&export_options, 0, sizeof(export_options));
  export_options.structure_size = sizeof(export_options);
  export_options.name = IREE_SV("media.image.decoded");
  export_options.tensor = decoded_image;
  return id4_pipeline_program_export(builder, &export_options);
}

const id4_vae_model_config_t* id4_vae_program_flux2_model_config(void) {
  static const id4_vae_model_config_t model = {
      // Latent-to-image scale factor along the width axis.
      .scale_x = 16,
      // Latent-to-image scale factor along the height axis.
      .scale_y = 16,
      // Latent-to-media scale factor along the temporal axis.
      .scale_t = 1,
      // Flux2 latent tensors use 32 channels.
      .latent_channel_count = 32,
      // Flux2 VAE decode produces RGB images.
      .decoded_channel_count = 3,
      // Minimum latent tile width accepted by this implementation.
      .min_tile_size_x = 4,
      // Minimum latent tile height accepted by this implementation.
      .min_tile_size_y = 4,
      // stable-diffusion.cpp default latent tile width.
      .default_tile_size_x = 32,
      // stable-diffusion.cpp default latent tile height.
      .default_tile_size_y = 32,
      // stable-diffusion.cpp clamps spatial overlap to 0.5.
      .max_overlap = 0.5f,
      // Supported Flux2 VAE operations for the initial ID4 path.
      .capabilities =
          ID4_VAE_CAPABILITY_DECODE | ID4_VAE_CAPABILITY_SPATIAL_TILING,
  };
  return &model;
}
