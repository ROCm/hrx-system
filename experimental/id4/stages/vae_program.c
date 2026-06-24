// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/vae_program.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/stages/vae_parameters.h"
#include "iree/hal/command_buffer.h"

enum {
  ID4_VAE_DECODE_WORKGROUP_SIZE_X = 256,
  ID4_VAE_CONFIG_VALUE_BUFFER_CAPACITY = 24,
  ID4_VAE_DECODE_CONFIG_CAPACITY = 24,
  ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY = 128,
  ID4_VAE_FLUX2_DECODER_LEVEL_COUNT = 4,
  ID4_VAE_FLUX2_DECODER_RESNET_BLOCK_COUNT = 3,
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

typedef uint32_t id4_vae_program_group_norm_flags_t;
enum id4_vae_program_group_norm_flag_bits_e {
  ID4_VAE_PROGRAM_GROUP_NORM_FLAG_NONE = 0u,
  ID4_VAE_PROGRAM_GROUP_NORM_FLAG_APPLY_SILU = 1u << 0,
};

typedef enum id4_vae_program_conv3x3_weight_layout_e {
  // Original source layout: OCxICxKYxKX.
  ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_SOURCE = 0,
  // Packed consumer layout: ICxKYxKXxOC.
  ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC,
} id4_vae_program_conv3x3_weight_layout_t;

static const float id4_vae_flux2_latent_mean[128] = {
    -0.0676f, -0.0715f, -0.0753f, -0.0745f, 0.0223f,  0.0180f,  0.0142f,
    0.0184f,  -0.0001f, -0.0063f, -0.0002f, -0.0031f, -0.0272f, -0.0281f,
    -0.0276f, -0.0290f, -0.0769f, -0.0672f, -0.0902f, -0.0892f, 0.0168f,
    0.0152f,  0.0079f,  0.0086f,  0.0083f,  0.0015f,  0.0003f,  -0.0043f,
    -0.0439f, -0.0419f, -0.0438f, -0.0431f, -0.0102f, -0.0132f, -0.0066f,
    -0.0048f, -0.0311f, -0.0306f, -0.0279f, -0.0180f, 0.0030f,  0.0015f,
    0.0126f,  0.0145f,  0.0347f,  0.0338f,  0.0337f,  0.0283f,  0.0020f,
    0.0047f,  0.0047f,  0.0050f,  0.0123f,  0.0081f,  0.0081f,  0.0146f,
    0.0681f,  0.0679f,  0.0767f,  0.0732f,  -0.0462f, -0.0474f, -0.0392f,
    -0.0511f, -0.0528f, -0.0477f, -0.0470f, -0.0517f, -0.0317f, -0.0316f,
    -0.0345f, -0.0283f, 0.0510f,  0.0445f,  0.0578f,  0.0458f,  -0.0412f,
    -0.0458f, -0.0487f, -0.0467f, -0.0088f, -0.0106f, -0.0088f, -0.0046f,
    -0.0376f, -0.0432f, -0.0436f, -0.0499f, 0.0118f,  0.0166f,  0.0203f,
    0.0279f,  0.0113f,  0.0129f,  0.0016f,  0.0072f,  -0.0118f, -0.0018f,
    -0.0141f, -0.0054f, -0.0091f, -0.0138f, -0.0145f, -0.0187f, 0.0323f,
    0.0305f,  0.0259f,  0.0300f,  0.0540f,  0.0614f,  0.0495f,  0.0590f,
    -0.0511f, -0.0603f, -0.0478f, -0.0524f, -0.0227f, -0.0274f, -0.0154f,
    -0.0255f, -0.0572f, -0.0565f, -0.0518f, -0.0496f, 0.0116f,  0.0054f,
    0.0163f,  0.0104f,
};

static const float id4_vae_flux2_latent_std[128] = {
    1.8029f, 1.7786f, 1.7868f, 1.7837f, 1.7717f, 1.7590f, 1.7610f, 1.7479f,
    1.7336f, 1.7373f, 1.7340f, 1.7343f, 1.8626f, 1.8527f, 1.8629f, 1.8589f,
    1.7593f, 1.7526f, 1.7556f, 1.7583f, 1.7363f, 1.7400f, 1.7355f, 1.7394f,
    1.7342f, 1.7246f, 1.7392f, 1.7304f, 1.7551f, 1.7513f, 1.7559f, 1.7488f,
    1.8449f, 1.8454f, 1.8550f, 1.8535f, 1.8240f, 1.7813f, 1.7854f, 1.7945f,
    1.8047f, 1.7876f, 1.7695f, 1.7676f, 1.7782f, 1.7667f, 1.7925f, 1.7848f,
    1.7579f, 1.7407f, 1.7483f, 1.7368f, 1.7961f, 1.7998f, 1.7920f, 1.7925f,
    1.7780f, 1.7747f, 1.7727f, 1.7749f, 1.7526f, 1.7447f, 1.7657f, 1.7495f,
    1.7775f, 1.7720f, 1.7813f, 1.7813f, 1.8162f, 1.8013f, 1.8023f, 1.8033f,
    1.7527f, 1.7331f, 1.7563f, 1.7482f, 1.7610f, 1.7507f, 1.7681f, 1.7613f,
    1.7665f, 1.7545f, 1.7828f, 1.7726f, 1.7896f, 1.7999f, 1.7864f, 1.7760f,
    1.7613f, 1.7625f, 1.7560f, 1.7577f, 1.7783f, 1.7671f, 1.7810f, 1.7799f,
    1.7201f, 1.7068f, 1.7265f, 1.7091f, 1.7793f, 1.7578f, 1.7502f, 1.7455f,
    1.7587f, 1.7500f, 1.7525f, 1.7362f, 1.7616f, 1.7572f, 1.7444f, 1.7430f,
    1.7509f, 1.7610f, 1.7634f, 1.7612f, 1.7254f, 1.7135f, 1.7321f, 1.7226f,
    1.7664f, 1.7624f, 1.7718f, 1.7664f, 1.7457f, 1.7441f, 1.7569f, 1.7530f,
};

static const uint32_t id4_vae_flux2_decoder_up_output_channel_counts
    [ID4_VAE_FLUX2_DECODER_LEVEL_COUNT] = {
        128,
        256,
        512,
        512,
};

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

static iree_status_t id4_vae_program_format_suffix(
    iree_string_view_t prefix, iree_string_view_t suffix, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%.*s%.*s", (int)prefix.size,
                        prefix.data, (int)suffix.size, suffix.data);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format VAE program name");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_vae_program_format_up_block_prefix(
    iree_string_view_t root, uint32_t level, uint32_t block, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%.*s.%u.block.%u",
                        (int)root.size, root.data, level, block);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format VAE up block prefix");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_vae_program_format_up_upsample_prefix(
    iree_string_view_t root, uint32_t level, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%.*s.%u.upsample",
                        (int)root.size, root.data, level);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format VAE upsample prefix");
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
  switch (model.implementation) {
    case ID4_VAE_IMPLEMENTATION_DIRECT:
      return iree_ok_status();
    case ID4_VAE_IMPLEMENTATION_FLUX2:
      if (model.latent_channel_count != 128) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Flux2 VAE implementation requires 128 public latent channels");
      }
      if (model.scale_x % 2 != 0 || model.scale_y % 2 != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Flux2 VAE implementation requires even spatial scale factors");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VAE implementation %d is invalid",
                              (int)model.implementation);
  }
}

static iree_status_t id4_vae_program_validate_request_shape(
    id4_vae_model_config_t model, id4_pipeline_program_shape_t latent_shape) {
  if (latent_shape.rank != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE decode latent shape must be rank-4 WHCB, got rank %u",
        latent_shape.rank);
  }
  for (uint32_t i = 0; i < latent_shape.rank; ++i) {
    if (latent_shape.dims[i] == 0 || latent_shape.dims[i] > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VAE decode latent shape dimension %u is out of range", i);
    }
  }
  if (latent_shape.dims[2] != model.latent_channel_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE decode latent channel count %" PRIu64
                            " does not match model channel count %u",
                            latent_shape.dims[2], model.latent_channel_count);
  }
  return iree_ok_status();
}

static uint32_t id4_vae_program_ceil_div_u32(uint32_t dividend,
                                             uint32_t divisor) {
  return dividend / divisor + (dividend % divisor != 0 ? 1 : 0);
}

static uint64_t id4_vae_program_ceil_div_u64(uint64_t dividend,
                                             uint64_t divisor) {
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

static iree_status_t id4_vae_program_whcb_element_count(
    uint32_t width, uint32_t height, uint32_t channel_count,
    uint32_t batch_count, uint64_t* out_element_count) {
  uint64_t element_count = width;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_mul_u64(element_count, height, &element_count));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_mul_u64(element_count, channel_count, &element_count));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_mul_u64(element_count, batch_count, &element_count));
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
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      tile_size_x * model.scale_x, tile_size_y * model.scale_y,
      model.decoded_channel_count, batch_count, &tile_element_count));
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
  const uint32_t latent_width = (uint32_t)request.latent_shape.dims[0];
  const uint32_t latent_height = (uint32_t)request.latent_shape.dims[1];
  const uint32_t batch_count = (uint32_t)request.latent_shape.dims[3];
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

typedef struct id4_vae_program_axis_tiling_t {
  // Number of tiles covering this axis.
  uint32_t tile_count;
  // Integer overlap between adjacent tiles.
  uint32_t overlap_pixels;
  // Integer step between adjacent tile origins.
  uint32_t tile_step;
  // Fractional overlap resolved for this axis.
  float overlap;
} id4_vae_program_axis_tiling_t;

typedef struct id4_vae_program_tile_location_t {
  // Tile ordinal along the width axis.
  uint32_t ordinal_x;
  // Tile ordinal along the height axis.
  uint32_t ordinal_y;
  // Requested latent origin along the width axis before final-edge adjustment.
  uint32_t requested_origin_x;
  // Requested latent origin along the height axis before final-edge adjustment.
  uint32_t requested_origin_y;
  // Adjusted latent origin along width used for fixed-size tile extraction.
  uint32_t origin_x;
  // Adjusted latent origin along height used for fixed-size tile extraction.
  uint32_t origin_y;
  // Decoded tile pixels skipped along width after final-edge adjustment.
  uint32_t decoded_skip_x;
  // Decoded tile pixels skipped along height after final-edge adjustment.
  uint32_t decoded_skip_y;
} id4_vae_program_tile_location_t;

typedef struct id4_vae_program_tile_io_config_t {
  // Element count to clear in the target image tensor.
  uint64_t clear_element_count;
  // Source tensor width.
  uint32_t source_width;
  // Source tensor height.
  uint32_t source_height;
  // Source tensor channel count.
  uint32_t source_channel_count;
  // Batch count shared by source, tile, and image tensors.
  uint32_t batch_count;
  // Tile tensor width.
  uint32_t tile_width;
  // Tile tensor height.
  uint32_t tile_height;
  // Tile origin along the source or image width axis.
  uint32_t tile_origin_x;
  // Tile origin along the source or image height axis.
  uint32_t tile_origin_y;
  // Tile element count for extract or merge dispatches.
  uint64_t tile_element_count;
  // Image tensor width.
  uint32_t image_width;
  // Image tensor height.
  uint32_t image_height;
  // Image tensor channel count.
  uint32_t image_channel_count;
  // Tile pixels skipped along width during merge.
  uint32_t tile_skip_x;
  // Tile pixels skipped along height during merge.
  uint32_t tile_skip_y;
  // Overlap width in image pixels.
  uint32_t overlap_x;
  // Overlap height in image pixels.
  uint32_t overlap_y;
} id4_vae_program_tile_io_config_t;

typedef struct id4_vae_program_flux2_decoder_tail_config_t {
  // Program-visible name prefix for tensors, taps, barriers, and dispatches.
  iree_string_view_t program_prefix;
  // Decoder conv_in activation tensor consumed by the tail.
  id4_pipeline_program_tensor_t input;
  // Decoder conv_in spatial width.
  uint32_t input_width;
  // Decoder conv_in spatial height.
  uint32_t input_height;
  // Batch count.
  uint32_t batch_count;
  // Decoded output spatial width.
  uint32_t output_width;
  // Decoded output spatial height.
  uint32_t output_height;
  // Decoded output channel count.
  uint32_t output_channel_count;
  // Decoded output tensor written by the tail.
  id4_pipeline_program_tensor_t output;
} id4_vae_program_flux2_decoder_tail_config_t;

static iree_status_t id4_vae_program_resolve_axis_tiling(
    uint32_t latent_size, uint32_t tile_size, float requested_overlap,
    id4_vae_program_axis_tiling_t* out_axis_tiling) {
  IREE_ASSERT_ARGUMENT(out_axis_tiling);
  memset(out_axis_tiling, 0, sizeof(*out_axis_tiling));
  if (latent_size == 0 || tile_size == 0 || tile_size > latent_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE axis tiling requires 0 < tile size <= latent size");
  }
  if (tile_size >= latent_size) {
    out_axis_tiling->tile_count = 1;
    out_axis_tiling->overlap_pixels = 0;
    out_axis_tiling->tile_step = tile_size;
    out_axis_tiling->overlap = 0.0f;
    return iree_ok_status();
  }

  uint32_t overlap_pixels = (uint32_t)((float)tile_size * requested_overlap);
  if (overlap_pixels >= tile_size) overlap_pixels = tile_size - 1;
  const uint32_t non_overlap = tile_size - overlap_pixels;
  uint32_t tile_count = (latent_size - overlap_pixels) / non_overlap;
  const uint32_t half_tile_without_overlap = tile_size / 2u - overlap_pixels;
  const uint64_t overshoot_source =
      ((uint64_t)tile_count + 1u) * non_overlap + overlap_pixels;
  const uint32_t overshoot = (uint32_t)(overshoot_source % latent_size);
  if (overshoot != non_overlap &&
      overshoot <= tile_count * half_tile_without_overlap) {
    ++tile_count;
  }

  float overlap = 0.0f;
  if (tile_count <= 2) {
    tile_count = 2;
    overlap = (float)(2u * tile_size - latent_size) / (float)tile_size;
  } else {
    overlap = (float)(tile_size * tile_count - latent_size) /
              (float)(tile_size * (tile_count - 1u));
  }
  if (overlap < 0.0f) overlap = 0.0f;
  overlap_pixels = (uint32_t)((float)tile_size * overlap);
  if (overlap_pixels >= tile_size) overlap_pixels = tile_size - 1;

  out_axis_tiling->tile_count = tile_count;
  out_axis_tiling->overlap_pixels = overlap_pixels;
  out_axis_tiling->tile_step = tile_size - overlap_pixels;
  out_axis_tiling->overlap = overlap;
  return iree_ok_status();
}

static id4_vae_program_tile_location_t id4_vae_program_resolve_tile_location(
    const id4_vae_decode_tiling_plan_t* tiling_plan,
    id4_vae_model_config_t model, uint32_t tile_ordinal_x,
    uint32_t tile_ordinal_y) {
  id4_vae_program_tile_location_t location = {
      .ordinal_x = tile_ordinal_x,
      .ordinal_y = tile_ordinal_y,
      .requested_origin_x = tile_ordinal_x * tiling_plan->tile_step_x,
      .requested_origin_y = tile_ordinal_y * tiling_plan->tile_step_y,
      .origin_x = tile_ordinal_x * tiling_plan->tile_step_x,
      .origin_y = tile_ordinal_y * tiling_plan->tile_step_y,
      .decoded_skip_x = 0,
      .decoded_skip_y = 0,
  };
  if (location.origin_x + tiling_plan->tile_size_x >
      tiling_plan->latent_width) {
    location.origin_x = tiling_plan->latent_width - tiling_plan->tile_size_x;
    location.decoded_skip_x =
        (location.requested_origin_x - location.origin_x) * model.scale_x;
  }
  if (location.origin_y + tiling_plan->tile_size_y >
      tiling_plan->latent_height) {
    location.origin_y = tiling_plan->latent_height - tiling_plan->tile_size_y;
    location.decoded_skip_y =
        (location.requested_origin_y - location.origin_y) * model.scale_y;
  }
  return location;
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

static iree_status_t id4_vae_program_constant_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const float* values, iree_host_size_t value_count,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_constant_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.name = name;
  options.dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32;
  options.shape = id4_pipeline_program_make_shape_rank1(value_count);
  options.data =
      iree_make_const_byte_span(values, value_count * sizeof(values[0]));
  return id4_pipeline_program_constant(builder, &options, out_tensor);
}

static iree_status_t id4_vae_program_parameter_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t key,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_parameter_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.key = key;
  options.dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32;
  options.shape = shape;
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

static iree_status_t id4_vae_program_resolve_conv3x3_weight(
    iree_string_view_t source_key, uint32_t input_channel_count,
    uint32_t output_channel_count,
    id4_vae_program_conv3x3_weight_layout_t layout, char* key_storage,
    iree_host_size_t key_storage_capacity, iree_string_view_t* out_key,
    id4_pipeline_program_shape_t* out_shape) {
  *out_key = source_key;
  *out_shape = id4_pipeline_program_make_shape_rank4(output_channel_count,
                                                     input_channel_count, 3, 3);
  switch (layout) {
    case ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_SOURCE:
      return iree_ok_status();
    case ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC:
      *out_shape = id4_pipeline_program_make_shape_rank4(
          input_channel_count, 3, 3, output_channel_count);
      return id4_vae_parameter_format_packed_conv3x3_weight_key(
          source_key, key_storage, key_storage_capacity, out_key);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VAE conv3x3 weight layout %d is invalid",
                              (int)layout);
  }
}

static iree_status_t id4_vae_program_tap_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_tensor_t tensor) {
  id4_pipeline_program_tap_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.name = name;
  options.tensor = tensor;
  return id4_pipeline_program_tap(builder, &options);
}

static iree_status_t id4_vae_program_barrier(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name) {
  id4_pipeline_program_barrier_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.name = name;
  return id4_pipeline_program_barrier(builder, &options);
}

static iree_hal_dispatch_config_t id4_vae_program_make_static_dispatch_config(
    uint32_t workgroup_count_x, uint32_t workgroup_count_y,
    uint32_t workgroup_count_z) {
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(workgroup_count_x, workgroup_count_y,
                                           workgroup_count_z);
  dispatch_config.workgroup_size[0] = ID4_VAE_DECODE_WORKGROUP_SIZE_X;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

static iree_hal_dispatch_config_t id4_vae_program_make_dispatch_config(
    uint32_t element_count) {
  return id4_vae_program_make_static_dispatch_config(
      id4_vae_program_ceil_div_u32(element_count,
                                   ID4_VAE_DECODE_WORKGROUP_SIZE_X),
      1, 1);
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
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.decode.tile_height"),
      tiling_plan.tile_size_y);
}

static iree_status_t id4_vae_program_build_conv1x1_bias_configs(
    uint32_t width, uint32_t height, uint32_t input_channel_count,
    uint32_t output_channel_count, uint32_t batch_count,
    uint64_t output_element_count,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv1x1_bias.width"), width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv1x1_bias.height"), height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv1x1_bias.input_channel_count"),
      input_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv1x1_bias.output_channel_count"),
      output_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv1x1_bias.batch_count"),
      batch_count));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv1x1_bias.output_element_count"),
      output_element_count);
}

static iree_status_t id4_vae_program_add_conv1x1_bias_oc4_configs(
    uint32_t output_channel_count, uint64_t output_element_count,
    id4_vae_program_config_list_t* config_list) {
  const uint32_t output_channel_tile_count =
      id4_vae_program_ceil_div_u32(output_channel_count, 4);
  const uint64_t output_tile_element_count = (output_element_count + 3) / 4;
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      config_list, IREE_SV("id4.vae.conv1x1_bias.output_channel_tile_count"),
      output_channel_tile_count));
  return id4_vae_program_config_list_add_u64(
      config_list, IREE_SV("id4.vae.conv1x1_bias.output_tile_element_count"),
      output_tile_element_count);
}

static iree_status_t id4_vae_program_build_conv3x3_bias_configs(
    uint32_t width, uint32_t height, uint32_t input_channel_count,
    uint32_t output_channel_count, uint32_t batch_count,
    uint64_t output_element_count,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv3x3_bias.width"), width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv3x3_bias.height"), height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv3x3_bias.input_channel_count"),
      input_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv3x3_bias.output_channel_count"),
      output_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv3x3_bias.batch_count"),
      batch_count));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.conv3x3_bias.output_element_count"),
      output_element_count);
}

static iree_status_t id4_vae_program_add_conv3x3_bias_output_tile_configs(
    uint32_t output_channel_count, uint32_t output_channel_tile_width,
    uint64_t output_element_count, id4_vae_program_config_list_t* config_list) {
  const uint32_t output_channel_tile_count = id4_vae_program_ceil_div_u32(
      output_channel_count, output_channel_tile_width);
  const uint64_t output_tile_element_count = id4_vae_program_ceil_div_u64(
      output_element_count, output_channel_tile_width);
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      config_list, IREE_SV("id4.vae.conv3x3_bias.output_channel_tile_count"),
      output_channel_tile_count));
  return id4_vae_program_config_list_add_u64(
      config_list, IREE_SV("id4.vae.conv3x3_bias.output_tile_element_count"),
      output_tile_element_count);
}

static iree_status_t id4_vae_program_build_group_norm_base_configs(
    uint32_t width, uint32_t height, uint32_t channel_count,
    uint32_t group_count, uint32_t batch_count,
    id4_vae_program_config_list_t* out_config_list) {
  const uint32_t channels_per_group = channel_count / group_count;
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.group_norm.width"), width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.group_norm.height"), height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.group_norm.channel_count"),
      channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.group_norm.group_count"), group_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.group_norm.channels_per_group"),
      channels_per_group));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.group_norm.batch_count"), batch_count);
}

static iree_status_t id4_vae_program_build_group_norm_stats_configs(
    uint32_t width, uint32_t height, uint32_t channel_count,
    uint32_t group_count, uint32_t batch_count, uint64_t group_element_count,
    id4_vae_program_config_list_t* out_config_list) {
  IREE_RETURN_IF_ERROR(id4_vae_program_build_group_norm_base_configs(
      width, height, channel_count, group_count, batch_count, out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.group_norm.group_element_count"),
      group_element_count));
  return iree_ok_status();
}

static iree_status_t id4_vae_program_build_group_norm_apply_configs(
    uint32_t width, uint32_t height, uint32_t channel_count,
    uint32_t group_count, uint32_t batch_count, uint64_t output_element_count,
    id4_vae_program_config_list_t* out_config_list) {
  IREE_RETURN_IF_ERROR(id4_vae_program_build_group_norm_base_configs(
      width, height, channel_count, group_count, batch_count, out_config_list));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.group_norm.output_element_count"),
      output_element_count);
}

static iree_status_t id4_vae_program_add_group_norm_apply_tile_configs(
    uint32_t channel_count, uint64_t output_element_count,
    uint32_t output_channel_tile_width,
    id4_vae_program_config_list_t* config_list) {
  const uint32_t output_channel_tile_count =
      id4_vae_program_ceil_div_u32(channel_count, output_channel_tile_width);
  const uint64_t output_tile_element_count = id4_vae_program_ceil_div_u64(
      output_element_count, output_channel_tile_width);
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      config_list, IREE_SV("id4.vae.group_norm.output_channel_tile_count"),
      output_channel_tile_count));
  return id4_vae_program_config_list_add_u64(
      config_list, IREE_SV("id4.vae.group_norm.output_tile_element_count"),
      output_tile_element_count);
}

static iree_status_t id4_vae_program_build_elementwise_add_configs(
    uint64_t element_count, id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.elementwise.add.element_count"),
      element_count);
}

static iree_status_t id4_vae_program_build_spatial_attention_configs(
    uint32_t width, uint32_t height, uint32_t channel_count,
    uint32_t batch_count, uint64_t token_count,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.spatial_attention.width"), width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.spatial_attention.height"), height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.spatial_attention.channel_count"),
      channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.spatial_attention.batch_count"),
      batch_count));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.spatial_attention.token_count"),
      token_count);
}

static iree_status_t id4_vae_program_build_upsample_conv3x3_bias_configs(
    uint32_t input_width, uint32_t input_height, uint32_t channel_count,
    uint32_t batch_count, uint32_t output_width, uint32_t output_height,
    uint64_t output_element_count,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.upsample_conv3x3_bias.input_width"),
      input_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.upsample_conv3x3_bias.input_height"),
      input_height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.upsample_conv3x3_bias.channel_count"),
      channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.upsample_conv3x3_bias.batch_count"),
      batch_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.upsample_conv3x3_bias.output_width"),
      output_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.upsample_conv3x3_bias.output_height"),
      output_height));
  return id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.upsample_conv3x3_bias.output_element_count"),
      output_element_count);
}

static iree_status_t
id4_vae_program_add_upsample_conv3x3_bias_channel_tile_config(
    uint32_t channel_count, uint32_t output_channel_tile_width,
    id4_vae_program_config_list_t* config_list) {
  const uint32_t output_channel_tile_count =
      id4_vae_program_ceil_div_u32(channel_count, output_channel_tile_width);
  return id4_vae_program_config_list_add_u64(
      config_list,
      IREE_SV("id4.vae.upsample_conv3x3_bias.output_channel_tile_count"),
      output_channel_tile_count);
}

static iree_status_t id4_vae_program_build_flux2_affine_configs(
    id4_vae_decode_tiling_plan_t tiling_plan,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.flux2_affine_pixel_shuffle.public_width"),
      tiling_plan.latent_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.flux2_affine_pixel_shuffle.public_height"),
      tiling_plan.latent_height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.flux2_affine_pixel_shuffle.public_channel_count"),
      tiling_plan.latent_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.flux2_affine_pixel_shuffle.batch_count"),
      tiling_plan.batch_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.flux2_affine_pixel_shuffle.internal_width"),
      tiling_plan.latent_width * 2u));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.flux2_affine_pixel_shuffle.internal_height"),
      tiling_plan.latent_height * 2u));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.flux2_affine_pixel_shuffle.internal_channel_count"),
      tiling_plan.latent_channel_count / 4u));
  return id4_vae_program_config_list_add_u64(
      out_config_list,
      IREE_SV("id4.vae.flux2_affine_pixel_shuffle.output_element_count"),
      tiling_plan.latent_element_count);
}

static iree_status_t id4_vae_program_build_tile_clear_configs(
    id4_vae_program_tile_io_config_t config,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.clear_element_count"),
      config.clear_element_count);
}

static iree_status_t id4_vae_program_build_tile_extract_configs(
    id4_vae_program_tile_io_config_t config,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.source_width"),
      config.source_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.source_height"),
      config.source_height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.source_channel_count"),
      config.source_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.batch_count"),
      config.batch_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_width"), config.tile_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_height"),
      config.tile_height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_origin_x"),
      config.tile_origin_x));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_origin_y"),
      config.tile_origin_y));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_element_count"),
      config.tile_element_count);
}

static iree_status_t id4_vae_program_build_tile_merge_configs(
    id4_vae_program_tile_io_config_t config,
    id4_vae_program_config_list_t* out_config_list) {
  memset(out_config_list, 0, sizeof(*out_config_list));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.image_width"),
      config.image_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.image_height"),
      config.image_height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.image_channel_count"),
      config.image_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.batch_count"),
      config.batch_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_width"), config.tile_width));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_height"),
      config.tile_height));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_origin_x"),
      config.tile_origin_x));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_origin_y"),
      config.tile_origin_y));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_skip_x"),
      config.tile_skip_x));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_skip_y"),
      config.tile_skip_y));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.tile_element_count"),
      config.tile_element_count));
  IREE_RETURN_IF_ERROR(id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.overlap_x"), config.overlap_x));
  return id4_vae_program_config_list_add_u64(
      out_config_list, IREE_SV("id4.vae.tile.overlap_y"), config.overlap_y);
}

static iree_status_t id4_vae_program_author_conv1x1_bias(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t input, uint32_t width, uint32_t height,
    uint32_t input_channel_count, uint32_t output_channel_count,
    uint32_t batch_count, iree_string_view_t weight_key,
    iree_string_view_t bias_key, iree_string_view_t output_name,
    id4_pipeline_program_tensor_t* out_output) {
  uint64_t output_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      width, height, output_channel_count, batch_count, &output_element_count));
  if (output_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE conv1x1 output element count %" PRIu64 " exceeds max count %u",
        output_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, weight_key,
      id4_pipeline_program_make_shape_rank4(output_channel_count,
                                            input_channel_count, 1, 1),
      &weight));
  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, bias_key,
      id4_pipeline_program_make_shape_rank1(output_channel_count), &bias));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, output_name,
      id4_pipeline_program_make_shape_rank4(width, height, output_channel_count,
                                            batch_count),
      &output));

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_conv1x1_bias_configs(
      width, height, input_channel_count, output_channel_count, batch_count,
      output_element_count, &config_list));
  iree_string_view_t function_name = IREE_SV("id4_vae_conv1x1_bias_f32");
  uint32_t dispatch_element_count = (uint32_t)output_element_count;
  if (batch_count == 1 && input_channel_count >= 4 &&
      input_channel_count % 4 == 0 && output_channel_count >= 4 &&
      output_channel_count % 4 == 0) {
    function_name = IREE_SV("id4_vae_conv1x1_bias_ic4_oc4_f32");
    dispatch_element_count = (uint32_t)(output_element_count / 4);
    IREE_RETURN_IF_ERROR(id4_vae_program_add_conv1x1_bias_oc4_configs(
        output_channel_count, output_element_count, &config_list));
  }
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("vae/conv1x1_bias_f32"), function_name);
  dispatch_options.dispatch_config =
      id4_vae_program_make_dispatch_config(dispatch_element_count);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  *out_output = output;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_dispatch_conv3x3_bias(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t input, uint32_t width, uint32_t height,
    uint32_t input_channel_count, uint32_t output_channel_count,
    uint32_t batch_count, iree_string_view_t weight_key,
    iree_string_view_t bias_key, id4_pipeline_program_tensor_t output) {
  uint64_t output_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      width, height, output_channel_count, batch_count, &output_element_count));
  if (output_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE conv3x3 output element count %" PRIu64 " exceeds max count %u",
        output_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  iree_string_view_t module_path = IREE_SV("vae/conv3x3_bias_f32");
  iree_string_view_t function_name = IREE_SV("id4_vae_conv3x3_bias_f32");
  uint32_t dispatch_element_count = (uint32_t)output_element_count;
  uint32_t dispatch_workgroup_count_x = id4_vae_program_ceil_div_u32(
      dispatch_element_count, ID4_VAE_DECODE_WORKGROUP_SIZE_X);
  uint32_t dispatch_workgroup_count_y = 1;
  uint32_t output_channel_tile_width = 1;
  uint32_t output_tile_config_width = 0;
  id4_vae_program_conv3x3_weight_layout_t weight_layout =
      ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_SOURCE;
  if (batch_count == 1 && input_channel_count >= 4 &&
      input_channel_count % 4 == 0 && output_channel_count >= 64 &&
      output_channel_count % 64 == 0) {
    module_path = IREE_SV("vae/conv3x3_bias_packed_block_f32");
    function_name = IREE_SV("id4_vae_conv3x3_bias_ic4_oc64_packed_block_f32");
    output_channel_tile_width = 64;
    weight_layout = ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC;
    dispatch_element_count =
        (uint32_t)(output_element_count / output_channel_count);
    dispatch_workgroup_count_x =
        id4_vae_program_ceil_div_u32(dispatch_element_count, 32);
    dispatch_workgroup_count_y =
        output_channel_count / output_channel_tile_width;
  } else if (batch_count == 1 && input_channel_count >= 4 &&
             input_channel_count % 4 == 0 && output_channel_count >= 16 &&
             output_channel_count % 16 == 0) {
    module_path = IREE_SV("vae/conv3x3_bias_packed_f32");
    function_name = IREE_SV("id4_vae_conv3x3_bias_ic4_oc16_packed_2d_f32");
    output_channel_tile_width = 16;
    output_tile_config_width = output_channel_tile_width;
    weight_layout = ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC;
    dispatch_element_count =
        (uint32_t)(output_element_count / output_channel_count);
    dispatch_workgroup_count_x = id4_vae_program_ceil_div_u32(
        dispatch_element_count, ID4_VAE_DECODE_WORKGROUP_SIZE_X);
    dispatch_workgroup_count_y =
        output_channel_count / output_channel_tile_width;
  } else if (batch_count == 1 && input_channel_count >= 4 &&
             input_channel_count % 4 == 0 && output_channel_count >= 8 &&
             output_channel_count % 8 == 0) {
    function_name = IREE_SV("id4_vae_conv3x3_bias_ic4_oc8_packed_f32");
    output_channel_tile_width = 8;
    output_tile_config_width = output_channel_tile_width;
    weight_layout = ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC;
    dispatch_element_count =
        (uint32_t)(output_element_count / output_channel_tile_width);
    dispatch_workgroup_count_x = id4_vae_program_ceil_div_u32(
        dispatch_element_count, ID4_VAE_DECODE_WORKGROUP_SIZE_X);
  } else if (batch_count == 1 && input_channel_count >= 4 &&
             input_channel_count % 4 == 0 && output_channel_count >= 4 &&
             output_channel_count % 4 == 0) {
    function_name = IREE_SV("id4_vae_conv3x3_bias_ic4_oc4_f32");
    output_channel_tile_width = 4;
    output_tile_config_width = output_channel_tile_width;
    dispatch_element_count =
        (uint32_t)(output_element_count / output_channel_tile_width);
    dispatch_workgroup_count_x = id4_vae_program_ceil_div_u32(
        dispatch_element_count, ID4_VAE_DECODE_WORKGROUP_SIZE_X);
  } else if (batch_count == 1 && input_channel_count >= 4 &&
             input_channel_count % 4 == 0) {
    function_name = IREE_SV("id4_vae_conv3x3_bias_ic4_f32");
  }
  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_conv3x3_bias_configs(
      width, height, input_channel_count, output_channel_count, batch_count,
      output_element_count, &config_list));
  if (output_tile_config_width != 0) {
    IREE_RETURN_IF_ERROR(id4_vae_program_add_conv3x3_bias_output_tile_configs(
        output_channel_count, output_tile_config_width, output_element_count,
        &config_list));
  }

  iree_string_view_t resolved_weight_key = iree_string_view_empty();
  id4_pipeline_program_shape_t weight_shape;
  char packed_weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_resolve_conv3x3_weight(
      weight_key, input_channel_count, output_channel_count, weight_layout,
      packed_weight_key_storage, sizeof(packed_weight_key_storage),
      &resolved_weight_key, &weight_shape));
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, resolved_weight_key, weight_shape, &weight));
  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, bias_key,
      id4_pipeline_program_make_shape_rank1(output_channel_count), &bias));

  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  dispatch_options.kernel =
      id4_pipeline_make_kernel_ref(module_path, function_name);
  dispatch_options.dispatch_config =
      id4_vae_program_make_static_dispatch_config(
          dispatch_workgroup_count_x, dispatch_workgroup_count_y, 1);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  return id4_pipeline_program_dispatch_loom(builder, &dispatch_options);
}

static iree_status_t id4_vae_program_author_conv3x3_bias(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t input, uint32_t width, uint32_t height,
    uint32_t input_channel_count, uint32_t output_channel_count,
    uint32_t batch_count, iree_string_view_t weight_key,
    iree_string_view_t bias_key, iree_string_view_t output_name,
    id4_pipeline_program_tensor_t* out_output) {
  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, output_name,
      id4_pipeline_program_make_shape_rank4(width, height, output_channel_count,
                                            batch_count),
      &output));
  IREE_RETURN_IF_ERROR(id4_vae_program_dispatch_conv3x3_bias(
      builder, dispatch_name, input, width, height, input_channel_count,
      output_channel_count, batch_count, weight_key, bias_key, output));
  *out_output = output;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_author_conv3x3_bias_add(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t shortcut,
    uint32_t width, uint32_t height, uint32_t channel_count,
    uint32_t batch_count, iree_string_view_t weight_key,
    iree_string_view_t bias_key, iree_string_view_t output_name,
    id4_pipeline_program_tensor_t* out_output) {
  if (batch_count != 1 || channel_count < 4 || channel_count % 4 != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE fused conv3x3 residual requires batch-1 channel tiles");
  }

  uint64_t output_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      width, height, channel_count, batch_count, &output_element_count));
  if (output_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VAE fused conv3x3 output element count %" PRIu64
                            " exceeds max count %u",
                            output_element_count,
                            ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  iree_string_view_t module_path = IREE_SV("vae/conv3x3_bias_f32");
  iree_string_view_t function_name =
      IREE_SV("id4_vae_conv3x3_bias_add_ic4_oc4_f32");
  uint32_t output_channel_tile_width = 4;
  uint32_t output_tile_config_width = output_channel_tile_width;
  uint32_t dispatch_element_count =
      (uint32_t)(output_element_count / output_channel_tile_width);
  uint32_t dispatch_workgroup_count_x = id4_vae_program_ceil_div_u32(
      dispatch_element_count, ID4_VAE_DECODE_WORKGROUP_SIZE_X);
  uint32_t dispatch_workgroup_count_y = 1;
  id4_vae_program_conv3x3_weight_layout_t weight_layout =
      ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_SOURCE;
  if (channel_count >= 64 && channel_count % 64 == 0) {
    module_path = IREE_SV("vae/conv3x3_bias_packed_block_f32");
    function_name =
        IREE_SV("id4_vae_conv3x3_bias_add_ic4_oc64_packed_block_f32");
    output_channel_tile_width = 64;
    output_tile_config_width = 0;
    weight_layout = ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC;
    dispatch_element_count = (uint32_t)(output_element_count / channel_count);
    dispatch_workgroup_count_x =
        id4_vae_program_ceil_div_u32(dispatch_element_count, 32);
    dispatch_workgroup_count_y = channel_count / output_channel_tile_width;
  } else if (channel_count >= 16 && channel_count % 16 == 0) {
    module_path = IREE_SV("vae/conv3x3_bias_packed_f32");
    function_name = IREE_SV("id4_vae_conv3x3_bias_add_ic4_oc16_packed_2d_f32");
    output_channel_tile_width = 16;
    output_tile_config_width = output_channel_tile_width;
    weight_layout = ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC;
    dispatch_element_count = (uint32_t)(output_element_count / channel_count);
    dispatch_workgroup_count_x = id4_vae_program_ceil_div_u32(
        dispatch_element_count, ID4_VAE_DECODE_WORKGROUP_SIZE_X);
    dispatch_workgroup_count_y = channel_count / output_channel_tile_width;
  } else if (channel_count >= 8 && channel_count % 8 == 0) {
    function_name = IREE_SV("id4_vae_conv3x3_bias_add_ic4_oc8_packed_f32");
    output_channel_tile_width = 8;
    output_tile_config_width = output_channel_tile_width;
    weight_layout = ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC;
    dispatch_element_count =
        (uint32_t)(output_element_count / output_channel_tile_width);
    dispatch_workgroup_count_x = id4_vae_program_ceil_div_u32(
        dispatch_element_count, ID4_VAE_DECODE_WORKGROUP_SIZE_X);
  }

  iree_string_view_t resolved_weight_key = iree_string_view_empty();
  id4_pipeline_program_shape_t weight_shape;
  char packed_weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_resolve_conv3x3_weight(
      weight_key, channel_count, channel_count, weight_layout,
      packed_weight_key_storage, sizeof(packed_weight_key_storage),
      &resolved_weight_key, &weight_shape));
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, resolved_weight_key, weight_shape, &weight));
  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, bias_key, id4_pipeline_program_make_shape_rank1(channel_count),
      &bias));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, output_name,
      id4_pipeline_program_make_shape_rank4(width, height, channel_count,
                                            batch_count),
      &output));

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_conv3x3_bias_configs(
      width, height, channel_count, channel_count, batch_count,
      output_element_count, &config_list));
  if (output_tile_config_width != 0) {
    IREE_RETURN_IF_ERROR(id4_vae_program_add_conv3x3_bias_output_tile_configs(
        channel_count, output_tile_config_width, output_element_count,
        &config_list));
  }
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),   id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),    id4_pipeline_program_read(shortcut),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  dispatch_options.kernel =
      id4_pipeline_make_kernel_ref(module_path, function_name);
  dispatch_options.dispatch_config =
      id4_vae_program_make_static_dispatch_config(
          dispatch_workgroup_count_x, dispatch_workgroup_count_y, 1);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  *out_output = output;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_author_group_norm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t input, uint32_t width, uint32_t height,
    uint32_t channel_count, uint32_t group_count, uint32_t batch_count,
    iree_string_view_t weight_key, iree_string_view_t bias_key,
    iree_string_view_t stats_name, iree_string_view_t stats_barrier_name,
    iree_string_view_t output_name, id4_vae_program_group_norm_flags_t flags,
    id4_pipeline_program_tensor_t* out_output) {
  if (group_count == 0 || channel_count % group_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE group norm channel count %u is not divisible by group count %u",
        channel_count, group_count);
  }

  uint64_t output_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      width, height, channel_count, batch_count, &output_element_count));
  if (output_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE group norm output element count %" PRIu64 " exceeds max count %u",
        output_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  uint64_t spatial_element_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_mul_u64(width, height, &spatial_element_count));
  uint64_t group_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_mul_u64(spatial_element_count,
                                               channel_count / group_count,
                                               &group_element_count));

  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, weight_key, id4_pipeline_program_make_shape_rank1(channel_count),
      &weight));
  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, bias_key, id4_pipeline_program_make_shape_rank1(channel_count),
      &bias));

  id4_pipeline_program_tensor_t stats = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, stats_name,
      id4_pipeline_program_make_shape_rank3(batch_count, group_count, 2),
      &stats));
  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, output_name,
      id4_pipeline_program_make_shape_rank4(width, height, channel_count,
                                            batch_count),
      &output));

  id4_pipeline_program_dispatch_binding_t stats_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(stats),
  };
  id4_vae_program_config_list_t stats_config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_group_norm_stats_configs(
      width, height, channel_count, group_count, batch_count,
      group_element_count, &stats_config_list));
  id4_pipeline_program_dispatch_loom_options_t stats_dispatch_options;
  memset(&stats_dispatch_options, 0, sizeof(stats_dispatch_options));
  stats_dispatch_options.structure_size = sizeof(stats_dispatch_options);
  stats_dispatch_options.name = dispatch_name;
  stats_dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("vae/group_norm_f32"), IREE_SV("id4_vae_group_norm_stats_f32"));
  stats_dispatch_options.dispatch_config =
      id4_vae_program_make_static_dispatch_config(group_count, batch_count, 1);
  stats_dispatch_options.config_binding_count = stats_config_list.count;
  stats_dispatch_options.config_bindings = stats_config_list.bindings;
  stats_dispatch_options.binding_count = IREE_ARRAYSIZE(stats_bindings);
  stats_dispatch_options.bindings = stats_bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &stats_dispatch_options));
  IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, stats_barrier_name));

  id4_pipeline_program_dispatch_binding_t apply_bindings[] = {
      id4_pipeline_program_read(input),   id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),    id4_pipeline_program_read(stats),
      id4_pipeline_program_write(output),
  };
  iree_string_view_t apply_module_path = IREE_SV("vae/group_norm_f32");
  iree_string_view_t apply_function_name =
      iree_any_bit_set(flags, ID4_VAE_PROGRAM_GROUP_NORM_FLAG_APPLY_SILU)
          ? IREE_SV("id4_vae_group_norm_silu_f32")
          : IREE_SV("id4_vae_group_norm_f32");
  uint32_t apply_dispatch_element_count = (uint32_t)output_element_count;
  uint32_t apply_dispatch_workgroup_count_y = 1;
  uint32_t apply_output_channel_tile_width = 1;
  id4_vae_program_config_list_t apply_config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_group_norm_apply_configs(
      width, height, channel_count, group_count, batch_count,
      output_element_count, &apply_config_list));
  const uint32_t channels_per_group = channel_count / group_count;
  if (iree_any_bit_set(flags, ID4_VAE_PROGRAM_GROUP_NORM_FLAG_APPLY_SILU) &&
      batch_count == 1 && channel_count >= 4 && channel_count % 4 == 0 &&
      channels_per_group % 4 == 0) {
    apply_function_name = IREE_SV("id4_vae_group_norm_silu_ic4_oc4_2d_f32");
    apply_output_channel_tile_width = 4;
    if (channels_per_group >= 16 && channels_per_group % 16 == 0) {
      apply_module_path = IREE_SV("vae/group_norm_oc16_f32");
      apply_function_name = IREE_SV("id4_vae_group_norm_silu_ic4_oc16_2d_f32");
      apply_output_channel_tile_width = 16;
    } else if (channels_per_group >= 8 && channels_per_group % 8 == 0) {
      apply_module_path = IREE_SV("vae/group_norm_oc8_f32");
      apply_function_name = IREE_SV("id4_vae_group_norm_silu_ic4_oc8_2d_f32");
      apply_output_channel_tile_width = 8;
    }
    apply_dispatch_workgroup_count_y =
        channel_count / apply_output_channel_tile_width;
    apply_dispatch_element_count =
        (uint32_t)(output_element_count / channel_count);
    IREE_RETURN_IF_ERROR(id4_vae_program_add_group_norm_apply_tile_configs(
        channel_count, output_element_count, apply_output_channel_tile_width,
        &apply_config_list));
  }
  id4_pipeline_program_dispatch_loom_options_t apply_dispatch_options;
  memset(&apply_dispatch_options, 0, sizeof(apply_dispatch_options));
  apply_dispatch_options.structure_size = sizeof(apply_dispatch_options);
  apply_dispatch_options.name = output_name;
  apply_dispatch_options.kernel =
      id4_pipeline_make_kernel_ref(apply_module_path, apply_function_name);
  apply_dispatch_options.dispatch_config =
      id4_vae_program_make_static_dispatch_config(
          id4_vae_program_ceil_div_u32(apply_dispatch_element_count,
                                       ID4_VAE_DECODE_WORKGROUP_SIZE_X),
          apply_dispatch_workgroup_count_y, 1);
  apply_dispatch_options.config_binding_count = apply_config_list.count;
  apply_dispatch_options.config_bindings = apply_config_list.bindings;
  apply_dispatch_options.binding_count = IREE_ARRAYSIZE(apply_bindings);
  apply_dispatch_options.bindings = apply_bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &apply_dispatch_options));

  *out_output = output;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_author_elementwise_add(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t lhs, id4_pipeline_program_tensor_t rhs,
    uint32_t width, uint32_t height, uint32_t channel_count,
    uint32_t batch_count, iree_string_view_t output_name,
    id4_pipeline_program_tensor_t* out_output) {
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      width, height, channel_count, batch_count, &element_count));
  if (element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VAE elementwise add element count %" PRIu64
                            " exceeds max count %u",
                            element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, output_name,
      id4_pipeline_program_make_shape_rank4(width, height, channel_count,
                                            batch_count),
      &output));

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_elementwise_add_configs(
      element_count, &config_list));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(lhs),
      id4_pipeline_program_read(rhs),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("elementwise/add_f32"), IREE_SV("id4_elementwise_add_f32"));
  dispatch_options.dispatch_config =
      id4_vae_program_make_dispatch_config((uint32_t)element_count);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  *out_output = output;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_author_spatial_attention(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, uint32_t width, uint32_t height,
    uint32_t channel_count, uint32_t batch_count,
    iree_string_view_t output_name, id4_pipeline_program_tensor_t* out_output) {
  uint64_t token_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_mul_u64(width, height, &token_count));
  if (token_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VAE spatial attention token count %" PRIu64
                            " exceeds max count %u",
                            token_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, output_name,
      id4_pipeline_program_make_shape_rank4(width, height, channel_count,
                                            batch_count),
      &output));

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_spatial_attention_configs(
      width, height, channel_count, batch_count, token_count, &config_list));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(value),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  iree_string_view_t module_path = IREE_SV("vae/spatial_attention_f32");
  iree_string_view_t function_name = IREE_SV("id4_vae_spatial_attention_f32");
  if (batch_count == 1 && channel_count >= 2 && channel_count <= 512 &&
      channel_count % 2 == 0) {
    module_path = IREE_SV("vae/spatial_attention_vec2_f32");
    function_name = IREE_SV("id4_vae_spatial_attention_vec2_f32");
  }
  dispatch_options.kernel =
      id4_pipeline_make_kernel_ref(module_path, function_name);
  dispatch_options.dispatch_config =
      id4_vae_program_make_static_dispatch_config((uint32_t)token_count,
                                                  batch_count, 1);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  *out_output = output;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_author_resnet_block(
    id4_pipeline_program_builder_t* builder,
    iree_string_view_t program_name_prefix,
    iree_string_view_t parameter_key_prefix,
    id4_pipeline_program_tensor_t input, uint32_t width, uint32_t height,
    uint32_t input_channel_count, uint32_t output_channel_count,
    uint32_t batch_count, id4_pipeline_program_tensor_t* out_output) {
  iree_string_view_t norm1_weight_key = iree_string_view_empty();
  char norm1_weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      parameter_key_prefix, IREE_SV(".norm1.weight"), norm1_weight_key_storage,
      sizeof(norm1_weight_key_storage), &norm1_weight_key));
  iree_string_view_t norm1_bias_key = iree_string_view_empty();
  char norm1_bias_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      parameter_key_prefix, IREE_SV(".norm1.bias"), norm1_bias_key_storage,
      sizeof(norm1_bias_key_storage), &norm1_bias_key));
  iree_string_view_t norm1_stats_name = iree_string_view_empty();
  char norm1_stats_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".norm1.stats"), norm1_stats_name_storage,
      sizeof(norm1_stats_name_storage), &norm1_stats_name));
  iree_string_view_t norm1_stats_barrier_name = iree_string_view_empty();
  char norm1_stats_barrier_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".after_norm1_stats"),
      norm1_stats_barrier_name_storage,
      sizeof(norm1_stats_barrier_name_storage), &norm1_stats_barrier_name));
  iree_string_view_t norm1_silu_name = iree_string_view_empty();
  char norm1_silu_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".norm1_silu"), norm1_silu_name_storage,
      sizeof(norm1_silu_name_storage), &norm1_silu_name));

  id4_pipeline_program_tensor_t norm1_silu =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_group_norm(
      builder, norm1_stats_name, input, width, height, input_channel_count, 32,
      batch_count, norm1_weight_key, norm1_bias_key, norm1_stats_name,
      norm1_stats_barrier_name, norm1_silu_name,
      ID4_VAE_PROGRAM_GROUP_NORM_FLAG_APPLY_SILU, &norm1_silu));
  iree_string_view_t after_norm1_silu_name = iree_string_view_empty();
  char after_norm1_silu_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".after_norm1_silu"),
      after_norm1_silu_name_storage, sizeof(after_norm1_silu_name_storage),
      &after_norm1_silu_name));
  IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, after_norm1_silu_name));

  iree_string_view_t conv1_weight_key = iree_string_view_empty();
  char conv1_weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      parameter_key_prefix, IREE_SV(".conv1.weight"), conv1_weight_key_storage,
      sizeof(conv1_weight_key_storage), &conv1_weight_key));
  iree_string_view_t conv1_bias_key = iree_string_view_empty();
  char conv1_bias_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      parameter_key_prefix, IREE_SV(".conv1.bias"), conv1_bias_key_storage,
      sizeof(conv1_bias_key_storage), &conv1_bias_key));
  iree_string_view_t conv1_name = iree_string_view_empty();
  char conv1_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".conv1"), conv1_name_storage,
      sizeof(conv1_name_storage), &conv1_name));

  id4_pipeline_program_tensor_t conv1 = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_conv3x3_bias(
      builder, conv1_name, norm1_silu, width, height, input_channel_count,
      output_channel_count, batch_count, conv1_weight_key, conv1_bias_key,
      conv1_name, &conv1));
  iree_string_view_t after_conv1_name = iree_string_view_empty();
  char after_conv1_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".after_conv1"), after_conv1_name_storage,
      sizeof(after_conv1_name_storage), &after_conv1_name));
  IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, after_conv1_name));

  iree_string_view_t norm2_weight_key = iree_string_view_empty();
  char norm2_weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      parameter_key_prefix, IREE_SV(".norm2.weight"), norm2_weight_key_storage,
      sizeof(norm2_weight_key_storage), &norm2_weight_key));
  iree_string_view_t norm2_bias_key = iree_string_view_empty();
  char norm2_bias_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      parameter_key_prefix, IREE_SV(".norm2.bias"), norm2_bias_key_storage,
      sizeof(norm2_bias_key_storage), &norm2_bias_key));
  iree_string_view_t norm2_stats_name = iree_string_view_empty();
  char norm2_stats_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".norm2.stats"), norm2_stats_name_storage,
      sizeof(norm2_stats_name_storage), &norm2_stats_name));
  iree_string_view_t norm2_stats_barrier_name = iree_string_view_empty();
  char norm2_stats_barrier_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".after_norm2_stats"),
      norm2_stats_barrier_name_storage,
      sizeof(norm2_stats_barrier_name_storage), &norm2_stats_barrier_name));
  iree_string_view_t norm2_silu_name = iree_string_view_empty();
  char norm2_silu_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".norm2_silu"), norm2_silu_name_storage,
      sizeof(norm2_silu_name_storage), &norm2_silu_name));

  id4_pipeline_program_tensor_t norm2_silu =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_group_norm(
      builder, norm2_stats_name, conv1, width, height, output_channel_count, 32,
      batch_count, norm2_weight_key, norm2_bias_key, norm2_stats_name,
      norm2_stats_barrier_name, norm2_silu_name,
      ID4_VAE_PROGRAM_GROUP_NORM_FLAG_APPLY_SILU, &norm2_silu));
  iree_string_view_t after_norm2_silu_name = iree_string_view_empty();
  char after_norm2_silu_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".after_norm2_silu"),
      after_norm2_silu_name_storage, sizeof(after_norm2_silu_name_storage),
      &after_norm2_silu_name));
  IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, after_norm2_silu_name));

  iree_string_view_t conv2_weight_key = iree_string_view_empty();
  char conv2_weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      parameter_key_prefix, IREE_SV(".conv2.weight"), conv2_weight_key_storage,
      sizeof(conv2_weight_key_storage), &conv2_weight_key));
  iree_string_view_t conv2_bias_key = iree_string_view_empty();
  char conv2_bias_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      parameter_key_prefix, IREE_SV(".conv2.bias"), conv2_bias_key_storage,
      sizeof(conv2_bias_key_storage), &conv2_bias_key));
  iree_string_view_t conv2_name = iree_string_view_empty();
  char conv2_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".conv2"), conv2_name_storage,
      sizeof(conv2_name_storage), &conv2_name));
  iree_string_view_t output_name = iree_string_view_empty();
  char output_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".output"), output_name_storage,
      sizeof(output_name_storage), &output_name));

  if (input_channel_count == output_channel_count && batch_count == 1 &&
      output_channel_count >= 4 && output_channel_count % 4 == 0) {
    id4_pipeline_program_tensor_t output =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_vae_program_author_conv3x3_bias_add(
        builder, conv2_name, norm2_silu, input, width, height,
        output_channel_count, batch_count, conv2_weight_key, conv2_bias_key,
        output_name, &output));
    iree_string_view_t output_barrier_name = iree_string_view_empty();
    char output_barrier_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
    IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
        program_name_prefix, IREE_SV(".after_output"),
        output_barrier_name_storage, sizeof(output_barrier_name_storage),
        &output_barrier_name));
    IREE_RETURN_IF_ERROR(
        id4_vae_program_tap_tensor(builder, output_name, output));
    IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, output_barrier_name));
    *out_output = output;
    return iree_ok_status();
  }

  id4_pipeline_program_tensor_t conv2 = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_conv3x3_bias(
      builder, conv2_name, norm2_silu, width, height, output_channel_count,
      output_channel_count, batch_count, conv2_weight_key, conv2_bias_key,
      conv2_name, &conv2));
  iree_string_view_t after_conv2_name = iree_string_view_empty();
  char after_conv2_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".after_conv2"), after_conv2_name_storage,
      sizeof(after_conv2_name_storage), &after_conv2_name));
  IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, after_conv2_name));

  id4_pipeline_program_tensor_t shortcut = input;
  if (input_channel_count != output_channel_count) {
    iree_string_view_t shortcut_weight_key = iree_string_view_empty();
    char shortcut_weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
    IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
        parameter_key_prefix, IREE_SV(".nin_shortcut.weight"),
        shortcut_weight_key_storage, sizeof(shortcut_weight_key_storage),
        &shortcut_weight_key));
    iree_string_view_t shortcut_bias_key = iree_string_view_empty();
    char shortcut_bias_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
    IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
        parameter_key_prefix, IREE_SV(".nin_shortcut.bias"),
        shortcut_bias_key_storage, sizeof(shortcut_bias_key_storage),
        &shortcut_bias_key));
    iree_string_view_t shortcut_name = iree_string_view_empty();
    char shortcut_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
    IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
        program_name_prefix, IREE_SV(".nin_shortcut"), shortcut_name_storage,
        sizeof(shortcut_name_storage), &shortcut_name));
    IREE_RETURN_IF_ERROR(id4_vae_program_author_conv1x1_bias(
        builder, shortcut_name, input, width, height, input_channel_count,
        output_channel_count, batch_count, shortcut_weight_key,
        shortcut_bias_key, shortcut_name, &shortcut));
    iree_string_view_t after_shortcut_name = iree_string_view_empty();
    char after_shortcut_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
    IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
        program_name_prefix, IREE_SV(".after_nin_shortcut"),
        after_shortcut_name_storage, sizeof(after_shortcut_name_storage),
        &after_shortcut_name));
    IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, after_shortcut_name));
  }

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_elementwise_add(
      builder, output_name, shortcut, conv2, width, height,
      output_channel_count, batch_count, output_name, &output));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_tap_tensor(builder, output_name, output));
  iree_string_view_t output_barrier_name = iree_string_view_empty();
  char output_barrier_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      program_name_prefix, IREE_SV(".after_output"),
      output_barrier_name_storage, sizeof(output_barrier_name_storage),
      &output_barrier_name));
  IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, output_barrier_name));

  *out_output = output;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_author_upsample_conv3x3_bias(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t input, uint32_t input_width,
    uint32_t input_height, uint32_t channel_count, uint32_t batch_count,
    iree_string_view_t weight_key, iree_string_view_t bias_key,
    iree_string_view_t output_name, id4_pipeline_program_tensor_t* out_output) {
  const uint32_t output_width = input_width * 2u;
  const uint32_t output_height = input_height * 2u;
  uint64_t output_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      output_width, output_height, channel_count, batch_count,
      &output_element_count));
  if (output_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE upsample output element count %" PRIu64 " exceeds max count %u",
        output_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, output_name,
      id4_pipeline_program_make_shape_rank4(output_width, output_height,
                                            channel_count, batch_count),
      &output));

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_upsample_conv3x3_bias_configs(
      input_width, input_height, channel_count, batch_count, output_width,
      output_height, output_element_count, &config_list));
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  iree_string_view_t function_name =
      IREE_SV("id4_vae_upsample_conv3x3_bias_f32");
  uint32_t output_channel_tile_width = 1;
  uint32_t dispatch_element_count = (uint32_t)output_element_count;
  uint32_t dispatch_workgroup_count_y = 1;
  id4_vae_program_conv3x3_weight_layout_t weight_layout =
      ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_SOURCE;
  if (batch_count == 1 && channel_count >= 16 && channel_count % 16 == 0) {
    function_name =
        IREE_SV("id4_vae_upsample_conv3x3_bias_ic4_oc16_packed_2d_f32");
    output_channel_tile_width = 16;
    weight_layout = ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC;
    dispatch_element_count = (uint32_t)(output_element_count / channel_count);
    dispatch_workgroup_count_y = channel_count / output_channel_tile_width;
    IREE_RETURN_IF_ERROR(
        id4_vae_program_add_upsample_conv3x3_bias_channel_tile_config(
            channel_count, output_channel_tile_width, &config_list));
  } else if (batch_count == 1 && channel_count >= 8 && channel_count % 8 == 0) {
    function_name = IREE_SV("id4_vae_upsample_conv3x3_bias_ic4_oc8_packed_f32");
    output_channel_tile_width = 8;
    weight_layout = ID4_VAE_PROGRAM_CONV3X3_WEIGHT_LAYOUT_PACKED_IC_KY_KX_OC;
    dispatch_element_count =
        (uint32_t)(output_element_count / output_channel_tile_width);
    IREE_RETURN_IF_ERROR(
        id4_vae_program_add_upsample_conv3x3_bias_channel_tile_config(
            channel_count, output_channel_tile_width, &config_list));
  } else if (batch_count == 1 && channel_count >= 4 && channel_count % 4 == 0) {
    function_name = IREE_SV("id4_vae_upsample_conv3x3_bias_ic4_oc4_f32");
    output_channel_tile_width = 4;
    dispatch_element_count =
        (uint32_t)(output_element_count / output_channel_tile_width);
    IREE_RETURN_IF_ERROR(
        id4_vae_program_add_upsample_conv3x3_bias_channel_tile_config(
            channel_count, output_channel_tile_width, &config_list));
  }

  iree_string_view_t resolved_weight_key = iree_string_view_empty();
  id4_pipeline_program_shape_t weight_shape;
  char packed_weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_resolve_conv3x3_weight(
      weight_key, channel_count, channel_count, weight_layout,
      packed_weight_key_storage, sizeof(packed_weight_key_storage),
      &resolved_weight_key, &weight_shape));
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, resolved_weight_key, weight_shape, &weight));
  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_parameter_tensor(
      builder, bias_key, id4_pipeline_program_make_shape_rank1(channel_count),
      &bias));

  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_write(output),
  };
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("vae/upsample_conv3x3_bias_f32"), function_name);
  dispatch_options.dispatch_config =
      id4_vae_program_make_static_dispatch_config(
          id4_vae_program_ceil_div_u32(dispatch_element_count,
                                       ID4_VAE_DECODE_WORKGROUP_SIZE_X),
          dispatch_workgroup_count_y, 1);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  *out_output = output;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_author_tile_clear(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_vae_program_tile_io_config_t config,
    id4_pipeline_program_tensor_t target) {
  if (config.clear_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE tile clear element count %" PRIu64 " exceeds max count %u",
        config.clear_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_build_tile_clear_configs(config, &config_list));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_write(target),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("vae/tile_io_f32"), IREE_SV("id4_vae_tile_clear_f32"));
  dispatch_options.dispatch_config = id4_vae_program_make_dispatch_config(
      (uint32_t)config.clear_element_count);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  return id4_pipeline_program_dispatch_loom(builder, &dispatch_options);
}

static iree_status_t id4_vae_program_author_tile_extract(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t source,
    id4_vae_program_tile_io_config_t config, iree_string_view_t tile_name,
    id4_pipeline_program_tensor_t* out_tile) {
  if (config.tile_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE tile extract element count %" PRIu64 " exceeds max count %u",
        config.tile_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  id4_pipeline_program_tensor_t tile = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, tile_name,
      id4_pipeline_program_make_shape_rank4(
          config.tile_width, config.tile_height, config.source_channel_count,
          config.batch_count),
      &tile));

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_build_tile_extract_configs(config, &config_list));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(source),
      id4_pipeline_program_write(tile),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("vae/tile_io_f32"), IREE_SV("id4_vae_tile_extract_f32"));
  dispatch_options.dispatch_config =
      id4_vae_program_make_dispatch_config((uint32_t)config.tile_element_count);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  *out_tile = tile;
  return iree_ok_status();
}

static iree_status_t id4_vae_program_author_tile_merge(
    id4_pipeline_program_builder_t* builder, iree_string_view_t dispatch_name,
    id4_pipeline_program_tensor_t tile, id4_vae_program_tile_io_config_t config,
    id4_pipeline_program_tensor_t image) {
  if (config.tile_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE tile merge element count %" PRIu64 " exceeds max count %u",
        config.tile_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_build_tile_merge_configs(config, &config_list));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(tile),
      id4_pipeline_program_read_write(image),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = dispatch_name;
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("vae/tile_io_f32"), IREE_SV("id4_vae_tile_merge_f32"));
  dispatch_options.dispatch_config =
      id4_vae_program_make_dispatch_config((uint32_t)config.tile_element_count);
  dispatch_options.config_binding_count = config_list.count;
  dispatch_options.config_bindings = config_list.bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  return id4_pipeline_program_dispatch_loom(builder, &dispatch_options);
}

static iree_status_t id4_vae_program_author_flux2_decoder_tail(
    id4_pipeline_program_builder_t* builder,
    id4_vae_program_flux2_decoder_tail_config_t config) {
  iree_string_view_t mid_block_1_norm1_stats_name = iree_string_view_empty();
  char mid_block_1_norm1_stats_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.block_1.norm1.stats"),
      mid_block_1_norm1_stats_name_storage,
      sizeof(mid_block_1_norm1_stats_name_storage),
      &mid_block_1_norm1_stats_name));
  iree_string_view_t after_mid_block_1_norm1_stats_name =
      iree_string_view_empty();
  char after_mid_block_1_norm1_stats_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_block_1_norm1_stats"),
      after_mid_block_1_norm1_stats_name_storage,
      sizeof(after_mid_block_1_norm1_stats_name_storage),
      &after_mid_block_1_norm1_stats_name));
  iree_string_view_t mid_block_1_norm1_silu_name = iree_string_view_empty();
  char
      mid_block_1_norm1_silu_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.block_1.norm1_silu"),
      mid_block_1_norm1_silu_name_storage,
      sizeof(mid_block_1_norm1_silu_name_storage),
      &mid_block_1_norm1_silu_name));
  id4_pipeline_program_tensor_t mid_block_1_norm1_silu =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_group_norm(
      builder, mid_block_1_norm1_stats_name, config.input, config.input_width,
      config.input_height, 512, 32, config.batch_count,
      IREE_SV("decoder.mid.block_1.norm1.weight"),
      IREE_SV("decoder.mid.block_1.norm1.bias"), mid_block_1_norm1_stats_name,
      after_mid_block_1_norm1_stats_name, mid_block_1_norm1_silu_name,
      ID4_VAE_PROGRAM_GROUP_NORM_FLAG_APPLY_SILU, &mid_block_1_norm1_silu));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, mid_block_1_norm1_silu_name, mid_block_1_norm1_silu));
  iree_string_view_t after_mid_block_1_norm1_silu_name =
      iree_string_view_empty();
  char after_mid_block_1_norm1_silu_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_block_1_norm1_silu"),
      after_mid_block_1_norm1_silu_name_storage,
      sizeof(after_mid_block_1_norm1_silu_name_storage),
      &after_mid_block_1_norm1_silu_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_block_1_norm1_silu_name));

  iree_string_view_t mid_block_1_conv1_name = iree_string_view_empty();
  char mid_block_1_conv1_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.block_1.conv1"),
      mid_block_1_conv1_name_storage, sizeof(mid_block_1_conv1_name_storage),
      &mid_block_1_conv1_name));
  id4_pipeline_program_tensor_t mid_block_1_conv1 =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_conv3x3_bias(
      builder, mid_block_1_conv1_name, mid_block_1_norm1_silu,
      config.input_width, config.input_height, 512, 512, config.batch_count,
      IREE_SV("decoder.mid.block_1.conv1.weight"),
      IREE_SV("decoder.mid.block_1.conv1.bias"), mid_block_1_conv1_name,
      &mid_block_1_conv1));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, mid_block_1_conv1_name, mid_block_1_conv1));
  iree_string_view_t after_mid_block_1_conv1_name = iree_string_view_empty();
  char after_mid_block_1_conv1_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_block_1_conv1"),
      after_mid_block_1_conv1_name_storage,
      sizeof(after_mid_block_1_conv1_name_storage),
      &after_mid_block_1_conv1_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_block_1_conv1_name));

  iree_string_view_t mid_block_1_norm2_stats_name = iree_string_view_empty();
  char mid_block_1_norm2_stats_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.block_1.norm2.stats"),
      mid_block_1_norm2_stats_name_storage,
      sizeof(mid_block_1_norm2_stats_name_storage),
      &mid_block_1_norm2_stats_name));
  iree_string_view_t after_mid_block_1_norm2_stats_name =
      iree_string_view_empty();
  char after_mid_block_1_norm2_stats_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_block_1_norm2_stats"),
      after_mid_block_1_norm2_stats_name_storage,
      sizeof(after_mid_block_1_norm2_stats_name_storage),
      &after_mid_block_1_norm2_stats_name));
  iree_string_view_t mid_block_1_norm2_silu_name = iree_string_view_empty();
  char
      mid_block_1_norm2_silu_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.block_1.norm2_silu"),
      mid_block_1_norm2_silu_name_storage,
      sizeof(mid_block_1_norm2_silu_name_storage),
      &mid_block_1_norm2_silu_name));
  id4_pipeline_program_tensor_t mid_block_1_norm2_silu =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_group_norm(
      builder, mid_block_1_norm2_stats_name, mid_block_1_conv1,
      config.input_width, config.input_height, 512, 32, config.batch_count,
      IREE_SV("decoder.mid.block_1.norm2.weight"),
      IREE_SV("decoder.mid.block_1.norm2.bias"), mid_block_1_norm2_stats_name,
      after_mid_block_1_norm2_stats_name, mid_block_1_norm2_silu_name,
      ID4_VAE_PROGRAM_GROUP_NORM_FLAG_APPLY_SILU, &mid_block_1_norm2_silu));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, mid_block_1_norm2_silu_name, mid_block_1_norm2_silu));
  iree_string_view_t after_mid_block_1_norm2_silu_name =
      iree_string_view_empty();
  char after_mid_block_1_norm2_silu_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_block_1_norm2_silu"),
      after_mid_block_1_norm2_silu_name_storage,
      sizeof(after_mid_block_1_norm2_silu_name_storage),
      &after_mid_block_1_norm2_silu_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_block_1_norm2_silu_name));

  iree_string_view_t mid_block_1_conv2_name = iree_string_view_empty();
  char mid_block_1_conv2_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.block_1.conv2"),
      mid_block_1_conv2_name_storage, sizeof(mid_block_1_conv2_name_storage),
      &mid_block_1_conv2_name));
  id4_pipeline_program_tensor_t mid_block_1_conv2 =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_conv3x3_bias(
      builder, mid_block_1_conv2_name, mid_block_1_norm2_silu,
      config.input_width, config.input_height, 512, 512, config.batch_count,
      IREE_SV("decoder.mid.block_1.conv2.weight"),
      IREE_SV("decoder.mid.block_1.conv2.bias"), mid_block_1_conv2_name,
      &mid_block_1_conv2));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, mid_block_1_conv2_name, mid_block_1_conv2));
  iree_string_view_t after_mid_block_1_conv2_name = iree_string_view_empty();
  char after_mid_block_1_conv2_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_block_1_conv2"),
      after_mid_block_1_conv2_name_storage,
      sizeof(after_mid_block_1_conv2_name_storage),
      &after_mid_block_1_conv2_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_block_1_conv2_name));

  iree_string_view_t mid_block_1_output_name = iree_string_view_empty();
  char mid_block_1_output_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.block_1.output"),
      mid_block_1_output_name_storage, sizeof(mid_block_1_output_name_storage),
      &mid_block_1_output_name));
  id4_pipeline_program_tensor_t mid_block_1_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_elementwise_add(
      builder, mid_block_1_output_name, config.input, mid_block_1_conv2,
      config.input_width, config.input_height, 512, config.batch_count,
      mid_block_1_output_name, &mid_block_1_output));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, mid_block_1_output_name, mid_block_1_output));
  iree_string_view_t after_mid_block_1_output_name = iree_string_view_empty();
  char after_mid_block_1_output_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_block_1_output"),
      after_mid_block_1_output_name_storage,
      sizeof(after_mid_block_1_output_name_storage),
      &after_mid_block_1_output_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_block_1_output_name));

  iree_string_view_t mid_attention_norm_stats_name = iree_string_view_empty();
  char mid_attention_norm_stats_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.attn_1.norm.stats"),
      mid_attention_norm_stats_name_storage,
      sizeof(mid_attention_norm_stats_name_storage),
      &mid_attention_norm_stats_name));
  iree_string_view_t after_mid_attention_norm_stats_name =
      iree_string_view_empty();
  char after_mid_attention_norm_stats_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_attn_1_norm_stats"),
      after_mid_attention_norm_stats_name_storage,
      sizeof(after_mid_attention_norm_stats_name_storage),
      &after_mid_attention_norm_stats_name));
  iree_string_view_t mid_attention_norm_name = iree_string_view_empty();
  char mid_attention_norm_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.attn_1.norm"),
      mid_attention_norm_name_storage, sizeof(mid_attention_norm_name_storage),
      &mid_attention_norm_name));
  id4_pipeline_program_tensor_t mid_attention_norm =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_group_norm(
      builder, mid_attention_norm_stats_name, mid_block_1_output,
      config.input_width, config.input_height, 512, 32, config.batch_count,
      IREE_SV("decoder.mid.attn_1.norm.weight"),
      IREE_SV("decoder.mid.attn_1.norm.bias"), mid_attention_norm_stats_name,
      after_mid_attention_norm_stats_name, mid_attention_norm_name,
      ID4_VAE_PROGRAM_GROUP_NORM_FLAG_NONE, &mid_attention_norm));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, mid_attention_norm_name, mid_attention_norm));
  iree_string_view_t after_mid_attention_norm_name = iree_string_view_empty();
  char after_mid_attention_norm_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_attn_1_norm"),
      after_mid_attention_norm_name_storage,
      sizeof(after_mid_attention_norm_name_storage),
      &after_mid_attention_norm_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_attention_norm_name));

  iree_string_view_t mid_attention_q_name = iree_string_view_empty();
  char mid_attention_q_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.attn_1.q"),
      mid_attention_q_name_storage, sizeof(mid_attention_q_name_storage),
      &mid_attention_q_name));
  id4_pipeline_program_tensor_t mid_attention_q =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_conv1x1_bias(
      builder, mid_attention_q_name, mid_attention_norm, config.input_width,
      config.input_height, 512, 512, config.batch_count,
      IREE_SV("decoder.mid.attn_1.q.weight"),
      IREE_SV("decoder.mid.attn_1.q.bias"), mid_attention_q_name,
      &mid_attention_q));
  iree_string_view_t mid_attention_k_name = iree_string_view_empty();
  char mid_attention_k_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.attn_1.k"),
      mid_attention_k_name_storage, sizeof(mid_attention_k_name_storage),
      &mid_attention_k_name));
  id4_pipeline_program_tensor_t mid_attention_k =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_conv1x1_bias(
      builder, mid_attention_k_name, mid_attention_norm, config.input_width,
      config.input_height, 512, 512, config.batch_count,
      IREE_SV("decoder.mid.attn_1.k.weight"),
      IREE_SV("decoder.mid.attn_1.k.bias"), mid_attention_k_name,
      &mid_attention_k));
  iree_string_view_t mid_attention_v_name = iree_string_view_empty();
  char mid_attention_v_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.attn_1.v"),
      mid_attention_v_name_storage, sizeof(mid_attention_v_name_storage),
      &mid_attention_v_name));
  id4_pipeline_program_tensor_t mid_attention_v =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_conv1x1_bias(
      builder, mid_attention_v_name, mid_attention_norm, config.input_width,
      config.input_height, 512, 512, config.batch_count,
      IREE_SV("decoder.mid.attn_1.v.weight"),
      IREE_SV("decoder.mid.attn_1.v.bias"), mid_attention_v_name,
      &mid_attention_v));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(builder, mid_attention_q_name,
                                                  mid_attention_q));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(builder, mid_attention_k_name,
                                                  mid_attention_k));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(builder, mid_attention_v_name,
                                                  mid_attention_v));
  iree_string_view_t after_mid_attention_qkv_name = iree_string_view_empty();
  char after_mid_attention_qkv_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_attn_1_qkv"),
      after_mid_attention_qkv_name_storage,
      sizeof(after_mid_attention_qkv_name_storage),
      &after_mid_attention_qkv_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_attention_qkv_name));

  iree_string_view_t mid_attention_name = iree_string_view_empty();
  char mid_attention_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.attn_1.attention"),
      mid_attention_name_storage, sizeof(mid_attention_name_storage),
      &mid_attention_name));
  id4_pipeline_program_tensor_t mid_attention =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_spatial_attention(
      builder, mid_attention_name, mid_attention_q, mid_attention_k,
      mid_attention_v, config.input_width, config.input_height, 512,
      config.batch_count, mid_attention_name, &mid_attention));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_tap_tensor(builder, mid_attention_name, mid_attention));
  iree_string_view_t after_mid_attention_name = iree_string_view_empty();
  char after_mid_attention_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_attn_1_attention"),
      after_mid_attention_name_storage,
      sizeof(after_mid_attention_name_storage), &after_mid_attention_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_attention_name));

  iree_string_view_t mid_attention_proj_name = iree_string_view_empty();
  char mid_attention_proj_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.attn_1.proj_out"),
      mid_attention_proj_name_storage, sizeof(mid_attention_proj_name_storage),
      &mid_attention_proj_name));
  id4_pipeline_program_tensor_t mid_attention_proj =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_conv1x1_bias(
      builder, mid_attention_proj_name, mid_attention, config.input_width,
      config.input_height, 512, 512, config.batch_count,
      IREE_SV("decoder.mid.attn_1.proj_out.weight"),
      IREE_SV("decoder.mid.attn_1.proj_out.bias"), mid_attention_proj_name,
      &mid_attention_proj));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, mid_attention_proj_name, mid_attention_proj));
  iree_string_view_t after_mid_attention_proj_name = iree_string_view_empty();
  char after_mid_attention_proj_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_attn_1_proj_out"),
      after_mid_attention_proj_name_storage,
      sizeof(after_mid_attention_proj_name_storage),
      &after_mid_attention_proj_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_attention_proj_name));

  iree_string_view_t mid_attention_output_name = iree_string_view_empty();
  char mid_attention_output_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.attn_1.output"),
      mid_attention_output_name_storage,
      sizeof(mid_attention_output_name_storage), &mid_attention_output_name));
  id4_pipeline_program_tensor_t mid_attention_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_elementwise_add(
      builder, mid_attention_output_name, mid_block_1_output,
      mid_attention_proj, config.input_width, config.input_height, 512,
      config.batch_count, mid_attention_output_name, &mid_attention_output));
  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, mid_attention_output_name, mid_attention_output));
  iree_string_view_t after_mid_attention_output_name = iree_string_view_empty();
  char after_mid_attention_output_name_storage
      [ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_mid_attn_1_output"),
      after_mid_attention_output_name_storage,
      sizeof(after_mid_attention_output_name_storage),
      &after_mid_attention_output_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_mid_attention_output_name));

  iree_string_view_t mid_block_2_prefix = iree_string_view_empty();
  char mid_block_2_prefix_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.mid.block_2"),
      mid_block_2_prefix_storage, sizeof(mid_block_2_prefix_storage),
      &mid_block_2_prefix));
  id4_pipeline_program_tensor_t mid_block_2_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_resnet_block(
      builder, mid_block_2_prefix, IREE_SV("decoder.mid.block_2"),
      mid_attention_output, config.input_width, config.input_height, 512, 512,
      config.batch_count, &mid_block_2_output));

  id4_pipeline_program_tensor_t decoder_latent = mid_block_2_output;
  uint32_t current_width = config.input_width;
  uint32_t current_height = config.input_height;
  uint32_t current_channel_count = 512;
  iree_string_view_t decoder_up_program_root = iree_string_view_empty();
  char decoder_up_program_root_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.up"),
      decoder_up_program_root_storage, sizeof(decoder_up_program_root_storage),
      &decoder_up_program_root));
  for (uint32_t level_index = 0;
       level_index < ID4_VAE_FLUX2_DECODER_LEVEL_COUNT; ++level_index) {
    const uint32_t level = ID4_VAE_FLUX2_DECODER_LEVEL_COUNT - 1u - level_index;
    const uint32_t output_channel_count =
        id4_vae_flux2_decoder_up_output_channel_counts[level];
    for (uint32_t block = 0; block < ID4_VAE_FLUX2_DECODER_RESNET_BLOCK_COUNT;
         ++block) {
      iree_string_view_t program_prefix = iree_string_view_empty();
      char program_prefix_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
      IREE_RETURN_IF_ERROR(id4_vae_program_format_up_block_prefix(
          decoder_up_program_root, level, block, program_prefix_storage,
          sizeof(program_prefix_storage), &program_prefix));
      iree_string_view_t parameter_prefix = iree_string_view_empty();
      char parameter_prefix_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
      IREE_RETURN_IF_ERROR(id4_vae_program_format_up_block_prefix(
          IREE_SV("decoder.up"), level, block, parameter_prefix_storage,
          sizeof(parameter_prefix_storage), &parameter_prefix));
      id4_pipeline_program_tensor_t block_output =
          id4_pipeline_program_tensor_invalid();
      IREE_RETURN_IF_ERROR(id4_vae_program_author_resnet_block(
          builder, program_prefix, parameter_prefix, decoder_latent,
          current_width, current_height, current_channel_count,
          output_channel_count, config.batch_count, &block_output));
      decoder_latent = block_output;
      current_channel_count = output_channel_count;
    }

    if (level != 0) {
      iree_string_view_t upsample_prefix = iree_string_view_empty();
      char upsample_prefix_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
      IREE_RETURN_IF_ERROR(id4_vae_program_format_up_upsample_prefix(
          decoder_up_program_root, level, upsample_prefix_storage,
          sizeof(upsample_prefix_storage), &upsample_prefix));
      iree_string_view_t parameter_prefix = iree_string_view_empty();
      char parameter_prefix_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
      IREE_RETURN_IF_ERROR(id4_vae_program_format_up_upsample_prefix(
          IREE_SV("decoder.up"), level, parameter_prefix_storage,
          sizeof(parameter_prefix_storage), &parameter_prefix));
      iree_string_view_t weight_key = iree_string_view_empty();
      char weight_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
      IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
          parameter_prefix, IREE_SV(".conv.weight"), weight_key_storage,
          sizeof(weight_key_storage), &weight_key));
      iree_string_view_t bias_key = iree_string_view_empty();
      char bias_key_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
      IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
          parameter_prefix, IREE_SV(".conv.bias"), bias_key_storage,
          sizeof(bias_key_storage), &bias_key));
      iree_string_view_t output_name = iree_string_view_empty();
      char output_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
      IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
          upsample_prefix, IREE_SV(".output"), output_name_storage,
          sizeof(output_name_storage), &output_name));
      id4_pipeline_program_tensor_t upsample_output =
          id4_pipeline_program_tensor_invalid();
      IREE_RETURN_IF_ERROR(id4_vae_program_author_upsample_conv3x3_bias(
          builder, upsample_prefix, decoder_latent, current_width,
          current_height, current_channel_count, config.batch_count, weight_key,
          bias_key, output_name, &upsample_output));
      IREE_RETURN_IF_ERROR(
          id4_vae_program_tap_tensor(builder, output_name, upsample_output));
      iree_string_view_t barrier_name = iree_string_view_empty();
      char barrier_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
      IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
          upsample_prefix, IREE_SV(".after_output"), barrier_name_storage,
          sizeof(barrier_name_storage), &barrier_name));
      IREE_RETURN_IF_ERROR(id4_vae_program_barrier(builder, barrier_name));
      decoder_latent = upsample_output;
      current_width *= 2u;
      current_height *= 2u;
    }
  }

  if (current_width != config.output_width ||
      current_height != config.output_height || current_channel_count != 128) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "Flux2 VAE decoder ended at %ux%ux%u instead of %ux%ux128",
        current_width, current_height, current_channel_count,
        config.output_width, config.output_height);
  }

  iree_string_view_t norm_out_stats_name = iree_string_view_empty();
  char norm_out_stats_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.norm_out.stats"),
      norm_out_stats_name_storage, sizeof(norm_out_stats_name_storage),
      &norm_out_stats_name));
  iree_string_view_t after_norm_out_stats_name = iree_string_view_empty();
  char after_norm_out_stats_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_norm_out_stats"),
      after_norm_out_stats_name_storage,
      sizeof(after_norm_out_stats_name_storage), &after_norm_out_stats_name));
  iree_string_view_t norm_out_silu_name = iree_string_view_empty();
  char norm_out_silu_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.norm_out_silu"),
      norm_out_silu_name_storage, sizeof(norm_out_silu_name_storage),
      &norm_out_silu_name));
  id4_pipeline_program_tensor_t norm_out_silu =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_author_group_norm(
      builder, norm_out_stats_name, decoder_latent, current_width,
      current_height, current_channel_count, 32, config.batch_count,
      IREE_SV("decoder.norm_out.weight"), IREE_SV("decoder.norm_out.bias"),
      norm_out_stats_name, after_norm_out_stats_name, norm_out_silu_name,
      ID4_VAE_PROGRAM_GROUP_NORM_FLAG_APPLY_SILU, &norm_out_silu));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_tap_tensor(builder, norm_out_silu_name, norm_out_silu));
  iree_string_view_t after_norm_out_silu_name = iree_string_view_empty();
  char after_norm_out_silu_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_norm_out_silu"),
      after_norm_out_silu_name_storage,
      sizeof(after_norm_out_silu_name_storage), &after_norm_out_silu_name));
  IREE_RETURN_IF_ERROR(
      id4_vae_program_barrier(builder, after_norm_out_silu_name));

  iree_string_view_t conv_out_name = iree_string_view_empty();
  char conv_out_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".decoder.conv_out"),
      conv_out_name_storage, sizeof(conv_out_name_storage), &conv_out_name));
  IREE_RETURN_IF_ERROR(id4_vae_program_dispatch_conv3x3_bias(
      builder, conv_out_name, norm_out_silu, current_width, current_height,
      current_channel_count, config.output_channel_count, config.batch_count,
      IREE_SV("decoder.conv_out.weight"), IREE_SV("decoder.conv_out.bias"),
      config.output));
  iree_string_view_t after_decoder_conv_out_name = iree_string_view_empty();
  char
      after_decoder_conv_out_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
  IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
      config.program_prefix, IREE_SV(".after_decoder_conv_out"),
      after_decoder_conv_out_name_storage,
      sizeof(after_decoder_conv_out_name_storage),
      &after_decoder_conv_out_name));
  return id4_vae_program_barrier(builder, after_decoder_conv_out_name);
}

iree_status_t id4_vae_program_resolve_decode_tiling(
    id4_vae_model_config_t model, id4_vae_decode_request_config_t request,
    id4_vae_decode_tiling_plan_t* out_tiling_plan) {
  if (!out_tiling_plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE decode tiling plan output is required");
  }
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

  const uint32_t latent_width = (uint32_t)request.latent_shape.dims[0];
  const uint32_t latent_height = (uint32_t)request.latent_shape.dims[1];
  const uint32_t latent_channel_count = (uint32_t)request.latent_shape.dims[2];
  const uint32_t batch_count = (uint32_t)request.latent_shape.dims[3];
  uint32_t tile_size_x = 0;
  uint32_t tile_size_y = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_resolve_tile_size(
      model, request, &tile_size_x, &tile_size_y));

  const float overlap = request.tiling.mode == ID4_VAE_TILING_MODE_DISABLED
                            ? 0.0f
                            : request.tiling.overlap;
  const uint32_t decoded_height = latent_height * model.scale_y;
  const uint32_t decoded_width = latent_width * model.scale_x;
  id4_vae_program_axis_tiling_t axis_x;
  IREE_RETURN_IF_ERROR(id4_vae_program_resolve_axis_tiling(
      latent_width, tile_size_x, overlap, &axis_x));
  id4_vae_program_axis_tiling_t axis_y;
  IREE_RETURN_IF_ERROR(id4_vae_program_resolve_axis_tiling(
      latent_height, tile_size_y, overlap, &axis_y));

  uint64_t decoded_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      decoded_width, decoded_height, model.decoded_channel_count, batch_count,
      &decoded_element_count));
  if (decoded_element_count == 0 ||
      decoded_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE decoded element count %" PRIu64 " exceeds max count %u",
        decoded_element_count, ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
  }

  uint64_t latent_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      latent_width, latent_height, latent_channel_count, batch_count,
      &latent_element_count));
  uint64_t tile_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
      tile_size_x * model.scale_x, tile_size_y * model.scale_y,
      model.decoded_channel_count, batch_count, &tile_element_count));

  out_tiling_plan->batch_count = batch_count;
  out_tiling_plan->latent_height = latent_height;
  out_tiling_plan->latent_width = latent_width;
  out_tiling_plan->latent_channel_count = latent_channel_count;
  out_tiling_plan->decoded_height = decoded_height;
  out_tiling_plan->decoded_width = decoded_width;
  out_tiling_plan->decoded_channel_count = model.decoded_channel_count;
  out_tiling_plan->tile_size_x = tile_size_x;
  out_tiling_plan->tile_size_y = tile_size_y;
  out_tiling_plan->tile_count_x = axis_x.tile_count;
  out_tiling_plan->tile_count_y = axis_y.tile_count;
  out_tiling_plan->overlap_pixels_x = axis_x.overlap_pixels;
  out_tiling_plan->overlap_pixels_y = axis_y.overlap_pixels;
  out_tiling_plan->tile_step_x = axis_x.tile_step;
  out_tiling_plan->tile_step_y = axis_y.tile_step;
  out_tiling_plan->overlap_x = axis_x.overlap;
  out_tiling_plan->overlap_y = axis_y.overlap;
  out_tiling_plan->overlap_milli =
      id4_vae_program_round_positive_to_u32(axis_x.overlap * 1000.0f);
  out_tiling_plan->decoded_element_count = decoded_element_count;
  out_tiling_plan->latent_element_count = latent_element_count;
  out_tiling_plan->tile_element_count = tile_element_count;
  out_tiling_plan->estimated_transient_peak =
      tile_element_count * sizeof(float);
  return iree_ok_status();
}

iree_status_t id4_vae_program_author_decode_from_tensor(
    const id4_vae_program_options_t* options,
    id4_pipeline_program_tensor_t latent, iree_string_view_t decoded_image_name,
    id4_pipeline_program_builder_t* builder,
    id4_pipeline_program_tensor_t* out_decoded_image) {
  if (!out_decoded_image) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE decoded image output is required");
  }
  *out_decoded_image = id4_pipeline_program_tensor_invalid();
  id4_vae_decode_tiling_plan_t tiling_plan;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_validate_decode_options(options, builder, &tiling_plan));
  if (!id4_pipeline_program_tensor_is_valid(latent)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE decode latent tensor is invalid");
  }
  if (iree_string_view_is_empty(decoded_image_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE decoded image tensor name is required");
  }

  id4_pipeline_program_shape_t decoded_shape =
      id4_pipeline_program_make_shape_rank4(
          tiling_plan.decoded_width, tiling_plan.decoded_height,
          tiling_plan.decoded_channel_count, tiling_plan.batch_count);

  id4_pipeline_program_tensor_t decoder_latent = latent;
  id4_vae_model_config_t decoder_model = options->model;
  id4_vae_decode_tiling_plan_t decoder_tiling_plan = tiling_plan;
  if (options->model.implementation == ID4_VAE_IMPLEMENTATION_FLUX2) {
    id4_pipeline_program_tensor_t latent_mean =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_vae_program_constant_tensor(
        builder, IREE_SV("vae.flux2.latent_mean"), id4_vae_flux2_latent_mean,
        IREE_ARRAYSIZE(id4_vae_flux2_latent_mean), &latent_mean));
    id4_pipeline_program_tensor_t latent_std =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_vae_program_constant_tensor(
        builder, IREE_SV("vae.flux2.latent_std"), id4_vae_flux2_latent_std,
        IREE_ARRAYSIZE(id4_vae_flux2_latent_std), &latent_std));

    const uint32_t internal_width = tiling_plan.latent_width * 2u;
    const uint32_t internal_height = tiling_plan.latent_height * 2u;
    const uint32_t internal_channel_count =
        tiling_plan.latent_channel_count / 4u;
    id4_pipeline_program_shape_t internal_shape =
        id4_pipeline_program_make_shape_rank4(internal_width, internal_height,
                                              internal_channel_count,
                                              tiling_plan.batch_count);
    id4_pipeline_program_tensor_t internal_latent =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
        builder, IREE_SV("vae.flux2.internal_latent"), internal_shape,
        &internal_latent));

    id4_vae_program_config_list_t affine_config_list;
    IREE_RETURN_IF_ERROR(id4_vae_program_build_flux2_affine_configs(
        tiling_plan, &affine_config_list));
    id4_pipeline_program_dispatch_binding_t affine_bindings[] = {
        id4_pipeline_program_read(latent),
        id4_pipeline_program_read(latent_mean),
        id4_pipeline_program_read(latent_std),
        id4_pipeline_program_write(internal_latent),
    };
    id4_pipeline_program_dispatch_loom_options_t affine_dispatch_options;
    memset(&affine_dispatch_options, 0, sizeof(affine_dispatch_options));
    affine_dispatch_options.structure_size = sizeof(affine_dispatch_options);
    affine_dispatch_options.name = IREE_SV("vae.flux2.affine_pixel_shuffle");
    affine_dispatch_options.kernel = id4_pipeline_make_kernel_ref(
        IREE_SV("vae/flux2_affine_pixel_shuffle_f32"),
        IREE_SV("id4_vae_flux2_affine_pixel_shuffle_f32"));
    affine_dispatch_options.dispatch_config =
        id4_vae_program_make_dispatch_config(
            (uint32_t)tiling_plan.latent_element_count);
    affine_dispatch_options.config_binding_count = affine_config_list.count;
    affine_dispatch_options.config_bindings = affine_config_list.bindings;
    affine_dispatch_options.binding_count = IREE_ARRAYSIZE(affine_bindings);
    affine_dispatch_options.bindings = affine_bindings;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_dispatch_loom(builder, &affine_dispatch_options));

    IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
        builder, IREE_SV("vae.flux2.internal_latent"), internal_latent));
    IREE_RETURN_IF_ERROR(
        id4_vae_program_barrier(builder, IREE_SV("vae.flux2.after_affine")));

    id4_pipeline_program_tensor_t post_quant =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_vae_program_author_conv1x1_bias(
        builder, IREE_SV("vae.flux2.post_quant_conv"), internal_latent,
        internal_width, internal_height, internal_channel_count,
        internal_channel_count, tiling_plan.batch_count,
        IREE_SV("decoder.post_quant_conv.weight"),
        IREE_SV("decoder.post_quant_conv.bias"),
        IREE_SV("vae.flux2.post_quant_conv"), &post_quant));

    IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
        builder, IREE_SV("vae.flux2.post_quant_conv"), post_quant));
    IREE_RETURN_IF_ERROR(id4_vae_program_barrier(
        builder, IREE_SV("vae.flux2.after_post_quant")));

    id4_pipeline_program_tensor_t decoder_conv_in =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_vae_program_author_conv3x3_bias(
        builder, IREE_SV("vae.decoder.conv_in"), post_quant, internal_width,
        internal_height, internal_channel_count, 512, tiling_plan.batch_count,
        IREE_SV("decoder.conv_in.weight"), IREE_SV("decoder.conv_in.bias"),
        IREE_SV("vae.decoder.conv_in"), &decoder_conv_in));

    IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
        builder, IREE_SV("vae.decoder.conv_in"), decoder_conv_in));
    IREE_RETURN_IF_ERROR(
        id4_vae_program_barrier(builder, IREE_SV("vae.after_decoder_conv_in")));

    id4_pipeline_program_tensor_t decoded_image =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_vae_program_import_tensor(
        builder, decoded_image_name, 0, decoded_shape, &decoded_image));

    const bool use_spatial_tiling =
        tiling_plan.tile_count_x > 1 || tiling_plan.tile_count_y > 1;
    if (!use_spatial_tiling) {
      id4_vae_program_flux2_decoder_tail_config_t tail_config = {
          .program_prefix = IREE_SV("vae"),
          .input = decoder_conv_in,
          .input_width = internal_width,
          .input_height = internal_height,
          .batch_count = tiling_plan.batch_count,
          .output_width = tiling_plan.decoded_width,
          .output_height = tiling_plan.decoded_height,
          .output_channel_count = tiling_plan.decoded_channel_count,
          .output = decoded_image,
      };
      IREE_RETURN_IF_ERROR(
          id4_vae_program_author_flux2_decoder_tail(builder, tail_config));
    } else {
      const uint32_t internal_tile_width = tiling_plan.tile_size_x * 2u;
      const uint32_t internal_tile_height = tiling_plan.tile_size_y * 2u;
      const uint32_t decoded_tile_width =
          tiling_plan.tile_size_x * options->model.scale_x;
      const uint32_t decoded_tile_height =
          tiling_plan.tile_size_y * options->model.scale_y;
      const uint32_t overlap_x =
          tiling_plan.overlap_pixels_x * options->model.scale_x;
      const uint32_t overlap_y =
          tiling_plan.overlap_pixels_y * options->model.scale_y;
      uint64_t internal_tile_element_count = 0;
      IREE_RETURN_IF_ERROR(id4_vae_program_whcb_element_count(
          internal_tile_width, internal_tile_height, 512,
          tiling_plan.batch_count, &internal_tile_element_count));
      if (internal_tile_element_count > ID4_VAE_DECODE_MAX_ELEMENT_COUNT) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "Flux2 VAE internal tile element count %" PRIu64
                                " exceeds max count %u",
                                internal_tile_element_count,
                                ID4_VAE_DECODE_MAX_ELEMENT_COUNT);
      }

      id4_vae_program_tile_io_config_t clear_config = {
          .clear_element_count = tiling_plan.decoded_element_count,
          .source_width = tiling_plan.decoded_width,
          .source_height = tiling_plan.decoded_height,
          .source_channel_count = tiling_plan.decoded_channel_count,
          .batch_count = tiling_plan.batch_count,
          .tile_width = decoded_tile_width,
          .tile_height = decoded_tile_height,
          .tile_origin_x = 0,
          .tile_origin_y = 0,
          .tile_element_count = tiling_plan.tile_element_count,
          .image_width = tiling_plan.decoded_width,
          .image_height = tiling_plan.decoded_height,
          .image_channel_count = tiling_plan.decoded_channel_count,
          .tile_skip_x = 0,
          .tile_skip_y = 0,
          .overlap_x = overlap_x,
          .overlap_y = overlap_y,
      };
      IREE_RETURN_IF_ERROR(id4_vae_program_author_tile_clear(
          builder, IREE_SV("vae.decode.clear"), clear_config, decoded_image));
      IREE_RETURN_IF_ERROR(
          id4_vae_program_barrier(builder, IREE_SV("vae.after_decode_clear")));

      for (uint32_t tile_y = 0; tile_y < tiling_plan.tile_count_y; ++tile_y) {
        for (uint32_t tile_x = 0; tile_x < tiling_plan.tile_count_x; ++tile_x) {
          id4_vae_program_tile_location_t location =
              id4_vae_program_resolve_tile_location(
                  &tiling_plan, options->model, tile_x, tile_y);
          iree_string_view_t tile_prefix = iree_string_view_empty();
          char tile_prefix_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
          int tile_prefix_length = snprintf(
              tile_prefix_storage, sizeof(tile_prefix_storage),
              "vae.tile.%u.%u", location.ordinal_y, location.ordinal_x);
          if (tile_prefix_length < 0 || (iree_host_size_t)tile_prefix_length >=
                                            sizeof(tile_prefix_storage)) {
            return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "failed to format VAE tile prefix");
          }
          tile_prefix = iree_make_string_view(
              tile_prefix_storage, (iree_host_size_t)tile_prefix_length);

          iree_string_view_t extract_name = iree_string_view_empty();
          char extract_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
          IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
              tile_prefix, IREE_SV(".extract_decoder_conv_in"),
              extract_name_storage, sizeof(extract_name_storage),
              &extract_name));
          iree_string_view_t source_tile_name = iree_string_view_empty();
          char source_tile_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
          IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
              tile_prefix, IREE_SV(".decoder.conv_in.tile"),
              source_tile_name_storage, sizeof(source_tile_name_storage),
              &source_tile_name));
          id4_vae_program_tile_io_config_t extract_config = {
              .clear_element_count = tiling_plan.decoded_element_count,
              .source_width = internal_width,
              .source_height = internal_height,
              .source_channel_count = 512,
              .batch_count = tiling_plan.batch_count,
              .tile_width = internal_tile_width,
              .tile_height = internal_tile_height,
              .tile_origin_x = location.origin_x * 2u,
              .tile_origin_y = location.origin_y * 2u,
              .tile_element_count = internal_tile_element_count,
              .image_width = tiling_plan.decoded_width,
              .image_height = tiling_plan.decoded_height,
              .image_channel_count = tiling_plan.decoded_channel_count,
              .tile_skip_x = 0,
              .tile_skip_y = 0,
              .overlap_x = overlap_x,
              .overlap_y = overlap_y,
          };
          id4_pipeline_program_tensor_t source_tile =
              id4_pipeline_program_tensor_invalid();
          IREE_RETURN_IF_ERROR(id4_vae_program_author_tile_extract(
              builder, extract_name, decoder_conv_in, extract_config,
              source_tile_name, &source_tile));
          iree_string_view_t after_extract_name = iree_string_view_empty();
          char after_extract_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
          IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
              tile_prefix, IREE_SV(".after_extract_decoder_conv_in"),
              after_extract_name_storage, sizeof(after_extract_name_storage),
              &after_extract_name));
          IREE_RETURN_IF_ERROR(
              id4_vae_program_barrier(builder, after_extract_name));

          iree_string_view_t decoded_tile_name = iree_string_view_empty();
          char decoded_tile_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
          IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
              tile_prefix, IREE_SV(".decoded_tile"), decoded_tile_name_storage,
              sizeof(decoded_tile_name_storage), &decoded_tile_name));
          id4_pipeline_program_tensor_t decoded_tile =
              id4_pipeline_program_tensor_invalid();
          IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
              builder, decoded_tile_name,
              id4_pipeline_program_make_shape_rank4(
                  decoded_tile_width, decoded_tile_height,
                  tiling_plan.decoded_channel_count, tiling_plan.batch_count),
              &decoded_tile));
          id4_vae_program_flux2_decoder_tail_config_t tail_config = {
              .program_prefix = tile_prefix,
              .input = source_tile,
              .input_width = internal_tile_width,
              .input_height = internal_tile_height,
              .batch_count = tiling_plan.batch_count,
              .output_width = decoded_tile_width,
              .output_height = decoded_tile_height,
              .output_channel_count = tiling_plan.decoded_channel_count,
              .output = decoded_tile,
          };
          IREE_RETURN_IF_ERROR(
              id4_vae_program_author_flux2_decoder_tail(builder, tail_config));

          iree_string_view_t merge_name = iree_string_view_empty();
          char merge_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
          IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
              tile_prefix, IREE_SV(".merge_decoded_tile"), merge_name_storage,
              sizeof(merge_name_storage), &merge_name));
          id4_vae_program_tile_io_config_t merge_config = {
              .clear_element_count = tiling_plan.decoded_element_count,
              .source_width = decoded_tile_width,
              .source_height = decoded_tile_height,
              .source_channel_count = tiling_plan.decoded_channel_count,
              .batch_count = tiling_plan.batch_count,
              .tile_width = decoded_tile_width,
              .tile_height = decoded_tile_height,
              .tile_origin_x = location.origin_x * options->model.scale_x,
              .tile_origin_y = location.origin_y * options->model.scale_y,
              .tile_element_count = tiling_plan.tile_element_count,
              .image_width = tiling_plan.decoded_width,
              .image_height = tiling_plan.decoded_height,
              .image_channel_count = tiling_plan.decoded_channel_count,
              .tile_skip_x = location.decoded_skip_x,
              .tile_skip_y = location.decoded_skip_y,
              .overlap_x = overlap_x,
              .overlap_y = overlap_y,
          };
          IREE_RETURN_IF_ERROR(id4_vae_program_author_tile_merge(
              builder, merge_name, decoded_tile, merge_config, decoded_image));
          iree_string_view_t after_merge_name = iree_string_view_empty();
          char after_merge_name_storage[ID4_VAE_PROGRAM_NAME_BUFFER_CAPACITY];
          IREE_RETURN_IF_ERROR(id4_vae_program_format_suffix(
              tile_prefix, IREE_SV(".after_merge_decoded_tile"),
              after_merge_name_storage, sizeof(after_merge_name_storage),
              &after_merge_name));
          IREE_RETURN_IF_ERROR(
              id4_vae_program_barrier(builder, after_merge_name));
        }
      }
    }

    id4_pipeline_program_export_options_t export_options;
    memset(&export_options, 0, sizeof(export_options));
    export_options.structure_size = sizeof(export_options);
    export_options.name = decoded_image_name;
    export_options.tensor = decoded_image;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_export(builder, &export_options));

    *out_decoded_image = decoded_image;
    return iree_ok_status();
  }

  id4_pipeline_program_shape_t tile_shape =
      id4_pipeline_program_make_shape_rank4(
          decoder_tiling_plan.tile_size_x * decoder_model.scale_x,
          decoder_tiling_plan.tile_size_y * decoder_model.scale_y,
          decoder_tiling_plan.decoded_channel_count,
          decoder_tiling_plan.batch_count);

  id4_pipeline_program_tensor_t decoded_tile =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_acquire_tensor(
      builder, IREE_SV("vae.decode.tile"), tile_shape, &decoded_tile));
  id4_pipeline_program_tensor_t decoded_image =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_import_tensor(
      builder, decoded_image_name, 0, decoded_shape, &decoded_image));

  id4_vae_program_config_list_t config_list;
  IREE_RETURN_IF_ERROR(id4_vae_program_build_decode_configs(
      decoder_model, decoder_tiling_plan, &config_list));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(decoder_latent),
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

  IREE_RETURN_IF_ERROR(id4_vae_program_tap_tensor(
      builder, IREE_SV("vae.decode.tile"), decoded_tile));

  id4_pipeline_program_export_options_t export_options;
  memset(&export_options, 0, sizeof(export_options));
  export_options.structure_size = sizeof(export_options);
  export_options.name = decoded_image_name;
  export_options.tensor = decoded_image;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_export(builder, &export_options));

  *out_decoded_image = decoded_image;
  return iree_ok_status();
}

iree_status_t id4_vae_program_author_decode(
    const id4_vae_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  id4_vae_decode_tiling_plan_t tiling_plan;
  IREE_RETURN_IF_ERROR(
      id4_vae_program_validate_decode_options(options, builder, &tiling_plan));

  id4_pipeline_program_tensor_t latent = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_program_import_tensor(
      builder, IREE_SV("media.latent.input"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      options->request.latent_shape, &latent));

  id4_pipeline_program_tensor_t decoded_image =
      id4_pipeline_program_tensor_invalid();
  return id4_vae_program_author_decode_from_tensor(
      options, latent, IREE_SV("media.image.decoded"), builder, &decoded_image);
}

const id4_vae_model_config_t* id4_vae_program_flux2_model_config(void) {
  static const id4_vae_model_config_t model = {
      // Latent-to-image scale factor along the width axis.
      .scale_x = 16,
      // Latent-to-image scale factor along the height axis.
      .scale_y = 16,
      // Latent-to-media scale factor along the temporal axis.
      .scale_t = 1,
      // Flux2 public diffusion latents use 128 channels.
      .latent_channel_count = 128,
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
      // Flux2 AutoEncoderKL implementation.
      .implementation = ID4_VAE_IMPLEMENTATION_FLUX2,
  };
  return &model;
}
