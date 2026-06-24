// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_decode_program.h"

#include <string.h>

#include "experimental/id4/stages/vae_program.h"

enum {
  ID4_IDEOGRAM4_DECODE_SUPPORTED_RANK = 4,
};

static const id4_ideogram4_decode_model_config_t
    id4_ideogram4_decode_program_ideogram4_model_config_value = {
        .vae =
            {
                .scale_x = 16,
                .scale_y = 16,
                .scale_t = 1,
                .latent_channel_count = 128,
                .decoded_channel_count = 3,
                .min_tile_size_x = 4,
                .min_tile_size_y = 4,
                .default_tile_size_x = 32,
                .default_tile_size_y = 32,
                .max_overlap = 0.5f,
                .capabilities = ID4_VAE_CAPABILITY_DECODE |
                                ID4_VAE_CAPABILITY_SPATIAL_TILING,
                .implementation = ID4_VAE_IMPLEMENTATION_FLUX2,
            },
};

static iree_status_t id4_ideogram4_decode_program_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_ideogram4_decode_program_validate_model_config(
    id4_ideogram4_decode_model_config_t model) {
  if (!iree_all_bits_set(model.vae.capabilities, ID4_VAE_CAPABILITY_DECODE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 decode VAE must support decode");
  }
  if (model.vae.latent_channel_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 decode VAE latent channel count must be nonzero");
  }
  return iree_ok_status();
}

static iree_status_t
id4_ideogram4_decode_program_validate_vae_activation_format(
    id4_vae_activation_format_t activation_format) {
  switch (activation_format) {
    case ID4_VAE_ACTIVATION_FORMAT_F32_CANONICAL:
    case ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 decode VAE activation format %" PRIu32
                              " is invalid",
                              (uint32_t)activation_format);
  }
}

static iree_status_t id4_ideogram4_decode_program_validate_options(
    const id4_ideogram4_decode_program_options_t* options,
    const id4_pipeline_program_builder_t* builder,
    uint64_t* out_diffusion_element_count) {
  *out_diffusion_element_count = 0;
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 decode program options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_decode_program_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram4 decode program")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 decode program extension structures are not supported");
  }
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 decode program builder is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_decode_program_validate_model_config(options->model));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_decode_program_validate_vae_activation_format(
          options->vae_activation_format));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_shape_element_count(
      options->request.diffusion_latent_shape, out_diffusion_element_count));
  if (*out_diffusion_element_count == 0 ||
      *out_diffusion_element_count >
          ID4_IDEOGRAM4_DECODE_MAX_DIFFUSION_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 decode diffusion latent element count %" PRIu64
        " exceeds max count %u",
        *out_diffusion_element_count,
        ID4_IDEOGRAM4_DECODE_MAX_DIFFUSION_ELEMENT_COUNT);
  }
  return id4_ideogram4_decode_program_validate_diffusion_latent_shape(
      options->model, options->request.diffusion_latent_shape);
}

static iree_status_t id4_ideogram4_decode_program_import_tensor(
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

iree_status_t id4_ideogram4_decode_program_validate_diffusion_latent_shape(
    id4_ideogram4_decode_model_config_t model,
    id4_pipeline_program_shape_t diffusion_latent_shape) {
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_decode_program_validate_model_config(model));
  if (diffusion_latent_shape.rank != ID4_IDEOGRAM4_DECODE_SUPPORTED_RANK) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 decode diffusion latent shape must be rank-4 WHCB, got "
        "rank %u",
        diffusion_latent_shape.rank);
  }
  for (uint32_t i = 0; i < diffusion_latent_shape.rank; ++i) {
    if (diffusion_latent_shape.dims[i] == 0 ||
        diffusion_latent_shape.dims[i] > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Ideogram4 decode diffusion latent dimension %u is out of range", i);
    }
  }
  if (diffusion_latent_shape.dims[2] != model.vae.latent_channel_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 decode diffusion latent channel count "
                            "%" PRIu64 " does not match VAE channel count %u",
                            diffusion_latent_shape.dims[2],
                            model.vae.latent_channel_count);
  }
  if (diffusion_latent_shape.dims[3] != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 decode diffusion latent batch dimension must be 1");
  }
  return iree_ok_status();
}

iree_status_t id4_ideogram4_decode_program_author_decode(
    const id4_ideogram4_decode_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  uint64_t diffusion_element_count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_decode_program_validate_options(
      options, builder, &diffusion_element_count));

  id4_pipeline_program_tensor_t diffusion_latent =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_decode_program_import_tensor(
      builder, IREE_SV("media.latent.diffusion"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      options->request.diffusion_latent_shape, &diffusion_latent));

  id4_vae_program_options_t vae_options;
  memset(&vae_options, 0, sizeof(vae_options));
  vae_options.structure_size = sizeof(vae_options);
  vae_options.model = options->model.vae;
  vae_options.request.latent_shape = options->request.diffusion_latent_shape;
  vae_options.request.tiling = options->request.vae_tiling;
  vae_options.activation_format = options->vae_activation_format;
  id4_pipeline_program_tensor_t decoded_image =
      id4_pipeline_program_tensor_invalid();
  return id4_vae_program_author_decode_from_tensor(
      &vae_options, diffusion_latent, IREE_SV("media.image.decoded"), builder,
      &decoded_image);
}

const id4_ideogram4_decode_model_config_t*
id4_ideogram4_decode_program_ideogram4_model_config(void) {
  return &id4_ideogram4_decode_program_ideogram4_model_config_value;
}
