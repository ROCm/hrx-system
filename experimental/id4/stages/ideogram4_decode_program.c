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

// Official patch-major latent affine reindexed to the ID4 public latent ABI.
static const float id4_ideogram4_latent_shift[128] = {
    0.01984364f,  0.02583857f,  0.01924137f,  0.02578168f,  0.10149707f,
    0.10190072f,  0.07652984f,  0.07993659f,  0.29689495f,  0.28402294f,
    0.2995608f,   0.28642181f,  0.27188619f,  0.26952152f,  0.2628057f,
    0.26038408f,  -0.21445648f, -0.21634675f, -0.22011674f, -0.22459419f,
    -0.15979549f, -0.17938656f, -0.12715361f, -0.14820155f, 0.05021099f,
    0.04358909f,  0.04879879f,  0.04059549f,  -0.15083604f, -0.15007621f,
    -0.14075719f, -0.14043529f, -0.15360136f, -0.1548502f,  -0.15935895f,
    -0.16111187f, -0.20131799f, -0.18971131f, -0.2123584f,  -0.2020305f,
    0.01922352f,  0.02710861f,  0.01974813f,  0.02602069f,  0.0622626f,
    0.05609494f,  0.05523547f,  0.04852717f,  0.10140969f,  0.10697846f,
    0.10011992f,  0.10432153f,  -0.06739428f, -0.06854968f, -0.06428964f,
    -0.06309942f, 0.3758261f,   0.38167698f,  0.37781868f,  0.38402443f,
    -0.233712f,   -0.24269937f, -0.21491644f, -0.22397003f, 0.35164491f,
    0.35705471f,  0.34254215f,  0.34814481f,  -0.02590912f, -0.03063305f,
    -0.03153528f, -0.03774432f, -0.0271935f,  -0.02946109f, -0.0310082f,
    -0.03381438f, -0.10833897f, -0.11244286f, -0.10761415f, -0.11245691f,
    -0.1476848f,  -0.14336038f, -0.14730405f, -0.14128767f, -0.01130957f,
    -0.01362137f, -0.02475182f, -0.02853208f, -0.2298372f,  -0.21863696f,
    -0.2285588f,  -0.21752016f, 0.23526423f,  0.23228983f,  0.2515081f,
    0.24872463f,  -0.10893522f, -0.11739769f, -0.10445128f, -0.11399775f,
    0.11957631f,  0.11693044f,  0.12446f,     0.1222687f,   0.04047799f,
    0.02563311f,  0.07062869f,  0.05620835f,  0.3134589f,   0.31356594f,
    0.30880162f,  0.309178f,    -0.17225064f, -0.17420591f, -0.18016875f,
    -0.18065738f, -0.18646109f, -0.19006285f, -0.18869164f, -0.19401479f,
    -0.34691978f, -0.34905377f, -0.34533499f, -0.34495114f, -0.03571246f,
    -0.04025005f, -0.0129177f,  -0.01760592f,
};

// Official patch-major latent scales reindexed to the ID4 public latent ABI.
static const float id4_ideogram4_latent_scale[128] = {
    1.63933691f, 1.63487607f, 1.64710857f, 1.65339716f, 1.70204478f,
    1.69513249f, 1.68163503f, 1.67540638f, 1.73642566f, 1.72933756f,
    1.74000294f, 1.73298504f, 1.90004803f, 1.91310663f, 1.92784786f,
    1.94067348f, 1.6675316f,  1.67035057f, 1.67411194f, 1.67893609f,
    1.69059584f, 1.72286863f, 1.67395548f, 1.70635117f, 1.56853198f,
    1.56719251f, 1.57406532f, 1.5730906f,  1.62314944f, 1.61934825f,
    1.62199356f, 1.61928553f, 1.89106626f, 1.88628859f, 1.87618195f,
    1.87148809f, 1.58086668f, 1.56911539f, 1.5584375f,  1.56244866f,
    1.60822129f, 1.59455129f, 1.57438785f, 1.56697152f, 1.60962993f,
    1.60829869f, 1.61711053f, 1.61584394f, 1.63322129f, 1.62470611f,
    1.63094305f, 1.62759496f, 1.56074359f, 1.56052853f, 1.55644029f,
    1.55480378f, 1.73419528f, 1.73677003f, 1.73124302f, 1.73484107f,
    1.7919265f,  1.77563606f, 1.80666627f, 1.79055143f, 1.64040632f,
    1.63732541f, 1.6463621f,  1.64688773f, 1.66802808f, 1.66370527f,
    1.65932006f, 1.66121492f, 1.60390303f, 1.59508952f, 1.60816188f,
    1.60135887f, 1.75480492f, 1.75153949f, 1.75682671f, 1.75254572f,
    1.63187587f, 1.63029275f, 1.64695873f, 1.64798332f, 1.64334594f,
    1.64517667f, 1.63121722f, 1.62989921f, 1.61722884f, 1.61659342f,
    1.61380832f, 1.61381592f, 1.60146046f, 1.59722044f, 1.60478651f,
    1.60792883f, 1.63459219f, 1.64103121f, 1.63396035f, 1.63939668f,
    1.55291476f, 1.5408531f,  1.53505068f, 1.53075757f, 1.68771497f,
    1.68610394f, 1.65534289f, 1.65371318f, 1.68415657f, 1.67772755f,
    1.67132281f, 1.66801185f, 1.78966054f, 1.78998563f, 1.80317197f,
    1.80029087f, 1.66631641f, 1.66621713f, 1.6767314f,  1.67591476f,
    1.65626686f, 1.65458955f, 1.65700938f, 1.65655173f, 1.65976433f,
    1.66041308f, 1.68426259f, 1.68533454f,
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
                .latent_affine =
                    {
                        .channel_count =
                            IREE_ARRAYSIZE(id4_ideogram4_latent_shift),
                        .scales = id4_ideogram4_latent_scale,
                        .offsets = id4_ideogram4_latent_shift,
                    },
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
  vae_options.request.attention_implementation =
      options->request.vae_attention_implementation;
  vae_options.parameter_scope = options->parameter_scope;
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
