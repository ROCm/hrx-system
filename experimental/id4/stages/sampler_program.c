// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/sampler_program.h"

#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/program.h"
#include "iree/hal/command_buffer.h"

enum {
  ID4_SAMPLER_WORKGROUP_SIZE_X = 256,
  ID4_SAMPLER_CONFIG_VALUE_BUFFER_CAPACITY = 16,
};

static iree_status_t id4_sampler_program_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

iree_string_view_t id4_sampler_program_cond_out_boundary_name(void) {
  return IREE_SV("cond_out");
}

iree_string_view_t id4_sampler_program_uncond_out_boundary_name(void) {
  return IREE_SV("uncond_out");
}

iree_string_view_t id4_sampler_program_x_t_boundary_name(void) {
  return IREE_SV("x_t");
}

iree_string_view_t id4_sampler_program_scalings_boundary_name(void) {
  return IREE_SV("scalings");
}

iree_string_view_t id4_sampler_program_guidance_boundary_name(void) {
  return IREE_SV("guidance");
}

iree_string_view_t id4_sampler_program_denoised_boundary_name(void) {
  return IREE_SV("denoised");
}

iree_string_view_t id4_sampler_program_guided_pred_tap_name(void) {
  return IREE_SV("guided_pred");
}

static iree_string_view_t id4_sampler_program_guided_pred_tensor_name(void) {
  return IREE_SV("guided_pred");
}

static iree_string_view_t id4_sampler_program_dispatch_name(void) {
  return IREE_SV("sampler.denoise_step.cfg_denoise");
}

static iree_string_view_t id4_sampler_program_element_count_config_key(void) {
  return IREE_SV("id4.sampler.element_count");
}

static id4_pipeline_kernel_ref_t id4_sampler_program_cfg_denoise_kernel_ref(
    void) {
  return id4_pipeline_make_kernel_ref(IREE_SV("sampler/cfg_denoise_f32"),
                                      IREE_SV("id4_sampler_cfg_denoise_f32"));
}

static iree_status_t id4_sampler_program_format_u64(
    uint64_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%" PRIu64, value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format sampler program string");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_sampler_program_request_element_count(
    id4_sampler_denoise_request_config_t request, uint64_t* out_element_count) {
  *out_element_count = 0;
  if (request.latent_shape.rank == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler latent shape rank must be nonzero");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_shape_element_count(
      request.latent_shape, out_element_count));
  if (*out_element_count == 0 ||
      *out_element_count > ID4_SAMPLER_DENOISE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "sampler latent element count %" PRIu64 " exceeds max count %u",
        *out_element_count, ID4_SAMPLER_DENOISE_MAX_ELEMENT_COUNT);
  }
  return iree_ok_status();
}

static iree_status_t id4_sampler_program_validate_options(
    const id4_sampler_program_options_t* options,
    const id4_pipeline_program_builder_t* builder,
    uint64_t* out_element_count) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler program options are required");
  }
  IREE_RETURN_IF_ERROR(id4_sampler_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("sampler program")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler program extension structures are not supported");
  }
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler program builder is required");
  }
  return id4_sampler_program_request_element_count(options->request,
                                                   out_element_count);
}

static uint32_t id4_sampler_program_ceil_div_u32(uint32_t dividend,
                                                 uint32_t divisor) {
  return dividend / divisor + (dividend % divisor != 0 ? 1 : 0);
}

static iree_hal_dispatch_config_t id4_sampler_program_make_dispatch_config(
    uint32_t element_count) {
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(
          id4_sampler_program_ceil_div_u32(element_count,
                                           ID4_SAMPLER_WORKGROUP_SIZE_X),
          1, 1);
  dispatch_config.workgroup_size[0] = ID4_SAMPLER_WORKGROUP_SIZE_X;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

static iree_status_t id4_sampler_program_import_tensor(
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

static iree_status_t id4_sampler_program_acquire_tensor(
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

iree_status_t id4_sampler_program_author_denoise_step(
    const id4_sampler_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_sampler_program_validate_options(options, builder, &element_count));

  id4_pipeline_program_tensor_t cond_out =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, id4_sampler_program_cond_out_boundary_name(),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      options->request.latent_shape, &cond_out));
  id4_pipeline_program_tensor_t uncond_out =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, id4_sampler_program_uncond_out_boundary_name(),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      options->request.latent_shape, &uncond_out));
  id4_pipeline_program_tensor_t x_t = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, id4_sampler_program_x_t_boundary_name(),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      options->request.latent_shape, &x_t));
  id4_pipeline_program_tensor_t scalings =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, id4_sampler_program_scalings_boundary_name(),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      id4_pipeline_program_make_shape_rank1(3), &scalings));
  id4_pipeline_program_tensor_t guidance =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, id4_sampler_program_guidance_boundary_name(),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      id4_pipeline_program_make_shape_rank1(3), &guidance));

  id4_pipeline_program_tensor_t guided_pred =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_acquire_tensor(
      builder, id4_sampler_program_guided_pred_tensor_name(),
      options->request.latent_shape, &guided_pred));
  id4_pipeline_program_tensor_t denoised =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, id4_sampler_program_denoised_boundary_name(), 0,
      options->request.latent_shape, &denoised));

  char element_count_buffer[ID4_SAMPLER_CONFIG_VALUE_BUFFER_CAPACITY];
  iree_string_view_t element_count_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_sampler_program_format_u64(
      element_count, element_count_buffer, IREE_ARRAYSIZE(element_count_buffer),
      &element_count_string));
  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      id4_pipeline_make_kernel_config_binding(
          id4_sampler_program_element_count_config_key(), element_count_string),
  };
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(cond_out),
      id4_pipeline_program_read(uncond_out),
      id4_pipeline_program_read(x_t),
      id4_pipeline_program_read(scalings),
      id4_pipeline_program_read(guidance),
      id4_pipeline_program_write(guided_pred),
      id4_pipeline_program_write(denoised),
  };

  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = id4_sampler_program_dispatch_name();
  dispatch_options.kernel = id4_sampler_program_cfg_denoise_kernel_ref();
  dispatch_options.dispatch_config =
      id4_sampler_program_make_dispatch_config((uint32_t)element_count);
  dispatch_options.config_binding_count = IREE_ARRAYSIZE(config_bindings);
  dispatch_options.config_bindings = config_bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_tap_options_t tap_options;
  memset(&tap_options, 0, sizeof(tap_options));
  tap_options.structure_size = sizeof(tap_options);
  tap_options.name = id4_sampler_program_guided_pred_tap_name();
  tap_options.tensor = guided_pred;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_tap(builder, &tap_options));

  id4_pipeline_program_export_options_t export_options;
  memset(&export_options, 0, sizeof(export_options));
  export_options.structure_size = sizeof(export_options);
  export_options.name = id4_sampler_program_denoised_boundary_name();
  export_options.tensor = denoised;
  return id4_pipeline_program_export(builder, &export_options);
}
