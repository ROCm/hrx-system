// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/sampler_program.h"

#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/program.h"

enum {
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
    id4_pipeline_program_shape_t latent_shape, uint64_t max_element_count,
    iree_string_view_t context, uint64_t* out_element_count) {
  *out_element_count = 0;
  if (latent_shape.rank == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s latent shape rank must be nonzero",
                            (int)context.size, context.data);
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_shape_element_count(
      latent_shape, out_element_count));
  if (*out_element_count == 0 || *out_element_count > max_element_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "%.*s latent element count %" PRIu64 " exceeds max count %" PRIu64,
        (int)context.size, context.data, *out_element_count, max_element_count);
  }
  return iree_ok_status();
}

static iree_status_t id4_sampler_program_validate_denoise_options(
    const id4_sampler_denoise_program_options_t* options,
    const id4_pipeline_program_builder_t* builder,
    uint64_t* out_element_count) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler denoise program options are required");
  }
  IREE_RETURN_IF_ERROR(id4_sampler_program_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("sampler denoise program")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler denoise program extension structures are not supported");
  }
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler program builder is required");
  }
  return id4_sampler_program_request_element_count(
      options->request.latent_shape, ID4_SAMPLER_DENOISE_MAX_ELEMENT_COUNT,
      IREE_SV("sampler denoise"), out_element_count);
}

static iree_status_t id4_sampler_program_validate_noise_options(
    const id4_sampler_noise_program_options_t* options,
    const id4_pipeline_program_builder_t* builder,
    uint64_t* out_element_count) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler noise program options are required");
  }
  IREE_RETURN_IF_ERROR(id4_sampler_program_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("sampler noise program")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler noise program extension structures are not supported");
  }
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler program builder is required");
  }
  IREE_RETURN_IF_ERROR(id4_sampler_program_request_element_count(
      options->request.latent_shape, ID4_SAMPLER_NOISE_MAX_ELEMENT_COUNT,
      IREE_SV("sampler noise"), out_element_count));
  const id4_pipeline_program_shape_t latent_shape =
      options->request.latent_shape;
  if (latent_shape.rank != 4 || latent_shape.dims[0] == 0 ||
      latent_shape.dims[1] == 0 || latent_shape.dims[2] < 4 ||
      latent_shape.dims[2] % 4 != 0 || latent_shape.dims[3] != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sampler noise latent shape must be nonempty "
        "[width, height, patch channels, 1] with channels divisible by 4");
  }
  if (options->request.generator_thread_count == 0 ||
      options->request.generator_thread_count >
          iree_align_uint64(*out_element_count, 256) ||
      options->request.generator_thread_count % 256 != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sampler noise generator thread count %" PRIu64
        " must be a nonzero multiple of 256 no larger than the rounded "
        "element count",
        options->request.generator_thread_count);
  }
  return iree_ok_status();
}

static iree_status_t id4_sampler_program_import_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_import_tensor_flags_t flags,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_import_tensor_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.flags = flags;
  options.name = name;
  options.dtype = dtype;
  options.shape = shape;
  return id4_pipeline_program_import_tensor(builder, &options, out_tensor);
}

iree_status_t id4_sampler_program_author_denoise_step(
    const id4_sampler_denoise_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(id4_sampler_program_validate_denoise_options(
      options, builder, &element_count));

  const iree_string_view_t x_next_name = IREE_SV("x_next");

  id4_pipeline_program_tensor_t cond_out =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, IREE_SV("cond_out"), ID4_PIPELINE_PROGRAM_DTYPE_F32,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      options->request.latent_shape, &cond_out));
  id4_pipeline_program_tensor_t uncond_out =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, IREE_SV("uncond_out"), ID4_PIPELINE_PROGRAM_DTYPE_F32,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      options->request.latent_shape, &uncond_out));
  id4_pipeline_program_tensor_t x_t = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, IREE_SV("x_t"), ID4_PIPELINE_PROGRAM_DTYPE_F32,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      options->request.latent_shape, &x_t));
  id4_pipeline_program_tensor_t step = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, IREE_SV("step"), ID4_PIPELINE_PROGRAM_DTYPE_F32,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      id4_pipeline_program_make_shape_rank1(3), &step));
  id4_pipeline_program_tensor_t x_next = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, x_next_name, ID4_PIPELINE_PROGRAM_DTYPE_F32, 0,
      options->request.latent_shape, &x_next));

  char element_count_buffer[ID4_SAMPLER_CONFIG_VALUE_BUFFER_CAPACITY];
  iree_string_view_t element_count_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_sampler_program_format_u64(
      element_count, element_count_buffer, IREE_ARRAYSIZE(element_count_buffer),
      &element_count_string));
  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.sampler.flow_euler.element_count"),
          element_count_string),
  };
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(cond_out),
      id4_pipeline_program_read(uncond_out),
      id4_pipeline_program_read(x_t),
      id4_pipeline_program_read(step),
      id4_pipeline_program_write(x_next),
  };

  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = IREE_SV("sampler.denoise_step.flow_euler");
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("sampler/flow_euler_f32"), IREE_SV("id4_sampler_flow_euler_f32"));
  dispatch_options.config_binding_count = IREE_ARRAYSIZE(config_bindings);
  dispatch_options.config_bindings = config_bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_export_options_t export_options;
  memset(&export_options, 0, sizeof(export_options));
  export_options.structure_size = sizeof(export_options);
  export_options.name = x_next_name;
  export_options.tensor = x_next;
  return id4_pipeline_program_export(builder, &export_options);
}

iree_status_t id4_sampler_program_author_noise(
    const id4_sampler_noise_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(id4_sampler_program_validate_noise_options(
      options, builder, &element_count));

  id4_pipeline_program_tensor_t seed = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, IREE_SV("seed"), ID4_PIPELINE_PROGRAM_DTYPE_I32,
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      id4_pipeline_program_make_shape_rank1(2), &seed));
  id4_pipeline_program_tensor_t x = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_sampler_program_import_tensor(
      builder, IREE_SV("x"), ID4_PIPELINE_PROGRAM_DTYPE_F32, 0,
      options->request.latent_shape, &x));

  char element_count_buffer[ID4_SAMPLER_CONFIG_VALUE_BUFFER_CAPACITY];
  iree_string_view_t element_count_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_sampler_program_format_u64(
      element_count, element_count_buffer, IREE_ARRAYSIZE(element_count_buffer),
      &element_count_string));
  char generator_thread_count_buffer[ID4_SAMPLER_CONFIG_VALUE_BUFFER_CAPACITY];
  iree_string_view_t generator_thread_count_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_sampler_program_format_u64(
      options->request.generator_thread_count, generator_thread_count_buffer,
      IREE_ARRAYSIZE(generator_thread_count_buffer),
      &generator_thread_count_string));
  char width_buffer[ID4_SAMPLER_CONFIG_VALUE_BUFFER_CAPACITY];
  iree_string_view_t width_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_sampler_program_format_u64(
      options->request.latent_shape.dims[0], width_buffer,
      IREE_ARRAYSIZE(width_buffer), &width_string));
  char height_buffer[ID4_SAMPLER_CONFIG_VALUE_BUFFER_CAPACITY];
  iree_string_view_t height_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_sampler_program_format_u64(
      options->request.latent_shape.dims[1], height_buffer,
      IREE_ARRAYSIZE(height_buffer), &height_string));
  char channel_count_buffer[ID4_SAMPLER_CONFIG_VALUE_BUFFER_CAPACITY];
  iree_string_view_t channel_count_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_sampler_program_format_u64(
      options->request.latent_shape.dims[2], channel_count_buffer,
      IREE_ARRAYSIZE(channel_count_buffer), &channel_count_string));
  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.sampler.noise.element_count"), element_count_string),
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.sampler.noise.generator_thread_count"),
          generator_thread_count_string),
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.sampler.noise.width"), width_string),
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.sampler.noise.height"), height_string),
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.sampler.noise.channel_count"), channel_count_string),
  };
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(seed),
      id4_pipeline_program_write(x),
  };

  id4_pipeline_program_dispatch_loom_options_t dispatch_options;
  memset(&dispatch_options, 0, sizeof(dispatch_options));
  dispatch_options.structure_size = sizeof(dispatch_options);
  dispatch_options.name = IREE_SV("sampler.noise.generate");
  dispatch_options.kernel = id4_pipeline_make_kernel_ref(
      IREE_SV("sampler/noise_f32"), IREE_SV("id4_sampler_noise_f32"));
  dispatch_options.config_binding_count = IREE_ARRAYSIZE(config_bindings);
  dispatch_options.config_bindings = config_bindings;
  dispatch_options.binding_count = IREE_ARRAYSIZE(bindings);
  dispatch_options.bindings = bindings;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_export_options_t export_options;
  memset(&export_options, 0, sizeof(export_options));
  export_options.structure_size = sizeof(export_options);
  export_options.name = IREE_SV("x");
  export_options.tensor = x;
  return id4_pipeline_program_export(builder, &export_options);
}
