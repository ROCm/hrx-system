// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_program.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/stages/ideogram4_dit_program_block.h"
#include "iree/hal/command_buffer.h"

enum {
  ID4_IDEOGRAM4_DIT_WORKGROUP_SIZE_X = 256,
  ID4_IDEOGRAM4_DIT_WMMA_WORKGROUP_SIZE_X = 32,
  ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK = 16,
  ID4_IDEOGRAM4_DIT_LINEAR_OUTPUT_ROW_BLOCK = 16,
  ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT = 268435456,
  ID4_IDEOGRAM4_DIT_ATTENTION_QUERY_BLOCK_SIZE = 8,
  ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY = 16,
  ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT = 7,
  ID4_IDEOGRAM4_DIT_LLM_HIDDEN_STATE_LAYER_COUNT = 13,
};

typedef struct id4_ideogram4_dit_program_config_value_t {
  // Loom config key selected by the dispatch authoring site.
  iree_string_view_t key;
  // Unsigned integer config value formatted for loomc.
  uint32_t value;
} id4_ideogram4_dit_program_config_value_t;

static const id4_ideogram4_dit_model_config_t
    id4_ideogram4_dit_program_ideogram4_model_config_value = {
        .layer_count = 34,
        .input_channel_count = 128,
        .hidden_size = 4608,
        .intermediate_size = 12288,
        .attention_head_count = 18,
        .adaln_size = 512,
        .llm_feature_count = 53248,
        .image_indicator_count = 2,
};

static iree_status_t id4_ideogram4_dit_program_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

iree_status_t id4_ideogram4_dit_program_format(char* buffer,
                                               iree_host_size_t buffer_capacity,
                                               iree_string_view_t* out_string,
                                               const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  int length = vsnprintf(buffer, buffer_capacity, format, varargs);
  va_end(varargs);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format Ideogram4 DiT program string");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_format_u32(
    uint32_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(buffer, buffer_capacity, out_string,
                                          "%" PRIu32, value);
}

iree_status_t id4_ideogram4_dit_program_format_layer_parameter(
    uint32_t layer_ordinal, iree_string_view_t suffix, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(
      buffer, buffer_capacity, out_string, "layers.%" PRIu32 ".%.*s",
      layer_ordinal, (int)suffix.size, suffix.data);
}

iree_status_t id4_ideogram4_dit_program_format_branch_layer_name(
    iree_string_view_t branch_name, uint32_t layer_ordinal,
    iree_string_view_t suffix, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(
      buffer, buffer_capacity, out_string, "%.*s.layers.%" PRIu32 ".%.*s",
      (int)branch_name.size, branch_name.data, layer_ordinal, (int)suffix.size,
      suffix.data);
}

bool id4_ideogram4_dit_program_checked_mul_u64(uint64_t lhs, uint64_t rhs,
                                               uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) return false;
  *out_result = lhs * rhs;
  return true;
}

static iree_status_t id4_ideogram4_dit_program_format_child_name(
    iree_string_view_t parent_name, iree_string_view_t child_suffix,
    char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  return id4_ideogram4_dit_program_format(
      buffer, buffer_capacity, out_string, "%.*s.%.*s", (int)parent_name.size,
      parent_name.data, (int)child_suffix.size, child_suffix.data);
}

static bool id4_ideogram4_dit_program_checked_add_u64(uint64_t lhs,
                                                      uint64_t rhs,
                                                      uint64_t* out_result) {
  if (rhs > UINT64_MAX - lhs) return false;
  *out_result = lhs + rhs;
  return true;
}

static uint32_t id4_ideogram4_dit_program_ceil_div_u32(uint32_t dividend,
                                                       uint32_t divisor) {
  return dividend / divisor + (dividend % divisor != 0 ? 1 : 0);
}

static iree_status_t id4_ideogram4_dit_program_validate_model_config(
    const id4_ideogram4_dit_model_config_t* model) {
  if (model->layer_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT layer count must be nonzero");
  }
  if (model->input_channel_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT input channel count must be "
                            "nonzero");
  }
  if ((model->input_channel_count % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT input channel count must be a "
                            "multiple of 4");
  }
  if (model->hidden_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must be nonzero");
  }
  if ((model->hidden_size % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must be a multiple of "
                            "4");
  }
  if (model->intermediate_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT intermediate size must be nonzero");
  }
  if (model->attention_head_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT attention head count must be "
                            "nonzero");
  }
  if ((model->hidden_size % model->attention_head_count) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must divide evenly by "
                            "attention head count");
  }
  if (((model->hidden_size / model->attention_head_count) % 2) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT attention head size must be even");
  }
  if (model->adaln_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT AdaLN size must be nonzero");
  }
  if (model->llm_feature_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must be nonzero");
  }
  if ((model->llm_feature_count % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must be a "
                            "multiple of 4");
  }
  if ((model->llm_feature_count %
       ID4_IDEOGRAM4_DIT_LLM_HIDDEN_STATE_LAYER_COUNT) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must divide "
                            "evenly by Qwen hidden-state layer count");
  }
  if (model->image_indicator_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT image indicator count must be 2");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_image_token_count(
    id4_ideogram4_dit_model_config_t model,
    id4_pipeline_program_shape_t latent_shape, uint32_t* out_token_count) {
  *out_token_count = 0;
  if (latent_shape.rank != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent shape rank must be 4");
  }
  if (latent_shape.dims[2] != model.input_channel_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent channel count %" PRIu64
                            " does not match model channel count %" PRIu32,
                            latent_shape.dims[2], model.input_channel_count);
  }
  if (latent_shape.dims[3] != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent batch dimension must be 1");
  }
  uint64_t token_count = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(
          latent_shape.dims[0], latent_shape.dims[1], &token_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT image token count overflow");
  }
  if (token_count == 0 ||
      token_count > ID4_IDEOGRAM4_DIT_PRELUDE_IMAGE_MAX_TOKEN_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT image token count %" PRIu64 " exceeds max count %u",
        token_count, ID4_IDEOGRAM4_DIT_PRELUDE_IMAGE_MAX_TOKEN_COUNT);
  }
  *out_token_count = (uint32_t)token_count;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_token_counts(
    const id4_ideogram4_dit_program_options_t* options,
    uint32_t* out_image_token_count, uint32_t* out_total_token_count) {
  *out_image_token_count = 0;
  *out_total_token_count = 0;

  uint32_t image_token_count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_image_token_count(
      options->model, options->request.latent_shape, &image_token_count));

  switch (options->request.conditioning_mode) {
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED:
      if (options->request.text_token_count != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT unconditioned requests must not provide text "
            "tokens");
      }
      *out_image_token_count = image_token_count;
      *out_total_token_count = image_token_count;
      return iree_ok_status();
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED: {
      if (options->request.text_token_count == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT conditioned requests require text tokens");
      }
      uint64_t total_token_count = 0;
      if (!id4_ideogram4_dit_program_checked_add_u64(
              options->request.text_token_count, image_token_count,
              &total_token_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "Ideogram4 DiT combined token count overflow");
      }
      if (total_token_count > ID4_IDEOGRAM4_DIT_PRELUDE_MAX_TOKEN_COUNT) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "Ideogram4 DiT combined token count %" PRIu64
                                " exceeds max count %u",
                                total_token_count,
                                ID4_IDEOGRAM4_DIT_PRELUDE_MAX_TOKEN_COUNT);
      }
      *out_image_token_count = image_token_count;
      *out_total_token_count = (uint32_t)total_token_count;
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 DiT conditioning mode %" PRIu32
                              " is invalid",
                              (uint32_t)options->request.conditioning_mode);
  }
}

static iree_status_t id4_ideogram4_dit_program_validate_options(
    const id4_ideogram4_dit_program_options_t* options,
    const id4_pipeline_program_builder_t* builder,
    uint32_t* out_image_token_count, uint32_t* out_total_token_count) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT program options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram4 DiT program")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 DiT program extension structures are not supported");
  }
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT program builder is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_validate_model_config(&options->model));
  return id4_ideogram4_dit_program_token_counts(options, out_image_token_count,
                                                out_total_token_count);
}

static iree_hal_dispatch_config_t
id4_ideogram4_dit_program_make_dispatch_config_with_workgroup_size(
    uint32_t workgroup_count_x, uint32_t workgroup_count_y,
    uint32_t workgroup_count_z, uint32_t workgroup_size_x) {
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(workgroup_count_x, workgroup_count_y,
                                           workgroup_count_z);
  dispatch_config.workgroup_size[0] = workgroup_size_x;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

iree_hal_dispatch_config_t id4_ideogram4_dit_program_make_dispatch_config(
    uint32_t workgroup_count_x, uint32_t workgroup_count_y,
    uint32_t workgroup_count_z) {
  return id4_ideogram4_dit_program_make_dispatch_config_with_workgroup_size(
      workgroup_count_x, workgroup_count_y, workgroup_count_z,
      ID4_IDEOGRAM4_DIT_WORKGROUP_SIZE_X);
}

iree_status_t id4_ideogram4_dit_program_make_element_dispatch_config(
    uint32_t element_count, iree_hal_dispatch_config_t* out_dispatch_config) {
  if (element_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT dispatch element count must be "
                            "nonzero");
  }
  const uint32_t workgroup_count_x = id4_ideogram4_dit_program_ceil_div_u32(
      element_count, ID4_IDEOGRAM4_DIT_WORKGROUP_SIZE_X);
  *out_dispatch_config =
      id4_ideogram4_dit_program_make_dispatch_config(workgroup_count_x, 1, 1);
  return iree_ok_status();
}

static iree_hal_dispatch_config_t
id4_ideogram4_dit_program_make_prelude_image_dispatch_config(
    uint32_t token_count, uint32_t hidden_size) {
  return id4_ideogram4_dit_program_make_dispatch_config(token_count,
                                                        hidden_size, 1);
}

static iree_status_t id4_ideogram4_dit_program_make_config_binding(
    iree_string_view_t key, uint32_t value, char* value_buffer,
    iree_host_size_t value_buffer_capacity,
    id4_pipeline_kernel_config_binding_t* out_binding) {
  iree_string_view_t value_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_u32(
      value, value_buffer, value_buffer_capacity, &value_string));
  *out_binding = id4_pipeline_make_kernel_config_binding(key, value_string);
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_make_config_bindings(
    iree_host_size_t config_value_count,
    const id4_ideogram4_dit_program_config_value_t* config_values,
    char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                      [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY],
    id4_pipeline_kernel_config_binding_t* out_bindings) {
  if (config_value_count > ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT config binding count %" PRIhsz " exceeds max count %u",
        config_value_count, ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT);
  }
  for (iree_host_size_t i = 0; i < config_value_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_binding(
        config_values[i].key, config_values[i].value, value_buffers[i],
        ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY, &out_bindings[i]));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_import_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_import_tensor_flags_t flags,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_import_tensor_options_t options = {
      .structure_size = sizeof(options),
      .flags = flags,
      .name = name,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_import_tensor(builder, &options, out_tensor);
}

iree_status_t id4_ideogram4_dit_program_parameter(
    id4_pipeline_program_builder_t* builder, iree_string_view_t key,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_parameter_options_t options = {
      .structure_size = sizeof(options),
      .key = key,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_parameter(builder, &options, out_tensor);
}

iree_status_t id4_ideogram4_dit_program_layer_parameter(
    id4_pipeline_program_builder_t* builder, uint32_t layer_ordinal,
    iree_string_view_t suffix, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  char key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_layer_parameter(
      layer_ordinal, suffix, key_buffer, IREE_ARRAYSIZE(key_buffer), &key));
  return id4_ideogram4_dit_program_parameter(builder, key, dtype, shape,
                                             out_tensor);
}

iree_status_t id4_ideogram4_dit_program_acquire_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    id4_pipeline_program_tensor_t* out_tensor) {
  id4_pipeline_program_acquire_tensor_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .dtype = dtype,
      .shape = shape,
  };
  return id4_pipeline_program_acquire_tensor(builder, &options, out_tensor);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_loom(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    iree_string_view_t module_path, iree_string_view_t function_name,
    iree_hal_dispatch_config_t dispatch_config,
    iree_host_size_t config_binding_count,
    const id4_pipeline_kernel_config_binding_t* config_bindings,
    iree_host_size_t binding_count,
    const id4_pipeline_program_dispatch_binding_t* bindings) {
  id4_pipeline_program_dispatch_loom_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .kernel = id4_pipeline_make_kernel_ref(module_path, function_name),
      .dispatch_config = dispatch_config,
      .config_binding_count = config_binding_count,
      .config_bindings = config_bindings,
      .binding_count = binding_count,
      .bindings = bindings,
  };
  return id4_pipeline_program_dispatch_loom(builder, &options);
}

iree_status_t id4_ideogram4_dit_program_barrier(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name) {
  id4_pipeline_program_barrier_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
  };
  return id4_pipeline_program_barrier(builder, &options);
}

iree_status_t id4_ideogram4_dit_program_tap(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_tensor_t tensor) {
  id4_pipeline_program_tap_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .tensor = tensor,
  };
  return id4_pipeline_program_tap(builder, &options);
}

static iree_status_t id4_ideogram4_dit_program_export(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_tensor_t tensor) {
  id4_pipeline_program_export_options_t options = {
      .structure_size = sizeof(options),
      .name = name,
      .tensor = tensor,
  };
  return id4_pipeline_program_export(builder, &options);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_prelude_image(
    id4_pipeline_program_builder_t* builder, uint32_t image_token_count,
    uint32_t total_token_count, uint32_t output_token_offset,
    uint32_t image_width, uint32_t image_height, uint32_t input_channel_count,
    uint32_t hidden_size, id4_pipeline_program_tensor_t image_tokens,
    id4_pipeline_program_tensor_t input_proj_weight,
    id4_pipeline_program_tensor_t input_proj_bias,
    id4_pipeline_program_tensor_t image_indicator,
    id4_pipeline_program_tensor_t image_indicator_embedding,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.prelude_image.token_count"), image_token_count},
      {IREE_SV("id4.ideogram4.prelude_image.output_token_count"),
       total_token_count},
      {IREE_SV("id4.ideogram4.prelude_image.output_token_offset"),
       output_token_offset},
      {IREE_SV("id4.ideogram4.prelude_image.image_width"), image_width},
      {IREE_SV("id4.ideogram4.prelude_image.image_height"), image_height},
      {IREE_SV("id4.ideogram4.prelude_image.input_channel_count"),
       input_channel_count},
      {IREE_SV("id4.ideogram4.prelude_image.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(image_tokens),
      id4_pipeline_program_read(input_proj_weight),
      id4_pipeline_program_read(input_proj_bias),
      id4_pipeline_program_read(image_indicator),
      id4_pipeline_program_read(image_indicator_embedding),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, IREE_SV("ideogram4.prelude.image_projection"),
      IREE_SV("ideogram4/prelude_image_bf16_f32"),
      IREE_SV("id4_ideogram4_prelude_image_bf16_f32"),
      id4_ideogram4_dit_program_make_prelude_image_dispatch_config(
          image_token_count, hidden_size),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_condition_rmsnorm(
    id4_pipeline_program_builder_t* builder, uint32_t text_token_count,
    uint32_t llm_feature_count, id4_pipeline_program_tensor_t condition,
    id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.condition_rmsnorm.token_count"),
       text_token_count},
      {IREE_SV("id4.ideogram4.condition_rmsnorm.feature_count"),
       llm_feature_count},
      {IREE_SV("id4.ideogram4.condition_rmsnorm.hidden_state_layer_count"),
       ID4_IDEOGRAM4_DIT_LLM_HIDDEN_STATE_LAYER_COUNT},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(condition),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
      IREE_SV("ideogram4/condition_rmsnorm_f32"),
      IREE_SV("id4_ideogram4_condition_rmsnorm_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(text_token_count, 1, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_condition_project(
    id4_pipeline_program_builder_t* builder, uint32_t text_token_count,
    uint32_t total_token_count, uint32_t llm_feature_count,
    uint32_t hidden_size, id4_pipeline_program_tensor_t condition,
    id4_pipeline_program_tensor_t weight, id4_pipeline_program_tensor_t bias,
    id4_pipeline_program_tensor_t image_indicator,
    id4_pipeline_program_tensor_t image_indicator_embedding,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.condition_project.text_token_count"),
       text_token_count},
      {IREE_SV("id4.ideogram4.condition_project.total_token_count"),
       total_token_count},
      {IREE_SV("id4.ideogram4.condition_project.feature_count"),
       llm_feature_count},
      {IREE_SV("id4.ideogram4.condition_project.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(condition),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_read(image_indicator),
      id4_pipeline_program_read(image_indicator_embedding),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, IREE_SV("ideogram4.cond.prelude.llm_cond_proj"),
      IREE_SV("ideogram4/condition_project_bf16_f32"),
      IREE_SV("id4_ideogram4_condition_project_bf16_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(
          id4_ideogram4_dit_program_ceil_div_u32(
              text_token_count, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK),
          id4_ideogram4_dit_program_ceil_div_u32(
              hidden_size, ID4_IDEOGRAM4_DIT_LINEAR_OUTPUT_ROW_BLOCK),
          1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_adaln_split(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t hidden_size, id4_pipeline_program_tensor_t raw_modulation,
    id4_pipeline_program_tensor_t scale_msa,
    id4_pipeline_program_tensor_t gate_msa,
    id4_pipeline_program_tensor_t scale_mlp,
    id4_pipeline_program_tensor_t gate_mlp) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.adaln_split.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(raw_modulation),
      id4_pipeline_program_write(scale_msa),
      id4_pipeline_program_write(gate_msa),
      id4_pipeline_program_write(scale_mlp),
      id4_pipeline_program_write(gate_mlp),
  };
  iree_hal_dispatch_config_t dispatch_config;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_element_dispatch_config(
      hidden_size, &dispatch_config));
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/adaln_split_f32"),
      IREE_SV("id4_ideogram4_adaln_split_f32"), dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_modulated_rmsnorm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t scale, id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.modulated_rmsnorm.token_count"), token_count},
      {IREE_SV("id4.ideogram4.modulated_rmsnorm.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/modulated_rmsnorm_f32"),
      IREE_SV("id4_ideogram4_modulated_rmsnorm_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(token_count, 1, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_rmsnorm_gated_residual(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t gate, id4_pipeline_program_tensor_t residual,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.rmsnorm_gated_residual.token_count"),
       token_count},
      {IREE_SV("id4.ideogram4.rmsnorm_gated_residual.hidden_size"),
       hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),   id4_pipeline_program_read(weight),
      id4_pipeline_program_read(gate),    id4_pipeline_program_read(residual),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/rmsnorm_gated_residual_f32"),
      IREE_SV("id4_ideogram4_rmsnorm_gated_residual_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(token_count, 1, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_mlp_gate_up_silu(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t intermediate_size,
    id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t gate_weight,
    id4_pipeline_program_tensor_t up_weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.token_count"), token_count},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.input_size"), input_size},
      {IREE_SV("id4.ideogram4.mlp_gate_up_silu.intermediate_size"),
       intermediate_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(gate_weight),
      id4_pipeline_program_read(up_weight),
      id4_pipeline_program_write(output),
  };
  const uint32_t token_tile_count = id4_ideogram4_dit_program_ceil_div_u32(
      token_count, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  const uint32_t intermediate_tile_count =
      id4_ideogram4_dit_program_ceil_div_u32(
          intermediate_size, ID4_IDEOGRAM4_DIT_LINEAR_OUTPUT_ROW_BLOCK);
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/mlp_gate_up_silu_f32"),
      IREE_SV("id4_ideogram4_mlp_gate_up_silu_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(
          token_tile_count, intermediate_tile_count, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_linear_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  if (token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 linear token count must be nonzero");
  }
  if ((input_size % ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 linear input size %" PRIu32
                            " must be a multiple of %u",
                            input_size, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  }
  if ((output_size % ID4_IDEOGRAM4_DIT_LINEAR_OUTPUT_ROW_BLOCK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 linear output size %" PRIu32 " must be a multiple of %u",
        output_size, ID4_IDEOGRAM4_DIT_LINEAR_OUTPUT_ROW_BLOCK);
  }
  uint64_t input_element_count_u64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(token_count, input_size,
                                                 &input_element_count_u64)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 linear input element count overflow");
  }
  if (input_element_count_u64 >
      ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 linear input element count %" PRIu64 " exceeds max %u",
        input_element_count_u64,
        ID4_IDEOGRAM4_DIT_LINEAR_INPUT_PACK_MAX_ELEMENT_COUNT);
  }
  const uint32_t input_element_count = (uint32_t)input_element_count_u64;
  const uint32_t body_token_count =
      token_count - (token_count % ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
  const uint32_t tail_token_count = token_count - body_token_count;

  char packed_input_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char pack_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_pack_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t packed_input_name = iree_string_view_empty();
  iree_string_view_t pack_dispatch_name = iree_string_view_empty();
  iree_string_view_t after_pack_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("input_bf16"), packed_input_name_buffer,
      IREE_ARRAYSIZE(packed_input_name_buffer), &packed_input_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("input_pack"), pack_dispatch_name_buffer,
      IREE_ARRAYSIZE(pack_dispatch_name_buffer), &pack_dispatch_name));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
      name, IREE_SV("after_input_pack"), after_pack_name_buffer,
      IREE_ARRAYSIZE(after_pack_name_buffer), &after_pack_name));

  id4_pipeline_program_tensor_t packed_input =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, packed_input_name, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(token_count, input_size),
      &packed_input));

  const id4_ideogram4_dit_program_config_value_t pack_config_values[] = {
      {IREE_SV("id4.ideogram4.linear_input_pack.token_count"), token_count},
      {IREE_SV("id4.ideogram4.linear_input_pack.input_size"), input_size},
      {IREE_SV("id4.ideogram4.linear_input_pack.element_count"),
       input_element_count},
  };
  char pack_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                         [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      pack_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(pack_config_values), pack_config_values,
      pack_value_buffers, pack_config_bindings));
  id4_pipeline_program_dispatch_binding_t pack_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(packed_input),
  };
  iree_hal_dispatch_config_t pack_dispatch_config;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_element_dispatch_config(
      input_element_count, &pack_dispatch_config));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
      builder, pack_dispatch_name,
      IREE_SV("ideogram4/linear_input_pack_f32_bf16"),
      IREE_SV("id4_ideogram4_linear_input_pack_f32_bf16"), pack_dispatch_config,
      IREE_ARRAYSIZE(pack_config_values), pack_config_bindings,
      IREE_ARRAYSIZE(pack_bindings), pack_bindings));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_pack_name));

  if (body_token_count != 0) {
    char body_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t body_dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, IREE_SV("wmma"), body_dispatch_name_buffer,
        IREE_ARRAYSIZE(body_dispatch_name_buffer), &body_dispatch_name));
    const id4_ideogram4_dit_program_config_value_t body_config_values[] = {
        {IREE_SV("id4.ideogram4.linear_wmma.token_count"), token_count},
        {IREE_SV("id4.ideogram4.linear_wmma.dispatch_token_count"),
         body_token_count},
        {IREE_SV("id4.ideogram4.linear_wmma.input_size"), input_size},
        {IREE_SV("id4.ideogram4.linear_wmma.output_size"), output_size},
    };
    char body_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        body_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
        IREE_ARRAYSIZE(body_config_values), body_config_values,
        body_value_buffers, body_config_bindings));
    id4_pipeline_program_dispatch_binding_t body_bindings[] = {
        id4_pipeline_program_read(packed_input),
        id4_pipeline_program_read(weight),
        id4_pipeline_program_write(output),
    };
    const uint32_t token_tile_count =
        body_token_count / ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK;
    const uint32_t output_row_tile_count =
        output_size / ID4_IDEOGRAM4_DIT_LINEAR_OUTPUT_ROW_BLOCK;
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
        builder, body_dispatch_name, IREE_SV("ideogram4/linear_bf16_f32_wmma"),
        IREE_SV("id4_ideogram4_linear_bf16_f32_wmma"),
        id4_ideogram4_dit_program_make_dispatch_config_with_workgroup_size(
            token_tile_count, output_row_tile_count, 1,
            ID4_IDEOGRAM4_DIT_WMMA_WORKGROUP_SIZE_X),
        IREE_ARRAYSIZE(body_config_values), body_config_bindings,
        IREE_ARRAYSIZE(body_bindings), body_bindings));
  }

  if (tail_token_count != 0) {
    if (body_token_count != 0) {
      char after_body_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
      iree_string_view_t after_body_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
          name, IREE_SV("after_wmma"), after_body_name_buffer,
          IREE_ARRAYSIZE(after_body_name_buffer), &after_body_name));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_dit_program_barrier(builder, after_body_name));
    }
    char tail_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
    iree_string_view_t tail_dispatch_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_child_name(
        name, IREE_SV("tail"), tail_dispatch_name_buffer,
        IREE_ARRAYSIZE(tail_dispatch_name_buffer), &tail_dispatch_name));
    const id4_ideogram4_dit_program_config_value_t tail_config_values[] = {
        {IREE_SV("id4.ideogram4.linear_tail.token_count"), token_count},
        {IREE_SV("id4.ideogram4.linear_tail.token_offset"), body_token_count},
        {IREE_SV("id4.ideogram4.linear_tail.dispatch_token_count"),
         tail_token_count},
        {IREE_SV("id4.ideogram4.linear_tail.input_size"), input_size},
        {IREE_SV("id4.ideogram4.linear_tail.output_size"), output_size},
    };
    char tail_value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                           [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
    id4_pipeline_kernel_config_binding_t
        tail_config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
        IREE_ARRAYSIZE(tail_config_values), tail_config_values,
        tail_value_buffers, tail_config_bindings));
    id4_pipeline_program_dispatch_binding_t tail_bindings[] = {
        id4_pipeline_program_read(packed_input),
        id4_pipeline_program_read(weight),
        id4_pipeline_program_write(output),
    };
    const uint32_t tail_token_tile_count =
        id4_ideogram4_dit_program_ceil_div_u32(
            tail_token_count, ID4_IDEOGRAM4_DIT_LINEAR_TOKEN_BLOCK);
    const uint32_t output_row_tile_count =
        id4_ideogram4_dit_program_ceil_div_u32(
            output_size, ID4_IDEOGRAM4_DIT_LINEAR_OUTPUT_ROW_BLOCK);
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_loom(
        builder, tail_dispatch_name, IREE_SV("ideogram4/linear_bf16_f32_tail"),
        IREE_SV("id4_ideogram4_linear_bf16_f32_tail"),
        id4_ideogram4_dit_program_make_dispatch_config(
            tail_token_tile_count, output_row_tile_count, 1),
        IREE_ARRAYSIZE(tail_config_values), tail_config_bindings,
        IREE_ARRAYSIZE(tail_bindings), tail_bindings));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_program_dispatch_silu(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t element_count, id4_pipeline_program_tensor_t input,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.silu.element_count"), element_count},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(output),
  };
  iree_hal_dispatch_config_t dispatch_config;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_element_dispatch_config(
      element_count, &dispatch_config));
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/silu_f32"),
      IREE_SV("id4_ideogram4_silu_f32"), dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_modulated_layernorm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t scale,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.modulated_layernorm.token_count"), token_count},
      {IREE_SV("id4.ideogram4.modulated_layernorm.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/modulated_layernorm_f32"),
      IREE_SV("id4_ideogram4_modulated_layernorm_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(token_count, 1, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_linear_bias_bf16_f32(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t token_offset, uint32_t dispatch_token_count,
    uint32_t input_size, uint32_t output_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t bias, id4_pipeline_program_tensor_t output) {
  if (token_offset > token_count ||
      dispatch_token_count > token_count - token_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT linear-bias dispatch token range overflow");
  }
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.linear_bias.token_count"), token_count},
      {IREE_SV("id4.ideogram4.linear_bias.token_offset"), token_offset},
      {IREE_SV("id4.ideogram4.linear_bias.dispatch_token_count"),
       dispatch_token_count},
      {IREE_SV("id4.ideogram4.linear_bias.input_size"), input_size},
      {IREE_SV("id4.ideogram4.linear_bias.output_size"), output_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/linear_bias_bf16_f32"),
      IREE_SV("id4_ideogram4_linear_bias_bf16_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(dispatch_token_count,
                                                     output_size, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_unpatchify_scale(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t width, uint32_t height, uint32_t input_channel_count,
    uint32_t total_token_count, uint32_t output_token_offset,
    id4_pipeline_program_tensor_t projected,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.unpatchify_scale.width"), width},
      {IREE_SV("id4.ideogram4.unpatchify_scale.height"), height},
      {IREE_SV("id4.ideogram4.unpatchify_scale.input_channel_count"),
       input_channel_count},
      {IREE_SV("id4.ideogram4.unpatchify_scale.total_token_count"),
       total_token_count},
      {IREE_SV("id4.ideogram4.unpatchify_scale.output_token_offset"),
       output_token_offset},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(projected),
      id4_pipeline_program_write(output),
  };
  uint64_t spatial_count64 = 0;
  uint64_t element_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(width, height,
                                                 &spatial_count64) ||
      !id4_ideogram4_dit_program_checked_mul_u64(
          spatial_count64, input_channel_count, &element_count64) ||
      element_count64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT unpatchify output size overflow");
  }
  iree_hal_dispatch_config_t dispatch_config;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_element_dispatch_config(
      (uint32_t)element_count64, &dispatch_config));
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/unpatchify_scale_f32"),
      IREE_SV("id4_ideogram4_unpatchify_scale_f32"), dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_qkv_split(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t hidden_size,
    id4_pipeline_program_tensor_t qkv, id4_pipeline_program_tensor_t query,
    id4_pipeline_program_tensor_t key, id4_pipeline_program_tensor_t value) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.qkv_split.token_count"), token_count},
      {IREE_SV("id4.ideogram4.qkv_split.hidden_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(qkv),
      id4_pipeline_program_write(query),
      id4_pipeline_program_write(key),
      id4_pipeline_program_write(value),
  };
  iree_hal_dispatch_config_t dispatch_config;
  uint64_t element_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(token_count, hidden_size,
                                                 &element_count64) ||
      element_count64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT QKV split size overflow");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_element_dispatch_config(
      (uint32_t)element_count64, &dispatch_config));
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/qkv_split_f32"),
      IREE_SV("id4_ideogram4_qkv_split_f32"), dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_qkv_norm_rotary(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t qkv,
    id4_pipeline_program_tensor_t norm_q_weight,
    id4_pipeline_program_tensor_t norm_k_weight,
    id4_pipeline_program_tensor_t position_embedding,
    id4_pipeline_program_tensor_t rotated_query,
    id4_pipeline_program_tensor_t rotated_key,
    id4_pipeline_program_tensor_t value) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.token_count"), token_count},
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.qkv_norm_rotary.head_size"), head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(qkv),
      id4_pipeline_program_read(norm_q_weight),
      id4_pipeline_program_read(norm_k_weight),
      id4_pipeline_program_read(position_embedding),
      id4_pipeline_program_write(rotated_query),
      id4_pipeline_program_write(rotated_key),
      id4_pipeline_program_write(value),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/qkv_norm_rotary_f32"),
      IREE_SV("id4_ideogram4_qkv_norm_rotary_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(token_count,
                                                     attention_head_count, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_head_rmsnorm(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t input, id4_pipeline_program_tensor_t weight,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.head_rmsnorm.token_count"), token_count},
      {IREE_SV("id4.ideogram4.head_rmsnorm.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.head_rmsnorm.head_size"), head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/head_rmsnorm_f32"),
      IREE_SV("id4_ideogram4_head_rmsnorm_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(token_count,
                                                     attention_head_count, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_rotary_apply(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t position_embedding,
    id4_pipeline_program_tensor_t rotated_query,
    id4_pipeline_program_tensor_t rotated_key) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.rotary_apply.token_count"), token_count},
      {IREE_SV("id4.ideogram4.rotary_apply.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.rotary_apply.head_size"), head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(position_embedding),
      id4_pipeline_program_write(rotated_query),
      id4_pipeline_program_write(rotated_key),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/rotary_apply_f32"),
      IREE_SV("id4_ideogram4_rotary_apply_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(token_count,
                                                     attention_head_count, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dispatch_attention(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t token_count, uint32_t attention_head_count, uint32_t head_size,
    id4_pipeline_program_tensor_t query, id4_pipeline_program_tensor_t key,
    id4_pipeline_program_tensor_t value, id4_pipeline_program_tensor_t output) {
  if (head_size > ID4_IDEOGRAM4_DIT_WORKGROUP_SIZE_X) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram4 DiT attention head size %u exceeds workgroup size %u",
        head_size, ID4_IDEOGRAM4_DIT_WORKGROUP_SIZE_X);
  }

  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.attention_query_block8_rounded.token_count"),
       token_count},
      {IREE_SV(
           "id4.ideogram4.attention_query_block8_rounded.attention_head_count"),
       attention_head_count},
      {IREE_SV("id4.ideogram4.attention_query_block8_rounded.head_size"),
       head_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(query),
      id4_pipeline_program_read(key),
      id4_pipeline_program_read(value),
      id4_pipeline_program_write(output),
  };
  const uint32_t query_tile_count =
      (token_count + ID4_IDEOGRAM4_DIT_ATTENTION_QUERY_BLOCK_SIZE - 1) /
      ID4_IDEOGRAM4_DIT_ATTENTION_QUERY_BLOCK_SIZE;
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/attention_query_block8_rounded_f32"),
      IREE_SV("id4_ideogram4_attention_query_block8_rounded_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(query_tile_count,
                                                     attention_head_count, 1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_dispatch_timestep_embedding(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint32_t hidden_size, id4_pipeline_program_tensor_t timestep,
    id4_pipeline_program_tensor_t output) {
  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.timestep_embedding.embedding_size"), hidden_size},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(timestep),
      id4_pipeline_program_write(output),
  };
  iree_hal_dispatch_config_t dispatch_config;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_element_dispatch_config(
      hidden_size, &dispatch_config));
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, name, IREE_SV("ideogram4/timestep_embedding_f32"),
      IREE_SV("id4_ideogram4_timestep_embedding_f32"), dispatch_config,
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

iree_status_t id4_ideogram4_dit_program_dense_bf16_f32(
    id4_pipeline_program_builder_t* builder,
    const id4_ideogram4_dit_program_dense_options_t* options,
    id4_pipeline_program_tensor_t* out_output) {
  if ((options->input_size % 4) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 DiT dense input size must be a multiple of 4");
  }
  if (options->output_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT dense output size must be nonzero");
  }
  if (options->activation_kind != ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY &&
      options->activation_kind != ID4_IDEOGRAM4_DIT_ACTIVATION_SILU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT dense activation kind %" PRIu32
                            " is invalid",
                            options->activation_kind);
  }

  char weight_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char bias_key_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t weight_key = iree_string_view_empty();
  iree_string_view_t bias_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      weight_key_buffer, IREE_ARRAYSIZE(weight_key_buffer), &weight_key,
      "%.*s.weight", (int)options->parameter_prefix.size,
      options->parameter_prefix.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      bias_key_buffer, IREE_ARRAYSIZE(bias_key_buffer), &bias_key, "%.*s.bias",
      (int)options->parameter_prefix.size, options->parameter_prefix.data));

  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
      builder, weight_key, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options->output_size,
                                            options->input_size),
      &weight));
  id4_pipeline_program_tensor_t bias = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
      builder, bias_key, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(options->output_size), &bias));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, options->output_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(options->output_size), out_output));

  const id4_ideogram4_dit_program_config_value_t config_values[] = {
      {IREE_SV("id4.ideogram4.dense_bias.input_size"), options->input_size},
      {IREE_SV("id4.ideogram4.dense_bias.output_size"), options->output_size},
      {IREE_SV("id4.ideogram4.dense_bias.activation_kind"),
       options->activation_kind},
  };
  char value_buffers[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT]
                    [ID4_IDEOGRAM4_DIT_CONFIG_VALUE_BUFFER_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_IDEOGRAM4_DIT_MAX_KERNEL_CONFIG_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_make_config_bindings(
      IREE_ARRAYSIZE(config_values), config_values, value_buffers,
      config_bindings));
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(options->input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_read(bias),
      id4_pipeline_program_write(*out_output),
  };
  return id4_ideogram4_dit_program_dispatch_loom(
      builder, options->operation_name,
      IREE_SV("ideogram4/dense_bias_bf16_f32"),
      IREE_SV("id4_ideogram4_dense_bias_bf16_f32"),
      id4_ideogram4_dit_program_make_dispatch_config(options->output_size, 1,
                                                     1),
      IREE_ARRAYSIZE(config_values), config_bindings, IREE_ARRAYSIZE(bindings),
      bindings);
}

static iree_status_t id4_ideogram4_dit_program_author_final_output(
    const id4_ideogram4_dit_program_options_t* options,
    id4_pipeline_program_builder_t* builder, iree_string_view_t branch_name,
    uint32_t total_token_count, id4_pipeline_program_tensor_t hidden,
    id4_pipeline_program_tensor_t adaln_input) {
  const uint32_t input_channel_count = options->model.input_channel_count;
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t adaln_size = options->model.adaln_size;
  const uint32_t text_token_count = options->request.text_token_count;
  const uint32_t width = (uint32_t)options->request.latent_shape.dims[0];
  const uint32_t height = (uint32_t)options->request.latent_shape.dims[1];
  uint64_t image_token_count64 = 0;
  if (!id4_ideogram4_dit_program_checked_mul_u64(width, height,
                                                 &image_token_count64) ||
      image_token_count64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram4 DiT final image token count overflow");
  }
  const uint32_t image_token_count = (uint32_t)image_token_count64;

  char final_silu_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_final_silu_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t final_silu_name = iree_string_view_empty();
  iree_string_view_t after_final_silu_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_silu_name_buffer, IREE_ARRAYSIZE(final_silu_name_buffer),
      &final_silu_name, "%.*s.final.adaln_input_silu", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_final_silu_name_buffer,
      IREE_ARRAYSIZE(after_final_silu_name_buffer), &after_final_silu_name,
      "%.*s.final.after_adaln_input_silu", (int)branch_name.size,
      branch_name.data));

  id4_pipeline_program_tensor_t final_silu =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, final_silu_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(adaln_size), &final_silu));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_silu(
      builder, final_silu_name, adaln_size, adaln_input, final_silu));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_final_silu_name));

  char final_scale_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_final_scale_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t final_scale_name = iree_string_view_empty();
  iree_string_view_t after_final_scale_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_scale_name_buffer, IREE_ARRAYSIZE(final_scale_name_buffer),
      &final_scale_name, "%.*s.final.scale", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_final_scale_name_buffer,
      IREE_ARRAYSIZE(after_final_scale_name_buffer), &after_final_scale_name,
      "%.*s.final.after_scale", (int)branch_name.size, branch_name.data));

  id4_pipeline_program_tensor_t final_scale =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t final_scale_options = {
      .operation_name = final_scale_name,
      .output_name = final_scale_name,
      .parameter_prefix = IREE_SV("final_layer.adaln_modulation"),
      .input = final_silu,
      .input_size = adaln_size,
      .output_size = hidden_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY,
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16_f32(
      builder, &final_scale_options, &final_scale));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_final_scale_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, final_scale_name, final_scale));

  char final_norm_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char
      final_norm_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_final_norm_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t final_norm_name = iree_string_view_empty();
  iree_string_view_t final_norm_dispatch_name = iree_string_view_empty();
  iree_string_view_t after_final_norm_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_norm_name_buffer, IREE_ARRAYSIZE(final_norm_name_buffer),
      &final_norm_name, "%.*s.final.normalized", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      final_norm_dispatch_name_buffer,
      IREE_ARRAYSIZE(final_norm_dispatch_name_buffer),
      &final_norm_dispatch_name, "%.*s.final.modulated_layernorm",
      (int)branch_name.size, branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_final_norm_name_buffer,
      IREE_ARRAYSIZE(after_final_norm_name_buffer), &after_final_norm_name,
      "%.*s.final.after_layernorm", (int)branch_name.size, branch_name.data));

  id4_pipeline_program_tensor_t final_norm =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, final_norm_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &final_norm));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_modulated_layernorm(
      builder, final_norm_dispatch_name, total_token_count, hidden_size, hidden,
      final_scale, final_norm));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_final_norm_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, final_norm_name, final_norm));

  id4_pipeline_program_tensor_t final_linear_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
      builder, IREE_SV("final_layer.linear.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(input_channel_count, hidden_size),
      &final_linear_weight));
  id4_pipeline_program_tensor_t final_linear_bias =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
      builder, IREE_SV("final_layer.linear.bias"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(input_channel_count),
      &final_linear_bias));

  char projected_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char projected_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_projected_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t projected_name = iree_string_view_empty();
  iree_string_view_t projected_dispatch_name = iree_string_view_empty();
  iree_string_view_t after_projected_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      projected_name_buffer, IREE_ARRAYSIZE(projected_name_buffer),
      &projected_name, "%.*s.final.projected", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      projected_dispatch_name_buffer,
      IREE_ARRAYSIZE(projected_dispatch_name_buffer), &projected_dispatch_name,
      "%.*s.final.linear", (int)branch_name.size, branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_projected_name_buffer, IREE_ARRAYSIZE(after_projected_name_buffer),
      &after_projected_name, "%.*s.final.after_linear", (int)branch_name.size,
      branch_name.data));

  id4_pipeline_program_tensor_t projected =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, projected_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(input_channel_count,
                                            image_token_count),
      &projected));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_linear_bias_bf16_f32(
      builder, projected_dispatch_name, total_token_count, text_token_count,
      image_token_count, hidden_size, input_channel_count, final_norm,
      final_linear_weight, final_linear_bias, projected));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_projected_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, projected_name, projected));

  id4_pipeline_program_tensor_t velocity =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("velocity"), 0, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      options->request.latent_shape, &velocity));

  char velocity_dispatch_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char after_velocity_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  char velocity_capture_name_buffer[ID4_IDEOGRAM4_DIT_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t velocity_dispatch_name = iree_string_view_empty();
  iree_string_view_t after_velocity_name = iree_string_view_empty();
  iree_string_view_t velocity_capture_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      velocity_dispatch_name_buffer,
      IREE_ARRAYSIZE(velocity_dispatch_name_buffer), &velocity_dispatch_name,
      "%.*s.final.unpatchify_scale", (int)branch_name.size, branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      after_velocity_name_buffer, IREE_ARRAYSIZE(after_velocity_name_buffer),
      &after_velocity_name, "%.*s.final.after_velocity", (int)branch_name.size,
      branch_name.data));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format(
      velocity_capture_name_buffer,
      IREE_ARRAYSIZE(velocity_capture_name_buffer), &velocity_capture_name,
      "%.*s.output.velocity", (int)branch_name.size, branch_name.data));

  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_unpatchify_scale(
      builder, velocity_dispatch_name, width, height, input_channel_count,
      image_token_count, 0, projected, velocity));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_velocity_name));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, velocity_capture_name, velocity));
  return id4_ideogram4_dit_program_export(builder, IREE_SV("velocity"),
                                          velocity);
}

iree_status_t id4_ideogram4_dit_program_author_forward(
    const id4_ideogram4_dit_program_options_t* options,
    id4_pipeline_program_builder_t* builder) {
  uint32_t image_token_count = 0;
  uint32_t total_token_count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_validate_options(
      options, builder, &image_token_count, &total_token_count));
  const uint32_t input_channel_count = options->model.input_channel_count;
  const uint32_t hidden_size = options->model.hidden_size;
  const uint32_t intermediate_size = options->model.intermediate_size;
  const uint32_t attention_head_count = options->model.attention_head_count;
  const uint32_t head_size = hidden_size / attention_head_count;
  const uint32_t adaln_size = options->model.adaln_size;
  const uint32_t llm_feature_count = options->model.llm_feature_count;
  const uint32_t text_token_count = options->request.text_token_count;
  const bool is_conditioned = options->request.conditioning_mode ==
                              ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  const iree_string_view_t branch_name =
      is_conditioned ? IREE_SV("ideogram4.cond") : IREE_SV("ideogram4.uncond");
  const iree_string_view_t prelude_hidden_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.hidden")
                     : IREE_SV("ideogram4.uncond.prelude.hidden");
  const iree_string_view_t timestep_embedding_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.timestep_embedding")
                     : IREE_SV("ideogram4.uncond.prelude.timestep_embedding");
  const iree_string_view_t timestep_mlp_in_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.t_embedding.mlp_in")
                     : IREE_SV("ideogram4.uncond.prelude.t_embedding.mlp_in");
  const iree_string_view_t timestep_mlp_out_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.t_embedding.mlp_out")
                     : IREE_SV("ideogram4.uncond.prelude.t_embedding.mlp_out");
  const iree_string_view_t adaln_input_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.adaln_input")
                     : IREE_SV("ideogram4.uncond.prelude.adaln_input");
  const iree_string_view_t after_inputs_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.after_inputs")
                     : IREE_SV("ideogram4.uncond.prelude.after_inputs");
  const iree_string_view_t after_timestep_mlp_in_name =
      is_conditioned
          ? IREE_SV("ideogram4.cond.prelude.after_t_embedding_mlp_in")
          : IREE_SV("ideogram4.uncond.prelude.after_t_embedding_mlp_in");
  const iree_string_view_t after_timestep_mlp_out_name =
      is_conditioned
          ? IREE_SV("ideogram4.cond.prelude.after_t_embedding_mlp_out")
          : IREE_SV("ideogram4.uncond.prelude.after_t_embedding_mlp_out");
  const iree_string_view_t after_adaln_name =
      is_conditioned ? IREE_SV("ideogram4.cond.prelude.after_adaln_proj")
                     : IREE_SV("ideogram4.uncond.prelude.after_adaln_proj");

  id4_pipeline_program_tensor_t x = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("x"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32, options->request.latent_shape, &x));
  id4_pipeline_program_tensor_t timestep =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("timestep"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32, id4_pipeline_program_make_shape_rank1(1),
      &timestep));
  id4_pipeline_program_tensor_t image_indicator =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("image_indicator"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_I32,
      id4_pipeline_program_make_shape_rank2(total_token_count, 1),
      &image_indicator));
  id4_pipeline_program_tensor_t position_embedding =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
      builder, IREE_SV("position_embedding"),
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank4(2, 2, head_size / 2,
                                            total_token_count),
      &position_embedding));
  id4_pipeline_program_tensor_t condition =
      id4_pipeline_program_tensor_invalid();
  if (is_conditioned) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_import_tensor(
        builder, IREE_SV("condition"),
        ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
        ID4_PIPELINE_PROGRAM_DTYPE_F32,
        id4_pipeline_program_make_shape_rank2(llm_feature_count,
                                              text_token_count),
        &condition));
  }

  id4_pipeline_program_tensor_t input_proj_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
      builder, IREE_SV("input_proj.weight"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(hidden_size, input_channel_count),
      &input_proj_weight));
  id4_pipeline_program_tensor_t input_proj_bias =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
      builder, IREE_SV("input_proj.bias"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(hidden_size), &input_proj_bias));
  id4_pipeline_program_tensor_t image_indicator_embedding =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
      builder, IREE_SV("embed_image_indicator.weight"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(
          options->model.image_indicator_count, hidden_size),
      &image_indicator_embedding));
  id4_pipeline_program_tensor_t llm_cond_norm_weight =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t llm_cond_proj_weight =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_t llm_cond_proj_bias =
      id4_pipeline_program_tensor_invalid();
  if (is_conditioned) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
        builder, IREE_SV("llm_cond_norm.weight"),
        ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank1(llm_feature_count),
        &llm_cond_norm_weight));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
        builder, IREE_SV("llm_cond_proj.weight"),
        ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank2(hidden_size, llm_feature_count),
        &llm_cond_proj_weight));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_parameter(
        builder, IREE_SV("llm_cond_proj.bias"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        id4_pipeline_program_make_shape_rank1(hidden_size),
        &llm_cond_proj_bias));
  }

  id4_pipeline_program_tensor_t prelude_hidden =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, prelude_hidden_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(hidden_size, total_token_count),
      &prelude_hidden));
  id4_pipeline_program_tensor_t timestep_embedding =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
      builder, timestep_embedding_name, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(hidden_size), &timestep_embedding));

  if (is_conditioned) {
    id4_pipeline_program_tensor_t condition_norm =
        id4_pipeline_program_tensor_invalid();
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_acquire_tensor(
        builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
        ID4_PIPELINE_PROGRAM_DTYPE_F32,
        id4_pipeline_program_make_shape_rank2(llm_feature_count,
                                              text_token_count),
        &condition_norm));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_condition_rmsnorm(
        builder, text_token_count, llm_feature_count, condition,
        llm_cond_norm_weight, condition_norm));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
        builder, IREE_SV("ideogram4.cond.prelude.after_llm_cond_norm")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
        condition_norm));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_condition_project(
        builder, text_token_count, total_token_count, llm_feature_count,
        hidden_size, condition_norm, llm_cond_proj_weight, llm_cond_proj_bias,
        image_indicator, image_indicator_embedding, prelude_hidden));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_barrier(
        builder, IREE_SV("ideogram4.cond.prelude.after_llm_cond_proj")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
        builder, IREE_SV("ideogram4.cond.prelude.llm_cond_proj"),
        prelude_hidden));
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_prelude_image(
      builder, image_token_count, total_token_count, text_token_count,
      (uint32_t)options->request.latent_shape.dims[0],
      (uint32_t)options->request.latent_shape.dims[1], input_channel_count,
      hidden_size, x, input_proj_weight, input_proj_bias, image_indicator,
      image_indicator_embedding, prelude_hidden));
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dispatch_timestep_embedding(
      builder, timestep_embedding_name, hidden_size, timestep,
      timestep_embedding));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_inputs_name));

  id4_pipeline_program_tensor_t timestep_mlp_in =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t mlp_in_options = {
      .operation_name = timestep_mlp_in_name,
      .output_name = timestep_mlp_in_name,
      .parameter_prefix = IREE_SV("t_embedding.mlp_in"),
      .input = timestep_embedding,
      .input_size = hidden_size,
      .output_size = hidden_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_SILU,
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16_f32(
      builder, &mlp_in_options, &timestep_mlp_in));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_timestep_mlp_in_name));

  id4_pipeline_program_tensor_t timestep_mlp_out =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t mlp_out_options = {
      .operation_name = timestep_mlp_out_name,
      .output_name = timestep_mlp_out_name,
      .parameter_prefix = IREE_SV("t_embedding.mlp_out"),
      .input = timestep_mlp_in,
      .input_size = hidden_size,
      .output_size = hidden_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_IDENTITY,
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16_f32(
      builder, &mlp_out_options, &timestep_mlp_out));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_timestep_mlp_out_name));

  id4_pipeline_program_tensor_t adaln_input =
      id4_pipeline_program_tensor_invalid();
  const id4_ideogram4_dit_program_dense_options_t adaln_options = {
      .operation_name = IREE_SV("ideogram4.prelude.adaln_proj"),
      .output_name = adaln_input_name,
      .parameter_prefix = IREE_SV("adaln_proj"),
      .input = timestep_mlp_out,
      .input_size = hidden_size,
      .output_size = adaln_size,
      .activation_kind = ID4_IDEOGRAM4_DIT_ACTIVATION_SILU,
  };
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_dense_bf16_f32(
      builder, &adaln_options, &adaln_input));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_barrier(builder, after_adaln_name));

  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_tap(
      builder, prelude_hidden_name, prelude_hidden));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_program_tap(builder, adaln_input_name, adaln_input));

  id4_pipeline_program_tensor_t layer_input = prelude_hidden;
  for (uint32_t layer_ordinal = 0; layer_ordinal < options->model.layer_count;
       ++layer_ordinal) {
    id4_pipeline_program_tensor_t layer_output =
        id4_pipeline_program_tensor_invalid();
    id4_ideogram4_dit_program_block_options_t block_options = {
        .builder = builder,
        .branch_name = branch_name,
        .layer_ordinal = layer_ordinal,
        .adaln_size = adaln_size,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .attention_head_count = attention_head_count,
        .total_token_count = total_token_count,
        .hidden_input = layer_input,
        .adaln_input = adaln_input,
        .position_embedding = position_embedding,
    };
    IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_author_transformer_block(
        &block_options, &layer_output));
    layer_input = layer_output;
  }
  return id4_ideogram4_dit_program_author_final_output(
      options, builder, branch_name, total_token_count, layer_input,
      adaln_input);
}

const id4_ideogram4_dit_model_config_t*
id4_ideogram4_dit_program_ideogram4_model_config(void) {
  return &id4_ideogram4_dit_program_ideogram4_model_config_value;
}
