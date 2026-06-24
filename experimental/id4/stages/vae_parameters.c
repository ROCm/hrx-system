// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/vae_parameters.h"

#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/program.h"

enum {
  ID4_VAE_PARAMETER_CONFIG_VALUE_CAPACITY = 24,
  ID4_VAE_PARAMETER_PACK_CONFIG_COUNT = 3,
  ID4_VAE_PARAMETER_CAST_CONFIG_COUNT = 1,
  ID4_VAE_PARAMETER_PACK_WORKGROUP_SIZE_X = 256,
};

static const iree_string_view_t id4_vae_parameter_packed_conv3x3_suffix =
    IREE_SVL(".packed_ic_ky_kx_oc");
static const iree_string_view_t id4_vae_parameter_bf16_suffix =
    IREE_SVL(".bf16");

typedef enum id4_vae_parameter_transform_kind_e {
  // Invalid parameter transform.
  ID4_VAE_PARAMETER_TRANSFORM_KIND_INVALID = 0,
  // Reorders F32 conv3x3 weights from OCxICxKYxKX to ICxKYxKXxOC.
  ID4_VAE_PARAMETER_TRANSFORM_KIND_PACK_CONV3X3_F32 = 1,
  // Casts dense F32 source storage to dense BF16 target storage.
  ID4_VAE_PARAMETER_TRANSFORM_KIND_CAST_F32_BF16 = 2,
} id4_vae_parameter_transform_kind_t;

typedef struct id4_vae_parameter_mapping_t {
  // Virtual parameter key visible to the program plan.
  iree_string_view_t virtual_key;
  // Source parameter key in the wrapped provider.
  iree_string_view_t source_key;
  // Transform used to materialize the virtual parameter.
  id4_vae_parameter_transform_kind_t transform_kind;
  // Dense source tensor byte length gathered from the wrapped provider.
  iree_device_size_t source_byte_length;
  // Dense target tensor byte length exposed by the virtual parameter.
  iree_device_size_t target_byte_length;
  // Number of scalar elements processed by the transform kernel.
  uint64_t element_count;
  // Number of input channels in the convolution weight.
  uint32_t input_channel_count;
  // Number of output channels in the convolution weight.
  uint32_t output_channel_count;
} id4_vae_parameter_mapping_t;

typedef struct id4_vae_parameter_provider_t {
  // Base provider interface; must be the first field.
  iree_io_parameter_provider_t base;
  // Allocator used for wrapper-owned metadata.
  iree_allocator_t host_allocator;
  // Wrapped source parameter provider.
  iree_io_parameter_provider_t* source_provider;
  // Plan borrowed for the wrapper lifetime.
  const id4_pipeline_plan_t* plan;
  // Loom kernel library borrowed for the wrapper lifetime.
  id4_pipeline_kernel_library_t* kernel_library;
  // Loom kernel cache borrowed for the wrapper lifetime.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // HAL executable cache borrowed for the wrapper lifetime.
  iree_hal_executable_cache_t* executable_cache;
  // Diagnostics sink borrowed for the wrapper lifetime.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
  // Number of derived parameter mappings.
  iree_host_size_t mapping_count;
  // Derived parameter mappings.
  id4_vae_parameter_mapping_t* mappings;
} id4_vae_parameter_provider_t;

typedef struct id4_vae_parameter_request_t {
  // Source parameter key.
  iree_string_view_t key;
  // Source and target span for the gather.
  iree_io_parameter_span_t span;
} id4_vae_parameter_request_t;

typedef struct id4_vae_parameter_request_enumerator_t {
  // Request count.
  iree_host_size_t count;
  // Request values.
  const id4_vae_parameter_request_t* requests;
} id4_vae_parameter_request_enumerator_t;

typedef struct id4_vae_parameter_transform_t {
  // Mapping describing the derived layout.
  const id4_vae_parameter_mapping_t* mapping;
  // Target span in the final parameter slab.
  iree_io_parameter_span_t target_span;
} id4_vae_parameter_transform_t;

typedef struct id4_vae_parameter_pack_config_t {
  // Number of config bindings.
  iree_host_size_t count;
  // Fixed-capacity config binding storage.
  id4_pipeline_kernel_config_binding_t
      bindings[ID4_VAE_PARAMETER_PACK_CONFIG_COUNT];
  // Fixed-capacity string storage backing binding values.
  char value_storage[ID4_VAE_PARAMETER_PACK_CONFIG_COUNT]
                    [ID4_VAE_PARAMETER_CONFIG_VALUE_CAPACITY];
} id4_vae_parameter_pack_config_t;

typedef struct id4_vae_parameter_cast_config_t {
  // Number of config bindings.
  iree_host_size_t count;
  // Fixed-capacity config binding storage.
  id4_pipeline_kernel_config_binding_t
      bindings[ID4_VAE_PARAMETER_CAST_CONFIG_COUNT];
  // Fixed-capacity string storage backing binding values.
  char value_storage[ID4_VAE_PARAMETER_CAST_CONFIG_COUNT]
                    [ID4_VAE_PARAMETER_CONFIG_VALUE_CAPACITY];
} id4_vae_parameter_cast_config_t;

static const iree_io_parameter_provider_vtable_t
    id4_vae_parameter_provider_vtable;

iree_status_t id4_vae_parameter_format_packed_conv3x3_weight_key(
    iree_string_view_t source_key, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_key) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_key);
  int length = snprintf(buffer, buffer_capacity, "%.*s%.*s",
                        (int)source_key.size, source_key.data,
                        (int)id4_vae_parameter_packed_conv3x3_suffix.size,
                        id4_vae_parameter_packed_conv3x3_suffix.data);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format VAE packed parameter key");
  }
  *out_key = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

bool id4_vae_parameter_parse_packed_conv3x3_weight_key(
    iree_string_view_t key, iree_string_view_t* out_source_key) {
  if (out_source_key) *out_source_key = iree_string_view_empty();
  const iree_host_size_t suffix_position =
      iree_string_view_find(key, id4_vae_parameter_packed_conv3x3_suffix, 0);
  if (suffix_position == IREE_STRING_VIEW_NPOS ||
      suffix_position + id4_vae_parameter_packed_conv3x3_suffix.size !=
          key.size) {
    return false;
  }
  if (suffix_position == 0) return false;
  if (out_source_key) {
    *out_source_key = iree_string_view_substr(key, 0, suffix_position);
  }
  return true;
}

iree_status_t id4_vae_parameter_format_bf16_weight_key(
    iree_string_view_t source_key, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_key) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_key);
  int length =
      snprintf(buffer, buffer_capacity, "%.*s%.*s", (int)source_key.size,
               source_key.data, (int)id4_vae_parameter_bf16_suffix.size,
               id4_vae_parameter_bf16_suffix.data);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format VAE BF16 parameter key");
  }
  *out_key = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

bool id4_vae_parameter_parse_bf16_weight_key(
    iree_string_view_t key, iree_string_view_t* out_source_key) {
  if (out_source_key) *out_source_key = iree_string_view_empty();
  const iree_host_size_t suffix_position =
      iree_string_view_find(key, id4_vae_parameter_bf16_suffix, 0);
  if (suffix_position == IREE_STRING_VIEW_NPOS ||
      suffix_position + id4_vae_parameter_bf16_suffix.size != key.size) {
    return false;
  }
  if (suffix_position == 0) return false;
  if (out_source_key) {
    *out_source_key = iree_string_view_substr(key, 0, suffix_position);
  }
  return true;
}

static id4_vae_parameter_provider_t* id4_vae_parameter_provider_cast(
    iree_io_parameter_provider_t* base_provider) {
  return (id4_vae_parameter_provider_t*)base_provider;
}

static iree_status_t id4_vae_parameter_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_vae_parameter_provider_validate_create_options(
    const id4_vae_parameter_provider_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE parameter provider options are required");
  }
  IREE_RETURN_IF_ERROR(id4_vae_parameter_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("VAE parameter provider")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VAE parameter provider extension structures are not supported");
  }
  if (!options->source_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE source parameter provider is required");
  }
  if (!options->plan || !id4_pipeline_plan_source_program(options->plan)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VAE parameter provider requires a program-backed plan");
  }
  if (!options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE parameter kernel library is required");
  }
  if (!options->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE parameter kernel cache is required");
  }
  if (!options->executable_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VAE parameter HAL executable cache is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("VAE parameter provider"));
}

static bool id4_vae_parameter_is_packed_conv3x3_tensor(
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_string_view_t* out_source_key) {
  if (!tensor || tensor->dtype != ID4_PIPELINE_PROGRAM_DTYPE_F32 ||
      tensor->shape.rank != 4 || tensor->shape.dims[1] != 3 ||
      tensor->shape.dims[2] != 3) {
    return false;
  }
  return id4_vae_parameter_parse_packed_conv3x3_weight_key(tensor->name,
                                                           out_source_key);
}

static bool id4_vae_parameter_is_bf16_tensor(
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_string_view_t* out_source_key) {
  if (!tensor || tensor->dtype != ID4_PIPELINE_PROGRAM_DTYPE_BF16) {
    return false;
  }
  return id4_vae_parameter_parse_bf16_weight_key(tensor->name, out_source_key);
}

static iree_status_t id4_vae_parameter_count_mappings(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_mapping_count) {
  *out_mapping_count = 0;
  const id4_pipeline_program_t* program =
      id4_pipeline_plan_source_program(plan);
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) continue;
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(program,
                                       op->payload.parameter.tensor.ordinal);
    iree_string_view_t source_key = iree_string_view_empty();
    if (id4_vae_parameter_is_packed_conv3x3_tensor(tensor, &source_key) ||
        id4_vae_parameter_is_bf16_tensor(tensor, &source_key)) {
      ++*out_mapping_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_vae_parameter_populate_packed_conv3x3_mapping(
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_string_view_t source_key, id4_vae_parameter_mapping_t* out_mapping) {
  if (tensor->shape.dims[0] > UINT32_MAX ||
      tensor->shape.dims[3] > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE packed conv3x3 parameter %.*s channel count is out of range",
        (int)tensor->name.size, tensor->name.data);
  }
  if (tensor->byte_length % sizeof(float) != 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE packed conv3x3 parameter %.*s byte length is not F32 aligned",
        (int)tensor->name.size, tensor->name.data);
  }
  *out_mapping = (id4_vae_parameter_mapping_t){
      // Virtual packed parameter key from the plan.
      .virtual_key = tensor->name,
      // Original source key in the wrapped provider.
      .source_key = source_key,
      // Transform used to materialize the virtual parameter.
      .transform_kind = ID4_VAE_PARAMETER_TRANSFORM_KIND_PACK_CONV3X3_F32,
      // Dense source byte length.
      .source_byte_length = tensor->byte_length,
      // Dense target byte length.
      .target_byte_length = tensor->byte_length,
      // F32 scalar element count.
      .element_count = tensor->byte_length / sizeof(float),
      // Input channel dimension in ICxKYxKXxOC layout.
      .input_channel_count = (uint32_t)tensor->shape.dims[0],
      // Output channel dimension in ICxKYxKXxOC layout.
      .output_channel_count = (uint32_t)tensor->shape.dims[3],
  };
  return iree_ok_status();
}

static iree_status_t id4_vae_parameter_populate_bf16_mapping(
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_string_view_t source_key, id4_vae_parameter_mapping_t* out_mapping) {
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_shape_element_count(tensor->shape, &element_count));
  if (element_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE BF16 parameter %.*s element count is out of range",
        (int)tensor->name.size, tensor->name.data);
  }
  *out_mapping = (id4_vae_parameter_mapping_t){
      // Virtual BF16 parameter key from the plan.
      .virtual_key = tensor->name,
      // Original F32 source key in the wrapped provider.
      .source_key = source_key,
      // Transform used to materialize the virtual parameter.
      .transform_kind = ID4_VAE_PARAMETER_TRANSFORM_KIND_CAST_F32_BF16,
      // Dense F32 source byte length.
      .source_byte_length = element_count * sizeof(float),
      // Dense BF16 target byte length.
      .target_byte_length = tensor->byte_length,
      // Scalar element count.
      .element_count = element_count,
      // Not used by the cast transform.
      .input_channel_count = 0,
      // Not used by the cast transform.
      .output_channel_count = 0,
  };
  return iree_ok_status();
}

static iree_status_t id4_vae_parameter_populate_mappings(
    const id4_pipeline_plan_t* plan, id4_vae_parameter_mapping_t* mappings) {
  const id4_pipeline_program_t* program =
      id4_pipeline_plan_source_program(plan);
  iree_host_size_t mapping_index = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) continue;
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(program,
                                       op->payload.parameter.tensor.ordinal);
    iree_string_view_t source_key = iree_string_view_empty();
    if (id4_vae_parameter_is_packed_conv3x3_tensor(tensor, &source_key)) {
      IREE_RETURN_IF_ERROR(id4_vae_parameter_populate_packed_conv3x3_mapping(
          tensor, source_key, &mappings[mapping_index++]));
      continue;
    }
    if (id4_vae_parameter_is_bf16_tensor(tensor, &source_key)) {
      IREE_RETURN_IF_ERROR(id4_vae_parameter_populate_bf16_mapping(
          tensor, source_key, &mappings[mapping_index++]));
      continue;
    }
  }
  return iree_ok_status();
}

static const id4_vae_parameter_mapping_t* id4_vae_parameter_find_mapping(
    const id4_vae_parameter_provider_t* provider, iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < provider->mapping_count; ++i) {
    if (iree_string_view_equal(provider->mappings[i].virtual_key, key)) {
      return &provider->mappings[i];
    }
  }
  return NULL;
}

static iree_status_t id4_vae_parameter_request_enumerator(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  id4_vae_parameter_request_enumerator_t* state =
      (id4_vae_parameter_request_enumerator_t*)user_data;
  if (i >= state->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VAE parameter request %" PRIhsz " is out of range",
                            i);
  }
  *out_key = state->requests[i].key;
  *out_span = state->requests[i].span;
  return iree_ok_status();
}

static iree_io_parameter_enumerator_t id4_vae_parameter_make_enumerator(
    id4_vae_parameter_request_enumerator_t* state) {
  return (iree_io_parameter_enumerator_t){
      // Enumerator callback.
      .fn = id4_vae_parameter_request_enumerator,
      // Enumerator state.
      .user_data = state,
  };
}

static iree_status_t id4_vae_parameter_format_u32(
    uint32_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%u", value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format VAE parameter config value");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_vae_parameter_pack_config_add_u32(
    id4_vae_parameter_pack_config_t* config, iree_string_view_t key,
    uint32_t value) {
  if (config->count >= IREE_ARRAYSIZE(config->bindings)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VAE parameter pack config capacity exceeded");
  }
  iree_string_view_t value_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_vae_parameter_format_u32(
      value, config->value_storage[config->count],
      IREE_ARRAYSIZE(config->value_storage[config->count]), &value_string));
  config->bindings[config->count++] =
      id4_pipeline_make_kernel_config_binding(key, value_string);
  return iree_ok_status();
}

static iree_status_t id4_vae_parameter_pack_config(
    const id4_vae_parameter_mapping_t* mapping,
    id4_vae_parameter_pack_config_t* out_config) {
  memset(out_config, 0, sizeof(*out_config));
  if (mapping->element_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE packed conv3x3 parameter %.*s element count is out of range",
        (int)mapping->virtual_key.size, mapping->virtual_key.data);
  }
  IREE_RETURN_IF_ERROR(id4_vae_parameter_pack_config_add_u32(
      out_config, IREE_SV("id4.vae.pack_conv3x3_weight.input_channel_count"),
      mapping->input_channel_count));
  IREE_RETURN_IF_ERROR(id4_vae_parameter_pack_config_add_u32(
      out_config, IREE_SV("id4.vae.pack_conv3x3_weight.output_channel_count"),
      mapping->output_channel_count));
  return id4_vae_parameter_pack_config_add_u32(
      out_config, IREE_SV("id4.vae.pack_conv3x3_weight.element_count"),
      (uint32_t)mapping->element_count);
}

static iree_status_t id4_vae_parameter_cast_config(
    const id4_vae_parameter_mapping_t* mapping,
    id4_vae_parameter_cast_config_t* out_config) {
  memset(out_config, 0, sizeof(*out_config));
  if (mapping->element_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VAE BF16 parameter %.*s element count is out of range",
        (int)mapping->virtual_key.size, mapping->virtual_key.data);
  }
  iree_string_view_t value_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_vae_parameter_format_u32(
      (uint32_t)mapping->element_count, out_config->value_storage[0],
      IREE_ARRAYSIZE(out_config->value_storage[0]), &value_string));
  out_config->bindings[0] = id4_pipeline_make_kernel_config_binding(
      IREE_SV("id4.elementwise.cast_f32_bf16.element_count"), value_string);
  out_config->count = 1;
  return iree_ok_status();
}

static iree_hal_dispatch_config_t id4_vae_parameter_transform_dispatch_config(
    const id4_vae_parameter_mapping_t* mapping) {
  const uint32_t element_count = (uint32_t)mapping->element_count;
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(
          element_count / ID4_VAE_PARAMETER_PACK_WORKGROUP_SIZE_X +
              (element_count % ID4_VAE_PARAMETER_PACK_WORKGROUP_SIZE_X != 0
                   ? 1
                   : 0),
          1, 1);
  dispatch_config.workgroup_size[0] = ID4_VAE_PARAMETER_PACK_WORKGROUP_SIZE_X;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

static iree_hal_semaphore_list_t id4_vae_parameter_one_semaphore_list(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return (iree_hal_semaphore_list_t){
      // One semaphore in the list.
      .count = 1,
      // Semaphore pointer list.
      .semaphores = semaphore,
      // Required or published payload value.
      .payload_values = payload_value,
  };
}

static iree_status_t id4_vae_parameter_create_semaphore(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(device, queue_affinity, 0,
                                   IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore);
}

static iree_status_t id4_vae_parameter_prepare_pack_executable(
    id4_vae_parameter_provider_t* provider,
    iree_hal_queue_affinity_t queue_affinity,
    const id4_vae_parameter_mapping_t* mapping,
    id4_pipeline_kernel_executable_t** out_executable,
    iree_hal_executable_function_t* out_function) {
  *out_executable = NULL;
  *out_function = iree_hal_executable_function_invalid();

  const id4_pipeline_kernel_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_lookup(
      provider->kernel_library, IREE_SV("vae/pack_conv3x3_weight_f32"),
      &module));
  id4_vae_parameter_pack_config_t config;
  IREE_RETURN_IF_ERROR(id4_vae_parameter_pack_config(mapping, &config));

  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = provider->executable_cache;
  prepare_options.queue_affinity = queue_affinity;
  prepare_options.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
  prepare_options.source_identifier = module->source_identifier;
  prepare_options.source_contents = module->source_contents;
  prepare_options.module_path = module->module_path;
  prepare_options.function_name = IREE_SV("id4_vae_pack_conv3x3_weight_f32");
  prepare_options.config_binding_count = config.count;
  prepare_options.config_bindings = config.bindings;
  prepare_options.diagnostic_artifact_flags =
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON;
  prepare_options.diagnostics_sink = provider->diagnostics_sink;

  id4_pipeline_kernel_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_prepare_executable(
      provider->kernel_cache, &prepare_options, &executable));
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  iree_status_t status = iree_hal_executable_lookup_function_by_name(
      id4_pipeline_kernel_executable_hal_executable(executable),
      prepare_options.function_name, &function);
  if (iree_status_is_ok(status)) {
    *out_executable = executable;
    *out_function = function;
    executable = NULL;
  }
  id4_pipeline_kernel_executable_release(executable);
  return status;
}

static iree_status_t id4_vae_parameter_prepare_cast_executable(
    id4_vae_parameter_provider_t* provider,
    iree_hal_queue_affinity_t queue_affinity,
    const id4_vae_parameter_mapping_t* mapping,
    id4_pipeline_kernel_executable_t** out_executable,
    iree_hal_executable_function_t* out_function) {
  *out_executable = NULL;
  *out_function = iree_hal_executable_function_invalid();

  const id4_pipeline_kernel_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_lookup(
      provider->kernel_library, IREE_SV("elementwise/cast_f32_bf16"), &module));
  id4_vae_parameter_cast_config_t config;
  IREE_RETURN_IF_ERROR(id4_vae_parameter_cast_config(mapping, &config));

  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = provider->executable_cache;
  prepare_options.queue_affinity = queue_affinity;
  prepare_options.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
  prepare_options.source_identifier = module->source_identifier;
  prepare_options.source_contents = module->source_contents;
  prepare_options.module_path = module->module_path;
  prepare_options.function_name = IREE_SV("id4_elementwise_cast_f32_bf16");
  prepare_options.config_binding_count = config.count;
  prepare_options.config_bindings = config.bindings;
  prepare_options.diagnostic_artifact_flags =
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON;
  prepare_options.diagnostics_sink = provider->diagnostics_sink;

  id4_pipeline_kernel_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_prepare_executable(
      provider->kernel_cache, &prepare_options, &executable));
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  iree_status_t status = iree_hal_executable_lookup_function_by_name(
      id4_pipeline_kernel_executable_hal_executable(executable),
      prepare_options.function_name, &function);
  if (iree_status_is_ok(status)) {
    *out_executable = executable;
    *out_function = function;
    executable = NULL;
  }
  id4_pipeline_kernel_executable_release(executable);
  return status;
}

static iree_status_t id4_vae_parameter_prepare_transform_executable(
    id4_vae_parameter_provider_t* provider,
    iree_hal_queue_affinity_t queue_affinity,
    const id4_vae_parameter_mapping_t* mapping,
    id4_pipeline_kernel_executable_t** out_executable,
    iree_hal_executable_function_t* out_function) {
  switch (mapping->transform_kind) {
    case ID4_VAE_PARAMETER_TRANSFORM_KIND_PACK_CONV3X3_F32:
      return id4_vae_parameter_prepare_pack_executable(
          provider, queue_affinity, mapping, out_executable, out_function);
    case ID4_VAE_PARAMETER_TRANSFORM_KIND_CAST_F32_BF16:
      return id4_vae_parameter_prepare_cast_executable(
          provider, queue_affinity, mapping, out_executable, out_function);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VAE parameter %.*s transform kind %d is invalid",
                              (int)mapping->virtual_key.size,
                              mapping->virtual_key.data,
                              (int)mapping->transform_kind);
  }
}

static iree_status_t id4_vae_parameter_submit_transform(
    id4_vae_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_string_view_t source_scope,
    iree_hal_buffer_t* target_buffer, id4_vae_parameter_transform_t transform,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  const id4_vae_parameter_mapping_t* mapping = transform.mapping;
  id4_pipeline_kernel_executable_t* executable = NULL;
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(id4_vae_parameter_prepare_transform_executable(
      provider, queue_affinity, mapping, &executable, &function));

  iree_hal_semaphore_t* alloca_semaphore = NULL;
  iree_hal_semaphore_t* gather_semaphore = NULL;
  iree_hal_semaphore_t* encode_semaphore = NULL;
  iree_hal_semaphore_t* cleanup_semaphore = NULL;
  iree_hal_buffer_t* staging_buffer = NULL;
  iree_status_t status = id4_vae_parameter_create_semaphore(
      device, queue_affinity, &alloca_semaphore);
  if (iree_status_is_ok(status)) {
    status = id4_vae_parameter_create_semaphore(device, queue_affinity,
                                                &gather_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = id4_vae_parameter_create_semaphore(device, queue_affinity,
                                                &encode_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = id4_vae_parameter_create_semaphore(device, queue_affinity,
                                                &cleanup_semaphore);
  }

  uint64_t alloca_payload_value = 1;
  uint64_t gather_payload_value = 1;
  uint64_t encode_payload_value = 1;
  uint64_t cleanup_payload_value = 1;
  iree_hal_semaphore_list_t alloca_signal_list =
      id4_vae_parameter_one_semaphore_list(&alloca_semaphore,
                                           &alloca_payload_value);
  iree_hal_semaphore_list_t gather_signal_list =
      id4_vae_parameter_one_semaphore_list(&gather_semaphore,
                                           &gather_payload_value);
  iree_hal_semaphore_list_t encode_signal_list =
      id4_vae_parameter_one_semaphore_list(&encode_semaphore,
                                           &encode_payload_value);
  iree_hal_semaphore_list_t cleanup_signal_list =
      id4_vae_parameter_one_semaphore_list(&cleanup_semaphore,
                                           &cleanup_payload_value);

  iree_hal_buffer_params_t staging_params = {0};
  staging_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  staging_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  staging_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  staging_params.queue_affinity = queue_affinity;
  staging_params.min_alignment = 16;

  bool alloca_submitted = false;
  bool gather_submitted = false;
  bool encode_submitted = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(
        device, queue_affinity, wait_semaphore_list, alloca_signal_list,
        /*pool=*/NULL, staging_params, mapping->source_byte_length,
        IREE_HAL_ALLOCA_FLAG_NONE, &staging_buffer);
    alloca_submitted = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    id4_vae_parameter_request_t source_request = {
        // Original source parameter key.
        .key = mapping->source_key,
        // Gather the entire source weight into the staging buffer.
        .span =
            {
                .parameter_offset = 0,
                .buffer_offset = 0,
                .length = mapping->source_byte_length,
            },
    };
    id4_vae_parameter_request_enumerator_t enumerator_state = {
        // One source request.
        .count = 1,
        // Source request storage.
        .requests = &source_request,
    };
    status = iree_io_parameter_provider_gather(
        provider->source_provider, device, queue_affinity, alloca_signal_list,
        gather_signal_list, source_scope, staging_buffer, 1,
        id4_vae_parameter_make_enumerator(&enumerator_state));
    gather_submitted = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_t bindings[2] = {
        iree_hal_make_buffer_ref(staging_buffer, 0,
                                 mapping->source_byte_length),
        iree_hal_make_buffer_ref(target_buffer,
                                 transform.target_span.buffer_offset,
                                 transform.target_span.length),
    };
    iree_hal_buffer_ref_list_t binding_list = {
        // Two direct buffer bindings.
        .count = IREE_ARRAYSIZE(bindings),
        // Direct source and target buffer refs.
        .values = bindings,
    };
    status = iree_hal_device_queue_dispatch(
        device, queue_affinity, gather_signal_list, encode_signal_list,
        id4_pipeline_kernel_executable_hal_executable(executable), function,
        id4_vae_parameter_transform_dispatch_config(mapping),
        iree_const_byte_span_empty(), binding_list,
        IREE_HAL_DISPATCH_FLAG_NONE);
    encode_submitted = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        device, queue_affinity, encode_signal_list, signal_semaphore_list,
        staging_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
  } else if (alloca_submitted && staging_buffer) {
    iree_hal_semaphore_list_t cleanup_wait_list =
        encode_submitted
            ? encode_signal_list
            : (gather_submitted ? gather_signal_list : alloca_signal_list);
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        device, queue_affinity, cleanup_wait_list, cleanup_signal_list,
        staging_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    if (iree_status_is_ok(cleanup_status)) {
      cleanup_status = iree_hal_semaphore_wait(
          cleanup_semaphore, cleanup_payload_value, iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE);
    }
    status = iree_status_join(status, cleanup_status);
  }

  iree_hal_buffer_release(staging_buffer);
  iree_hal_semaphore_release(cleanup_semaphore);
  iree_hal_semaphore_release(encode_semaphore);
  iree_hal_semaphore_release(gather_semaphore);
  iree_hal_semaphore_release(alloca_semaphore);
  id4_pipeline_kernel_executable_release(executable);
  return status;
}

static iree_status_t id4_vae_parameter_submit_transforms(
    id4_vae_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_string_view_t source_scope,
    iree_hal_buffer_t* target_buffer,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t transform_count,
    const id4_vae_parameter_transform_t* transforms) {
  iree_hal_semaphore_t** chain_semaphores = NULL;
  if (transform_count > 1) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        provider->host_allocator, transform_count - 1,
        sizeof(chain_semaphores[0]), (void**)&chain_semaphores));
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i + 1 < transform_count && iree_status_is_ok(status); ++i) {
    status = id4_vae_parameter_create_semaphore(device, queue_affinity,
                                                &chain_semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < transform_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_semaphore_t* wait_semaphore =
        i == 0 ? NULL : chain_semaphores[i - 1];
    uint64_t wait_value = 1;
    iree_hal_semaphore_list_t transform_wait_list =
        i == 0 ? wait_semaphore_list
               : id4_vae_parameter_one_semaphore_list(&wait_semaphore,
                                                      &wait_value);
    iree_hal_semaphore_t* signal_semaphore =
        i + 1 == transform_count ? NULL : chain_semaphores[i];
    uint64_t signal_value = 1;
    iree_hal_semaphore_list_t transform_signal_list =
        i + 1 == transform_count ? signal_semaphore_list
                                 : id4_vae_parameter_one_semaphore_list(
                                       &signal_semaphore, &signal_value);
    status = id4_vae_parameter_submit_transform(
        provider, device, queue_affinity, source_scope, target_buffer,
        transforms[i], transform_wait_list, transform_signal_list);
  }
  if (!iree_status_is_ok(status) && signal_semaphore_list.count != 0) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  for (iree_host_size_t i = 0; i + 1 < transform_count; ++i) {
    iree_hal_semaphore_release(chain_semaphores[i]);
  }
  iree_allocator_free(provider->host_allocator, chain_semaphores);
  return status;
}

static void id4_vae_parameter_provider_destroy(
    iree_io_parameter_provider_t* base_provider) {
  id4_vae_parameter_provider_t* provider =
      id4_vae_parameter_provider_cast(base_provider);
  iree_allocator_t host_allocator = provider->host_allocator;
  iree_allocator_free(host_allocator, provider->mappings);
  iree_io_parameter_provider_release(provider->source_provider);
  iree_allocator_free(host_allocator, provider);
}

static iree_status_t id4_vae_parameter_provider_notify(
    iree_io_parameter_provider_t* base_provider,
    iree_io_parameter_provider_signal_t signal) {
  id4_vae_parameter_provider_t* provider =
      id4_vae_parameter_provider_cast(base_provider);
  return iree_io_parameter_provider_notify(provider->source_provider, signal);
}

static bool id4_vae_parameter_provider_query_support(
    iree_io_parameter_provider_t* base_provider, iree_string_view_t scope) {
  id4_vae_parameter_provider_t* provider =
      id4_vae_parameter_provider_cast(base_provider);
  return iree_io_parameter_provider_query_support(provider->source_provider,
                                                  scope);
}

static iree_status_t id4_vae_parameter_provider_load(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_params_t target_params,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator,
    iree_io_parameter_emitter_t emitter) {
  id4_vae_parameter_provider_t* provider =
      id4_vae_parameter_provider_cast(base_provider);
  return iree_io_parameter_provider_load(
      provider->source_provider, device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, source_scope, target_params, count, enumerator,
      emitter);
}

static iree_status_t id4_vae_parameter_provider_gather(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  id4_vae_parameter_provider_t* provider =
      id4_vae_parameter_provider_cast(base_provider);
  id4_vae_parameter_request_t* direct_requests = NULL;
  id4_vae_parameter_transform_t* transforms = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      provider->host_allocator, count, sizeof(direct_requests[0]),
      (void**)&direct_requests);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(provider->host_allocator, count,
                                    sizeof(transforms[0]), (void**)&transforms);
  }

  iree_host_size_t direct_request_count = 0;
  iree_host_size_t transform_count = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_string_view_t key = iree_string_view_empty();
    iree_io_parameter_span_t span = {0};
    status = enumerator.fn(enumerator.user_data, i, &key, &span);
    if (!iree_status_is_ok(status)) break;
    const id4_vae_parameter_mapping_t* mapping =
        id4_vae_parameter_find_mapping(provider, key);
    if (mapping) {
      if (span.parameter_offset != 0 ||
          span.length != mapping->target_byte_length) {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "VAE derived parameter %.*s requires a whole-parameter "
            "gather request",
            (int)mapping->virtual_key.size, mapping->virtual_key.data);
        break;
      }
      transforms[transform_count++] = (id4_vae_parameter_transform_t){
          // Mapping for the virtual parameter.
          .mapping = mapping,
          // Target span in the final parameter slab.
          .target_span = span,
      };
    } else {
      direct_requests[direct_request_count++] = (id4_vae_parameter_request_t){
          // Direct source key.
          .key = key,
          // Direct source and target span.
          .span = span,
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(provider->host_allocator, transforms);
    iree_allocator_free(provider->host_allocator, direct_requests);
    return status;
  }

  if (transform_count == 0) {
    id4_vae_parameter_request_enumerator_t direct_enumerator_state = {
        // Direct request count.
        .count = direct_request_count,
        // Direct requests.
        .requests = direct_requests,
    };
    status = iree_io_parameter_provider_gather(
        provider->source_provider, device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, source_scope, target_buffer,
        direct_request_count,
        id4_vae_parameter_make_enumerator(&direct_enumerator_state));
  } else {
    iree_hal_semaphore_t* direct_done_semaphore = NULL;
    uint64_t direct_done_value = 1;
    iree_hal_semaphore_list_t transform_wait_list = wait_semaphore_list;
    if (direct_request_count != 0) {
      status = id4_vae_parameter_create_semaphore(device, queue_affinity,
                                                  &direct_done_semaphore);
      iree_hal_semaphore_list_t direct_signal_list =
          id4_vae_parameter_one_semaphore_list(&direct_done_semaphore,
                                               &direct_done_value);
      if (iree_status_is_ok(status)) {
        id4_vae_parameter_request_enumerator_t direct_enumerator_state = {
            // Direct request count.
            .count = direct_request_count,
            // Direct requests.
            .requests = direct_requests,
        };
        status = iree_io_parameter_provider_gather(
            provider->source_provider, device, queue_affinity,
            wait_semaphore_list, direct_signal_list, source_scope,
            target_buffer, direct_request_count,
            id4_vae_parameter_make_enumerator(&direct_enumerator_state));
      }
      transform_wait_list = id4_vae_parameter_one_semaphore_list(
          &direct_done_semaphore, &direct_done_value);
    }
    if (iree_status_is_ok(status)) {
      status = id4_vae_parameter_submit_transforms(
          provider, device, queue_affinity, source_scope, target_buffer,
          transform_wait_list, signal_semaphore_list, transform_count,
          transforms);
    } else if (signal_semaphore_list.count != 0) {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
    }
    iree_hal_semaphore_release(direct_done_semaphore);
  }

  iree_allocator_free(provider->host_allocator, transforms);
  iree_allocator_free(provider->host_allocator, direct_requests);
  return status;
}

static iree_status_t id4_vae_parameter_provider_scatter(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  id4_vae_parameter_provider_t* provider =
      id4_vae_parameter_provider_cast(base_provider);
  return iree_io_parameter_provider_scatter(
      provider->source_provider, device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, source_buffer, target_scope, count, enumerator);
}

static const iree_io_parameter_provider_vtable_t
    id4_vae_parameter_provider_vtable = {
        // Destroys the wrapper provider.
        .destroy = id4_vae_parameter_provider_destroy,
        // Forwards notifications to the source provider.
        .notify = id4_vae_parameter_provider_notify,
        // Forwards scope support checks to the source provider.
        .query_support = id4_vae_parameter_provider_query_support,
        // Forwards load requests to the source provider.
        .load = id4_vae_parameter_provider_load,
        // Gathers direct and derived VAE parameter requests.
        .gather = id4_vae_parameter_provider_gather,
        // Forwards scatter requests to the source provider.
        .scatter = id4_vae_parameter_provider_scatter,
};

iree_status_t id4_vae_parameter_provider_create(
    const id4_vae_parameter_provider_create_options_t* options,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  IREE_RETURN_IF_ERROR(
      id4_vae_parameter_provider_validate_create_options(options));

  iree_host_size_t mapping_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_vae_parameter_count_mappings(options->plan, &mapping_count));

  id4_vae_parameter_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_atomic_ref_count_init(&provider->base.ref_count);
  provider->base.vtable = &id4_vae_parameter_provider_vtable;
  provider->host_allocator = host_allocator;
  provider->source_provider = options->source_provider;
  iree_io_parameter_provider_retain(provider->source_provider);
  provider->plan = options->plan;
  provider->kernel_library = options->kernel_library;
  provider->kernel_cache = options->kernel_cache;
  provider->executable_cache = options->executable_cache;
  provider->diagnostics_sink = options->diagnostics_sink;
  provider->mapping_count = mapping_count;

  iree_status_t status = iree_ok_status();
  if (mapping_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, mapping_count,
                                         sizeof(provider->mappings[0]),
                                         (void**)&provider->mappings);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_vae_parameter_populate_mappings(options->plan, provider->mappings);
  }
  if (iree_status_is_ok(status)) {
    *out_provider = (iree_io_parameter_provider_t*)provider;
  } else {
    iree_io_parameter_provider_release((iree_io_parameter_provider_t*)provider);
  }
  return status;
}
