// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl_prefill.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/region.h"
#include "iree/base/internal/arena.h"

#define ID4_QWEN3_VL_PREFILL_STAGE_BINDING_COUNT 4
#define ID4_QWEN3_VL_PREFILL_STAGE_INPUT_BINDING_SLOT 0
#define ID4_QWEN3_VL_PREFILL_STAGE_WEIGHT_BINDING_SLOT 1
#define ID4_QWEN3_VL_PREFILL_STAGE_OUTPUT_BINDING_SLOT 2
#define ID4_QWEN3_VL_PREFILL_STAGE_LOCAL_BINDING_SLOT 3
#define ID4_QWEN3_VL_PREFILL_STAGE_BF16_ELEMENT_BYTE_LENGTH 2
#define ID4_QWEN3_VL_PREFILL_STAGE_RMSNORM_MODULE_PATH "qwen3_vl/rmsnorm"
#define ID4_QWEN3_VL_PREFILL_STAGE_RMSNORM_FUNCTION_NAME \
  "id4_qwen3_vl_rmsnorm_f32"
#define ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_MODULE_PATH \
  "qwen3_vl/linear_bf16_f32"
#define ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_FUNCTION_NAME \
  "id4_qwen3_vl_linear_bf16_f32"

typedef struct id4_qwen3_vl_prefill_stage_parameter_layout_t {
  // Gather span for the input RMSNorm weight vector in the parameter slab.
  iree_io_parameter_span_t rmsnorm_weight;
  // Gather span for the Q projection weight matrix in the parameter slab.
  iree_io_parameter_span_t q_projection_weight;
  // Total byte length of the packed parameter slab.
  iree_device_size_t total_byte_length;
} id4_qwen3_vl_prefill_stage_parameter_layout_t;

typedef struct id4_qwen3_vl_prefill_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Number of token rows in the prefill hidden-state tensor.
  uint32_t token_count;
  // True after load has completed.
  bool is_loaded;
} id4_qwen3_vl_prefill_stage_t;

typedef struct id4_qwen3_vl_prefill_stage_bundle_payload_t {
  // Prepared RMSNorm kernel executable retained for command-buffer validity.
  id4_pipeline_kernel_executable_t* rmsnorm_executable;
  // Prepared Q projection executable retained for command-buffer validity.
  id4_pipeline_kernel_executable_t* q_projection_executable;
  // Prepared reusable region issued by the Qwen3-VL prefill stage.
  id4_pipeline_prepared_region_t* prepared_region;
  // Input hidden-state buffer retained for issue-time binding.
  iree_hal_buffer_t* input_buffer;
  // Output hidden-state buffer retained for issue-time binding and readback.
  iree_hal_buffer_t* output_buffer;
  // Hidden-state tensor byte length.
  iree_device_size_t hidden_states_byte_length;
  // Fixed issue-time binding table storage.
  iree_hal_buffer_binding_t bindings[ID4_QWEN3_VL_PREFILL_STAGE_BINDING_COUNT];
} id4_qwen3_vl_prefill_stage_bundle_payload_t;

static id4_qwen3_vl_prefill_stage_t* id4_qwen3_vl_prefill_stage_cast(
    id4_pipeline_stage_t* base_stage) {
  return (id4_qwen3_vl_prefill_stage_t*)base_stage;
}

static const id4_qwen3_vl_prefill_stage_t*
id4_qwen3_vl_prefill_stage_const_cast(const id4_pipeline_stage_t* base_stage) {
  return (const id4_qwen3_vl_prefill_stage_t*)base_stage;
}

static iree_status_t id4_qwen3_vl_prefill_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static uint64_t id4_qwen3_vl_prefill_stage_hidden_element_count(
    const id4_qwen3_vl_prefill_stage_t* stage) {
  return (uint64_t)stage->token_count * ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE;
}

static iree_status_t id4_qwen3_vl_prefill_stage_f32_byte_length(
    uint64_t element_count, iree_device_size_t* out_byte_length) {
  if (!iree_device_size_checked_mul(
          element_count, ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH,
          out_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL prefill f32 byte length overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_prefill_stage_hidden_states_byte_length(
    const id4_qwen3_vl_prefill_stage_t* stage,
    iree_device_size_t* out_byte_length) {
  return id4_qwen3_vl_prefill_stage_f32_byte_length(
      id4_qwen3_vl_prefill_stage_hidden_element_count(stage), out_byte_length);
}

static iree_status_t id4_qwen3_vl_prefill_stage_weight_byte_length(
    const id4_qwen3_vl_prefill_stage_t* stage,
    iree_device_size_t* out_byte_length) {
  (void)stage;
  return id4_qwen3_vl_prefill_stage_f32_byte_length(
      ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE, out_byte_length);
}

static iree_status_t id4_qwen3_vl_prefill_stage_q_projection_weight_byte_length(
    const id4_qwen3_vl_prefill_stage_t* stage,
    iree_device_size_t* out_byte_length) {
  (void)stage;
  uint64_t element_count = (uint64_t)ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE *
                           ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE;
  if (!iree_device_size_checked_mul(
          element_count, ID4_QWEN3_VL_PREFILL_STAGE_BF16_ELEMENT_BYTE_LENGTH,
          out_byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Qwen3-VL prefill Q projection weight byte length overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_prefill_stage_parameter_layout(
    const id4_qwen3_vl_prefill_stage_t* stage,
    id4_qwen3_vl_prefill_stage_parameter_layout_t* out_layout) {
  memset(out_layout, 0, sizeof(*out_layout));
  iree_device_size_t slab_byte_length = 0;
  iree_device_size_t rmsnorm_weight_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_weight_byte_length(
      stage, &rmsnorm_weight_byte_length));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_pack_span(
      rmsnorm_weight_byte_length, /*alignment=*/16, &slab_byte_length,
      &out_layout->rmsnorm_weight));
  iree_device_size_t q_projection_weight_byte_length = 0;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_prefill_stage_q_projection_weight_byte_length(
          stage, &q_projection_weight_byte_length));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_pack_span(
      q_projection_weight_byte_length, /*alignment=*/16, &slab_byte_length,
      &out_layout->q_projection_weight));
  out_layout->total_byte_length = slab_byte_length;
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_prefill_stage_validate_create_options(
    const id4_qwen3_vl_prefill_stage_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL prefill stage create options are "
                            "required");
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Qwen3-VL prefill stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen3-VL prefill stage create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL prefill stage device group is required");
  }
  if (options->token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL prefill token count must be nonzero");
  }
  iree_device_size_t byte_length = 0;
  const uint64_t element_count =
      (uint64_t)options->token_count * ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_prefill_stage_f32_byte_length(element_count, &byte_length));
  id4_qwen3_vl_prefill_stage_parameter_layout_t parameter_layout;
  id4_qwen3_vl_prefill_stage_t layout_stage;
  memset(&layout_stage, 0, sizeof(layout_stage));
  return id4_qwen3_vl_prefill_stage_parameter_layout(&layout_stage,
                                                     &parameter_layout);
}

static iree_status_t id4_qwen3_vl_prefill_stage_format_u32(
    uint32_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%" PRIu32, value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format Qwen3-VL prefill config value");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_prefill_stage_format_specialization_key(
    const id4_qwen3_vl_prefill_stage_t* stage, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  int length =
      snprintf(buffer, buffer_capacity,
               "id4_qwen3_vl_prefill_rmsnorm_f32:token_count=%" PRIu32
               ":hidden_size=%" PRIu32,
               stage->token_count, ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "failed to format Qwen3-VL prefill specialization key");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t
id4_qwen3_vl_prefill_stage_format_q_projection_specialization_key(
    const id4_qwen3_vl_prefill_stage_t* stage, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  int length = snprintf(
      buffer, buffer_capacity,
      "id4_qwen3_vl_prefill_q_projection_bf16_f32:"
      "token_count=%" PRIu32 ":input_size=%" PRIu32 ":output_size=%" PRIu32,
      stage->token_count, ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE,
      ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "failed to format Qwen3-VL prefill Q projection specialization key");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t
id4_qwen3_vl_prefill_stage_lifecycle_event(iree_string_view_t key,
                                           iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the Qwen3-VL prefill stage.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name used across Qwen3-VL prefill diagnostics.
      .stage_name = IREE_SV("qwen3_vl.prefill"),
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_qwen3_vl_prefill_stage_emit_lifecycle(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_qwen3_vl_prefill_stage_lifecycle_event(key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static id4_pipeline_tensor_shape_t id4_qwen3_vl_prefill_stage_make_hidden_shape(
    const id4_qwen3_vl_prefill_stage_t* stage) {
  id4_pipeline_tensor_shape_t shape;
  memset(&shape, 0, sizeof(shape));
  shape.rank = 2;
  shape.dims[0] = stage->token_count;
  shape.dims[1] = ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE;
  return shape;
}

static id4_pipeline_tensor_shape_t id4_qwen3_vl_prefill_stage_make_weight_shape(
    const id4_qwen3_vl_prefill_stage_t* stage) {
  id4_pipeline_tensor_shape_t shape;
  memset(&shape, 0, sizeof(shape));
  shape.rank = 1;
  shape.dims[0] = ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE;
  return shape;
}

static id4_pipeline_tensor_shape_t
id4_qwen3_vl_prefill_stage_make_projection_weight_shape(
    const id4_qwen3_vl_prefill_stage_t* stage) {
  id4_pipeline_tensor_shape_t shape;
  memset(&shape, 0, sizeof(shape));
  shape.rank = 2;
  shape.dims[0] = ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE;
  shape.dims[1] = ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE;
  return shape;
}

static iree_hal_dispatch_config_t
id4_qwen3_vl_prefill_stage_make_rmsnorm_dispatch_config(
    const id4_qwen3_vl_prefill_stage_t* stage) {
  // This mirrors kernels/qwen3_vl/rmsnorm.loom kernel.launch.config until the
  // runtime can evaluate Loom launch regions directly.
  const uint32_t workgroup_size_x = 256;
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(stage->token_count, 1, 1);
  dispatch_config.workgroup_size[0] = workgroup_size_x;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

static iree_hal_dispatch_config_t
id4_qwen3_vl_prefill_stage_make_q_projection_dispatch_config(
    const id4_qwen3_vl_prefill_stage_t* stage) {
  // This mirrors kernels/qwen3_vl/linear_bf16_f32.loom kernel.launch.config
  // until the runtime can evaluate Loom launch regions directly.
  const uint32_t workgroup_size_x = 256;
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(
          stage->token_count, ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE, 1);
  dispatch_config.workgroup_size[0] = workgroup_size_x;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  return dispatch_config;
}

static iree_status_t id4_qwen3_vl_prefill_stage_author_rmsnorm(
    id4_qwen3_vl_prefill_stage_t* stage, id4_pipeline_region_builder_t* builder,
    iree_hal_executable_t* hal_executable,
    iree_hal_executable_function_t function,
    iree_io_parameter_span_t weight_span,
    id4_pipeline_tensor_t* out_output_tensor) {
  iree_device_size_t hidden_states_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_hidden_states_byte_length(
      stage, &hidden_states_byte_length));

  id4_pipeline_tensor_shape_t hidden_shape =
      id4_qwen3_vl_prefill_stage_make_hidden_shape(stage);
  id4_pipeline_tensor_shape_t weight_shape =
      id4_qwen3_vl_prefill_stage_make_weight_shape(stage);

  id4_pipeline_tensor_import_t input_import;
  memset(&input_import, 0, sizeof(input_import));
  input_import.layout.name = IREE_SV("qwen3_vl.prefill.hidden_states.input");
  input_import.layout.shape = hidden_shape;
  input_import.layout.byte_length = hidden_states_byte_length;
  input_import.layout.alignment =
      ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH;
  input_import.binding_slot = ID4_QWEN3_VL_PREFILL_STAGE_INPUT_BINDING_SLOT;
  input_import.offset = 0;
  input_import.flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;

  id4_pipeline_tensor_t input_tensor;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_import_tensor(builder, &input_import, &input_tensor));

  id4_pipeline_tensor_import_t weight_import;
  memset(&weight_import, 0, sizeof(weight_import));
  weight_import.layout.name =
      IREE_SV("qwen3_vl.prefill.layer0.input_rmsnorm.weight");
  weight_import.layout.shape = weight_shape;
  weight_import.layout.byte_length = weight_span.length;
  weight_import.layout.alignment =
      ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH;
  weight_import.binding_slot = ID4_QWEN3_VL_PREFILL_STAGE_WEIGHT_BINDING_SLOT;
  weight_import.offset = weight_span.buffer_offset;
  weight_import.flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;

  id4_pipeline_tensor_t weight_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      builder, &weight_import, &weight_tensor));

  id4_pipeline_tensor_layout_t output_layout;
  memset(&output_layout, 0, sizeof(output_layout));
  output_layout.name = IREE_SV("qwen3_vl.prefill.hidden_states.rmsnorm");
  output_layout.shape = hidden_shape;
  output_layout.byte_length = hidden_states_byte_length;
  output_layout.alignment = ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH;

  id4_pipeline_tensor_t output_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_acquire_tensor(
      builder, &output_layout, &output_tensor));

  char token_count_value_buffer[16];
  char hidden_size_value_buffer[16];
  char specialization_key_buffer[160];
  iree_string_view_t token_count_value = iree_string_view_empty();
  iree_string_view_t hidden_size_value = iree_string_view_empty();
  iree_string_view_t specialization_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_u32(
      stage->token_count, token_count_value_buffer,
      IREE_ARRAYSIZE(token_count_value_buffer), &token_count_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_u32(
      ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE, hidden_size_value_buffer,
      IREE_ARRAYSIZE(hidden_size_value_buffer), &hidden_size_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_specialization_key(
      stage, specialization_key_buffer,
      IREE_ARRAYSIZE(specialization_key_buffer), &specialization_key));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          // Config key for the token count.
          .key = IREE_SV("id4.qwen3_vl.rmsnorm.token_count"),
          // Config value for the token count.
          .value = token_count_value,
      },
      {
          // Config key for the hidden size.
          .key = IREE_SV("id4.qwen3_vl.rmsnorm.hidden_size"),
          // Config value for the hidden size.
          .value = hidden_size_value,
      },
  };
  id4_pipeline_region_loom_kernel_t kernel;
  memset(&kernel, 0, sizeof(kernel));
  kernel.specialization_key = specialization_key;
  kernel.module_path = IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_RMSNORM_MODULE_PATH);
  kernel.function_name =
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_RMSNORM_FUNCTION_NAME);
  kernel.executable = hal_executable;
  kernel.function = function;
  kernel.binding_count = 3;
  kernel.constant_byte_length = 0;
  kernel.config_binding_count = IREE_ARRAYSIZE(config_bindings);
  kernel.config_bindings = config_bindings;

  iree_hal_dispatch_config_t dispatch_config =
      id4_qwen3_vl_prefill_stage_make_rmsnorm_dispatch_config(stage);

  id4_pipeline_region_dispatch_binding_t bindings[] = {
      {
          // Hidden states are read by the RMSNorm kernel.
          .tensor = input_tensor,
          // The dispatch reads hidden states.
          .access = ID4_PIPELINE_TENSOR_ACCESS_READ,
      },
      {
          // RMSNorm weights are read by the RMSNorm kernel.
          .tensor = weight_tensor,
          // The dispatch reads RMSNorm weights.
          .access = ID4_PIPELINE_TENSOR_ACCESS_READ,
      },
      {
          // Normalized hidden states are written by the RMSNorm kernel.
          .tensor = output_tensor,
          // The dispatch writes normalized hidden states.
          .access = ID4_PIPELINE_TENSOR_ACCESS_WRITE,
      },
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_region_dispatch_loom(
      builder, &kernel, dispatch_config, iree_const_byte_span_empty(),
      IREE_ARRAYSIZE(bindings), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  *out_output_tensor = output_tensor;
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_prefill_stage_author_dispatch_barrier(
    id4_pipeline_region_builder_t* builder) {
  iree_hal_memory_barrier_t memory_barrier = {
      // Dispatch writes completed before the barrier.
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      // Dispatch reads allowed after the barrier.
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
  };
  return id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL);
}

static iree_status_t id4_qwen3_vl_prefill_stage_author_q_projection(
    id4_qwen3_vl_prefill_stage_t* stage, id4_pipeline_region_builder_t* builder,
    id4_pipeline_tensor_t input_tensor, iree_hal_executable_t* hal_executable,
    iree_hal_executable_function_t function,
    iree_io_parameter_span_t weight_span) {
  iree_device_size_t hidden_states_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_hidden_states_byte_length(
      stage, &hidden_states_byte_length));

  id4_pipeline_tensor_shape_t hidden_shape =
      id4_qwen3_vl_prefill_stage_make_hidden_shape(stage);
  id4_pipeline_tensor_shape_t weight_shape =
      id4_qwen3_vl_prefill_stage_make_projection_weight_shape(stage);

  id4_pipeline_tensor_import_t weight_import;
  memset(&weight_import, 0, sizeof(weight_import));
  weight_import.layout.name =
      IREE_SV("qwen3_vl.prefill.layer0.self_attn.q_proj.weight");
  weight_import.layout.shape = weight_shape;
  weight_import.layout.byte_length = weight_span.length;
  weight_import.layout.alignment =
      ID4_QWEN3_VL_PREFILL_STAGE_BF16_ELEMENT_BYTE_LENGTH;
  weight_import.binding_slot = ID4_QWEN3_VL_PREFILL_STAGE_WEIGHT_BINDING_SLOT;
  weight_import.offset = weight_span.buffer_offset;
  weight_import.flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;

  id4_pipeline_tensor_t weight_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      builder, &weight_import, &weight_tensor));

  id4_pipeline_tensor_import_t output_import;
  memset(&output_import, 0, sizeof(output_import));
  output_import.layout.name =
      IREE_SV("qwen3_vl.prefill.layer0.self_attn.q_proj.output");
  output_import.layout.shape = hidden_shape;
  output_import.layout.byte_length = hidden_states_byte_length;
  output_import.layout.alignment =
      ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH;
  output_import.binding_slot = ID4_QWEN3_VL_PREFILL_STAGE_OUTPUT_BINDING_SLOT;
  output_import.offset = 0;

  id4_pipeline_tensor_t output_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      builder, &output_import, &output_tensor));

  char token_count_value_buffer[16];
  char hidden_size_value_buffer[16];
  char specialization_key_buffer[192];
  iree_string_view_t token_count_value = iree_string_view_empty();
  iree_string_view_t hidden_size_value = iree_string_view_empty();
  iree_string_view_t specialization_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_u32(
      stage->token_count, token_count_value_buffer,
      IREE_ARRAYSIZE(token_count_value_buffer), &token_count_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_u32(
      ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE, hidden_size_value_buffer,
      IREE_ARRAYSIZE(hidden_size_value_buffer), &hidden_size_value));
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_prefill_stage_format_q_projection_specialization_key(
          stage, specialization_key_buffer,
          IREE_ARRAYSIZE(specialization_key_buffer), &specialization_key));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          // Config key for the token count.
          .key = IREE_SV("id4.qwen3_vl.linear.token_count"),
          // Config value for the token count.
          .value = token_count_value,
      },
      {
          // Config key for the linear input size.
          .key = IREE_SV("id4.qwen3_vl.linear.input_size"),
          // Config value for the linear input size.
          .value = hidden_size_value,
      },
      {
          // Config key for the linear output size.
          .key = IREE_SV("id4.qwen3_vl.linear.output_size"),
          // Config value for the linear output size.
          .value = hidden_size_value,
      },
  };
  id4_pipeline_region_loom_kernel_t kernel;
  memset(&kernel, 0, sizeof(kernel));
  kernel.specialization_key = specialization_key;
  kernel.module_path =
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_MODULE_PATH);
  kernel.function_name =
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_FUNCTION_NAME);
  kernel.executable = hal_executable;
  kernel.function = function;
  kernel.binding_count = 3;
  kernel.constant_byte_length = 0;
  kernel.config_binding_count = IREE_ARRAYSIZE(config_bindings);
  kernel.config_bindings = config_bindings;

  iree_hal_dispatch_config_t dispatch_config =
      id4_qwen3_vl_prefill_stage_make_q_projection_dispatch_config(stage);

  id4_pipeline_region_dispatch_binding_t bindings[] = {
      {
          // Normalized hidden states are read by the Q projection.
          .tensor = input_tensor,
          // The dispatch reads normalized hidden states.
          .access = ID4_PIPELINE_TENSOR_ACCESS_READ,
      },
      {
          // Q projection weights are read by the linear kernel.
          .tensor = weight_tensor,
          // The dispatch reads Q projection weights.
          .access = ID4_PIPELINE_TENSOR_ACCESS_READ,
      },
      {
          // Q projection outputs are written by the linear kernel.
          .tensor = output_tensor,
          // The dispatch writes Q projection outputs.
          .access = ID4_PIPELINE_TENSOR_ACCESS_WRITE,
      },
  };
  return id4_pipeline_region_dispatch_loom(
      builder, &kernel, dispatch_config, iree_const_byte_span_empty(),
      IREE_ARRAYSIZE(bindings), bindings, IREE_HAL_DISPATCH_FLAG_NONE);
}

static iree_status_t id4_qwen3_vl_prefill_stage_author_region(
    id4_qwen3_vl_prefill_stage_t* stage, id4_pipeline_region_builder_t* builder,
    iree_hal_executable_t* rmsnorm_executable,
    iree_hal_executable_function_t rmsnorm_function,
    iree_hal_executable_t* q_projection_executable,
    iree_hal_executable_function_t q_projection_function) {
  id4_qwen3_vl_prefill_stage_parameter_layout_t parameter_layout;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_prefill_stage_parameter_layout(stage, &parameter_layout));

  id4_pipeline_tensor_t normalized_tensor;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_author_rmsnorm(
      stage, builder, rmsnorm_executable, rmsnorm_function,
      parameter_layout.rmsnorm_weight, &normalized_tensor));
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_prefill_stage_author_dispatch_barrier(builder));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_author_q_projection(
      stage, builder, normalized_tensor, q_projection_executable,
      q_projection_function, parameter_layout.q_projection_weight));
  return id4_pipeline_region_release_tensor(builder, normalized_tensor);
}

static iree_status_t id4_qwen3_vl_prefill_stage_create_dry_run_builder(
    id4_qwen3_vl_prefill_stage_t* stage, iree_arena_block_pool_t* block_pool,
    id4_pipeline_region_builder_t** out_builder) {
  id4_pipeline_region_builder_create_options_t builder_options;
  memset(&builder_options, 0, sizeof(builder_options));
  builder_options.structure_size = sizeof(builder_options);
  builder_options.region_name = IREE_SV("qwen3_vl.prefill.forward");
  builder_options.mode = ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN;
  builder_options.block_pool = block_pool;
  builder_options.binding_capacity = ID4_QWEN3_VL_PREFILL_STAGE_BINDING_COUNT;
  builder_options.local_binding_slot =
      ID4_QWEN3_VL_PREFILL_STAGE_LOCAL_BINDING_SLOT;

  id4_pipeline_region_builder_t* builder = NULL;
  iree_status_t status = id4_pipeline_region_builder_create(
      &builder_options, stage->host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_author_region(
        stage, builder, NULL, iree_hal_executable_function_invalid(), NULL,
        iree_hal_executable_function_invalid());
  }
  if (iree_status_is_ok(status)) {
    *out_builder = builder;
  } else {
    id4_pipeline_region_builder_destroy(builder);
  }
  return status;
}

static void id4_qwen3_vl_prefill_stage_make_host_tensor_buffer_params(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_usage_t usage,
    iree_hal_buffer_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                     IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                     IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  out_params->access = IREE_HAL_MEMORY_ACCESS_ALL;
  out_params->usage = usage | IREE_HAL_BUFFER_USAGE_MAPPING;
  out_params->queue_affinity = queue_affinity;
  out_params->min_alignment =
      ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH;
}

static iree_status_t id4_qwen3_vl_prefill_stage_create_plan(
    const id4_pipeline_stage_plan_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan,
    id4_pipeline_stage_t* base_stage) {
  id4_qwen3_vl_prefill_stage_t* stage =
      id4_qwen3_vl_prefill_stage_cast(base_stage);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   stage->host_allocator, &block_pool);
  id4_pipeline_region_builder_t* builder = NULL;
  iree_status_t status = id4_qwen3_vl_prefill_stage_create_dry_run_builder(
      stage, &block_pool, &builder);

  id4_pipeline_region_statistics_t region_statistics;
  memset(&region_statistics, 0, sizeof(region_statistics));
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_statistics(builder, &region_statistics);
  }
  iree_host_size_t kernel_count = 0;
  if (iree_status_is_ok(status)) {
    kernel_count = id4_pipeline_region_builder_kernel_count(builder);
  }
  id4_pipeline_kernel_plan_t* kernels = NULL;
  if (iree_status_is_ok(status) && kernel_count != 0) {
    kernels = (id4_pipeline_kernel_plan_t*)iree_alloca(kernel_count *
                                                       sizeof(kernels[0]));
    memset(kernels, 0, kernel_count * sizeof(kernels[0]));
    for (iree_host_size_t i = 0; i < kernel_count; ++i) {
      const id4_pipeline_region_kernel_plan_t* region_kernel =
          id4_pipeline_region_builder_kernel_at(builder, i);
      if (!region_kernel) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "Qwen3-VL prefill kernel plan %" PRIhsz " is missing", i);
        break;
      }
      kernels[i].specialization_key = region_kernel->specialization_key;
      kernels[i].module_path = region_kernel->module_path;
      kernels[i].function_name = region_kernel->function_name;
      kernels[i].placement_id = 0;
      kernels[i].config_binding_count = region_kernel->config_binding_count;
      kernels[i].config_bindings = region_kernel->config_bindings;
    }
  }

  iree_device_size_t hidden_states_byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_hidden_states_byte_length(
        stage, &hidden_states_byte_length);
  }
  id4_qwen3_vl_prefill_stage_parameter_layout_t parameter_layout;
  if (iree_status_is_ok(status)) {
    status =
        id4_qwen3_vl_prefill_stage_parameter_layout(stage, &parameter_layout);
  }

  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = options->device_index;
  placement.queue_affinity = options->queue_affinity;

  id4_pipeline_parameter_request_t weight_requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_INPUT_RMSNORM_WEIGHT_KEY),
          parameter_layout.rmsnorm_weight),
      id4_pipeline_parameter_request(
          IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_WEIGHT_KEY),
          parameter_layout.q_projection_weight),
  };

  id4_pipeline_parameter_slab_plan_t parameter_slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_PARAMETER_SCOPE),
          /*placement_id=*/0, options->queue_affinity,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
          parameter_layout.total_byte_length, /*alignment=*/16,
          IREE_ARRAYSIZE(weight_requests), weight_requests);

  iree_hal_buffer_params_t input_params;
  id4_qwen3_vl_prefill_stage_make_host_tensor_buffer_params(
      options->queue_affinity,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      &input_params);
  iree_hal_buffer_params_t output_params;
  id4_qwen3_vl_prefill_stage_make_host_tensor_buffer_params(
      options->queue_affinity,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      &output_params);
  iree_hal_buffer_params_t local_slab_params;
  memset(&local_slab_params, 0, sizeof(local_slab_params));
  local_slab_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_slab_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  local_slab_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  local_slab_params.queue_affinity = options->queue_affinity;
  local_slab_params.min_alignment = 16;
  id4_pipeline_memory_slab_plan_t memory_slabs[] = {
      {
          // Input hidden-state slab for the prefill region.
          .name = IREE_SV("qwen3_vl.prefill.hidden_states.input"),
          // Plan-local placement identifier.
          .placement_id = 0,
          // Issue-time binding-table slot for input hidden states.
          .binding_slot = ID4_QWEN3_VL_PREFILL_STAGE_INPUT_BINDING_SLOT,
          // HAL buffer parameters for input hidden states.
          .params = input_params,
          // Input hidden-state tensor byte length.
          .byte_length = hidden_states_byte_length,
          // Input hidden-state base alignment.
          .alignment = ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH,
          // Input hidden-state live byte count.
          .high_water_mark = hidden_states_byte_length,
      },
      {
          // Output hidden-state slab for the prefill region.
          .name = IREE_SV("qwen3_vl.prefill.hidden_states.output"),
          // Plan-local placement identifier.
          .placement_id = 0,
          // Issue-time binding-table slot for output hidden states.
          .binding_slot = ID4_QWEN3_VL_PREFILL_STAGE_OUTPUT_BINDING_SLOT,
          // HAL buffer parameters for output hidden states.
          .params = output_params,
          // Output hidden-state tensor byte length.
          .byte_length = hidden_states_byte_length,
          // Output hidden-state base alignment.
          .alignment = ID4_QWEN3_VL_PREFILL_STAGE_F32_ELEMENT_BYTE_LENGTH,
          // Output hidden-state live byte count.
          .high_water_mark = hidden_states_byte_length,
      },
      {
          // Local transient slab for reusable prefill region storage.
          .name = IREE_SV("qwen3_vl.prefill.local"),
          // Plan-local placement identifier.
          .placement_id = 0,
          // Issue-time binding-table slot for local transient storage.
          .binding_slot = ID4_QWEN3_VL_PREFILL_STAGE_LOCAL_BINDING_SLOT,
          // HAL buffer parameters for local transient storage.
          .params = local_slab_params,
          // Required local transient slab byte length.
          .byte_length = region_statistics.local_slab_byte_length,
          // Local transient slab base alignment.
          .alignment = 16,
          // Peak local transient live byte count.
          .high_water_mark = region_statistics.local_slab_high_water_mark,
      },
  };

  id4_pipeline_region_plan_t region;
  memset(&region, 0, sizeof(region));
  region.name = IREE_SV("qwen3_vl.prefill.forward");
  region.placement_id = 0;
  region.binding_capacity = ID4_QWEN3_VL_PREFILL_STAGE_BINDING_COUNT;
  region.local_binding_slot = ID4_QWEN3_VL_PREFILL_STAGE_LOCAL_BINDING_SLOT;
  region.statistics = region_statistics;

  id4_pipeline_diagnostic_tap_plan_t output_tap;
  memset(&output_tap, 0, sizeof(output_tap));
  output_tap.name =
      IREE_SV("qwen3_vl.prefill.hidden_states.after_q_projection");
  output_tap.region_id = 0;
  output_tap.after_operation_ordinal = 2;
  output_tap.target_name =
      IREE_SV("qwen3_vl.prefill.layer0.self_attn.q_proj.output");

  if (iree_status_is_ok(status)) {
    id4_pipeline_plan_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.stage_name = IREE_SV("qwen3_vl.prefill");
    create_options.device_group = base_stage->services.device_group;
    create_options.placement_count = 1;
    create_options.placements = &placement;
    create_options.parameter_slab_count = 1;
    create_options.parameter_slabs = &parameter_slab;
    create_options.memory_slab_count = IREE_ARRAYSIZE(memory_slabs);
    create_options.memory_slabs = memory_slabs;
    create_options.kernel_count = kernel_count;
    create_options.kernels = kernels;
    create_options.region_count = 1;
    create_options.regions = &region;
    create_options.diagnostic_tap_count = 1;
    create_options.diagnostic_taps = &output_tap;
    create_options.diagnostics_sink = options->diagnostics_sink;
    status =
        id4_pipeline_plan_create(&create_options, host_allocator, out_plan);
  }
  id4_pipeline_region_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_qwen3_vl_prefill_stage_prepare_kernel_executable(
    id4_qwen3_vl_prefill_stage_t* stage,
    iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_kernel_library_t* kernel_library,
    iree_string_view_t module_path, iree_host_size_t config_binding_count,
    const id4_pipeline_kernel_config_binding_t* config_bindings,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_kernel_executable_t** out_executable) {
  if (!stage->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL prefill stage kernel cache is required "
                            "for preparation");
  }
  if (!stage->base.services.executable_cache) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL prefill stage HAL executable cache is required for "
        "preparation");
  }
  const id4_pipeline_kernel_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_kernel_library_lookup(kernel_library, module_path, &module));

  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = stage->base.services.executable_cache;
  prepare_options.queue_affinity = queue_affinity;
  prepare_options.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
  prepare_options.source_identifier = module->source_identifier;
  prepare_options.source_contents = module->source_contents;
  prepare_options.module_path = module->module_path;
  prepare_options.config_binding_count = config_binding_count;
  prepare_options.config_bindings = config_bindings;
  prepare_options.diagnostic_artifact_flags =
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_TEXT |
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON |
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_EMIT_MANIFEST_JSON;
  prepare_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_kernel_cache_prepare_executable(
      stage->kernel_cache, &prepare_options, out_executable);
}

static iree_status_t id4_qwen3_vl_prefill_stage_prepare_rmsnorm_executable(
    id4_qwen3_vl_prefill_stage_t* stage,
    iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_kernel_library_t* kernel_library,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_kernel_executable_t** out_executable) {
  char token_count_value_buffer[16];
  char hidden_size_value_buffer[16];
  iree_string_view_t token_count_value = iree_string_view_empty();
  iree_string_view_t hidden_size_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_u32(
      stage->token_count, token_count_value_buffer,
      IREE_ARRAYSIZE(token_count_value_buffer), &token_count_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_u32(
      ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE, hidden_size_value_buffer,
      IREE_ARRAYSIZE(hidden_size_value_buffer), &hidden_size_value));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          // Config key for the token count.
          .key = IREE_SV("id4.qwen3_vl.rmsnorm.token_count"),
          // Config value for the token count.
          .value = token_count_value,
      },
      {
          // Config key for the hidden size.
          .key = IREE_SV("id4.qwen3_vl.rmsnorm.hidden_size"),
          // Config value for the hidden size.
          .value = hidden_size_value,
      },
  };
  return id4_qwen3_vl_prefill_stage_prepare_kernel_executable(
      stage, queue_affinity, kernel_library,
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_RMSNORM_MODULE_PATH),
      IREE_ARRAYSIZE(config_bindings), config_bindings, diagnostics_sink,
      out_executable);
}

static iree_status_t id4_qwen3_vl_prefill_stage_prepare_q_projection_executable(
    id4_qwen3_vl_prefill_stage_t* stage,
    iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_kernel_library_t* kernel_library,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_kernel_executable_t** out_executable) {
  char token_count_value_buffer[16];
  char hidden_size_value_buffer[16];
  iree_string_view_t token_count_value = iree_string_view_empty();
  iree_string_view_t hidden_size_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_u32(
      stage->token_count, token_count_value_buffer,
      IREE_ARRAYSIZE(token_count_value_buffer), &token_count_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_format_u32(
      ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE, hidden_size_value_buffer,
      IREE_ARRAYSIZE(hidden_size_value_buffer), &hidden_size_value));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          // Config key for the token count.
          .key = IREE_SV("id4.qwen3_vl.linear.token_count"),
          // Config value for the token count.
          .value = token_count_value,
      },
      {
          // Config key for the linear input size.
          .key = IREE_SV("id4.qwen3_vl.linear.input_size"),
          // Config value for the linear input size.
          .value = hidden_size_value,
      },
      {
          // Config key for the linear output size.
          .key = IREE_SV("id4.qwen3_vl.linear.output_size"),
          // Config value for the linear output size.
          .value = hidden_size_value,
      },
  };
  return id4_qwen3_vl_prefill_stage_prepare_kernel_executable(
      stage, queue_affinity, kernel_library,
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_MODULE_PATH),
      IREE_ARRAYSIZE(config_bindings), config_bindings, diagnostics_sink,
      out_executable);
}

static iree_status_t id4_qwen3_vl_prefill_stage_allocate_hidden_buffer(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_buffer_usage_t usage, iree_device_size_t byte_length,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params;
  id4_qwen3_vl_prefill_stage_make_host_tensor_buffer_params(queue_affinity,
                                                            usage, &params);
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t id4_qwen3_vl_prefill_stage_join_prepare_readiness(
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (signal_semaphore_list.count == 0) return iree_ok_status();
  return iree_hal_semaphore_list_wait(signal_semaphore_list,
                                      iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_qwen3_vl_prefill_stage_record_region(
    id4_qwen3_vl_prefill_stage_t* stage, iree_hal_device_group_t* device_group,
    iree_host_size_t device_index, iree_hal_queue_affinity_t queue_affinity,
    id4_pipeline_kernel_executable_t* rmsnorm_executable,
    id4_pipeline_kernel_executable_t* q_projection_executable,
    id4_pipeline_prepared_region_t** out_prepared_region) {
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, device_index);
  if (!device) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL prefill stage placement device is required");
  }

  iree_hal_executable_t* rmsnorm_hal_executable =
      id4_pipeline_kernel_executable_hal_executable(rmsnorm_executable);
  iree_hal_executable_function_t rmsnorm_function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
      rmsnorm_hal_executable,
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_RMSNORM_FUNCTION_NAME),
      &rmsnorm_function));
  iree_hal_executable_t* q_projection_hal_executable =
      id4_pipeline_kernel_executable_hal_executable(q_projection_executable);
  iree_hal_executable_function_t q_projection_function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
      q_projection_hal_executable,
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_FUNCTION_NAME),
      &q_projection_function));

  iree_hal_command_buffer_t* command_buffer = NULL;
  id4_pipeline_region_builder_t* builder = NULL;
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   stage->host_allocator, &block_pool);

  iree_status_t status = iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, queue_affinity,
      ID4_QWEN3_VL_PREFILL_STAGE_BINDING_COUNT, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_create_options_t builder_options;
    memset(&builder_options, 0, sizeof(builder_options));
    builder_options.structure_size = sizeof(builder_options);
    builder_options.region_name = IREE_SV("qwen3_vl.prefill.forward");
    builder_options.mode = ID4_PIPELINE_REGION_BUILDER_MODE_RECORD;
    builder_options.block_pool = &block_pool;
    builder_options.command_buffer = command_buffer;
    builder_options.binding_capacity = ID4_QWEN3_VL_PREFILL_STAGE_BINDING_COUNT;
    builder_options.local_binding_slot =
        ID4_QWEN3_VL_PREFILL_STAGE_LOCAL_BINDING_SLOT;
    status = id4_pipeline_region_builder_create(
        &builder_options, stage->host_allocator, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_author_region(
        stage, builder, rmsnorm_hal_executable, rmsnorm_function,
        q_projection_hal_executable, q_projection_function);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_params_t local_slab_params;
    memset(&local_slab_params, 0, sizeof(local_slab_params));
    local_slab_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    local_slab_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    local_slab_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    local_slab_params.queue_affinity = queue_affinity;
    local_slab_params.min_alignment = 16;

    id4_pipeline_prepared_region_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.device_group = device_group;
    create_options.device_index = device_index;
    create_options.queue_affinity = queue_affinity;
    create_options.local_slab_params = local_slab_params;
    create_options.local_slab_alloca_flags = IREE_HAL_ALLOCA_FLAG_NONE;
    create_options.local_slab_dealloca_flags = IREE_HAL_DEALLOCA_FLAG_NONE;
    status = id4_pipeline_prepared_region_create(
        builder, &create_options, stage->host_allocator, out_prepared_region);
  }

  id4_pipeline_region_builder_destroy(builder);
  iree_hal_command_buffer_release(command_buffer);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static void id4_qwen3_vl_prefill_stage_bundle_payload_destroy(
    id4_pipeline_bundle_t* bundle, void* raw_payload) {
  (void)bundle;
  id4_qwen3_vl_prefill_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_prefill_stage_bundle_payload_t*)raw_payload;
  id4_pipeline_prepared_region_release(payload->prepared_region);
  id4_pipeline_kernel_executable_release(payload->q_projection_executable);
  id4_pipeline_kernel_executable_release(payload->rmsnorm_executable);
  iree_hal_buffer_release(payload->output_buffer);
  iree_hal_buffer_release(payload->input_buffer);
}

static void id4_qwen3_vl_prefill_stage_destroy(
    id4_pipeline_stage_t* base_stage) {
  id4_qwen3_vl_prefill_stage_t* stage =
      id4_qwen3_vl_prefill_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  id4_pipeline_stage_deinitialize(base_stage);
  iree_allocator_free(host_allocator, stage);
}

static iree_status_t id4_qwen3_vl_prefill_stage_load(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_qwen3_vl_prefill_stage_t* stage =
      id4_qwen3_vl_prefill_stage_cast(base_stage);
  stage->is_loaded = true;
  return id4_qwen3_vl_prefill_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.load"),
      IREE_SV("loaded Qwen3-VL prefill stage"));
}

static iree_status_t id4_qwen3_vl_prefill_stage_plan(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  const id4_qwen3_vl_prefill_stage_t* stage =
      id4_qwen3_vl_prefill_stage_const_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL prefill stage must be loaded before "
                            "planning");
  }
  return id4_qwen3_vl_prefill_stage_create_plan(options, stage->host_allocator,
                                                out_plan, base_stage);
}

static iree_status_t id4_qwen3_vl_prefill_stage_validate_prepare_inputs(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options) {
  if (!iree_string_view_equal(id4_pipeline_plan_stage_name(plan),
                              IREE_SV("qwen3_vl.prefill"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL prefill stage prepare requires a Qwen3-VL prefill plan");
  }
  if (id4_pipeline_plan_placement_count(plan) != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL prefill plan must have one placement");
  }
  if (id4_pipeline_plan_parameter_slab_count(plan) != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL prefill plan must have one parameter slab");
  }
  if (id4_pipeline_plan_memory_slab_count(plan) != 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL prefill plan must have three memory "
                            "slabs");
  }
  if (!options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL prefill kernel library is required");
  }
  if (!options->parameter_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL prefill parameter provider is required");
  }
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL prefill prepare requires a signal semaphore list for "
        "parameter loading readiness");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_prefill_stage_prepare(
    id4_pipeline_stage_t* base_stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  id4_qwen3_vl_prefill_stage_t* stage =
      id4_qwen3_vl_prefill_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL prefill stage must be loaded before "
                            "preparation");
  }
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_prefill_stage_validate_prepare_inputs(plan, options));

  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(plan, 0);
  iree_hal_device_group_t* device_group = id4_pipeline_plan_device_group(plan);
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, placement->device_index);
  if (!device) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL prefill stage placement device is required");
  }

  iree_device_size_t hidden_states_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_hidden_states_byte_length(
      stage, &hidden_states_byte_length));
  id4_qwen3_vl_prefill_stage_parameter_layout_t parameter_layout;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_prefill_stage_parameter_layout(stage, &parameter_layout));

  id4_pipeline_parameter_slab_set_t* parameter_slabs = NULL;
  id4_pipeline_kernel_executable_t* rmsnorm_executable = NULL;
  id4_pipeline_kernel_executable_t* q_projection_executable = NULL;
  id4_pipeline_prepared_region_t* prepared_region = NULL;
  iree_hal_buffer_t* input_buffer = NULL;
  iree_hal_buffer_t* output_buffer = NULL;
  id4_pipeline_bundle_t* bundle = NULL;
  bool parameter_load_submitted = false;

  iree_status_t status = id4_pipeline_plan_load_parameter_slabs(
      plan, options->parameter_provider, options->wait_semaphore_list,
      options->signal_semaphore_list, options->diagnostics_sink,
      stage->host_allocator, &parameter_slabs);
  parameter_load_submitted = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_prepare_rmsnorm_executable(
        stage, placement->queue_affinity, options->kernel_library,
        options->diagnostics_sink, &rmsnorm_executable);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_prepare_q_projection_executable(
        stage, placement->queue_affinity, options->kernel_library,
        options->diagnostics_sink, &q_projection_executable);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_allocate_hidden_buffer(
        device, placement->queue_affinity,
        IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
            IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
        hidden_states_byte_length, &input_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_allocate_hidden_buffer(
        device, placement->queue_affinity,
        IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
            IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
        hidden_states_byte_length, &output_buffer);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_map_zero(output_buffer, 0, hidden_states_byte_length);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_record_region(
        stage, device_group, placement->device_index, placement->queue_affinity,
        rmsnorm_executable, q_projection_executable, &prepared_region);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_bundle_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.plan = plan;
    create_options.parameter_slabs = parameter_slabs;
    create_options.readiness_semaphore_list = options->signal_semaphore_list;
    create_options.payload_size =
        sizeof(id4_qwen3_vl_prefill_stage_bundle_payload_t);
    create_options.payload_alignment =
        iree_alignof(id4_qwen3_vl_prefill_stage_bundle_payload_t);
    create_options.payload_destroy =
        id4_qwen3_vl_prefill_stage_bundle_payload_destroy;
    status = id4_pipeline_bundle_create(&create_options, stage->host_allocator,
                                        &bundle);
  }
  if (iree_status_is_ok(status)) {
    id4_qwen3_vl_prefill_stage_bundle_payload_t* payload =
        (id4_qwen3_vl_prefill_stage_bundle_payload_t*)
            id4_pipeline_bundle_payload(bundle);
    payload->rmsnorm_executable = rmsnorm_executable;
    rmsnorm_executable = NULL;
    payload->q_projection_executable = q_projection_executable;
    q_projection_executable = NULL;
    payload->prepared_region = prepared_region;
    prepared_region = NULL;
    payload->input_buffer = input_buffer;
    input_buffer = NULL;
    payload->output_buffer = output_buffer;
    output_buffer = NULL;
    payload->hidden_states_byte_length = hidden_states_byte_length;
    payload->bindings[ID4_QWEN3_VL_PREFILL_STAGE_INPUT_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Input hidden-state buffer read by the RMSNorm kernel.
            .buffer = payload->input_buffer,
            // Input hidden-state tensor starts at byte zero.
            .offset = 0,
            // Full input hidden-state tensor byte length.
            .length = hidden_states_byte_length,
        };
    payload->bindings[ID4_QWEN3_VL_PREFILL_STAGE_WEIGHT_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Loaded packed parameter slab retained by the bundle.
            .buffer =
                id4_pipeline_parameter_slab_set_buffer_at(parameter_slabs, 0),
            // Packed parameter slab starts at byte zero.
            .offset = 0,
            // Full packed parameter slab byte length.
            .length = parameter_layout.total_byte_length,
        };
    payload->bindings[ID4_QWEN3_VL_PREFILL_STAGE_OUTPUT_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Output hidden-state buffer written by the Q projection kernel.
            .buffer = payload->output_buffer,
            // Output hidden-state tensor starts at byte zero.
            .offset = 0,
            // Full output hidden-state tensor byte length.
            .length = hidden_states_byte_length,
        };
    payload->bindings[ID4_QWEN3_VL_PREFILL_STAGE_LOCAL_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Local slab binding is patched by prepared-region issue.
            .buffer = NULL,
            // Local slab binding starts at byte zero once patched.
            .offset = 0,
            // Local slab length is supplied by prepared-region issue.
            .length = 0,
        };
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_prefill_stage_emit_lifecycle(
        options->diagnostics_sink, IREE_SV("stage.prepare"),
        IREE_SV("prepared Qwen3-VL prefill bundle"));
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    if (parameter_load_submitted) {
      status = iree_status_join(
          status, id4_qwen3_vl_prefill_stage_join_prepare_readiness(
                      options->signal_semaphore_list));
    }
    id4_pipeline_bundle_release(bundle);
    id4_pipeline_prepared_region_release(prepared_region);
    id4_pipeline_kernel_executable_release(q_projection_executable);
    id4_pipeline_kernel_executable_release(rmsnorm_executable);
    iree_hal_buffer_release(output_buffer);
    iree_hal_buffer_release(input_buffer);
  }
  id4_pipeline_parameter_slab_set_release(parameter_slabs);
  return status;
}

static iree_status_t id4_qwen3_vl_prefill_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  (void)base_stage;
  id4_qwen3_vl_prefill_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_prefill_stage_bundle_payload_t*)id4_pipeline_bundle_payload(
          bundle);
  if (!payload || !payload->prepared_region) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL prefill bundle payload is not prepared");
  }

  IREE_RETURN_IF_ERROR(id4_qwen3_vl_prefill_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.issue"),
      IREE_SV("issued Qwen3-VL prefill bundle")));

  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(readiness_list.count,
                                  options->wait_semaphore_list.count,
                                  &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL prefill stage wait list count overflow");
  }
  iree_hal_semaphore_t** wait_semaphores = NULL;
  uint64_t* wait_payload_values = NULL;
  if (wait_count != 0) {
    wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        wait_count * sizeof(wait_semaphores[0]));
    wait_payload_values =
        (uint64_t*)iree_alloca(wait_count * sizeof(wait_payload_values[0]));
  }
  for (iree_host_size_t i = 0; i < readiness_list.count; ++i) {
    wait_semaphores[i] = readiness_list.semaphores[i];
    wait_payload_values[i] = readiness_list.payload_values[i];
  }
  for (iree_host_size_t i = 0; i < options->wait_semaphore_list.count; ++i) {
    const iree_host_size_t target_index = readiness_list.count + i;
    wait_semaphores[target_index] = options->wait_semaphore_list.semaphores[i];
    wait_payload_values[target_index] =
        options->wait_semaphore_list.payload_values[i];
  }
  iree_hal_semaphore_list_t combined_wait_list = {
      // Number of semaphores waited on by the issue.
      .count = wait_count,
      // Stack-local semaphore pointer array valid for this issue call.
      .semaphores = wait_semaphores,
      // Stack-local payload value array valid for this issue call.
      .payload_values = wait_payload_values,
  };

  iree_hal_buffer_binding_table_t binding_table = {
      // Number of issue-time binding table slots.
      .count = ID4_QWEN3_VL_PREFILL_STAGE_BINDING_COUNT,
      // Fixed binding storage owned by the bundle payload.
      .bindings = payload->bindings,
  };
  id4_pipeline_prepared_region_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = combined_wait_list;
  issue_options.signal_semaphore_list = options->signal_semaphore_list;
  issue_options.binding_table = binding_table;
  issue_options.execute_flags = IREE_HAL_EXECUTE_FLAG_NONE;
  return id4_pipeline_prepared_region_issue(payload->prepared_region,
                                            &issue_options);
}

static const id4_pipeline_stage_vtable_t id4_qwen3_vl_prefill_stage_vtable = {
    // Destroys the Qwen3-VL prefill stage.
    id4_qwen3_vl_prefill_stage_destroy,
    // Loads immutable Qwen3-VL prefill stage state.
    id4_qwen3_vl_prefill_stage_load,
    // Builds an inspectable Qwen3-VL prefill execution plan.
    id4_qwen3_vl_prefill_stage_plan,
    // Prepares a reusable Qwen3-VL prefill execution bundle.
    id4_qwen3_vl_prefill_stage_prepare,
    // Issues one Qwen3-VL prefill bundle execution.
    id4_qwen3_vl_prefill_stage_issue,
};

iree_status_t id4_qwen3_vl_prefill_stage_create(
    const id4_qwen3_vl_prefill_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = NULL;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_prefill_stage_validate_create_options(options));

  id4_qwen3_vl_prefill_stage_t* stage = NULL;
  bool base_initialized = false;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*stage), (void**)&stage);
  if (iree_status_is_ok(status)) {
    memset(stage, 0, sizeof(*stage));
    stage->host_allocator = host_allocator;
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_stage_initialize(&id4_qwen3_vl_prefill_stage_vtable,
                                           &options->services, &stage->base);
    base_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    stage->kernel_cache = options->kernel_cache;
    id4_pipeline_kernel_cache_retain(stage->kernel_cache);
    stage->token_count = options->token_count;
  }
  if (iree_status_is_ok(status)) {
    *out_stage = &stage->base;
  } else if (stage) {
    if (base_initialized) {
      id4_pipeline_stage_release(&stage->base);
    } else {
      iree_allocator_free(host_allocator, stage);
    }
  }
  return status;
}

iree_hal_buffer_t* id4_qwen3_vl_prefill_stage_bundle_input_buffer(
    id4_pipeline_bundle_t* bundle) {
  id4_qwen3_vl_prefill_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_prefill_stage_bundle_payload_t*)id4_pipeline_bundle_payload(
          bundle);
  return payload ? payload->input_buffer : NULL;
}

iree_hal_buffer_t* id4_qwen3_vl_prefill_stage_bundle_output_buffer(
    id4_pipeline_bundle_t* bundle) {
  id4_qwen3_vl_prefill_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_prefill_stage_bundle_payload_t*)id4_pipeline_bundle_payload(
          bundle);
  return payload ? payload->output_buffer : NULL;
}

iree_device_size_t id4_qwen3_vl_prefill_stage_bundle_hidden_states_byte_length(
    const id4_pipeline_bundle_t* bundle) {
  const id4_qwen3_vl_prefill_stage_bundle_payload_t* payload =
      (const id4_qwen3_vl_prefill_stage_bundle_payload_t*)
          id4_pipeline_bundle_const_payload(bundle);
  return payload ? payload->hidden_states_byte_length : 0;
}
