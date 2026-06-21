// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/region.h"
#include "iree/base/internal/arena.h"

#define ID4_QWEN3_VL_STAGE_BINDING_COUNT 4
#define ID4_QWEN3_VL_STAGE_SELECTED_HIDDEN_STATES_BINDING_SLOT 0
#define ID4_QWEN3_VL_STAGE_TOKEN_WEIGHTS_BINDING_SLOT 1
#define ID4_QWEN3_VL_STAGE_CONDITION_BINDING_SLOT 2
#define ID4_QWEN3_VL_STAGE_LOCAL_BINDING_SLOT 3
#define ID4_QWEN3_VL_STAGE_MAX_CONDITION_ELEMENT_COUNT ((uint64_t)1048576)
#define ID4_QWEN3_VL_STAGE_STATS_BYTE_LENGTH 8

typedef struct id4_qwen3_vl_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Source identifier owned by the stage.
  iree_string_view_t source_identifier;
  // Textual Loom source contents owned by the stage.
  iree_const_byte_span_t source_contents;
  // Loom module name owned by the stage.
  iree_string_view_t module_name;
  // HAL executable identifier owned by the stage.
  iree_string_view_t executable_identifier;
  // Exported apply-token-weights HAL function name owned by the stage.
  iree_string_view_t apply_token_weights_function_name;
  // Exported normalize-token-weights HAL function name owned by the stage.
  iree_string_view_t normalize_token_weights_function_name;
  // Number of hidden rows in stable-diffusion.cpp tensor order.
  uint32_t hidden_row_count;
  // Number of token columns in stable-diffusion.cpp tensor order.
  uint32_t token_count;
  // Configured X workgroup size.
  uint32_t workgroup_size_x;
  // True after load has completed.
  bool is_loaded;
} id4_qwen3_vl_stage_t;

typedef struct id4_qwen3_vl_stage_bundle_payload_t {
  // Prepared kernel executable retained for command-buffer validity.
  id4_pipeline_kernel_executable_t* executable;
  // Prepared reusable region issued by the Qwen3-VL stage.
  id4_pipeline_prepared_region_t* prepared_region;
  // Selected-hidden-states input buffer retained for issue-time binding.
  iree_hal_buffer_t* selected_hidden_states_buffer;
  // Token-weights input buffer retained for issue-time binding.
  iree_hal_buffer_t* token_weights_buffer;
  // Condition output buffer retained for issue-time binding and readback.
  iree_hal_buffer_t* condition_buffer;
  // Token-weights tensor byte length.
  iree_device_size_t token_weights_byte_length;
  // Condition tensor byte length.
  iree_device_size_t condition_byte_length;
  // Fixed issue-time binding table storage.
  iree_hal_buffer_binding_t bindings[ID4_QWEN3_VL_STAGE_BINDING_COUNT];
} id4_qwen3_vl_stage_bundle_payload_t;

static id4_qwen3_vl_stage_t* id4_qwen3_vl_stage_cast(
    id4_pipeline_stage_t* base_stage) {
  return (id4_qwen3_vl_stage_t*)base_stage;
}

static const id4_qwen3_vl_stage_t* id4_qwen3_vl_stage_const_cast(
    const id4_pipeline_stage_t* base_stage) {
  return (const id4_qwen3_vl_stage_t*)base_stage;
}

static iree_status_t id4_qwen3_vl_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_qwen3_vl_stage_copy_string(
    iree_string_view_t value, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  memset(out_value, 0, sizeof(*out_value));
  if (iree_string_view_is_empty(value)) return iree_ok_status();

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, value.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, value.data, value.size);
  storage[value.size] = 0;
  *out_value = iree_make_string_view(storage, value.size);
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_copy_bytes(
    iree_const_byte_span_t value, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  memset(out_value, 0, sizeof(*out_value));
  if (value.data_length == 0) return iree_ok_status();

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, value.data_length, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, value.data, value.data_length);
  *out_value = iree_make_const_byte_span(storage, value.data_length);
  return iree_ok_status();
}

static void id4_qwen3_vl_stage_free_string(iree_string_view_t* value,
                                           iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static void id4_qwen3_vl_stage_free_bytes(iree_const_byte_span_t* value,
                                          iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static uint64_t id4_qwen3_vl_stage_condition_element_count(
    const id4_qwen3_vl_stage_t* stage) {
  return (uint64_t)stage->hidden_row_count * stage->token_count;
}

static iree_status_t id4_qwen3_vl_stage_condition_byte_length(
    uint64_t element_count, iree_device_size_t* out_byte_length) {
  if (!iree_device_size_checked_mul(element_count,
                                    ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH,
                                    out_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL condition byte length overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_token_weights_byte_length(
    uint32_t token_count, iree_device_size_t* out_byte_length) {
  if (!iree_device_size_checked_mul(token_count,
                                    ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH,
                                    out_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL token weights byte length overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_workgroups_x(
    uint64_t element_count, uint32_t workgroup_size_x,
    uint32_t* out_workgroups_x) {
  const uint64_t workgroups_x =
      (element_count + workgroup_size_x - 1) / workgroup_size_x;
  if (workgroups_x > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL condition workgroup count overflow");
  }
  *out_workgroups_x = (uint32_t)workgroups_x;
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_validate_create_options(
    const id4_qwen3_vl_stage_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Qwen3-VL stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen3-VL stage create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage device group is required");
  }
  if (iree_string_view_is_empty(options->source_identifier)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage source identifier is required");
  }
  if (iree_string_view_is_empty(options->module_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage module name is required");
  }
  if (iree_string_view_is_empty(options->executable_identifier)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage executable identifier is required");
  }
  if (iree_string_view_is_empty(options->apply_token_weights_function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage apply-token-weights function name "
                            "is required");
  }
  if (iree_string_view_is_empty(
          options->normalize_token_weights_function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage normalize-token-weights function "
                            "name is required");
  }
  if (options->hidden_row_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL hidden row count must be nonzero");
  }
  if (options->token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL token count must be nonzero");
  }
  const uint64_t element_count =
      (uint64_t)options->hidden_row_count * options->token_count;
  if (element_count > ID4_QWEN3_VL_STAGE_MAX_CONDITION_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Qwen3-VL condition element count %" PRIu64 " exceeds max %" PRIu64,
        element_count, ID4_QWEN3_VL_STAGE_MAX_CONDITION_ELEMENT_COUNT);
  }
  if (options->workgroup_size_x == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL workgroup size must be nonzero");
  }
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_stage_condition_byte_length(element_count, &byte_length));
  return id4_qwen3_vl_stage_token_weights_byte_length(options->token_count,
                                                      &byte_length);
}

static iree_status_t id4_qwen3_vl_stage_format_u64(
    uint64_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%" PRIu64, value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format Qwen3-VL config value");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_format_u32(
    uint32_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%" PRIu32, value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format Qwen3-VL config value");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t id4_qwen3_vl_stage_lifecycle_event(
    iree_string_view_t key, iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the concrete Qwen3-VL stage.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name used across Qwen3-VL diagnostics.
      .stage_name = IREE_SV("qwen3_vl"),
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_qwen3_vl_stage_emit_lifecycle(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_qwen3_vl_stage_lifecycle_event(key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static id4_pipeline_tensor_shape_t id4_qwen3_vl_stage_make_condition_shape(
    const id4_qwen3_vl_stage_t* stage) {
  id4_pipeline_tensor_shape_t shape;
  memset(&shape, 0, sizeof(shape));
  shape.rank = 2;
  shape.dims[0] = stage->hidden_row_count;
  shape.dims[1] = stage->token_count;
  return shape;
}

static id4_pipeline_tensor_shape_t id4_qwen3_vl_stage_make_token_shape(
    const id4_qwen3_vl_stage_t* stage) {
  id4_pipeline_tensor_shape_t shape;
  memset(&shape, 0, sizeof(shape));
  shape.rank = 1;
  shape.dims[0] = stage->token_count;
  return shape;
}

static id4_pipeline_tensor_shape_t id4_qwen3_vl_stage_make_stats_shape(void) {
  id4_pipeline_tensor_shape_t shape;
  memset(&shape, 0, sizeof(shape));
  shape.rank = 1;
  shape.dims[0] = 2;
  return shape;
}

static iree_status_t id4_qwen3_vl_stage_author_region(
    id4_qwen3_vl_stage_t* stage, id4_pipeline_region_builder_t* builder,
    iree_hal_executable_t* hal_executable,
    iree_hal_executable_function_t apply_function,
    iree_hal_executable_function_t normalize_function) {
  const uint64_t element_count =
      id4_qwen3_vl_stage_condition_element_count(stage);
  iree_device_size_t condition_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_condition_byte_length(
      element_count, &condition_byte_length));
  iree_device_size_t token_weights_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_token_weights_byte_length(
      stage->token_count, &token_weights_byte_length));

  id4_pipeline_tensor_shape_t condition_shape =
      id4_qwen3_vl_stage_make_condition_shape(stage);
  id4_pipeline_tensor_shape_t token_shape =
      id4_qwen3_vl_stage_make_token_shape(stage);
  id4_pipeline_tensor_import_t selected_import;
  memset(&selected_import, 0, sizeof(selected_import));
  selected_import.layout.name =
      IREE_SV("qwen3_vl.encoder.selected_hidden_states");
  selected_import.layout.shape = condition_shape;
  selected_import.layout.byte_length = condition_byte_length;
  selected_import.layout.alignment = ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH;
  selected_import.binding_slot =
      ID4_QWEN3_VL_STAGE_SELECTED_HIDDEN_STATES_BINDING_SLOT;
  selected_import.offset = 0;
  selected_import.flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;

  id4_pipeline_tensor_t selected_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      builder, &selected_import, &selected_tensor));

  id4_pipeline_tensor_import_t token_weights_import;
  memset(&token_weights_import, 0, sizeof(token_weights_import));
  token_weights_import.layout.name = IREE_SV("qwen3_vl.prompt.token_weights");
  token_weights_import.layout.shape = token_shape;
  token_weights_import.layout.byte_length = token_weights_byte_length;
  token_weights_import.layout.alignment =
      ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH;
  token_weights_import.binding_slot =
      ID4_QWEN3_VL_STAGE_TOKEN_WEIGHTS_BINDING_SLOT;
  token_weights_import.offset = 0;
  token_weights_import.flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;

  id4_pipeline_tensor_t token_weights_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      builder, &token_weights_import, &token_weights_tensor));

  id4_pipeline_tensor_import_t condition_import;
  memset(&condition_import, 0, sizeof(condition_import));
  condition_import.layout.name = IREE_SV("qwen3_vl.encoder.condition");
  condition_import.layout.shape = condition_shape;
  condition_import.layout.byte_length = condition_byte_length;
  condition_import.layout.alignment =
      ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH;
  condition_import.binding_slot = ID4_QWEN3_VL_STAGE_CONDITION_BINDING_SLOT;
  condition_import.offset = 0;

  id4_pipeline_tensor_t condition_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      builder, &condition_import, &condition_tensor));

  id4_pipeline_tensor_layout_t stats_layout;
  memset(&stats_layout, 0, sizeof(stats_layout));
  stats_layout.name = IREE_SV("qwen3_vl.condition.stats");
  stats_layout.shape = id4_qwen3_vl_stage_make_stats_shape();
  stats_layout.byte_length = ID4_QWEN3_VL_STAGE_STATS_BYTE_LENGTH;
  stats_layout.alignment = 8;

  id4_pipeline_tensor_t stats_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_acquire_tensor(
      builder, &stats_layout, &stats_tensor));

  id4_pipeline_region_kernel_t apply_kernel;
  memset(&apply_kernel, 0, sizeof(apply_kernel));
  apply_kernel.name = stage->apply_token_weights_function_name;
  apply_kernel.executable = hal_executable;
  apply_kernel.function = apply_function;
  apply_kernel.binding_count = 4;
  apply_kernel.constant_byte_length = 0;

  iree_hal_dispatch_config_t apply_dispatch_config =
      iree_hal_make_static_dispatch_config(1, 1, 1);
  apply_dispatch_config.workgroup_size[0] = stage->workgroup_size_x;
  apply_dispatch_config.workgroup_size[1] = 1;
  apply_dispatch_config.workgroup_size[2] = 1;

  id4_pipeline_region_dispatch_binding_t apply_bindings[] = {
      {
          // Selected hidden states are read by the token-weight kernel.
          .tensor = selected_tensor,
          // The dispatch reads selected hidden states.
          .access = ID4_PIPELINE_TENSOR_ACCESS_READ,
      },
      {
          // Token weights are read by the token-weight kernel.
          .tensor = token_weights_tensor,
          // The dispatch reads token weights.
          .access = ID4_PIPELINE_TENSOR_ACCESS_READ,
      },
      {
          // Condition tensor is written by the token-weight kernel.
          .tensor = condition_tensor,
          // The dispatch writes the condition tensor.
          .access = ID4_PIPELINE_TENSOR_ACCESS_WRITE,
      },
      {
          // Stats tensor receives original and weighted sums.
          .tensor = stats_tensor,
          // The dispatch writes token-weight normalization stats.
          .access = ID4_PIPELINE_TENSOR_ACCESS_WRITE,
      },
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_region_dispatch(
      builder, &apply_kernel, apply_dispatch_config,
      iree_const_byte_span_empty(), IREE_ARRAYSIZE(apply_bindings),
      apply_bindings, IREE_HAL_DISPATCH_FLAG_NONE));

  iree_hal_memory_barrier_t dispatch_barrier = {
      // Prior dispatch writes condition and stats.
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      // The next dispatch reads stats and reads/writes condition.
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &dispatch_barrier,
      /*buffer_barrier_count=*/0, NULL));

  id4_pipeline_region_kernel_t normalize_kernel;
  memset(&normalize_kernel, 0, sizeof(normalize_kernel));
  normalize_kernel.name = stage->normalize_token_weights_function_name;
  normalize_kernel.executable = hal_executable;
  normalize_kernel.function = normalize_function;
  normalize_kernel.binding_count = 2;
  normalize_kernel.constant_byte_length = 0;

  uint32_t workgroups_x = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_workgroups_x(
      element_count, stage->workgroup_size_x, &workgroups_x));
  iree_hal_dispatch_config_t normalize_dispatch_config =
      iree_hal_make_static_dispatch_config(workgroups_x, 1, 1);
  normalize_dispatch_config.workgroup_size[0] = stage->workgroup_size_x;
  normalize_dispatch_config.workgroup_size[1] = 1;
  normalize_dispatch_config.workgroup_size[2] = 1;

  id4_pipeline_region_dispatch_binding_t normalize_bindings[] = {
      {
          // Condition is read and rewritten by normalization.
          .tensor = condition_tensor,
          // The dispatch reads and writes the condition tensor.
          .access = ID4_PIPELINE_TENSOR_ACCESS_READ |
                    ID4_PIPELINE_TENSOR_ACCESS_WRITE,
      },
      {
          // Stats are read by normalization.
          .tensor = stats_tensor,
          // The dispatch reads token-weight normalization stats.
          .access = ID4_PIPELINE_TENSOR_ACCESS_READ,
      },
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_region_dispatch(
      builder, &normalize_kernel, normalize_dispatch_config,
      iree_const_byte_span_empty(), IREE_ARRAYSIZE(normalize_bindings),
      normalize_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  return id4_pipeline_region_release_tensor(builder, stats_tensor);
}

static iree_status_t id4_qwen3_vl_stage_dry_run_region(
    id4_qwen3_vl_stage_t* stage,
    id4_pipeline_region_statistics_t* out_statistics) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   stage->host_allocator, &block_pool);

  id4_pipeline_region_builder_t* builder = NULL;
  id4_pipeline_region_builder_create_options_t builder_options;
  memset(&builder_options, 0, sizeof(builder_options));
  builder_options.structure_size = sizeof(builder_options);
  builder_options.region_name = IREE_SV("qwen3_vl.condition_forward");
  builder_options.mode = ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN;
  builder_options.block_pool = &block_pool;
  builder_options.binding_capacity = ID4_QWEN3_VL_STAGE_BINDING_COUNT;
  builder_options.local_binding_slot = ID4_QWEN3_VL_STAGE_LOCAL_BINDING_SLOT;

  iree_status_t status = id4_pipeline_region_builder_create(
      &builder_options, stage->host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_author_region(
        stage, builder, NULL, iree_hal_executable_function_invalid(),
        iree_hal_executable_function_invalid());
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_statistics(builder, out_statistics);
  }

  id4_pipeline_region_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_qwen3_vl_stage_format_specialization_key(
    const id4_qwen3_vl_stage_t* stage, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity,
                        "id4_qwen3_vl_condition_f32:hidden_row_count=%" PRIu32
                        ":token_count=%" PRIu32 ":workgroup_size_x=%" PRIu32,
                        stage->hidden_row_count, stage->token_count,
                        stage->workgroup_size_x);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format Qwen3-VL specialization key");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static void id4_qwen3_vl_stage_make_tensor_buffer_params(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_usage_t usage,
    iree_hal_buffer_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                     IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                     IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  out_params->access = IREE_HAL_MEMORY_ACCESS_ALL;
  out_params->usage = usage | IREE_HAL_BUFFER_USAGE_MAPPING;
  out_params->queue_affinity = queue_affinity;
  out_params->min_alignment = ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH;
}

static iree_status_t id4_qwen3_vl_stage_create_plan(
    const id4_pipeline_stage_plan_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan,
    id4_pipeline_stage_t* base_stage) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  id4_pipeline_region_statistics_t region_statistics;
  memset(&region_statistics, 0, sizeof(region_statistics));
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_stage_dry_run_region(stage, &region_statistics));

  const uint64_t element_count =
      id4_qwen3_vl_stage_condition_element_count(stage);
  iree_device_size_t condition_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_condition_byte_length(
      element_count, &condition_byte_length));
  iree_device_size_t token_weights_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_token_weights_byte_length(
      stage->token_count, &token_weights_byte_length));

  char hidden_row_count_value_buffer[16];
  char token_count_value_buffer[16];
  char workgroup_size_x_value_buffer[16];
  char specialization_key_buffer[160];
  iree_string_view_t hidden_row_count_value = iree_string_view_empty();
  iree_string_view_t token_count_value = iree_string_view_empty();
  iree_string_view_t workgroup_size_x_value = iree_string_view_empty();
  iree_string_view_t specialization_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_format_u32(
      stage->hidden_row_count, hidden_row_count_value_buffer,
      IREE_ARRAYSIZE(hidden_row_count_value_buffer), &hidden_row_count_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_format_u32(
      stage->token_count, token_count_value_buffer,
      IREE_ARRAYSIZE(token_count_value_buffer), &token_count_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_format_u32(
      stage->workgroup_size_x, workgroup_size_x_value_buffer,
      IREE_ARRAYSIZE(workgroup_size_x_value_buffer), &workgroup_size_x_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_format_specialization_key(
      stage, specialization_key_buffer,
      IREE_ARRAYSIZE(specialization_key_buffer), &specialization_key));

  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = options->device_index;
  placement.queue_affinity = options->queue_affinity;

  iree_hal_buffer_params_t selected_params;
  id4_qwen3_vl_stage_make_tensor_buffer_params(
      options->queue_affinity,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      &selected_params);
  iree_hal_buffer_params_t token_weights_params;
  id4_qwen3_vl_stage_make_tensor_buffer_params(
      options->queue_affinity,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      &token_weights_params);
  iree_hal_buffer_params_t condition_params;
  id4_qwen3_vl_stage_make_tensor_buffer_params(
      options->queue_affinity,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      &condition_params);
  iree_hal_buffer_params_t local_slab_params;
  memset(&local_slab_params, 0, sizeof(local_slab_params));
  local_slab_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_slab_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  local_slab_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  local_slab_params.queue_affinity = options->queue_affinity;
  local_slab_params.min_alignment = 16;

  id4_pipeline_memory_slab_plan_t memory_slabs[] = {
      {
          // Input tensor slab for selected hidden states.
          .name = IREE_SV("qwen3_vl.selected_hidden_states"),
          // Plan-local placement identifier.
          .placement_id = 0,
          // Issue-time binding-table slot for selected hidden states.
          .binding_slot =
              ID4_QWEN3_VL_STAGE_SELECTED_HIDDEN_STATES_BINDING_SLOT,
          // HAL buffer parameters for selected hidden states.
          .params = selected_params,
          // Selected-hidden-states tensor byte length.
          .byte_length = condition_byte_length,
          // Selected-hidden-states base alignment.
          .alignment = ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH,
          // Selected-hidden-states live byte count.
          .high_water_mark = condition_byte_length,
      },
      {
          // Input tensor slab for per-token prompt weights.
          .name = IREE_SV("qwen3_vl.token_weights"),
          // Plan-local placement identifier.
          .placement_id = 0,
          // Issue-time binding-table slot for token weights.
          .binding_slot = ID4_QWEN3_VL_STAGE_TOKEN_WEIGHTS_BINDING_SLOT,
          // HAL buffer parameters for token weights.
          .params = token_weights_params,
          // Token-weights tensor byte length.
          .byte_length = token_weights_byte_length,
          // Token-weights base alignment.
          .alignment = ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH,
          // Token-weights live byte count.
          .high_water_mark = token_weights_byte_length,
      },
      {
          // Output tensor slab for the Qwen condition tensor.
          .name = IREE_SV("qwen3_vl.condition"),
          // Plan-local placement identifier.
          .placement_id = 0,
          // Issue-time binding-table slot for the condition tensor.
          .binding_slot = ID4_QWEN3_VL_STAGE_CONDITION_BINDING_SLOT,
          // HAL buffer parameters for the condition tensor.
          .params = condition_params,
          // Condition tensor byte length.
          .byte_length = condition_byte_length,
          // Condition tensor base alignment.
          .alignment = ID4_QWEN3_VL_STAGE_F32_ELEMENT_BYTE_LENGTH,
          // Condition tensor live byte count.
          .high_water_mark = condition_byte_length,
      },
      {
          // Local transient slab for Qwen condition epilogue temporaries.
          .name = IREE_SV("qwen3_vl.local"),
          // Plan-local placement identifier.
          .placement_id = 0,
          // Issue-time binding-table slot for the local transient slab.
          .binding_slot = ID4_QWEN3_VL_STAGE_LOCAL_BINDING_SLOT,
          // HAL buffer parameters for local transients.
          .params = local_slab_params,
          // Required local slab byte length.
          .byte_length = region_statistics.local_slab_byte_length,
          // Local slab base alignment.
          .alignment = 16,
          // Peak local transient byte count.
          .high_water_mark = region_statistics.local_slab_high_water_mark,
      },
  };

  id4_pipeline_plan_config_binding_t config_bindings[] = {
      {
          // Config key for the hidden row count.
          .key = IREE_SV("id4.qwen3_vl.condition.hidden_row_count"),
          // Config value for the hidden row count.
          .value = hidden_row_count_value,
      },
      {
          // Config key for the token count.
          .key = IREE_SV("id4.qwen3_vl.condition.token_count"),
          // Config value for the token count.
          .value = token_count_value,
      },
      {
          // Config key for the X workgroup size.
          .key = IREE_SV("id4.qwen3_vl.condition.workgroup_size_x"),
          // Config value for the X workgroup size.
          .value = workgroup_size_x_value,
      },
  };
  id4_pipeline_kernel_plan_t kernels[2];
  memset(kernels, 0, sizeof(kernels));
  kernels[0].specialization_key = specialization_key;
  kernels[0].source_identifier = stage->source_identifier;
  kernels[0].module_name = stage->module_name;
  kernels[0].executable_identifier = stage->executable_identifier;
  kernels[0].function_name = stage->apply_token_weights_function_name;
  kernels[0].placement_id = 0;
  kernels[0].config_binding_count = IREE_ARRAYSIZE(config_bindings);
  kernels[0].config_bindings = config_bindings;
  kernels[1].specialization_key = specialization_key;
  kernels[1].source_identifier = stage->source_identifier;
  kernels[1].module_name = stage->module_name;
  kernels[1].executable_identifier = stage->executable_identifier;
  kernels[1].function_name = stage->normalize_token_weights_function_name;
  kernels[1].placement_id = 0;
  kernels[1].config_binding_count = IREE_ARRAYSIZE(config_bindings);
  kernels[1].config_bindings = config_bindings;

  id4_pipeline_region_plan_t region;
  memset(&region, 0, sizeof(region));
  region.name = IREE_SV("qwen3_vl.condition_forward");
  region.placement_id = 0;
  region.binding_capacity = ID4_QWEN3_VL_STAGE_BINDING_COUNT;
  region.local_binding_slot = ID4_QWEN3_VL_STAGE_LOCAL_BINDING_SLOT;
  region.statistics = region_statistics;

  id4_pipeline_diagnostic_tap_plan_t taps[] = {
      {
          // Tap for inspecting the selected hidden states entering the stage.
          .name = IREE_SV("qwen3_vl.selected_hidden_states.before_forward"),
          // Region containing the tapped value.
          .region_id = 0,
          // Operation ordinal after which the input can be inspected.
          .after_operation_ordinal = 0,
          // Tensor name exposed by the tap.
          .target_name = IREE_SV("qwen3_vl.encoder.selected_hidden_states"),
      },
      {
          // Tap for inspecting token weights entering the epilogue.
          .name = IREE_SV("qwen3_vl.token_weights.before_forward"),
          // Region containing the tapped value.
          .region_id = 0,
          // Operation ordinal before the input is consumed.
          .after_operation_ordinal = 0,
          // Tensor name exposed by the tap.
          .target_name = IREE_SV("qwen3_vl.prompt.token_weights"),
      },
      {
          // Tap for inspecting the condition tensor produced by the stage.
          .name = IREE_SV("qwen3_vl.condition.after_forward"),
          // Region containing the tapped value.
          .region_id = 0,
          // Operation ordinal after which the normalized output can be
          // inspected.
          .after_operation_ordinal = 2,
          // Tensor name exposed by the tap.
          .target_name = IREE_SV("qwen3_vl.encoder.condition"),
      },
  };

  id4_pipeline_plan_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.stage_name = IREE_SV("qwen3_vl");
  create_options.device_group = base_stage->services.device_group;
  create_options.placement_count = 1;
  create_options.placements = &placement;
  create_options.parameter_slab_count = 0;
  create_options.parameter_slabs = NULL;
  create_options.memory_slab_count = IREE_ARRAYSIZE(memory_slabs);
  create_options.memory_slabs = memory_slabs;
  create_options.kernel_count = IREE_ARRAYSIZE(kernels);
  create_options.kernels = kernels;
  create_options.region_count = 1;
  create_options.regions = &region;
  create_options.diagnostic_tap_count = IREE_ARRAYSIZE(taps);
  create_options.diagnostic_taps = taps;
  create_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_plan_create(&create_options, host_allocator, out_plan);
}

static iree_status_t id4_qwen3_vl_stage_prepare_kernel_executable(
    id4_qwen3_vl_stage_t* stage, iree_hal_queue_affinity_t queue_affinity,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_kernel_executable_t** out_executable) {
  if (!stage->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage kernel cache is required for "
                            "preparation");
  }
  if (!stage->base.services.executable_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage HAL executable cache is required "
                            "for preparation");
  }
  if (!stage->source_contents.data || stage->source_contents.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage source contents are required for "
                            "preparation");
  }

  char hidden_row_count_value_buffer[16];
  char token_count_value_buffer[16];
  char workgroup_size_x_value_buffer[16];
  iree_string_view_t hidden_row_count_value = iree_string_view_empty();
  iree_string_view_t token_count_value = iree_string_view_empty();
  iree_string_view_t workgroup_size_x_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_format_u32(
      stage->hidden_row_count, hidden_row_count_value_buffer,
      IREE_ARRAYSIZE(hidden_row_count_value_buffer), &hidden_row_count_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_format_u32(
      stage->token_count, token_count_value_buffer,
      IREE_ARRAYSIZE(token_count_value_buffer), &token_count_value));
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_format_u32(
      stage->workgroup_size_x, workgroup_size_x_value_buffer,
      IREE_ARRAYSIZE(workgroup_size_x_value_buffer), &workgroup_size_x_value));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          // Config key for the hidden row count.
          .key = IREE_SV("id4.qwen3_vl.condition.hidden_row_count"),
          // Config value for the hidden row count.
          .value = hidden_row_count_value,
      },
      {
          // Config key for the token count.
          .key = IREE_SV("id4.qwen3_vl.condition.token_count"),
          // Config value for the token count.
          .value = token_count_value,
      },
      {
          // Config key for the X workgroup size.
          .key = IREE_SV("id4.qwen3_vl.condition.workgroup_size_x"),
          // Config value for the X workgroup size.
          .value = workgroup_size_x_value,
      },
  };
  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = stage->base.services.executable_cache;
  prepare_options.queue_affinity = queue_affinity;
  prepare_options.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
  prepare_options.source_identifier = stage->source_identifier;
  prepare_options.source_contents = stage->source_contents;
  prepare_options.module_name = stage->module_name;
  prepare_options.executable_identifier = stage->executable_identifier;
  prepare_options.config_binding_count = IREE_ARRAYSIZE(config_bindings);
  prepare_options.config_bindings = config_bindings;
  prepare_options.diagnostic_artifact_flags =
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_TEXT |
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON |
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_EMIT_MANIFEST_JSON;
  prepare_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_kernel_cache_prepare_executable(
      stage->kernel_cache, &prepare_options, out_executable);
}

static iree_status_t id4_qwen3_vl_stage_allocate_selected_hidden_states_buffer(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params;
  id4_qwen3_vl_stage_make_tensor_buffer_params(
      queue_affinity,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      &params);
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t id4_qwen3_vl_stage_allocate_token_weights_buffer(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params;
  id4_qwen3_vl_stage_make_tensor_buffer_params(
      queue_affinity,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      &params);
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t id4_qwen3_vl_stage_allocate_condition_buffer(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params;
  id4_qwen3_vl_stage_make_tensor_buffer_params(
      queue_affinity,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      &params);
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t id4_qwen3_vl_stage_record_region(
    id4_qwen3_vl_stage_t* stage, iree_hal_device_group_t* device_group,
    iree_host_size_t device_index, iree_hal_queue_affinity_t queue_affinity,
    id4_pipeline_kernel_executable_t* executable,
    id4_pipeline_prepared_region_t** out_prepared_region) {
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, device_index);
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage placement device is required");
  }

  iree_hal_executable_t* hal_executable =
      id4_pipeline_kernel_executable_hal_executable(executable);
  iree_hal_executable_function_t apply_function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
      hal_executable, stage->apply_token_weights_function_name,
      &apply_function));
  iree_hal_executable_function_t normalize_function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
      hal_executable, stage->normalize_token_weights_function_name,
      &normalize_function));

  iree_hal_command_buffer_t* command_buffer = NULL;
  id4_pipeline_region_builder_t* builder = NULL;
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   stage->host_allocator, &block_pool);

  iree_status_t status = iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, queue_affinity,
      ID4_QWEN3_VL_STAGE_BINDING_COUNT, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_create_options_t builder_options;
    memset(&builder_options, 0, sizeof(builder_options));
    builder_options.structure_size = sizeof(builder_options);
    builder_options.region_name = IREE_SV("qwen3_vl.condition_forward");
    builder_options.mode = ID4_PIPELINE_REGION_BUILDER_MODE_RECORD;
    builder_options.block_pool = &block_pool;
    builder_options.command_buffer = command_buffer;
    builder_options.binding_capacity = ID4_QWEN3_VL_STAGE_BINDING_COUNT;
    builder_options.local_binding_slot = ID4_QWEN3_VL_STAGE_LOCAL_BINDING_SLOT;
    status = id4_pipeline_region_builder_create(
        &builder_options, stage->host_allocator, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_author_region(
        stage, builder, hal_executable, apply_function, normalize_function);
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

static void id4_qwen3_vl_stage_bundle_payload_destroy(
    id4_pipeline_bundle_t* bundle, void* raw_payload) {
  (void)bundle;
  id4_qwen3_vl_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_stage_bundle_payload_t*)raw_payload;
  id4_pipeline_prepared_region_release(payload->prepared_region);
  id4_pipeline_kernel_executable_release(payload->executable);
  iree_hal_buffer_release(payload->condition_buffer);
  iree_hal_buffer_release(payload->token_weights_buffer);
  iree_hal_buffer_release(payload->selected_hidden_states_buffer);
}

static void id4_qwen3_vl_stage_destroy(id4_pipeline_stage_t* base_stage) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  id4_qwen3_vl_stage_free_string(&stage->normalize_token_weights_function_name,
                                 host_allocator);
  id4_qwen3_vl_stage_free_string(&stage->apply_token_weights_function_name,
                                 host_allocator);
  id4_qwen3_vl_stage_free_string(&stage->executable_identifier, host_allocator);
  id4_qwen3_vl_stage_free_string(&stage->module_name, host_allocator);
  id4_qwen3_vl_stage_free_bytes(&stage->source_contents, host_allocator);
  id4_qwen3_vl_stage_free_string(&stage->source_identifier, host_allocator);
  id4_pipeline_stage_deinitialize(base_stage);
  iree_allocator_free(host_allocator, stage);
}

static iree_status_t id4_qwen3_vl_stage_load(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  stage->is_loaded = true;
  return id4_qwen3_vl_stage_emit_lifecycle(options->diagnostics_sink,
                                           IREE_SV("stage.load"),
                                           IREE_SV("loaded Qwen3-VL stage"));
}

static iree_status_t id4_qwen3_vl_stage_plan(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  const id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_const_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL stage must be loaded before planning");
  }
  return id4_qwen3_vl_stage_create_plan(options, stage->host_allocator,
                                        out_plan, base_stage);
}

static iree_status_t id4_qwen3_vl_stage_validate_prepare_inputs(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options) {
  if (!iree_string_view_equal(id4_pipeline_plan_stage_name(plan),
                              IREE_SV("qwen3_vl"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage prepare requires a Qwen3-VL plan");
  }
  if (id4_pipeline_plan_placement_count(plan) != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage plan must have one placement");
  }
  if (id4_pipeline_plan_parameter_slab_count(plan) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL condition plan must not have parameter "
                            "slabs");
  }
  if (id4_pipeline_plan_memory_slab_count(plan) != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL condition plan must have four memory "
                            "slabs");
  }
  if (options->parameter_provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL condition prepare does not accept a parameter provider");
  }
  if (options->wait_semaphore_list.count != 0 ||
      options->signal_semaphore_list.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL condition prepare has no HAL work and requires empty "
        "prepare semaphore lists");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_prepare(
    id4_pipeline_stage_t* base_stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL stage must be loaded before preparation");
  }
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_stage_validate_prepare_inputs(plan, options));

  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(plan, 0);
  iree_hal_device_group_t* device_group = id4_pipeline_plan_device_group(plan);
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, placement->device_index);
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage placement device is required");
  }

  const uint64_t element_count =
      id4_qwen3_vl_stage_condition_element_count(stage);
  iree_device_size_t condition_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_condition_byte_length(
      element_count, &condition_byte_length));
  iree_device_size_t token_weights_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_token_weights_byte_length(
      stage->token_count, &token_weights_byte_length));

  id4_pipeline_kernel_executable_t* executable = NULL;
  id4_pipeline_prepared_region_t* prepared_region = NULL;
  iree_hal_buffer_t* selected_hidden_states_buffer = NULL;
  iree_hal_buffer_t* token_weights_buffer = NULL;
  iree_hal_buffer_t* condition_buffer = NULL;
  id4_pipeline_bundle_t* bundle = NULL;

  iree_status_t status = id4_qwen3_vl_stage_prepare_kernel_executable(
      stage, placement->queue_affinity, options->diagnostics_sink, &executable);
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_allocate_selected_hidden_states_buffer(
        device, placement->queue_affinity, condition_byte_length,
        &selected_hidden_states_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_allocate_token_weights_buffer(
        device, placement->queue_affinity, token_weights_byte_length,
        &token_weights_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_allocate_condition_buffer(
        device, placement->queue_affinity, condition_byte_length,
        &condition_buffer);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_map_zero(condition_buffer, 0, condition_byte_length);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_record_region(
        stage, device_group, placement->device_index, placement->queue_affinity,
        executable, &prepared_region);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_bundle_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.plan = plan;
    create_options.parameter_slabs = NULL;
    create_options.readiness_semaphore_list = iree_hal_semaphore_list_empty();
    create_options.payload_size = sizeof(id4_qwen3_vl_stage_bundle_payload_t);
    create_options.payload_alignment =
        iree_alignof(id4_qwen3_vl_stage_bundle_payload_t);
    create_options.payload_destroy = id4_qwen3_vl_stage_bundle_payload_destroy;
    status = id4_pipeline_bundle_create(&create_options, stage->host_allocator,
                                        &bundle);
  }
  if (iree_status_is_ok(status)) {
    id4_qwen3_vl_stage_bundle_payload_t* payload =
        (id4_qwen3_vl_stage_bundle_payload_t*)id4_pipeline_bundle_payload(
            bundle);
    payload->executable = executable;
    executable = NULL;
    payload->prepared_region = prepared_region;
    prepared_region = NULL;
    payload->selected_hidden_states_buffer = selected_hidden_states_buffer;
    selected_hidden_states_buffer = NULL;
    payload->token_weights_buffer = token_weights_buffer;
    token_weights_buffer = NULL;
    payload->condition_buffer = condition_buffer;
    condition_buffer = NULL;
    payload->token_weights_byte_length = token_weights_byte_length;
    payload->condition_byte_length = condition_byte_length;
    payload->bindings[ID4_QWEN3_VL_STAGE_SELECTED_HIDDEN_STATES_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Selected-hidden-states input buffer read by the kernel.
            .buffer = payload->selected_hidden_states_buffer,
            // Selected-hidden-states tensor starts at byte zero.
            .offset = 0,
            // Full selected-hidden-states tensor byte length.
            .length = condition_byte_length,
        };
    payload->bindings[ID4_QWEN3_VL_STAGE_CONDITION_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Condition output buffer written by the kernel.
            .buffer = payload->condition_buffer,
            // Condition tensor starts at byte zero.
            .offset = 0,
            // Full condition tensor byte length.
            .length = condition_byte_length,
        };
    payload->bindings[ID4_QWEN3_VL_STAGE_TOKEN_WEIGHTS_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Token-weights input buffer read by the kernel.
            .buffer = payload->token_weights_buffer,
            // Token-weights tensor starts at byte zero.
            .offset = 0,
            // Full token-weights tensor byte length.
            .length = token_weights_byte_length,
        };
    payload->bindings[ID4_QWEN3_VL_STAGE_LOCAL_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Local slab slot is filled by prepared-region issue.
            .buffer = NULL,
            // No caller-provided local slab offset.
            .offset = 0,
            // No caller-provided local slab length.
            .length = 0,
        };
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_emit_lifecycle(
        options->diagnostics_sink, IREE_SV("stage.prepare"),
        IREE_SV("prepared Qwen3-VL bundle"));
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    id4_pipeline_bundle_release(bundle);
    id4_pipeline_prepared_region_release(prepared_region);
    id4_pipeline_kernel_executable_release(executable);
    iree_hal_buffer_release(condition_buffer);
    iree_hal_buffer_release(token_weights_buffer);
    iree_hal_buffer_release(selected_hidden_states_buffer);
  }
  return status;
}

static iree_status_t id4_qwen3_vl_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  (void)base_stage;
  id4_qwen3_vl_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_stage_bundle_payload_t*)id4_pipeline_bundle_payload(bundle);
  if (!payload || !payload->prepared_region) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL bundle payload is not prepared");
  }

  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.issue"),
      IREE_SV("issued Qwen3-VL bundle")));

  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(readiness_list.count,
                                  options->wait_semaphore_list.count,
                                  &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL stage wait list count overflow");
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
      .count = ID4_QWEN3_VL_STAGE_BINDING_COUNT,
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

static const id4_pipeline_stage_vtable_t id4_qwen3_vl_stage_vtable = {
    // Destroys the concrete Qwen3-VL stage.
    id4_qwen3_vl_stage_destroy,
    // Loads immutable Qwen3-VL stage state.
    id4_qwen3_vl_stage_load,
    // Builds an inspectable Qwen3-VL execution plan.
    id4_qwen3_vl_stage_plan,
    // Prepares a reusable Qwen3-VL execution bundle.
    id4_qwen3_vl_stage_prepare,
    // Issues one Qwen3-VL bundle execution.
    id4_qwen3_vl_stage_issue,
};

iree_status_t id4_qwen3_vl_stage_create(
    const id4_qwen3_vl_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = NULL;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_validate_create_options(options));

  id4_qwen3_vl_stage_t* stage = NULL;
  bool base_initialized = false;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*stage), (void**)&stage);
  if (iree_status_is_ok(status)) {
    memset(stage, 0, sizeof(*stage));
    stage->host_allocator = host_allocator;
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_stage_initialize(&id4_qwen3_vl_stage_vtable,
                                           &options->services, &stage->base);
    base_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    stage->kernel_cache = options->kernel_cache;
    id4_pipeline_kernel_cache_retain(stage->kernel_cache);
    stage->hidden_row_count = options->hidden_row_count;
    stage->token_count = options->token_count;
    stage->workgroup_size_x = options->workgroup_size_x;
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_copy_string(
        options->source_identifier, host_allocator, &stage->source_identifier);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_copy_bytes(
        options->source_contents, host_allocator, &stage->source_contents);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_copy_string(
        options->module_name, host_allocator, &stage->module_name);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_copy_string(options->executable_identifier,
                                            host_allocator,
                                            &stage->executable_identifier);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_copy_string(
        options->apply_token_weights_function_name, host_allocator,
        &stage->apply_token_weights_function_name);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_copy_string(
        options->normalize_token_weights_function_name, host_allocator,
        &stage->normalize_token_weights_function_name);
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

iree_hal_buffer_t* id4_qwen3_vl_stage_bundle_selected_hidden_states_buffer(
    id4_pipeline_bundle_t* bundle) {
  id4_qwen3_vl_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_stage_bundle_payload_t*)id4_pipeline_bundle_payload(bundle);
  return payload ? payload->selected_hidden_states_buffer : NULL;
}

iree_hal_buffer_t* id4_qwen3_vl_stage_bundle_token_weights_buffer(
    id4_pipeline_bundle_t* bundle) {
  id4_qwen3_vl_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_stage_bundle_payload_t*)id4_pipeline_bundle_payload(bundle);
  return payload ? payload->token_weights_buffer : NULL;
}

iree_hal_buffer_t* id4_qwen3_vl_stage_bundle_condition_buffer(
    id4_pipeline_bundle_t* bundle) {
  id4_qwen3_vl_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_stage_bundle_payload_t*)id4_pipeline_bundle_payload(bundle);
  return payload ? payload->condition_buffer : NULL;
}

iree_device_size_t id4_qwen3_vl_stage_bundle_token_weights_byte_length(
    const id4_pipeline_bundle_t* bundle) {
  const id4_qwen3_vl_stage_bundle_payload_t* payload =
      (const id4_qwen3_vl_stage_bundle_payload_t*)
          id4_pipeline_bundle_const_payload(bundle);
  return payload ? payload->token_weights_byte_length : 0;
}

iree_device_size_t id4_qwen3_vl_stage_bundle_condition_byte_length(
    const id4_pipeline_bundle_t* bundle) {
  const id4_qwen3_vl_stage_bundle_payload_t* payload =
      (const id4_qwen3_vl_stage_bundle_payload_t*)
          id4_pipeline_bundle_const_payload(bundle);
  return payload ? payload->condition_byte_length : 0;
}
