// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/smoke.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/region.h"
#include "iree/base/internal/arena.h"

#define ID4_SMOKE_STAGE_BINDING_COUNT 4
#define ID4_SMOKE_STAGE_OUTPUT_BINDING_SLOT 0
#define ID4_SMOKE_STAGE_LOCAL_BINDING_SLOT 1
#define ID4_SMOKE_STAGE_PARAMETER_BINDING_SLOT 2
#define ID4_SMOKE_STAGE_TAP_BINDING_SLOT 3
#define ID4_SMOKE_STAGE_OUTPUT_ELEMENT_COUNT 1

typedef uint32_t id4_smoke_stage_region_flags_t;
enum id4_smoke_stage_region_flag_bits_e {
  // Records the post-dispatch output tensor copy into the diagnostic tap slot.
  ID4_SMOKE_STAGE_REGION_FLAG_CAPTURE_DIAGNOSTIC_TAP = 1u << 0,
};

typedef struct id4_smoke_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Loom module path owned by the stage.
  iree_string_view_t module_path;
  // Exported HAL function name owned by the stage.
  iree_string_view_t function_name;
  // True after load has completed.
  bool is_loaded;
} id4_smoke_stage_t;

typedef struct id4_smoke_stage_bundle_payload_t {
  // Prepared kernel executable retained for command-buffer validity.
  id4_pipeline_kernel_executable_t* executable;
  // Prepared reusable region issued by the smoke stage.
  id4_pipeline_prepared_region_t* prepared_region;
  // Output buffer retained for issue-time binding and readback.
  iree_hal_buffer_t* output_buffer;
  // Fixed issue-time binding table storage.
  iree_hal_buffer_binding_t bindings[ID4_SMOKE_STAGE_BINDING_COUNT];
} id4_smoke_stage_bundle_payload_t;

static id4_smoke_stage_t* id4_smoke_stage_cast(
    id4_pipeline_stage_t* base_stage) {
  return (id4_smoke_stage_t*)base_stage;
}

static const id4_smoke_stage_t* id4_smoke_stage_const_cast(
    const id4_pipeline_stage_t* base_stage) {
  return (const id4_smoke_stage_t*)base_stage;
}

static iree_status_t id4_smoke_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_smoke_stage_copy_string(
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

static void id4_smoke_stage_free_string(iree_string_view_t* value,
                                        iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static iree_status_t id4_smoke_stage_validate_create_options(
    const id4_smoke_stage_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_smoke_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("smoke stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "smoke stage create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage device group is required");
  }
  if (iree_string_view_is_empty(options->module_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage module path is required");
  }
  if (iree_string_view_is_empty(options->function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage function name is required");
  }
  return iree_ok_status();
}

static iree_status_t id4_smoke_stage_format_u32(
    uint32_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  int length = snprintf(buffer, buffer_capacity, "%" PRIu32, value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format smoke config value");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t id4_smoke_stage_lifecycle_event(
    iree_string_view_t key, iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the concrete smoke stage.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name used across smoke diagnostics.
      .stage_name = IREE_SV("smoke"),
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_smoke_stage_emit_lifecycle(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_smoke_stage_lifecycle_event(key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static id4_pipeline_tensor_shape_t id4_smoke_stage_make_vector_shape(
    uint64_t element_count) {
  id4_pipeline_tensor_shape_t shape;
  memset(&shape, 0, sizeof(shape));
  shape.rank = 1;
  shape.dims[0] = element_count;
  return shape;
}

static iree_status_t id4_smoke_stage_format_specialization_key(
    char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  int length =
      snprintf(buffer, buffer_capacity, "id4_smoke_configured:element_count=%u",
               ID4_SMOKE_STAGE_OUTPUT_ELEMENT_COUNT);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format smoke specialization key");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_smoke_stage_author_region(
    id4_smoke_stage_t* stage, id4_pipeline_region_builder_t* builder,
    id4_smoke_stage_region_flags_t region_flags,
    iree_hal_executable_t* hal_executable,
    iree_hal_executable_function_t function,
    iree_hal_dispatch_config_t dispatch_config) {
  id4_pipeline_tensor_import_t output_import;
  memset(&output_import, 0, sizeof(output_import));
  output_import.layout.name = IREE_SV("smoke.output");
  output_import.layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_I32;
  output_import.layout.shape = id4_smoke_stage_make_vector_shape(1);
  output_import.layout.byte_length = ID4_SMOKE_STAGE_OUTPUT_BYTE_LENGTH;
  output_import.layout.alignment = 4;
  output_import.binding_slot = ID4_SMOKE_STAGE_OUTPUT_BINDING_SLOT;
  output_import.offset = 0;

  id4_pipeline_tensor_t output_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      builder, &output_import, &output_tensor));

  id4_pipeline_tensor_layout_t scratch_layout;
  memset(&scratch_layout, 0, sizeof(scratch_layout));
  scratch_layout.name = IREE_SV("smoke.scratch");
  scratch_layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_I32;
  scratch_layout.shape = id4_smoke_stage_make_vector_shape(4);
  scratch_layout.byte_length = 16;
  scratch_layout.alignment = 16;
  id4_pipeline_tensor_t scratch_tensor;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_acquire_tensor(
      builder, &scratch_layout, &scratch_tensor));

  char element_count_value_buffer[16];
  char specialization_key_buffer[128];
  iree_string_view_t element_count_value = iree_string_view_empty();
  iree_string_view_t specialization_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_smoke_stage_format_u32(
      ID4_SMOKE_STAGE_OUTPUT_ELEMENT_COUNT, element_count_value_buffer,
      IREE_ARRAYSIZE(element_count_value_buffer), &element_count_value));
  IREE_RETURN_IF_ERROR(id4_smoke_stage_format_specialization_key(
      specialization_key_buffer, IREE_ARRAYSIZE(specialization_key_buffer),
      &specialization_key));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          // Config key for the logical output element count.
          .key = IREE_SV("id4.smoke.element_count"),
          // Config value for the logical output element count.
          .value = element_count_value,
      },
  };
  id4_pipeline_region_loom_kernel_t kernel;
  memset(&kernel, 0, sizeof(kernel));
  kernel.specialization_key = specialization_key;
  kernel.module_path = stage->module_path;
  kernel.function_name = stage->function_name;
  kernel.executable = hal_executable;
  kernel.function = function;
  kernel.binding_count = 1;
  kernel.constant_byte_length = 0;
  kernel.config_binding_count = IREE_ARRAYSIZE(config_bindings);
  kernel.config_bindings = config_bindings;

  id4_pipeline_region_dispatch_binding_t binding;
  memset(&binding, 0, sizeof(binding));
  binding.tensor = output_tensor;
  binding.access = ID4_PIPELINE_TENSOR_ACCESS_WRITE;
  IREE_RETURN_IF_ERROR(id4_pipeline_region_dispatch_loom(
      builder, &kernel, dispatch_config, iree_const_byte_span_empty(),
      /*binding_count=*/1, &binding, IREE_HAL_DISPATCH_FLAG_NONE));

  if (!iree_all_bits_set(region_flags,
                         ID4_SMOKE_STAGE_REGION_FLAG_CAPTURE_DIAGNOSTIC_TAP)) {
    return iree_ok_status();
  }

  id4_pipeline_tensor_import_t tap_import;
  memset(&tap_import, 0, sizeof(tap_import));
  tap_import.layout.name = IREE_SV("smoke.output.after_dispatch");
  tap_import.layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_I32;
  tap_import.layout.shape = id4_smoke_stage_make_vector_shape(1);
  tap_import.layout.byte_length = ID4_SMOKE_STAGE_OUTPUT_BYTE_LENGTH;
  tap_import.layout.alignment = 4;
  tap_import.binding_slot = ID4_SMOKE_STAGE_TAP_BINDING_SLOT;
  tap_import.offset = 0;

  id4_pipeline_tensor_t tap_tensor;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_import_tensor(builder, &tap_import, &tap_tensor));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/NULL,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_copy_tensor(
      builder, output_tensor, tap_tensor, IREE_HAL_COPY_FLAG_NONE));
  return id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/NULL,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL);
}

static iree_status_t id4_smoke_stage_create_dry_run_builder(
    id4_smoke_stage_t* stage, iree_arena_block_pool_t* block_pool,
    id4_smoke_stage_region_flags_t region_flags,
    id4_pipeline_region_builder_t** out_builder) {
  id4_pipeline_region_builder_create_options_t builder_options;
  memset(&builder_options, 0, sizeof(builder_options));
  builder_options.structure_size = sizeof(builder_options);
  builder_options.region_name = IREE_SV("smoke.region");
  builder_options.mode = ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN;
  builder_options.block_pool = block_pool;
  builder_options.binding_capacity = ID4_SMOKE_STAGE_BINDING_COUNT;
  builder_options.local_binding_slot = ID4_SMOKE_STAGE_LOCAL_BINDING_SLOT;

  id4_pipeline_region_builder_t* builder = NULL;
  iree_status_t status = id4_pipeline_region_builder_create(
      &builder_options, stage->host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    status =
        id4_smoke_stage_author_region(stage, builder, region_flags, NULL,
                                      iree_hal_executable_function_invalid(),
                                      (iree_hal_dispatch_config_t){0});
  }
  if (iree_status_is_ok(status)) {
    *out_builder = builder;
  } else {
    id4_pipeline_region_builder_destroy(builder);
  }
  return status;
}

static iree_status_t id4_smoke_stage_create_plan(
    const id4_pipeline_stage_plan_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan,
    id4_pipeline_stage_t* base_stage) {
  id4_smoke_stage_t* stage = id4_smoke_stage_cast(base_stage);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   stage->host_allocator, &block_pool);
  const iree_string_view_t smoke_tap_name =
      IREE_SV("smoke.output.after_dispatch");
  id4_smoke_stage_region_flags_t region_flags = 0;
  if (iree_all_bits_set(options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS)) {
    if (options->diagnostic_tap_names.count != 1 ||
        !iree_string_view_equal(options->diagnostic_tap_names.values[0],
                                smoke_tap_name)) {
      iree_arena_block_pool_deinitialize(&block_pool);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "smoke stage diagnostic tap capture requires tap `%.*s`",
          (int)smoke_tap_name.size, smoke_tap_name.data);
    }
    region_flags |= ID4_SMOKE_STAGE_REGION_FLAG_CAPTURE_DIAGNOSTIC_TAP;
  }
  id4_pipeline_region_builder_t* builder = NULL;
  iree_status_t status = id4_smoke_stage_create_dry_run_builder(
      stage, &block_pool, region_flags, &builder);

  id4_pipeline_region_statistics_t region_statistics;
  memset(&region_statistics, 0, sizeof(region_statistics));
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_statistics(builder, &region_statistics);
  }
  iree_host_size_t local_lifetime_count = 0;
  id4_pipeline_region_local_lifetime_t* local_lifetimes = NULL;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_region_builder_clone_local_lifetimes(
        builder, host_allocator, &local_lifetime_count, &local_lifetimes);
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
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "smoke kernel plan %" PRIhsz " is missing", i);
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

  id4_pipeline_parameter_request_t request = id4_pipeline_parameter_request(
      IREE_SV("smoke.weight"),
      id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                  /*buffer_offset=*/0, /*length=*/16));

  id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_device_local_parameter_slab_plan(
          IREE_SV("smoke"), /*placement_id=*/0,
          ID4_SMOKE_STAGE_PARAMETER_BINDING_SLOT, options->queue_affinity,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
          /*byte_length=*/16, /*alignment=*/16, /*request_count=*/1, &request);
  id4_pipeline_parameter_load_step_t load_step =
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("parameters.gather"), IREE_SV("smoke"),
          /*target_slab_index=*/0, /*request_offset=*/0,
          /*request_count=*/1);

  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = options->device_index;
  placement.queue_affinity = options->queue_affinity;

  iree_hal_buffer_params_t local_slab_params;
  memset(&local_slab_params, 0, sizeof(local_slab_params));
  local_slab_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_slab_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  local_slab_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  local_slab_params.queue_affinity = options->queue_affinity;
  local_slab_params.min_alignment = 16;

  id4_pipeline_memory_slab_plan_t memory_slab;
  memset(&memory_slab, 0, sizeof(memory_slab));
  memory_slab.name = IREE_SV("smoke.local");
  memory_slab.placement_id = 0;
  memory_slab.binding_slot = ID4_SMOKE_STAGE_LOCAL_BINDING_SLOT;
  memory_slab.params = local_slab_params;
  memory_slab.byte_length = region_statistics.local_slab_byte_length;
  memory_slab.alignment = 16;
  memory_slab.high_water_mark = region_statistics.local_slab_high_water_mark;

  id4_pipeline_region_plan_t region;
  memset(&region, 0, sizeof(region));
  region.name = IREE_SV("smoke.region");
  region.placement_id = 0;
  region.binding_capacity = ID4_SMOKE_STAGE_BINDING_COUNT;
  region.local_binding_slot = ID4_SMOKE_STAGE_LOCAL_BINDING_SLOT;
  region.statistics = region_statistics;
  region.local_lifetime_count = local_lifetime_count;
  region.local_lifetimes = local_lifetimes;

  id4_pipeline_diagnostic_tap_plan_t tap;
  memset(&tap, 0, sizeof(tap));
  tap.name = smoke_tap_name;
  tap.region_id = 0;
  tap.placement_id = 0;
  tap.binding_slot = ID4_SMOKE_STAGE_TAP_BINDING_SLOT;
  tap.after_operation_ordinal = 0;
  tap.target_name = IREE_SV("smoke.output");
  tap.layout.name = smoke_tap_name;
  tap.layout.dtype = ID4_PIPELINE_TENSOR_DTYPE_I32;
  tap.layout.shape = id4_smoke_stage_make_vector_shape(1);
  tap.layout.byte_length = ID4_SMOKE_STAGE_OUTPUT_BYTE_LENGTH;
  tap.layout.alignment = 4;

  if (iree_status_is_ok(status)) {
    id4_pipeline_plan_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.stage_name = IREE_SV("smoke");
    create_options.device_group = base_stage->services.device_group;
    create_options.placement_count = 1;
    create_options.placements = &placement;
    create_options.parameter_slab_count = 1;
    create_options.parameter_slabs = &slab;
    create_options.parameter_load_step_count = 1;
    create_options.parameter_load_steps = &load_step;
    create_options.memory_slab_count = 1;
    create_options.memory_slabs = &memory_slab;
    create_options.kernel_count = kernel_count;
    create_options.kernels = kernels;
    create_options.region_count = 1;
    create_options.regions = &region;
    if (iree_all_bits_set(region_flags,
                          ID4_SMOKE_STAGE_REGION_FLAG_CAPTURE_DIAGNOSTIC_TAP)) {
      create_options.diagnostic_tap_count = 1;
      create_options.diagnostic_taps = &tap;
    }
    create_options.diagnostics_sink = options->diagnostics_sink;
    status =
        id4_pipeline_plan_create(&create_options, host_allocator, out_plan);
  }
  id4_pipeline_region_local_lifetime_list_release(
      local_lifetime_count, local_lifetimes, host_allocator);
  id4_pipeline_region_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_smoke_stage_prepare_kernel_executable(
    id4_smoke_stage_t* stage, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_kernel_library_t* kernel_library,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_kernel_executable_t** out_executable) {
  const id4_pipeline_kernel_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_lookup(
      kernel_library, stage->module_path, &module));

  char element_count_value_buffer[16];
  iree_string_view_t element_count_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_smoke_stage_format_u32(
      ID4_SMOKE_STAGE_OUTPUT_ELEMENT_COUNT, element_count_value_buffer,
      IREE_ARRAYSIZE(element_count_value_buffer), &element_count_value));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          // Config key for the logical output element count.
          .key = IREE_SV("id4.smoke.element_count"),
          // Config value for the logical output element count.
          .value = element_count_value,
      },
  };
  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = stage->base.services.executable_cache;
  prepare_options.queue_affinity = queue_affinity;
  prepare_options.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
  prepare_options.source_identifier = module->source_identifier;
  prepare_options.source_contents = module->source_contents;
  prepare_options.module_path = module->module_path;
  prepare_options.function_name = stage->function_name;
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

static iree_status_t id4_smoke_stage_allocate_output_buffer(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_MAPPING;
  params.queue_affinity = queue_affinity;
  params.min_alignment = 4;
  return iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params,
      ID4_SMOKE_STAGE_OUTPUT_BYTE_LENGTH, out_buffer);
}

static iree_status_t id4_smoke_stage_join_prepare_readiness(
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (signal_semaphore_list.count == 0) return iree_ok_status();
  return iree_hal_semaphore_list_wait(signal_semaphore_list,
                                      iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_smoke_stage_record_region(
    id4_smoke_stage_t* stage, iree_hal_device_group_t* device_group,
    iree_host_size_t device_index, iree_hal_queue_affinity_t queue_affinity,
    id4_smoke_stage_region_flags_t region_flags,
    iree_hal_command_buffer_mode_t command_buffer_mode,
    id4_pipeline_kernel_executable_t* executable,
    id4_pipeline_prepared_region_t** out_prepared_region) {
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, device_index);
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage placement device is required");
  }

  iree_hal_executable_t* hal_executable =
      id4_pipeline_kernel_executable_hal_executable(executable);
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
      hal_executable, stage->function_name, &function));

  iree_hal_command_buffer_t* command_buffer = NULL;
  id4_pipeline_region_builder_t* builder = NULL;
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   stage->host_allocator, &block_pool);

  iree_hal_command_category_t command_categories =
      IREE_HAL_COMMAND_CATEGORY_DISPATCH;
  if (iree_all_bits_set(region_flags,
                        ID4_SMOKE_STAGE_REGION_FLAG_CAPTURE_DIAGNOSTIC_TAP)) {
    command_categories = IREE_HAL_COMMAND_CATEGORY_ANY;
  }
  iree_status_t status = iree_hal_command_buffer_create(
      device, command_buffer_mode, command_categories, queue_affinity,
      ID4_SMOKE_STAGE_BINDING_COUNT, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_create_options_t builder_options;
    memset(&builder_options, 0, sizeof(builder_options));
    builder_options.structure_size = sizeof(builder_options);
    builder_options.region_name = IREE_SV("smoke.region");
    builder_options.mode = ID4_PIPELINE_REGION_BUILDER_MODE_RECORD;
    builder_options.block_pool = &block_pool;
    builder_options.command_buffer = command_buffer;
    builder_options.binding_capacity = ID4_SMOKE_STAGE_BINDING_COUNT;
    builder_options.local_binding_slot = ID4_SMOKE_STAGE_LOCAL_BINDING_SLOT;
    status = id4_pipeline_region_builder_create(
        &builder_options, stage->host_allocator, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = id4_smoke_stage_author_region(
        stage, builder, region_flags, hal_executable, function,
        id4_pipeline_kernel_executable_dispatch_config(executable));
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

static void id4_smoke_stage_bundle_payload_destroy(
    id4_pipeline_bundle_t* bundle, void* raw_payload) {
  (void)bundle;
  id4_smoke_stage_bundle_payload_t* payload =
      (id4_smoke_stage_bundle_payload_t*)raw_payload;
  id4_pipeline_prepared_region_release(payload->prepared_region);
  id4_pipeline_kernel_executable_release(payload->executable);
  iree_hal_buffer_release(payload->output_buffer);
}

static void id4_smoke_stage_destroy(id4_pipeline_stage_t* base_stage) {
  id4_smoke_stage_t* stage = id4_smoke_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  id4_smoke_stage_free_string(&stage->function_name, host_allocator);
  id4_smoke_stage_free_string(&stage->module_path, host_allocator);
  id4_pipeline_stage_deinitialize(base_stage);
  iree_allocator_free(host_allocator, stage);
}

static iree_status_t id4_smoke_stage_load(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_smoke_stage_t* stage = id4_smoke_stage_cast(base_stage);
  stage->is_loaded = true;
  return id4_smoke_stage_emit_lifecycle(options->diagnostics_sink,
                                        IREE_SV("stage.load"),
                                        IREE_SV("loaded smoke stage"));
}

static iree_status_t id4_smoke_stage_plan(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  const id4_smoke_stage_t* stage = id4_smoke_stage_const_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "smoke stage must be loaded before planning");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "smoke stage plan extension structures are not supported");
  }
  return id4_smoke_stage_create_plan(options, stage->host_allocator, out_plan,
                                     base_stage);
}

static iree_status_t id4_smoke_stage_prepare(
    id4_pipeline_stage_t* base_stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  id4_smoke_stage_t* stage = id4_smoke_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "smoke stage must be loaded before preparation");
  }
  if (!iree_string_view_equal(id4_pipeline_plan_stage_name(plan),
                              IREE_SV("smoke"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage prepare requires a smoke plan");
  }
  if (id4_pipeline_plan_placement_count(plan) != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage plan must have one placement");
  }
  if (id4_pipeline_plan_parameter_slab_count(plan) != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage plan must have one parameter slab");
  }
  if (!options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage kernel library is required");
  }
  if (!options->parameter_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage parameter provider is required");
  }
  if (!stage->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage kernel cache is required for "
                            "preparation");
  }
  if (!stage->base.services.executable_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage HAL executable cache is required for "
                            "preparation");
  }
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(plan, 0);
  iree_hal_device_group_t* device_group = id4_pipeline_plan_device_group(plan);
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, placement->device_index);
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke stage placement device is required");
  }

  id4_pipeline_parameter_slab_set_t* parameter_slabs = NULL;
  id4_pipeline_kernel_executable_t* executable = NULL;
  id4_pipeline_prepared_region_t* prepared_region = NULL;
  iree_hal_buffer_t* output_buffer = NULL;
  id4_pipeline_bundle_t* bundle = NULL;
  bool parameter_load_submitted = false;
  id4_smoke_stage_region_flags_t region_flags = 0;
  if (id4_pipeline_plan_diagnostic_tap_count(plan) != 0) {
    region_flags |= ID4_SMOKE_STAGE_REGION_FLAG_CAPTURE_DIAGNOSTIC_TAP;
  }

  id4_pipeline_parameter_slab_set_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.provider = options->parameter_provider;
  load_options.kernel_library = options->kernel_library;
  load_options.kernel_cache = stage->kernel_cache;
  load_options.executable_cache = stage->base.services.executable_cache;
  load_options.diagnostic_artifact_flags =
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON |
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_EMIT_MANIFEST_JSON;
  load_options.wait_semaphore_list = options->wait_semaphore_list;
  load_options.signal_semaphore_list = options->signal_semaphore_list;
  load_options.diagnostics_sink = options->diagnostics_sink;
  iree_status_t status = id4_pipeline_plan_load_parameter_slabs(
      plan, &load_options, stage->host_allocator, &parameter_slabs);
  parameter_load_submitted = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = id4_smoke_stage_prepare_kernel_executable(
        stage, placement->queue_affinity, options->kernel_library,
        options->diagnostics_sink, &executable);
  }
  if (iree_status_is_ok(status)) {
    status = id4_smoke_stage_allocate_output_buffer(
        device, placement->queue_affinity, &output_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_zero(output_buffer, 0,
                                      ID4_SMOKE_STAGE_OUTPUT_BYTE_LENGTH);
  }
  if (iree_status_is_ok(status)) {
    status = id4_smoke_stage_record_region(
        stage, device_group, placement->device_index, placement->queue_affinity,
        region_flags, options->command_buffer_mode, executable,
        &prepared_region);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_bundle_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.plan = plan;
    create_options.parameter_slabs = parameter_slabs;
    create_options.readiness_semaphore_list = options->signal_semaphore_list;
    create_options.payload_size = sizeof(id4_smoke_stage_bundle_payload_t);
    create_options.payload_alignment =
        iree_alignof(id4_smoke_stage_bundle_payload_t);
    create_options.payload_destroy = id4_smoke_stage_bundle_payload_destroy;
    status = id4_pipeline_bundle_create(&create_options, stage->host_allocator,
                                        &bundle);
  }
  if (iree_status_is_ok(status)) {
    id4_smoke_stage_bundle_payload_t* payload =
        (id4_smoke_stage_bundle_payload_t*)id4_pipeline_bundle_payload(bundle);
    payload->executable = executable;
    executable = NULL;
    payload->prepared_region = prepared_region;
    prepared_region = NULL;
    payload->output_buffer = output_buffer;
    output_buffer = NULL;
    payload->bindings[ID4_SMOKE_STAGE_OUTPUT_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Output buffer written by the smoke kernel.
            .buffer = payload->output_buffer,
            // Output tensor starts at byte zero.
            .offset = 0,
            // Output tensor byte length.
            .length = ID4_SMOKE_STAGE_OUTPUT_BYTE_LENGTH,
        };
    payload->bindings[ID4_SMOKE_STAGE_LOCAL_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Local slab slot patched by prepared-region issue.
            .buffer = NULL,
            // No caller-provided local slab offset.
            .offset = 0,
            // No caller-provided local slab length.
            .length = 0,
        };
    payload->bindings[ID4_SMOKE_STAGE_PARAMETER_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Loaded parameter slab retained by the bundle.
            .buffer =
                id4_pipeline_parameter_slab_set_buffer_at(parameter_slabs, 0),
            // Parameter slab starts at byte zero.
            .offset = 0,
            // Full parameter slab byte length.
            .length = 16,
        };
    payload->bindings[ID4_SMOKE_STAGE_TAP_BINDING_SLOT] =
        (iree_hal_buffer_binding_t){
            // Diagnostic tap slot patched from issue options when planned.
            .buffer = NULL,
            // No caller-provided tap offset at prepare time.
            .offset = 0,
            // No caller-provided tap length at prepare time.
            .length = 0,
        };
  }
  if (iree_status_is_ok(status)) {
    status = id4_smoke_stage_emit_lifecycle(options->diagnostics_sink,
                                            IREE_SV("stage.prepare"),
                                            IREE_SV("prepared smoke bundle"));
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    if (parameter_load_submitted) {
      status = iree_status_join(status, id4_smoke_stage_join_prepare_readiness(
                                            options->signal_semaphore_list));
    }
    id4_pipeline_bundle_release(bundle);
    id4_pipeline_prepared_region_release(prepared_region);
    id4_pipeline_kernel_executable_release(executable);
    iree_hal_buffer_release(output_buffer);
  }
  id4_pipeline_parameter_slab_set_release(parameter_slabs);
  return status;
}

static iree_status_t id4_smoke_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  (void)base_stage;
  id4_smoke_stage_bundle_payload_t* payload =
      (id4_smoke_stage_bundle_payload_t*)id4_pipeline_bundle_payload(bundle);
  if (!payload || !payload->prepared_region) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "smoke bundle payload is not prepared");
  }

  IREE_RETURN_IF_ERROR(id4_smoke_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.issue"),
      IREE_SV("issued smoke bundle")));

  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(readiness_list.count,
                                  options->wait_semaphore_list.count,
                                  &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "smoke stage wait list count overflow");
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

  iree_hal_buffer_binding_t bindings[ID4_SMOKE_STAGE_BINDING_COUNT];
  memcpy(bindings, payload->bindings, sizeof(bindings));
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(id4_pipeline_bundle_plan(bundle));
  if (diagnostic_tap_count != 0) {
    bindings[ID4_SMOKE_STAGE_TAP_BINDING_SLOT] =
        options->diagnostic_tap_bindings[0];
  }

  iree_hal_buffer_binding_table_t binding_table = {
      // Number of issue-time binding table slots.
      .count = ID4_SMOKE_STAGE_BINDING_COUNT,
      // Per-issue binding table containing caller-owned tap bindings.
      .bindings = bindings,
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

static const id4_pipeline_stage_vtable_t id4_smoke_stage_vtable = {
    // Destroys the concrete smoke stage.
    id4_smoke_stage_destroy,
    // Loads immutable smoke stage state.
    id4_smoke_stage_load,
    // Builds an inspectable smoke execution plan.
    id4_smoke_stage_plan,
    // Prepares a reusable smoke execution bundle.
    id4_smoke_stage_prepare,
    // Issues one smoke bundle execution.
    id4_smoke_stage_issue,
};

iree_status_t id4_smoke_stage_create(
    const id4_smoke_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = NULL;
  IREE_RETURN_IF_ERROR(id4_smoke_stage_validate_create_options(options));

  id4_smoke_stage_t* stage = NULL;
  bool base_initialized = false;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*stage), (void**)&stage);
  if (iree_status_is_ok(status)) {
    memset(stage, 0, sizeof(*stage));
    stage->host_allocator = host_allocator;
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_stage_initialize(&id4_smoke_stage_vtable,
                                           &options->services, &stage->base);
    base_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    stage->kernel_cache = options->kernel_cache;
    id4_pipeline_kernel_cache_retain(stage->kernel_cache);
  }
  if (iree_status_is_ok(status)) {
    status = id4_smoke_stage_copy_string(options->module_path, host_allocator,
                                         &stage->module_path);
  }
  if (iree_status_is_ok(status)) {
    status = id4_smoke_stage_copy_string(options->function_name, host_allocator,
                                         &stage->function_name);
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

iree_hal_buffer_t* id4_smoke_stage_bundle_output_buffer(
    id4_pipeline_bundle_t* bundle) {
  id4_smoke_stage_bundle_payload_t* payload =
      (id4_smoke_stage_bundle_payload_t*)id4_pipeline_bundle_payload(bundle);
  return payload ? payload->output_buffer : NULL;
}
