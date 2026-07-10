// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_prepare.h"

#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/parameter_window.h"
#include "experimental/id4/pipeline/program_region.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/buffer_transfer.h"

typedef struct id4_pipeline_program_prepared_kernel_t {
  // Prepared kernel executable retained by the prepared program.
  id4_pipeline_kernel_executable_t* executable;
  // HAL executable function selected for the planned export name.
  iree_hal_executable_function_t function;
  // Static HAL dispatch configuration resolved by the kernel cache.
  iree_hal_dispatch_config_t dispatch_config;
} id4_pipeline_program_prepared_kernel_t;

struct id4_pipeline_program_prepared_t {
  // Reference count for shared prepared program ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for prepared program storage.
  iree_allocator_t host_allocator;
  // Plan retained for kernel, region, and binding metadata.
  id4_pipeline_plan_t* plan;
  // Number of sealed prepared regions.
  iree_host_size_t region_count;
  // Sealed prepared regions in plan region order.
  id4_pipeline_prepared_region_t** prepared_regions;
  // Whether prepared regions use compact region-local parameter bindings.
  bool uses_compact_parameter_windows;
  // Compact parameter windows retained in plan region order.
  id4_pipeline_parameter_window_t** parameter_windows;
  // Compact parameter loading schedules retained in plan region order.
  id4_pipeline_parameter_window_schedule_t** parameter_window_schedules;
  // Number of prepared kernel entries.
  iree_host_size_t kernel_count;
  // Prepared kernel entries in plan kernel order.
  id4_pipeline_program_prepared_kernel_t* kernels;
  // Number of materialized constant slab buffers.
  iree_host_size_t constant_slab_count;
  // Device buffers containing program-owned constants in plan slab order.
  iree_hal_buffer_t** constant_slab_buffers;
  // HAL queue-alloca flags used for plan-shared transient slabs.
  iree_hal_alloca_flags_t shared_slab_alloca_flags;
  // HAL queue-dealloca flags used for plan-shared transient slabs.
  iree_hal_dealloca_flags_t shared_slab_dealloca_flags;
};

typedef struct id4_pipeline_program_prepare_context_t {
  // Prepare options borrowed for recording.
  const id4_pipeline_program_prepare_options_t* options;
  // Prepared object receiving executables and the sealed region.
  id4_pipeline_program_prepared_t* prepared;
  // Optional compact parameter window used while recording one region.
  const id4_pipeline_parameter_window_t* parameter_window;
} id4_pipeline_program_prepare_context_t;

typedef uint32_t id4_pipeline_program_issue_wait_flags_t;

enum {
  ID4_PIPELINE_PROGRAM_ISSUE_WAIT_FLAG_INCLUDE_BUNDLE_READINESS = 1u << 0,
  ID4_PIPELINE_PROGRAM_ISSUE_WAIT_FLAG_INCLUDE_PARAMETER_READINESS = 1u << 1,
};

static iree_status_t id4_pipeline_program_emit_lifecycle(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_string_view_t stage_name, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the program runtime.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stage owning the program execution.
      .stage_name = stage_name,
      // Stable event key for this lifecycle boundary.
      .key = key,
      // Human-readable event context.
      .message = message,
      // No parameter slab payload is attached to lifecycle events.
      .parameter_slab = NULL,
      // No parameter loading payload is attached to lifecycle events.
      .parameter_load = NULL,
      // No kernel payload is attached to lifecycle events.
      .kernel = NULL,
      // No timing payload is attached to lifecycle events.
      .timing = NULL,
  };
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_pipeline_program_wait_after_region_issue(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_string_view_t stage_name, iree_string_view_t region_name,
    iree_hal_semaphore_list_t signal_list) {
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  return id4_pipeline_program_emit_lifecycle(
      diagnostics_sink, stage_name, IREE_SV("program.region.issue.completed"),
      region_name);
}

typedef struct id4_pipeline_program_parameter_window_slot_t {
  // Plan region currently using this slot, or IREE_HOST_SIZE_MAX when idle.
  iree_host_size_t region_index;
  // Compact parameter schedule for the active region.
  const id4_pipeline_parameter_window_schedule_t* schedule;
  // Stack-owned slice of queue-allocated compact parameter slab buffers.
  iree_hal_buffer_t** buffers;
  // Stack-owned slice of per-buffer alloca completion semaphores.
  iree_hal_semaphore_t** alloca_semaphores;
  // Stack-owned slice of per-buffer alloca completion payload values.
  uint64_t* alloca_payload_values;
  // Stack-owned slice of per-load-group completion semaphores.
  iree_hal_semaphore_t** load_semaphores;
  // Stack-owned slice of per-load-group completion payload values.
  uint64_t* load_payload_values;
  // Number of compact parameter buffer allocas submitted for this slot.
  iree_host_size_t alloca_submitted_count;
  // Number of compact parameter load groups submitted for this slot.
  iree_host_size_t load_submitted_count;
  // Latest cleanup edge for this slot's queue-deallocated buffers.
  iree_hal_semaphore_t* cleanup_semaphore;
  // Payload value paired with cleanup_semaphore.
  uint64_t cleanup_payload_value;
} id4_pipeline_program_parameter_window_slot_t;

static iree_status_t id4_pipeline_program_prepare_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_program_prepare_validate_options(
    const id4_pipeline_program_prepare_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepare_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program prepare")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program prepare extension structures are not supported");
  }
  const id4_pipeline_program_prepare_flags_t allowed_flags =
      ID4_PIPELINE_PROGRAM_PREPARE_FLAG_COMPACT_PARAMETER_WINDOWS;
  if (options->flags & ~allowed_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare flags 0x%08" PRIx32
                            " contain unsupported bits",
                            options->flags & ~allowed_flags);
  }
  if (!options->program) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare program is required");
  }
  if (!options->plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare plan is required");
  }
  if (!iree_string_view_equal(id4_pipeline_program_name(options->program),
                              id4_pipeline_plan_stage_name(options->plan))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program prepare requires the program and plan names to match");
  }
  if (!options->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare kernel cache is required");
  }
  if (!options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare kernel library is required");
  }
  if (!options->executable_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare HAL executable cache is required");
  }
  if (id4_pipeline_plan_region_count(options->plan) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare requires at least one region");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("program prepare")));
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_create_empty(
    const id4_pipeline_program_prepare_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_program_prepared_t** out_prepared) {
  IREE_ASSERT_ARGUMENT(out_prepared);
  *out_prepared = NULL;

  const iree_host_size_t kernel_count =
      id4_pipeline_plan_kernel_count(options->plan);
  const iree_host_size_t region_count =
      id4_pipeline_plan_region_count(options->plan);
  const iree_host_size_t constant_slab_count =
      id4_pipeline_plan_constant_slab_count(options->plan);
  id4_pipeline_program_prepared_t* prepared = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*prepared), (void**)&prepared);
  if (iree_status_is_ok(status)) {
    memset(prepared, 0, sizeof(*prepared));
    iree_atomic_ref_count_init(&prepared->ref_count);
    prepared->host_allocator = host_allocator;
    prepared->plan = (id4_pipeline_plan_t*)options->plan;
    id4_pipeline_plan_retain(prepared->plan);
    prepared->region_count = region_count;
    prepared->uses_compact_parameter_windows = iree_all_bits_set(
        options->flags,
        ID4_PIPELINE_PROGRAM_PREPARE_FLAG_COMPACT_PARAMETER_WINDOWS);
    prepared->kernel_count = kernel_count;
    prepared->constant_slab_count = constant_slab_count;
    prepared->shared_slab_alloca_flags = options->local_slab_alloca_flags;
    prepared->shared_slab_dealloca_flags = options->local_slab_dealloca_flags;
  }
  if (iree_status_is_ok(status) && region_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, region_count,
                                         sizeof(prepared->prepared_regions[0]),
                                         (void**)&prepared->prepared_regions);
  }
  if (iree_status_is_ok(status) && region_count != 0) {
    memset(prepared->prepared_regions, 0,
           region_count * sizeof(prepared->prepared_regions[0]));
  }
  if (iree_status_is_ok(status) && prepared->uses_compact_parameter_windows &&
      region_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, region_count,
                                         sizeof(prepared->parameter_windows[0]),
                                         (void**)&prepared->parameter_windows);
  }
  if (iree_status_is_ok(status) && prepared->uses_compact_parameter_windows &&
      region_count != 0) {
    memset(prepared->parameter_windows, 0,
           region_count * sizeof(prepared->parameter_windows[0]));
  }
  if (iree_status_is_ok(status) && prepared->uses_compact_parameter_windows &&
      region_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, region_count,
        sizeof(prepared->parameter_window_schedules[0]),
        (void**)&prepared->parameter_window_schedules);
  }
  if (iree_status_is_ok(status) && prepared->uses_compact_parameter_windows &&
      region_count != 0) {
    memset(prepared->parameter_window_schedules, 0,
           region_count * sizeof(prepared->parameter_window_schedules[0]));
  }
  if (iree_status_is_ok(status) && kernel_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, kernel_count,
                                         sizeof(prepared->kernels[0]),
                                         (void**)&prepared->kernels);
  }
  if (iree_status_is_ok(status) && kernel_count != 0) {
    memset(prepared->kernels, 0, kernel_count * sizeof(prepared->kernels[0]));
    for (iree_host_size_t i = 0; i < kernel_count; ++i) {
      prepared->kernels[i].function = iree_hal_executable_function_invalid();
    }
  }
  if (iree_status_is_ok(status) && constant_slab_count != 0) {
    status =
        iree_allocator_malloc_array(host_allocator, constant_slab_count,
                                    sizeof(prepared->constant_slab_buffers[0]),
                                    (void**)&prepared->constant_slab_buffers);
  }
  if (iree_status_is_ok(status) && constant_slab_count != 0) {
    memset(prepared->constant_slab_buffers, 0,
           constant_slab_count * sizeof(prepared->constant_slab_buffers[0]));
  }
  if (iree_status_is_ok(status)) {
    *out_prepared = prepared;
  } else {
    id4_pipeline_program_prepared_release(prepared);
  }
  return status;
}

static iree_status_t id4_pipeline_program_prepare_kernel(
    id4_pipeline_program_prepared_t* prepared,
    const id4_pipeline_program_prepare_options_t* options,
    iree_host_size_t kernel_index) {
  const id4_pipeline_kernel_plan_t* kernel_plan =
      id4_pipeline_plan_kernel_at(options->plan, kernel_index);
  if (!kernel_plan) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program kernel plan %" PRIhsz " is missing",
                            kernel_index);
  }
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(options->plan, kernel_plan->placement_id);
  if (!placement) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program kernel %.*s references missing placement %u",
        (int)kernel_plan->specialization_key.size,
        kernel_plan->specialization_key.data, kernel_plan->placement_id);
  }
  const id4_pipeline_kernel_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_lookup(
      options->kernel_library, kernel_plan->module_path, &module));

  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = options->executable_cache;
  prepare_options.queue_affinity = placement->queue_affinity;
  prepare_options.caching_mode = options->executable_caching_mode;
  prepare_options.source_identifier = module->source_identifier;
  prepare_options.source_contents = module->source_contents;
  prepare_options.module_path = kernel_plan->module_path;
  prepare_options.function_name = kernel_plan->function_name;
  prepare_options.config_binding_count = kernel_plan->config_binding_count;
  prepare_options.config_bindings = kernel_plan->config_bindings;
  prepare_options.diagnostic_artifact_flags =
      options->diagnostic_artifact_flags;
  prepare_options.diagnostics_sink = options->diagnostics_sink;
  id4_pipeline_kernel_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_prepare_executable(
      options->kernel_cache, &prepare_options, &executable));

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  iree_status_t status = iree_hal_executable_lookup_function_by_name(
      id4_pipeline_kernel_executable_hal_executable(executable),
      kernel_plan->function_name, &function);
  if (iree_status_is_ok(status)) {
    prepared->kernels[kernel_index].executable = executable;
    prepared->kernels[kernel_index].function = function;
    prepared->kernels[kernel_index].dispatch_config =
        id4_pipeline_kernel_executable_dispatch_config(executable);
    executable = NULL;
  }
  id4_pipeline_kernel_executable_release(executable);
  return status;
}

static iree_status_t id4_pipeline_program_prepare_kernels(
    id4_pipeline_program_prepared_t* prepared,
    const id4_pipeline_program_prepare_options_t* options) {
  for (iree_host_size_t i = 0; i < prepared->kernel_count; ++i) {
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_prepare_kernel(prepared, options, i));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepare_find_constant_operation(
    const id4_pipeline_program_t* program, iree_host_size_t constant_ordinal,
    const id4_pipeline_program_op_t** out_op,
    const id4_pipeline_program_tensor_record_t** out_tensor) {
  *out_op = NULL;
  *out_tensor = NULL;
  iree_host_size_t current_ordinal = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT) continue;
    if (current_ordinal++ != constant_ordinal) continue;
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(program,
                                       op->payload.constant.tensor.ordinal);
    if (!tensor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program constant tensor %u is missing",
                              op->payload.constant.tensor.ordinal);
    }
    *out_op = op;
    *out_tensor = tensor;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "program constant %" PRIhsz " is missing",
                          constant_ordinal);
}

static iree_status_t id4_pipeline_program_prepare_constant_slab_buffer(
    id4_pipeline_program_prepared_t* prepared,
    const id4_pipeline_program_prepare_options_t* options,
    iree_host_size_t slab_index, iree_host_size_t first_constant_ordinal,
    iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;
  const id4_pipeline_constant_slab_plan_t* slab =
      id4_pipeline_plan_constant_slab_at(options->plan, slab_index);
  if (!slab) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program constant slab %" PRIhsz " is missing",
                            slab_index);
  }
  if (slab->byte_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program constant slab %.*s byte length %" PRIu64
                            " exceeds host allocation capacity",
                            (int)slab->name.size, slab->name.data,
                            (uint64_t)slab->byte_length);
  }
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(options->plan, slab->placement_id);
  if (!placement) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program constant slab %.*s references missing "
                            "placement %u",
                            (int)slab->name.size, slab->name.data,
                            slab->placement_id);
  }
  iree_hal_device_group_t* device_group =
      id4_pipeline_plan_device_group(options->plan);
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, placement->device_index);
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program constant slab %.*s device is required",
                            (int)slab->name.size, slab->name.data);
  }

  uint8_t* host_bytes = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      prepared->host_allocator, (iree_host_size_t)slab->byte_length,
      sizeof(host_bytes[0]), (void**)&host_bytes);
  if (iree_status_is_ok(status)) {
    memset(host_bytes, 0, (iree_host_size_t)slab->byte_length);
  }
  for (iree_host_size_t i = 0;
       i < slab->request_count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_constant_request_t* request = &slab->requests[i];
    const id4_pipeline_program_op_t* constant_op = NULL;
    const id4_pipeline_program_tensor_record_t* tensor = NULL;
    status = id4_pipeline_program_prepare_find_constant_operation(
        options->program, first_constant_ordinal + i, &constant_op, &tensor);
    if (!iree_status_is_ok(status)) break;
    if (!iree_string_view_equal(request->name, tensor->name)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program constant slab %.*s request %.*s does not match constant "
          "tensor %.*s",
          (int)slab->name.size, slab->name.data, (int)request->name.size,
          request->name.data, (int)tensor->name.size, tensor->name.data);
      break;
    }
    if (constant_op->payload.constant.data_length != request->span.length) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "program constant %.*s byte length %" PRIhsz
                           " does not match planned request length %" PRIu64,
                           (int)tensor->name.size, tensor->name.data,
                           constant_op->payload.constant.data_length,
                           (uint64_t)request->span.length);
      break;
    }
    memcpy(host_bytes + request->span.buffer_offset,
           constant_op->payload.constant.data,
           constant_op->payload.constant.data_length);
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), slab->target_params,
        slab->byte_length, &buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_h2d(
        device, host_bytes, buffer, 0, slab->byte_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  iree_allocator_free(prepared->host_allocator, host_bytes);
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t id4_pipeline_program_prepare_constant_slabs(
    id4_pipeline_program_prepared_t* prepared,
    const id4_pipeline_program_prepare_options_t* options) {
  iree_host_size_t first_constant_ordinal = 0;
  for (iree_host_size_t i = 0; i < prepared->constant_slab_count; ++i) {
    const id4_pipeline_constant_slab_plan_t* slab =
        id4_pipeline_plan_constant_slab_at(options->plan, i);
    if (!slab) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program constant slab %" PRIhsz " is missing",
                              i);
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_program_prepare_constant_slab_buffer(
        prepared, options, i, first_constant_ordinal,
        &prepared->constant_slab_buffers[i]));
    first_constant_ordinal += slab->request_count;
  }
  return iree_ok_status();
}

static iree_host_size_t id4_pipeline_program_prepare_find_kernel(
    const id4_pipeline_plan_t* plan, iree_string_view_t specialization_key) {
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_kernel_count(plan); ++i) {
    const id4_pipeline_kernel_plan_t* kernel =
        id4_pipeline_plan_kernel_at(plan, i);
    if (kernel && iree_string_view_equal(kernel->specialization_key,
                                         specialization_key)) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static id4_pipeline_tensor_shape_t id4_pipeline_program_prepare_convert_shape(
    id4_pipeline_program_shape_t source) {
  id4_pipeline_tensor_shape_t target;
  memset(&target, 0, sizeof(target));
  target.rank = source.rank;
  memcpy(target.dims, source.dims, sizeof(target.dims));
  return target;
}

static iree_status_t id4_pipeline_program_prepare_resolve_import(
    void* user_data, const id4_pipeline_program_import_op_t* import_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t import_ordinal, id4_pipeline_tensor_import_t* out_import) {
  (void)import_op;
  (void)tensor;
  id4_pipeline_program_prepare_context_t* context =
      (id4_pipeline_program_prepare_context_t*)user_data;
  const id4_pipeline_boundary_tensor_plan_t* boundary =
      id4_pipeline_plan_boundary_tensor_at(context->options->plan,
                                           import_ordinal);
  if (!boundary) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program boundary tensor %" PRIhsz " is missing",
                            import_ordinal);
  }
  out_import->layout = boundary->layout;
  out_import->binding_slot = boundary->binding_slot;
  out_import->offset = 0;
  out_import->flags =
      iree_all_bits_set(boundary->flags,
                        ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED)
          ? ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED
          : 0;
  return iree_ok_status();
}

static const id4_pipeline_parameter_window_slab_t*
id4_pipeline_program_prepare_find_window_slab(
    const id4_pipeline_parameter_window_t* parameter_window,
    iree_host_size_t original_slab_index) {
  const iree_host_size_t slab_count =
      id4_pipeline_parameter_window_slab_count(parameter_window);
  for (iree_host_size_t i = 0; i < slab_count; ++i) {
    const id4_pipeline_parameter_window_slab_t* slab =
        id4_pipeline_parameter_window_slab_at(parameter_window, i);
    if (slab && slab->original_slab_index == original_slab_index) return slab;
  }
  return NULL;
}

static iree_status_t id4_pipeline_program_prepare_validate_parameter_tensor(
    const id4_pipeline_parameter_tensor_plan_t* parameter_tensor,
    const id4_pipeline_program_parameter_op_t* parameter_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t parameter_ordinal) {
  if (!parameter_tensor) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program parameter tensor %" PRIhsz
                            " is missing from the plan",
                            parameter_ordinal);
  }
  if (parameter_tensor->program_tensor_ordinal !=
      parameter_op->tensor.ordinal) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program parameter tensor %" PRIhsz
        " plan ordinal %u does not match program tensor ordinal %u",
        parameter_ordinal, parameter_tensor->program_tensor_ordinal,
        parameter_op->tensor.ordinal);
  }
  if (!iree_string_view_equal(parameter_tensor->layout.name, tensor->name) ||
      parameter_tensor->layout.byte_length != tensor->byte_length ||
      parameter_tensor->layout.dtype !=
          id4_pipeline_program_region_convert_dtype(tensor->dtype)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program parameter tensor %" PRIhsz
                            " plan layout does not match program tensor %.*s",
                            parameter_ordinal, (int)tensor->name.size,
                            tensor->name.data);
  }
  if (parameter_tensor->layout.shape.rank != tensor->shape.rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program parameter tensor %" PRIhsz
                            " plan rank does not match program tensor %.*s",
                            parameter_ordinal, (int)tensor->name.size,
                            tensor->name.data);
  }
  for (uint32_t i = 0; i < tensor->shape.rank; ++i) {
    if (parameter_tensor->layout.shape.dims[i] != tensor->shape.dims[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program parameter tensor %" PRIhsz
                              " plan shape does not match program tensor %.*s",
                              parameter_ordinal, (int)tensor->name.size,
                              tensor->name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_prepare_resolve_window_parameter_tensor(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* parameter_window,
    const id4_pipeline_parameter_tensor_plan_t* parameter_tensor,
    iree_host_size_t parameter_ordinal,
    id4_pipeline_tensor_import_t* out_import) {
  const id4_pipeline_parameter_window_request_t* first_window_request =
      id4_pipeline_parameter_window_resolve_request(
          parameter_window, parameter_tensor->global_request_offset);
  if (!first_window_request) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program parameter tensor %" PRIhsz
                            " is outside the compact parameter window",
                            parameter_ordinal);
  }
  const id4_pipeline_parameter_window_slab_t* window_slab =
      id4_pipeline_program_prepare_find_window_slab(
          parameter_window, first_window_request->original_slab_index);
  if (!window_slab) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program parameter tensor %" PRIhsz
        " references missing compact slab for original slab %" PRIhsz,
        parameter_ordinal, first_window_request->original_slab_index);
  }
  const id4_pipeline_parameter_slab_plan_t* original_slab =
      id4_pipeline_plan_parameter_slab_at(
          plan, parameter_tensor->parameter_slab_index);
  const id4_pipeline_parameter_request_table_t* original_request_table =
      id4_pipeline_plan_parameter_request_table_at(
          plan, parameter_tensor->parameter_slab_index);
  if (!original_slab || !original_request_table) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program parameter tensor %" PRIhsz
                            " references missing original slab %" PRIhsz,
                            parameter_ordinal,
                            parameter_tensor->parameter_slab_index);
  }
  for (iree_host_size_t i = 0; i < parameter_tensor->request_count; ++i) {
    const iree_host_size_t slab_request_index =
        parameter_tensor->request_offset + i;
    const id4_pipeline_parameter_request_t* original_request =
        &original_request_table->values[slab_request_index];
    const id4_pipeline_parameter_window_request_t* window_request =
        id4_pipeline_parameter_window_resolve_request(
            parameter_window, parameter_tensor->global_request_offset + i);
    if (!window_request) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program parameter tensor %" PRIhsz
                              " request %" PRIhsz
                              " is outside the compact parameter window",
                              parameter_ordinal, i);
    }
    if (window_request->original_slab_index !=
            parameter_tensor->parameter_slab_index ||
        window_request->original_request_index != slab_request_index) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program parameter tensor %" PRIhsz
                              " compact request %" PRIhsz
                              " resolves to the wrong original request",
                              parameter_ordinal, i);
    }
    if (original_request->span.buffer_offset < parameter_tensor->offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program parameter tensor %" PRIhsz
                              " request %" PRIhsz
                              " starts before tensor storage",
                              parameter_ordinal, i);
    }
    const iree_device_size_t tensor_relative_offset =
        original_request->span.buffer_offset - parameter_tensor->offset;
    iree_device_size_t expected_window_offset = 0;
    if (!iree_device_size_checked_add(first_window_request->span.buffer_offset,
                                      tensor_relative_offset,
                                      &expected_window_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program parameter tensor %" PRIhsz
                              " compact request offset overflows",
                              parameter_ordinal);
    }
    if (window_request->span.buffer_offset != expected_window_offset ||
        window_request->span.length != original_request->span.length) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "program parameter tensor %" PRIhsz
                              " is not dense in the compact parameter window",
                              parameter_ordinal);
    }
  }
  out_import->layout = parameter_tensor->layout;
  out_import->binding_slot = window_slab->binding_slot;
  out_import->offset = first_window_request->span.buffer_offset;
  out_import->flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepare_resolve_parameter(
    void* user_data, const id4_pipeline_program_parameter_op_t* parameter_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t parameter_ordinal,
    id4_pipeline_tensor_import_t* out_import) {
  id4_pipeline_program_prepare_context_t* context =
      (id4_pipeline_program_prepare_context_t*)user_data;
  const id4_pipeline_parameter_tensor_plan_t* parameter_tensor =
      id4_pipeline_plan_parameter_tensor_at(context->options->plan,
                                            parameter_ordinal);
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepare_validate_parameter_tensor(
      parameter_tensor, parameter_op, tensor, parameter_ordinal));
  if (context->parameter_window) {
    return id4_pipeline_program_prepare_resolve_window_parameter_tensor(
        context->options->plan, context->parameter_window, parameter_tensor,
        parameter_ordinal, out_import);
  }

  const id4_pipeline_parameter_slab_plan_t* slab =
      id4_pipeline_plan_parameter_slab_at(
          context->options->plan, parameter_tensor->parameter_slab_index);
  if (!slab) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program parameter tensor %" PRIhsz " references missing slab %" PRIhsz,
        parameter_ordinal, parameter_tensor->parameter_slab_index);
  }
  out_import->layout = parameter_tensor->layout;
  out_import->binding_slot = slab->binding_slot;
  out_import->offset = parameter_tensor->offset;
  out_import->flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepare_resolve_constant(
    void* user_data, const id4_pipeline_program_constant_op_t* constant_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t constant_ordinal,
    id4_pipeline_tensor_import_t* out_import) {
  (void)constant_op;
  id4_pipeline_program_prepare_context_t* context =
      (id4_pipeline_program_prepare_context_t*)user_data;
  iree_host_size_t remaining_ordinal = constant_ordinal;
  for (iree_host_size_t slab_index = 0;
       slab_index <
       id4_pipeline_plan_constant_slab_count(context->options->plan);
       ++slab_index) {
    const id4_pipeline_constant_slab_plan_t* slab =
        id4_pipeline_plan_constant_slab_at(context->options->plan, slab_index);
    if (!slab) continue;
    if (remaining_ordinal >= slab->request_count) {
      remaining_ordinal -= slab->request_count;
      continue;
    }
    const id4_pipeline_constant_request_t* request =
        &slab->requests[remaining_ordinal];
    out_import->layout = (id4_pipeline_tensor_layout_t){
        // Constant tensor diagnostic name.
        .name = tensor->name,
        // Constant tensor element type.
        .dtype = id4_pipeline_program_region_convert_dtype(tensor->dtype),
        // Constant tensor shape.
        .shape = id4_pipeline_program_prepare_convert_shape(tensor->shape),
        // Dense constant tensor byte length.
        .byte_length = tensor->byte_length,
        // Constant subrange alignment is already represented by the plan
        // span offset.
        .alignment = 0,
    };
    out_import->binding_slot = slab->binding_slot;
    out_import->offset = request->span.buffer_offset;
    out_import->flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "program constant tensor %" PRIhsz " is missing",
                          constant_ordinal);
}

static iree_status_t id4_pipeline_program_prepare_resolve_shared_tensor(
    void* user_data, const id4_pipeline_program_acquire_op_t* acquire_op,
    const id4_pipeline_program_tensor_record_t* tensor, bool* out_is_shared,
    id4_pipeline_tensor_import_t* out_import) {
  (void)tensor;
  *out_is_shared = false;
  memset(out_import, 0, sizeof(*out_import));
  id4_pipeline_program_prepare_context_t* context =
      (id4_pipeline_program_prepare_context_t*)user_data;
  const id4_pipeline_plan_t* plan = context->options->plan;
  const iree_host_size_t shared_tensor_count =
      id4_pipeline_plan_shared_tensor_count(plan);
  for (iree_host_size_t i = 0; i < shared_tensor_count; ++i) {
    const id4_pipeline_shared_tensor_plan_t* shared_tensor =
        id4_pipeline_plan_shared_tensor_at(plan, i);
    if (!shared_tensor ||
        shared_tensor->program_tensor_ordinal != acquire_op->tensor.ordinal) {
      continue;
    }
    const id4_pipeline_memory_slab_plan_t* slab =
        id4_pipeline_plan_memory_slab_at(plan,
                                         shared_tensor->memory_slab_index);
    if (!slab || slab->scope != ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program shared tensor %.*s references a missing plan-shared slab",
          (int)shared_tensor->layout.name.size,
          shared_tensor->layout.name.data);
    }
    *out_is_shared = true;
    out_import->layout = shared_tensor->layout;
    out_import->binding_slot = slab->binding_slot;
    out_import->offset = shared_tensor->offset;
    out_import->flags = 0;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepare_resolve_kernel(
    void* user_data, const id4_pipeline_program_dispatch_loom_op_t* dispatch_op,
    iree_string_view_t specialization_key, iree_host_size_t dispatch_ordinal,
    id4_pipeline_program_region_kernel_resolution_t* out_resolution) {
  (void)dispatch_op;
  (void)dispatch_ordinal;
  id4_pipeline_program_prepare_context_t* context =
      (id4_pipeline_program_prepare_context_t*)user_data;
  const iree_host_size_t kernel_index =
      id4_pipeline_program_prepare_find_kernel(context->options->plan,
                                               specialization_key);
  if (kernel_index == IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "program kernel specialization %.*s was not prepared",
        (int)specialization_key.size, specialization_key.data);
  }
  id4_pipeline_program_prepared_kernel_t* kernel =
      &context->prepared->kernels[kernel_index];
  if (!kernel->executable ||
      !iree_hal_executable_function_is_valid(kernel->function)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "program kernel specialization %.*s is not ready",
                            (int)specialization_key.size,
                            specialization_key.data);
  }
  out_resolution->executable =
      id4_pipeline_kernel_executable_hal_executable(kernel->executable);
  out_resolution->function = kernel->function;
  out_resolution->dispatch_config = kernel->dispatch_config;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepare_resolve_tap(
    void* user_data, const id4_pipeline_program_tap_op_t* tap_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t tap_ordinal, id4_pipeline_tensor_import_t* out_import) {
  (void)tap_op;
  (void)tensor;
  id4_pipeline_program_prepare_context_t* context =
      (id4_pipeline_program_prepare_context_t*)user_data;
  const id4_pipeline_diagnostic_tap_plan_t* tap =
      id4_pipeline_plan_diagnostic_tap_at(context->options->plan, tap_ordinal);
  if (!tap) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program diagnostic tap %" PRIhsz " is missing",
                            tap_ordinal);
  }
  out_import->layout = tap->layout;
  out_import->binding_slot = tap->binding_slot;
  out_import->offset = 0;
  out_import->flags = 0;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepare_record_region(
    id4_pipeline_program_prepared_t* prepared,
    const id4_pipeline_program_prepare_options_t* options,
    iree_host_size_t region_index) {
  if (region_index > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program region index %" PRIhsz " exceeds uint32_t",
                            region_index);
  }
  const uint32_t region_id = (uint32_t)region_index;
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(options->plan, region_index);
  if (!region) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program region plan %" PRIhsz " is missing",
                            region_index);
  }
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(options->plan, region->placement_id);
  if (!placement) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program region references missing placement %u",
                            region->placement_id);
  }
  iree_hal_device_group_t* device_group =
      id4_pipeline_plan_device_group(options->plan);
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, placement->device_index);
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region placement device is required");
  }

  const id4_pipeline_memory_slab_plan_t* local_slab = NULL;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_memory_slab_count(options->plan); ++i) {
    const id4_pipeline_memory_slab_plan_t* slab =
        id4_pipeline_plan_memory_slab_at(options->plan, i);
    if (slab && slab->scope == ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL &&
        slab->region_id == region_id) {
      local_slab = slab;
      break;
    }
  }
  if (region->statistics.local_slab_byte_length != 0 && !local_slab) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region local slab plan is required");
  }
  if (local_slab && local_slab->placement_id != region->placement_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program region local slab placement %u does not match region "
        "placement %u",
        local_slab->placement_id, region->placement_id);
  }
  if (local_slab && local_slab->binding_slot != region->local_binding_slot) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program region local slab binding slot %u does not match region "
        "local binding slot %u",
        local_slab->binding_slot, region->local_binding_slot);
  }
  if (local_slab &&
      local_slab->byte_length != region->statistics.local_slab_byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program region local slab byte length %" PRIu64
        " does not match region requirement %" PRIu64,
        (uint64_t)local_slab->byte_length,
        (uint64_t)region->statistics.local_slab_byte_length);
  }

  iree_string_view_t* planned_tap_names = NULL;
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(options->plan);
  iree_status_t status = iree_ok_status();
  if (diagnostic_tap_count != 0) {
    status = iree_allocator_malloc_array(
        prepared->host_allocator, diagnostic_tap_count,
        sizeof(planned_tap_names[0]), (void**)&planned_tap_names);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < diagnostic_tap_count; ++i) {
      const id4_pipeline_diagnostic_tap_plan_t* tap =
          id4_pipeline_plan_diagnostic_tap_at(options->plan, i);
      if (!tap) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "program diagnostic tap %" PRIhsz " is missing from the plan", i);
        break;
      }
      planned_tap_names[i] = tap->name;
    }
  }
  if (iree_status_is_ok(status) && prepared->uses_compact_parameter_windows) {
    id4_pipeline_parameter_window_create_options_t window_options;
    memset(&window_options, 0, sizeof(window_options));
    window_options.structure_size = sizeof(window_options);
    window_options.plan = options->plan;
    window_options.region_offset = region_index;
    window_options.region_count = 1;
    status = id4_pipeline_parameter_window_create(
        &window_options, prepared->host_allocator,
        &prepared->parameter_windows[region_index]);
  }
  if (iree_status_is_ok(status) && prepared->uses_compact_parameter_windows) {
    id4_pipeline_parameter_window_schedule_create_options_t schedule_options;
    memset(&schedule_options, 0, sizeof(schedule_options));
    schedule_options.structure_size = sizeof(schedule_options);
    schedule_options.plan = prepared->plan;
    schedule_options.window = prepared->parameter_windows[region_index];
    status = id4_pipeline_parameter_window_schedule_create(
        &schedule_options, prepared->host_allocator,
        &prepared->parameter_window_schedules[region_index]);
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  id4_pipeline_region_builder_t* builder = NULL;
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   prepared->host_allocator, &block_pool);

  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_create(
        device, options->command_buffer_mode, IREE_HAL_COMMAND_CATEGORY_ANY,
        placement->queue_affinity, region->binding_capacity, &command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_create_options_t builder_options;
    memset(&builder_options, 0, sizeof(builder_options));
    builder_options.structure_size = sizeof(builder_options);
    builder_options.region_name = region->name;
    builder_options.mode = ID4_PIPELINE_REGION_BUILDER_MODE_RECORD;
    builder_options.block_pool = &block_pool;
    builder_options.command_buffer = command_buffer;
    builder_options.binding_capacity = region->binding_capacity;
    builder_options.local_binding_slot = region->local_binding_slot;
    status = id4_pipeline_region_builder_create(
        &builder_options, prepared->host_allocator, &builder);
  }
  id4_pipeline_program_prepare_context_t context = {
      // Prepare options borrowed during lowering.
      .options = options,
      // Prepared object containing resolved kernel executables.
      .prepared = prepared,
      // Compact parameter window used by this region, if enabled.
      .parameter_window = prepared->uses_compact_parameter_windows
                              ? prepared->parameter_windows[region_index]
                              : NULL,
  };
  if (iree_status_is_ok(status)) {
    id4_pipeline_program_region_lower_options_t lower_options;
    memset(&lower_options, 0, sizeof(lower_options));
    lower_options.structure_size = sizeof(lower_options);
    lower_options.program = options->program;
    lower_options.source_operation_offset = region->source_operation_offset;
    lower_options.source_operation_count = region->source_operation_count;
    lower_options.builder = builder;
    lower_options.tap_mode = diagnostic_tap_count == 0
                                 ? ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_IGNORE
                                 : ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_CAPTURE;
    lower_options.captured_tap_names = (iree_string_view_list_t){
        // Number of diagnostic taps selected by the plan.
        .count = diagnostic_tap_count,
        // Plan-owned tap names copied into prepare transient storage.
        .values = planned_tap_names,
    };
    lower_options.local_tensor_alignment = region->local_tensor_alignment;
    lower_options.user_data = &context;
    lower_options.resolve_import = id4_pipeline_program_prepare_resolve_import;
    lower_options.resolve_parameter =
        id4_pipeline_program_prepare_resolve_parameter;
    lower_options.resolve_constant =
        id4_pipeline_program_prepare_resolve_constant;
    lower_options.resolve_shared_tensor =
        id4_pipeline_program_prepare_resolve_shared_tensor;
    lower_options.resolve_kernel = id4_pipeline_program_prepare_resolve_kernel;
    lower_options.resolve_tap = id4_pipeline_program_prepare_resolve_tap;
    status = id4_pipeline_program_region_lower(&lower_options,
                                               prepared->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_prepared_region_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.device_group = device_group;
    create_options.device_index = placement->device_index;
    create_options.queue_affinity = placement->queue_affinity;
    create_options.local_slab_params =
        local_slab ? local_slab->params : (iree_hal_buffer_params_t){0};
    create_options.local_slab_alloca_flags = options->local_slab_alloca_flags;
    create_options.local_slab_dealloca_flags =
        options->local_slab_dealloca_flags;
    status = id4_pipeline_prepared_region_create(
        builder, &create_options, prepared->host_allocator,
        &prepared->prepared_regions[region_index]);
  }

  id4_pipeline_region_builder_destroy(builder);
  iree_hal_command_buffer_release(command_buffer);
  iree_allocator_free(prepared->host_allocator, planned_tap_names);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_pipeline_program_prepare_record_regions(
    id4_pipeline_program_prepared_t* prepared,
    const id4_pipeline_program_prepare_options_t* options) {
  for (iree_host_size_t i = 0; i < prepared->region_count; ++i) {
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_prepare_record_region(prepared, options, i));
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_program_prepare(
    const id4_pipeline_program_prepare_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_program_prepared_t** out_prepared) {
  IREE_ASSERT_ARGUMENT(out_prepared);
  *out_prepared = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepare_validate_options(options));

  id4_pipeline_program_prepared_t* prepared = NULL;
  iree_status_t status = id4_pipeline_program_prepared_create_empty(
      options, host_allocator, &prepared);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_prepare_kernels(prepared, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_prepare_constant_slabs(prepared, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_prepare_record_regions(prepared, options);
  }
  if (iree_status_is_ok(status)) {
    *out_prepared = prepared;
  } else {
    id4_pipeline_program_prepared_release(prepared);
  }
  return status;
}

void id4_pipeline_program_prepared_retain(
    id4_pipeline_program_prepared_t* prepared) {
  if (!prepared) return;
  iree_atomic_ref_count_inc(&prepared->ref_count);
}

static void id4_pipeline_program_prepared_destroy(
    id4_pipeline_program_prepared_t* prepared) {
  iree_allocator_t host_allocator = prepared->host_allocator;
  for (iree_host_size_t i = 0; i < prepared->region_count; ++i) {
    id4_pipeline_prepared_region_release(prepared->prepared_regions[i]);
  }
  iree_allocator_free(host_allocator, prepared->prepared_regions);
  for (iree_host_size_t i = 0; i < prepared->region_count; ++i) {
    id4_pipeline_parameter_window_release(
        prepared->parameter_windows ? prepared->parameter_windows[i] : NULL);
    id4_pipeline_parameter_window_schedule_release(
        prepared->parameter_window_schedules
            ? prepared->parameter_window_schedules[i]
            : NULL);
  }
  iree_allocator_free(host_allocator, prepared->parameter_windows);
  iree_allocator_free(host_allocator, prepared->parameter_window_schedules);
  for (iree_host_size_t i = 0; i < prepared->kernel_count; ++i) {
    id4_pipeline_kernel_executable_release(prepared->kernels[i].executable);
  }
  iree_allocator_free(host_allocator, prepared->kernels);
  for (iree_host_size_t i = 0; i < prepared->constant_slab_count; ++i) {
    iree_hal_buffer_release(prepared->constant_slab_buffers[i]);
  }
  iree_allocator_free(host_allocator, prepared->constant_slab_buffers);
  id4_pipeline_plan_release(prepared->plan);
  iree_allocator_free(host_allocator, prepared);
}

void id4_pipeline_program_prepared_release(
    id4_pipeline_program_prepared_t* prepared) {
  if (prepared && iree_atomic_ref_count_dec(&prepared->ref_count) == 1) {
    id4_pipeline_program_prepared_destroy(prepared);
  }
}

static iree_status_t id4_pipeline_program_prepared_make_initial_wait_list(
    id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options,
    id4_pipeline_program_issue_wait_flags_t flags,
    iree_hal_semaphore_t** semaphores, uint64_t* payload_values,
    iree_hal_semaphore_list_t* out_wait_list) {
  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  const iree_host_size_t readiness_count =
      iree_all_bits_set(
          flags, ID4_PIPELINE_PROGRAM_ISSUE_WAIT_FLAG_INCLUDE_BUNDLE_READINESS)
          ? readiness_list.count
          : 0;
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(
          readiness_count, options->wait_semaphore_list.count, &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue wait list count overflow");
  }
  for (iree_host_size_t i = 0; i < readiness_count; ++i) {
    semaphores[i] = readiness_list.semaphores[i];
    payload_values[i] = readiness_list.payload_values[i];
  }
  for (iree_host_size_t i = 0; i < options->wait_semaphore_list.count; ++i) {
    const iree_host_size_t target_index = readiness_count + i;
    semaphores[target_index] = options->wait_semaphore_list.semaphores[i];
    payload_values[target_index] =
        options->wait_semaphore_list.payload_values[i];
  }
  *out_wait_list = (iree_hal_semaphore_list_t){
      // Number of wait semaphores.
      .count = wait_count,
      // Combined readiness and caller wait semaphores.
      .semaphores = wait_count == 0 ? NULL : semaphores,
      // Combined readiness and caller wait payload values.
      .payload_values = wait_count == 0 ? NULL : payload_values,
  };
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_prepared_max_region_parameter_load_group_count(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_count) {
  *out_count = 0;
  const iree_host_size_t region_count = id4_pipeline_plan_region_count(plan);
  for (iree_host_size_t i = 0; i < region_count; ++i) {
    const id4_pipeline_region_plan_t* region =
        id4_pipeline_plan_region_at(plan, i);
    if (!region) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program issue region plan %" PRIhsz " is missing", i);
    }
    *out_count = iree_max(*out_count, region->parameter_load_group_count);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_make_region_wait_list(
    const id4_pipeline_region_plan_t* region,
    id4_pipeline_parameter_slab_set_t* parameter_slabs,
    iree_hal_semaphore_list_t base_wait_list,
    iree_hal_semaphore_list_t explicit_parameter_wait_list,
    id4_pipeline_program_issue_wait_flags_t flags,
    iree_hal_semaphore_t** semaphores, uint64_t* payload_values,
    iree_hal_semaphore_list_t* out_wait_list) {
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(out_wait_list);
  *out_wait_list = iree_hal_semaphore_list_empty();
  const bool uses_explicit_parameter_waits =
      explicit_parameter_wait_list.count != 0;
  const bool includes_retained_parameter_waits = iree_all_bits_set(
      flags, ID4_PIPELINE_PROGRAM_ISSUE_WAIT_FLAG_INCLUDE_PARAMETER_READINESS);
  if (region->parameter_load_group_count != 0 &&
      includes_retained_parameter_waits && !uses_explicit_parameter_waits &&
      !parameter_slabs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program issue region %.*s requires parameter load groups but has no "
        "parameter slab set",
        (int)region->name.size, region->name.data);
  }

  const iree_host_size_t parameter_wait_count =
      uses_explicit_parameter_waits       ? explicit_parameter_wait_list.count
      : includes_retained_parameter_waits ? region->parameter_load_group_count
                                          : 0;
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(base_wait_list.count, parameter_wait_count,
                                  &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue region wait list count overflow");
  }
  for (iree_host_size_t i = 0; i < base_wait_list.count; ++i) {
    semaphores[i] = base_wait_list.semaphores[i];
    payload_values[i] = base_wait_list.payload_values[i];
  }
  if (uses_explicit_parameter_waits) {
    for (iree_host_size_t i = 0; i < explicit_parameter_wait_list.count; ++i) {
      const iree_host_size_t target_index = base_wait_list.count + i;
      semaphores[target_index] = explicit_parameter_wait_list.semaphores[i];
      payload_values[target_index] =
          explicit_parameter_wait_list.payload_values[i];
    }
  } else if (includes_retained_parameter_waits) {
    for (iree_host_size_t i = 0; i < region->parameter_load_group_count; ++i) {
      const iree_host_size_t target_index = base_wait_list.count + i;
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_set_load_group_ready_at(
          parameter_slabs, region->parameter_load_groups[i],
          &semaphores[target_index], &payload_values[target_index]));
    }
  }
  *out_wait_list = (iree_hal_semaphore_list_t){
      // Number of region wait semaphores.
      .count = wait_count,
      // Combined structural and parameter readiness semaphores.
      .semaphores = wait_count == 0 ? NULL : semaphores,
      // Combined structural and parameter readiness payloads.
      .payload_values = wait_count == 0 ? NULL : payload_values,
  };
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_prepared_submit_region_parameter_load_groups(
    const id4_pipeline_plan_t* plan, iree_host_size_t submit_region_id,
    const id4_pipeline_region_plan_t* region,
    id4_pipeline_parameter_slab_issue_context_t* parameter_issue_context,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(parameter_issue_context);
  for (iree_host_size_t i = 0; i < region->parameter_load_group_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_submit_parameter_load_group(
        plan, parameter_issue_context, region->parameter_load_groups[i],
        submit_region_id, diagnostics_sink));
  }
  return iree_ok_status();
}

static iree_host_size_t
id4_pipeline_program_prepared_find_load_group_first_region(
    const id4_pipeline_plan_t* plan, iree_host_size_t group_index) {
  const iree_host_size_t region_count = id4_pipeline_plan_region_count(plan);
  for (iree_host_size_t i = 0; i < region_count; ++i) {
    const id4_pipeline_region_plan_t* region =
        id4_pipeline_plan_region_at(plan, i);
    if (!region) continue;
    for (iree_host_size_t j = 0; j < region->parameter_load_group_count; ++j) {
      if (region->parameter_load_groups[j] == group_index) return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t
id4_pipeline_program_prepared_submit_parameter_window_load_groups(
    const id4_pipeline_plan_t* plan, iree_host_size_t submit_region_id,
    const id4_pipeline_parameter_window_schedule_t* schedule,
    id4_pipeline_parameter_slab_issue_context_t* parameter_issue_context,
    iree_hal_buffer_t** parameter_window_buffers,
    iree_hal_semaphore_list_t target_wait_list,
    iree_hal_semaphore_t** load_semaphores, uint64_t* load_payload_values,
    iree_host_size_t* out_load_signal_count,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(parameter_issue_context);
  IREE_ASSERT_ARGUMENT(out_load_signal_count);
  *out_load_signal_count = 0;
  const iree_host_size_t load_count =
      id4_pipeline_parameter_window_schedule_load_count(schedule);
  const id4_pipeline_parameter_slab_load_t* loads =
      id4_pipeline_parameter_window_schedule_loads(schedule);
  const iree_host_size_t load_step_count =
      id4_pipeline_parameter_window_schedule_load_step_count(schedule);
  const id4_pipeline_parameter_load_step_t* load_steps =
      id4_pipeline_parameter_window_schedule_load_steps(schedule);
  const iree_host_size_t load_group_count =
      id4_pipeline_parameter_window_schedule_load_group_count(schedule);
  for (iree_host_size_t compact_group_index = 0;
       compact_group_index < load_group_count; ++compact_group_index) {
    id4_pipeline_parameter_load_group_t compact_group;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_group_at(
        load_step_count, load_steps, compact_group_index, &compact_group));
    if (compact_group.target_slab_index >= load_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program compact parameter load group %" PRIhsz
          " target slab %" PRIhsz " exceeds load count %" PRIhsz,
          compact_group_index, compact_group.target_slab_index, load_count);
    }
    const id4_pipeline_parameter_slab_load_t* load =
        &loads[compact_group.target_slab_index];
    const iree_host_size_t original_group_index =
        id4_pipeline_parameter_window_schedule_original_load_group_at(
            schedule, compact_group_index);
    id4_pipeline_parameter_load_group_context_t group_context = {
        // Original plan-local load group ordinal.
        .group_index = original_group_index,
        // First planned region that consumes the original load group.
        .first_consumer_region_id =
            id4_pipeline_program_prepared_find_load_group_first_region(
                plan, original_group_index),
        // Region currently submitting the compact load group.
        .submit_region_id = submit_region_id,
    };
    load_payload_values[*out_load_signal_count] = 1;
    iree_status_t status = iree_hal_semaphore_create(
        load->device, load->queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &load_semaphores[*out_load_signal_count]);
    if (!iree_status_is_ok(status)) return status;
    iree_hal_semaphore_t* load_signal_semaphore =
        load_semaphores[*out_load_signal_count];
    uint64_t load_signal_payload_value =
        load_payload_values[*out_load_signal_count];
    iree_hal_semaphore_list_t load_signal_list = {
        .count = 1,
        .semaphores = &load_signal_semaphore,
        .payload_values = &load_signal_payload_value,
    };
    status =
        id4_pipeline_parameter_slab_issue_context_submit_load_group_to_buffers(
            parameter_issue_context, load_count, loads, load_step_count,
            load_steps, compact_group_index, load_count,
            parameter_window_buffers, target_wait_list, load_signal_list,
            group_context, id4_pipeline_plan_stage_name(plan),
            diagnostics_sink);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_release(load_semaphores[*out_load_signal_count]);
      load_semaphores[*out_load_signal_count] = NULL;
      return status;
    }
    ++*out_load_signal_count;
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_prepared_submit_region_parameter_load_window(
    const id4_pipeline_plan_t* plan, iree_host_size_t current_region_id,
    iree_host_size_t region_count, iree_host_size_t prefetch_region_distance,
    id4_pipeline_parameter_slab_issue_context_t* parameter_issue_context,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(parameter_issue_context);
  if (region_count == 0) return iree_ok_status();
  if (current_region_id >= region_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue current region %" PRIhsz
                            " is outside region count %" PRIhsz,
                            current_region_id, region_count);
  }

  const iree_host_size_t remaining_region_count =
      region_count - current_region_id - 1;
  const iree_host_size_t lookahead_region_count =
      iree_min(prefetch_region_distance, remaining_region_count);
  const iree_host_size_t last_region_id =
      current_region_id + lookahead_region_count;
  for (iree_host_size_t region_id = current_region_id;
       region_id <= last_region_id; ++region_id) {
    const id4_pipeline_region_plan_t* region =
        id4_pipeline_plan_region_at(plan, region_id);
    if (!region) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program issue region plan %" PRIhsz " is missing", region_id);
    }
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_prepared_submit_region_parameter_load_groups(
            plan, current_region_id, region, parameter_issue_context,
            diagnostics_sink));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_make_binding_table(
    id4_pipeline_program_prepared_t* prepared, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options,
    iree_host_size_t region_index,
    const id4_pipeline_parameter_window_schedule_t* parameter_window_schedule,
    iree_hal_buffer_t* const* parameter_window_buffers,
    iree_hal_buffer_t* const* memory_slab_buffers,
    iree_hal_buffer_binding_t* bindings,
    iree_hal_buffer_binding_table_t* out_binding_table) {
  const id4_pipeline_plan_t* plan = id4_pipeline_bundle_plan(bundle);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, region_index);
  if (!region) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue region plan %" PRIhsz " is missing",
                            region_index);
  }
  if (region->binding_capacity != 0) {
    memset(bindings, 0, region->binding_capacity * sizeof(bindings[0]));
  }

  if (parameter_window_schedule) {
    const iree_host_size_t load_count =
        id4_pipeline_parameter_window_schedule_load_count(
            parameter_window_schedule);
    const id4_pipeline_parameter_slab_load_t* loads =
        id4_pipeline_parameter_window_schedule_loads(parameter_window_schedule);
    if (load_count != 0 && !parameter_window_buffers) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program issue compact parameter buffer table is required");
    }
    for (iree_host_size_t i = 0; i < load_count; ++i) {
      const id4_pipeline_parameter_slab_plan_t* slab = loads[i].slab;
      iree_hal_buffer_t* buffer = parameter_window_buffers[i];
      if (!slab || !buffer) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "program issue compact parameter slab %" PRIhsz " is missing", i);
      }
      bindings[slab->binding_slot] = (iree_hal_buffer_binding_t){
          // Queue-allocated compact parameter slab buffer.
          .buffer = buffer,
          // Compact parameter slabs are bound from byte zero.
          .offset = 0,
          // Compact parameter slab byte length.
          .length = slab->byte_length,
      };
    }
  } else {
    id4_pipeline_parameter_slab_set_t* parameter_slabs =
        id4_pipeline_bundle_parameter_slabs(bundle);
    const iree_host_size_t parameter_slab_count =
        id4_pipeline_plan_parameter_slab_count(plan);
    if (parameter_slab_count != 0 && !parameter_slabs) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program issue parameter slab set is required");
    }
    if (parameter_slab_count != 0 &&
        id4_pipeline_parameter_slab_set_count(parameter_slabs) !=
            parameter_slab_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program issue parameter slab count mismatch");
    }
    for (iree_host_size_t i = 0; i < parameter_slab_count; ++i) {
      const id4_pipeline_parameter_slab_plan_t* slab =
          id4_pipeline_plan_parameter_slab_at(plan, i);
      iree_hal_buffer_t* buffer =
          id4_pipeline_parameter_slab_set_buffer_at(parameter_slabs, i);
      if (!slab || !buffer) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "program issue parameter slab %" PRIhsz " is missing", i);
      }
      bindings[slab->binding_slot] = (iree_hal_buffer_binding_t){
          // Loaded parameter slab buffer retained by the bundle.
          .buffer = buffer,
          // Parameter slabs are bound from byte zero.
          .offset = 0,
          // Full parameter slab byte length.
          .length = slab->byte_length,
      };
    }
  }

  const iree_host_size_t constant_slab_count =
      id4_pipeline_plan_constant_slab_count(plan);
  if (constant_slab_count != prepared->constant_slab_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program issue constant slab count mismatch");
  }
  for (iree_host_size_t i = 0; i < constant_slab_count; ++i) {
    const id4_pipeline_constant_slab_plan_t* slab =
        id4_pipeline_plan_constant_slab_at(plan, i);
    iree_hal_buffer_t* buffer = prepared->constant_slab_buffers[i];
    if (!slab || !buffer) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program issue constant slab %" PRIhsz " is missing", i);
    }
    bindings[slab->binding_slot] = (iree_hal_buffer_binding_t){
        // Prepared constant slab buffer owned by the prepared program.
        .buffer = buffer,
        // Constant slabs are bound from byte zero.
        .offset = 0,
        // Full constant slab byte length.
        .length = slab->byte_length,
    };
  }

  const iree_host_size_t memory_slab_count =
      id4_pipeline_plan_memory_slab_count(plan);
  for (iree_host_size_t i = 0; i < memory_slab_count; ++i) {
    const id4_pipeline_memory_slab_plan_t* slab =
        id4_pipeline_plan_memory_slab_at(plan, i);
    if (!slab || slab->scope != ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED) {
      continue;
    }
    iree_hal_buffer_t* buffer =
        memory_slab_buffers ? memory_slab_buffers[i] : NULL;
    if (!buffer) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program issue plan-shared memory slab %" PRIhsz " is missing", i);
    }
    bindings[slab->binding_slot] = (iree_hal_buffer_binding_t){
        // Allocated shared transient slab buffer for this issue.
        .buffer = buffer,
        // Shared slabs are bound from byte zero.
        .offset = 0,
        // Full shared slab byte length.
        .length = slab->byte_length,
    };
  }

  const iree_host_size_t boundary_tensor_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  if (options->boundary_binding_count != boundary_tensor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program issue boundary binding count mismatch");
  }
  for (iree_host_size_t i = 0; i < boundary_tensor_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    bindings[boundary->binding_slot] = options->boundary_bindings[i];
  }

  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  if (options->diagnostic_tap_binding_count != diagnostic_tap_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program issue diagnostic tap binding count mismatch");
  }
  for (iree_host_size_t i = 0; i < diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (!tap) continue;
    bindings[tap->binding_slot] = options->diagnostic_tap_bindings[i];
  }

  const bool local_slot_is_trailing =
      region->local_binding_slot + 1 == region->binding_capacity;
  const iree_host_size_t binding_count =
      region->statistics.local_slab_byte_length == 0 && local_slot_is_trailing
          ? region->local_binding_slot
          : region->binding_capacity;
  *out_binding_table = (iree_hal_buffer_binding_table_t){
      // Exact issue-time binding count expected by the prepared region.
      .count = binding_count,
      // Stack-local full binding table for this issue call.
      .bindings = bindings,
  };
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_max_binding_capacity(
    const id4_pipeline_program_prepared_t* prepared,
    iree_host_size_t* out_binding_capacity) {
  *out_binding_capacity = 0;
  for (iree_host_size_t i = 0; i < prepared->region_count; ++i) {
    const id4_pipeline_region_plan_t* region =
        id4_pipeline_plan_region_at(prepared->plan, i);
    if (!region) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program issue region plan %" PRIhsz " is missing", i);
    }
    *out_binding_capacity =
        iree_max(*out_binding_capacity, region->binding_capacity);
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_prepared_max_parameter_window_load_count(
    const id4_pipeline_program_prepared_t* prepared,
    iree_host_size_t* out_load_count) {
  *out_load_count = 0;
  if (!prepared->uses_compact_parameter_windows) return iree_ok_status();
  for (iree_host_size_t i = 0; i < prepared->region_count; ++i) {
    const id4_pipeline_parameter_window_schedule_t* schedule =
        prepared->parameter_window_schedules[i];
    if (!schedule) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "program compact parameter schedule %" PRIhsz " is missing", i);
    }
    *out_load_count =
        iree_max(*out_load_count,
                 id4_pipeline_parameter_window_schedule_load_count(schedule));
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_prepared_max_parameter_window_load_group_count(
    const id4_pipeline_program_prepared_t* prepared,
    iree_host_size_t* out_load_group_count) {
  *out_load_group_count = 0;
  if (!prepared->uses_compact_parameter_windows) return iree_ok_status();
  for (iree_host_size_t i = 0; i < prepared->region_count; ++i) {
    const id4_pipeline_parameter_window_schedule_t* schedule =
        prepared->parameter_window_schedules[i];
    if (!schedule) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "program compact parameter schedule %" PRIhsz " is missing", i);
    }
    *out_load_group_count = iree_max(
        *out_load_group_count,
        id4_pipeline_parameter_window_schedule_load_group_count(schedule));
  }
  return iree_ok_status();
}

static iree_hal_semaphore_list_t id4_pipeline_program_one_semaphore_list(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return (iree_hal_semaphore_list_t){
      // Number of semaphores in the list.
      .count = 1,
      // Single semaphore pointer.
      .semaphores = semaphore,
      // Single payload value pointer.
      .payload_values = payload_value,
  };
}

static iree_hal_semaphore_list_t id4_pipeline_program_many_semaphore_list(
    iree_host_size_t count, iree_hal_semaphore_t** semaphores,
    uint64_t* payload_values) {
  return (iree_hal_semaphore_list_t){
      // Number of semaphores in the list.
      .count = count,
      // Semaphore pointer list.
      .semaphores = count == 0 ? NULL : semaphores,
      // Payload values paired with semaphores.
      .payload_values = count == 0 ? NULL : payload_values,
  };
}

static void id4_pipeline_program_release_semaphores(
    iree_host_size_t count, iree_hal_semaphore_t** semaphores) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_semaphore_release(semaphores[i]);
  }
}

static void id4_pipeline_program_release_buffers(iree_host_size_t count,
                                                 iree_hal_buffer_t** buffers) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_buffer_release(buffers[i]);
    buffers[i] = NULL;
  }
}

static iree_status_t id4_pipeline_program_prepared_create_internal_signals(
    id4_pipeline_program_prepared_t* prepared,
    iree_host_size_t internal_signal_count, iree_hal_semaphore_t** semaphores,
    uint64_t* payload_values) {
  iree_hal_device_group_t* device_group =
      id4_pipeline_plan_device_group(prepared->plan);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < internal_signal_count && iree_status_is_ok(status); ++i) {
    payload_values[i] = 1;
    const id4_pipeline_region_plan_t* source_region =
        id4_pipeline_plan_region_at(prepared->plan, i);
    const id4_pipeline_region_plan_t* target_region =
        id4_pipeline_plan_region_at(prepared->plan, i + 1);
    if (!source_region || !target_region) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "program internal region edge %" PRIhsz
                                " references a missing region",
                                i);
      break;
    }
    const id4_pipeline_device_placement_t* source_placement =
        id4_pipeline_plan_placement_at(prepared->plan,
                                       source_region->placement_id);
    const id4_pipeline_device_placement_t* target_placement =
        id4_pipeline_plan_placement_at(prepared->plan,
                                       target_region->placement_id);
    if (!source_placement || !target_placement) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "program internal region edge %" PRIhsz
                                " references a missing placement",
                                i);
      break;
    }
    if (source_placement->device_index != target_placement->device_index) {
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "program internal region edge %" PRIhsz
                                " crosses devices; multi-device issue "
                                "sequencing is not implemented",
                                i);
      break;
    }
    iree_hal_device_t* device = iree_hal_device_group_device_at(
        device_group, source_placement->device_index);
    if (!device) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program internal region edge %" PRIhsz " device is required", i);
      break;
    }
    status = iree_hal_semaphore_create(
        device, source_placement->queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphores[i]);
  }
  return status;
}

static iree_status_t id4_pipeline_program_prepared_count_shared_slabs(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_shared_slab_count) {
  *out_shared_slab_count = 0;
  const iree_host_size_t memory_slab_count =
      id4_pipeline_plan_memory_slab_count(plan);
  for (iree_host_size_t i = 0; i < memory_slab_count; ++i) {
    const id4_pipeline_memory_slab_plan_t* slab =
        id4_pipeline_plan_memory_slab_at(plan, i);
    if (!slab) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program memory slab %" PRIhsz " is missing", i);
    }
    if (slab->scope == ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED) {
      ++*out_shared_slab_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_collect_shared_slabs(
    const id4_pipeline_plan_t* plan, iree_host_size_t* shared_slab_indices,
    iree_host_size_t* out_shared_slab_count) {
  *out_shared_slab_count = 0;
  const iree_host_size_t memory_slab_count =
      id4_pipeline_plan_memory_slab_count(plan);
  for (iree_host_size_t i = 0; i < memory_slab_count; ++i) {
    const id4_pipeline_memory_slab_plan_t* slab =
        id4_pipeline_plan_memory_slab_at(plan, i);
    if (!slab) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program memory slab %" PRIhsz " is missing", i);
    }
    if (slab->scope == ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED) {
      shared_slab_indices[*out_shared_slab_count] = i;
      ++*out_shared_slab_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_validate_shared_slab(
    const id4_pipeline_program_prepared_t* prepared,
    const id4_pipeline_memory_slab_plan_t* slab) {
  if (!slab) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program shared memory slab is required");
  }
  const id4_pipeline_region_plan_t* first_region =
      id4_pipeline_plan_region_at(prepared->plan, 0);
  if (!first_region) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program first region is missing");
  }
  if (slab->placement_id != first_region->placement_id) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program shared memory slab %.*s placement %u differs from first "
        "region placement %u",
        (int)slab->name.size, slab->name.data, slab->placement_id,
        first_region->placement_id);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_shared_slab_device(
    const id4_pipeline_program_prepared_t* prepared,
    const id4_pipeline_memory_slab_plan_t* slab, iree_hal_device_t** out_device,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  *out_device = NULL;
  *out_queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_prepared_validate_shared_slab(prepared, slab));
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(prepared->plan, slab->placement_id);
  if (!placement) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program shared memory slab references missing "
                            "placement %u",
                            slab->placement_id);
  }
  iree_hal_device_group_t* device_group =
      id4_pipeline_plan_device_group(prepared->plan);
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, placement->device_index);
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program shared memory slab device is required");
  }
  *out_device = device;
  *out_queue_affinity = placement->queue_affinity;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_issue_shared_allocas(
    id4_pipeline_program_prepared_t* prepared,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_hal_semaphore_list_t wait_list, iree_host_size_t shared_slab_count,
    const iree_host_size_t* shared_slab_indices,
    iree_hal_buffer_t** memory_slab_buffers,
    iree_hal_semaphore_t** alloca_semaphores, uint64_t* alloca_payload_values,
    iree_host_size_t* out_submitted_count) {
  *out_submitted_count = 0;
  const id4_pipeline_plan_t* plan = prepared->plan;
  for (iree_host_size_t i = 0; i < shared_slab_count; ++i) {
    const iree_host_size_t memory_slab_index = shared_slab_indices[i];
    const id4_pipeline_memory_slab_plan_t* slab =
        id4_pipeline_plan_memory_slab_at(plan, memory_slab_index);
    if (!slab || slab->scope != ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program shared memory slab index %" PRIhsz
                              " is invalid",
                              memory_slab_index);
    }
    iree_hal_device_t* device = NULL;
    iree_hal_queue_affinity_t queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_shared_slab_device(
        prepared, slab, &device, &queue_affinity));
    alloca_payload_values[i] = 1;
    iree_status_t status = iree_hal_semaphore_create(
        device, queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &alloca_semaphores[i]);
    if (iree_status_is_ok(status)) {
      iree_hal_semaphore_list_t signal_list =
          id4_pipeline_program_one_semaphore_list(&alloca_semaphores[i],
                                                  &alloca_payload_values[i]);
      status = id4_pipeline_program_emit_lifecycle(
          diagnostics_sink, id4_pipeline_plan_stage_name(prepared->plan),
          IREE_SV("program.shared_alloca.issue.begin"), slab->name);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_release(alloca_semaphores[i]);
        alloca_semaphores[i] = NULL;
        return status;
      }
      status = iree_hal_device_queue_alloca(
          device, queue_affinity, wait_list, signal_list, /*pool=*/NULL,
          slab->params, slab->byte_length, prepared->shared_slab_alloca_flags,
          &memory_slab_buffers[memory_slab_index]);
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_program_emit_lifecycle(
            diagnostics_sink, id4_pipeline_plan_stage_name(prepared->plan),
            IREE_SV("program.shared_alloca.issue.submitted"), slab->name);
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_release(alloca_semaphores[i]);
      alloca_semaphores[i] = NULL;
      iree_hal_buffer_release(memory_slab_buffers[memory_slab_index]);
      memory_slab_buffers[memory_slab_index] = NULL;
      return status;
    }
    ++*out_submitted_count;
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_prepared_issue_parameter_window_allocas(
    const id4_pipeline_parameter_window_schedule_t* schedule,
    iree_hal_semaphore_list_t wait_list, iree_hal_buffer_t** buffers,
    iree_hal_semaphore_t** alloca_semaphores, uint64_t* alloca_payload_values,
    iree_host_size_t* out_submitted_count) {
  IREE_ASSERT_ARGUMENT(out_submitted_count);
  *out_submitted_count = 0;
  const iree_host_size_t load_count =
      id4_pipeline_parameter_window_schedule_load_count(schedule);
  const id4_pipeline_parameter_slab_load_t* loads =
      id4_pipeline_parameter_window_schedule_loads(schedule);
  if (load_count != 0 && !loads) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "program compact parameter schedule has no load descriptors");
  }
  for (iree_host_size_t i = 0; i < load_count; ++i) {
    const id4_pipeline_parameter_slab_load_t* load = &loads[i];
    if (!load->slab) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program compact parameter load %" PRIhsz " has no slab plan", i);
    }
    alloca_payload_values[i] = 1;
    iree_status_t status = iree_hal_semaphore_create(
        load->device, load->queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &alloca_semaphores[i]);
    if (iree_status_is_ok(status)) {
      iree_hal_semaphore_list_t signal_list =
          id4_pipeline_program_one_semaphore_list(&alloca_semaphores[i],
                                                  &alloca_payload_values[i]);
      status = iree_hal_device_queue_alloca(
          load->device, load->queue_affinity, wait_list, signal_list,
          /*pool=*/NULL, load->slab->target_params, load->slab->byte_length,
          IREE_HAL_ALLOCA_FLAG_NONE, &buffers[i]);
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_release(alloca_semaphores[i]);
      alloca_semaphores[i] = NULL;
      iree_hal_buffer_release(buffers[i]);
      buffers[i] = NULL;
      return status;
    }
    ++*out_submitted_count;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_prepared_dealloca_shared_slabs(
    id4_pipeline_program_prepared_t* prepared,
    iree_hal_semaphore_list_t initial_wait_list,
    iree_hal_semaphore_list_t final_signal_list,
    iree_host_size_t shared_slab_count,
    const iree_host_size_t* shared_slab_indices,
    iree_hal_buffer_t** memory_slab_buffers) {
  iree_hal_semaphore_t* chain_semaphore = NULL;
  uint64_t chain_payload_value = 1;
  iree_hal_semaphore_list_t wait_list = initial_wait_list;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < shared_slab_count && iree_status_is_ok(status); ++i) {
    const iree_host_size_t memory_slab_index = shared_slab_indices[i];
    const id4_pipeline_memory_slab_plan_t* slab =
        id4_pipeline_plan_memory_slab_at(prepared->plan, memory_slab_index);
    if (!slab || slab->scope != ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "program shared memory slab index %" PRIhsz
                                " is invalid",
                                memory_slab_index);
      break;
    }
    if (!memory_slab_buffers[memory_slab_index]) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "program shared memory slab buffer %" PRIhsz
                                " is missing",
                                memory_slab_index);
      break;
    }
    iree_hal_device_t* device = NULL;
    iree_hal_queue_affinity_t queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    status = id4_pipeline_program_prepared_shared_slab_device(
        prepared, slab, &device, &queue_affinity);
    if (!iree_status_is_ok(status)) break;

    iree_hal_semaphore_t* next_chain_semaphore = NULL;
    uint64_t next_chain_payload_value = 1;
    iree_hal_semaphore_list_t signal_list = final_signal_list;
    if (i + 1 < shared_slab_count) {
      status = iree_hal_semaphore_create(
          device, queue_affinity, /*initial_value=*/0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &next_chain_semaphore);
      if (!iree_status_is_ok(status)) break;
      signal_list = id4_pipeline_program_one_semaphore_list(
          &next_chain_semaphore, &next_chain_payload_value);
    }

    status = iree_hal_device_queue_dealloca(
        device, queue_affinity, wait_list, signal_list,
        memory_slab_buffers[memory_slab_index],
        prepared->shared_slab_dealloca_flags);
    iree_hal_semaphore_release(chain_semaphore);
    chain_semaphore = next_chain_semaphore;
    chain_payload_value = next_chain_payload_value;
    wait_list = iree_hal_semaphore_list_empty();
    if (chain_semaphore) {
      wait_list = id4_pipeline_program_one_semaphore_list(&chain_semaphore,
                                                          &chain_payload_value);
    }
  }
  iree_hal_semaphore_release(chain_semaphore);
  return status;
}

static iree_status_t id4_pipeline_program_prepared_dealloca_parameter_window(
    const id4_pipeline_parameter_window_schedule_t* schedule,
    iree_hal_semaphore_list_t initial_wait_list,
    iree_hal_semaphore_list_t final_signal_list, iree_host_size_t buffer_count,
    iree_hal_buffer_t** buffers) {
  const iree_host_size_t load_count =
      id4_pipeline_parameter_window_schedule_load_count(schedule);
  const id4_pipeline_parameter_slab_load_t* loads =
      id4_pipeline_parameter_window_schedule_loads(schedule);
  if (buffer_count > load_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program compact parameter dealloca buffer count "
                            "%" PRIhsz " exceeds load count %" PRIhsz,
                            buffer_count, load_count);
  }
  iree_hal_semaphore_t* chain_semaphore = NULL;
  uint64_t chain_payload_value = 1;
  iree_hal_semaphore_list_t wait_list = initial_wait_list;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < buffer_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_parameter_slab_load_t* load = &loads[i];
    if (!buffers[i]) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program compact parameter buffer %" PRIhsz " is missing", i);
      break;
    }

    iree_hal_semaphore_t* next_chain_semaphore = NULL;
    uint64_t next_chain_payload_value = 1;
    iree_hal_semaphore_list_t signal_list = final_signal_list;
    if (i + 1 < buffer_count) {
      status = iree_hal_semaphore_create(
          load->device, load->queue_affinity, /*initial_value=*/0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &next_chain_semaphore);
      if (!iree_status_is_ok(status)) break;
      signal_list = id4_pipeline_program_one_semaphore_list(
          &next_chain_semaphore, &next_chain_payload_value);
    }

    status = iree_hal_device_queue_dealloca(load->device, load->queue_affinity,
                                            wait_list, signal_list, buffers[i],
                                            IREE_HAL_DEALLOCA_FLAG_NONE);
    iree_hal_semaphore_release(chain_semaphore);
    chain_semaphore = next_chain_semaphore;
    chain_payload_value = next_chain_payload_value;
    wait_list = iree_hal_semaphore_list_empty();
    if (chain_semaphore) {
      wait_list = id4_pipeline_program_one_semaphore_list(&chain_semaphore,
                                                          &chain_payload_value);
    }
  }
  iree_hal_semaphore_release(chain_semaphore);
  return status;
}

static iree_status_t
id4_pipeline_program_prepared_cleanup_failed_parameter_window(
    const id4_pipeline_parameter_window_schedule_t* schedule,
    iree_host_size_t alloca_submitted_count,
    iree_hal_semaphore_list_t alloca_wait_list,
    iree_host_size_t load_submitted_count,
    iree_hal_semaphore_list_t load_wait_list, iree_hal_buffer_t** buffers) {
  if (alloca_submitted_count == 0) return iree_ok_status();
  iree_hal_semaphore_list_t cleanup_wait_list =
      load_submitted_count == 0 ? alloca_wait_list : load_wait_list;
  return id4_pipeline_program_prepared_dealloca_parameter_window(
      schedule, cleanup_wait_list, iree_hal_semaphore_list_empty(),
      alloca_submitted_count, buffers);
}

static iree_status_t
id4_pipeline_program_prepared_cleanup_and_release_failed_parameter_window(
    iree_status_t status,
    const id4_pipeline_parameter_window_schedule_t* schedule,
    iree_host_size_t alloca_submitted_count,
    iree_hal_semaphore_list_t alloca_wait_list,
    iree_host_size_t load_submitted_count,
    iree_hal_semaphore_list_t load_wait_list, iree_hal_buffer_t** buffers,
    iree_hal_semaphore_t** alloca_semaphores,
    iree_hal_semaphore_t** load_semaphores) {
  iree_status_t cleanup_status =
      id4_pipeline_program_prepared_cleanup_failed_parameter_window(
          schedule, alloca_submitted_count, alloca_wait_list,
          load_submitted_count, load_wait_list, buffers);
  status = iree_status_join(status, cleanup_status);
  id4_pipeline_program_release_buffers(alloca_submitted_count, buffers);
  id4_pipeline_program_release_semaphores(alloca_submitted_count,
                                          alloca_semaphores);
  id4_pipeline_program_release_semaphores(load_submitted_count,
                                          load_semaphores);
  return status;
}

static void id4_pipeline_program_parameter_window_slot_initialize(
    id4_pipeline_program_parameter_window_slot_t* slot,
    iree_hal_buffer_t** buffers, iree_hal_semaphore_t** alloca_semaphores,
    uint64_t* alloca_payload_values, iree_hal_semaphore_t** load_semaphores,
    uint64_t* load_payload_values) {
  memset(slot, 0, sizeof(*slot));
  slot->region_index = IREE_HOST_SIZE_MAX;
  slot->buffers = buffers;
  slot->alloca_semaphores = alloca_semaphores;
  slot->alloca_payload_values = alloca_payload_values;
  slot->load_semaphores = load_semaphores;
  slot->load_payload_values = load_payload_values;
}

static iree_status_t
id4_pipeline_program_parameter_window_slot_alloca_wait_list(
    const id4_pipeline_program_parameter_window_slot_t* slot,
    iree_hal_semaphore_list_t base_wait_list,
    iree_hal_semaphore_t** wait_semaphores, uint64_t* wait_payload_values,
    iree_hal_semaphore_list_t* out_wait_list) {
  IREE_ASSERT_ARGUMENT(slot);
  IREE_ASSERT_ARGUMENT(out_wait_list);
  *out_wait_list = iree_hal_semaphore_list_empty();
  const iree_host_size_t cleanup_wait_count = slot->cleanup_semaphore ? 1 : 0;
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(base_wait_list.count, cleanup_wait_count,
                                  &wait_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program compact parameter window alloca wait list count overflows");
  }
  for (iree_host_size_t i = 0; i < base_wait_list.count; ++i) {
    wait_semaphores[i] = base_wait_list.semaphores[i];
    wait_payload_values[i] = base_wait_list.payload_values[i];
  }
  if (slot->cleanup_semaphore) {
    wait_semaphores[base_wait_list.count] = slot->cleanup_semaphore;
    wait_payload_values[base_wait_list.count] = slot->cleanup_payload_value;
  }
  *out_wait_list = (iree_hal_semaphore_list_t){
      // Number of waits needed before this slot can allocate buffers.
      .count = wait_count,
      // Caller and slot-reuse cleanup wait semaphores.
      .semaphores = wait_count == 0 ? NULL : wait_semaphores,
      // Payload values paired with wait_semaphores.
      .payload_values = wait_count == 0 ? NULL : wait_payload_values,
  };
  return iree_ok_status();
}

static iree_hal_semaphore_list_t
id4_pipeline_program_parameter_window_slot_alloca_completion_wait_list(
    const id4_pipeline_program_parameter_window_slot_t* slot) {
  return id4_pipeline_program_many_semaphore_list(slot->alloca_submitted_count,
                                                  slot->alloca_semaphores,
                                                  slot->alloca_payload_values);
}

static iree_hal_semaphore_list_t
id4_pipeline_program_parameter_window_slot_load_completion_wait_list(
    const id4_pipeline_program_parameter_window_slot_t* slot) {
  return id4_pipeline_program_many_semaphore_list(slot->load_submitted_count,
                                                  slot->load_semaphores,
                                                  slot->load_payload_values);
}

static void id4_pipeline_program_parameter_window_slot_release_transient_refs(
    id4_pipeline_program_parameter_window_slot_t* slot) {
  id4_pipeline_program_release_buffers(slot->alloca_submitted_count,
                                       slot->buffers);
  id4_pipeline_program_release_semaphores(slot->alloca_submitted_count,
                                          slot->alloca_semaphores);
  id4_pipeline_program_release_semaphores(slot->load_submitted_count,
                                          slot->load_semaphores);
  slot->alloca_submitted_count = 0;
  slot->load_submitted_count = 0;
  slot->schedule = NULL;
  slot->region_index = IREE_HOST_SIZE_MAX;
}

static iree_status_t
id4_pipeline_program_prepared_cleanup_active_parameter_window_slot(
    id4_pipeline_program_parameter_window_slot_t* slot) {
  if (slot->region_index == IREE_HOST_SIZE_MAX ||
      slot->alloca_submitted_count == 0) {
    return iree_ok_status();
  }
  iree_hal_semaphore_list_t alloca_wait_list =
      id4_pipeline_program_parameter_window_slot_alloca_completion_wait_list(
          slot);
  iree_hal_semaphore_list_t load_wait_list =
      id4_pipeline_program_parameter_window_slot_load_completion_wait_list(
          slot);
  return id4_pipeline_program_prepared_cleanup_failed_parameter_window(
      slot->schedule, slot->alloca_submitted_count, alloca_wait_list,
      slot->load_submitted_count, load_wait_list, slot->buffers);
}

static iree_status_t
id4_pipeline_program_prepared_cleanup_active_parameter_window_slots(
    iree_status_t status, iree_host_size_t slot_count,
    id4_pipeline_program_parameter_window_slot_t* slots) {
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    iree_status_t cleanup_status =
        id4_pipeline_program_prepared_cleanup_active_parameter_window_slot(
            &slots[i]);
    status = iree_status_join(status, cleanup_status);
    id4_pipeline_program_parameter_window_slot_release_transient_refs(
        &slots[i]);
  }
  return status;
}

static iree_status_t id4_pipeline_program_prepared_submit_parameter_window_slot(
    id4_pipeline_program_prepared_t* prepared,
    iree_host_size_t submit_region_id, iree_host_size_t window_region_id,
    id4_pipeline_parameter_slab_issue_context_t* parameter_issue_context,
    iree_hal_semaphore_list_t base_wait_list,
    iree_hal_semaphore_t** alloca_wait_semaphores,
    uint64_t* alloca_wait_payload_values,
    id4_pipeline_program_parameter_window_slot_t* slot,
    iree_host_size_t max_load_count, iree_host_size_t max_load_group_count,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_ASSERT_ARGUMENT(prepared);
  IREE_ASSERT_ARGUMENT(parameter_issue_context);
  IREE_ASSERT_ARGUMENT(slot);
  if (slot->region_index != IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "program compact parameter window slot is still active for region "
        "%" PRIhsz,
        slot->region_index);
  }
  const id4_pipeline_parameter_window_schedule_t* schedule =
      prepared->parameter_window_schedules[window_region_id];
  if (!schedule) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "program compact parameter schedule %" PRIhsz
                            " is missing",
                            window_region_id);
  }

  slot->region_index = window_region_id;
  slot->schedule = schedule;
  const iree_host_size_t load_count =
      id4_pipeline_parameter_window_schedule_load_count(schedule);
  const iree_host_size_t load_group_count =
      id4_pipeline_parameter_window_schedule_load_group_count(schedule);
  if (load_count > max_load_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "program compact parameter window load count exceeds scratch capacity");
  }
  if (load_group_count > max_load_group_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "program compact parameter window load group count exceeds scratch "
        "capacity");
  }
  if (load_count != 0) {
    memset(slot->buffers, 0, load_count * sizeof(slot->buffers[0]));
    memset(slot->alloca_semaphores, 0,
           load_count * sizeof(slot->alloca_semaphores[0]));
    memset(slot->alloca_payload_values, 0,
           load_count * sizeof(slot->alloca_payload_values[0]));
  }
  if (load_group_count != 0) {
    memset(slot->load_semaphores, 0,
           load_group_count * sizeof(slot->load_semaphores[0]));
    memset(slot->load_payload_values, 0,
           load_group_count * sizeof(slot->load_payload_values[0]));
  }
  if (load_count == 0) return iree_ok_status();

  iree_hal_semaphore_list_t alloca_wait_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_parameter_window_slot_alloca_wait_list(
          slot, base_wait_list, alloca_wait_semaphores,
          alloca_wait_payload_values, &alloca_wait_list));
  iree_status_t status =
      id4_pipeline_program_prepared_issue_parameter_window_allocas(
          schedule, alloca_wait_list, slot->buffers, slot->alloca_semaphores,
          slot->alloca_payload_values, &slot->alloca_submitted_count);
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_release(slot->cleanup_semaphore);
    slot->cleanup_semaphore = NULL;
    slot->cleanup_payload_value = 0;
  }
  iree_hal_semaphore_list_t alloca_completion_wait_list =
      id4_pipeline_program_parameter_window_slot_alloca_completion_wait_list(
          slot);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_prepared_submit_parameter_window_load_groups(
        prepared->plan, submit_region_id, schedule, parameter_issue_context,
        slot->buffers, alloca_completion_wait_list, slot->load_semaphores,
        slot->load_payload_values, &slot->load_submitted_count,
        diagnostics_sink);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t load_completion_wait_list =
        id4_pipeline_program_parameter_window_slot_load_completion_wait_list(
            slot);
    status =
        id4_pipeline_program_prepared_cleanup_and_release_failed_parameter_window(
            status, schedule, slot->alloca_submitted_count,
            alloca_completion_wait_list, slot->load_submitted_count,
            load_completion_wait_list, slot->buffers, slot->alloca_semaphores,
            slot->load_semaphores);
    slot->alloca_submitted_count = 0;
    slot->load_submitted_count = 0;
    slot->schedule = NULL;
    slot->region_index = IREE_HOST_SIZE_MAX;
  }
  return status;
}

static iree_status_t
id4_pipeline_program_prepared_submit_parameter_window_prefetch(
    id4_pipeline_program_prepared_t* prepared,
    iree_host_size_t current_region_id, iree_host_size_t* inout_next_region_id,
    iree_host_size_t prefetch_region_distance,
    id4_pipeline_parameter_slab_issue_context_t* parameter_issue_context,
    iree_hal_semaphore_list_t base_wait_list,
    iree_hal_semaphore_t** alloca_wait_semaphores,
    uint64_t* alloca_wait_payload_values, iree_host_size_t slot_count,
    id4_pipeline_program_parameter_window_slot_t* slots,
    iree_host_size_t max_load_count, iree_host_size_t max_load_group_count,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_ASSERT_ARGUMENT(prepared);
  IREE_ASSERT_ARGUMENT(inout_next_region_id);
  if (slot_count == 0) return iree_ok_status();
  const iree_host_size_t remaining_region_count =
      prepared->region_count - current_region_id - 1;
  const iree_host_size_t lookahead_region_count =
      iree_min(prefetch_region_distance, remaining_region_count);
  const iree_host_size_t last_region_id =
      current_region_id + lookahead_region_count;
  while (*inout_next_region_id <= last_region_id) {
    const iree_host_size_t window_region_id = *inout_next_region_id;
    id4_pipeline_program_parameter_window_slot_t* slot =
        &slots[window_region_id % slot_count];
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_prepared_submit_parameter_window_slot(
            prepared, current_region_id, window_region_id,
            parameter_issue_context, base_wait_list, alloca_wait_semaphores,
            alloca_wait_payload_values, slot, max_load_count,
            max_load_group_count, diagnostics_sink));
    ++*inout_next_region_id;
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_prepared_release_parameter_window_slot_after_issue(
    id4_pipeline_program_parameter_window_slot_t* slot,
    iree_host_size_t region_index,
    iree_hal_semaphore_list_t region_signal_list) {
  if (slot->region_index != region_index) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "program compact parameter window slot for region %" PRIhsz
        " is active for region %" PRIhsz,
        region_index, slot->region_index);
  }
  if (slot->alloca_submitted_count != 0) {
    iree_hal_semaphore_t* next_cleanup_semaphore = NULL;
    uint64_t next_cleanup_payload_value = 1;
    const id4_pipeline_parameter_slab_load_t* loads =
        id4_pipeline_parameter_window_schedule_loads(slot->schedule);
    iree_status_t status = iree_hal_semaphore_create(
        loads[0].device, loads[0].queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &next_cleanup_semaphore);
    if (iree_status_is_ok(status)) {
      iree_hal_semaphore_list_t cleanup_signal_list =
          id4_pipeline_program_one_semaphore_list(&next_cleanup_semaphore,
                                                  &next_cleanup_payload_value);
      status = id4_pipeline_program_prepared_dealloca_parameter_window(
          slot->schedule, region_signal_list, cleanup_signal_list,
          slot->alloca_submitted_count, slot->buffers);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_semaphore_release(slot->cleanup_semaphore);
      slot->cleanup_semaphore = next_cleanup_semaphore;
      slot->cleanup_payload_value = next_cleanup_payload_value;
      next_cleanup_semaphore = NULL;
    }
    iree_hal_semaphore_release(next_cleanup_semaphore);
    if (!iree_status_is_ok(status)) {
      id4_pipeline_program_parameter_window_slot_release_transient_refs(slot);
      return status;
    }
  }
  id4_pipeline_program_parameter_window_slot_release_transient_refs(slot);
  return iree_ok_status();
}

static iree_host_size_t id4_pipeline_program_parameter_window_cleanup_count(
    iree_host_size_t slot_count,
    const id4_pipeline_program_parameter_window_slot_t* slots) {
  iree_host_size_t cleanup_count = 0;
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    cleanup_count += slots[i].cleanup_semaphore ? 1 : 0;
  }
  return cleanup_count;
}

static void id4_pipeline_program_release_parameter_window_slots(
    iree_host_size_t slot_count,
    id4_pipeline_program_parameter_window_slot_t* slots) {
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    id4_pipeline_program_parameter_window_slot_release_transient_refs(
        &slots[i]);
    iree_hal_semaphore_release(slots[i].cleanup_semaphore);
    slots[i].cleanup_semaphore = NULL;
    slots[i].cleanup_payload_value = 0;
  }
}

static iree_status_t id4_pipeline_program_validate_final_signal_list(
    iree_hal_semaphore_list_t signal_list) {
  if (signal_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program issue final signal is required");
  }
  if (!signal_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program issue final signal semaphore array is "
                            "required");
  }
  if (!signal_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program issue final signal payload array is "
                            "required");
  }
  for (iree_host_size_t i = 0; i < signal_list.count; ++i) {
    if (!signal_list.semaphores[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program issue final signal semaphore %" PRIhsz " is NULL", i);
    }
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_program_prepared_issue(
    id4_pipeline_program_prepared_t* prepared, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  IREE_ASSERT_ARGUMENT(prepared);
  IREE_ASSERT_ARGUMENT(bundle);
  IREE_ASSERT_ARGUMENT(options);
  if (id4_pipeline_bundle_plan(bundle) != prepared->plan) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program issue bundle plan does not match prepared program plan");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_final_signal_list(
      options->signal_semaphore_list));
  if (prepared->region_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "program prepared regions are missing");
  }
  for (iree_host_size_t i = 0; i < prepared->region_count; ++i) {
    if (!prepared->prepared_regions[i]) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "program prepared region %" PRIhsz " is missing",
                              i);
    }
  }

  iree_host_size_t max_binding_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_max_binding_capacity(
      prepared, &max_binding_capacity));
  iree_hal_buffer_binding_t* bindings = NULL;
  if (max_binding_capacity != 0) {
    bindings = (iree_hal_buffer_binding_t*)iree_alloca(max_binding_capacity *
                                                       sizeof(bindings[0]));
  }

  iree_host_size_t max_parameter_window_load_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_prepared_max_parameter_window_load_count(
          prepared, &max_parameter_window_load_count));
  iree_host_size_t max_parameter_window_load_group_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_prepared_max_parameter_window_load_group_count(
          prepared, &max_parameter_window_load_group_count));

  iree_host_size_t parameter_window_slot_count = 0;
  if (prepared->uses_compact_parameter_windows) {
    iree_host_size_t requested_slot_count = 0;
    if (!iree_host_size_checked_add(
            options->parameter_load_prefetch_region_distance, 1,
            &requested_slot_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program compact parameter prefetch distance overflows");
    }
    parameter_window_slot_count =
        iree_min(prepared->region_count, requested_slot_count);
  }
  id4_pipeline_program_parameter_window_slot_t* parameter_window_slots = NULL;
  if (parameter_window_slot_count != 0) {
    parameter_window_slots =
        (id4_pipeline_program_parameter_window_slot_t*)iree_alloca(
            parameter_window_slot_count * sizeof(parameter_window_slots[0]));
  }
  iree_host_size_t total_parameter_window_load_count = 0;
  if (!iree_host_size_checked_mul(parameter_window_slot_count,
                                  max_parameter_window_load_count,
                                  &total_parameter_window_load_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program compact parameter window buffer scratch count overflows");
  }
  iree_host_size_t total_parameter_window_load_group_count = 0;
  if (!iree_host_size_checked_mul(parameter_window_slot_count,
                                  max_parameter_window_load_group_count,
                                  &total_parameter_window_load_group_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program compact parameter window load scratch count overflows");
  }
  iree_hal_buffer_t** parameter_window_buffers = NULL;
  iree_hal_semaphore_t** parameter_window_alloca_semaphores = NULL;
  uint64_t* parameter_window_alloca_payload_values = NULL;
  if (total_parameter_window_load_count != 0) {
    parameter_window_buffers =
        (iree_hal_buffer_t**)iree_alloca(total_parameter_window_load_count *
                                         sizeof(parameter_window_buffers[0]));
    parameter_window_alloca_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        total_parameter_window_load_count *
        sizeof(parameter_window_alloca_semaphores[0]));
    parameter_window_alloca_payload_values = (uint64_t*)iree_alloca(
        total_parameter_window_load_count *
        sizeof(parameter_window_alloca_payload_values[0]));
  }
  iree_hal_semaphore_t** parameter_window_load_semaphores = NULL;
  uint64_t* parameter_window_load_payload_values = NULL;
  if (total_parameter_window_load_group_count != 0) {
    parameter_window_load_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        total_parameter_window_load_group_count *
        sizeof(parameter_window_load_semaphores[0]));
    parameter_window_load_payload_values =
        (uint64_t*)iree_alloca(total_parameter_window_load_group_count *
                               sizeof(parameter_window_load_payload_values[0]));
  }
  for (iree_host_size_t i = 0; i < parameter_window_slot_count; ++i) {
    id4_pipeline_program_parameter_window_slot_initialize(
        &parameter_window_slots[i],
        max_parameter_window_load_count == 0
            ? NULL
            : parameter_window_buffers + i * max_parameter_window_load_count,
        max_parameter_window_load_count == 0
            ? NULL
            : parameter_window_alloca_semaphores +
                  i * max_parameter_window_load_count,
        max_parameter_window_load_count == 0
            ? NULL
            : parameter_window_alloca_payload_values +
                  i * max_parameter_window_load_count,
        max_parameter_window_load_group_count == 0
            ? NULL
            : parameter_window_load_semaphores +
                  i * max_parameter_window_load_group_count,
        max_parameter_window_load_group_count == 0
            ? NULL
            : parameter_window_load_payload_values +
                  i * max_parameter_window_load_group_count);
  }

  id4_pipeline_parameter_slab_set_t* parameter_slabs =
      id4_pipeline_bundle_parameter_slabs(bundle);
  iree_host_size_t planned_load_group_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_load_group_count(
      prepared->plan, &planned_load_group_count));
  const iree_host_size_t slab_set_load_group_count =
      id4_pipeline_parameter_slab_set_load_group_count(parameter_slabs);
  const bool uses_deferred_parameter_loads =
      id4_pipeline_parameter_slab_set_has_deferred_load_context(
          parameter_slabs);
  if (uses_deferred_parameter_loads &&
      planned_load_group_count != slab_set_load_group_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program issue parameter load group count mismatch: plan has %" PRIhsz
        " groups but slab set has %" PRIhsz,
        planned_load_group_count, slab_set_load_group_count);
  }
  const bool uses_region_parameter_waits =
      uses_deferred_parameter_loads && slab_set_load_group_count != 0;
  if (prepared->uses_compact_parameter_windows &&
      !uses_deferred_parameter_loads) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "compact parameter windows require deferred parameter loads");
  }
  id4_pipeline_program_issue_wait_flags_t initial_wait_flags = 0;
  if (!uses_region_parameter_waits) {
    initial_wait_flags |=
        ID4_PIPELINE_PROGRAM_ISSUE_WAIT_FLAG_INCLUDE_BUNDLE_READINESS;
  }

  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  const iree_host_size_t included_readiness_count =
      iree_all_bits_set(
          initial_wait_flags,
          ID4_PIPELINE_PROGRAM_ISSUE_WAIT_FLAG_INCLUDE_BUNDLE_READINESS)
          ? readiness_list.count
          : 0;
  iree_host_size_t initial_wait_count = 0;
  if (!iree_host_size_checked_add(included_readiness_count,
                                  options->wait_semaphore_list.count,
                                  &initial_wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue wait list count overflow");
  }
  iree_hal_semaphore_t** wait_semaphores = NULL;
  uint64_t* wait_payload_values = NULL;
  if (initial_wait_count != 0) {
    wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        initial_wait_count * sizeof(wait_semaphores[0]));
    wait_payload_values = (uint64_t*)iree_alloca(
        initial_wait_count * sizeof(wait_payload_values[0]));
  }
  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_make_initial_wait_list(
      bundle, options, initial_wait_flags, wait_semaphores, wait_payload_values,
      &wait_list));
  iree_hal_semaphore_t** parameter_window_alloca_wait_semaphores = NULL;
  uint64_t* parameter_window_alloca_wait_payload_values = NULL;
  if (parameter_window_slot_count != 0) {
    iree_host_size_t parameter_window_alloca_wait_capacity = 0;
    if (!iree_host_size_checked_add(initial_wait_count, 1,
                                    &parameter_window_alloca_wait_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program compact parameter window alloca wait scratch overflows");
    }
    if (parameter_window_alloca_wait_capacity != 0) {
      parameter_window_alloca_wait_semaphores =
          (iree_hal_semaphore_t**)iree_alloca(
              parameter_window_alloca_wait_capacity *
              sizeof(parameter_window_alloca_wait_semaphores[0]));
      parameter_window_alloca_wait_payload_values = (uint64_t*)iree_alloca(
          parameter_window_alloca_wait_capacity *
          sizeof(parameter_window_alloca_wait_payload_values[0]));
    }
  }

  const iree_host_size_t memory_slab_count =
      id4_pipeline_plan_memory_slab_count(prepared->plan);
  iree_hal_buffer_t** memory_slab_buffers = NULL;
  if (memory_slab_count != 0) {
    memory_slab_buffers = (iree_hal_buffer_t**)iree_alloca(
        memory_slab_count * sizeof(memory_slab_buffers[0]));
    memset(memory_slab_buffers, 0,
           memory_slab_count * sizeof(memory_slab_buffers[0]));
  }

  iree_host_size_t shared_slab_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_count_shared_slabs(
      prepared->plan, &shared_slab_count));
  iree_host_size_t* shared_slab_indices = NULL;
  iree_hal_semaphore_t** shared_alloca_semaphores = NULL;
  uint64_t* shared_alloca_payload_values = NULL;
  if (shared_slab_count != 0) {
    shared_slab_indices = (iree_host_size_t*)iree_alloca(
        shared_slab_count * sizeof(shared_slab_indices[0]));
    shared_alloca_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        shared_slab_count * sizeof(shared_alloca_semaphores[0]));
    memset(shared_alloca_semaphores, 0,
           shared_slab_count * sizeof(shared_alloca_semaphores[0]));
    shared_alloca_payload_values = (uint64_t*)iree_alloca(
        shared_slab_count * sizeof(shared_alloca_payload_values[0]));
    memset(shared_alloca_payload_values, 0,
           shared_slab_count * sizeof(shared_alloca_payload_values[0]));
    IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_collect_shared_slabs(
        prepared->plan, shared_slab_indices, &shared_slab_count));
  }

  const iree_host_size_t internal_signal_count = prepared->region_count - 1;
  iree_hal_semaphore_t** internal_semaphores = NULL;
  uint64_t* internal_payload_values = NULL;
  if (internal_signal_count != 0) {
    internal_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        internal_signal_count * sizeof(internal_semaphores[0]));
    memset(internal_semaphores, 0,
           internal_signal_count * sizeof(internal_semaphores[0]));
    internal_payload_values = (uint64_t*)iree_alloca(
        internal_signal_count * sizeof(internal_payload_values[0]));
  }

  iree_host_size_t max_region_parameter_load_group_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_prepared_max_region_parameter_load_group_count(
          prepared->plan, &max_region_parameter_load_group_count));
  iree_host_size_t max_region_base_wait_count = initial_wait_count;
  max_region_base_wait_count =
      iree_max(max_region_base_wait_count, shared_slab_count);
  if (internal_signal_count != 0) {
    max_region_base_wait_count = iree_max(max_region_base_wait_count, 1);
  }
  const iree_host_size_t max_region_parameter_wait_count =
      iree_max(max_region_parameter_load_group_count,
               max_parameter_window_load_group_count);
  iree_host_size_t max_region_wait_count = 0;
  if (!iree_host_size_checked_add(max_region_base_wait_count,
                                  max_region_parameter_wait_count,
                                  &max_region_wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue region wait list count overflow");
  }
  iree_hal_semaphore_t** region_wait_semaphores = NULL;
  uint64_t* region_wait_payload_values = NULL;
  if (max_region_wait_count != 0) {
    region_wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        max_region_wait_count * sizeof(region_wait_semaphores[0]));
    region_wait_payload_values = (uint64_t*)iree_alloca(
        max_region_wait_count * sizeof(region_wait_payload_values[0]));
  }

  iree_status_t status = iree_ok_status();
  iree_hal_semaphore_t* shared_completion_semaphore = NULL;
  uint64_t shared_completion_payload_value = 1;
  iree_hal_semaphore_t* shared_cleanup_semaphore = NULL;
  uint64_t shared_cleanup_payload_value = 1;
  if (shared_slab_count != 0) {
    const id4_pipeline_memory_slab_plan_t* first_shared_slab =
        id4_pipeline_plan_memory_slab_at(prepared->plan,
                                         shared_slab_indices[0]);
    iree_hal_device_t* shared_slab_device = NULL;
    iree_hal_queue_affinity_t shared_slab_queue_affinity =
        IREE_HAL_QUEUE_AFFINITY_ANY;
    status = id4_pipeline_program_prepared_shared_slab_device(
        prepared, first_shared_slab, &shared_slab_device,
        &shared_slab_queue_affinity);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_create(
          shared_slab_device, shared_slab_queue_affinity, /*initial_value=*/0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &shared_completion_semaphore);
    }
    if (iree_status_is_ok(status) && uses_deferred_parameter_loads) {
      status = iree_hal_semaphore_create(
          shared_slab_device, shared_slab_queue_affinity, /*initial_value=*/0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &shared_cleanup_semaphore);
    }
  }

  iree_hal_semaphore_t* execution_completion_semaphore = NULL;
  uint64_t execution_completion_payload_value = 1;
  if (uses_deferred_parameter_loads && shared_slab_count == 0) {
    const id4_pipeline_region_plan_t* last_region =
        id4_pipeline_plan_region_at(prepared->plan, prepared->region_count - 1);
    if (!last_region) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program final region is missing");
    }
    const id4_pipeline_device_placement_t* placement =
        id4_pipeline_plan_placement_at(prepared->plan,
                                       last_region->placement_id);
    if (!placement) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program final region placement is missing");
    }
    iree_hal_device_t* device = iree_hal_device_group_device_at(
        id4_pipeline_plan_device_group(prepared->plan),
        placement->device_index);
    if (!device) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program final region device is required");
    }
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        device, placement->queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &execution_completion_semaphore));
  }

  id4_pipeline_parameter_slab_issue_context_t* parameter_issue_context = NULL;
  if (iree_status_is_ok(status) && uses_deferred_parameter_loads) {
    status = id4_pipeline_plan_create_parameter_slab_issue_context(
        prepared->plan, parameter_slabs, prepared->host_allocator,
        &parameter_issue_context);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_prepared_create_internal_signals(
        prepared, internal_signal_count, internal_semaphores,
        internal_payload_values);
  }
  iree_host_size_t shared_alloca_submitted_count = 0;
  if (iree_status_is_ok(status) && shared_slab_count != 0) {
    status = id4_pipeline_program_prepared_issue_shared_allocas(
        prepared, options->diagnostics_sink, wait_list, shared_slab_count,
        shared_slab_indices, memory_slab_buffers, shared_alloca_semaphores,
        shared_alloca_payload_values, &shared_alloca_submitted_count);
  }

  iree_hal_semaphore_list_t first_region_wait_list = wait_list;
  if (iree_status_is_ok(status) && shared_slab_count != 0) {
    first_region_wait_list = (iree_hal_semaphore_list_t){
        // All shared slabs must be allocated before region execution.
        .count = shared_slab_count,
        // Per-slab alloca completion semaphores.
        .semaphores = shared_alloca_semaphores,
        // Per-slab alloca completion payloads.
        .payload_values = shared_alloca_payload_values,
    };
  }

  bool region_submitted = false;
  const bool wait_after_each_region = iree_all_bits_set(
      options->flags, ID4_PIPELINE_STAGE_ISSUE_FLAG_WAIT_AFTER_EACH_REGION);
  iree_hal_semaphore_t* cleanup_wait_semaphore = NULL;
  uint64_t cleanup_wait_payload_value = 0;
  iree_host_size_t next_parameter_window_region = 0;
  for (iree_host_size_t i = 0;
       i < prepared->region_count && iree_status_is_ok(status); ++i) {
    if (prepared->uses_compact_parameter_windows) {
      status = id4_pipeline_program_prepared_submit_parameter_window_prefetch(
          prepared, i, &next_parameter_window_region,
          options->parameter_load_prefetch_region_distance,
          parameter_issue_context, wait_list,
          parameter_window_alloca_wait_semaphores,
          parameter_window_alloca_wait_payload_values,
          parameter_window_slot_count, parameter_window_slots,
          max_parameter_window_load_count,
          max_parameter_window_load_group_count, options->diagnostics_sink);
      if (!iree_status_is_ok(status)) break;
    }

    iree_hal_semaphore_list_t structural_region_wait_list = wait_list;
    iree_hal_semaphore_t* internal_wait_semaphore = NULL;
    uint64_t internal_wait_payload_value = 0;
    if (i != 0) {
      internal_wait_semaphore = internal_semaphores[i - 1];
      internal_wait_payload_value = internal_payload_values[i - 1];
      structural_region_wait_list = id4_pipeline_program_one_semaphore_list(
          &internal_wait_semaphore, &internal_wait_payload_value);
    }
    iree_hal_semaphore_list_t base_region_wait_list =
        structural_region_wait_list;
    if (i == 0 && shared_slab_count != 0) {
      base_region_wait_list = first_region_wait_list;
    }
    const id4_pipeline_region_plan_t* region =
        id4_pipeline_plan_region_at(prepared->plan, i);
    if (!region) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program issue region plan %" PRIhsz " is missing", i);
      break;
    }
    id4_pipeline_program_parameter_window_slot_t* parameter_window_slot = NULL;
    const id4_pipeline_parameter_window_schedule_t* parameter_window_schedule =
        NULL;
    iree_hal_semaphore_list_t parameter_window_load_wait_list =
        iree_hal_semaphore_list_empty();
    if (prepared->uses_compact_parameter_windows) {
      parameter_window_slot =
          &parameter_window_slots[i % parameter_window_slot_count];
      if (parameter_window_slot->region_index != i) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "program compact parameter window for region %" PRIhsz
            " is not prefetched",
            i);
        break;
      }
      parameter_window_schedule = parameter_window_slot->schedule;
      parameter_window_load_wait_list =
          id4_pipeline_program_parameter_window_slot_load_completion_wait_list(
              parameter_window_slot);
    } else if (uses_deferred_parameter_loads) {
      status =
          id4_pipeline_program_prepared_submit_region_parameter_load_window(
              prepared->plan, i, prepared->region_count,
              options->parameter_load_prefetch_region_distance,
              parameter_issue_context, options->diagnostics_sink);
      if (!iree_status_is_ok(status)) break;
    }
    iree_hal_semaphore_list_t region_wait_list =
        iree_hal_semaphore_list_empty();
    status = id4_pipeline_program_prepared_make_region_wait_list(
        region, parameter_slabs, base_region_wait_list,
        parameter_window_load_wait_list,
        uses_region_parameter_waits
            ? ID4_PIPELINE_PROGRAM_ISSUE_WAIT_FLAG_INCLUDE_PARAMETER_READINESS
            : 0,
        region_wait_semaphores, region_wait_payload_values, &region_wait_list);
    if (!iree_status_is_ok(status)) break;

    iree_hal_semaphore_list_t region_signal_list =
        options->signal_semaphore_list;
    iree_hal_semaphore_t* internal_signal_semaphore = NULL;
    uint64_t internal_signal_payload_value = 0;
    if (i + 1 < prepared->region_count) {
      internal_signal_semaphore = internal_semaphores[i];
      internal_signal_payload_value = internal_payload_values[i];
      region_signal_list = id4_pipeline_program_one_semaphore_list(
          &internal_signal_semaphore, &internal_signal_payload_value);
    } else if (shared_slab_count != 0) {
      region_signal_list = id4_pipeline_program_one_semaphore_list(
          &shared_completion_semaphore, &shared_completion_payload_value);
    } else if (uses_deferred_parameter_loads) {
      region_signal_list = id4_pipeline_program_one_semaphore_list(
          &execution_completion_semaphore, &execution_completion_payload_value);
    }

    iree_hal_buffer_binding_table_t binding_table =
        iree_hal_buffer_binding_table_empty();
    status = id4_pipeline_program_prepared_make_binding_table(
        prepared, bundle, options, i, parameter_window_schedule,
        parameter_window_slot ? parameter_window_slot->buffers : NULL,
        memory_slab_buffers, bindings, &binding_table);
    if (!iree_status_is_ok(status)) break;

    id4_pipeline_prepared_region_issue_options_t issue_options;
    memset(&issue_options, 0, sizeof(issue_options));
    issue_options.structure_size = sizeof(issue_options);
    issue_options.wait_semaphore_list = region_wait_list;
    issue_options.signal_semaphore_list = region_signal_list;
    issue_options.binding_table = binding_table;
    issue_options.execute_flags = IREE_HAL_EXECUTE_FLAG_NONE;
    status = id4_pipeline_program_emit_lifecycle(
        options->diagnostics_sink, id4_pipeline_plan_stage_name(prepared->plan),
        IREE_SV("program.region.issue.begin"), region->name);
    if (!iree_status_is_ok(status)) break;
    status = id4_pipeline_prepared_region_issue(prepared->prepared_regions[i],
                                                &issue_options);
    if (!iree_status_is_ok(status)) break;
    status = id4_pipeline_program_emit_lifecycle(
        options->diagnostics_sink, id4_pipeline_plan_stage_name(prepared->plan),
        IREE_SV("program.region.issue.submitted"), region->name);
    if (!iree_status_is_ok(status)) break;
    if (iree_status_is_ok(status)) {
      region_submitted = true;
      if (i + 1 < prepared->region_count) {
        cleanup_wait_semaphore = internal_semaphores[i];
        cleanup_wait_payload_value = internal_payload_values[i];
      } else if (shared_slab_count != 0) {
        cleanup_wait_semaphore = shared_completion_semaphore;
        cleanup_wait_payload_value = shared_completion_payload_value;
      } else if (uses_deferred_parameter_loads) {
        cleanup_wait_semaphore = execution_completion_semaphore;
        cleanup_wait_payload_value = execution_completion_payload_value;
      }
      if (wait_after_each_region) {
        status = id4_pipeline_program_wait_after_region_issue(
            options->diagnostics_sink,
            id4_pipeline_plan_stage_name(prepared->plan), region->name,
            region_signal_list);
        if (!iree_status_is_ok(status)) break;
      } else if (i + 1 >= options->region_submission_window) {
        const iree_host_size_t wait_region_index =
            i + 1 - options->region_submission_window;
        if (wait_region_index + 1 < prepared->region_count) {
          status = iree_hal_semaphore_wait(
              internal_semaphores[wait_region_index],
              internal_payload_values[wait_region_index],
              iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
          if (!iree_status_is_ok(status)) break;
        }
      }
      if (parameter_window_slot) {
        status =
            id4_pipeline_program_prepared_release_parameter_window_slot_after_issue(
                parameter_window_slot, i, region_signal_list);
        if (!iree_status_is_ok(status)) break;
      }
    }
  }
  if (!iree_status_is_ok(status) && parameter_window_slot_count != 0) {
    status =
        id4_pipeline_program_prepared_cleanup_active_parameter_window_slots(
            status, parameter_window_slot_count, parameter_window_slots);
  }

  iree_hal_semaphore_list_t parameter_cleanup_wait_list =
      iree_hal_semaphore_list_empty();
  bool parameter_issue_context_finish_attempted = false;
  bool shared_dealloca_attempted = false;
  if (iree_status_is_ok(status) && shared_slab_count != 0) {
    iree_hal_semaphore_list_t shared_completion_wait_list =
        id4_pipeline_program_one_semaphore_list(
            &shared_completion_semaphore, &shared_completion_payload_value);
    iree_hal_semaphore_list_t shared_dealloca_signal_list =
        options->signal_semaphore_list;
    if (uses_deferred_parameter_loads) {
      shared_dealloca_signal_list = id4_pipeline_program_one_semaphore_list(
          &shared_cleanup_semaphore, &shared_cleanup_payload_value);
    }
    shared_dealloca_attempted = true;
    status = id4_pipeline_program_prepared_dealloca_shared_slabs(
        prepared, shared_completion_wait_list, shared_dealloca_signal_list,
        shared_slab_count, shared_slab_indices, memory_slab_buffers);
  }
  if (iree_status_is_ok(status) && uses_deferred_parameter_loads) {
    parameter_issue_context_finish_attempted = true;
    status = id4_pipeline_parameter_slab_issue_context_finish(
        parameter_issue_context, &parameter_cleanup_wait_list);
  }
  if (iree_status_is_ok(status) && uses_deferred_parameter_loads) {
    iree_hal_device_t* final_device = NULL;
    iree_hal_queue_affinity_t final_queue_affinity =
        IREE_HAL_QUEUE_AFFINITY_ANY;
    iree_hal_semaphore_t* execution_wait_semaphore = NULL;
    uint64_t execution_wait_payload_value = 0;
    if (shared_slab_count != 0) {
      const id4_pipeline_memory_slab_plan_t* first_shared_slab =
          id4_pipeline_plan_memory_slab_at(prepared->plan,
                                           shared_slab_indices[0]);
      status = id4_pipeline_program_prepared_shared_slab_device(
          prepared, first_shared_slab, &final_device, &final_queue_affinity);
      execution_wait_semaphore = shared_cleanup_semaphore;
      execution_wait_payload_value = shared_cleanup_payload_value;
    } else {
      const id4_pipeline_region_plan_t* last_region =
          id4_pipeline_plan_region_at(prepared->plan,
                                      prepared->region_count - 1);
      if (!last_region) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "program final region is missing");
      } else {
        const id4_pipeline_device_placement_t* placement =
            id4_pipeline_plan_placement_at(prepared->plan,
                                           last_region->placement_id);
        if (!placement) {
          status =
              iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                               "program final region placement is missing");
        } else {
          final_device = iree_hal_device_group_device_at(
              id4_pipeline_plan_device_group(prepared->plan),
              placement->device_index);
          final_queue_affinity = placement->queue_affinity;
          execution_wait_semaphore = execution_completion_semaphore;
          execution_wait_payload_value = execution_completion_payload_value;
        }
      }
    }
    if (iree_status_is_ok(status) && !final_device) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "program final signal device is required");
    }
    if (iree_status_is_ok(status)) {
      iree_host_size_t final_wait_count = 0;
      const iree_host_size_t parameter_window_cleanup_count =
          id4_pipeline_program_parameter_window_cleanup_count(
              parameter_window_slot_count, parameter_window_slots);
      iree_host_size_t parameter_wait_count = 0;
      if (!iree_host_size_checked_add(parameter_cleanup_wait_list.count,
                                      parameter_window_cleanup_count,
                                      &parameter_wait_count) ||
          !iree_host_size_checked_add(1, parameter_wait_count,
                                      &final_wait_count)) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "program final wait list count overflows");
      } else {
        iree_hal_semaphore_t** final_wait_semaphores =
            (iree_hal_semaphore_t**)iree_alloca(
                final_wait_count * sizeof(final_wait_semaphores[0]));
        uint64_t* final_wait_payload_values = (uint64_t*)iree_alloca(
            final_wait_count * sizeof(final_wait_payload_values[0]));
        final_wait_semaphores[0] = execution_wait_semaphore;
        final_wait_payload_values[0] = execution_wait_payload_value;
        iree_host_size_t final_wait_offset = 1;
        for (iree_host_size_t i = 0; i < parameter_window_slot_count; ++i) {
          if (!parameter_window_slots[i].cleanup_semaphore) continue;
          final_wait_semaphores[final_wait_offset] =
              parameter_window_slots[i].cleanup_semaphore;
          final_wait_payload_values[final_wait_offset] =
              parameter_window_slots[i].cleanup_payload_value;
          ++final_wait_offset;
        }
        const iree_host_size_t parameter_cleanup_offset =
            1 + parameter_window_cleanup_count;
        for (iree_host_size_t i = 0; i < parameter_cleanup_wait_list.count;
             ++i) {
          final_wait_semaphores[parameter_cleanup_offset + i] =
              parameter_cleanup_wait_list.semaphores[i];
          final_wait_payload_values[parameter_cleanup_offset + i] =
              parameter_cleanup_wait_list.payload_values[i];
        }
        iree_hal_semaphore_list_t final_wait_list = {
            // Completion and issue-local cleanup edges.
            .count = final_wait_count,
            // Semaphores that must complete before the caller signal.
            .semaphores = final_wait_semaphores,
            // Payload values paired with final_wait_semaphores.
            .payload_values = final_wait_payload_values,
        };
        status = iree_hal_device_queue_barrier(
            final_device, final_queue_affinity, final_wait_list,
            options->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
      }
    }
  } else if (!iree_status_is_ok(status) && shared_alloca_submitted_count != 0 &&
             !shared_dealloca_attempted) {
    iree_hal_semaphore_list_t cleanup_wait_list =
        iree_hal_semaphore_list_empty();
    if (region_submitted && cleanup_wait_semaphore) {
      cleanup_wait_list = id4_pipeline_program_one_semaphore_list(
          &cleanup_wait_semaphore, &cleanup_wait_payload_value);
    } else {
      cleanup_wait_list = (iree_hal_semaphore_list_t){
          // Shared alloca submissions that must complete before cleanup.
          .count = shared_alloca_submitted_count,
          // Per-slab alloca completion semaphores.
          .semaphores = shared_alloca_semaphores,
          // Per-slab alloca completion payloads.
          .payload_values = shared_alloca_payload_values,
      };
    }
    iree_status_t cleanup_status =
        id4_pipeline_program_prepared_dealloca_shared_slabs(
            prepared, cleanup_wait_list, iree_hal_semaphore_list_empty(),
            shared_alloca_submitted_count, shared_slab_indices,
            memory_slab_buffers);
    status = iree_status_join(status, cleanup_status);
  }
  if (!iree_status_is_ok(status) && parameter_issue_context &&
      !parameter_issue_context_finish_attempted) {
    iree_hal_semaphore_list_t cleanup_wait_list =
        iree_hal_semaphore_list_empty();
    iree_status_t cleanup_status =
        id4_pipeline_parameter_slab_issue_context_finish(
            parameter_issue_context, &cleanup_wait_list);
    status = iree_status_join(status, cleanup_status);
  }

  for (iree_host_size_t i = 0; i < memory_slab_count; ++i) {
    iree_hal_buffer_release(memory_slab_buffers[i]);
  }
  for (iree_host_size_t i = 0; i < shared_slab_count; ++i) {
    iree_hal_semaphore_release(shared_alloca_semaphores[i]);
  }
  iree_hal_semaphore_release(shared_completion_semaphore);
  iree_hal_semaphore_release(shared_cleanup_semaphore);
  iree_hal_semaphore_release(execution_completion_semaphore);
  id4_pipeline_program_release_parameter_window_slots(
      parameter_window_slot_count, parameter_window_slots);
  id4_pipeline_parameter_slab_issue_context_release(parameter_issue_context);
  id4_pipeline_program_release_semaphores(internal_signal_count,
                                          internal_semaphores);
  return status;
}
