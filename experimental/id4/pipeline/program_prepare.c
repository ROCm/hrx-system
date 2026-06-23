// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_prepare.h"

#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/program_region.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/buffer_transfer.h"

typedef struct id4_pipeline_program_prepared_kernel_t {
  // Prepared kernel executable retained by the prepared program.
  id4_pipeline_kernel_executable_t* executable;
  // HAL executable function selected for the planned export name.
  iree_hal_executable_function_t function;
} id4_pipeline_program_prepared_kernel_t;

struct id4_pipeline_program_prepared_t {
  // Reference count for shared prepared program ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for prepared program storage.
  iree_allocator_t host_allocator;
  // Plan retained for kernel, region, and binding metadata.
  id4_pipeline_plan_t* plan;
  // Sealed prepared region recorded from the semantic program.
  id4_pipeline_prepared_region_t* prepared_region;
  // Number of prepared kernel entries.
  iree_host_size_t kernel_count;
  // Prepared kernel entries in plan kernel order.
  id4_pipeline_program_prepared_kernel_t* kernels;
  // Number of materialized constant slab buffers.
  iree_host_size_t constant_slab_count;
  // Device buffers containing program-owned constants in plan slab order.
  iree_hal_buffer_t** constant_slab_buffers;
};

typedef struct id4_pipeline_program_prepare_context_t {
  // Prepare options borrowed for recording.
  const id4_pipeline_program_prepare_options_t* options;
  // Prepared object receiving executables and the sealed region.
  id4_pipeline_program_prepared_t* prepared;
} id4_pipeline_program_prepare_context_t;

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
  if (id4_pipeline_plan_region_count(options->plan) != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program prepare requires exactly one region");
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
    prepared->kernel_count = kernel_count;
    prepared->constant_slab_count = constant_slab_count;
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

static iree_status_t id4_pipeline_program_prepare_resolve_parameter(
    void* user_data, const id4_pipeline_program_parameter_op_t* parameter_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t parameter_ordinal,
    id4_pipeline_tensor_import_t* out_import) {
  (void)parameter_op;
  id4_pipeline_program_prepare_context_t* context =
      (id4_pipeline_program_prepare_context_t*)user_data;
  iree_host_size_t remaining_ordinal = parameter_ordinal;
  for (iree_host_size_t slab_index = 0;
       slab_index <
       id4_pipeline_plan_parameter_slab_count(context->options->plan);
       ++slab_index) {
    const id4_pipeline_parameter_slab_plan_t* slab =
        id4_pipeline_plan_parameter_slab_at(context->options->plan, slab_index);
    if (!slab) continue;
    if (remaining_ordinal >= slab->request_count) {
      remaining_ordinal -= slab->request_count;
      continue;
    }
    const id4_pipeline_parameter_request_t* request =
        &slab->requests[remaining_ordinal];
    out_import->layout = (id4_pipeline_tensor_layout_t){
        // Parameter tensor diagnostic name.
        .name = tensor->name,
        // Parameter tensor element type.
        .dtype = id4_pipeline_program_region_convert_dtype(tensor->dtype),
        // Parameter tensor shape.
        .shape = id4_pipeline_program_prepare_convert_shape(tensor->shape),
        // Dense parameter tensor byte length.
        .byte_length = tensor->byte_length,
        // Parameter subrange alignment is already represented by the plan
        // span offset.
        .alignment = 0,
    };
    out_import->binding_slot = slab->binding_slot;
    out_import->offset = request->span.buffer_offset;
    out_import->flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "program parameter tensor %" PRIhsz " is missing",
                          parameter_ordinal);
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
    const id4_pipeline_program_prepare_options_t* options) {
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(options->plan, 0);
  if (!region) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program region plan is missing");
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
    if (slab && slab->binding_slot == region->local_binding_slot) {
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
  };
  if (iree_status_is_ok(status)) {
    id4_pipeline_program_region_lower_options_t lower_options;
    memset(&lower_options, 0, sizeof(lower_options));
    lower_options.structure_size = sizeof(lower_options);
    lower_options.program = options->program;
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
    status = id4_pipeline_prepared_region_create(builder, &create_options,
                                                 prepared->host_allocator,
                                                 &prepared->prepared_region);
  }

  id4_pipeline_region_builder_destroy(builder);
  iree_hal_command_buffer_release(command_buffer);
  iree_allocator_free(prepared->host_allocator, planned_tap_names);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
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
    status = id4_pipeline_program_prepare_record_region(prepared, options);
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
  id4_pipeline_prepared_region_release(prepared->prepared_region);
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

static iree_status_t id4_pipeline_program_prepared_make_wait_list(
    id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options,
    iree_hal_semaphore_t** semaphores, uint64_t* payload_values,
    iree_hal_semaphore_list_t* out_wait_list) {
  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(readiness_list.count,
                                  options->wait_semaphore_list.count,
                                  &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue wait list count overflow");
  }
  for (iree_host_size_t i = 0; i < readiness_list.count; ++i) {
    semaphores[i] = readiness_list.semaphores[i];
    payload_values[i] = readiness_list.payload_values[i];
  }
  for (iree_host_size_t i = 0; i < options->wait_semaphore_list.count; ++i) {
    const iree_host_size_t target_index = readiness_list.count + i;
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

static iree_status_t id4_pipeline_program_prepared_make_binding_table(
    id4_pipeline_program_prepared_t* prepared, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options,
    iree_hal_buffer_binding_t* bindings,
    iree_hal_buffer_binding_table_t* out_binding_table) {
  const id4_pipeline_plan_t* plan = id4_pipeline_bundle_plan(bundle);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  if (!region) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue region plan is missing");
  }
  memset(bindings, 0, region->binding_capacity * sizeof(bindings[0]));

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
    bindings[tap->binding_slot] = options->diagnostic_tap_bindings[i];
  }

  *out_binding_table = (iree_hal_buffer_binding_table_t){
      // Exact issue-time binding count expected by the prepared region.
      .count = region->binding_capacity,
      // Stack-local full binding table for this issue call.
      .bindings = bindings,
  };
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
  if (!prepared->prepared_region) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "program prepared region is missing");
  }

  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(prepared->plan, 0);
  if (!region) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue region plan is missing");
  }
  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(readiness_list.count,
                                  options->wait_semaphore_list.count,
                                  &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue wait list count overflow");
  }
  iree_hal_semaphore_t** wait_semaphores = NULL;
  uint64_t* wait_payload_values = NULL;
  if (wait_count != 0) {
    wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        wait_count * sizeof(wait_semaphores[0]));
    wait_payload_values =
        (uint64_t*)iree_alloca(wait_count * sizeof(wait_payload_values[0]));
  }
  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_make_wait_list(
      bundle, options, wait_semaphores, wait_payload_values, &wait_list));

  iree_hal_buffer_binding_t* bindings = NULL;
  if (region->binding_capacity != 0) {
    bindings = (iree_hal_buffer_binding_t*)iree_alloca(
        region->binding_capacity * sizeof(bindings[0]));
  }
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_make_binding_table(
      prepared, bundle, options, bindings, &binding_table));

  id4_pipeline_prepared_region_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = wait_list;
  issue_options.signal_semaphore_list = options->signal_semaphore_list;
  issue_options.binding_table = binding_table;
  issue_options.execute_flags = IREE_HAL_EXECUTE_FLAG_NONE;
  return id4_pipeline_prepared_region_issue(prepared->prepared_region,
                                            &issue_options);
}
