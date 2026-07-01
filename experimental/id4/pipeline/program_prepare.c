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
    iree_host_size_t region_index,
    iree_hal_buffer_t* const* memory_slab_buffers,
    iree_hal_buffer_binding_t* bindings,
    iree_hal_buffer_binding_table_t* out_binding_table) {
  if (region_index > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program issue region index %" PRIhsz
                            " exceeds uint32_t",
                            region_index);
  }
  const uint32_t region_id = (uint32_t)region_index;
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
    if (!tap || tap->region_id != region_id) continue;
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

static void id4_pipeline_program_release_internal_signals(
    iree_host_size_t internal_signal_count, iree_hal_semaphore_t** semaphores) {
  for (iree_host_size_t i = 0; i < internal_signal_count; ++i) {
    iree_hal_semaphore_release(semaphores[i]);
  }
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
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        device, queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &alloca_semaphores[i]));
    iree_hal_semaphore_list_t signal_list =
        id4_pipeline_program_one_semaphore_list(&alloca_semaphores[i],
                                                &alloca_payload_values[i]);
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_alloca(
        device, queue_affinity, wait_list, signal_list, /*pool=*/NULL,
        slab->params, slab->byte_length, prepared->shared_slab_alloca_flags,
        &memory_slab_buffers[memory_slab_index]));
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

  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(bundle);
  iree_host_size_t initial_wait_count = 0;
  if (!iree_host_size_checked_add(readiness_list.count,
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
  IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_make_wait_list(
      bundle, options, wait_semaphores, wait_payload_values, &wait_list));

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

  iree_hal_semaphore_t* shared_completion_semaphore = NULL;
  uint64_t shared_completion_payload_value = 1;
  if (shared_slab_count != 0) {
    const id4_pipeline_memory_slab_plan_t* first_shared_slab =
        id4_pipeline_plan_memory_slab_at(prepared->plan,
                                         shared_slab_indices[0]);
    iree_hal_device_t* shared_slab_device = NULL;
    iree_hal_queue_affinity_t shared_slab_queue_affinity =
        IREE_HAL_QUEUE_AFFINITY_ANY;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_prepared_shared_slab_device(
        prepared, first_shared_slab, &shared_slab_device,
        &shared_slab_queue_affinity));
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        shared_slab_device, shared_slab_queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &shared_completion_semaphore));
  }

  iree_status_t status = id4_pipeline_program_prepared_create_internal_signals(
      prepared, internal_signal_count, internal_semaphores,
      internal_payload_values);
  iree_host_size_t shared_alloca_submitted_count = 0;
  if (iree_status_is_ok(status) && shared_slab_count != 0) {
    status = id4_pipeline_program_prepared_issue_shared_allocas(
        prepared, wait_list, shared_slab_count, shared_slab_indices,
        memory_slab_buffers, shared_alloca_semaphores,
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
  iree_hal_semaphore_t* cleanup_wait_semaphore = NULL;
  uint64_t cleanup_wait_payload_value = 0;
  for (iree_host_size_t i = 0;
       i < prepared->region_count && iree_status_is_ok(status); ++i) {
    iree_hal_semaphore_list_t region_wait_list = first_region_wait_list;
    iree_hal_semaphore_t* internal_wait_semaphore = NULL;
    uint64_t internal_wait_payload_value = 0;
    if (i != 0) {
      internal_wait_semaphore = internal_semaphores[i - 1];
      internal_wait_payload_value = internal_payload_values[i - 1];
      region_wait_list = id4_pipeline_program_one_semaphore_list(
          &internal_wait_semaphore, &internal_wait_payload_value);
    }

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
    }

    iree_hal_buffer_binding_table_t binding_table =
        iree_hal_buffer_binding_table_empty();
    status = id4_pipeline_program_prepared_make_binding_table(
        prepared, bundle, options, i, memory_slab_buffers, bindings,
        &binding_table);
    if (!iree_status_is_ok(status)) break;

    id4_pipeline_prepared_region_issue_options_t issue_options;
    memset(&issue_options, 0, sizeof(issue_options));
    issue_options.structure_size = sizeof(issue_options);
    issue_options.wait_semaphore_list = region_wait_list;
    issue_options.signal_semaphore_list = region_signal_list;
    issue_options.binding_table = binding_table;
    issue_options.execute_flags = IREE_HAL_EXECUTE_FLAG_NONE;
    status = id4_pipeline_prepared_region_issue(prepared->prepared_regions[i],
                                                &issue_options);
    if (iree_status_is_ok(status)) {
      region_submitted = true;
      if (i + 1 < prepared->region_count) {
        cleanup_wait_semaphore = internal_semaphores[i];
        cleanup_wait_payload_value = internal_payload_values[i];
      } else if (shared_slab_count != 0) {
        cleanup_wait_semaphore = shared_completion_semaphore;
        cleanup_wait_payload_value = shared_completion_payload_value;
      }
    }
  }

  if (iree_status_is_ok(status) && shared_slab_count != 0) {
    iree_hal_semaphore_list_t shared_completion_wait_list =
        id4_pipeline_program_one_semaphore_list(
            &shared_completion_semaphore, &shared_completion_payload_value);
    status = id4_pipeline_program_prepared_dealloca_shared_slabs(
        prepared, shared_completion_wait_list, options->signal_semaphore_list,
        shared_slab_count, shared_slab_indices, memory_slab_buffers);
  } else if (!iree_status_is_ok(status) && shared_alloca_submitted_count != 0) {
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

  for (iree_host_size_t i = 0; i < memory_slab_count; ++i) {
    iree_hal_buffer_release(memory_slab_buffers[i]);
  }
  for (iree_host_size_t i = 0; i < shared_slab_count; ++i) {
    iree_hal_semaphore_release(shared_alloca_semaphores[i]);
  }
  iree_hal_semaphore_release(shared_completion_semaphore);
  id4_pipeline_program_release_internal_signals(internal_signal_count,
                                                internal_semaphores);
  return status;
}
