// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_plan.h"

#include <string.h>

typedef struct id4_pipeline_program_plan_counts_t {
  // Number of parameter operations in the program.
  iree_host_size_t parameter_count;
  // Number of Loom dispatch operations in the program.
  iree_host_size_t dispatch_count;
  // Number of barrier operations in the program.
  iree_host_size_t barrier_count;
  // Number of executable region operations in the program.
  iree_host_size_t region_operation_count;
  // Number of diagnostic tap operations in the program.
  iree_host_size_t tap_count;
} id4_pipeline_program_plan_counts_t;

static iree_status_t id4_pipeline_program_plan_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_program_plan_validate_options(
    const id4_pipeline_program_plan_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program plan options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_plan_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program plan")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program plan extension structures are not supported");
  }
  if (!options->program) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program is required");
  }
  if (!options->device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program plan device group is required");
  }
  if (options->placement_count == 0 || !options->placements) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program plan placements are required");
  }
  if (options->parameter_slab_alignment != 0 &&
      !iree_device_size_is_power_of_two(options->parameter_slab_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program plan parameter slab alignment must be a power of two");
  }
  if (options->parameter_request_alignment != 0 &&
      !iree_device_size_is_power_of_two(options->parameter_request_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program plan parameter request alignment must be a power of two");
  }
  return iree_ok_status();
}

static id4_pipeline_program_plan_counts_t id4_pipeline_program_plan_count_ops(
    const id4_pipeline_program_t* program) {
  id4_pipeline_program_plan_counts_t counts;
  memset(&counts, 0, sizeof(counts));
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (!op) continue;
    switch (op->kind) {
      case ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER:
        ++counts.parameter_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
        ++counts.dispatch_count;
        ++counts.region_operation_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
        ++counts.barrier_count;
        ++counts.region_operation_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
        ++counts.tap_count;
        break;
      default:
        break;
    }
  }
  return counts;
}

static bool id4_pipeline_program_plan_kernel_config_equal(
    iree_host_size_t lhs_count,
    const id4_pipeline_kernel_config_binding_t* lhs_values,
    iree_host_size_t rhs_count,
    const id4_pipeline_kernel_config_binding_t* rhs_values) {
  if (lhs_count != rhs_count) return false;
  for (iree_host_size_t i = 0; i < lhs_count; ++i) {
    if (!iree_string_view_equal(lhs_values[i].key, rhs_values[i].key)) {
      return false;
    }
    if (!iree_string_view_equal(lhs_values[i].value, rhs_values[i].value)) {
      return false;
    }
  }
  return true;
}

static bool id4_pipeline_program_plan_kernel_equal(
    const id4_pipeline_kernel_plan_t* lhs,
    const id4_pipeline_program_dispatch_loom_op_t* rhs) {
  return iree_string_view_equal(lhs->module_path, rhs->kernel.module_path) &&
         iree_string_view_equal(lhs->function_name,
                                rhs->kernel.function_name) &&
         id4_pipeline_program_plan_kernel_config_equal(
             lhs->config_binding_count, lhs->config_bindings,
             rhs->config_binding_count, rhs->config_bindings);
}

static iree_host_size_t id4_pipeline_program_plan_find_kernel(
    iree_host_size_t kernel_count, const id4_pipeline_kernel_plan_t* kernels,
    const id4_pipeline_program_dispatch_loom_op_t* dispatch) {
  for (iree_host_size_t i = 0; i < kernel_count; ++i) {
    if (id4_pipeline_program_plan_kernel_equal(&kernels[i], dispatch)) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t id4_pipeline_program_plan_make_specialization_key(
    const id4_pipeline_program_dispatch_loom_op_t* dispatch,
    iree_allocator_t host_allocator, iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_string_builder_append_string(&builder, dispatch->kernel.module_path);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "::");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder,
                                               dispatch->kernel.function_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "[");
  }
  for (iree_host_size_t i = 0;
       i < dispatch->config_binding_count && iree_status_is_ok(status); ++i) {
    if (i != 0) {
      status = iree_string_builder_append_cstring(&builder, ",");
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(
          &builder, dispatch->config_bindings[i].key);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(&builder, "=");
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(
          &builder, dispatch->config_bindings[i].value);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "]");
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t key_size = iree_string_builder_size(&builder);
    char* storage = iree_string_builder_take_storage(&builder);
    *out_key = iree_make_string_view(storage, key_size);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static void id4_pipeline_program_plan_release_specialization_keys(
    iree_host_size_t kernel_count, id4_pipeline_kernel_plan_t* kernels,
    iree_allocator_t host_allocator) {
  if (!kernels) return;
  for (iree_host_size_t i = 0; i < kernel_count; ++i) {
    iree_allocator_free(host_allocator,
                        (void*)kernels[i].specialization_key.data);
    kernels[i].specialization_key = iree_string_view_empty();
  }
}

static iree_status_t id4_pipeline_program_plan_build_parameter_requests(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts,
    id4_pipeline_parameter_request_t* requests,
    iree_device_size_t* out_parameter_slab_byte_length) {
  *out_parameter_slab_byte_length = 0;
  if (counts.parameter_count == 0) return iree_ok_status();

  iree_host_size_t request_index = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) continue;
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(options->program,
                                       op->payload.parameter.tensor.ordinal);
    if (!tensor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program parameter tensor %u is missing",
                              op->payload.parameter.tensor.ordinal);
    }
    iree_io_parameter_span_t span;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_pack_span(
        tensor->byte_length, options->parameter_request_alignment,
        out_parameter_slab_byte_length, &span));
    requests[request_index] =
        id4_pipeline_parameter_request(tensor->name, span);
    ++request_index;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_build_kernels(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts,
    id4_pipeline_kernel_plan_t* kernels, iree_allocator_t host_allocator,
    iree_host_size_t* out_kernel_count) {
  *out_kernel_count = 0;
  if (counts.dispatch_count == 0) return iree_ok_status();

  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      continue;
    }
    const id4_pipeline_program_dispatch_loom_op_t* dispatch =
        &op->payload.dispatch_loom;
    if (id4_pipeline_program_plan_find_kernel(*out_kernel_count, kernels,
                                              dispatch) != IREE_HOST_SIZE_MAX) {
      continue;
    }
    id4_pipeline_kernel_plan_t* kernel = &kernels[*out_kernel_count];
    memset(kernel, 0, sizeof(*kernel));
    IREE_RETURN_IF_ERROR(id4_pipeline_program_plan_make_specialization_key(
        dispatch, host_allocator, &kernel->specialization_key));
    kernel->module_path = dispatch->kernel.module_path;
    kernel->function_name = dispatch->kernel.function_name;
    kernel->placement_id = options->kernel_placement_id;
    kernel->config_binding_count = dispatch->config_binding_count;
    kernel->config_bindings = dispatch->config_bindings;
    ++*out_kernel_count;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_build_taps(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts,
    id4_pipeline_diagnostic_tap_plan_t* taps) {
  if (counts.tap_count == 0) return iree_ok_status();

  iree_host_size_t tap_index = 0;
  iree_host_size_t region_operation_count = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op) continue;
    switch (op->kind) {
      case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
      case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
        ++region_operation_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_TAP: {
        if (region_operation_count == 0) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "program tap %.*s has no preceding "
                                  "executable region operation",
                                  (int)op->payload.tap.name.size,
                                  op->payload.tap.name.data);
        }
        const id4_pipeline_program_tensor_record_t* tensor =
            id4_pipeline_program_tensor_at(options->program,
                                           op->payload.tap.tensor.ordinal);
        if (!tensor) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "program tap tensor %u is missing",
                                  op->payload.tap.tensor.ordinal);
        }
        taps[tap_index].name = op->payload.tap.name;
        taps[tap_index].region_id = 0;
        taps[tap_index].after_operation_ordinal = region_operation_count - 1;
        taps[tap_index].target_name = tensor->name;
        ++tap_index;
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_program_create_plan(
    const id4_pipeline_program_plan_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_plan_validate_options(options));

  id4_pipeline_program_plan_counts_t counts =
      id4_pipeline_program_plan_count_ops(options->program);

  id4_pipeline_parameter_request_t* parameter_requests = NULL;
  id4_pipeline_kernel_plan_t* kernels = NULL;
  id4_pipeline_diagnostic_tap_plan_t* taps = NULL;
  iree_status_t status = iree_ok_status();
  if (counts.parameter_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.parameter_count,
                                         sizeof(parameter_requests[0]),
                                         (void**)&parameter_requests);
  }
  if (iree_status_is_ok(status) && counts.dispatch_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.dispatch_count,
                                         sizeof(kernels[0]), (void**)&kernels);
  }
  if (iree_status_is_ok(status) && counts.tap_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.tap_count,
                                         sizeof(taps[0]), (void**)&taps);
  }

  iree_device_size_t parameter_slab_byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_plan_build_parameter_requests(
        options, counts, parameter_requests, &parameter_slab_byte_length);
  }

  iree_host_size_t kernel_count = 0;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_plan_build_kernels(
        options, counts, kernels, host_allocator, &kernel_count);
  }

  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_plan_build_taps(options, counts, taps);
  }

  id4_pipeline_parameter_slab_plan_t parameter_slab;
  memset(&parameter_slab, 0, sizeof(parameter_slab));
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    parameter_slab = id4_pipeline_make_parameter_slab_plan(
        options->parameter_scope, options->parameter_slab_placement_id,
        options->parameter_slab_target_params, parameter_slab_byte_length,
        options->parameter_slab_alignment, counts.parameter_count,
        parameter_requests);
  }

  id4_pipeline_region_plan_t region;
  memset(&region, 0, sizeof(region));
  const bool has_region = counts.region_operation_count != 0;
  if (iree_status_is_ok(status) && has_region) {
    region.name = id4_pipeline_program_name(options->program);
    region.placement_id = options->region_placement_id;
    region.binding_capacity = options->region_binding_capacity;
    region.local_binding_slot = options->region_local_binding_slot;
    region.statistics.operation_count = counts.region_operation_count;
    region.statistics.dispatch_count = counts.dispatch_count;
    region.statistics.barrier_count = counts.barrier_count;
    region.statistics.current_epoch = (uint32_t)counts.barrier_count;
  } else if (iree_status_is_ok(status) && counts.tap_count != 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "program has diagnostic taps but no executable "
                              "region operations");
  }

  if (iree_status_is_ok(status)) {
    id4_pipeline_plan_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.stage_name = id4_pipeline_program_name(options->program);
    create_options.device_group = options->device_group;
    create_options.placement_count = options->placement_count;
    create_options.placements = options->placements;
    create_options.parameter_slab_count = counts.parameter_count == 0 ? 0 : 1;
    create_options.parameter_slabs =
        counts.parameter_count == 0 ? NULL : &parameter_slab;
    create_options.kernel_count = kernel_count;
    create_options.kernels = kernel_count == 0 ? NULL : kernels;
    create_options.region_count = has_region ? 1 : 0;
    create_options.regions = has_region ? &region : NULL;
    create_options.diagnostic_tap_count = counts.tap_count;
    create_options.diagnostic_taps = counts.tap_count == 0 ? NULL : taps;
    create_options.diagnostics_sink = options->diagnostics_sink;
    status =
        id4_pipeline_plan_create(&create_options, host_allocator, out_plan);
  }

  id4_pipeline_program_plan_release_specialization_keys(kernel_count, kernels,
                                                        host_allocator);
  iree_allocator_free(host_allocator, taps);
  iree_allocator_free(host_allocator, kernels);
  iree_allocator_free(host_allocator, parameter_requests);
  return status;
}
