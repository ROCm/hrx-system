// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_plan.h"

#include <string.h>

#include "experimental/id4/pipeline/program_region.h"
#include "iree/base/internal/arena.h"

typedef struct id4_pipeline_program_plan_counts_t {
  // Number of external tensor import operations in the program.
  iree_host_size_t import_count;
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

typedef struct id4_pipeline_program_plan_lowering_context_t {
  // Program lowering options.
  const id4_pipeline_program_plan_options_t* options;
  // Parameter requests in program parameter-operation order.
  const id4_pipeline_parameter_request_t* parameter_requests;
  // Boundary tensor plans in program import-operation order.
  const id4_pipeline_boundary_tensor_plan_t* boundary_tensors;
  // Diagnostic tap plans in program tap-operation order.
  const id4_pipeline_diagnostic_tap_plan_t* diagnostic_taps;
} id4_pipeline_program_plan_lowering_context_t;

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
  const id4_pipeline_program_plan_flags_t allowed_flags =
      ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  if (iree_any_bit_set(options->flags, ~allowed_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported program plan flags 0x%x",
                            options->flags);
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
  if (options->region_local_slab_alignment != 0 &&
      !iree_device_size_is_power_of_two(options->region_local_slab_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program plan region local slab alignment must be a power of two");
  }
  if (options->region_local_tensor_alignment != 0 &&
      !iree_device_size_is_power_of_two(
          options->region_local_tensor_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program plan region local tensor alignment must be a power of two");
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
      case ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT:
        ++counts.import_count;
        break;
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

static id4_pipeline_tensor_shape_t id4_pipeline_program_plan_convert_shape(
    id4_pipeline_program_shape_t source) {
  id4_pipeline_tensor_shape_t target;
  memset(&target, 0, sizeof(target));
  target.rank = source.rank;
  memcpy(target.dims, source.dims, sizeof(target.dims));
  return target;
}

static bool id4_pipeline_program_plan_tensor_is_exported(
    const id4_pipeline_program_t* program,
    id4_pipeline_program_tensor_t tensor) {
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT) continue;
    if (op->payload.export_value.tensor.ordinal == tensor.ordinal) return true;
  }
  return false;
}

static bool id4_pipeline_program_plan_captures_diagnostic_taps(
    const id4_pipeline_program_plan_options_t* options) {
  return iree_all_bits_set(
      options->flags, ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS);
}

static iree_status_t id4_pipeline_program_plan_validate_exports(
    const id4_pipeline_program_t* program) {
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT) continue;
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(program,
                                       op->payload.export_value.tensor.ordinal);
    if (!tensor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program export tensor %u is missing",
                              op->payload.export_value.tensor.ordinal);
    }
    const id4_pipeline_program_op_t* producer =
        id4_pipeline_program_operation_at(program,
                                          tensor->producer_operation_ordinal);
    if (!producer || producer->kind != ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "program export %.*s must reference an imported "
                              "boundary tensor",
                              (int)op->payload.export_value.name.size,
                              op->payload.export_value.name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_validate_boundary_slots(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts) {
  for (iree_host_size_t i = 0; i < counts.import_count; ++i) {
    if (i > UINT32_MAX ||
        options->region_boundary_binding_slot_base > UINT32_MAX - (uint32_t)i) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program boundary binding slot overflow");
    }
    const uint32_t binding_slot =
        options->region_boundary_binding_slot_base + (uint32_t)i;
    if (binding_slot >= options->region_binding_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program boundary binding slot %u exceeds "
                              "region binding capacity %" PRIhsz,
                              binding_slot, options->region_binding_capacity);
    }
    if (binding_slot == options->region_local_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program boundary binding slot must not match "
                              "the local slab binding slot");
    }
    if (counts.parameter_count != 0 &&
        binding_slot == options->parameter_slab_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program boundary binding slot must not match "
                              "the parameter slab binding slot");
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_validate_diagnostic_tap_slots(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts) {
  if (!id4_pipeline_program_plan_captures_diagnostic_taps(options)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < counts.tap_count; ++i) {
    if (i > UINT32_MAX ||
        options->diagnostic_tap_binding_slot_base > UINT32_MAX - (uint32_t)i) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program diagnostic tap binding slot overflow");
    }
    const uint32_t binding_slot =
        options->diagnostic_tap_binding_slot_base + (uint32_t)i;
    if (binding_slot >= options->region_binding_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program diagnostic tap binding slot %u exceeds region binding "
          "capacity %" PRIhsz,
          binding_slot, options->region_binding_capacity);
    }
    if (binding_slot == options->region_local_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program diagnostic tap binding slot must not "
                              "match the local slab binding slot");
    }
    if (counts.parameter_count != 0 &&
        binding_slot == options->parameter_slab_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program diagnostic tap binding slot must not "
                              "match the parameter slab binding slot");
    }
    for (iree_host_size_t boundary_index = 0;
         boundary_index < counts.import_count; ++boundary_index) {
      const uint32_t boundary_slot =
          options->region_boundary_binding_slot_base + (uint32_t)boundary_index;
      if (binding_slot == boundary_slot) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "program diagnostic tap binding slot must not match boundary "
            "binding slot %u",
            boundary_slot);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_build_boundary_tensors(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts,
    id4_pipeline_boundary_tensor_plan_t* boundary_tensors) {
  if (counts.import_count == 0) return iree_ok_status();

  iree_host_size_t boundary_index = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT) continue;
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(options->program,
                                       op->payload.import_value.tensor.ordinal);
    if (!tensor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program import tensor %u is missing",
                              op->payload.import_value.tensor.ordinal);
    }
    id4_pipeline_boundary_tensor_flags_t flags =
        ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED;
    if (iree_all_bits_set(
            op->payload.import_value.flags,
            ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED)) {
      flags |= ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED;
    }
    if (id4_pipeline_program_plan_tensor_is_exported(
            options->program, op->payload.import_value.tensor)) {
      flags |= ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED;
    }
    boundary_tensors[boundary_index] = (id4_pipeline_boundary_tensor_plan_t){
        // Tensor layout and diagnostic name.
        .layout =
            {
                // Stable tensor name.
                .name = tensor->name,
                // Tensor element type.
                .dtype =
                    id4_pipeline_program_region_convert_dtype(tensor->dtype),
                // Tensor shape.
                .shape = id4_pipeline_program_plan_convert_shape(tensor->shape),
                // Dense tensor byte length.
                .byte_length = tensor->byte_length,
                // No additional external buffer base alignment.
                .alignment = 0,
            },
        // Boundary behavior flags.
        .flags = flags,
        // Semantic programs currently lower into one executable region.
        .region_id = 0,
        // External tensors follow the executable region placement.
        .placement_id = options->region_placement_id,
        // Binding slot assigned from the configured boundary range.
        .binding_slot = options->region_boundary_binding_slot_base +
                        (uint32_t)boundary_index,
    };
    ++boundary_index;
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
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_format_dispatch_specialization_key(
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
  if (!id4_pipeline_program_plan_captures_diagnostic_taps(options) ||
      counts.tap_count == 0) {
    return iree_ok_status();
  }

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
        taps[tap_index].placement_id = options->region_placement_id;
        taps[tap_index].binding_slot =
            options->diagnostic_tap_binding_slot_base + (uint32_t)tap_index;
        taps[tap_index].after_operation_ordinal = region_operation_count - 1;
        taps[tap_index].target_name = tensor->name;
        taps[tap_index].layout = (id4_pipeline_tensor_layout_t){
            // Stable capture record name.
            .name = op->payload.tap.name,
            // Captured tensor element type.
            .dtype = id4_pipeline_program_region_convert_dtype(tensor->dtype),
            // Captured tensor shape.
            .shape = id4_pipeline_program_plan_convert_shape(tensor->shape),
            // Dense captured tensor byte length.
            .byte_length = tensor->byte_length,
            // Tap bindings are standalone issue-time buffers.
            .alignment = 0,
        };
        ++tap_index;
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_resolve_import(
    void* user_data, const id4_pipeline_program_import_op_t* import_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t import_ordinal, id4_pipeline_tensor_import_t* out_import) {
  (void)import_op;
  (void)tensor;
  id4_pipeline_program_plan_lowering_context_t* context =
      (id4_pipeline_program_plan_lowering_context_t*)user_data;
  const id4_pipeline_boundary_tensor_plan_t* boundary =
      &context->boundary_tensors[import_ordinal];
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

static iree_status_t id4_pipeline_program_plan_resolve_parameter(
    void* user_data, const id4_pipeline_program_parameter_op_t* parameter_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t parameter_ordinal,
    id4_pipeline_tensor_import_t* out_import) {
  (void)parameter_op;
  id4_pipeline_program_plan_lowering_context_t* context =
      (id4_pipeline_program_plan_lowering_context_t*)user_data;
  const id4_pipeline_parameter_request_t* request =
      &context->parameter_requests[parameter_ordinal];
  out_import->layout = (id4_pipeline_tensor_layout_t){
      // Parameter tensor diagnostic name.
      .name = tensor->name,
      // Parameter tensor element type.
      .dtype = id4_pipeline_program_region_convert_dtype(tensor->dtype),
      // Parameter tensor shape.
      .shape = id4_pipeline_program_plan_convert_shape(tensor->shape),
      // Dense parameter tensor byte length.
      .byte_length = tensor->byte_length,
      // Required subrange alignment in the packed slab.
      .alignment = context->options->parameter_request_alignment,
  };
  out_import->binding_slot = context->options->parameter_slab_binding_slot;
  out_import->offset = request->span.buffer_offset;
  out_import->flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_resolve_tap(
    void* user_data, const id4_pipeline_program_tap_op_t* tap_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t tap_ordinal, id4_pipeline_tensor_import_t* out_import) {
  (void)tap_op;
  (void)tensor;
  id4_pipeline_program_plan_lowering_context_t* context =
      (id4_pipeline_program_plan_lowering_context_t*)user_data;
  const id4_pipeline_diagnostic_tap_plan_t* tap =
      &context->diagnostic_taps[tap_ordinal];
  out_import->layout = tap->layout;
  out_import->binding_slot = tap->binding_slot;
  out_import->offset = 0;
  out_import->flags = 0;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_dry_run_region(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts,
    const id4_pipeline_parameter_request_t* parameter_requests,
    const id4_pipeline_boundary_tensor_plan_t* boundary_tensors,
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_taps,
    iree_allocator_t host_allocator,
    id4_pipeline_region_statistics_t* out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  if (counts.region_operation_count == 0) return iree_ok_status();

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096, host_allocator,
                                   &block_pool);

  id4_pipeline_region_builder_t* builder = NULL;
  id4_pipeline_region_builder_create_options_t builder_options = {
      // Size of this structure for versioning.
      .structure_size = sizeof(builder_options),
      // Region name copied by the builder.
      .region_name = id4_pipeline_program_name(options->program),
      // Dry-run validates the same region operations without HAL recording.
      .mode = ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN,
      // Arena block pool used for region transient metadata.
      .block_pool = &block_pool,
      // Exact issue-time binding-table capacity.
      .binding_capacity = options->region_binding_capacity,
      // Binding-table slot reserved for the local transient slab.
      .local_binding_slot = options->region_local_binding_slot,
  };
  iree_status_t status = id4_pipeline_region_builder_create(
      &builder_options, host_allocator, &builder);

  id4_pipeline_program_plan_lowering_context_t lowering_context = {
      // Program lowering options.
      .options = options,
      // Parameter request table.
      .parameter_requests = parameter_requests,
      // Boundary tensor table.
      .boundary_tensors = boundary_tensors,
      // Diagnostic tap tensor table.
      .diagnostic_taps = diagnostic_taps,
  };
  if (iree_status_is_ok(status)) {
    id4_pipeline_program_region_lower_options_t lower_options = {
        // Size of this structure for versioning.
        .structure_size = sizeof(lower_options),
        // Semantic program being planned.
        .program = options->program,
        // Dry-run region builder.
        .builder = builder,
        // Diagnostic tap lowering policy.
        .tap_mode = id4_pipeline_program_plan_captures_diagnostic_taps(options)
                        ? ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_CAPTURE
                        : ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_IGNORE,
        // Required local tensor alignment.
        .local_tensor_alignment = options->region_local_tensor_alignment,
        // Planner resolver context.
        .user_data = &lowering_context,
        // Resolves boundary tensor imports.
        .resolve_import = id4_pipeline_program_plan_resolve_import,
        // Resolves parameter tensor imports.
        .resolve_parameter = id4_pipeline_program_plan_resolve_parameter,
        // Resolves diagnostic tap imports.
        .resolve_tap = id4_pipeline_program_plan_resolve_tap,
    };
    status = id4_pipeline_program_region_lower(&lower_options, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_statistics(builder, out_statistics);
  }

  id4_pipeline_region_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_pipeline_program_plan_make_local_slab_name(
    iree_string_view_t region_name, iree_allocator_t host_allocator,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_string_builder_append_string(&builder, region_name);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, ".local");
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t name_size = iree_string_builder_size(&builder);
    char* storage = iree_string_builder_take_storage(&builder);
    *out_name = iree_make_string_view(storage, name_size);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

iree_status_t id4_pipeline_program_create_plan(
    const id4_pipeline_program_plan_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_plan_validate_options(options));

  id4_pipeline_program_plan_counts_t counts =
      id4_pipeline_program_plan_count_ops(options->program);
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_plan_validate_exports(options->program));
  if (counts.parameter_count != 0) {
    if (options->parameter_slab_binding_slot >=
        options->region_binding_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program parameter slab binding slot %u exceeds "
                              "region binding capacity %" PRIhsz,
                              options->parameter_slab_binding_slot,
                              options->region_binding_capacity);
    }
    if (options->parameter_slab_binding_slot ==
        options->region_local_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program parameter slab binding slot must not "
                              "match the local slab binding slot");
    }
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_plan_validate_boundary_slots(options, counts));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_plan_validate_diagnostic_tap_slots(options, counts));

  id4_pipeline_parameter_request_t* parameter_requests = NULL;
  id4_pipeline_boundary_tensor_plan_t* boundary_tensors = NULL;
  id4_pipeline_kernel_plan_t* kernels = NULL;
  id4_pipeline_diagnostic_tap_plan_t* taps = NULL;
  const iree_host_size_t planned_tap_count =
      id4_pipeline_program_plan_captures_diagnostic_taps(options)
          ? counts.tap_count
          : 0;
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
  if (iree_status_is_ok(status) && counts.import_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.import_count,
                                         sizeof(boundary_tensors[0]),
                                         (void**)&boundary_tensors);
  }
  if (iree_status_is_ok(status) && planned_tap_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, planned_tap_count,
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
    status = id4_pipeline_program_plan_build_boundary_tensors(options, counts,
                                                              boundary_tensors);
  }

  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_plan_build_taps(options, counts, taps);
  }

  id4_pipeline_region_statistics_t region_statistics;
  memset(&region_statistics, 0, sizeof(region_statistics));
  const bool has_region = counts.region_operation_count != 0;
  if (iree_status_is_ok(status) && has_region) {
    status = id4_pipeline_program_plan_dry_run_region(
        options, counts, parameter_requests, boundary_tensors, taps,
        host_allocator, &region_statistics);
  } else if (iree_status_is_ok(status) && planned_tap_count != 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "program has diagnostic taps but no executable "
                              "region operations");
  }

  id4_pipeline_parameter_slab_plan_t parameter_slab;
  memset(&parameter_slab, 0, sizeof(parameter_slab));
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    parameter_slab = id4_pipeline_make_parameter_slab_plan(
        options->parameter_scope, options->parameter_slab_placement_id,
        options->parameter_slab_binding_slot,
        options->parameter_slab_target_params, parameter_slab_byte_length,
        options->parameter_slab_alignment, counts.parameter_count,
        parameter_requests);
  }

  iree_string_view_t local_slab_name = iree_string_view_empty();
  id4_pipeline_memory_slab_plan_t local_slab;
  memset(&local_slab, 0, sizeof(local_slab));
  const bool has_local_slab = region_statistics.local_slab_byte_length != 0;
  if (iree_status_is_ok(status) && has_local_slab) {
    status = id4_pipeline_program_plan_make_local_slab_name(
        id4_pipeline_program_name(options->program), host_allocator,
        &local_slab_name);
  }
  if (iree_status_is_ok(status) && has_local_slab) {
    local_slab = (id4_pipeline_memory_slab_plan_t){
        // Human-readable local slab name.
        .name = local_slab_name,
        // Local transient slab follows the executable region placement.
        .placement_id = options->region_placement_id,
        // Binding-table slot reserved for the local transient slab.
        .binding_slot = options->region_local_binding_slot,
        // HAL buffer parameters used for local transient allocation.
        .params = options->region_local_slab_params,
        // Planned local slab byte length.
        .byte_length = region_statistics.local_slab_byte_length,
        // Required local slab base alignment.
        .alignment = options->region_local_slab_alignment,
        // Peak live local slab bytes observed by dry-run lowering.
        .high_water_mark = region_statistics.local_slab_high_water_mark,
    };
  }

  id4_pipeline_region_plan_t region;
  memset(&region, 0, sizeof(region));
  if (iree_status_is_ok(status) && has_region) {
    region.name = id4_pipeline_program_name(options->program);
    region.placement_id = options->region_placement_id;
    region.binding_capacity = options->region_binding_capacity;
    region.local_binding_slot = options->region_local_binding_slot;
    region.local_tensor_alignment = options->region_local_tensor_alignment;
    region.statistics = region_statistics;
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
    create_options.memory_slab_count = has_local_slab ? 1 : 0;
    create_options.memory_slabs = has_local_slab ? &local_slab : NULL;
    create_options.boundary_tensor_count = counts.import_count;
    create_options.boundary_tensors =
        counts.import_count == 0 ? NULL : boundary_tensors;
    create_options.kernel_count = kernel_count;
    create_options.kernels = kernel_count == 0 ? NULL : kernels;
    create_options.region_count = has_region ? 1 : 0;
    create_options.regions = has_region ? &region : NULL;
    create_options.diagnostic_tap_count = planned_tap_count;
    create_options.diagnostic_taps = planned_tap_count == 0 ? NULL : taps;
    create_options.diagnostics_sink = options->diagnostics_sink;
    status =
        id4_pipeline_plan_create(&create_options, host_allocator, out_plan);
  }

  iree_allocator_free(host_allocator, (void*)local_slab_name.data);
  id4_pipeline_program_plan_release_specialization_keys(kernel_count, kernels,
                                                        host_allocator);
  iree_allocator_free(host_allocator, taps);
  iree_allocator_free(host_allocator, kernels);
  iree_allocator_free(host_allocator, boundary_tensors);
  iree_allocator_free(host_allocator, parameter_requests);
  return status;
}
