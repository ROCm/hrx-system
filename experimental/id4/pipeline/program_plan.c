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
  // Number of provider gather requests produced by parameter operations.
  iree_host_size_t parameter_request_count;
  // Number of program-owned constant operations in the program.
  iree_host_size_t constant_count;
  // Number of acquired transient tensors in the program.
  iree_host_size_t acquire_count;
  // Number of Loom dispatch operations in the program.
  iree_host_size_t dispatch_count;
  // Number of barrier operations in the program.
  iree_host_size_t barrier_count;
  // Number of stage-region cut operations in the program.
  iree_host_size_t region_cut_count;
  // Number of executable region operations in the program.
  iree_host_size_t region_operation_count;
  // Number of diagnostic tap operations in the program.
  iree_host_size_t tap_count;
} id4_pipeline_program_plan_counts_t;

typedef struct id4_pipeline_program_plan_parameter_request_range_t {
  // First gather request ordinal backing a parameter tensor.
  iree_host_size_t request_offset;
  // Number of gather requests backing a parameter tensor.
  iree_host_size_t request_count;
} id4_pipeline_program_plan_parameter_request_range_t;

typedef struct id4_pipeline_program_plan_lowering_context_t {
  // Program lowering options.
  const id4_pipeline_program_plan_options_t* options;
  // Shared tensor plans in program tensor ordinal lookup order.
  const id4_pipeline_shared_tensor_plan_t* shared_tensors;
  // Number of shared tensor plans.
  iree_host_size_t shared_tensor_count;
  // Parameter requests in provider gather enumeration order.
  const id4_pipeline_parameter_request_t* parameter_requests;
  // Parameter request ranges in program parameter-operation order.
  const id4_pipeline_program_plan_parameter_request_range_t*
      parameter_request_ranges;
  // Constant requests in program constant-operation order.
  const id4_pipeline_constant_request_t* constant_requests;
  // Boundary tensor plans in program import-operation order.
  const id4_pipeline_boundary_tensor_plan_t* boundary_tensors;
  // Diagnostic tap plans in program tap-operation order.
  const id4_pipeline_diagnostic_tap_plan_t* diagnostic_taps;
} id4_pipeline_program_plan_lowering_context_t;

typedef struct id4_pipeline_program_plan_parameter_load_record_t {
  // Prepare-time source-to-execution transformation.
  id4_pipeline_program_parameter_encoding_t encoding;
  // First request ordinal owned by this parameter.
  iree_host_size_t request_offset;
  // Number of requests owned by this parameter.
  iree_host_size_t request_count;
  // Provider scope for direct gather records.
  iree_string_view_t source_scope;
  // Number of provider source descriptors.
  iree_host_size_t source_count;
  // Provider source descriptors borrowed from temporary planning storage.
  const id4_pipeline_parameter_load_source_t* sources;
} id4_pipeline_program_plan_parameter_load_record_t;

typedef struct id4_pipeline_program_plan_region_range_t {
  // Region diagnostic name borrowed from the program or closing cut.
  iree_string_view_t name;
  // First source-program operation in the region interval.
  iree_host_size_t source_operation_offset;
  // Number of source-program operations in the region interval.
  iree_host_size_t source_operation_count;
  // Number of dispatch/barrier operations in the region interval.
  iree_host_size_t region_operation_count;
} id4_pipeline_program_plan_region_range_t;

typedef struct id4_pipeline_program_plan_shared_tensor_record_t {
  // Final shared tensor plan entry.
  id4_pipeline_shared_tensor_plan_t plan;
  // Source-program operation ordinal that acquires the tensor.
  iree_host_size_t acquire_operation_ordinal;
  // Last source-program operation ordinal that may use the tensor.
  iree_host_size_t last_use_operation_ordinal;
} id4_pipeline_program_plan_shared_tensor_record_t;

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
      ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS |
      ID4_PIPELINE_PROGRAM_PLAN_FLAG_REGION_PER_DISPATCH;
  if (iree_any_bit_set(options->flags, ~allowed_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported program plan flags 0x%x",
                            options->flags);
  }
  if (iree_all_bits_set(
          options->flags,
          ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS)) {
    if (options->diagnostic_tap_names.count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program diagnostic tap capture requires at least one tap name");
    }
    if (!options->diagnostic_tap_names.values) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program diagnostic tap capture requires a tap name list");
    }
    for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
      iree_string_view_t name = options->diagnostic_tap_names.values[i];
      if (iree_string_view_is_empty(name)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "program diagnostic tap capture name %" PRIhsz " is empty", i);
      }
      for (iree_host_size_t j = 0; j < i; ++j) {
        if (iree_string_view_equal(name,
                                   options->diagnostic_tap_names.values[j])) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "program diagnostic tap capture name `%.*s` is duplicated",
              (int)name.size, name.data);
        }
      }
    }
  } else {
    if (options->diagnostic_tap_names.count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program diagnostic tap names require diagnostic tap capture");
    }
    if (options->diagnostic_tap_names.values) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program diagnostic tap name list requires diagnostic tap capture");
    }
  }
  if (iree_string_view_is_empty(options->stage_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program plan stage name is required");
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
  if (options->constant_slab_alignment != 0 &&
      !iree_device_size_is_power_of_two(options->constant_slab_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program plan constant slab alignment must be a power of two");
  }
  if (options->constant_request_alignment != 0 &&
      !iree_device_size_is_power_of_two(options->constant_request_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program plan constant request alignment must be a power of two");
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

static bool id4_pipeline_program_plan_captures_diagnostic_taps(
    const id4_pipeline_program_plan_options_t* options) {
  return iree_all_bits_set(
      options->flags, ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS);
}

static bool id4_pipeline_program_plan_uses_dispatch_regions(
    const id4_pipeline_program_plan_options_t* options) {
  return iree_all_bits_set(options->flags,
                           ID4_PIPELINE_PROGRAM_PLAN_FLAG_REGION_PER_DISPATCH);
}

static bool id4_pipeline_program_plan_tap_name_requested(
    const id4_pipeline_program_plan_options_t* options,
    iree_string_view_t name) {
  if (!id4_pipeline_program_plan_captures_diagnostic_taps(options)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
    if (iree_string_view_equal(options->diagnostic_tap_names.values[i], name)) {
      return true;
    }
  }
  return false;
}

static id4_pipeline_program_plan_counts_t id4_pipeline_program_plan_count_ops(
    const id4_pipeline_program_plan_options_t* options) {
  id4_pipeline_program_plan_counts_t counts;
  memset(&counts, 0, sizeof(counts));
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op) continue;
    switch (op->kind) {
      case ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT:
        ++counts.import_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER:
        ++counts.parameter_count;
        counts.parameter_request_count +=
            op->payload.parameter.source_span_count == 0
                ? 1
                : op->payload.parameter.source_span_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT:
        ++counts.constant_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE:
        ++counts.acquire_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
        ++counts.dispatch_count;
        ++counts.region_operation_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
        ++counts.barrier_count;
        ++counts.region_operation_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_REGION_CUT:
        ++counts.region_cut_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
        if (id4_pipeline_program_plan_tap_name_requested(
                options, op->payload.tap.name)) {
          ++counts.tap_count;
        }
        break;
      default:
        break;
    }
  }
  return counts;
}

static bool id4_pipeline_program_plan_op_is_region_operation(
    const id4_pipeline_program_op_t* op) {
  return op && (op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM ||
                op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER);
}

static iree_host_size_t id4_pipeline_program_plan_count_region_operations(
    const id4_pipeline_program_t* program, iree_host_size_t operation_offset,
    iree_host_size_t operation_limit) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = operation_offset; i < operation_limit; ++i) {
    if (id4_pipeline_program_plan_op_is_region_operation(
            id4_pipeline_program_operation_at(program, i))) {
      ++count;
    }
  }
  return count;
}

static void id4_pipeline_program_plan_append_region_range(
    const id4_pipeline_program_t* program, iree_string_view_t name,
    iree_host_size_t operation_offset, iree_host_size_t operation_limit,
    iree_host_size_t* range_count,
    id4_pipeline_program_plan_region_range_t* ranges) {
  if (operation_limit <= operation_offset) return;
  const iree_host_size_t region_operation_count =
      id4_pipeline_program_plan_count_region_operations(
          program, operation_offset, operation_limit);
  if (region_operation_count == 0) return;
  ranges[*range_count] = (id4_pipeline_program_plan_region_range_t){
      // Region diagnostic name borrowed from the program or closing cut.
      .name = name,
      // First source-program operation in the interval.
      .source_operation_offset = operation_offset,
      // Number of source-program operations in the interval.
      .source_operation_count = operation_limit - operation_offset,
      // Dispatch/barrier operation count in the interval.
      .region_operation_count = region_operation_count,
  };
  ++*range_count;
}

static void id4_pipeline_program_plan_build_region_ranges(
    const id4_pipeline_program_plan_options_t* options,
    iree_host_size_t* out_range_count,
    id4_pipeline_program_plan_region_range_t* ranges) {
  *out_range_count = 0;
  const id4_pipeline_program_t* program = options->program;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  iree_host_size_t operation_offset = 0;
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (id4_pipeline_program_plan_uses_dispatch_regions(options) && op &&
        op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      id4_pipeline_program_plan_append_region_range(
          program, op->payload.dispatch_loom.name, operation_offset, i + 1,
          out_range_count, ranges);
      operation_offset = i + 1;
      continue;
    }
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_REGION_CUT) continue;
    id4_pipeline_program_plan_append_region_range(
        program, op->payload.region_cut.name, operation_offset, i,
        out_range_count, ranges);
    operation_offset = i + 1;
  }
  id4_pipeline_program_plan_append_region_range(
      program, id4_pipeline_program_name(program), operation_offset,
      operation_count, out_range_count, ranges);
}

static iree_status_t id4_pipeline_program_plan_find_region_index_for_operation(
    iree_host_size_t range_count,
    const id4_pipeline_program_plan_region_range_t* ranges,
    iree_host_size_t operation_ordinal, iree_host_size_t* out_region_index) {
  *out_region_index = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    const iree_host_size_t range_offset = ranges[i].source_operation_offset;
    const iree_host_size_t range_limit =
        ranges[i].source_operation_offset + ranges[i].source_operation_count;
    if (operation_ordinal >= range_offset && operation_ordinal < range_limit) {
      *out_region_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "program operation %" PRIhsz
                          " is not contained by an executable region",
                          operation_ordinal);
}

static bool id4_pipeline_program_plan_operation_uses_tensor(
    const id4_pipeline_program_plan_options_t* options,
    const id4_pipeline_program_op_t* op, id4_pipeline_program_tensor_t tensor) {
  if (!op) return false;
  switch (op->kind) {
    case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
      for (iree_host_size_t i = 0; i < op->payload.dispatch_loom.binding_count;
           ++i) {
        if (op->payload.dispatch_loom.bindings[i].tensor.ordinal ==
            tensor.ordinal) {
          return true;
        }
      }
      return false;
    case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
      return id4_pipeline_program_plan_tap_name_requested(
                 options, op->payload.tap.name) &&
             op->payload.tap.tensor.ordinal == tensor.ordinal;
    case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT:
      return op->payload.export_value.tensor.ordinal == tensor.ordinal;
    default:
      return false;
  }
}

static iree_host_size_t id4_pipeline_program_plan_find_last_tensor_use(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_tensor_t tensor) {
  iree_host_size_t last_use_operation_ordinal = IREE_HOST_SIZE_MAX;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (id4_pipeline_program_plan_operation_uses_tensor(options, op, tensor)) {
      last_use_operation_ordinal = i;
    }
  }
  return last_use_operation_ordinal;
}

static iree_status_t id4_pipeline_program_plan_try_reuse_shared_range(
    iree_host_size_t shared_tensor_count,
    const id4_pipeline_program_plan_shared_tensor_record_t* shared_tensors,
    iree_host_size_t acquire_region_id,
    iree_host_size_t acquire_operation_ordinal, iree_device_size_t byte_length,
    iree_device_size_t alignment, iree_device_size_t* out_offset,
    bool* out_found) {
  *out_offset = 0;
  *out_found = false;
  const iree_device_size_t effective_alignment = alignment == 0 ? 1 : alignment;
  for (iree_host_size_t i = 0; i < shared_tensor_count; ++i) {
    const id4_pipeline_program_plan_shared_tensor_record_t* candidate_record =
        &shared_tensors[i];
    if (candidate_record->last_use_operation_ordinal >=
        acquire_operation_ordinal) {
      continue;
    }
    const id4_pipeline_shared_tensor_plan_t* candidate =
        &candidate_record->plan;
    if (candidate->last_use_region_id >= acquire_region_id) continue;
    if (candidate->layout.byte_length < byte_length) continue;
    if (!iree_device_size_has_alignment(candidate->offset,
                                        effective_alignment)) {
      continue;
    }
    iree_device_size_t candidate_end = 0;
    if (!iree_device_size_checked_add(candidate->offset, byte_length,
                                      &candidate_end)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "shared transient candidate range overflow");
    }
    bool overlaps_live_range = false;
    for (iree_host_size_t j = 0; j < shared_tensor_count; ++j) {
      const id4_pipeline_program_plan_shared_tensor_record_t* live_record =
          &shared_tensors[j];
      if (live_record->acquire_operation_ordinal > acquire_operation_ordinal ||
          live_record->last_use_operation_ordinal < acquire_operation_ordinal) {
        continue;
      }
      const id4_pipeline_shared_tensor_plan_t* live = &live_record->plan;
      iree_device_size_t live_end = 0;
      if (!iree_device_size_checked_add(live->offset, live->layout.byte_length,
                                        &live_end)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "shared transient live range overflow");
      }
      if (candidate->offset < live_end && live->offset < candidate_end) {
        overlaps_live_range = true;
        break;
      }
    }
    if (overlaps_live_range) continue;
    *out_offset = candidate->offset;
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_bump_shared_range(
    iree_device_size_t byte_length, iree_device_size_t alignment,
    iree_device_size_t* io_shared_slab_byte_length,
    iree_device_size_t* out_offset) {
  const iree_device_size_t effective_alignment = alignment == 0 ? 1 : alignment;
  iree_device_size_t aligned_offset = 0;
  if (!iree_device_size_checked_align(*io_shared_slab_byte_length,
                                      effective_alignment, &aligned_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "shared transient slab alignment overflow");
  }
  iree_device_size_t allocation_end = 0;
  if (!iree_device_size_checked_add(aligned_offset, byte_length,
                                    &allocation_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "shared transient slab allocation overflow");
  }
  *io_shared_slab_byte_length = allocation_end;
  *out_offset = aligned_offset;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_shared_tensor_high_water_mark(
    iree_host_size_t shared_tensor_count,
    const id4_pipeline_program_plan_shared_tensor_record_t* shared_tensors,
    iree_device_size_t* out_high_water_mark) {
  *out_high_water_mark = 0;
  iree_device_size_t high_water_mark = 0;
  for (iree_host_size_t point_index = 0; point_index < shared_tensor_count;
       ++point_index) {
    const iree_host_size_t operation_ordinal =
        shared_tensors[point_index].acquire_operation_ordinal;
    iree_device_size_t live_byte_length = 0;
    for (iree_host_size_t tensor_index = 0; tensor_index < shared_tensor_count;
         ++tensor_index) {
      const id4_pipeline_program_plan_shared_tensor_record_t* shared_tensor =
          &shared_tensors[tensor_index];
      if (shared_tensor->acquire_operation_ordinal <= operation_ordinal &&
          shared_tensor->last_use_operation_ordinal >= operation_ordinal) {
        if (!iree_device_size_checked_add(
                live_byte_length, shared_tensor->plan.layout.byte_length,
                &live_byte_length)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "shared transient live byte length overflow");
        }
      }
    }
    high_water_mark = iree_max(high_water_mark, live_byte_length);
  }
  *out_high_water_mark = high_water_mark;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_build_shared_tensors(
    const id4_pipeline_program_plan_options_t* options,
    iree_host_size_t range_count,
    const id4_pipeline_program_plan_region_range_t* ranges,
    id4_pipeline_program_plan_shared_tensor_record_t* shared_tensors,
    iree_host_size_t* out_shared_tensor_count,
    iree_device_size_t* out_shared_slab_byte_length,
    iree_device_size_t* out_shared_slab_high_water_mark) {
  *out_shared_tensor_count = 0;
  *out_shared_slab_byte_length = 0;
  *out_shared_slab_high_water_mark = 0;
  if (range_count < 2) return iree_ok_status();

  for (iree_host_size_t region_index = 0; region_index < range_count;
       ++region_index) {
    const id4_pipeline_program_plan_region_range_t* range =
        &ranges[region_index];
    const iree_host_size_t operation_limit =
        range->source_operation_offset + range->source_operation_count;
    for (iree_host_size_t i = range->source_operation_offset;
         i < operation_limit; ++i) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(options->program, i);
      if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE) continue;

      const id4_pipeline_program_acquire_op_t* acquire = &op->payload.acquire;
      const iree_host_size_t last_use_operation_ordinal =
          id4_pipeline_program_plan_find_last_tensor_use(options,
                                                         acquire->tensor);
      if (last_use_operation_ordinal == IREE_HOST_SIZE_MAX) continue;

      iree_host_size_t last_use_region_id = IREE_HOST_SIZE_MAX;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_program_plan_find_region_index_for_operation(
              range_count, ranges, last_use_operation_ordinal,
              &last_use_region_id));
      if (last_use_region_id == region_index) continue;
      if (region_index > UINT32_MAX || last_use_region_id > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "shared transient region index overflow");
      }

      const id4_pipeline_program_tensor_record_t* tensor =
          id4_pipeline_program_tensor_at(options->program,
                                         acquire->tensor.ordinal);
      if (!tensor) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "program acquire tensor %u is missing",
                                acquire->tensor.ordinal);
      }
      id4_pipeline_tensor_shape_t shape;
      memset(&shape, 0, sizeof(shape));
      shape.rank = tensor->shape.rank;
      memcpy(shape.dims, tensor->shape.dims, sizeof(shape.dims));
      const id4_pipeline_tensor_layout_t layout = {
          // Shared tensor diagnostic name.
          .name = tensor->name,
          // Shared tensor element type.
          .dtype = id4_pipeline_program_region_convert_dtype(tensor->dtype),
          // Shared tensor shape.
          .shape = shape,
          // Dense shared tensor byte length.
          .byte_length = tensor->byte_length,
          // Required shared tensor alignment.
          .alignment = options->region_local_tensor_alignment,
      };

      iree_device_size_t offset = 0;
      bool found_reusable_range = false;
      IREE_RETURN_IF_ERROR(id4_pipeline_program_plan_try_reuse_shared_range(
          *out_shared_tensor_count, shared_tensors, region_index, i,
          layout.byte_length, layout.alignment, &offset,
          &found_reusable_range));
      if (!found_reusable_range) {
        IREE_RETURN_IF_ERROR(id4_pipeline_program_plan_bump_shared_range(
            layout.byte_length, layout.alignment, out_shared_slab_byte_length,
            &offset));
      }

      id4_pipeline_program_plan_shared_tensor_record_t* record =
          &shared_tensors[*out_shared_tensor_count];
      record->plan = (id4_pipeline_shared_tensor_plan_t){
          // Shared tensor layout.
          .layout = layout,
          // Semantic program tensor ordinal.
          .program_tensor_ordinal = acquire->tensor.ordinal,
          // Single plan-shared slab emitted below.
          .memory_slab_index = 0,
          // Byte offset into the shared slab.
          .offset = offset,
          // Region acquiring this tensor.
          .acquire_region_id = (uint32_t)region_index,
          // Last region that may use this tensor.
          .last_use_region_id = (uint32_t)last_use_region_id,
      };
      record->acquire_operation_ordinal = i;
      record->last_use_operation_ordinal = last_use_operation_ordinal;
      ++*out_shared_tensor_count;
    }
  }

  return id4_pipeline_program_plan_shared_tensor_high_water_mark(
      *out_shared_tensor_count, shared_tensors,
      out_shared_slab_high_water_mark);
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

static id4_pipeline_tensor_shape_t id4_pipeline_program_plan_convert_shape(
    id4_pipeline_program_shape_t source) {
  id4_pipeline_tensor_shape_t target;
  memset(&target, 0, sizeof(target));
  target.rank = source.rank;
  memcpy(target.dims, source.dims, sizeof(target.dims));
  return target;
}

static iree_status_t id4_pipeline_program_plan_make_parameter_load_source(
    const id4_pipeline_program_parameter_source_t* source,
    id4_pipeline_parameter_load_source_t* out_source) {
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_tensor_byte_length(
      source->dtype, source->shape, &byte_length));
  *out_source = id4_pipeline_parameter_load_source(
      source->source_scope, source->key,
      id4_pipeline_program_region_convert_dtype(source->dtype),
      id4_pipeline_program_plan_convert_shape(source->shape), byte_length);
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_build_parameter_requests(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts,
    id4_pipeline_parameter_request_t* requests,
    id4_pipeline_parameter_tensor_plan_t* parameter_tensors,
    id4_pipeline_program_plan_parameter_request_range_t* request_ranges,
    id4_pipeline_program_plan_parameter_load_record_t* load_records,
    id4_pipeline_parameter_load_source_t* load_sources,
    iree_device_size_t* out_parameter_slab_byte_length) {
  *out_parameter_slab_byte_length = 0;
  if (counts.parameter_count == 0) return iree_ok_status();

  iree_host_size_t request_index = 0;
  iree_host_size_t parameter_index = 0;
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
    id4_pipeline_program_plan_parameter_load_record_t* load_record =
        &load_records[parameter_index];
    memset(load_record, 0, sizeof(*load_record));
    load_record->encoding = op->payload.parameter.encoding;
    load_record->request_offset = request_index;
    load_record->request_count = op->payload.parameter.source_span_count == 0
                                     ? 1
                                     : op->payload.parameter.source_span_count;
    load_record->source_count = op->payload.parameter.source_count;
    load_record->sources =
        &load_sources[parameter_index *
                      ID4_PIPELINE_PROGRAM_PARAMETER_MAX_SOURCE_COUNT];
    id4_pipeline_parameter_load_source_t* mutable_sources =
        (id4_pipeline_parameter_load_source_t*)load_record->sources;
    for (iree_host_size_t j = 0; j < load_record->source_count; ++j) {
      IREE_RETURN_IF_ERROR(id4_pipeline_program_plan_make_parameter_load_source(
          &op->payload.parameter.sources[j], &mutable_sources[j]));
    }
    if (load_record->encoding ==
        ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT) {
      load_record->source_scope = mutable_sources[0].source_scope;
    }
    request_ranges[parameter_index] =
        (id4_pipeline_program_plan_parameter_request_range_t){
            // First gather request ordinal backing this parameter.
            .request_offset = load_record->request_offset,
            // Number of gather requests backing this parameter.
            .request_count = load_record->request_count,
        };
    parameter_tensors[parameter_index] = (id4_pipeline_parameter_tensor_plan_t){
        // Parameter tensor layout.
        .layout =
            {
                // Parameter tensor diagnostic name.
                .name = tensor->name,
                // Parameter tensor element type.
                .dtype =
                    id4_pipeline_program_region_convert_dtype(tensor->dtype),
                // Parameter tensor shape.
                .shape = id4_pipeline_program_plan_convert_shape(tensor->shape),
                // Dense parameter tensor byte length.
                .byte_length = tensor->byte_length,
                // Required parameter subrange alignment.
                .alignment = options->parameter_request_alignment,
            },
        // Semantic program tensor ordinal.
        .program_tensor_ordinal = op->payload.parameter.tensor.ordinal,
        // Single parameter slab emitted below.
        .parameter_slab_index = 0,
        // First request in the single parameter slab.
        .request_offset = load_record->request_offset,
        // Number of requests populating this tensor.
        .request_count = load_record->request_count,
        // Plan-global request ordinal matches the single-slab ordinal.
        .global_request_offset = load_record->request_offset,
        // Base byte offset of the tensor in the parameter slab.
        .offset = span.buffer_offset,
    };
    if (op->payload.parameter.source_span_count == 0) {
      requests[request_index] =
          id4_pipeline_parameter_request(tensor->name, span);
      ++request_index;
    } else {
      for (iree_host_size_t j = 0; j < op->payload.parameter.source_span_count;
           ++j) {
        const id4_pipeline_program_parameter_source_span_t* source_span =
            &op->payload.parameter.source_spans[j];
        iree_device_size_t buffer_offset = 0;
        if (!iree_device_size_checked_add(span.buffer_offset,
                                          source_span->target_offset,
                                          &buffer_offset)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "program parameter %.*s source span target offset overflow",
              (int)tensor->name.size, tensor->name.data);
        }
        requests[request_index] = id4_pipeline_parameter_request(
            op->payload.parameter.sources[0].key,
            id4_pipeline_parameter_span(source_span->source_offset,
                                        buffer_offset, source_span->length));
        ++request_index;
      }
    }
    ++parameter_index;
  }
  if (request_index != counts.parameter_request_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "program parameter request count %" PRIhsz
                            " does not match counted requests %" PRIhsz,
                            request_index, counts.parameter_request_count);
  }
  if (parameter_index != counts.parameter_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "program parameter count %" PRIhsz
        " does not match counted parameter operations %" PRIhsz,
        parameter_index, counts.parameter_count);
  }
  return iree_ok_status();
}

static iree_host_size_t id4_pipeline_program_plan_parameter_readiness_group_key(
    const iree_host_size_t* request_readiness_group_keys,
    iree_host_size_t request_index) {
  return request_readiness_group_keys
             ? request_readiness_group_keys[request_index]
             : ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE;
}

static iree_status_t
id4_pipeline_program_plan_parameter_load_record_readiness_group_key(
    const iree_host_size_t* request_readiness_group_keys,
    const id4_pipeline_program_plan_parameter_load_record_t* record,
    iree_host_size_t* out_readiness_group_key) {
  if (!request_readiness_group_keys) {
    *out_readiness_group_key = ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE;
    return iree_ok_status();
  }
  if (record->request_count == 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "parameter load record at request %" PRIhsz
                            " has no requests",
                            record->request_offset);
  }
  const iree_host_size_t readiness_group_key =
      id4_pipeline_program_plan_parameter_readiness_group_key(
          request_readiness_group_keys, record->request_offset);
  for (iree_host_size_t i = 1; i < record->request_count; ++i) {
    const iree_host_size_t request_index = record->request_offset + i;
    const iree_host_size_t span_readiness_group_key =
        id4_pipeline_program_plan_parameter_readiness_group_key(
            request_readiness_group_keys, request_index);
    if (span_readiness_group_key != readiness_group_key) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "parameter load record at request %" PRIhsz
                              " has mixed readiness groups across source spans",
                              record->request_offset);
    }
  }
  *out_readiness_group_key = readiness_group_key;
  return iree_ok_status();
}

static bool
id4_pipeline_program_plan_parameter_load_step_precedes_for_submission(
    const id4_pipeline_parameter_load_step_t* lhs,
    const id4_pipeline_parameter_load_step_t* rhs) {
  const iree_host_size_t lhs_key = lhs->readiness_group_key;
  const iree_host_size_t rhs_key = rhs->readiness_group_key;
  if (lhs_key == rhs_key) return true;
  if (lhs_key == ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE) {
    return false;
  }
  if (rhs_key == ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE) {
    return true;
  }
  return lhs_key < rhs_key;
}

static void id4_pipeline_program_plan_order_parameter_load_steps_for_submission(
    iree_host_size_t load_step_count,
    id4_pipeline_parameter_load_step_t* load_steps) {
  for (iree_host_size_t i = 1; i < load_step_count; ++i) {
    id4_pipeline_parameter_load_step_t step = load_steps[i];
    iree_host_size_t j = i;
    while (
        j > 0 &&
        !id4_pipeline_program_plan_parameter_load_step_precedes_for_submission(
            &load_steps[j - 1], &step)) {
      load_steps[j] = load_steps[j - 1];
      --j;
    }
    load_steps[j] = step;
  }
}

static iree_status_t id4_pipeline_program_plan_build_parameter_load_steps(
    iree_host_size_t parameter_count,
    const id4_pipeline_program_plan_parameter_load_record_t* load_records,
    const iree_host_size_t* request_readiness_group_keys,
    id4_pipeline_parameter_load_step_t* load_steps,
    iree_host_size_t* direct_request_indices,
    iree_host_size_t* out_load_step_count) {
  *out_load_step_count = 0;
  if (parameter_count == 0) return iree_ok_status();

  for (iree_host_size_t i = 0; i < parameter_count; ++i) {
    const id4_pipeline_program_plan_parameter_load_record_t* record =
        &load_records[i];
    iree_host_size_t readiness_group_key = 0;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_plan_parameter_load_record_readiness_group_key(
            request_readiness_group_keys, record, &readiness_group_key));
    switch (record->encoding) {
      case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT:
        continue;
      case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16:
        load_steps[*out_load_step_count] =
            id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
                IREE_SV("parameters.encode_fp8_e4m3_scaled_to_bf16"),
                record->source_count, record->sources,
                /*target_slab_index=*/0,
                /*request_offset=*/record->request_offset);
        load_steps[*out_load_step_count].readiness_group_key =
            readiness_group_key;
        break;
      case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE:
        load_steps[*out_load_step_count] =
            id4_pipeline_parameter_encode_bf16_linear_rhs_tile_load_step(
                IREE_SV("parameters.encode_bf16_linear_rhs_tile"),
                record->source_count, record->sources,
                /*target_slab_index=*/0,
                /*request_offset=*/record->request_offset);
        load_steps[*out_load_step_count].readiness_group_key =
            readiness_group_key;
        break;
      case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE:
        load_steps[*out_load_step_count] =
            id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile_load_step(
                IREE_SV("parameters.encode_fp8_e4m3_scaled_to_bf16_linear_rhs_"
                        "tile"),
                record->source_count, record->sources,
                /*target_slab_index=*/0,
                /*request_offset=*/record->request_offset);
        load_steps[*out_load_step_count].readiness_group_key =
            readiness_group_key;
        break;
      case ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE:
        load_steps[*out_load_step_count] =
            id4_pipeline_parameter_encode_fp8_e4m3_linear_rhs_tile_load_step(
                IREE_SV("parameters.encode_fp8_e4m3_linear_rhs_tile"),
                record->source_count, record->sources,
                /*target_slab_index=*/0,
                /*request_offset=*/record->request_offset);
        load_steps[*out_load_step_count].readiness_group_key =
            readiness_group_key;
        break;
      default:
        continue;
    }
    ++*out_load_step_count;
  }

  iree_host_size_t direct_request_index_count = 0;
  for (iree_host_size_t i = 0; i < parameter_count; ++i) {
    const id4_pipeline_program_plan_parameter_load_record_t* record =
        &load_records[i];
    if (record->encoding != ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT) {
      continue;
    }
    iree_host_size_t readiness_group_key = 0;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_plan_parameter_load_record_readiness_group_key(
            request_readiness_group_keys, record, &readiness_group_key));
    bool source_scope_already_planned = false;
    for (iree_host_size_t j = 0; j < i; ++j) {
      const id4_pipeline_program_plan_parameter_load_record_t* previous =
          &load_records[j];
      iree_host_size_t previous_readiness_group_key = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_program_plan_parameter_load_record_readiness_group_key(
              request_readiness_group_keys, previous,
              &previous_readiness_group_key));
      if (previous->encoding ==
              ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT &&
          previous_readiness_group_key == readiness_group_key &&
          iree_string_view_equal(previous->source_scope,
                                 record->source_scope)) {
        source_scope_already_planned = true;
        break;
      }
    }
    if (source_scope_already_planned) {
      continue;
    }

    const iree_host_size_t request_index_start = direct_request_index_count;
    for (iree_host_size_t j = i; j < parameter_count; ++j) {
      const id4_pipeline_program_plan_parameter_load_record_t* candidate =
          &load_records[j];
      iree_host_size_t candidate_readiness_group_key = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_program_plan_parameter_load_record_readiness_group_key(
              request_readiness_group_keys, candidate,
              &candidate_readiness_group_key));
      if (candidate->encoding ==
              ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT &&
          candidate_readiness_group_key == readiness_group_key &&
          iree_string_view_equal(candidate->source_scope,
                                 record->source_scope)) {
        for (iree_host_size_t k = 0; k < candidate->request_count; ++k) {
          direct_request_indices[direct_request_index_count++] =
              candidate->request_offset + k;
        }
      }
    }

    load_steps[*out_load_step_count] = id4_pipeline_parameter_gather_load_step(
        IREE_SV("parameters.gather"), record->source_scope,
        /*target_slab_index=*/0, direct_request_indices[request_index_start],
        direct_request_index_count - request_index_start);
    load_steps[*out_load_step_count].readiness_group_key = readiness_group_key;
    for (iree_host_size_t j = request_index_start;
         j < direct_request_index_count; ++j) {
      const iree_host_size_t expected_request_index =
          direct_request_indices[request_index_start] + j - request_index_start;
      if (direct_request_indices[j] != expected_request_index) {
        load_steps[*out_load_step_count] =
            id4_pipeline_parameter_indexed_gather_load_step(
                IREE_SV("parameters.gather"), record->source_scope,
                /*target_slab_index=*/0,
                direct_request_index_count - request_index_start,
                &direct_request_indices[request_index_start]);
        load_steps[*out_load_step_count].readiness_group_key =
            readiness_group_key;
        break;
      }
    }
    ++*out_load_step_count;
  }

  id4_pipeline_program_plan_order_parameter_load_steps_for_submission(
      *out_load_step_count, load_steps);
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_plan_build_parameter_request_ranges_by_tensor(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts,
    const id4_pipeline_program_plan_parameter_request_range_t* request_ranges,
    id4_pipeline_program_plan_parameter_request_range_t*
        request_ranges_by_tensor) {
  const iree_host_size_t tensor_count =
      id4_pipeline_program_tensor_count(options->program);
  for (iree_host_size_t i = 0; i < tensor_count; ++i) {
    request_ranges_by_tensor[i] =
        (id4_pipeline_program_plan_parameter_request_range_t){
            // No request range is assigned to this tensor.
            .request_offset = IREE_HOST_SIZE_MAX,
            // No requests are assigned to this tensor.
            .request_count = 0,
        };
  }

  iree_host_size_t parameter_index = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) continue;
    const uint32_t tensor_ordinal = op->payload.parameter.tensor.ordinal;
    if ((iree_host_size_t)tensor_ordinal >= tensor_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program parameter tensor %u is missing",
                              tensor_ordinal);
    }
    if (request_ranges_by_tensor[tensor_ordinal].request_offset !=
        IREE_HOST_SIZE_MAX) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "program parameter tensor %u has multiple request ordinals",
          tensor_ordinal);
    }
    request_ranges_by_tensor[tensor_ordinal] = request_ranges[parameter_index];
    ++parameter_index;
  }
  if (parameter_index != counts.parameter_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "program parameter count %" PRIhsz
        " does not match counted parameter operations %" PRIhsz,
        parameter_index, counts.parameter_count);
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_plan_mark_parameter_tensor_first_reader(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_tensor_t tensor,
    const id4_pipeline_program_plan_parameter_request_range_t*
        parameter_request_ranges_by_tensor,
    iree_host_size_t region_index,
    iree_host_size_t* parameter_request_readiness_group_keys) {
  const iree_host_size_t tensor_count =
      id4_pipeline_program_tensor_count(options->program);
  if ((iree_host_size_t)tensor.ordinal >= tensor_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program tensor %u is missing", tensor.ordinal);
  }
  const id4_pipeline_program_plan_parameter_request_range_t range =
      parameter_request_ranges_by_tensor[tensor.ordinal];
  if (range.request_offset == IREE_HOST_SIZE_MAX) return iree_ok_status();
  for (iree_host_size_t i = 0; i < range.request_count; ++i) {
    const iree_host_size_t request_index = range.request_offset + i;
    if (parameter_request_readiness_group_keys[request_index] ==
        ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE) {
      parameter_request_readiness_group_keys[request_index] = region_index;
    }
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_plan_mark_parameter_first_readers_for_op(
    const id4_pipeline_program_plan_options_t* options,
    const id4_pipeline_program_op_t* op,
    const id4_pipeline_program_plan_parameter_request_range_t*
        parameter_request_ranges_by_tensor,
    iree_host_size_t region_index,
    iree_host_size_t* parameter_request_readiness_group_keys) {
  if (!op) return iree_ok_status();
  switch (op->kind) {
    case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
      for (iree_host_size_t i = 0; i < op->payload.dispatch_loom.binding_count;
           ++i) {
        const id4_pipeline_program_dispatch_binding_t* binding =
            &op->payload.dispatch_loom.bindings[i];
        if (!iree_any_bit_set(binding->access,
                              ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ)) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            id4_pipeline_program_plan_mark_parameter_tensor_first_reader(
                options, binding->tensor, parameter_request_ranges_by_tensor,
                region_index, parameter_request_readiness_group_keys));
      }
      return iree_ok_status();
    case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
      if (!id4_pipeline_program_plan_tap_name_requested(options,
                                                        op->payload.tap.name)) {
        return iree_ok_status();
      }
      return id4_pipeline_program_plan_mark_parameter_tensor_first_reader(
          options, op->payload.tap.tensor, parameter_request_ranges_by_tensor,
          region_index, parameter_request_readiness_group_keys);
    default:
      return iree_ok_status();
  }
}

static iree_status_t
id4_pipeline_program_plan_build_parameter_request_readiness_group_keys(
    const id4_pipeline_program_plan_options_t* options,
    iree_host_size_t parameter_request_count, iree_host_size_t range_count,
    const id4_pipeline_program_plan_region_range_t* ranges,
    const id4_pipeline_program_plan_parameter_request_range_t*
        parameter_request_ranges_by_tensor,
    iree_host_size_t* parameter_request_readiness_group_keys) {
  for (iree_host_size_t i = 0; i < parameter_request_count; ++i) {
    parameter_request_readiness_group_keys[i] =
        ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE;
  }
  for (iree_host_size_t region_index = 0; region_index < range_count;
       ++region_index) {
    const id4_pipeline_program_plan_region_range_t* range =
        &ranges[region_index];
    const iree_host_size_t operation_limit =
        range->source_operation_offset + range->source_operation_count;
    for (iree_host_size_t operation_index = range->source_operation_offset;
         operation_index < operation_limit; ++operation_index) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(options->program, operation_index);
      IREE_RETURN_IF_ERROR(
          id4_pipeline_program_plan_mark_parameter_first_readers_for_op(
              options, op, parameter_request_ranges_by_tensor, region_index,
              parameter_request_readiness_group_keys));
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_assign_request_load_group(
    iree_host_size_t parameter_request_count, iree_host_size_t request_index,
    iree_host_size_t group_index, iree_host_size_t* request_load_groups) {
  if (request_index >= parameter_request_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step request index %" PRIhsz
                            " exceeds parameter request count %" PRIhsz,
                            request_index, parameter_request_count);
  }
  if (request_load_groups[request_index] != IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "parameter request %" PRIhsz
                            " is assigned to multiple load groups",
                            request_index);
  }
  request_load_groups[request_index] = group_index;
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_plan_build_parameter_request_load_groups(
    iree_host_size_t parameter_request_count, iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_host_size_t* request_load_groups,
    iree_host_size_t* out_load_group_count) {
  *out_load_group_count = 0;
  for (iree_host_size_t i = 0; i < parameter_request_count; ++i) {
    request_load_groups[i] = IREE_HOST_SIZE_MAX;
  }

  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_group_count(
      load_step_count, load_steps, out_load_group_count));
  for (iree_host_size_t group_index = 0; group_index < *out_load_group_count;
       ++group_index) {
    id4_pipeline_parameter_load_group_t group;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_group_at(
        load_step_count, load_steps, group_index, &group));
    for (iree_host_size_t step_ordinal = 0; step_ordinal < group.step_count;
         ++step_ordinal) {
      const id4_pipeline_parameter_load_step_t* step =
          &load_steps[group.step_offset + step_ordinal];
      for (iree_host_size_t request_ordinal = 0;
           request_ordinal < step->request_count; ++request_ordinal) {
        const iree_host_size_t request_index =
            step->request_indices ? step->request_indices[request_ordinal]
                                  : step->request_offset + request_ordinal;
        IREE_RETURN_IF_ERROR(
            id4_pipeline_program_plan_assign_request_load_group(
                parameter_request_count, request_index, group_index,
                request_load_groups));
      }
    }
  }
  for (iree_host_size_t i = 0; i < parameter_request_count; ++i) {
    if (request_load_groups[i] == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "parameter request %" PRIhsz " has no load group",
                              i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_mark_parameter_tensor_read(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_tensor_t tensor,
    const id4_pipeline_program_plan_parameter_request_range_t*
        parameter_request_ranges_by_tensor,
    const iree_host_size_t* parameter_load_groups_by_request,
    iree_host_size_t parameter_load_group_count,
    bool* parameter_load_group_used) {
  const iree_host_size_t tensor_count =
      id4_pipeline_program_tensor_count(options->program);
  if ((iree_host_size_t)tensor.ordinal >= tensor_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program tensor %u is missing", tensor.ordinal);
  }
  const id4_pipeline_program_plan_parameter_request_range_t range =
      parameter_request_ranges_by_tensor[tensor.ordinal];
  if (range.request_offset == IREE_HOST_SIZE_MAX) return iree_ok_status();

  for (iree_host_size_t i = 0; i < range.request_count; ++i) {
    const iree_host_size_t request_index = range.request_offset + i;
    const iree_host_size_t group_index =
        parameter_load_groups_by_request[request_index];
    if (group_index >= parameter_load_group_count) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "parameter request %" PRIhsz " load group %" PRIhsz
          " exceeds load group count %" PRIhsz,
          request_index, group_index, parameter_load_group_count);
    }
    parameter_load_group_used[group_index] = true;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_mark_parameter_reads_for_op(
    const id4_pipeline_program_plan_options_t* options,
    const id4_pipeline_program_op_t* op,
    const id4_pipeline_program_plan_parameter_request_range_t*
        parameter_request_ranges_by_tensor,
    const iree_host_size_t* parameter_load_groups_by_request,
    iree_host_size_t parameter_load_group_count,
    bool* parameter_load_group_used) {
  if (!op) return iree_ok_status();
  switch (op->kind) {
    case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
      for (iree_host_size_t i = 0; i < op->payload.dispatch_loom.binding_count;
           ++i) {
        const id4_pipeline_program_dispatch_binding_t* binding =
            &op->payload.dispatch_loom.bindings[i];
        if (!iree_any_bit_set(binding->access,
                              ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ)) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            id4_pipeline_program_plan_mark_parameter_tensor_read(
                options, binding->tensor, parameter_request_ranges_by_tensor,
                parameter_load_groups_by_request, parameter_load_group_count,
                parameter_load_group_used));
      }
      return iree_ok_status();
    case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
      if (!id4_pipeline_program_plan_tap_name_requested(options,
                                                        op->payload.tap.name)) {
        return iree_ok_status();
      }
      return id4_pipeline_program_plan_mark_parameter_tensor_read(
          options, op->payload.tap.tensor, parameter_request_ranges_by_tensor,
          parameter_load_groups_by_request, parameter_load_group_count,
          parameter_load_group_used);
    default:
      return iree_ok_status();
  }
}

static iree_status_t
id4_pipeline_program_plan_build_region_parameter_load_groups(
    const id4_pipeline_program_plan_options_t* options,
    iree_host_size_t range_count,
    const id4_pipeline_program_plan_region_range_t* ranges,
    const id4_pipeline_program_plan_parameter_request_range_t*
        parameter_request_ranges_by_tensor,
    const iree_host_size_t* parameter_load_groups_by_request,
    iree_host_size_t parameter_load_group_count,
    iree_host_size_t* region_parameter_load_group_counts,
    iree_host_size_t** region_parameter_load_groups,
    iree_allocator_t host_allocator) {
  if (parameter_load_group_count == 0 || range_count == 0) {
    return iree_ok_status();
  }

  bool* parameter_load_group_used = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(host_allocator, parameter_load_group_count,
                                  sizeof(parameter_load_group_used[0]),
                                  (void**)&parameter_load_group_used));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t region_index = 0;
       region_index < range_count && iree_status_is_ok(status);
       ++region_index) {
    memset(parameter_load_group_used, 0,
           parameter_load_group_count * sizeof(parameter_load_group_used[0]));
    const id4_pipeline_program_plan_region_range_t* range =
        &ranges[region_index];
    const iree_host_size_t operation_limit =
        range->source_operation_offset + range->source_operation_count;
    for (iree_host_size_t operation_index = range->source_operation_offset;
         operation_index < operation_limit && iree_status_is_ok(status);
         ++operation_index) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(options->program, operation_index);
      status = id4_pipeline_program_plan_mark_parameter_reads_for_op(
          options, op, parameter_request_ranges_by_tensor,
          parameter_load_groups_by_request, parameter_load_group_count,
          parameter_load_group_used);
    }
    if (!iree_status_is_ok(status)) break;

    iree_host_size_t group_count = 0;
    for (iree_host_size_t i = 0; i < parameter_load_group_count; ++i) {
      if (parameter_load_group_used[i]) ++group_count;
    }
    region_parameter_load_group_counts[region_index] = group_count;
    if (group_count == 0) continue;

    iree_host_size_t* groups = NULL;
    status = iree_allocator_malloc_array(host_allocator, group_count,
                                         sizeof(groups[0]), (void**)&groups);
    if (!iree_status_is_ok(status)) break;
    region_parameter_load_groups[region_index] = groups;
    for (iree_host_size_t i = 0, group_ordinal = 0;
         i < parameter_load_group_count; ++i) {
      if (parameter_load_group_used[i]) groups[group_ordinal++] = i;
    }
  }

  iree_allocator_free(host_allocator, parameter_load_group_used);
  return status;
}

static iree_status_t id4_pipeline_program_plan_build_constant_requests(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts,
    id4_pipeline_constant_request_t* requests,
    iree_device_size_t* out_constant_slab_byte_length) {
  *out_constant_slab_byte_length = 0;
  if (counts.constant_count == 0) return iree_ok_status();

  iree_host_size_t request_index = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT) continue;
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(options->program,
                                       op->payload.constant.tensor.ordinal);
    if (!tensor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program constant tensor %u is missing",
                              op->payload.constant.tensor.ordinal);
    }
    iree_io_parameter_span_t span;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_pack_span(
        tensor->byte_length, options->constant_request_alignment,
        out_constant_slab_byte_length, &span));
    requests[request_index] = (id4_pipeline_constant_request_t){
        // Constant tensor diagnostic name.
        .name = tensor->name,
        // Packed target span inside the constant slab.
        .span = span,
    };
    ++request_index;
  }
  return iree_ok_status();
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

static iree_status_t id4_pipeline_program_plan_validate_diagnostic_tap_names(
    const id4_pipeline_program_plan_options_t* options) {
  if (!id4_pipeline_program_plan_captures_diagnostic_taps(options)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
    iree_string_view_t requested_name = options->diagnostic_tap_names.values[i];
    iree_host_size_t match_count = 0;
    const iree_host_size_t operation_count =
        id4_pipeline_program_operation_count(options->program);
    for (iree_host_size_t j = 0; j < operation_count; ++j) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(options->program, j);
      if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_TAP) continue;
      if (iree_string_view_equal(requested_name, op->payload.tap.name)) {
        ++match_count;
      }
    }
    if (match_count == 0) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "program diagnostic tap `%.*s` was not found",
                              (int)requested_name.size, requested_name.data);
    }
    if (match_count > 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program diagnostic tap `%.*s` is ambiguous",
                              (int)requested_name.size, requested_name.data);
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
    if (counts.constant_count != 0 &&
        binding_slot == options->constant_slab_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program boundary binding slot must not match "
                              "the constant slab binding slot");
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
  for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
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
    if (counts.constant_count != 0 &&
        binding_slot == options->constant_slab_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program diagnostic tap binding slot must not "
                              "match the constant slab binding slot");
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

static iree_status_t id4_pipeline_program_plan_validate_shared_slot(
    const id4_pipeline_program_plan_options_t* options,
    id4_pipeline_program_plan_counts_t counts) {
  const uint32_t binding_slot = options->region_shared_binding_slot;
  if (binding_slot >= options->region_binding_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program shared binding slot %u exceeds region "
                            "binding capacity %" PRIhsz,
                            binding_slot, options->region_binding_capacity);
  }
  if (binding_slot == options->region_local_binding_slot) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program shared binding slot must not match the "
                            "local slab binding slot");
  }
  if (counts.parameter_count != 0 &&
      binding_slot == options->parameter_slab_binding_slot) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program shared binding slot must not match the "
                            "parameter slab binding slot");
  }
  if (counts.constant_count != 0 &&
      binding_slot == options->constant_slab_binding_slot) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program shared binding slot must not match the "
                            "constant slab binding slot");
  }
  for (iree_host_size_t i = 0; i < counts.import_count; ++i) {
    if (i > UINT32_MAX ||
        options->region_boundary_binding_slot_base > UINT32_MAX - (uint32_t)i) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program boundary binding slot overflow");
    }
    const uint32_t boundary_slot =
        options->region_boundary_binding_slot_base + (uint32_t)i;
    if (binding_slot == boundary_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program shared binding slot must not match "
                              "boundary binding slot %u",
                              boundary_slot);
    }
  }
  if (!id4_pipeline_program_plan_captures_diagnostic_taps(options)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
    if (i > UINT32_MAX ||
        options->diagnostic_tap_binding_slot_base > UINT32_MAX - (uint32_t)i) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program diagnostic tap binding slot overflow");
    }
    const uint32_t tap_slot =
        options->diagnostic_tap_binding_slot_base + (uint32_t)i;
    if (binding_slot == tap_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program shared binding slot must not match "
                              "diagnostic tap binding slot %u",
                              tap_slot);
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
    iree_host_size_t region_range_count,
    const id4_pipeline_program_plan_region_range_t* region_ranges,
    id4_pipeline_diagnostic_tap_plan_t* taps) {
  if (!id4_pipeline_program_plan_captures_diagnostic_taps(options) ||
      counts.tap_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t tap_index = 0;
  for (iree_host_size_t region_index = 0; region_index < region_range_count;
       ++region_index) {
    if (region_index > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "program region index %" PRIhsz " exceeds uint32_t", region_index);
    }
    const id4_pipeline_program_plan_region_range_t* range =
        &region_ranges[region_index];
    const iree_host_size_t operation_limit =
        range->source_operation_offset + range->source_operation_count;
    iree_host_size_t region_operation_count = 0;
    for (iree_host_size_t i = range->source_operation_offset;
         i < operation_limit; ++i) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(options->program, i);
      if (!op) continue;
      switch (op->kind) {
        case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
        case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
          ++region_operation_count;
          break;
        case ID4_PIPELINE_PROGRAM_OP_KIND_TAP: {
          if (!id4_pipeline_program_plan_tap_name_requested(
                  options, op->payload.tap.name)) {
            break;
          }
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
          taps[tap_index].region_id = (uint32_t)region_index;
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
  }
  if (tap_index != options->diagnostic_tap_names.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "program planned %" PRIhsz
                            " diagnostic taps but expected %" PRIhsz,
                            tap_index, options->diagnostic_tap_names.count);
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
  const id4_pipeline_program_plan_parameter_request_range_t request_range =
      context->parameter_request_ranges[parameter_ordinal];
  const id4_pipeline_parameter_request_t* request =
      &context->parameter_requests[request_range.request_offset];
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

static iree_status_t id4_pipeline_program_plan_resolve_constant(
    void* user_data, const id4_pipeline_program_constant_op_t* constant_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t constant_ordinal,
    id4_pipeline_tensor_import_t* out_import) {
  (void)constant_op;
  id4_pipeline_program_plan_lowering_context_t* context =
      (id4_pipeline_program_plan_lowering_context_t*)user_data;
  const id4_pipeline_constant_request_t* request =
      &context->constant_requests[constant_ordinal];
  out_import->layout = (id4_pipeline_tensor_layout_t){
      // Constant tensor diagnostic name.
      .name = tensor->name,
      // Constant tensor element type.
      .dtype = id4_pipeline_program_region_convert_dtype(tensor->dtype),
      // Constant tensor shape.
      .shape = id4_pipeline_program_plan_convert_shape(tensor->shape),
      // Dense constant tensor byte length.
      .byte_length = tensor->byte_length,
      // Required subrange alignment in the packed slab.
      .alignment = context->options->constant_request_alignment,
  };
  out_import->binding_slot = context->options->constant_slab_binding_slot;
  out_import->offset = request->span.buffer_offset;
  out_import->flags = ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_plan_resolve_shared_tensor(
    void* user_data, const id4_pipeline_program_acquire_op_t* acquire_op,
    const id4_pipeline_program_tensor_record_t* tensor, bool* out_is_shared,
    id4_pipeline_tensor_import_t* out_import) {
  (void)tensor;
  *out_is_shared = false;
  memset(out_import, 0, sizeof(*out_import));
  id4_pipeline_program_plan_lowering_context_t* context =
      (id4_pipeline_program_plan_lowering_context_t*)user_data;
  for (iree_host_size_t i = 0; i < context->shared_tensor_count; ++i) {
    const id4_pipeline_shared_tensor_plan_t* shared_tensor =
        &context->shared_tensors[i];
    if (shared_tensor->program_tensor_ordinal != acquire_op->tensor.ordinal) {
      continue;
    }
    *out_is_shared = true;
    out_import->layout = shared_tensor->layout;
    out_import->binding_slot = context->options->region_shared_binding_slot;
    out_import->offset = shared_tensor->offset;
    out_import->flags = 0;
    return iree_ok_status();
  }
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
    const id4_pipeline_program_plan_region_range_t* range,
    id4_pipeline_program_plan_counts_t counts,
    iree_host_size_t shared_tensor_count,
    const id4_pipeline_shared_tensor_plan_t* shared_tensors,
    const id4_pipeline_parameter_request_t* parameter_requests,
    const id4_pipeline_program_plan_parameter_request_range_t*
        parameter_request_ranges,
    const id4_pipeline_constant_request_t* constant_requests,
    const id4_pipeline_boundary_tensor_plan_t* boundary_tensors,
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_taps,
    iree_allocator_t host_allocator,
    id4_pipeline_region_statistics_t* out_statistics,
    iree_host_size_t* out_local_lifetime_count,
    id4_pipeline_region_local_lifetime_t** out_local_lifetimes) {
  (void)counts;
  memset(out_statistics, 0, sizeof(*out_statistics));
  *out_local_lifetime_count = 0;
  *out_local_lifetimes = NULL;
  if (range->region_operation_count == 0) return iree_ok_status();

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096, host_allocator,
                                   &block_pool);

  id4_pipeline_region_builder_t* builder = NULL;
  id4_pipeline_region_builder_create_options_t builder_options = {
      // Size of this structure for versioning.
      .structure_size = sizeof(builder_options),
      // Region name copied by the builder.
      .region_name = range->name,
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
      // Shared tensor table.
      .shared_tensors = shared_tensors,
      // Number of shared tensor records.
      .shared_tensor_count = shared_tensor_count,
      // Parameter request table.
      .parameter_requests = parameter_requests,
      // Parameter request ranges in program parameter-operation order.
      .parameter_request_ranges = parameter_request_ranges,
      // Constant request table.
      .constant_requests = constant_requests,
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
        // Dry-run emits the selected source-program interval.
        .source_operation_offset = range->source_operation_offset,
        // Dry-run emits the selected source-program interval.
        .source_operation_count = range->source_operation_count,
        // Dry-run region builder.
        .builder = builder,
        // Diagnostic tap lowering policy.
        .tap_mode = id4_pipeline_program_plan_captures_diagnostic_taps(options)
                        ? ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_CAPTURE
                        : ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_IGNORE,
        // Selected diagnostic tap names.
        .captured_tap_names = options->diagnostic_tap_names,
        // Required local tensor alignment.
        .local_tensor_alignment = options->region_local_tensor_alignment,
        // Planner resolver context.
        .user_data = &lowering_context,
        // Resolves boundary tensor imports.
        .resolve_import = id4_pipeline_program_plan_resolve_import,
        // Resolves parameter tensor imports.
        .resolve_parameter = id4_pipeline_program_plan_resolve_parameter,
        // Resolves constant tensor imports.
        .resolve_constant = id4_pipeline_program_plan_resolve_constant,
        // Resolves plan-shared acquired tensor imports.
        .resolve_shared_tensor =
            id4_pipeline_program_plan_resolve_shared_tensor,
        // Resolves diagnostic tap imports.
        .resolve_tap = id4_pipeline_program_plan_resolve_tap,
    };
    status = id4_pipeline_program_region_lower(&lower_options, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_builder_statistics(builder, out_statistics);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_region_builder_clone_local_lifetimes(
        builder, host_allocator, out_local_lifetime_count, out_local_lifetimes);
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

static iree_status_t id4_pipeline_program_plan_make_constant_slab_name(
    iree_string_view_t program_name, iree_allocator_t host_allocator,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_string_builder_append_string(&builder, program_name);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, ".constants");
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t name_size = iree_string_builder_size(&builder);
    char* storage = iree_string_builder_take_storage(&builder);
    *out_name = iree_make_string_view(storage, name_size);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t id4_pipeline_program_plan_make_shared_slab_name(
    iree_string_view_t program_name, iree_allocator_t host_allocator,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_string_builder_append_string(&builder, program_name);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, ".shared");
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
      id4_pipeline_program_plan_count_ops(options);
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_plan_validate_exports(options->program));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_plan_validate_diagnostic_tap_names(options));
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
  if (counts.constant_count != 0) {
    if (options->constant_slab_binding_slot >=
        options->region_binding_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program constant slab binding slot %u exceeds "
                              "region binding capacity %" PRIhsz,
                              options->constant_slab_binding_slot,
                              options->region_binding_capacity);
    }
    if (options->constant_slab_binding_slot ==
        options->region_local_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program constant slab binding slot must not "
                              "match the local slab binding slot");
    }
    if (counts.parameter_count != 0 &&
        options->constant_slab_binding_slot ==
            options->parameter_slab_binding_slot) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program constant slab binding slot must not "
                              "match the parameter slab binding slot");
    }
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_plan_validate_boundary_slots(options, counts));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_plan_validate_diagnostic_tap_slots(options, counts));

  id4_pipeline_parameter_request_t* parameter_requests = NULL;
  id4_pipeline_parameter_tensor_plan_t* parameter_tensors = NULL;
  id4_pipeline_program_plan_parameter_load_record_t* parameter_load_records =
      NULL;
  id4_pipeline_parameter_load_source_t* parameter_load_sources = NULL;
  id4_pipeline_parameter_load_step_t* parameter_load_steps = NULL;
  iree_host_size_t* parameter_load_step_request_indices = NULL;
  id4_pipeline_program_plan_parameter_request_range_t*
      parameter_request_ranges = NULL;
  id4_pipeline_program_plan_parameter_request_range_t*
      parameter_request_ranges_by_tensor = NULL;
  iree_host_size_t* parameter_request_readiness_group_keys = NULL;
  iree_host_size_t* parameter_load_groups_by_request = NULL;
  id4_pipeline_constant_request_t* constant_requests = NULL;
  id4_pipeline_boundary_tensor_plan_t* boundary_tensors = NULL;
  id4_pipeline_kernel_plan_t* kernels = NULL;
  id4_pipeline_diagnostic_tap_plan_t* taps = NULL;
  id4_pipeline_program_plan_region_range_t* region_ranges = NULL;
  id4_pipeline_program_plan_shared_tensor_record_t* shared_tensor_records =
      NULL;
  id4_pipeline_shared_tensor_plan_t* shared_tensors = NULL;
  id4_pipeline_region_statistics_t* region_statistics = NULL;
  iree_host_size_t* region_local_lifetime_counts = NULL;
  id4_pipeline_region_local_lifetime_t** region_local_lifetimes = NULL;
  iree_host_size_t* region_parameter_load_group_counts = NULL;
  iree_host_size_t** region_parameter_load_groups = NULL;
  id4_pipeline_region_plan_t* regions = NULL;
  id4_pipeline_memory_slab_plan_t* memory_slabs = NULL;
  iree_string_view_t shared_slab_name = iree_string_view_empty();
  iree_string_view_t* local_slab_names = NULL;
  iree_host_size_t parameter_load_step_count = 0;
  iree_host_size_t parameter_load_group_count = 0;
  iree_host_size_t region_range_count = 0;
  iree_host_size_t shared_tensor_count = 0;
  iree_device_size_t shared_slab_byte_length = 0;
  iree_device_size_t shared_slab_high_water_mark = 0;
  const iree_host_size_t planned_tap_count =
      id4_pipeline_program_plan_captures_diagnostic_taps(options)
          ? options->diagnostic_tap_names.count
          : 0;
  iree_status_t status = iree_ok_status();
  if (counts.region_operation_count != 0) {
    const iree_host_size_t region_range_capacity =
        id4_pipeline_program_plan_uses_dispatch_regions(options) &&
                counts.dispatch_count != 0
            ? counts.region_operation_count
            : counts.region_cut_count + 1;
    status = iree_allocator_malloc_array(host_allocator, region_range_capacity,
                                         sizeof(region_ranges[0]),
                                         (void**)&region_ranges);
  }
  if (iree_status_is_ok(status) && counts.region_operation_count != 0) {
    id4_pipeline_program_plan_build_region_ranges(options, &region_range_count,
                                                  region_ranges);
    if (region_range_count == 0) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "program has executable operations but no "
                                "executable region ranges");
    }
  }
  if (iree_status_is_ok(status) && counts.parameter_request_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.parameter_count,
                                         sizeof(parameter_request_ranges[0]),
                                         (void**)&parameter_request_ranges);
  }
  if (iree_status_is_ok(status) && counts.parameter_request_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, counts.parameter_request_count,
        sizeof(parameter_requests[0]), (void**)&parameter_requests);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.parameter_count,
                                         sizeof(parameter_tensors[0]),
                                         (void**)&parameter_tensors);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.parameter_count,
                                         sizeof(parameter_load_records[0]),
                                         (void**)&parameter_load_records);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    const iree_host_size_t parameter_load_source_capacity =
        counts.parameter_count *
        ID4_PIPELINE_PROGRAM_PARAMETER_MAX_SOURCE_COUNT;
    status = iree_allocator_malloc_array(
        host_allocator, parameter_load_source_capacity,
        sizeof(parameter_load_sources[0]), (void**)&parameter_load_sources);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.parameter_count,
                                         sizeof(parameter_load_steps[0]),
                                         (void**)&parameter_load_steps);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, counts.parameter_request_count,
        sizeof(parameter_load_step_request_indices[0]),
        (void**)&parameter_load_step_request_indices);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, id4_pipeline_program_tensor_count(options->program),
        sizeof(parameter_request_ranges_by_tensor[0]),
        (void**)&parameter_request_ranges_by_tensor);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, counts.parameter_request_count,
        sizeof(parameter_request_readiness_group_keys[0]),
        (void**)&parameter_request_readiness_group_keys);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, counts.parameter_request_count,
        sizeof(parameter_load_groups_by_request[0]),
        (void**)&parameter_load_groups_by_request);
  }
  if (iree_status_is_ok(status) && counts.constant_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, counts.constant_count,
                                         sizeof(constant_requests[0]),
                                         (void**)&constant_requests);
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
  if (iree_status_is_ok(status) && counts.acquire_count != 0 &&
      region_range_count > 1) {
    status = iree_allocator_malloc_array(host_allocator, counts.acquire_count,
                                         sizeof(shared_tensor_records[0]),
                                         (void**)&shared_tensor_records);
  }
  if (iree_status_is_ok(status) && counts.acquire_count != 0 &&
      region_range_count > 1) {
    status = iree_allocator_malloc_array(host_allocator, counts.acquire_count,
                                         sizeof(shared_tensors[0]),
                                         (void**)&shared_tensors);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, region_range_count,
                                         sizeof(region_statistics[0]),
                                         (void**)&region_statistics);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    memset(region_statistics, 0,
           region_range_count * sizeof(region_statistics[0]));
    status =
        iree_allocator_malloc_array(host_allocator, region_range_count,
                                    sizeof(region_local_lifetime_counts[0]),
                                    (void**)&region_local_lifetime_counts);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    memset(region_local_lifetime_counts, 0,
           region_range_count * sizeof(region_local_lifetime_counts[0]));
    status = iree_allocator_malloc_array(host_allocator, region_range_count,
                                         sizeof(region_local_lifetimes[0]),
                                         (void**)&region_local_lifetimes);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    memset(region_local_lifetimes, 0,
           region_range_count * sizeof(region_local_lifetimes[0]));
    status = iree_allocator_malloc_array(
        host_allocator, region_range_count,
        sizeof(region_parameter_load_group_counts[0]),
        (void**)&region_parameter_load_group_counts);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    memset(region_parameter_load_group_counts, 0,
           region_range_count * sizeof(region_parameter_load_group_counts[0]));
    status =
        iree_allocator_malloc_array(host_allocator, region_range_count,
                                    sizeof(region_parameter_load_groups[0]),
                                    (void**)&region_parameter_load_groups);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    memset(region_parameter_load_groups, 0,
           region_range_count * sizeof(region_parameter_load_groups[0]));
    status = iree_allocator_malloc_array(host_allocator, region_range_count,
                                         sizeof(regions[0]), (void**)&regions);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    memset(regions, 0, region_range_count * sizeof(regions[0]));
    status = iree_allocator_malloc_array(host_allocator, region_range_count + 1,
                                         sizeof(memory_slabs[0]),
                                         (void**)&memory_slabs);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    memset(memory_slabs, 0, (region_range_count + 1) * sizeof(memory_slabs[0]));
    status = iree_allocator_malloc_array(host_allocator, region_range_count,
                                         sizeof(local_slab_names[0]),
                                         (void**)&local_slab_names);
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    memset(local_slab_names, 0,
           region_range_count * sizeof(local_slab_names[0]));
  }

  iree_device_size_t parameter_slab_byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_plan_build_parameter_requests(
        options, counts, parameter_requests, parameter_tensors,
        parameter_request_ranges, parameter_load_records,
        parameter_load_sources, &parameter_slab_byte_length);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = id4_pipeline_program_plan_build_parameter_request_ranges_by_tensor(
        options, counts, parameter_request_ranges,
        parameter_request_ranges_by_tensor);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status =
        id4_pipeline_program_plan_build_parameter_request_readiness_group_keys(
            options, counts.parameter_request_count, region_range_count,
            region_ranges, parameter_request_ranges_by_tensor,
            parameter_request_readiness_group_keys);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = id4_pipeline_program_plan_build_parameter_load_steps(
        counts.parameter_count, parameter_load_records,
        parameter_request_readiness_group_keys, parameter_load_steps,
        parameter_load_step_request_indices, &parameter_load_step_count);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0) {
    status = id4_pipeline_program_plan_build_parameter_request_load_groups(
        counts.parameter_request_count, parameter_load_step_count,
        parameter_load_steps, parameter_load_groups_by_request,
        &parameter_load_group_count);
  }
  if (iree_status_is_ok(status) && counts.parameter_count != 0 &&
      region_range_count != 0) {
    status = id4_pipeline_program_plan_build_region_parameter_load_groups(
        options, region_range_count, region_ranges,
        parameter_request_ranges_by_tensor, parameter_load_groups_by_request,
        parameter_load_group_count, region_parameter_load_group_counts,
        region_parameter_load_groups, host_allocator);
  }

  iree_device_size_t constant_slab_byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_plan_build_constant_requests(
        options, counts, constant_requests, &constant_slab_byte_length);
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
    status = id4_pipeline_program_plan_build_taps(
        options, counts, region_range_count, region_ranges, taps);
  }

  if (iree_status_is_ok(status) && shared_tensor_records) {
    status = id4_pipeline_program_plan_build_shared_tensors(
        options, region_range_count, region_ranges, shared_tensor_records,
        &shared_tensor_count, &shared_slab_byte_length,
        &shared_slab_high_water_mark);
  }
  if (iree_status_is_ok(status) && shared_tensor_count != 0) {
    status = id4_pipeline_program_plan_validate_shared_slot(options, counts);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < shared_tensor_count; ++i) {
      shared_tensors[i] = shared_tensor_records[i].plan;
    }
  }

  if (iree_status_is_ok(status) && region_range_count != 0) {
    for (iree_host_size_t i = 0;
         i < region_range_count && iree_status_is_ok(status); ++i) {
      status = id4_pipeline_program_plan_dry_run_region(
          options, &region_ranges[i], counts, shared_tensor_count,
          shared_tensors, parameter_requests, parameter_request_ranges,
          constant_requests, boundary_tensors, taps, host_allocator,
          &region_statistics[i], &region_local_lifetime_counts[i],
          &region_local_lifetimes[i]);
    }
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
        options->parameter_slab_alignment, counts.parameter_request_count,
        parameter_requests);
  }

  iree_string_view_t constant_slab_name = iree_string_view_empty();
  id4_pipeline_constant_slab_plan_t constant_slab;
  memset(&constant_slab, 0, sizeof(constant_slab));
  if (iree_status_is_ok(status) && counts.constant_count != 0) {
    status = id4_pipeline_program_plan_make_constant_slab_name(
        id4_pipeline_program_name(options->program), host_allocator,
        &constant_slab_name);
  }
  if (iree_status_is_ok(status) && counts.constant_count != 0) {
    constant_slab = (id4_pipeline_constant_slab_plan_t){
        // Human-readable constant slab name.
        .name = constant_slab_name,
        // Constant slab placement selected by the stage.
        .placement_id = options->constant_slab_placement_id,
        // Binding-table slot reserved for embedded constants.
        .binding_slot = options->constant_slab_binding_slot,
        // HAL buffer parameters used for constant slab allocation.
        .target_params = options->constant_slab_target_params,
        // Planned packed constant slab byte length.
        .byte_length = constant_slab_byte_length,
        // Required constant slab base alignment.
        .alignment = options->constant_slab_alignment,
        // Number of embedded constant tensor requests.
        .request_count = counts.constant_count,
        // Constant requests in program operation order.
        .requests = constant_requests,
    };
  }

  iree_host_size_t memory_slab_count = 0;
  if (iree_status_is_ok(status) && shared_tensor_count != 0) {
    status = id4_pipeline_program_plan_make_shared_slab_name(
        id4_pipeline_program_name(options->program), host_allocator,
        &shared_slab_name);
  }
  if (iree_status_is_ok(status) && shared_tensor_count != 0) {
    memory_slabs[memory_slab_count] = (id4_pipeline_memory_slab_plan_t){
        // Human-readable shared transient slab name.
        .name = shared_slab_name,
        // Shared transient slab is visible to every executable region.
        .scope = ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED,
        // Plan-shared slabs are not owned by one region.
        .region_id = 0,
        // Shared transient slab follows the executable region placement.
        .placement_id = options->region_placement_id,
        // Binding-table slot reserved for the shared transient slab.
        .binding_slot = options->region_shared_binding_slot,
        // HAL buffer parameters used for shared transient allocation.
        .params = options->region_local_slab_params,
        // Planned shared slab byte length.
        .byte_length = shared_slab_byte_length,
        // Required shared slab base alignment.
        .alignment = options->region_local_slab_alignment,
        // Peak concurrently live shared tensor bytes.
        .high_water_mark = shared_slab_high_water_mark,
    };
    ++memory_slab_count;
  }
  if (iree_status_is_ok(status) && region_range_count != 0) {
    for (iree_host_size_t i = 0;
         i < region_range_count && iree_status_is_ok(status); ++i) {
      if (i > UINT32_MAX) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "program region index %" PRIhsz " exceeds uint32_t", i);
        break;
      }
      const id4_pipeline_program_plan_region_range_t* range = &region_ranges[i];
      id4_pipeline_region_plan_t* region = &regions[i];
      region->name = range->name;
      region->source_operation_offset = range->source_operation_offset;
      region->source_operation_count = range->source_operation_count;
      region->placement_id = options->region_placement_id;
      region->binding_capacity = options->region_binding_capacity;
      region->local_binding_slot = options->region_local_binding_slot;
      region->local_tensor_alignment = options->region_local_tensor_alignment;
      region->statistics = region_statistics[i];
      region->local_lifetime_count = region_local_lifetime_counts[i];
      region->local_lifetimes = region_local_lifetimes[i];
      region->parameter_load_group_count =
          region_parameter_load_group_counts
              ? region_parameter_load_group_counts[i]
              : 0;
      region->parameter_load_groups =
          region_parameter_load_groups ? region_parameter_load_groups[i] : NULL;
      if (region_statistics[i].local_slab_byte_length == 0) continue;

      status = id4_pipeline_program_plan_make_local_slab_name(
          range->name, host_allocator, &local_slab_names[i]);
      if (!iree_status_is_ok(status)) break;
      memory_slabs[memory_slab_count] = (id4_pipeline_memory_slab_plan_t){
          // Human-readable local slab name.
          .name = local_slab_names[i],
          // Local transient slab is scoped to one executable region.
          .scope = ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL,
          // Executable region owning this local slab.
          .region_id = (uint32_t)i,
          // Local transient slab follows the executable region placement.
          .placement_id = options->region_placement_id,
          // Binding-table slot reserved for the local transient slab.
          .binding_slot = options->region_local_binding_slot,
          // HAL buffer parameters used for local transient allocation.
          .params = options->region_local_slab_params,
          // Planned local slab byte length.
          .byte_length = region_statistics[i].local_slab_byte_length,
          // Required local slab base alignment.
          .alignment = options->region_local_slab_alignment,
          // Peak concurrently live local tensor bytes observed by dry-run.
          .high_water_mark = region_statistics[i].local_slab_high_water_mark,
      };
      ++memory_slab_count;
    }
  }

  if (iree_status_is_ok(status)) {
    id4_pipeline_plan_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.stage_name = options->stage_name;
    create_options.source_program = options->program;
    create_options.device_group = options->device_group;
    create_options.placement_count = options->placement_count;
    create_options.placements = options->placements;
    create_options.parameter_slab_count = counts.parameter_count == 0 ? 0 : 1;
    create_options.parameter_slabs =
        counts.parameter_count == 0 ? NULL : &parameter_slab;
    create_options.parameter_tensor_count = counts.parameter_count;
    create_options.parameter_tensors =
        counts.parameter_count == 0 ? NULL : parameter_tensors;
    create_options.parameter_load_step_count = parameter_load_step_count;
    create_options.parameter_load_steps =
        parameter_load_step_count == 0 ? NULL : parameter_load_steps;
    create_options.constant_slab_count = counts.constant_count == 0 ? 0 : 1;
    create_options.constant_slabs =
        counts.constant_count == 0 ? NULL : &constant_slab;
    create_options.memory_slab_count = memory_slab_count;
    create_options.memory_slabs = memory_slab_count == 0 ? NULL : memory_slabs;
    create_options.shared_tensor_count = shared_tensor_count;
    create_options.shared_tensors =
        shared_tensor_count == 0 ? NULL : shared_tensors;
    create_options.boundary_tensor_count = counts.import_count;
    create_options.boundary_tensors =
        counts.import_count == 0 ? NULL : boundary_tensors;
    create_options.kernel_count = kernel_count;
    create_options.kernels = kernel_count == 0 ? NULL : kernels;
    create_options.region_count = region_range_count;
    create_options.regions = region_range_count == 0 ? NULL : regions;
    create_options.diagnostic_tap_count = planned_tap_count;
    create_options.diagnostic_taps = planned_tap_count == 0 ? NULL : taps;
    create_options.diagnostics_sink = options->diagnostics_sink;
    status =
        id4_pipeline_plan_create(&create_options, host_allocator, out_plan);
  }

  iree_allocator_free(host_allocator, (void*)constant_slab_name.data);
  iree_allocator_free(host_allocator, (void*)shared_slab_name.data);
  for (iree_host_size_t i = 0; i < region_range_count; ++i) {
    if (local_slab_names) {
      iree_allocator_free(host_allocator, (void*)local_slab_names[i].data);
    }
    if (region_parameter_load_groups) {
      iree_allocator_free(host_allocator, region_parameter_load_groups[i]);
    }
    if (region_local_lifetime_counts && region_local_lifetimes) {
      id4_pipeline_region_local_lifetime_list_release(
          region_local_lifetime_counts[i], region_local_lifetimes[i],
          host_allocator);
    }
  }
  id4_pipeline_program_plan_release_specialization_keys(kernel_count, kernels,
                                                        host_allocator);
  iree_allocator_free(host_allocator, local_slab_names);
  iree_allocator_free(host_allocator, memory_slabs);
  iree_allocator_free(host_allocator, regions);
  iree_allocator_free(host_allocator, region_parameter_load_groups);
  iree_allocator_free(host_allocator, region_parameter_load_group_counts);
  iree_allocator_free(host_allocator, region_local_lifetimes);
  iree_allocator_free(host_allocator, region_local_lifetime_counts);
  iree_allocator_free(host_allocator, region_statistics);
  iree_allocator_free(host_allocator, shared_tensors);
  iree_allocator_free(host_allocator, shared_tensor_records);
  iree_allocator_free(host_allocator, region_ranges);
  iree_allocator_free(host_allocator, taps);
  iree_allocator_free(host_allocator, kernels);
  iree_allocator_free(host_allocator, boundary_tensors);
  iree_allocator_free(host_allocator, constant_requests);
  iree_allocator_free(host_allocator, parameter_load_groups_by_request);
  iree_allocator_free(host_allocator, parameter_request_readiness_group_keys);
  iree_allocator_free(host_allocator, parameter_request_ranges_by_tensor);
  iree_allocator_free(host_allocator, parameter_request_ranges);
  iree_allocator_free(host_allocator, parameter_load_step_request_indices);
  iree_allocator_free(host_allocator, parameter_load_steps);
  iree_allocator_free(host_allocator, parameter_load_sources);
  iree_allocator_free(host_allocator, parameter_load_records);
  iree_allocator_free(host_allocator, parameter_tensors);
  iree_allocator_free(host_allocator, parameter_requests);
  return status;
}
