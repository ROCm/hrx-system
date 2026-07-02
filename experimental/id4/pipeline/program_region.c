// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_region.h"

#include <string.h>

typedef struct id4_pipeline_program_region_counts_t {
  // Maximum tensor binding count used by any Loom dispatch operation.
  iree_host_size_t max_dispatch_binding_count;
} id4_pipeline_program_region_counts_t;

typedef struct id4_pipeline_program_region_initialized_ranges_t {
  // Tensor-relative byte ranges initialized by prior program operations.
  id4_pipeline_region_tensor_byte_range_t* values;
  // Number of entries in values.
  iree_host_size_t count;
} id4_pipeline_program_region_initialized_ranges_t;

typedef struct id4_pipeline_program_region_context_t {
  // Program lowering options.
  const id4_pipeline_program_region_lower_options_t* options;
  // Program tensor ordinal to region tensor handle map.
  id4_pipeline_tensor_t* tensor_map;
  // Number of entries in tensor_map.
  iree_host_size_t tensor_map_count;
  // Last semantic operation ordinal that uses each tensor.
  iree_host_size_t* last_use_ordinals;
  // True for tensors introduced by semantic acquire operations.
  uint8_t* local_tensor_bits;
  // True after lowering has released the corresponding local tensor.
  uint8_t* released_tensor_bits;
  // Reusable dispatch binding scratch for one operation.
  id4_pipeline_region_dispatch_binding_t* dispatch_bindings;
  // Host allocator used for temporary specialization keys.
  iree_allocator_t host_allocator;
} id4_pipeline_program_region_context_t;

static iree_status_t id4_pipeline_program_region_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_program_region_validate_options(
    const id4_pipeline_program_region_lower_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region lower options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program region")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program region extension structures are not supported");
  }
  if (!options->program) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region program is required");
  }
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  if (options->source_operation_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region source operation count is zero");
  }
  if (options->source_operation_offset > operation_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program region source operation offset %" PRIhsz
                            " exceeds operation count %" PRIhsz,
                            options->source_operation_offset, operation_count);
  }
  if (options->source_operation_count >
      operation_count - options->source_operation_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program region source operation range [%" PRIhsz ", %" PRIhsz
        ") exceeds operation count %" PRIhsz,
        options->source_operation_offset,
        options->source_operation_offset + options->source_operation_count,
        operation_count);
  }
  if (!options->builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region builder is required");
  }
  if (options->local_tensor_alignment != 0 &&
      !iree_device_size_is_power_of_two(options->local_tensor_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program region local tensor alignment must be a power of two");
  }
  if (id4_pipeline_region_builder_mode(options->builder) ==
          ID4_PIPELINE_REGION_BUILDER_MODE_RECORD &&
      !options->resolve_kernel) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region kernel resolver is required in "
                            "record mode");
  }
  switch (options->tap_mode) {
    case ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_IGNORE:
      if (options->captured_tap_names.count != 0 ||
          options->captured_tap_names.values) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "program region captured tap names require capture mode");
      }
      return iree_ok_status();
    case ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_CAPTURE:
      if (!options->resolve_tap) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "program region tap resolver is required in capture mode");
      }
      if (options->captured_tap_names.count == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "program region capture mode requires captured tap names");
      }
      if (!options->captured_tap_names.values) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "program region capture mode requires a captured tap name list");
      }
      for (iree_host_size_t i = 0; i < options->captured_tap_names.count; ++i) {
        if (iree_string_view_is_empty(options->captured_tap_names.values[i])) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "program region captured tap name %" PRIhsz " is empty", i);
        }
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported program region tap mode %d",
                              (int)options->tap_mode);
  }
}

static bool id4_pipeline_program_region_captures_tap(
    const id4_pipeline_program_region_lower_options_t* options,
    iree_string_view_t name) {
  if (options->tap_mode != ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_CAPTURE) {
    return false;
  }
  for (iree_host_size_t i = 0; i < options->captured_tap_names.count; ++i) {
    if (iree_string_view_equal(options->captured_tap_names.values[i], name)) {
      return true;
    }
  }
  return false;
}

static id4_pipeline_program_region_counts_t
id4_pipeline_program_region_count_ops(
    const id4_pipeline_program_region_lower_options_t* options) {
  id4_pipeline_program_region_counts_t counts;
  memset(&counts, 0, sizeof(counts));
  const iree_host_size_t operation_limit =
      options->source_operation_offset + options->source_operation_count;
  for (iree_host_size_t i = options->source_operation_offset;
       i < operation_limit; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      continue;
    }
    counts.max_dispatch_binding_count =
        iree_max(counts.max_dispatch_binding_count,
                 op->payload.dispatch_loom.binding_count);
  }
  return counts;
}

static bool id4_pipeline_program_region_operation_uses_tensor(
    const id4_pipeline_program_region_lower_options_t* options,
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
      return id4_pipeline_program_region_captures_tap(options,
                                                      op->payload.tap.name) &&
             op->payload.tap.tensor.ordinal == tensor.ordinal;
    case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT:
      return op->payload.export_value.tensor.ordinal == tensor.ordinal;
    default:
      return false;
  }
}

static bool id4_pipeline_program_region_range_uses_tensor(
    const id4_pipeline_program_region_lower_options_t* options,
    id4_pipeline_program_tensor_t tensor) {
  const iree_host_size_t operation_limit =
      options->source_operation_offset + options->source_operation_count;
  for (iree_host_size_t i = options->source_operation_offset;
       i < operation_limit; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (id4_pipeline_program_region_operation_uses_tensor(options, op,
                                                          tensor)) {
      return true;
    }
  }
  return false;
}

static iree_status_t id4_pipeline_program_region_binding_write_range(
    const id4_pipeline_program_dispatch_binding_t* binding,
    const id4_pipeline_program_tensor_record_t* tensor,
    id4_pipeline_program_tensor_byte_range_t* out_range) {
  IREE_ASSERT_ARGUMENT(out_range);
  *out_range = (id4_pipeline_program_tensor_byte_range_t){0, 0};
  if (!iree_all_bits_set(
          binding->flags,
          ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_WRITE_RANGE)) {
    *out_range = (id4_pipeline_program_tensor_byte_range_t){
        // Whole tensor writes start at the tensor base.
        .offset = 0,
        // Whole tensor write byte coverage.
        .length = tensor->byte_length,
    };
    return iree_ok_status();
  }
  iree_device_size_t write_end = 0;
  if (!iree_device_size_checked_add(binding->write_range.offset,
                                    binding->write_range.length, &write_end) ||
      write_end > tensor->byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program tensor %.*s write range [%" PRIu64
                            ", %" PRIu64
                            ") exceeds tensor byte length %" PRIu64,
                            (int)tensor->name.size, tensor->name.data,
                            (uint64_t)binding->write_range.offset,
                            (uint64_t)write_end, (uint64_t)tensor->byte_length);
  }
  if (binding->write_range.length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program tensor %.*s write range is empty",
                            (int)tensor->name.size, tensor->name.data);
  }
  *out_range = binding->write_range;
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_program_region_collect_initialized_write_ranges_before_range(
    const id4_pipeline_program_region_lower_options_t* options,
    id4_pipeline_program_tensor_t tensor,
    const id4_pipeline_program_tensor_record_t* tensor_record,
    iree_allocator_t host_allocator,
    id4_pipeline_program_region_initialized_ranges_t* out_ranges) {
  out_ranges->values = NULL;
  out_ranges->count = 0;

  iree_host_size_t range_count = 0;
  for (iree_host_size_t i = 0; i < options->source_operation_offset; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      continue;
    }
    for (iree_host_size_t j = 0; j < op->payload.dispatch_loom.binding_count;
         ++j) {
      const id4_pipeline_program_dispatch_binding_t* binding =
          &op->payload.dispatch_loom.bindings[j];
      if (binding->tensor.ordinal == tensor.ordinal &&
          iree_all_bits_set(binding->access,
                            ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE)) {
        ++range_count;
      }
    }
  }
  if (range_count == 0) return iree_ok_status();

  id4_pipeline_region_tensor_byte_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, range_count, sizeof(ranges[0]), (void**)&ranges));
  iree_host_size_t range_index = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < options->source_operation_offset && iree_status_is_ok(status); ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      continue;
    }
    for (iree_host_size_t j = 0; j < op->payload.dispatch_loom.binding_count &&
                                 iree_status_is_ok(status);
         ++j) {
      const id4_pipeline_program_dispatch_binding_t* binding =
          &op->payload.dispatch_loom.bindings[j];
      if (binding->tensor.ordinal != tensor.ordinal ||
          !iree_all_bits_set(binding->access,
                             ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE)) {
        continue;
      }
      id4_pipeline_program_tensor_byte_range_t write_range;
      status = id4_pipeline_program_region_binding_write_range(
          binding, tensor_record, &write_range);
      if (!iree_status_is_ok(status)) break;
      ranges[range_index++] = (id4_pipeline_region_tensor_byte_range_t){
          // Byte offset from the start of the logical tensor.
          .offset = write_range.offset,
          // Byte length of the initialized interval.
          .length = write_range.length,
      };
    }
  }
  if (iree_status_is_ok(status)) {
    out_ranges->values = ranges;
    out_ranges->count = range_index;
  } else {
    iree_allocator_free(host_allocator, ranges);
  }
  return status;
}

static iree_status_t
id4_pipeline_program_region_tensor_fully_written_before_range(
    const id4_pipeline_program_region_lower_options_t* options,
    id4_pipeline_program_tensor_t tensor,
    const id4_pipeline_program_tensor_record_t* tensor_record,
    bool* out_fully_written) {
  IREE_ASSERT_ARGUMENT(out_fully_written);
  *out_fully_written = false;

  iree_device_size_t covered_end = 0;
  bool changed = true;
  while (changed && covered_end < tensor_record->byte_length) {
    changed = false;
    for (iree_host_size_t i = 0; i < options->source_operation_offset; ++i) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(options->program, i);
      if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
        continue;
      }
      for (iree_host_size_t j = 0; j < op->payload.dispatch_loom.binding_count;
           ++j) {
        const id4_pipeline_program_dispatch_binding_t* binding =
            &op->payload.dispatch_loom.bindings[j];
        if (binding->tensor.ordinal != tensor.ordinal ||
            !iree_all_bits_set(binding->access,
                               ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE)) {
          continue;
        }
        id4_pipeline_program_tensor_byte_range_t write_range;
        IREE_RETURN_IF_ERROR(id4_pipeline_program_region_binding_write_range(
            binding, tensor_record, &write_range));
        if (write_range.offset > covered_end) continue;
        iree_device_size_t write_end = 0;
        if (!iree_device_size_checked_add(write_range.offset,
                                          write_range.length, &write_end)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "program tensor %.*s write range overflow",
                                  (int)tensor_record->name.size,
                                  tensor_record->name.data);
        }
        if (write_end <= covered_end) continue;
        covered_end = write_end;
        changed = true;
      }
    }
  }
  *out_fully_written = covered_end >= tensor_record->byte_length;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_populate_prior_initialization(
    const id4_pipeline_program_region_lower_options_t* options,
    id4_pipeline_program_tensor_t tensor,
    const id4_pipeline_program_tensor_record_t* tensor_record,
    iree_allocator_t host_allocator, id4_pipeline_tensor_import_t* import,
    id4_pipeline_program_region_initialized_ranges_t* out_ranges) {
  out_ranges->values = NULL;
  out_ranges->count = 0;
  import->initialized_ranges = NULL;
  import->initialized_range_count = 0;
  if (iree_all_bits_set(import->flags,
                        ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED)) {
    return iree_ok_status();
  }

  bool fully_written = false;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_region_tensor_fully_written_before_range(
          options, tensor, tensor_record, &fully_written));
  if (fully_written) {
    import->flags |= ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_region_collect_initialized_write_ranges_before_range(
          options, tensor, tensor_record, host_allocator, out_ranges));
  import->initialized_ranges = out_ranges->values;
  import->initialized_range_count = out_ranges->count;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_lookup_local_producer(
    const id4_pipeline_program_region_lower_options_t* options,
    id4_pipeline_program_tensor_t tensor,
    const id4_pipeline_program_tensor_record_t** out_record,
    const id4_pipeline_program_op_t** out_producer) {
  if (out_record) *out_record = NULL;
  *out_producer = NULL;
  const id4_pipeline_program_tensor_record_t* record =
      id4_pipeline_program_tensor_at(options->program, tensor.ordinal);
  if (!record) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program tensor %u is missing", tensor.ordinal);
  }
  const id4_pipeline_program_op_t* producer = id4_pipeline_program_operation_at(
      options->program, record->producer_operation_ordinal);
  if (!producer) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program tensor %u producer operation %" PRIhsz
                            " is missing",
                            tensor.ordinal, record->producer_operation_ordinal);
  }
  if (out_record) *out_record = record;
  if (producer->kind == ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE) {
    *out_producer = producer;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_resolve_shared_acquire(
    const id4_pipeline_program_region_lower_options_t* options,
    const id4_pipeline_program_acquire_op_t* acquire_op,
    const id4_pipeline_program_tensor_record_t* tensor, bool* out_is_shared,
    id4_pipeline_tensor_import_t* out_import) {
  *out_is_shared = false;
  memset(out_import, 0, sizeof(*out_import));
  if (!options->resolve_shared_tensor) return iree_ok_status();
  return options->resolve_shared_tensor(options->user_data, acquire_op, tensor,
                                        out_is_shared, out_import);
}

static iree_status_t id4_pipeline_program_region_validate_local_tensor_range(
    const id4_pipeline_program_region_lower_options_t* options,
    id4_pipeline_program_tensor_t tensor) {
  const id4_pipeline_program_tensor_record_t* tensor_record = NULL;
  const id4_pipeline_program_op_t* producer = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_lookup_local_producer(
      options, tensor, &tensor_record, &producer));
  if (!producer) return iree_ok_status();

  const iree_host_size_t operation_offset = options->source_operation_offset;
  const iree_host_size_t operation_limit =
      options->source_operation_offset + options->source_operation_count;
  const iree_host_size_t producer_ordinal = producer->ordinal;
  if (producer_ordinal < operation_offset) {
    bool is_shared = false;
    id4_pipeline_tensor_import_t import;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_region_resolve_shared_acquire(
        options, &producer->payload.acquire, tensor_record, &is_shared,
        &import));
    if (is_shared) return iree_ok_status();
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program tensor %u is a local transient produced before source "
        "operation range [%" PRIhsz ", %" PRIhsz
        ") and requires shared transient lowering",
        tensor.ordinal, operation_offset, operation_limit);
  }
  if (producer_ordinal >= operation_limit) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "program tensor %u is produced after source operation range [%" PRIhsz
        ", %" PRIhsz ")",
        tensor.ordinal, operation_offset, operation_limit);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_validate_local_residency(
    const id4_pipeline_program_region_lower_options_t* options) {
  const iree_host_size_t operation_limit =
      options->source_operation_offset + options->source_operation_count;
  for (iree_host_size_t i = options->source_operation_offset;
       i < operation_limit; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op) continue;
    switch (op->kind) {
      case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
        for (iree_host_size_t j = 0;
             j < op->payload.dispatch_loom.binding_count; ++j) {
          IREE_RETURN_IF_ERROR(
              id4_pipeline_program_region_validate_local_tensor_range(
                  options, op->payload.dispatch_loom.bindings[j].tensor));
        }
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
        if (id4_pipeline_program_region_captures_tap(options,
                                                     op->payload.tap.name)) {
          IREE_RETURN_IF_ERROR(
              id4_pipeline_program_region_validate_local_tensor_range(
                  options, op->payload.tap.tensor));
        }
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT: {
        IREE_RETURN_IF_ERROR(
            id4_pipeline_program_region_validate_local_tensor_range(
                options, op->payload.export_value.tensor));
        break;
      }
      default:
        break;
    }
  }

  const iree_host_size_t tensor_count =
      id4_pipeline_program_tensor_count(options->program);
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t tensor_index = 0; tensor_index < tensor_count;
       ++tensor_index) {
    id4_pipeline_program_tensor_t tensor = {
        // Program-local tensor ordinal.
        .ordinal = (uint32_t)tensor_index,
    };
    const id4_pipeline_program_tensor_record_t* tensor_record = NULL;
    const id4_pipeline_program_op_t* producer = NULL;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_region_lookup_local_producer(
        options, tensor, &tensor_record, &producer));
    if (!producer || producer->ordinal < options->source_operation_offset ||
        producer->ordinal >= operation_limit) {
      continue;
    }
    bool is_shared = false;
    id4_pipeline_tensor_import_t import;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_region_resolve_shared_acquire(
        options, &producer->payload.acquire, tensor_record, &is_shared,
        &import));
    if (is_shared) continue;
    for (iree_host_size_t i = operation_limit; i < operation_count; ++i) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(options->program, i);
      if (id4_pipeline_program_region_operation_uses_tensor(options, op,
                                                            tensor)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "program tensor %u is a local transient used after source "
            "operation range [%" PRIhsz ", %" PRIhsz
            ") and requires shared transient lowering",
            tensor.ordinal, options->source_operation_offset, operation_limit);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_note_tensor_last_use(
    id4_pipeline_program_tensor_t tensor, iree_host_size_t operation_ordinal,
    iree_host_size_t tensor_count, iree_host_size_t* last_use_ordinals) {
  if ((iree_host_size_t)tensor.ordinal >= tensor_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program tensor %u exceeds tensor count %" PRIhsz,
                            tensor.ordinal, tensor_count);
  }
  last_use_ordinals[tensor.ordinal] = operation_ordinal;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_build_liveness(
    const id4_pipeline_program_region_lower_options_t* options,
    iree_host_size_t tensor_count, iree_host_size_t* last_use_ordinals,
    uint8_t* local_tensor_bits) {
  for (iree_host_size_t i = 0; i < tensor_count; ++i) {
    last_use_ordinals[i] = IREE_HOST_SIZE_MAX;
    local_tensor_bits[i] = 0;
  }

  const iree_host_size_t operation_limit =
      options->source_operation_offset + options->source_operation_count;
  for (iree_host_size_t i = options->source_operation_offset;
       i < operation_limit; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op) continue;
    switch (op->kind) {
      case ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE:
        if ((iree_host_size_t)op->payload.acquire.tensor.ordinal >=
            tensor_count) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "program acquire tensor %u exceeds tensor count %" PRIhsz,
              op->payload.acquire.tensor.ordinal, tensor_count);
        }
        const id4_pipeline_program_tensor_record_t* tensor_record =
            id4_pipeline_program_tensor_at(options->program,
                                           op->payload.acquire.tensor.ordinal);
        if (!tensor_record) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "program acquire tensor %u is missing",
                                  op->payload.acquire.tensor.ordinal);
        }
        bool is_shared = false;
        id4_pipeline_tensor_import_t import;
        IREE_RETURN_IF_ERROR(id4_pipeline_program_region_resolve_shared_acquire(
            options, &op->payload.acquire, tensor_record, &is_shared, &import));
        if (is_shared) break;
        local_tensor_bits[op->payload.acquire.tensor.ordinal] = 1;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
        for (iree_host_size_t j = 0;
             j < op->payload.dispatch_loom.binding_count; ++j) {
          IREE_RETURN_IF_ERROR(id4_pipeline_program_region_note_tensor_last_use(
              op->payload.dispatch_loom.bindings[j].tensor, op->ordinal,
              tensor_count, last_use_ordinals));
        }
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
        if (id4_pipeline_program_region_captures_tap(options,
                                                     op->payload.tap.name)) {
          IREE_RETURN_IF_ERROR(id4_pipeline_program_region_note_tensor_last_use(
              op->payload.tap.tensor, op->ordinal, tensor_count,
              last_use_ordinals));
        }
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT: {
        IREE_RETURN_IF_ERROR(id4_pipeline_program_region_note_tensor_last_use(
            op->payload.export_value.tensor, op->ordinal, tensor_count,
            last_use_ordinals));
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_program_format_dispatch_specialization_key(
    const id4_pipeline_program_dispatch_loom_op_t* dispatch_op,
    iree_allocator_t host_allocator, iree_string_view_t* out_key) {
  IREE_ASSERT_ARGUMENT(out_key);
  *out_key = iree_string_view_empty();
  if (!dispatch_op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program dispatch operation is required");
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status = iree_string_builder_append_string(
      &builder, dispatch_op->kernel.module_path);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "::");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &builder, dispatch_op->kernel.function_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "[");
  }
  for (iree_host_size_t i = 0;
       i < dispatch_op->config_binding_count && iree_status_is_ok(status);
       ++i) {
    if (i != 0) {
      status = iree_string_builder_append_cstring(&builder, ",");
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(
          &builder, dispatch_op->config_bindings[i].key);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(&builder, "=");
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(
          &builder, dispatch_op->config_bindings[i].value);
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

static id4_pipeline_tensor_shape_t id4_pipeline_program_region_convert_shape(
    id4_pipeline_program_shape_t source) {
  id4_pipeline_tensor_shape_t target;
  memset(&target, 0, sizeof(target));
  target.rank = source.rank;
  memcpy(target.dims, source.dims, sizeof(target.dims));
  return target;
}

id4_pipeline_tensor_dtype_t id4_pipeline_program_region_convert_dtype(
    id4_pipeline_program_dtype_t dtype) {
  switch (dtype) {
    case ID4_PIPELINE_PROGRAM_DTYPE_F32:
      return ID4_PIPELINE_TENSOR_DTYPE_F32;
    case ID4_PIPELINE_PROGRAM_DTYPE_F16:
      return ID4_PIPELINE_TENSOR_DTYPE_F16;
    case ID4_PIPELINE_PROGRAM_DTYPE_BF16:
      return ID4_PIPELINE_TENSOR_DTYPE_BF16;
    case ID4_PIPELINE_PROGRAM_DTYPE_I32:
      return ID4_PIPELINE_TENSOR_DTYPE_I32;
    case ID4_PIPELINE_PROGRAM_DTYPE_U32:
      return ID4_PIPELINE_TENSOR_DTYPE_U32;
    case ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3:
      return ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3;
    default:
      return ID4_PIPELINE_TENSOR_DTYPE_INVALID;
  }
}

static id4_pipeline_tensor_t id4_pipeline_program_region_invalid_tensor(void) {
  id4_pipeline_tensor_t tensor;
  memset(&tensor, 0, sizeof(tensor));
  tensor.storage_class = ID4_PIPELINE_TENSOR_STORAGE_CLASS_INVALID;
  tensor.ordinal = UINT32_MAX;
  return tensor;
}

static bool id4_pipeline_program_region_tensor_is_valid(
    id4_pipeline_tensor_t tensor) {
  return tensor.storage_class != ID4_PIPELINE_TENSOR_STORAGE_CLASS_INVALID;
}

static iree_status_t id4_pipeline_program_region_tensor_at(
    const id4_pipeline_program_t* program, id4_pipeline_program_tensor_t tensor,
    const id4_pipeline_program_tensor_record_t** out_record) {
  *out_record = NULL;
  const id4_pipeline_program_tensor_record_t* record =
      id4_pipeline_program_tensor_at(program, tensor.ordinal);
  if (!record) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program tensor %u is missing", tensor.ordinal);
  }
  *out_record = record;
  return iree_ok_status();
}

static id4_pipeline_tensor_layout_t id4_pipeline_program_region_tensor_layout(
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_device_size_t alignment) {
  return (id4_pipeline_tensor_layout_t){
      // Tensor diagnostic name.
      .name = tensor->name,
      // Tensor element type.
      .dtype = id4_pipeline_program_region_convert_dtype(tensor->dtype),
      // Tensor shape.
      .shape = id4_pipeline_program_region_convert_shape(tensor->shape),
      // Dense tensor byte length.
      .byte_length = tensor->byte_length,
      // Required base alignment.
      .alignment = alignment,
  };
}

static iree_status_t id4_pipeline_program_region_convert_tensor_access(
    id4_pipeline_program_tensor_access_flags_t source,
    id4_pipeline_tensor_access_flags_t* out_target) {
  *out_target = 0;
  const id4_pipeline_program_tensor_access_flags_t allowed =
      ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ |
      ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE;
  if (source == 0 || iree_any_bit_set(source, ~allowed)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported program tensor access flags 0x%x",
                            source);
  }
  if (iree_all_bits_set(source, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ)) {
    *out_target |= ID4_PIPELINE_TENSOR_ACCESS_READ;
  }
  if (iree_all_bits_set(source, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE)) {
    *out_target |= ID4_PIPELINE_TENSOR_ACCESS_WRITE;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_store_tensor(
    id4_pipeline_program_region_context_t* context,
    id4_pipeline_program_tensor_t program_tensor,
    id4_pipeline_tensor_t region_tensor) {
  if ((iree_host_size_t)program_tensor.ordinal >= context->tensor_map_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program tensor %u exceeds tensor map count %" PRIhsz,
        program_tensor.ordinal, context->tensor_map_count);
  }
  context->tensor_map[program_tensor.ordinal] = region_tensor;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_load_tensor(
    id4_pipeline_program_region_context_t* context,
    id4_pipeline_program_tensor_t program_tensor,
    id4_pipeline_tensor_t* out_region_tensor) {
  *out_region_tensor = id4_pipeline_program_region_invalid_tensor();
  if ((iree_host_size_t)program_tensor.ordinal >= context->tensor_map_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program tensor %u exceeds tensor map count %" PRIhsz,
        program_tensor.ordinal, context->tensor_map_count);
  }
  id4_pipeline_tensor_t region_tensor =
      context->tensor_map[program_tensor.ordinal];
  if (!id4_pipeline_program_region_tensor_is_valid(region_tensor)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "program tensor %u has not been lowered",
                            program_tensor.ordinal);
  }
  *out_region_tensor = region_tensor;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_release_last_use_tensor(
    id4_pipeline_program_region_context_t* context,
    id4_pipeline_program_tensor_t program_tensor,
    iree_host_size_t operation_ordinal) {
  if ((iree_host_size_t)program_tensor.ordinal >= context->tensor_map_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program tensor %u exceeds tensor map count %" PRIhsz,
        program_tensor.ordinal, context->tensor_map_count);
  }
  if (!context->local_tensor_bits[program_tensor.ordinal] ||
      context->released_tensor_bits[program_tensor.ordinal] ||
      context->last_use_ordinals[program_tensor.ordinal] != operation_ordinal) {
    return iree_ok_status();
  }
  id4_pipeline_tensor_t region_tensor =
      id4_pipeline_program_region_invalid_tensor();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_load_tensor(
      context, program_tensor, &region_tensor));
  IREE_RETURN_IF_ERROR(id4_pipeline_region_release_tensor(
      context->options->builder, region_tensor));
  context->released_tensor_bits[program_tensor.ordinal] = 1;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_release_last_uses(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_op_t* op) {
  switch (op->kind) {
    case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
      for (iree_host_size_t i = 0; i < op->payload.dispatch_loom.binding_count;
           ++i) {
        IREE_RETURN_IF_ERROR(
            id4_pipeline_program_region_release_last_use_tensor(
                context, op->payload.dispatch_loom.bindings[i].tensor,
                op->ordinal));
      }
      return iree_ok_status();
    case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
      if (id4_pipeline_program_region_captures_tap(context->options,
                                                   op->payload.tap.name)) {
        IREE_RETURN_IF_ERROR(
            id4_pipeline_program_region_release_last_use_tensor(
                context, op->payload.tap.tensor, op->ordinal));
      }
      return iree_ok_status();
    case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT:
      return id4_pipeline_program_region_release_last_use_tensor(
          context, op->payload.export_value.tensor, op->ordinal);
    default:
      return iree_ok_status();
  }
}

static iree_status_t id4_pipeline_program_region_lower_import(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_import_op_t* import_op,
    iree_host_size_t import_ordinal) {
  if (!context->options->resolve_import) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region import resolver is required");
  }
  const id4_pipeline_program_tensor_record_t* tensor = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_tensor_at(
      context->options->program, import_op->tensor, &tensor));
  id4_pipeline_tensor_import_t import;
  memset(&import, 0, sizeof(import));
  IREE_RETURN_IF_ERROR(context->options->resolve_import(
      context->options->user_data, import_op, tensor, import_ordinal, &import));
  id4_pipeline_program_region_initialized_ranges_t initialized_ranges;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_region_populate_prior_initialization(
          context->options, import_op->tensor, tensor, context->host_allocator,
          &import, &initialized_ranges));
  id4_pipeline_tensor_t region_tensor =
      id4_pipeline_program_region_invalid_tensor();
  iree_status_t status = id4_pipeline_region_import_tensor(
      context->options->builder, &import, &region_tensor);
  iree_allocator_free(context->host_allocator, initialized_ranges.values);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_region_store_tensor(
        context, import_op->tensor, region_tensor);
  }
  return status;
}

static iree_status_t id4_pipeline_program_region_lower_parameter(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_parameter_op_t* parameter_op,
    iree_host_size_t parameter_ordinal) {
  if (!context->options->resolve_parameter) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region parameter resolver is required");
  }
  const id4_pipeline_program_tensor_record_t* tensor = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_tensor_at(
      context->options->program, parameter_op->tensor, &tensor));
  id4_pipeline_tensor_import_t import;
  memset(&import, 0, sizeof(import));
  IREE_RETURN_IF_ERROR(context->options->resolve_parameter(
      context->options->user_data, parameter_op, tensor, parameter_ordinal,
      &import));
  id4_pipeline_tensor_t region_tensor =
      id4_pipeline_program_region_invalid_tensor();
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      context->options->builder, &import, &region_tensor));
  return id4_pipeline_program_region_store_tensor(context, parameter_op->tensor,
                                                  region_tensor);
}

static iree_status_t id4_pipeline_program_region_lower_constant(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_constant_op_t* constant_op,
    iree_host_size_t constant_ordinal) {
  if (!context->options->resolve_constant) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program region constant resolver is required");
  }
  const id4_pipeline_program_tensor_record_t* tensor = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_tensor_at(
      context->options->program, constant_op->tensor, &tensor));
  id4_pipeline_tensor_import_t import;
  memset(&import, 0, sizeof(import));
  IREE_RETURN_IF_ERROR(context->options->resolve_constant(
      context->options->user_data, constant_op, tensor, constant_ordinal,
      &import));
  id4_pipeline_tensor_t region_tensor =
      id4_pipeline_program_region_invalid_tensor();
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      context->options->builder, &import, &region_tensor));
  return id4_pipeline_program_region_store_tensor(context, constant_op->tensor,
                                                  region_tensor);
}

static iree_status_t id4_pipeline_program_region_lower_acquire(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_acquire_op_t* acquire_op) {
  const id4_pipeline_program_tensor_record_t* tensor = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_tensor_at(
      context->options->program, acquire_op->tensor, &tensor));
  bool is_shared = false;
  id4_pipeline_tensor_import_t import;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_resolve_shared_acquire(
      context->options, acquire_op, tensor, &is_shared, &import));
  if (is_shared) {
    id4_pipeline_program_region_initialized_ranges_t initialized_ranges;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_region_populate_prior_initialization(
            context->options, acquire_op->tensor, tensor,
            context->host_allocator, &import, &initialized_ranges));
    id4_pipeline_tensor_t region_tensor =
        id4_pipeline_program_region_invalid_tensor();
    iree_status_t status = id4_pipeline_region_import_tensor(
        context->options->builder, &import, &region_tensor);
    iree_allocator_free(context->host_allocator, initialized_ranges.values);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_program_region_store_tensor(
          context, acquire_op->tensor, region_tensor);
    }
    return status;
  }
  id4_pipeline_tensor_layout_t layout =
      id4_pipeline_program_region_tensor_layout(
          tensor, context->options->local_tensor_alignment);
  id4_pipeline_tensor_t region_tensor =
      id4_pipeline_program_region_invalid_tensor();
  IREE_RETURN_IF_ERROR(id4_pipeline_region_acquire_tensor(
      context->options->builder, &layout, &region_tensor));
  return id4_pipeline_program_region_store_tensor(context, acquire_op->tensor,
                                                  region_tensor);
}

static iree_status_t id4_pipeline_program_region_lower_dispatch_binding(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_dispatch_binding_t* program_binding,
    id4_pipeline_region_dispatch_binding_t* region_binding) {
  memset(region_binding, 0, sizeof(*region_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_load_tensor(
      context, program_binding->tensor, &region_binding->tensor));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_convert_tensor_access(
      program_binding->access, &region_binding->access));
  if (iree_all_bits_set(
          program_binding->flags,
          ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_WRITE_RANGE)) {
    region_binding->flags =
        ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_WRITE_RANGE;
    region_binding->write_range = (id4_pipeline_region_tensor_byte_range_t){
        // Byte offset from the start of the logical tensor.
        .offset = program_binding->write_range.offset,
        // Byte length of the interval.
        .length = program_binding->write_range.length,
    };
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_resolve_kernel(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_dispatch_loom_op_t* dispatch_op,
    iree_string_view_t specialization_key, iree_host_size_t dispatch_ordinal,
    id4_pipeline_program_region_kernel_resolution_t* out_resolution) {
  memset(out_resolution, 0, sizeof(*out_resolution));
  out_resolution->function = iree_hal_executable_function_invalid();
  if (id4_pipeline_region_builder_mode(context->options->builder) !=
      ID4_PIPELINE_REGION_BUILDER_MODE_RECORD) {
    return iree_ok_status();
  }
  return context->options->resolve_kernel(context->options->user_data,
                                          dispatch_op, specialization_key,
                                          dispatch_ordinal, out_resolution);
}

static iree_status_t id4_pipeline_program_region_validate_dispatch_config(
    const id4_pipeline_program_dispatch_loom_op_t* dispatch_op,
    iree_hal_dispatch_config_t dispatch_config) {
  if (dispatch_config.workgroup_count_ref.buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program dispatch %.*s resolved indirect workgroup counts",
        (int)dispatch_op->name.size, dispatch_op->name.data);
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(dispatch_config.workgroup_count); ++i) {
    if (dispatch_config.workgroup_count[i] == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "program dispatch %.*s resolved zero workgroup count dimension "
          "%" PRIhsz,
          (int)dispatch_op->name.size, dispatch_op->name.data, i);
    }
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(dispatch_config.workgroup_size); ++i) {
    if (dispatch_config.workgroup_size[i] == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "program dispatch %.*s resolved zero workgroup size dimension "
          "%" PRIhsz,
          (int)dispatch_op->name.size, dispatch_op->name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_region_lower_dispatch(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_dispatch_loom_op_t* dispatch_op,
    iree_host_size_t dispatch_ordinal) {
  for (iree_host_size_t i = 0; i < dispatch_op->binding_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_program_region_lower_dispatch_binding(
        context, &dispatch_op->bindings[i], &context->dispatch_bindings[i]));
  }

  iree_string_view_t specialization_key = iree_string_view_empty();
  iree_status_t status =
      id4_pipeline_program_format_dispatch_specialization_key(
          dispatch_op, context->host_allocator, &specialization_key);
  id4_pipeline_program_region_kernel_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  resolution.function = iree_hal_executable_function_invalid();
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_region_resolve_kernel(
        context, dispatch_op, specialization_key, dispatch_ordinal,
        &resolution);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_region_loom_kernel_t kernel = {
        // Stable specialization key.
        .specialization_key = specialization_key,
        // Stable module path.
        .module_path = dispatch_op->kernel.module_path,
        // Exported Loom kernel function.
        .function_name = dispatch_op->kernel.function_name,
        // Prepared HAL executable for record mode.
        .executable = resolution.executable,
        // Prepared HAL executable function for record mode.
        .function = resolution.function,
        // Exact tensor binding count expected by the kernel ABI.
        .binding_count = dispatch_op->binding_count,
        // This program layer does not author dispatch constants yet.
        .constant_byte_length = 0,
        // Number of config bindings used by this specialization.
        .config_binding_count = dispatch_op->config_binding_count,
        // Config bindings borrowed from the immutable semantic program.
        .config_bindings = dispatch_op->config_bindings,
    };
    iree_hal_dispatch_config_t dispatch_config;
    memset(&dispatch_config, 0, sizeof(dispatch_config));
    if (id4_pipeline_region_builder_mode(context->options->builder) ==
        ID4_PIPELINE_REGION_BUILDER_MODE_RECORD) {
      dispatch_config = resolution.dispatch_config;
      status = id4_pipeline_program_region_validate_dispatch_config(
          dispatch_op, dispatch_config);
    }
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_region_dispatch_loom(
          context->options->builder, &kernel, dispatch_config,
          iree_const_byte_span_empty(), dispatch_op->binding_count,
          context->dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE);
    }
  }
  iree_allocator_free(context->host_allocator, (void*)specialization_key.data);
  return status;
}

static iree_status_t id4_pipeline_program_region_lower_barrier(
    id4_pipeline_program_region_context_t* context) {
  const iree_hal_memory_barrier_t memory_barrier = {
      // Prior dispatch reads/writes that must complete before the next epoch.
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      // Following dispatch reads/writes in the next epoch.
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  return id4_pipeline_region_barrier(
      context->options->builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, /*memory_barriers=*/&memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL);
}

static iree_status_t id4_pipeline_program_region_lower_tap(
    id4_pipeline_program_region_context_t* context,
    const id4_pipeline_program_tap_op_t* tap_op, iree_host_size_t tap_ordinal) {
  if (context->options->tap_mode ==
      ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_IGNORE) {
    return iree_ok_status();
  }
  if (!id4_pipeline_program_region_captures_tap(context->options,
                                                tap_op->name)) {
    return iree_ok_status();
  }
  const id4_pipeline_program_tensor_record_t* tensor = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_tensor_at(
      context->options->program, tap_op->tensor, &tensor));

  id4_pipeline_tensor_import_t import;
  memset(&import, 0, sizeof(import));
  IREE_RETURN_IF_ERROR(context->options->resolve_tap(
      context->options->user_data, tap_op, tensor, tap_ordinal, &import));

  id4_pipeline_tensor_t source_tensor =
      id4_pipeline_program_region_invalid_tensor();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_load_tensor(
      context, tap_op->tensor, &source_tensor));

  id4_pipeline_tensor_t target_tensor =
      id4_pipeline_program_region_invalid_tensor();
  IREE_RETURN_IF_ERROR(id4_pipeline_region_import_tensor(
      context->options->builder, &import, &target_tensor));
  const iree_hal_memory_barrier_t dispatch_to_transfer_barrier = {
      // Prior dispatch writes that make the tapped tensor readable.
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      // The diagnostic copy reads the tapped tensor.
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_region_barrier(
      context->options->builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1,
      /*memory_barriers=*/&dispatch_to_transfer_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_region_copy_tensor(context->options->builder, source_tensor,
                                      target_tensor, IREE_HAL_COPY_FLAG_NONE));
  const iree_hal_memory_barrier_t transfer_to_dispatch_barrier = {
      // The diagnostic copy reads the source tensor and writes the capture
      // buffer before following work.
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      // Following dispatches may read or write reused tensors in the next
      // epoch.
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  return id4_pipeline_region_barrier(
      context->options->builder, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1,
      /*memory_barriers=*/&transfer_to_dispatch_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL);
}

iree_status_t id4_pipeline_program_region_lower(
    const id4_pipeline_program_region_lower_options_t* options,
    iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(id4_pipeline_program_region_validate_options(options));

  id4_pipeline_program_region_counts_t counts =
      id4_pipeline_program_region_count_ops(options);
  id4_pipeline_tensor_t* tensor_map = NULL;
  id4_pipeline_region_dispatch_binding_t* dispatch_bindings = NULL;
  iree_host_size_t* last_use_ordinals = NULL;
  uint8_t* local_tensor_bits = NULL;
  uint8_t* released_tensor_bits = NULL;
  const iree_host_size_t tensor_count =
      id4_pipeline_program_tensor_count(options->program);
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, tensor_count, sizeof(tensor_map[0]), (void**)&tensor_map);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < tensor_count; ++i) {
      tensor_map[i] = id4_pipeline_program_region_invalid_tensor();
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, tensor_count,
                                         sizeof(last_use_ordinals[0]),
                                         (void**)&last_use_ordinals);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, tensor_count,
                                         sizeof(local_tensor_bits[0]),
                                         (void**)&local_tensor_bits);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, tensor_count,
                                         sizeof(released_tensor_bits[0]),
                                         (void**)&released_tensor_bits);
  }
  if (iree_status_is_ok(status)) {
    memset(released_tensor_bits, 0,
           tensor_count * sizeof(released_tensor_bits[0]));
    status = id4_pipeline_program_region_validate_local_residency(options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_region_build_liveness(
        options, tensor_count, last_use_ordinals, local_tensor_bits);
  }
  if (iree_status_is_ok(status) && counts.max_dispatch_binding_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, counts.max_dispatch_binding_count,
        sizeof(dispatch_bindings[0]), (void**)&dispatch_bindings);
  }

  id4_pipeline_program_region_context_t context = {
      // Program lowering options.
      .options = options,
      // Semantic tensor to region tensor map.
      .tensor_map = tensor_map,
      // Number of semantic tensor map entries.
      .tensor_map_count = tensor_count,
      // Last operation that uses each semantic tensor.
      .last_use_ordinals = last_use_ordinals,
      // Local-acquire origin marker for each semantic tensor.
      .local_tensor_bits = local_tensor_bits,
      // Lowering-time release marker for each semantic tensor.
      .released_tensor_bits = released_tensor_bits,
      // Reusable dispatch binding scratch.
      .dispatch_bindings = dispatch_bindings,
      // Host allocator used for temporary specialization keys.
      .host_allocator = host_allocator,
  };

  iree_host_size_t import_ordinal = 0;
  iree_host_size_t parameter_ordinal = 0;
  iree_host_size_t constant_ordinal = 0;
  iree_host_size_t dispatch_ordinal = 0;
  iree_host_size_t tap_ordinal = 0;
  const iree_host_size_t operation_offset = options->source_operation_offset;
  const iree_host_size_t operation_limit =
      options->source_operation_offset + options->source_operation_count;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!op) continue;
    const bool emit_operation = i >= operation_offset && i < operation_limit;
    switch (op->kind) {
      case ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT:
        status = id4_pipeline_program_region_lower_import(
            &context, &op->payload.import_value, import_ordinal++);
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER:
        if (id4_pipeline_program_region_range_uses_tensor(
                options, op->payload.parameter.tensor)) {
          status = id4_pipeline_program_region_lower_parameter(
              &context, &op->payload.parameter, parameter_ordinal);
        }
        ++parameter_ordinal;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT:
        status = id4_pipeline_program_region_lower_constant(
            &context, &op->payload.constant, constant_ordinal++);
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE: {
        bool should_lower_acquire = emit_operation;
        if (!should_lower_acquire && i < operation_offset &&
            id4_pipeline_program_region_range_uses_tensor(
                options, op->payload.acquire.tensor)) {
          const id4_pipeline_program_tensor_record_t* tensor_record = NULL;
          status = id4_pipeline_program_region_tensor_at(
              options->program, op->payload.acquire.tensor, &tensor_record);
          if (iree_status_is_ok(status)) {
            bool is_shared = false;
            id4_pipeline_tensor_import_t import;
            status = id4_pipeline_program_region_resolve_shared_acquire(
                options, &op->payload.acquire, tensor_record, &is_shared,
                &import);
            should_lower_acquire = is_shared;
          }
        }
        if (iree_status_is_ok(status) && should_lower_acquire) {
          status = id4_pipeline_program_region_lower_acquire(
              &context, &op->payload.acquire);
        }
        break;
      }
      case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
        if (emit_operation) {
          status = id4_pipeline_program_region_lower_dispatch(
              &context, &op->payload.dispatch_loom, dispatch_ordinal);
        }
        ++dispatch_ordinal;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
        if (emit_operation) {
          status = id4_pipeline_program_region_lower_barrier(&context);
        }
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_REGION_CUT:
        if (emit_operation) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "program region cut %.*s cannot be emitted inside source "
              "operation range [%" PRIhsz ", %" PRIhsz ")",
              (int)op->payload.region_cut.name.size,
              op->payload.region_cut.name.data, operation_offset,
              operation_limit);
        }
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
        if (id4_pipeline_program_region_captures_tap(options,
                                                     op->payload.tap.name)) {
          if (emit_operation) {
            status = id4_pipeline_program_region_lower_tap(
                &context, &op->payload.tap, tap_ordinal);
          }
          ++tap_ordinal;
        }
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT:
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported program operation kind %d",
                                  (int)op->kind);
        break;
    }
    if (iree_status_is_ok(status) && emit_operation) {
      status = id4_pipeline_program_region_release_last_uses(&context, op);
    }
    if (i + 1 >= operation_limit) break;
  }

  iree_allocator_free(host_allocator, released_tensor_bits);
  iree_allocator_free(host_allocator, local_tensor_bits);
  iree_allocator_free(host_allocator, last_use_ordinals);
  iree_allocator_free(host_allocator, dispatch_bindings);
  iree_allocator_free(host_allocator, tensor_map);
  return status;
}
