// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

typedef struct id4_pipeline_program_tensor_state_t {
  // Tensor metadata copied into the builder.
  id4_pipeline_program_tensor_record_t record;
  // True when the tensor contents may be read.
  bool initialized;
} id4_pipeline_program_tensor_state_t;

struct id4_pipeline_program_builder_t {
  // Host allocator used for builder storage.
  iree_allocator_t host_allocator;
  // Arena used for append-only builder storage.
  iree_arena_allocator_t arena;
  // Allocator view over arena used by shared copy helpers.
  iree_allocator_t arena_allocator;
  // Program name copied into the builder.
  iree_string_view_t program_name;
  // Tensor states authored so far.
  id4_pipeline_program_tensor_state_t* tensor_states;
  // Number of tensor states authored so far.
  iree_host_size_t tensor_count;
  // Allocated tensor-state capacity.
  iree_host_size_t tensor_capacity;
  // Operation records authored so far.
  id4_pipeline_program_op_t* operations;
  // Number of operation records authored so far.
  iree_host_size_t operation_count;
  // Allocated operation-record capacity.
  iree_host_size_t operation_capacity;
};

struct id4_pipeline_program_t {
  // Reference count for shared program ownership.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for program storage.
  iree_allocator_t host_allocator;
  // Program name copied into this immutable object.
  iree_string_view_t name;
  // Number of tensor records in this program.
  iree_host_size_t tensor_count;
  // Tensor records owned by this program.
  id4_pipeline_program_tensor_record_t* tensors;
  // Number of operation records in this program.
  iree_host_size_t operation_count;
  // Operation records owned by this program.
  id4_pipeline_program_op_t* operations;
};

static iree_status_t id4_pipeline_program_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_program_copy_string(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_string_view_t* out_target) {
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  if (source.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program string is too large to copy");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, source.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.size);
  storage[source.size] = 0;
  *out_target = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void id4_pipeline_program_free_string(iree_string_view_t* value,
                                             iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static iree_status_t id4_pipeline_program_validate_shape(
    id4_pipeline_program_shape_t shape, iree_string_view_t tensor_name) {
  if (shape.rank > ID4_PIPELINE_PROGRAM_TENSOR_MAX_RANK) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor %.*s rank %u exceeds max rank %u",
                            (int)tensor_name.size, tensor_name.data, shape.rank,
                            ID4_PIPELINE_PROGRAM_TENSOR_MAX_RANK);
  }
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (shape.dims[i] == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tensor %.*s dimension %u is zero",
                              (int)tensor_name.size, tensor_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_validate_tensor_metadata(
    iree_string_view_t name, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t shape, iree_device_size_t* out_byte_length) {
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor name is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_shape(shape, name));
  return id4_pipeline_program_tensor_byte_length(dtype, shape, out_byte_length);
}

static iree_status_t id4_pipeline_program_validate_access(
    id4_pipeline_program_tensor_access_flags_t access,
    iree_string_view_t dispatch_name, iree_host_size_t binding_index) {
  if (access == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch %.*s binding %" PRIhsz " access is required",
        (int)dispatch_name.size, dispatch_name.data, binding_index);
  }
  if (iree_any_bit_set(access, ~(ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ |
                                 ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch %.*s binding %" PRIhsz " access has unsupported bits 0x%x",
        (int)dispatch_name.size, dispatch_name.data, binding_index, access);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_validate_dispatch_binding_flags(
    id4_pipeline_program_dispatch_binding_flags_t flags,
    iree_string_view_t dispatch_name, iree_host_size_t binding_index) {
  const id4_pipeline_program_dispatch_binding_flags_t allowed_flags =
      ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_WRITE_RANGE;
  if (!iree_any_bit_set(flags, ~allowed_flags)) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "dispatch %.*s binding %" PRIhsz " has unsupported flags 0x%x",
      (int)dispatch_name.size, dispatch_name.data, binding_index, flags);
}

static iree_status_t id4_pipeline_program_validate_dispatch_binding_range(
    const id4_pipeline_program_dispatch_binding_t* binding,
    const id4_pipeline_program_tensor_record_t* tensor_record,
    iree_string_view_t dispatch_name, iree_host_size_t binding_index) {
  if (!iree_all_bits_set(
          binding->flags,
          ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_WRITE_RANGE)) {
    return iree_ok_status();
  }
  if (!iree_any_bit_set(binding->access,
                        ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch %.*s binding %" PRIhsz
                            " has write coverage without write access",
                            (int)dispatch_name.size, dispatch_name.data,
                            binding_index);
  }
  if (binding->write_range.length == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch %.*s binding %" PRIhsz " write coverage is empty",
        (int)dispatch_name.size, dispatch_name.data, binding_index);
  }
  iree_device_size_t end = 0;
  if (!iree_device_size_checked_add(binding->write_range.offset,
                                    binding->write_range.length, &end) ||
      end > tensor_record->byte_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch %.*s binding %" PRIhsz " write coverage [%" PRIu64
        ", %" PRIu64 ") exceeds tensor %.*s byte length %" PRIu64,
        (int)dispatch_name.size, dispatch_name.data, binding_index,
        (uint64_t)binding->write_range.offset, (uint64_t)end,
        (int)tensor_record->name.size, tensor_record->name.data,
        (uint64_t)tensor_record->byte_length);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_validate_import_flags(
    id4_pipeline_program_import_tensor_flags_t flags,
    iree_string_view_t tensor_name) {
  const id4_pipeline_program_import_tensor_flags_t allowed_flags =
      ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED;
  if (!iree_any_bit_set(flags, ~allowed_flags)) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "import tensor %.*s has unsupported flags 0x%x",
                          (int)tensor_name.size, tensor_name.data, flags);
}

static iree_status_t id4_pipeline_program_validate_kernel_ref(
    id4_pipeline_kernel_ref_t kernel, iree_string_view_t dispatch_name) {
  if (iree_string_view_is_empty(kernel.module_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch %.*s module path is required",
                            (int)dispatch_name.size, dispatch_name.data);
  }
  if (iree_string_view_is_empty(kernel.function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch %.*s function name is required",
                            (int)dispatch_name.size, dispatch_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_validate_config_bindings(
    iree_string_view_t dispatch_name, iree_host_size_t config_binding_count,
    const id4_pipeline_kernel_config_binding_t* config_bindings) {
  if (config_binding_count != 0 && !config_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch %.*s config binding array is required",
                            (int)dispatch_name.size, dispatch_name.data);
  }
  for (iree_host_size_t i = 0; i < config_binding_count; ++i) {
    if (iree_string_view_is_empty(config_bindings[i].key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch %.*s config binding %" PRIhsz
                              " key is required",
                              (int)dispatch_name.size, dispatch_name.data, i);
    }
    if (iree_string_view_is_empty(config_bindings[i].value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch %.*s config binding %" PRIhsz
                              " value is required",
                              (int)dispatch_name.size, dispatch_name.data, i);
    }
  }
  return iree_ok_status();
}

static void id4_pipeline_program_free_config_bindings(
    iree_host_size_t config_binding_count,
    const id4_pipeline_kernel_config_binding_t* config_bindings,
    iree_allocator_t host_allocator) {
  if (!config_bindings) return;
  id4_pipeline_kernel_config_binding_t* mutable_bindings =
      (id4_pipeline_kernel_config_binding_t*)config_bindings;
  for (iree_host_size_t i = 0; i < config_binding_count; ++i) {
    id4_pipeline_program_free_string(&mutable_bindings[i].value,
                                     host_allocator);
    id4_pipeline_program_free_string(&mutable_bindings[i].key, host_allocator);
  }
  iree_allocator_free(host_allocator, mutable_bindings);
}

static iree_status_t id4_pipeline_program_copy_config_bindings(
    iree_host_size_t source_count,
    const id4_pipeline_kernel_config_binding_t* source_values,
    iree_allocator_t host_allocator, iree_host_size_t* out_target_count,
    const id4_pipeline_kernel_config_binding_t** out_target_values) {
  *out_target_count = 0;
  *out_target_values = NULL;
  if (source_count == 0) return iree_ok_status();

  id4_pipeline_kernel_config_binding_t* target_values = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, source_count,
                                                   sizeof(target_values[0]),
                                                   (void**)&target_values));
  memset(target_values, 0, source_count * sizeof(target_values[0]));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < source_count && iree_status_is_ok(status);
       ++i) {
    status = id4_pipeline_program_copy_string(
        source_values[i].key, host_allocator, &target_values[i].key);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_program_copy_string(
          source_values[i].value, host_allocator, &target_values[i].value);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_target_count = source_count;
    *out_target_values = target_values;
  } else {
    id4_pipeline_program_free_config_bindings(source_count, target_values,
                                              host_allocator);
  }
  return status;
}

static void id4_pipeline_program_free_dispatch_bindings(
    const id4_pipeline_program_dispatch_binding_t* bindings,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, (void*)bindings);
}

static void id4_pipeline_program_free_constant_data(
    const uint8_t* data, iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, (void*)data);
}

static iree_status_t id4_pipeline_program_copy_constant_data(
    iree_const_byte_span_t source, iree_allocator_t host_allocator,
    const uint8_t** out_data) {
  *out_data = NULL;
  if (source.data_length == 0) return iree_ok_status();
  uint8_t* target = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, source.data_length, sizeof(target[0]), (void**)&target));
  memcpy(target, source.data, source.data_length);
  *out_data = target;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_copy_dispatch_bindings(
    iree_host_size_t source_count,
    const id4_pipeline_program_dispatch_binding_t* source_values,
    iree_allocator_t host_allocator,
    const id4_pipeline_program_dispatch_binding_t** out_target_values) {
  *out_target_values = NULL;
  if (source_count == 0) return iree_ok_status();

  id4_pipeline_program_dispatch_binding_t* target_values = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, source_count,
                                                   sizeof(target_values[0]),
                                                   (void**)&target_values));
  memcpy(target_values, source_values, source_count * sizeof(target_values[0]));
  *out_target_values = target_values;
  return iree_ok_status();
}

static void id4_pipeline_program_free_tensor_record(
    id4_pipeline_program_tensor_record_t* record,
    iree_allocator_t host_allocator) {
  if (!record) return;
  id4_pipeline_program_free_string(&record->name, host_allocator);
  memset(record, 0, sizeof(*record));
}

static iree_status_t id4_pipeline_program_copy_tensor_record(
    const id4_pipeline_program_tensor_record_t* source,
    iree_allocator_t host_allocator,
    id4_pipeline_program_tensor_record_t* target) {
  memset(target, 0, sizeof(*target));
  target->dtype = source->dtype;
  target->shape = source->shape;
  target->byte_length = source->byte_length;
  target->producer_operation_ordinal = source->producer_operation_ordinal;
  return id4_pipeline_program_copy_string(source->name, host_allocator,
                                          &target->name);
}

static void id4_pipeline_program_free_op(id4_pipeline_program_op_t* op,
                                         iree_allocator_t host_allocator) {
  if (!op) return;
  switch (op->kind) {
    case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
      id4_pipeline_program_free_dispatch_bindings(
          op->payload.dispatch_loom.bindings, host_allocator);
      id4_pipeline_program_free_config_bindings(
          op->payload.dispatch_loom.config_binding_count,
          op->payload.dispatch_loom.config_bindings, host_allocator);
      id4_pipeline_program_free_string(
          &op->payload.dispatch_loom.kernel.function_name, host_allocator);
      id4_pipeline_program_free_string(
          &op->payload.dispatch_loom.kernel.module_path, host_allocator);
      id4_pipeline_program_free_string(&op->payload.dispatch_loom.name,
                                       host_allocator);
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT:
      id4_pipeline_program_free_constant_data(op->payload.constant.data,
                                              host_allocator);
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
      id4_pipeline_program_free_string(&op->payload.barrier.name,
                                       host_allocator);
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
      id4_pipeline_program_free_string(&op->payload.tap.name, host_allocator);
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT:
      id4_pipeline_program_free_string(&op->payload.export_value.name,
                                       host_allocator);
      break;
    default:
      break;
  }
  memset(op, 0, sizeof(*op));
}

static iree_status_t id4_pipeline_program_copy_op(
    const id4_pipeline_program_op_t* source, iree_allocator_t host_allocator,
    id4_pipeline_program_op_t* target) {
  memset(target, 0, sizeof(*target));
  target->kind = source->kind;
  target->ordinal = source->ordinal;
  iree_status_t status = iree_ok_status();
  switch (source->kind) {
    case ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT:
      target->payload.import_value = source->payload.import_value;
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER:
      target->payload.parameter = source->payload.parameter;
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT:
      target->payload.constant.tensor = source->payload.constant.tensor;
      target->payload.constant.data_length =
          source->payload.constant.data_length;
      status = id4_pipeline_program_copy_constant_data(
          iree_make_const_byte_span(source->payload.constant.data,
                                    source->payload.constant.data_length),
          host_allocator, &target->payload.constant.data);
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE:
      target->payload.acquire = source->payload.acquire;
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
      target->payload.dispatch_loom.binding_count =
          source->payload.dispatch_loom.binding_count;
      target->payload.dispatch_loom.kernel.module_path =
          iree_string_view_empty();
      target->payload.dispatch_loom.kernel.function_name =
          iree_string_view_empty();
      status = id4_pipeline_program_copy_string(
          source->payload.dispatch_loom.name, host_allocator,
          &target->payload.dispatch_loom.name);
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_program_copy_string(
            source->payload.dispatch_loom.kernel.module_path, host_allocator,
            &target->payload.dispatch_loom.kernel.module_path);
      }
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_program_copy_string(
            source->payload.dispatch_loom.kernel.function_name, host_allocator,
            &target->payload.dispatch_loom.kernel.function_name);
      }
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_program_copy_config_bindings(
            source->payload.dispatch_loom.config_binding_count,
            source->payload.dispatch_loom.config_bindings, host_allocator,
            &target->payload.dispatch_loom.config_binding_count,
            &target->payload.dispatch_loom.config_bindings);
      }
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_program_copy_dispatch_bindings(
            source->payload.dispatch_loom.binding_count,
            source->payload.dispatch_loom.bindings, host_allocator,
            &target->payload.dispatch_loom.bindings);
      }
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
      status = id4_pipeline_program_copy_string(source->payload.barrier.name,
                                                host_allocator,
                                                &target->payload.barrier.name);
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
      target->payload.tap.tensor = source->payload.tap.tensor;
      status = id4_pipeline_program_copy_string(
          source->payload.tap.name, host_allocator, &target->payload.tap.name);
      break;
    case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT:
      target->payload.export_value.tensor = source->payload.export_value.tensor;
      status = id4_pipeline_program_copy_string(
          source->payload.export_value.name, host_allocator,
          &target->payload.export_value.name);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "program operation kind %d is invalid",
                                (int)source->kind);
      break;
  }
  if (!iree_status_is_ok(status)) {
    id4_pipeline_program_free_op(target, host_allocator);
  }
  return status;
}

static bool id4_pipeline_program_shape_equal(id4_pipeline_program_shape_t lhs,
                                             id4_pipeline_program_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static bool id4_pipeline_program_builder_find_tensor_by_name(
    const id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_tensor_t* out_tensor,
    id4_pipeline_program_tensor_state_t** out_state) {
  for (iree_host_size_t i = 0; i < builder->tensor_count; ++i) {
    if (iree_string_view_equal(builder->tensor_states[i].record.name, name)) {
      if (out_tensor) {
        out_tensor->ordinal = (uint32_t)i;
      }
      if (out_state) {
        *out_state = &builder->tensor_states[i];
      }
      return true;
    }
  }
  if (out_tensor) {
    *out_tensor = id4_pipeline_program_tensor_invalid();
  }
  if (out_state) {
    *out_state = NULL;
  }
  return false;
}

static bool id4_pipeline_program_builder_has_tensor_name(
    const id4_pipeline_program_builder_t* builder, iree_string_view_t name) {
  return id4_pipeline_program_builder_find_tensor_by_name(
      builder, name, /*out_tensor=*/NULL, /*out_state=*/NULL);
}

static bool id4_pipeline_program_builder_has_named_op(
    const id4_pipeline_program_builder_t* builder,
    id4_pipeline_program_op_kind_t kind, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < builder->operation_count; ++i) {
    const id4_pipeline_program_op_t* op = &builder->operations[i];
    if (op->kind != kind) continue;
    iree_string_view_t op_name = iree_string_view_empty();
    switch (kind) {
      case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
        op_name = op->payload.dispatch_loom.name;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
        op_name = op->payload.barrier.name;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
        op_name = op->payload.tap.name;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT:
        op_name = op->payload.export_value.name;
        break;
      default:
        break;
    }
    if (iree_string_view_equal(op_name, name)) return true;
  }
  return false;
}

static iree_status_t id4_pipeline_program_builder_validate_tensor(
    const id4_pipeline_program_builder_t* builder,
    id4_pipeline_program_tensor_t tensor,
    id4_pipeline_program_tensor_state_t** out_state) {
  if (!id4_pipeline_program_tensor_is_valid(tensor) ||
      tensor.ordinal >= builder->tensor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program tensor ordinal %u is invalid",
                            tensor.ordinal);
  }
  *out_state = &builder->tensor_states[tensor.ordinal];
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_builder_add_tensor(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    iree_host_size_t producer_operation_ordinal, bool initialized,
    id4_pipeline_program_tensor_t* out_tensor) {
  *out_tensor = id4_pipeline_program_tensor_invalid();
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_tensor_metadata(
      name, dtype, shape, &byte_length));
  if (id4_pipeline_program_builder_has_tensor_name(builder, name)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "program tensor %.*s already exists",
                            (int)name.size, name.data);
  }
  if (builder->tensor_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program tensor ordinal overflow");
  }

  if (builder->tensor_count == builder->tensor_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &builder->arena, builder->tensor_count, builder->tensor_count + 1,
        sizeof(builder->tensor_states[0]), &builder->tensor_capacity,
        (void**)&builder->tensor_states));
  }
  id4_pipeline_program_tensor_state_t* state =
      &builder->tensor_states[builder->tensor_count];
  memset(state, 0, sizeof(*state));
  id4_pipeline_program_tensor_record_t source_record = {
      .name = name,
      .dtype = dtype,
      .shape = shape,
      .byte_length = byte_length,
      .producer_operation_ordinal = producer_operation_ordinal,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_program_copy_tensor_record(
      &source_record, builder->arena_allocator, &state->record));
  state->initialized = initialized;
  out_tensor->ordinal = (uint32_t)builder->tensor_count;
  ++builder->tensor_count;
  return iree_ok_status();
}

static void id4_pipeline_program_builder_remove_last_tensor(
    id4_pipeline_program_builder_t* builder) {
  if (builder->tensor_count == 0) return;
  --builder->tensor_count;
  memset(&builder->tensor_states[builder->tensor_count], 0,
         sizeof(builder->tensor_states[builder->tensor_count]));
}

static iree_status_t id4_pipeline_program_builder_append_op(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_op_t* source_op) {
  if (builder->operation_count == builder->operation_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &builder->arena, builder->operation_count, builder->operation_count + 1,
        sizeof(builder->operations[0]), &builder->operation_capacity,
        (void**)&builder->operations));
  }
  id4_pipeline_program_op_t* target_op =
      &builder->operations[builder->operation_count];
  iree_status_t status = id4_pipeline_program_copy_op(
      source_op, builder->arena_allocator, target_op);
  if (iree_status_is_ok(status)) {
    ++builder->operation_count;
  }
  return status;
}

static iree_status_t id4_pipeline_program_builder_validate_create_options(
    const id4_pipeline_program_builder_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("program builder create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program builder create extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->program_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program name is required");
  }
  if (!options->block_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder block pool is required");
  }
  return iree_ok_status();
}

static void id4_pipeline_program_destroy(id4_pipeline_program_t* program) {
  iree_allocator_t host_allocator = program->host_allocator;
  for (iree_host_size_t i = 0; i < program->operation_count; ++i) {
    id4_pipeline_program_free_op(&program->operations[i], host_allocator);
  }
  iree_allocator_free(host_allocator, program->operations);
  for (iree_host_size_t i = 0; i < program->tensor_count; ++i) {
    id4_pipeline_program_free_tensor_record(&program->tensors[i],
                                            host_allocator);
  }
  iree_allocator_free(host_allocator, program->tensors);
  id4_pipeline_program_free_string(&program->name, host_allocator);
  iree_allocator_free(host_allocator, program);
}

iree_device_size_t id4_pipeline_program_dtype_byte_length(
    id4_pipeline_program_dtype_t dtype) {
  switch (dtype) {
    case ID4_PIPELINE_PROGRAM_DTYPE_F32:
    case ID4_PIPELINE_PROGRAM_DTYPE_I32:
    case ID4_PIPELINE_PROGRAM_DTYPE_U32:
      return 4;
    case ID4_PIPELINE_PROGRAM_DTYPE_F16:
    case ID4_PIPELINE_PROGRAM_DTYPE_BF16:
      return 2;
    default:
      return 0;
  }
}

iree_status_t id4_pipeline_program_shape_element_count(
    id4_pipeline_program_shape_t shape, uint64_t* out_element_count) {
  IREE_ASSERT_ARGUMENT(out_element_count);
  *out_element_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_validate_shape(shape, IREE_SV("<anonymous>")));
  uint64_t element_count = 1;
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (element_count > UINT64_MAX / shape.dims[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor element count overflow");
    }
    element_count *= shape.dims[i];
  }
  *out_element_count = element_count;
  return iree_ok_status();
}

iree_status_t id4_pipeline_program_tensor_byte_length(
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    iree_device_size_t* out_byte_length) {
  IREE_ASSERT_ARGUMENT(out_byte_length);
  *out_byte_length = 0;
  iree_device_size_t dtype_byte_length =
      id4_pipeline_program_dtype_byte_length(dtype);
  if (dtype_byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program tensor dtype %d is invalid", (int)dtype);
  }
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_shape_element_count(shape, &element_count));
  if (element_count > IREE_DEVICE_SIZE_MAX / dtype_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor byte length overflow");
  }
  *out_byte_length = (iree_device_size_t)element_count * dtype_byte_length;
  return iree_ok_status();
}

iree_status_t id4_pipeline_program_builder_create(
    const id4_pipeline_program_builder_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_program_builder_t** out_builder) {
  IREE_ASSERT_ARGUMENT(out_builder);
  *out_builder = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_builder_validate_create_options(options));

  id4_pipeline_program_builder_t* builder = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*builder),
                                             (void**)&builder));
  memset(builder, 0, sizeof(*builder));
  builder->host_allocator = host_allocator;
  iree_arena_initialize(options->block_pool, &builder->arena);
  builder->arena_allocator = iree_arena_allocator(&builder->arena);
  iree_status_t status = id4_pipeline_program_copy_string(
      options->program_name, builder->arena_allocator, &builder->program_name);
  if (iree_status_is_ok(status)) {
    *out_builder = builder;
  } else {
    id4_pipeline_program_builder_destroy(builder);
  }
  return status;
}

void id4_pipeline_program_builder_destroy(
    id4_pipeline_program_builder_t* builder) {
  if (!builder) return;
  iree_allocator_t host_allocator = builder->host_allocator;
  iree_arena_deinitialize(&builder->arena);
  iree_allocator_free(host_allocator, builder);
}

iree_status_t id4_pipeline_program_import_tensor(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_import_tensor_options_t* options,
    id4_pipeline_program_tensor_t* out_tensor) {
  IREE_ASSERT_ARGUMENT(out_tensor);
  *out_tensor = id4_pipeline_program_tensor_invalid();
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program import tensor options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("program import tensor")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program import tensor extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_import_flags(
      options->flags, options->name));

  iree_host_size_t operation_ordinal = builder->operation_count;
  id4_pipeline_program_tensor_t tensor = id4_pipeline_program_tensor_invalid();
  const bool initialized = iree_all_bits_set(
      options->flags, ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED);
  IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_add_tensor(
      builder, options->name, options->dtype, options->shape, operation_ordinal,
      initialized, &tensor));
  id4_pipeline_program_op_t op = {
      .kind = ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT,
      .ordinal = operation_ordinal,
      .payload.import_value =
          {
              .flags = options->flags,
              .tensor = tensor,
          },
  };
  iree_status_t status = id4_pipeline_program_builder_append_op(builder, &op);
  if (iree_status_is_ok(status)) {
    *out_tensor = tensor;
  } else {
    id4_pipeline_program_builder_remove_last_tensor(builder);
  }
  return status;
}

iree_status_t id4_pipeline_program_parameter(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_parameter_options_t* options,
    id4_pipeline_program_tensor_t* out_tensor) {
  IREE_ASSERT_ARGUMENT(out_tensor);
  *out_tensor = id4_pipeline_program_tensor_invalid();
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program parameter options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program parameter")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program parameter extension structures are not supported");
  }
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_tensor_metadata(
      options->key, options->dtype, options->shape, &byte_length));

  id4_pipeline_program_tensor_t existing_tensor =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_tensor_state_t* existing_state = NULL;
  if (id4_pipeline_program_builder_find_tensor_by_name(
          builder, options->key, &existing_tensor, &existing_state)) {
    if (existing_state->record.producer_operation_ordinal >=
        builder->operation_count) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "program tensor %.*s has invalid producer operation %" PRIhsz,
          (int)options->key.size, options->key.data,
          existing_state->record.producer_operation_ordinal);
    }
    const id4_pipeline_program_op_t* producer =
        &builder->operations[existing_state->record.producer_operation_ordinal];
    if (producer->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "program tensor %.*s already exists and is not a parameter",
          (int)options->key.size, options->key.data);
    }
    if (existing_state->record.dtype != options->dtype ||
        !id4_pipeline_program_shape_equal(existing_state->record.shape,
                                          options->shape) ||
        existing_state->record.byte_length != byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program parameter %.*s was requested with incompatible metadata",
          (int)options->key.size, options->key.data);
    }
    *out_tensor = existing_tensor;
    return iree_ok_status();
  }

  iree_host_size_t operation_ordinal = builder->operation_count;
  id4_pipeline_program_tensor_t tensor = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_add_tensor(
      builder, options->key, options->dtype, options->shape, operation_ordinal,
      /*initialized=*/true, &tensor));
  id4_pipeline_program_op_t op = {
      .kind = ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER,
      .ordinal = operation_ordinal,
      .payload.parameter.tensor = tensor,
  };
  iree_status_t status = id4_pipeline_program_builder_append_op(builder, &op);
  if (iree_status_is_ok(status)) {
    *out_tensor = tensor;
  } else {
    id4_pipeline_program_builder_remove_last_tensor(builder);
  }
  return status;
}

iree_status_t id4_pipeline_program_constant(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_constant_options_t* options,
    id4_pipeline_program_tensor_t* out_tensor) {
  IREE_ASSERT_ARGUMENT(out_tensor);
  *out_tensor = id4_pipeline_program_tensor_invalid();
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program constant options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program constant")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program constant extension structures are not supported");
  }
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_tensor_metadata(
      options->name, options->dtype, options->shape, &byte_length));
  if (options->data.data_length != byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program constant %.*s data length %" PRIhsz
                            " does not match tensor byte length %" PRIu64,
                            (int)options->name.size, options->name.data,
                            options->data.data_length, (uint64_t)byte_length);
  }
  if (byte_length != 0 && !options->data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program constant %.*s data is required",
                            (int)options->name.size, options->name.data);
  }

  iree_host_size_t operation_ordinal = builder->operation_count;
  id4_pipeline_program_tensor_t tensor = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_add_tensor(
      builder, options->name, options->dtype, options->shape, operation_ordinal,
      /*initialized=*/true, &tensor));
  id4_pipeline_program_op_t op = {
      .kind = ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT,
      .ordinal = operation_ordinal,
      .payload.constant =
          {
              .tensor = tensor,
              .data_length = options->data.data_length,
              .data = options->data.data,
          },
  };
  iree_status_t status = id4_pipeline_program_builder_append_op(builder, &op);
  if (iree_status_is_ok(status)) {
    *out_tensor = tensor;
  } else {
    id4_pipeline_program_builder_remove_last_tensor(builder);
  }
  return status;
}

iree_status_t id4_pipeline_program_acquire_tensor(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_acquire_tensor_options_t* options,
    id4_pipeline_program_tensor_t* out_tensor) {
  IREE_ASSERT_ARGUMENT(out_tensor);
  *out_tensor = id4_pipeline_program_tensor_invalid();
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program acquire tensor options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("program acquire tensor")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program acquire tensor extension structures are not supported");
  }

  iree_host_size_t operation_ordinal = builder->operation_count;
  id4_pipeline_program_tensor_t tensor = id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_add_tensor(
      builder, options->name, options->dtype, options->shape, operation_ordinal,
      /*initialized=*/false, &tensor));
  id4_pipeline_program_op_t op = {
      .kind = ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE,
      .ordinal = operation_ordinal,
      .payload.acquire.tensor = tensor,
  };
  iree_status_t status = id4_pipeline_program_builder_append_op(builder, &op);
  if (iree_status_is_ok(status)) {
    *out_tensor = tensor;
  } else {
    id4_pipeline_program_builder_remove_last_tensor(builder);
  }
  return status;
}

iree_status_t id4_pipeline_program_dispatch_loom(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_dispatch_loom_options_t* options) {
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program dispatch options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program dispatch")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program dispatch extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program dispatch name is required");
  }
  if (id4_pipeline_program_builder_has_named_op(
          builder, ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM, options->name)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "program dispatch %.*s already exists",
                            (int)options->name.size, options->name.data);
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_validate_kernel_ref(options->kernel, options->name));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_config_bindings(
      options->name, options->config_binding_count, options->config_bindings));
  if (options->binding_count == 0 || !options->bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch %.*s binding array is required",
                            (int)options->name.size, options->name.data);
  }
  for (iree_host_size_t i = 0; i < options->binding_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_access(
        options->bindings[i].access, options->name, i));
    IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_dispatch_binding_flags(
        options->bindings[i].flags, options->name, i));
    id4_pipeline_program_tensor_state_t* state = NULL;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_validate_tensor(
        builder, options->bindings[i].tensor, &state));
    IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_dispatch_binding_range(
        &options->bindings[i], &state->record, options->name, i));
    if (iree_any_bit_set(options->bindings[i].access,
                         ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ) &&
        !state->initialized) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "dispatch %.*s reads uninitialized tensor %.*s",
                              (int)options->name.size, options->name.data,
                              (int)state->record.name.size,
                              state->record.name.data);
    }
  }

  id4_pipeline_program_op_t op = {
      .kind = ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM,
      .ordinal = builder->operation_count,
      .payload.dispatch_loom =
          {
              .name = options->name,
              .kernel = options->kernel,
              .config_binding_count = options->config_binding_count,
              .config_bindings = options->config_bindings,
              .binding_count = options->binding_count,
              .bindings = options->bindings,
          },
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_append_op(builder, &op));
  for (iree_host_size_t i = 0; i < options->binding_count; ++i) {
    if (!iree_any_bit_set(options->bindings[i].access,
                          ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE)) {
      continue;
    }
    id4_pipeline_program_tensor_state_t* state = NULL;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_validate_tensor(
        builder, options->bindings[i].tensor, &state));
    state->initialized = true;
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_program_barrier(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_barrier_options_t* options) {
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program barrier options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program barrier")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program barrier extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program barrier name is required");
  }
  if (id4_pipeline_program_builder_has_named_op(
          builder, ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER, options->name)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "program barrier %.*s already exists",
                            (int)options->name.size, options->name.data);
  }

  id4_pipeline_program_op_t op = {
      .kind = ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER,
      .ordinal = builder->operation_count,
      .payload.barrier.name = options->name,
  };
  return id4_pipeline_program_builder_append_op(builder, &op);
}

iree_status_t id4_pipeline_program_tap(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_tap_options_t* options) {
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program tap options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program tap")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program tap extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program tap name is required");
  }
  if (id4_pipeline_program_builder_has_named_op(
          builder, ID4_PIPELINE_PROGRAM_OP_KIND_TAP, options->name)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "program tap %.*s already exists",
                            (int)options->name.size, options->name.data);
  }
  id4_pipeline_program_tensor_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_validate_tensor(
      builder, options->tensor, &state));
  if (!state->initialized) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "tap %.*s captures uninitialized tensor %.*s",
                            (int)options->name.size, options->name.data,
                            (int)state->record.name.size,
                            state->record.name.data);
  }

  id4_pipeline_program_op_t op = {
      .kind = ID4_PIPELINE_PROGRAM_OP_KIND_TAP,
      .ordinal = builder->operation_count,
      .payload.tap =
          {
              .name = options->name,
              .tensor = options->tensor,
          },
  };
  return id4_pipeline_program_builder_append_op(builder, &op);
}

iree_status_t id4_pipeline_program_export(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_export_options_t* options) {
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program export options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("program export")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program export extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program export name is required");
  }
  if (id4_pipeline_program_builder_has_named_op(
          builder, ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT, options->name)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "program export %.*s already exists",
                            (int)options->name.size, options->name.data);
  }
  id4_pipeline_program_tensor_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_builder_validate_tensor(
      builder, options->tensor, &state));
  if (!state->initialized) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "export %.*s captures uninitialized tensor %.*s",
                            (int)options->name.size, options->name.data,
                            (int)state->record.name.size,
                            state->record.name.data);
  }

  id4_pipeline_program_op_t op = {
      .kind = ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT,
      .ordinal = builder->operation_count,
      .payload.export_value =
          {
              .name = options->name,
              .tensor = options->tensor,
          },
  };
  return id4_pipeline_program_builder_append_op(builder, &op);
}

iree_status_t id4_pipeline_program_builder_seal(
    const id4_pipeline_program_builder_t* builder,
    iree_allocator_t host_allocator, id4_pipeline_program_t** out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = NULL;
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program builder is required");
  }
  if (builder->operation_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION, "program %.*s has no operations",
        (int)builder->program_name.size, builder->program_name.data);
  }

  id4_pipeline_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*program),
                                             (void**)&program));
  memset(program, 0, sizeof(*program));
  iree_atomic_ref_count_init(&program->ref_count);
  program->host_allocator = host_allocator;

  iree_status_t status = id4_pipeline_program_copy_string(
      builder->program_name, host_allocator, &program->name);
  if (iree_status_is_ok(status) && builder->tensor_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, builder->tensor_count,
                                         sizeof(program->tensors[0]),
                                         (void**)&program->tensors);
    if (iree_status_is_ok(status)) {
      memset(program->tensors, 0,
             builder->tensor_count * sizeof(program->tensors[0]));
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < builder->tensor_count && iree_status_is_ok(status); ++i) {
      status = id4_pipeline_program_copy_tensor_record(
          &builder->tensor_states[i].record, host_allocator,
          &program->tensors[i]);
      if (iree_status_is_ok(status)) {
        ++program->tensor_count;
      }
    }
  }
  if (iree_status_is_ok(status) && builder->operation_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, builder->operation_count,
        sizeof(program->operations[0]), (void**)&program->operations);
    if (iree_status_is_ok(status)) {
      memset(program->operations, 0,
             builder->operation_count * sizeof(program->operations[0]));
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < builder->operation_count && iree_status_is_ok(status); ++i) {
      status = id4_pipeline_program_copy_op(
          &builder->operations[i], host_allocator, &program->operations[i]);
      if (iree_status_is_ok(status)) {
        ++program->operation_count;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_program = program;
  } else {
    id4_pipeline_program_destroy(program);
  }
  return status;
}

void id4_pipeline_program_retain(id4_pipeline_program_t* program) {
  if (!program) return;
  iree_atomic_ref_count_inc(&program->ref_count);
}

void id4_pipeline_program_release(id4_pipeline_program_t* program) {
  if (program && iree_atomic_ref_count_dec(&program->ref_count) == 1) {
    id4_pipeline_program_destroy(program);
  }
}

iree_string_view_t id4_pipeline_program_name(
    const id4_pipeline_program_t* program) {
  return program ? program->name : iree_string_view_empty();
}

iree_host_size_t id4_pipeline_program_tensor_count(
    const id4_pipeline_program_t* program) {
  return program ? program->tensor_count : 0;
}

const id4_pipeline_program_tensor_record_t* id4_pipeline_program_tensor_at(
    const id4_pipeline_program_t* program, iree_host_size_t index) {
  if (!program || index >= program->tensor_count) return NULL;
  return &program->tensors[index];
}

iree_host_size_t id4_pipeline_program_operation_count(
    const id4_pipeline_program_t* program) {
  return program ? program->operation_count : 0;
}

const id4_pipeline_program_op_t* id4_pipeline_program_operation_at(
    const id4_pipeline_program_t* program, iree_host_size_t index) {
  if (!program || index >= program->operation_count) return NULL;
  return &program->operations[index];
}
