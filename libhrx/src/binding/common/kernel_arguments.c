// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/kernel_arguments.h"

#include <inttypes.h>
#include <string.h>

iree_status_t iree_hal_streaming_validate_prepacked_kernel_arguments(
    const iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  if (params->buffer_size > 0 && !params->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pre-packed kernel arguments require storage when length is non-zero");
  }

  const iree_host_size_t required_size = symbol->parameters.direct_arg_bytes;
  if (params->buffer_size < required_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pre-packed kernel arguments are shorter than the native argument "
        "extent (%" PRIhsz " < %" PRIhsz ")",
        params->buffer_size, required_size);
  }
  return iree_ok_status();
}

// Loads fixed-width device pointer bytes without imposing a host lvalue type or
// alignment requirement on caller-owned kernel argument storage.
static iree_hal_streaming_deviceptr_t
iree_hal_streaming_load_device_pointer_bytes(const void* source) {
  iree_hal_streaming_deviceptr_t value = 0;
  memcpy(&value, source, sizeof(value));
  return value;
}

static bool iree_hal_streaming_buffer_can_import_for_context(
    const iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) return false;
  if (buffer->is_managed) return true;
  return iree_all_bits_set(
      (iree_hal_memory_type_t)buffer->memory_type,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
}

static iree_status_t iree_hal_streaming_device_buffer_for_context(
    iree_hal_streaming_context_t* context, iree_hal_streaming_buffer_t* buffer,
    iree_hal_buffer_t** out_buffer,
    iree_hal_streaming_deviceptr_t* out_device_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (out_device_ptr) *out_device_ptr = 0;

  if (buffer->context == context) {
    *out_buffer = buffer->buffer;
    if (out_device_ptr) *out_device_ptr = buffer->device_ptr;
    return iree_ok_status();
  }
  if (!iree_hal_streaming_buffer_can_import_for_context(buffer)) {
    return iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }
  if (buffer->is_managed &&
      (!buffer->host_ptr ||
       (iree_hal_streaming_deviceptr_t)(uintptr_t)buffer->host_ptr !=
           buffer->device_ptr)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "cross-device managed memory requires one stable host/device address");
  }
  if (!buffer->buffer || buffer->device_ptr == 0 || buffer->size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation is missing device import metadata");
  }
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&buffer->context_import_mutex);
  for (iree_hal_streaming_context_import_t* import = buffer->context_imports;
       import; import = import->next) {
    if (import->context == context) {
      *out_buffer = import->buffer;
      if (out_device_ptr) *out_device_ptr = buffer->device_ptr;
      iree_slim_mutex_unlock(&buffer->context_import_mutex);
      return iree_ok_status();
    }
  }

  iree_hal_buffer_t* imported_buffer = NULL;
  const bool import_host_allocation = iree_all_bits_set(
      (iree_hal_memory_type_t)buffer->memory_type,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  iree_hal_buffer_params_t params = {
      .usage = iree_hal_buffer_allowed_usage(buffer->buffer),
      .access = iree_hal_buffer_allowed_access(buffer->buffer),
      .type = (iree_hal_memory_type_t)buffer->memory_type,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      .min_alignment = 0,
  };
  iree_hal_external_buffer_t external_buffer = {
      .type = import_host_allocation
                  ? IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION
                  : IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
      .size = buffer->size,
  };
  if (import_host_allocation) {
    external_buffer.handle.host_allocation.ptr = buffer->host_ptr;
  } else {
    external_buffer.handle.device_allocation.ptr = buffer->device_ptr;
  }
  status = iree_hal_allocator_import_buffer(
      context->device_allocator, params, &external_buffer,
      iree_hal_buffer_release_callback_null(), &imported_buffer);

  iree_hal_streaming_context_import_t* import = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(buffer->context->host_allocator,
                                   sizeof(*import), (void**)&import);
  }
  if (iree_status_is_ok(status)) {
    import->next = buffer->context_imports;
    import->context = context;
    iree_hal_streaming_context_retain(context);
    import->buffer = imported_buffer;
    buffer->context_imports = import;
    imported_buffer = NULL;
    *out_buffer = import->buffer;
    if (out_device_ptr) *out_device_ptr = buffer->device_ptr;
  }
  iree_slim_mutex_unlock(&buffer->context_import_mutex);
  iree_hal_buffer_release(imported_buffer);
  return status;
}

static iree_status_t iree_hal_streaming_lookup_kernel_buffer_ref(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_pointer,
    iree_hal_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  *out_ref = (iree_hal_buffer_ref_t){0};

  // This resolves explicit pointer arguments described by kernel metadata into
  // HAL buffer bindings. It is not a lifetime analysis: HIP device pointers can
  // be hidden in opaque kernarg bytes or device memory, so free/unregister
  // paths must conservatively order destruction without relying on lookup
  // coverage.
  iree_hal_streaming_buffer_ref_t stream_ref;
  iree_hal_streaming_context_t* owner_context = NULL;
  iree_status_t status =
      iree_hal_streaming_memory_lookup(context, device_pointer, &stream_ref);
  if (iree_status_is_ok(status) ||
      iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    if (!iree_status_is_ok(status)) return status;
  } else {
    iree_status_ignore(status);
    if (!iree_hal_streaming_context_has_peer_contexts(context)) {
      return iree_status_from_code(IREE_STATUS_NOT_FOUND);
    }
    status = iree_hal_streaming_memory_lookup_range_across_contexts(
        device_pointer, 1, &owner_context, &stream_ref);
    if (!iree_status_is_ok(status)) return status;

    if (!iree_hal_streaming_buffer_can_import_for_context(stream_ref.buffer)) {
      iree_hal_streaming_context_release(owner_context);
      return iree_status_from_code(IREE_STATUS_NOT_FOUND);
    }
  }

  iree_hal_buffer_t* device_buffer = NULL;
  status = iree_hal_streaming_device_buffer_for_context(
      context, stream_ref.buffer, &device_buffer, NULL);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_release(owner_context);
    return status;
  }
  const iree_device_size_t length =
      stream_ref.offset < stream_ref.buffer->size
          ? stream_ref.buffer->size - stream_ref.offset
          : 0;
  *out_ref = iree_hal_make_buffer_ref(device_buffer, stream_ref.offset, length);
  iree_hal_streaming_context_release(owner_context);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_unpack_parameters(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_parameter_info_t* parameters,
    const void* parameter_buffer_ptr, void* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(parameters);
  if (iree_hal_streaming_parameter_info_is_empty(parameters)) {
    return iree_ok_status();
  }
  const bool requires_parameter_storage = parameters->buffer_size > 0 ||
                                          parameters->binding_count > 0 ||
                                          parameters->copy_count > 0;
  if (!requires_parameter_storage) {
    return iree_ok_status();
  }
  if (!parameter_buffer_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel parameter buffer is required");
  }
  if (parameters->copy_count > 0 && !out_constants) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel constant storage is required");
  }
  if (!out_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel binding list is required");
  }
  if (parameters->binding_count > 0 && !out_bindings->values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel binding storage is required");
  }

  const uint8_t* parameter_buffer = (const uint8_t*)parameter_buffer_ptr;

  // Copy constant data spans into the HAL constants table. Native ABI packing
  // uses a separate destination offset so dense constants do not imply ABI
  // layout.
  uint8_t* constants = (uint8_t*)out_constants;
  const iree_hal_streaming_parameter_op_t* op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    if (copy_op.size > 0) {
      memcpy(constants + copy_op.constant_destination_offset,
             parameter_buffer + copy_op.source_offset, copy_op.size);
    }
  }

  // Resolve bindings, if any.
  // A NULL HIP kernel pointer is a valid literal kernarg for optional buffers.
  // Represent it as a zeroed direct binding; the AMDGPU direct queue path
  // materializes that as a zero pointer in the final kernarg block.
  iree_hal_buffer_ref_t* bindings =
      (iree_hal_buffer_ref_t*)out_bindings->values;
  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    const iree_hal_streaming_deviceptr_t device_pointer =
        iree_hal_streaming_load_device_pointer_bytes(parameter_buffer +
                                                     resolve_op.source_offset);
    // Kernel metadata identifies pointer slots but not the dynamic object
    // extent. Resolve with an unknown length; the HAL buffer reference owns
    // the allocation for dispatch.

    if (!device_pointer) {
      bindings[resolve_op.destination_ordinal] = (iree_hal_buffer_ref_t){0};
      continue;
    }

    iree_status_t lookup_status = iree_hal_streaming_lookup_kernel_buffer_ref(
        context, device_pointer, &bindings[resolve_op.destination_ordinal]);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_pack_raw_argument_list(
    const iree_hal_streaming_parameter_info_t* parameters,
    void** parameter_list, void* out_constants,
    iree_host_size_t* out_constants_size) {
  IREE_ASSERT_ARGUMENT(parameters);
  IREE_ASSERT_ARGUMENT(out_constants_size);

  if (iree_hal_streaming_parameter_info_is_empty(parameters)) {
    *out_constants_size = 0;
    return iree_ok_status();
  }

  *out_constants_size = parameters->direct_arg_bytes
                            ? parameters->direct_arg_bytes
                            : parameters->constant_bytes;
  if (*out_constants_size == 0) {
    *out_constants_size = parameters->buffer_size;
  }
  if (*out_constants_size == 0) return iree_ok_status();
  if (!out_constants || (!parameter_list && (parameters->buffer_size > 0 ||
                                             parameters->binding_count > 0 ||
                                             parameters->copy_count > 0))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "raw kernel arguments require parameter storage");
  }
  if ((parameters->copy_count > 0 || parameters->binding_count > 0) &&
      !parameters->ops) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "raw kernel argument operations are missing");
  }
  const uint32_t parameter_count =
      (uint32_t)parameters->copy_count + parameters->binding_count;
  if (IREE_UNLIKELY(parameter_count > UINT16_MAX)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "raw kernel argument operation count exceeds the ordinal range");
  }

  uint8_t* constants = (uint8_t*)out_constants;
  memset(constants, 0, *out_constants_size);
  const iree_hal_streaming_parameter_op_t* copy_ops = parameters->ops;
  const iree_hal_streaming_parameter_op_t* resolve_ops =
      parameters->ops + parameters->copy_count;
  iree_host_size_t copy_index = 0;
  iree_host_size_t resolve_index = 0;
  bool has_previous_source_ordinal = false;
  uint16_t previous_source_ordinal = 0;
  iree_host_size_t written_end = 0;
  while (copy_index < parameters->copy_count ||
         resolve_index < parameters->binding_count) {
    bool use_copy = false;
    if (copy_index == parameters->copy_count) {
      use_copy = false;
    } else if (resolve_index == parameters->binding_count) {
      use_copy = true;
    } else {
      const uint16_t copy_source_ordinal =
          copy_ops[copy_index].copy.source_ordinal;
      const uint16_t resolve_source_ordinal =
          resolve_ops[resolve_index].resolve.source_ordinal;
      if (IREE_UNLIKELY(copy_source_ordinal == resolve_source_ordinal)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "kernel argument source ordinal %u appears more than once",
            copy_source_ordinal);
      }
      use_copy = copy_source_ordinal < resolve_source_ordinal;
    }

    const uint16_t source_ordinal =
        use_copy ? copy_ops[copy_index].copy.source_ordinal
                 : resolve_ops[resolve_index].resolve.source_ordinal;
    if (IREE_UNLIKELY(source_ordinal >= parameter_count ||
                      (has_previous_source_ordinal &&
                       source_ordinal <= previous_source_ordinal))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "kernel argument source ordinals are out of range or not strictly "
          "ordered");
    }
    has_previous_source_ordinal = true;
    previous_source_ordinal = source_ordinal;

    const iree_host_size_t destination_offset =
        use_copy
            ? copy_ops[copy_index].copy.native_abi_destination_offset
            : resolve_ops[resolve_index].resolve.native_abi_destination_offset;
    const iree_host_size_t argument_size =
        use_copy ? copy_ops[copy_index].copy.size
                 : sizeof(iree_hal_streaming_deviceptr_t);
    if (IREE_UNLIKELY(destination_offset < written_end ||
                      destination_offset > *out_constants_size ||
                      argument_size >
                          *out_constants_size - destination_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "kernel argument layout overlaps or exceeds kernarg size");
    }

    void* param_ptr = parameter_list[source_ordinal];
    if (!param_ptr) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel argument %" PRIu32 " is NULL",
                              (uint32_t)source_ordinal);
    }
    if (use_copy) {
      memcpy(constants + destination_offset, param_ptr, argument_size);
      ++copy_index;
    } else {
      const iree_hal_streaming_deviceptr_t device_pointer =
          iree_hal_streaming_load_device_pointer_bytes(param_ptr);
      memcpy(constants + destination_offset, &device_pointer,
             sizeof(device_pointer));
      ++resolve_index;
    }
    written_end = destination_offset + argument_size;
  }

  return iree_ok_status();
}
