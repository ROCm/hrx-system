// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/vm/buffer.h"

#include "iree/vm/buffer_provider.h"

typedef struct loom_vm_buffer_mapping_t {
  // Persistent mapping borrowing the independently retained HAL buffer.
  iree_hal_buffer_mapping_t mapping;
  // Start of the mapping relative to the retained HAL view.
  iree_device_size_t byte_offset;
  // Allocator owning this mapping descriptor.
  iree_allocator_t host_allocator;
} loom_vm_buffer_mapping_t;

static void loom_vm_buffer_release_mapping(void* user_data,
                                           iree_byte_span_t storage) {
  loom_vm_buffer_mapping_t* mapping = user_data;
  iree_hal_buffer_release(mapping->mapping.buffer);
  iree_allocator_free(mapping->host_allocator, mapping);
}

// VM views retain a flattened root. Its storage callback identifies this
// adapter without relying on the invocation or its original argument list.
static loom_vm_buffer_mapping_t* loom_vm_buffer_mapping(
    iree_vm_buffer_t* buffer) {
  if (!buffer) {
    return NULL;
  }
  iree_vm_buffer_t* root = buffer->root ? buffer->root : buffer;
  return root->release_callback.function == loom_vm_buffer_release_mapping
             ? root->release_callback.user_data
             : NULL;
}

static iree_device_size_t loom_vm_buffer_mapping_offset(
    const loom_vm_buffer_mapping_t* mapping, const iree_vm_buffer_t* buffer) {
  return buffer->data ? (iree_device_size_t)(buffer->data -
                                             mapping->mapping.contents.data)
                      : 0;
}

iree_status_t loom_vm_buffer_import(iree_hal_buffer_t* source,
                                    iree_device_size_t byte_offset,
                                    iree_device_size_t byte_length,
                                    iree_allocator_t host_allocator,
                                    iree_vm_buffer_t** out_buffer) {
  *out_buffer = NULL;
  const iree_hal_memory_access_t access =
      iree_hal_buffer_allowed_access(source);
  iree_hal_buffer_mapping_t mapped = {0};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      source, IREE_HAL_MAPPING_MODE_PERSISTENT, access,
      IREE_HAL_BUFFER_MAP_FLAG_MAY_ALIAS, byte_offset, byte_length, &mapped));
  loom_vm_buffer_mapping_t* mapping = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*mapping),
                                             (void**)&mapping));
  *mapping = (loom_vm_buffer_mapping_t){
      .mapping = mapped,
      .byte_offset = byte_offset,
      .host_allocator = host_allocator,
  };
  iree_vm_buffer_access_flags_t vm_access = 0;
  if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
    vm_access |= IREE_VM_BUFFER_ACCESS_FLAG_READ;
  }
  if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    vm_access |= IREE_VM_BUFFER_ACCESS_FLAG_WRITE;
  }
  iree_status_t status =
      iree_vm_buffer_wrap(vm_access, mapped.contents,
                          (iree_vm_buffer_release_callback_t){
                              .function = loom_vm_buffer_release_mapping,
                              .user_data = mapping,
                          },
                          host_allocator, out_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_retain(source);
  } else {
    iree_allocator_free(host_allocator, mapping);
  }
  return status;
}

iree_status_t loom_vm_buffer_invalidate(iree_vm_buffer_t* buffer) {
  loom_vm_buffer_mapping_t* mapping = loom_vm_buffer_mapping(buffer);
  if (!mapping ||
      !iree_any_bit_set(iree_vm_buffer_access(buffer),
                        IREE_VM_BUFFER_ACCESS_FLAG_READ) ||
      iree_any_bit_set(iree_hal_buffer_memory_type(mapping->mapping.buffer),
                       IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    return iree_ok_status();
  }
  return iree_hal_buffer_mapping_invalidate_range(
      &mapping->mapping, loom_vm_buffer_mapping_offset(mapping, buffer),
      iree_vm_buffer_length(buffer));
}

iree_status_t loom_vm_buffer_flush(iree_vm_buffer_t* buffer) {
  loom_vm_buffer_mapping_t* mapping = loom_vm_buffer_mapping(buffer);
  if (!mapping ||
      !iree_any_bit_set(iree_vm_buffer_access(buffer),
                        IREE_VM_BUFFER_ACCESS_FLAG_WRITE) ||
      iree_any_bit_set(iree_hal_buffer_memory_type(mapping->mapping.buffer),
                       IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    return iree_ok_status();
  }
  return iree_hal_buffer_mapping_flush_range(
      &mapping->mapping, loom_vm_buffer_mapping_offset(mapping, buffer),
      iree_vm_buffer_length(buffer));
}

static void loom_vm_buffer_release_storage(void* user_data,
                                           iree_hal_buffer_t* buffer) {
  iree_vm_buffer_release(user_data);
}

iree_status_t loom_vm_buffer_export(iree_vm_buffer_t* source,
                                    iree_allocator_t host_allocator,
                                    iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;
  const iree_host_size_t length = iree_vm_buffer_length(source);
  loom_vm_buffer_mapping_t* mapping = loom_vm_buffer_mapping(source);
  if (mapping) {
    return iree_hal_buffer_subspan(
        mapping->mapping.buffer,
        mapping->byte_offset + loom_vm_buffer_mapping_offset(mapping, source),
        length, host_allocator, out_buffer);
  }
  const iree_vm_buffer_access_flags_t vm_access = iree_vm_buffer_access(source);
  iree_hal_memory_access_t access = IREE_HAL_MEMORY_ACCESS_NONE;
  if (iree_any_bit_set(vm_access, IREE_VM_BUFFER_ACCESS_FLAG_READ)) {
    access |= IREE_HAL_MEMORY_ACCESS_READ;
  }
  if (iree_any_bit_set(vm_access, IREE_VM_BUFFER_ACCESS_FLAG_WRITE)) {
    access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  }
  void* data = iree_any_bit_set(vm_access, IREE_VM_BUFFER_ACCESS_FLAG_WRITE)
                   ? iree_vm_buffer_data(source)
                   : (void*)iree_vm_buffer_const_data(source);
  IREE_RETURN_IF_ERROR(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      access,
      IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      length, iree_make_byte_span(data, length),
      (iree_hal_buffer_release_callback_t){
          .fn = loom_vm_buffer_release_storage,
          .user_data = source,
      },
      host_allocator, out_buffer));
  iree_vm_buffer_retain(source);
  return iree_ok_status();
}
