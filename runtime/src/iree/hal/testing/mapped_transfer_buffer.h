// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TESTING_MAPPED_TRANSFER_BUFFER_H_
#define IREE_HAL_TESTING_MAPPED_TRANSFER_BUFFER_H_

#include <array>
#include <cstring>
#include <memory>

#include "iree/hal/buffer.h"

namespace iree::hal::testing {

// Models a device allocation whose host cache changes only at the buffer's
// explicit invalidate/flush boundary. Coherent mappings expose device bytes
// directly; noncoherent mappings expose an independently stale host cache.
struct MappedTransferBuffer : iree_hal_buffer_t {
  // Bytes visible to completed device work.
  std::array<uint8_t, 64> device_contents = {};
  // Stale host cache for noncoherent mappings.
  std::array<uint8_t, 64> host_contents = {};
  // Prepared CPU execution view, independent of public mapping permission.
  iree_hal_buffer_native_binding_t host_binding = {};
  // Outstanding mappings that must be returned even after a cache error.
  size_t mapping_count = 0;
  // Source transitions performed by the caller.
  size_t invalidate_count = 0;
  // Target transitions performed by the caller.
  size_t flush_count = 0;
  // Last exact native maintenance range, including the mapped view offset.
  struct {
    // Byte offset from the native allocation base.
    iree_device_size_t offset = 0;
    // Number of bytes passed to the native operation.
    iree_device_size_t length = 0;
  } maintenance_range;
  // Injected native invalidation result, without owning a status allocation.
  iree_status_code_t invalidate_code = IREE_STATUS_OK;
  // Injected native publication result, without owning a status allocation.
  iree_status_code_t flush_code = IREE_STATUS_OK;

  explicit MappedTransferBuffer(iree_hal_memory_type_t coherence) {
    static const iree_hal_buffer_vtable_t vtable = {
        .destroy =
            [](iree_hal_buffer_t* base) {
              delete static_cast<MappedTransferBuffer*>(base);
            },
        .export_range = nullptr,
        .map_range =
            [](iree_hal_buffer_t* base, iree_hal_mapping_mode_t,
               iree_hal_memory_access_t, iree_hal_buffer_map_flags_t,
               iree_device_size_t offset, iree_device_size_t length,
               iree_hal_buffer_mapping_t* mapping) {
              auto* buffer = static_cast<MappedTransferBuffer*>(base);
              ++buffer->mapping_count;
              auto& contents =
                  iree_all_bits_set(iree_hal_buffer_memory_type(base),
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT)
                      ? buffer->device_contents
                      : buffer->host_contents;
              mapping->contents =
                  iree_make_byte_span(contents.data() + offset, length);
              return iree_ok_status();
            },
        .unmap_range =
            [](iree_hal_buffer_t* base, iree_device_size_t, iree_device_size_t,
               iree_hal_buffer_mapping_t*) {
              --static_cast<MappedTransferBuffer*>(base)->mapping_count;
              return iree_ok_status();
            },
        .invalidate_range =
            [](iree_hal_buffer_t* base, iree_device_size_t offset,
               iree_device_size_t length) {
              auto* buffer = static_cast<MappedTransferBuffer*>(base);
              ++buffer->invalidate_count;
              buffer->maintenance_range = {offset, length};
              if (buffer->invalidate_code != IREE_STATUS_OK) {
                return iree_status_from_code(buffer->invalidate_code);
              }
              memcpy(buffer->host_contents.data() + offset,
                     buffer->device_contents.data() + offset, length);
              return iree_ok_status();
            },
        .flush_range =
            [](iree_hal_buffer_t* base, iree_device_size_t offset,
               iree_device_size_t length) {
              auto* buffer = static_cast<MappedTransferBuffer*>(base);
              ++buffer->flush_count;
              buffer->maintenance_range = {offset, length};
              if (buffer->flush_code != IREE_STATUS_OK) {
                return iree_status_from_code(buffer->flush_code);
              }
              if (!iree_all_bits_set(iree_hal_buffer_memory_type(base),
                                     IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
                memcpy(buffer->device_contents.data() + offset,
                       buffer->host_contents.data() + offset, length);
              }
              return iree_ok_status();
            },
        .query_memory = nullptr,
        .allocation = nullptr,
    };
    iree_hal_buffer_initialize(
        iree_hal_buffer_placement_undefined(), this, device_contents.size(), 0,
        device_contents.size(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL | coherence,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING, &vtable,
        this);
    host_binding.host_pointer =
        iree_all_bits_set(coherence, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)
            ? device_contents.data()
            : host_contents.data();
    host_binding_index = 0;
    memory.bindings = &host_binding;
  }
};

using MappedTransferBufferPtr =
    std::unique_ptr<MappedTransferBuffer, decltype(&iree_hal_buffer_release)>;

inline MappedTransferBufferPtr MakeMappedTransferBuffer(
    iree_hal_memory_type_t coherence = 0) {
  return MappedTransferBufferPtr(new MappedTransferBuffer(coherence),
                                 iree_hal_buffer_release);
}

}  // namespace iree::hal::testing

#endif  // IREE_HAL_TESTING_MAPPED_TRANSFER_BUFFER_H_
