// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/memory.h"

#include <drm/amdxdna_accel.h>
#include <limits.h>
#include <stdint.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/dma_buf.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"
#include "libamdf/src/xdna/umd/drm/context.h"
#include "libamdf/src/xdna/umd/drm/device.h"
#include "libamdf/src/xdna/umd/drm/memory.h"
#include "libamdf/src/xdna/umd/drm/memory_profile.h"

struct amdf_xdna_umd_host_mapping_t {
  // Host allocator copied for independent mapping teardown.
  amdf_allocator_t host_allocator;
  // View into the attachment's persistent mapping, never independently
  // unmapped.
  void* pointer;
  // Native cache-line length in bytes, qualified during device creation.
  uint32_t cache_line_size;
};

amdf_status_t amdf_xdna_umd_memory_destroy(amdf_xdna_umd_memory_t* memory) {
  const amdf_status_t status = amdf_linux_xdna_buffer_deinitialize(
      memory->device->descriptor, &memory->buffer);
  if (amdf_status_is_ok(status)) {
    amdf_free(memory->device->host_allocator, memory);
  }
  return status;
}

void amdf_xdna_umd_memory_abandon(amdf_xdna_umd_memory_t* memory) {
  amdf_free(memory->device->host_allocator, memory);
}

static amdf_status_t amdf_linux_xdna_memory_query_dma_buf(
    amdf_xdna_umd_memory_t* memory, amdf_linux_dma_buf_info_t* out_info) {
  int descriptor = -1;
  amdf_status_t status = amdf_linux_xdna_buffer_export_dma_buf(
      memory->device->descriptor, &memory->buffer, &descriptor);
  amdf_linux_dma_buf_info_t info;
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_dma_buf_query(descriptor, &info);
  }
  if (amdf_status_is_ok(status) &&
      info.byte_length != memory->buffer.byte_length) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
  if (!amdf_status_is_ok(close_status)) status = close_status;
  if (amdf_status_is_ok(status)) *out_info = info;
  return status;
}

// The native mapping is external input: its complete translated range must fit
// the target's shim DMA aperture before any public address is published.
amdf_status_t amdf_linux_xdna_memory_translate_dma_address(
    const amdf_xdna_umd_memory_t* memory, uint64_t byte_offset,
    uint64_t byte_length, uint64_t* out_address) {
  const amdf_xdna_endpoint_profile_t* profile = memory->device->profile;
  const uint64_t maximum_address =
      ((UINT64_C(1) << profile->dma.address_bit_count) - 1) -
      profile->dma.byte_offset;
  if (memory->buffer.device_address > maximum_address ||
      byte_offset > maximum_address - memory->buffer.device_address ||
      byte_length - 1 >
          maximum_address - memory->buffer.device_address - byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  *out_address =
      memory->buffer.device_address + byte_offset + profile->dma.byte_offset;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_device_query_memory_profile(
    amdf_xdna_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  return amdf_linux_xdna_query_memory_profile(
      device->profile, device->page_size, memory_profile_ordinal, out_profile);
}

void amdf_xdna_umd_context_query_memory_profile(
    amdf_xdna_umd_context_t* context,
    amdf_memory_native_profile_t* out_profile) {
  const uint64_t page_size = context->device->page_size;
  const uint64_t heap_byte_length = context->device->heap.byte_length;
  const uint64_t maximum_address =
      (UINT64_C(1) << context->device->profile->dma.address_bit_count) - 1;
  *out_profile = (amdf_memory_native_profile_t){
      .ordinal = 0,
      .memory_class = AMDF_MEMORY_CLASS_PRIVATE,
      .roles =
          AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      .guaranteed_flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .supported_flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .guaranteed_device_access = AMDF_MEMORY_ACCESS_READ |
                                  AMDF_MEMORY_ACCESS_WRITE |
                                  AMDF_MEMORY_ACCESS_EXECUTE,
      .supported_device_access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE,
      .device_address =
          {
              .address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE,
              .address_bit_count =
                  context->device->profile->dma.address_bit_count,
              .maximum_address = maximum_address,
              .minimum_alignment = page_size,
          },
      .allocation =
          {
              .maximum_byte_length = heap_byte_length,
              .byte_length_granularity = 1,
              .minimum_alignment = page_size,
              .maximum_alignment = page_size,
              .native_byte_length_granularity = page_size,
          },
      .host_mapping =
          {
              .maximum_byte_length = heap_byte_length,
              .byte_offset_granularity = 1,
              .byte_length_granularity = 1,
              .supported_access =
                  AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
          },
      .address_kinds = (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE) |
                       (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA),
  };
}

amdf_status_t amdf_xdna_umd_memory_prepare_private(
    amdf_xdna_umd_context_t* context,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result) {
  amdf_xdna_umd_device_t* device = context->device;
  amdf_xdna_umd_memory_t* memory = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*memory),
                  amdf_alignof(amdf_xdna_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) return status;
  memory->device = device;
  *memory_state = memory;
  const size_t byte_length =
      (create_info->byte_length + device->page_size - 1) &
      ~(device->page_size - 1);
  status = amdf_linux_xdna_buffer_create(device->descriptor, AMDXDNA_BO_DEV,
                                         byte_length, &memory->buffer);
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_buffer_attach(device->descriptor,
                                           device->page_size, device->page_size,
                                           &device->heap, &memory->buffer);
  }
  uint64_t dma_address = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_memory_translate_dma_address(
        memory, 0, create_info->byte_length, &dma_address);
  }
  if (amdf_status_is_ok(status)) {
    *out_result = (amdf_xdna_umd_memory_result_t){
        .flags = profile->guaranteed_flags,
        .byte_length = create_info->byte_length,
        .alignment = device->page_size,
        .native_allocation_byte_length = byte_length,
        .native_allocation_granularity = device->page_size,
        .device_address = memory->buffer.device_address,
        .address_kinds = profile->address_kinds,
        .dma_address = dma_address,
    };
  }
  return status;
}

amdf_status_t amdf_xdna_umd_memory_prepare_import(
    amdf_xdna_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result) {
  amdf_assert(external_memory->type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD &&
              "selected XDNA import profile must consume DMA-BUF memory");
  if (external_memory->payload.file_descriptor > INT_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const int descriptor = (int)external_memory->payload.file_descriptor;
  amdf_linux_dma_buf_info_t dma_buf_info;
  amdf_status_t status = amdf_linux_dma_buf_query(descriptor, &dma_buf_info);
  if (!amdf_status_is_ok(status)) return status;
  if (amdf_physical_memory_id_is_valid(&external_memory->physical_backing_id) &&
      !amdf_physical_memory_id_is_equal(&external_memory->physical_backing_id,
                                        &dma_buf_info.physical_backing_id)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  if (external_memory->source_byte_offset > dma_buf_info.byte_length ||
      external_memory->byte_length >
          dma_buf_info.byte_length - external_memory->source_byte_offset ||
      dma_buf_info.byte_length > SIZE_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint64_t alignment = import_info->minimum_alignment > device->page_size
                                 ? import_info->minimum_alignment
                                 : device->page_size;
  if (alignment > PTRDIFF_MAX ||
      dma_buf_info.byte_length > (uint64_t)PTRDIFF_MAX - alignment) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }

  amdf_xdna_umd_memory_t* memory = NULL;
  status = amdf_calloc(device->host_allocator, sizeof(*memory),
                       amdf_alignof(amdf_xdna_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) return status;
  memory->device = device;
  *memory_state = memory;
  memory->source_byte_offset = external_memory->source_byte_offset;
  memory->physical_backing_id = dma_buf_info.physical_backing_id;
  status = amdf_linux_xdna_buffer_import_dma_buf(
      device->descriptor, descriptor, (size_t)dma_buf_info.byte_length,
      &memory->buffer);
  if (amdf_status_is_ok(status)) {
    status =
        amdf_linux_xdna_buffer_attach(device->descriptor, (size_t)alignment,
                                      device->page_size, NULL, &memory->buffer);
  }
  uint64_t dma_address = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_memory_translate_dma_address(
        memory, external_memory->source_byte_offset,
        external_memory->byte_length, &dma_address);
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t offset_alignment =
        external_memory->source_byte_offset == 0
            ? alignment
            : external_memory->source_byte_offset &
                  (UINT64_C(0) - external_memory->source_byte_offset);
    const uint64_t logical_alignment =
        offset_alignment < alignment ? offset_alignment : alignment;
    const amdf_xdna_umd_memory_result_t result = {
        .flags = profile->guaranteed_flags,
        .source_byte_offset = external_memory->source_byte_offset,
        .byte_length = external_memory->byte_length,
        .alignment = logical_alignment,
        .native_allocation_byte_length = dma_buf_info.byte_length,
        .native_allocation_granularity = device->page_size,
        .physical_backing_id = dma_buf_info.physical_backing_id,
        .device_address =
            memory->buffer.device_address + external_memory->source_byte_offset,
        .address_kinds = profile->address_kinds,
        .dma_address = dma_address,
    };
    *out_result = result;
  }
  return status;
}

amdf_status_t amdf_xdna_umd_memory_export(
    amdf_xdna_umd_memory_t* memory,
    const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  (void)export_info;
  int descriptor = -1;
  amdf_status_t status = amdf_linux_xdna_buffer_export_dma_buf(
      memory->device->descriptor, &memory->buffer, &descriptor);
  amdf_linux_dma_buf_info_t info;
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_dma_buf_query(descriptor, &info);
  }
  if (amdf_status_is_ok(status) &&
      (info.byte_length != memory->buffer.byte_length ||
       !amdf_physical_memory_id_is_equal(&info.physical_backing_id,
                                         &memory->physical_backing_id))) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (amdf_status_is_ok(status)) {
    *out_value = (amdf_external_memory_t){
        .payload.file_descriptor = descriptor,
        .release = amdf_linux_dma_buf_release,
    };
  } else {
    const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
    if (!amdf_status_is_ok(close_status)) status = close_status;
  }
  return status;
}

amdf_status_t amdf_xdna_umd_memory_describe_site(
    amdf_xdna_umd_memory_t* memory, const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  (void)memory;
  const amdf_queue_family_info_t* family = query->queue_family_info;
  if (family->command_type != AMDF_QUEUE_COMMAND_TYPE_XDNA ||
      family->format_version != AMDF_XDNA_QUEUE_FORMAT_VERSION_1 ||
      (family->roles & AMDF_QUEUE_ROLE_COMPUTE) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_memory_site_description_t description = {0};
  if ((query->access_info->access & AMDF_MEMORY_ACCESS_READ) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((query->access_info->access & AMDF_MEMORY_ACCESS_WRITE) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  description.release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  description.acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  *out_description = description;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_memory_prepare(
    amdf_xdna_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result) {
  const bool registers_host =
      (profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0;
  uint64_t alignment = 0;
  size_t native_byte_length = 0;
  void* native_host_pointer = NULL;
  uint64_t source_byte_offset = 0;
  if (registers_host) {
    const uintptr_t host_address =
        (uintptr_t)create_info->registered_host_pointer;
    const uintptr_t host_page_address =
        host_address & ~(uintptr_t)(device->page_size - 1);
    source_byte_offset = host_address - host_page_address;
    if (create_info->byte_length > SIZE_MAX - source_byte_offset) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    const size_t unaligned_native_byte_length =
        (size_t)(source_byte_offset + create_info->byte_length);
    if (unaligned_native_byte_length > SIZE_MAX - (device->page_size - 1)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    native_byte_length =
        (unaligned_native_byte_length + device->page_size - 1) &
        ~(device->page_size - 1);
    native_host_pointer = (void*)host_page_address;
    alignment = create_info->minimum_alignment == 0
                    ? 1
                    : create_info->minimum_alignment;
  } else {
    alignment = create_info->minimum_alignment > device->page_size
                    ? create_info->minimum_alignment
                    : device->page_size;
    native_byte_length = (create_info->byte_length + device->page_size - 1) &
                         ~(device->page_size - 1);
  }

  amdf_xdna_umd_memory_t* memory = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*memory),
                  amdf_alignof(amdf_xdna_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) return status;
  memory->device = device;
  *memory_state = memory;
  memory->source_byte_offset = source_byte_offset;
  if (registers_host) {
    status = amdf_linux_xdna_buffer_register_host_pages(
        device->descriptor, native_host_pointer, native_byte_length,
        &memory->buffer);
    if (amdf_status_is_ok(status)) {
      status = amdf_linux_xdna_buffer_attach(device->descriptor, alignment,
                                             device->page_size, NULL,
                                             &memory->buffer);
    }
  } else {
    status = amdf_linux_xdna_buffer_create(device->descriptor, AMDXDNA_BO_SHARE,
                                           native_byte_length, &memory->buffer);
    if (amdf_status_is_ok(status)) {
      status = amdf_linux_xdna_buffer_attach(device->descriptor, alignment,
                                             device->page_size, NULL,
                                             &memory->buffer);
    }
  }
  if (amdf_status_is_ok(status) && !registers_host) {
    amdf_linux_dma_buf_info_t info;
    status = amdf_linux_xdna_memory_query_dma_buf(memory, &info);
    if (amdf_status_is_ok(status)) {
      memory->physical_backing_id = info.physical_backing_id;
    }
  }
  uint64_t dma_address = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_memory_translate_dma_address(
        memory, source_byte_offset,
        registers_host ? create_info->byte_length : native_byte_length,
        &dma_address);
  }
  if (amdf_status_is_ok(status)) {
    const amdf_xdna_umd_memory_result_t result = {
        .flags = profile->guaranteed_flags | create_info->required_flags,
        .source_byte_offset = source_byte_offset,
        .byte_length =
            registers_host ? create_info->byte_length : native_byte_length,
        .alignment = alignment,
        .native_allocation_byte_length = native_byte_length,
        .native_allocation_granularity = device->page_size,
        .physical_backing_id = memory->physical_backing_id,
        .device_address = memory->buffer.device_address + source_byte_offset,
        .address_kinds = profile->address_kinds,
        .dma_address = dma_address,
    };
    *out_result = result;
  }
  return status;
}

amdf_status_t amdf_xdna_umd_memory_map(
    amdf_xdna_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_xdna_umd_host_mapping_t** out_mapping,
    amdf_xdna_umd_host_mapping_result_t* out_result) {
  amdf_xdna_umd_host_mapping_t* mapping = NULL;
  amdf_status_t status =
      amdf_calloc(memory->device->host_allocator, sizeof(*mapping),
                  amdf_alignof(amdf_xdna_umd_host_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) return status;
  mapping->host_allocator = memory->device->host_allocator;
  mapping->pointer = (uint8_t*)memory->buffer.host_pointer +
                     memory->source_byte_offset + map_info->byte_offset;
  mapping->cache_line_size = memory->device->cache_line_size;
  amdf_xdna_umd_host_mapping_result_t result = {0};
  result.flags = capabilities->supported_access;
  result.pointer = mapping->pointer;
  result.byte_length = map_info->byte_length;
  result.cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
  result.cache_line_size = mapping->cache_line_size;
  result.flush = (amdf_cache_transition_t){
      .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
      .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
      .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
      .host_fence_before = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .range_granularity = mapping->cache_line_size,
  };
  result.invalidate = (amdf_cache_transition_t){
      .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
      .host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE,
      .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
      .host_fence_before = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .range_granularity = mapping->cache_line_size,
  };
  *out_result = result;
  *out_mapping = mapping;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_host_mapping_cache_control(
    amdf_xdna_umd_host_mapping_t* mapping,
    amdf_host_cache_operation_t operation, uint64_t byte_offset,
    uint64_t byte_length) {
  // The public mapping boundary has already validated operation and range.
  (void)operation;
  amdf_linux_host_cache_transfer((uint8_t*)mapping->pointer + byte_offset,
                                 byte_length, mapping->cache_line_size);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_host_mapping_destroy(
    amdf_xdna_umd_host_mapping_t* mapping) {
  amdf_free(mapping->host_allocator, mapping);
  return AMDF_STATUS_OK;
}
