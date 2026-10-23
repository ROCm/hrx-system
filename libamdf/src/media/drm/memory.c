// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/memory.h"

#include <emmintrin.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/host_mapping.h"
#include "libamdf/src/platform/linux/dma_buf.h"
#include "libamdf/src/platform/linux/file.h"

static amdf_status_t amdf_media_drm_describe_site(
    const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  amdf_memory_site_description_t result = {0};
  if ((query->access & AMDF_MEMORY_ACCESS_READ) != 0) {
    result.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((query->access & AMDF_MEMORY_ACCESS_WRITE) != 0) {
    result.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  // Native job submission and engine completion publish memory across the
  // uncached media VM. These operations do not supply execution dependencies.
  result.release.kind = result.acquire.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
  result.release.executor = result.acquire.executor =
      AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API;
  result.release.operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
  result.acquire.operation = AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM;
  *out_description = result;
  return AMDF_STATUS_OK;
}

static amdf_memory_host_description_t amdf_media_drm_describe_host(
    const void* data, amdf_external_memory_type_t external_type,
    amdf_memory_flags_t flags) {
  (void)data;
  (void)external_type;
  (void)flags;
  amdf_memory_host_description_t result = {
      .cacheability = AMDF_HOST_CACHEABILITY_WRITE_COMBINED,
      .flush = {.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
                .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
                .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE},
      .invalidate = {.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                     .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
                     .host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                     .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE},
  };
  return result;
}

static amdf_status_t amdf_media_drm_memory_describe_site(
    amdf_memory_t* memory, uint32_t ordinal, uint32_t family,
    amdf_memory_site_description_t* out_description) {
  const amdf_memory_access_state_t* access = &memory->accesses[ordinal];
  if (family >= access->device->queue_family_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_memory_site_query_t query = {
      .access = access->info.access,
      .queue_family_info = &access->device->queue_families[family],
  };
  return amdf_media_drm_describe_site(&query, out_description);
}

static amdf_status_t amdf_media_drm_memory_export(
    amdf_memory_t* memory, uint32_t ordinal,
    const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  const amdf_media_drm_memory_t* native =
      amdf_media_drm_memory(memory, ordinal);
  int descriptor = -1;
  const amdf_status_t status =
      amdf_media_drm_buffer_export(native->buffer, &descriptor);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *out_value = (amdf_external_memory_t){
      .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
      .payload = {.file_descriptor = descriptor},
      .source_byte_offset = native->byte_offset + export_info->byte_offset,
      .byte_length = export_info->byte_length,
      .release = amdf_linux_dma_buf_release,
  };
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_media_drm_mapping_cache_control(
    amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t byte_offset, uint64_t byte_length) {
  (void)mapping;
  (void)operation;
  (void)byte_offset;
  (void)byte_length;
  _mm_mfence();
  return AMDF_STATUS_OK;
}

static void amdf_media_drm_mapping_destroy(amdf_host_mapping_t* mapping) {
  (void)mapping;
}

static const amdf_host_mapping_vtable_t amdf_media_drm_mapping_operations = {
    amdf_media_drm_mapping_cache_control, amdf_media_drm_mapping_destroy};

static amdf_status_t amdf_media_drm_memory_map(
    amdf_memory_t* memory, uint32_t ordinal,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping) {
  amdf_host_mapping_t* mapping = NULL;
  const amdf_status_t status =
      amdf_calloc(memory->host_allocator, sizeof(*mapping),
                  amdf_alignof(amdf_host_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_media_drm_memory_t* native =
      amdf_media_drm_memory(memory, ordinal);
  const amdf_memory_host_description_t host =
      amdf_media_drm_describe_host(NULL, 0, 0);
  amdf_host_mapping_initialize(mapping, &amdf_media_drm_mapping_operations,
                               memory);
  mapping->info = (amdf_host_mapping_info_t){
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping->info),
      .flags = map_info->flags,
      .cacheability = host.cacheability,
      .pointer = (uint8_t*)native->buffer->host_pointer + native->byte_offset +
                 map_info->byte_offset,
      .memory_byte_offset = map_info->byte_offset,
      .byte_length = map_info->byte_length,
      .byte_offset_granularity = capabilities->byte_offset_granularity,
      .byte_length_granularity = capabilities->byte_length_granularity,
      .flush = host.flush,
      .invalidate = host.invalidate,
  };
  *out_mapping = mapping;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_media_drm_memory_destroy(amdf_memory_t* memory,
                                                   uint32_t ordinal) {
  amdf_media_drm_memory_t* native = memory->accesses[ordinal].native;
  return amdf_media_drm_buffer_release(&native->buffer);
}

static uint64_t amdf_media_drm_memory_remaining(const amdf_memory_t* memory,
                                                uint32_t ordinal) {
  const amdf_media_drm_memory_t* native = memory->accesses[ordinal].native;
  return native->buffer != NULL ? native->buffer->byte_length : 0;
}

static void amdf_media_drm_memory_abandon(amdf_memory_t* memory,
                                          uint32_t ordinal) {
  amdf_media_drm_memory_t* native = memory->accesses[ordinal].native;
  amdf_media_drm_buffer_abandon(&native->buffer);
  amdf_free(memory->host_allocator, native);
  memory->accesses[ordinal].native = NULL;
}

static const amdf_memory_vtable_t amdf_media_drm_memory_operations = {
    amdf_media_drm_memory_export,  amdf_media_drm_memory_describe_site,
    amdf_media_drm_memory_map,     NULL,
    amdf_media_drm_memory_destroy, amdf_media_drm_memory_remaining,
    amdf_media_drm_memory_abandon};

amdf_status_t amdf_media_drm_memory_profile(
    amdf_device_t* base, uint32_t ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (ordinal > 1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_media_drm_device_t* device = (const amdf_media_drm_device_t*)base;
  const uint64_t maximum = device->address_end - device->address_begin;
  const uint64_t page_size = device->page_size;
  amdf_memory_native_profile_t profile = {
      .ordinal = ordinal,
      .memory_class = AMDF_MEMORY_CLASS_SYSTEM,
      .roles = AMDF_MEMORY_PROFILE_ROLE_EXPORT,
      .guaranteed_flags =
          AMDF_MEMORY_FLAG_SHAREABLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .guaranteed_device_access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .device_address = {.address_domain_ordinal =
                             AMDF_ADDRESS_DOMAIN_ORDINAL_NONE,
                         .address_bit_count = 48,
                         .maximum_address = device->address_end - 1,
                         .minimum_alignment = ordinal == 0 ? page_size : 1},
      .external_memory_support_count = 1,
      .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
  };
  profile.supported_device_access = profile.guaranteed_device_access;
  profile.external_memory_support[0] = (amdf_external_memory_support_t){
      .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
      .flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
               AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
               AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API,
      .source_offset_alignment = 1,
      .byte_length_alignment = 1,
      .maximum_byte_length = maximum,
  };
  if (ordinal == 0) {
    profile.roles |=
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.guaranteed_flags |= AMDF_MEMORY_FLAG_HOST_VISIBLE;
    profile.allocation = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum,
        .byte_length_granularity = 1,
        .native_byte_length_granularity = page_size,
        .minimum_alignment = page_size,
        .maximum_alignment = page_size,
    };
    profile.host_mapping = (amdf_host_mapping_capabilities_t){
        .maximum_byte_length = maximum,
        .byte_offset_granularity = 1,
        .byte_length_granularity = 1,
        .supported_access =
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    profile.visibility.describe_host = amdf_media_drm_describe_host;
  } else {
    profile.roles |= AMDF_MEMORY_PROFILE_ROLE_IMPORT |
                     AMDF_MEMORY_NATIVE_PROFILE_ROLE_HOST_IMPORT;
    profile.import = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum,
        .byte_length_granularity = 1,
        .minimum_alignment = 1,
        .maximum_alignment = page_size,
    };
    profile.external_memory_support[0].flags |=
        AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT;
  }
  profile.supported_flags = profile.guaranteed_flags;
  profile.visibility.describe_site = amdf_media_drm_describe_site;
  *out_profile = profile;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_media_drm_memory_attach(
    amdf_memory_t* memory, uint32_t ordinal, int descriptor,
    uint64_t byte_offset, uint64_t byte_length, uint64_t native_byte_length,
    amdf_memory_access_t device_access, amdf_memory_info_t* out_info) {
  amdf_memory_access_state_t* access = &memory->accesses[ordinal];
  amdf_media_drm_device_t* device = (amdf_media_drm_device_t*)access->device;
  amdf_media_drm_memory_t* native = NULL;
  amdf_status_t status =
      amdf_calloc(memory->host_allocator, sizeof(*native),
                  amdf_alignof(amdf_media_drm_memory_t), (void**)&native);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  access->vtable = &amdf_media_drm_memory_operations;
  access->native = native;
  native->byte_offset = byte_offset;
  status = amdf_media_drm_buffer_prepare(device, descriptor, native_byte_length,
                                         descriptor < 0, &native->buffer);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_device_info_t device_info = {0};
  access->device->vtable->query_info(access->device, &device_info);
  access->info.device_id = device_info.id;
  access->info.reset_epoch = device_info.reset_epoch;
  access->info.access = device_access;
  access->info.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  access->info.address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE;
  access->info.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  access->addresses[AMDF_MEMORY_ADDRESS_GPU] =
      native->buffer->address + byte_offset;
  out_info->type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  out_info->structure_size = sizeof(*out_info);
  out_info->memory_class = AMDF_MEMORY_CLASS_SYSTEM;
  out_info->flags = AMDF_MEMORY_FLAG_SHAREABLE |
                    (descriptor < 0 ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0);
  out_info->byte_length = byte_length;
  out_info->native_allocation_byte_length = native_byte_length;
  out_info->native_allocation_granularity = device->page_size;
  out_info->alignment = byte_offset != 0 && byte_offset % device->page_size != 0
                            ? byte_offset & (~byte_offset + 1)
                            : device->page_size;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_media_drm_memory_prepare(
    amdf_memory_t* memory, const amdf_memory_native_group_t* group,
    const amdf_memory_native_create_info_t* create_info,
    amdf_memory_info_t* out_info) {
  const uint32_t ordinal = group->access_ordinals[0];
  const amdf_media_drm_device_t* device =
      (const amdf_media_drm_device_t*)memory->accesses[ordinal].device;
  const uint64_t native_length =
      (create_info->byte_length + device->page_size - 1) &
      ~(device->page_size - 1);
  amdf_status_t status = amdf_media_drm_memory_attach(
      memory, ordinal, -1, 0, create_info->byte_length, native_length,
      create_info->device_access, out_info);
  int descriptor = -1;
  if (amdf_status_is_ok(status)) {
    const amdf_media_drm_memory_t* native =
        amdf_media_drm_memory(memory, ordinal);
    status = amdf_media_drm_buffer_export(native->buffer, &descriptor);
  }
  if (amdf_status_is_ok(status)) {
    amdf_linux_dma_buf_info_t information;
    status = amdf_linux_dma_buf_query(descriptor, &information);
    if (amdf_status_is_ok(status)) {
      out_info->physical_backing_id = information.physical_backing_id;
    }
  }
  const amdf_status_t cleanup = amdf_linux_file_close(&descriptor);
  return amdf_status_is_ok(status) ? cleanup : status;
}

amdf_status_t amdf_media_drm_memory_prepare_import(
    amdf_memory_t* memory, uint32_t ordinal,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_memory_info_t* out_info) {
  (void)profile;
  amdf_linux_dma_buf_info_t info;
  amdf_status_t status = amdf_linux_dma_buf_query(
      (int)external_memory->payload.file_descriptor, &info);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_media_drm_device_t* device =
      (const amdf_media_drm_device_t*)memory->accesses[ordinal].device;
  if (info.byte_length % device->page_size != 0 ||
      external_memory->source_byte_offset > info.byte_length ||
      external_memory->byte_length >
          info.byte_length - external_memory->source_byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  status = amdf_media_drm_memory_attach(
      memory, ordinal, (int)external_memory->payload.file_descriptor,
      external_memory->source_byte_offset, external_memory->byte_length,
      info.byte_length, import_info->device_access, out_info);
  if (amdf_status_is_ok(status)) {
    out_info->physical_backing_id = info.physical_backing_id;
    out_info->source_byte_offset = external_memory->source_byte_offset;
  }
  return status;
}

amdf_status_t amdf_media_drm_memory_range(
    amdf_media_drm_device_t* device, amdf_memory_t* memory, uint32_t ordinal,
    uint64_t byte_offset, uint64_t byte_length,
    amdf_media_drm_buffer_t** out_buffer, uint64_t* out_address) {
  if (memory == NULL || ordinal >= memory->info.access_count ||
      memory->accesses[ordinal].device != &device->base.base) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (byte_offset > memory->info.byte_length || byte_length == 0 ||
      byte_length > memory->info.byte_length - byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_media_drm_memory_t* native =
      amdf_media_drm_memory(memory, ordinal);
  *out_buffer = native->buffer;
  *out_address = native->buffer->address + native->byte_offset + byte_offset;
  return AMDF_STATUS_OK;
}
