// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/rocjitsu/memory.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/device_state.h"
#include "libamdf/src/gpu/umd/rocjitsu/buffer.h"
#include "libamdf/src/gpu/umd/rocjitsu/device.h"

struct amdf_gpu_umd_memory_t {
  // Allocator owning this resource metadata.
  amdf_allocator_t host_allocator;
  // Partial or complete allocation state retained through explicit destruction.
  amdf_rocjitsu_buffer_t* buffer;
};

static amdf_memory_host_description_t amdf_rocjitsu_describe_host(
    const void* data, amdf_external_memory_type_t external_type,
    amdf_memory_flags_t flags, amdf_host_memory_properties_t properties) {
  (void)data;
  (void)external_type;
  (void)flags;
  // CPU execution and the host view share ordinary coherent memfd pages.
  return (amdf_memory_host_description_t){
      .properties = properties,
      .flush = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE},
      .invalidate = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE},
  };
}

static amdf_memory_site_description_t amdf_rocjitsu_describe_program(
    amdf_memory_access_t access, const amdf_device_t* device,
    amdf_memory_atomic_capabilities_t operations) {
  return amdf_gpu_umd_memory_describe_program_site(access, &device->endpoint_id,
                                                   operations);
}

static bool amdf_rocjitsu_query_access(
    const amdf_memory_native_profile_t* backing,
    const amdf_memory_native_profile_t* candidate,
    amdf_memory_native_profile_t* out_profile) {
  if (backing->construction.data != candidate->construction.data ||
      backing->memory_class != candidate->memory_class) {
    return false;
  }
  *out_profile = *candidate;
  return true;
}

amdf_status_t amdf_rocjitsu_device_query_memory_profile(
    amdf_gpu_umd_device_t* device, uint32_t ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (ordinal >= 2) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint64_t maximum_length = UINT64_C(1) << 46;
  const bool local = ordinal == 1;
  const amdf_memory_flags_t flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                                    AMDF_MEMORY_FLAG_HOST_COHERENT |
                                    AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                                    (local ? AMDF_MEMORY_FLAG_DEVICE_LOCAL : 0);
  *out_profile = (amdf_memory_native_profile_t){
      .memory_class =
          local ? AMDF_MEMORY_CLASS_LOCAL : AMDF_MEMORY_CLASS_SYSTEM,
      .roles =
          AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      .guaranteed_flags = flags,
      .supported_flags = flags | AMDF_MEMORY_FLAG_QUEUE_STORAGE,
      .guaranteed_device_access = AMDF_MEMORY_ACCESS_READ,
      .supported_device_access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE,
      .device_address = {.address_bit_count = 48,
                         .minimum_address = 4096,
                         .maximum_address = (UINT64_C(1) << 48) - 1,
                         .minimum_alignment = 4096},
      .allocation = {.maximum_byte_length = maximum_length,
                     .byte_length_granularity = 1,
                     .minimum_alignment = 4096,
                     .maximum_alignment = maximum_length,
                     .native_byte_length_granularity = 4096},
      .host_mapping = {.properties = AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                                     AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK,
                       .features = AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS |
                                   AMDF_HOST_MAPPING_FEATURE_BUFFERED_FILE_IO,
                       .maximum_byte_length = maximum_length,
                       .byte_offset_granularity = 1,
                       .byte_length_granularity = 1,
                       .supported_access = AMDF_MEMORY_MAP_FLAG_READ |
                                           AMDF_MEMORY_MAP_FLAG_WRITE},
      .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
      .atomic_operations_32 =
          AMDF_ATOMIC_OPERATION_STORE | AMDF_ATOMIC_OPERATION_WAIT,
      .atomic_operations_64 = AMDF_ATOMIC_OPERATION_STORE,
      .construction = {.query_access = amdf_rocjitsu_query_access,
                       .data = device->gpu->source},
      .visibility = {.describe_site =
                         amdf_gpu_umd_memory_describe_system_store_site,
                     .describe_program = amdf_rocjitsu_describe_program,
                     .describe_host = amdf_rocjitsu_describe_host},
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_memory_prepare(
    amdf_gpu_umd_device_t* device, uint32_t peer_count,
    amdf_gpu_umd_device_t* const* peers,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_umd_memory_t** state, amdf_gpu_umd_memory_result_t* out_result) {
  amdf_gpu_umd_memory_t* memory = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*memory),
                  amdf_alignof(amdf_gpu_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memory->host_allocator = device->host_allocator;
  *state = memory;
  uint32_t* peer_ids = NULL;
  if (peer_count != 0) {
    status = amdf_malloc(device->host_allocator,
                         ((size_t)peer_count + 1) * sizeof(*peer_ids),
                         amdf_alignof(uint32_t), (void**)&peer_ids);
    if (amdf_status_is_ok(status)) {
      peer_ids[0] = device->gpu->native.gpu_id;
      for (uint32_t i = 0; i < peer_count; ++i) {
        peer_ids[i + 1] = peers[i]->gpu->native.gpu_id;
      }
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_rocjitsu_buffer_prepare(
        device->gpu->source, peer_count + 1,
        peer_ids != NULL ? peer_ids : &device->gpu->native.gpu_id,
        profile->memory_class, create_info->byte_length,
        create_info->minimum_alignment, create_info->device_access,
        &memory->buffer);
  }
  amdf_free(device->host_allocator, peer_ids);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_rocjitsu_buffer_t* buffer = memory->buffer;
  *out_result = (amdf_gpu_umd_memory_result_t){
      .flags = profile->guaranteed_flags | create_info->required_flags,
      .byte_length = create_info->byte_length,
      .alignment = buffer->alignment,
      .native_allocation_byte_length = buffer->byte_length,
      .native_allocation_granularity = 4096,
      .physical_backing_id = {{device->gpu->source->generation,
                               buffer->handle}},
      .device_address = (uintptr_t)buffer->pointer,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_memory_destroy(amdf_gpu_umd_memory_t* memory) {
  if (memory == NULL) {
    return AMDF_STATUS_OK;
  }
  const amdf_status_t status = amdf_rocjitsu_buffer_destroy(memory->buffer);
  if (amdf_status_is_ok(status)) {
    amdf_free(memory->host_allocator, memory);
  }
  return status;
}

uint64_t amdf_rocjitsu_memory_remaining_backing_byte_length(
    const amdf_gpu_umd_memory_t* memory) {
  return memory != NULL && memory->buffer != NULL &&
                 (memory->buffer->allocated || memory->buffer->host_mapped)
             ? memory->buffer->byte_length
             : 0;
}

void amdf_rocjitsu_memory_abandon(amdf_gpu_umd_memory_t* memory) {
  if (memory == NULL) {
    return;
  }
  amdf_rocjitsu_buffer_abandon(memory->buffer);
  amdf_free(memory->host_allocator, memory);
}

amdf_status_t amdf_rocjitsu_memory_map(
    amdf_gpu_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_gpu_umd_host_mapping_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result) {
  *out_result = (amdf_gpu_umd_host_mapping_result_t){
      .flags = capabilities->supported_access,
      .pointer = (uint8_t*)memory->buffer->pointer + map_info->byte_offset,
      .byte_length = map_info->byte_length,
      .visibility =
          amdf_rocjitsu_describe_host(NULL, 0, 0, capabilities->properties),
  };
  *out_mapping = NULL;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_host_mapping_cache_control(
    amdf_gpu_umd_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t byte_offset, uint64_t byte_length) {
  (void)mapping;
  (void)operation;
  (void)byte_offset;
  (void)byte_length;
  // The coherent host policy advertises no explicit cache operation.
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

void amdf_rocjitsu_host_mapping_destroy(amdf_gpu_umd_host_mapping_t* mapping) {
  (void)mapping;
}
