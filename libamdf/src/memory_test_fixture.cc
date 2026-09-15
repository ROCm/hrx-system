// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/memory_test_fixture.h"

#include <cstring>

#include "libamdf/src/allocator.h"
#include "libamdf/src/child_tracker.h"

namespace amdf::testing {

void AMDF_CALL RecordRelease(void* user_data, amdf_external_memory_type_t type,
                             amdf_external_memory_payload_t payload) {
  auto* state = static_cast<ReleaseState*>(user_data);
  ++state->count;
  state->type = type;
  state->payload = payload;
}

static FakeMemory* NativeMemory(amdf_memory_t* memory, uint32_t ordinal) {
  return static_cast<FakeMemory*>(
      memory->accesses[memory->accesses[ordinal].native_owner_ordinal].native);
}

static amdf_status_t FakeMemoryExport(
    amdf_memory_t* base_memory, uint32_t access_ordinal,
    const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  (void)export_info;
  auto* memory = NativeMemory(base_memory, access_ordinal);
  ++memory->export_call_count;
  if (!amdf_status_is_ok(memory->export_status)) {
    return memory->export_status;
  }
  out_value->payload.file_descriptor = 73;
  out_value->release = RecordRelease;
  out_value->release_user_data = memory->export_release_state;
  return AMDF_STATUS_OK;
}

static amdf_status_t FakeMemoryDescribeSite(
    amdf_memory_t* base_memory, uint32_t access_ordinal,
    uint32_t queue_family_ordinal,
    amdf_memory_site_description_t* out_description) {
  auto* memory = NativeMemory(base_memory, access_ordinal);
  ++memory->site_description_count;
  memory->last_queue_family_ordinal = queue_family_ordinal;
  if (!amdf_status_is_ok(memory->site_status)) {
    return memory->site_status;
  }
  *out_description = memory->site_description;
  return AMDF_STATUS_OK;
}

static amdf_status_t FakeMemoryMap(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping) {
  (void)capabilities;
  (void)map_info;
  (void)out_mapping;
  auto* fake_memory = NativeMemory(memory, access_ordinal);
  ++fake_memory->map_call_count;
  if (!amdf_status_is_ok(fake_memory->map_status)) {
    return fake_memory->map_status;
  }
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t FakeMemoryDestroyNative(amdf_memory_t* base_memory,
                                             uint32_t access_ordinal) {
  auto* memory =
      static_cast<FakeMemory*>(base_memory->accesses[access_ordinal].native);
  ++memory->device->destroy_call_count;
  if (memory->device->release_order != nullptr) {
    memory->device->release_order->push_back(
        memory->device->backing_id.words[0]);
  }
  if (!amdf_status_is_ok(memory->destroy_status)) {
    return memory->destroy_status;
  }
  amdf_free(base_memory->host_allocator, memory);
  base_memory->accesses[access_ordinal].native = nullptr;
  return AMDF_STATUS_OK;
}

static void FakeMemoryAbandonNative(amdf_memory_t* memory,
                                    uint32_t access_ordinal) {
  ++static_cast<FakeMemory*>(memory->accesses[access_ordinal].native)
        ->device->abandon_call_count;
  amdf_free(memory->host_allocator, memory->accesses[access_ordinal].native);
  memory->accesses[access_ordinal].native = nullptr;
}

static const amdf_memory_vtable_t kFakeMemoryVtable = {
    .export_external = FakeMemoryExport,
    .describe_site = FakeMemoryDescribeSite,
    .map = FakeMemoryMap,
    .destroy_native = FakeMemoryDestroyNative,
    .abandon_native = FakeMemoryAbandonNative,
};

static amdf_status_t PrepareFakeMemory(
    amdf_memory_t* base_memory, uint32_t access_ordinal,
    amdf_memory_access_t device_access, uint32_t memory_profile_ordinal,
    uint64_t source_byte_offset, uint64_t byte_length,
    amdf_physical_memory_id_t backing_id, amdf_memory_info_t* out_info) {
  auto* device = reinterpret_cast<FakeDevice*>(
      base_memory->accesses[access_ordinal].device);
  const amdf_allocator_t host_allocator =
      amdf_device_host_allocator(&device->base);
  FakeMemory* memory = nullptr;
  const amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*memory), amdf_alignof(FakeMemory),
                  reinterpret_cast<void**>(&memory));
  if (!amdf_status_is_ok(status)) return status;
  base_memory->accesses[access_ordinal].native = memory;
  memory->device = device;
  memory->export_status = AMDF_STATUS_OK;
  memory->map_status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  memory->site_status = AMDF_STATUS_OK;
  memory->destroy_status = device->destroy_status;
  memory->export_release_state = &device->export_release;
  memory->site_description.capabilities =
      AMDF_MEMORY_SITE_CAPABILITY_READ | AMDF_MEMORY_SITE_CAPABILITY_WRITE |
      AMDF_MEMORY_SITE_CAPABILITY_MAPPING_SOURCE |
      AMDF_MEMORY_SITE_CAPABILITY_MAPPING_TARGET |
      AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN |
      AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN;
  memory->site_description.release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  memory->site_description.acquire.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
  memory->site_description.acquire.executor =
      AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE;
  memory->site_description.acquire.operation =
      AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM;
  memory->site_description.acquire.range_granularity = 64;
  memory->site_description.mapping_domain.words[0] = 1;
  memory->site_description.atomic_domain.words[0] = 2;
  memory->site_description.atomic_reach.scope_32 = AMDF_ATOMIC_SCOPE_SYSTEM;
  memory->site_description.atomic_reach.scope_64 = AMDF_ATOMIC_SCOPE_DEVICE;
  memory->site_description.release_fixed_cost_nanoseconds = 17;
  memory->site_description.acquire_fixed_cost_nanoseconds = 25;
  out_info->type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  out_info->structure_size = sizeof(*out_info);
  out_info->memory_profile_ordinal = memory_profile_ordinal;
  out_info->memory_class = AMDF_MEMORY_CLASS_SYSTEM;
  base_memory->accesses[access_ordinal].info.access = device_access;
  base_memory->accesses[access_ordinal].info.atomic_operations_32 =
      device->profile.atomic_operations_32;
  base_memory->accesses[access_ordinal].info.atomic_operations_64 =
      device->profile.atomic_operations_64;
  base_memory->accesses[access_ordinal].info.address_domain_ordinal = 0;
  out_info->flags = device->profile.supported_flags & AMDF_MEMORY_BACKING_FLAGS;
  base_memory->accesses[access_ordinal].info.flags =
      device->profile.guaranteed_flags & AMDF_MEMORY_ACCESS_FLAGS;
  out_info->source_byte_offset = source_byte_offset;
  out_info->byte_length = byte_length;
  out_info->alignment = 4096;
  out_info->native_allocation_byte_length = source_byte_offset + byte_length;
  out_info->native_allocation_granularity = 4096;
  out_info->physical_backing_id = backing_id;
  std::memcpy(base_memory->accesses[access_ordinal].addresses,
              device->addresses.data(),
              sizeof(base_memory->accesses[access_ordinal].addresses));
  base_memory->accesses[access_ordinal].info.address_kinds =
      device->profile.address_kinds;
  base_memory->accesses[access_ordinal].info.reset_epoch = 1;
  return AMDF_STATUS_OK;
}

static amdf_status_t FakeDeviceQueryMemoryProfile(
    amdf_device_t* base_device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  auto* device = reinterpret_cast<FakeDevice*>(base_device);
  if (!amdf_status_is_ok(device->profile_status)) {
    return device->profile_status;
  }
  if (memory_profile_ordinal > 1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  *out_profile = device->profile;
  out_profile->ordinal = memory_profile_ordinal;
  if (memory_profile_ordinal == 0) {
    out_profile->roles &= ~AMDF_MEMORY_PROFILE_ROLE_IMPORT;
    out_profile->import = {};
  } else {
    out_profile->roles &= ~AMDF_MEMORY_PROFILE_ROLE_CREATE;
    out_profile->allocation = {};
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t FakeDeviceMemoryPrepare(
    amdf_memory_t* memory, const amdf_memory_native_group_t* group,
    const amdf_memory_native_create_info_t* create_info,
    amdf_memory_info_t* out_info) {
  const uint32_t access_ordinal = group->access_ordinals[0];
  const amdf_memory_native_profile_t* profile =
      &group->profiles[access_ordinal];
  auto* device =
      reinterpret_cast<FakeDevice*>(memory->accesses[access_ordinal].device);
  memory->accesses[access_ordinal].vtable = &kFakeMemoryVtable;
  ++device->create_call_count;
  device->prepared_access_count = group->access_count;
  device->registered_host_pointer = create_info->registered_host_pointer;
  const bool registration =
      (profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0;
  const uint64_t granularity =
      registration ? profile->registration.native_byte_length_granularity
                   : profile->allocation.native_byte_length_granularity;
  const uint64_t native_length =
      (create_info->byte_length + granularity - 1) & ~(granularity - 1);
  const amdf_status_t status = PrepareFakeMemory(
      memory, access_ordinal, create_info->device_access, profile->ordinal, 0,
      native_length, device->backing_id, out_info);
  if (amdf_status_is_ok(status)) {
    for (uint32_t i = 1; i < group->access_count; ++i) {
      const uint32_t ordinal = group->access_ordinals[i];
      auto& member = memory->accesses[ordinal];
      member.vtable = &kFakeMemoryVtable;
      member.info = memory->accesses[access_ordinal].info;
      member.info.ordinal = ordinal;
      member.info.address_domain_ordinal =
          group->profiles[ordinal].device_address.address_domain_ordinal;
      std::memcpy(member.addresses, memory->accesses[access_ordinal].addresses,
                  sizeof(member.addresses));
    }
  }
  return amdf_status_is_ok(status) ? device->create_status : status;
}

static amdf_status_t FakeDeviceMemoryPrepareImport(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_memory_info_t* out_info) {
  auto* device =
      reinterpret_cast<FakeDevice*>(memory->accesses[access_ordinal].device);
  memory->accesses[access_ordinal].vtable = &kFakeMemoryVtable;
  ++device->import_call_count;
  if (device->import_failure_stage == ImportFailureStage::kBeforeAttachment) {
    return device->import_status;
  }

  const amdf_status_t status = PrepareFakeMemory(
      memory, access_ordinal, import_info->device_access, profile->ordinal,
      external_memory->source_byte_offset, external_memory->byte_length,
      external_memory->physical_backing_id, out_info);
  if (!amdf_status_is_ok(status)) return status;
  if (device->import_failure_stage == ImportFailureStage::kAfterAttachment) {
    return device->import_status;
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t FakeDeviceDestroyNative(amdf_device_t* device) {
  (void)device;
  return AMDF_STATUS_OK;
}

static const amdf_device_vtable_t kFakeDeviceVtable = {
    .query_memory_profile = FakeDeviceQueryMemoryProfile,
    .memory_prepare = FakeDeviceMemoryPrepare,
    .memory_prepare_import = FakeDeviceMemoryPrepareImport,
    .destroy_native = FakeDeviceDestroyNative,
};

void InitializeFakeDevice(uint64_t identity, amdf_instance_t* instance,
                          FakeDevice* out_device) {
  std::memset(out_device, 0, sizeof(*out_device));
  out_device->base.host_allocator = instance->host_allocator;
  out_device->base.vtable = &kFakeDeviceVtable;
  out_device->base.provider_instance = instance;
  out_device->base.engine_kind = AMDF_ENGINE_KIND_GPU;
  amdf_child_tracker_initialize(&out_device->base.children);
  out_device->request.device = &out_device->base;
  out_device->request.requirements.access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  out_device->request.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  out_device->profile.ordinal = 0;
  out_device->profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
  out_device->profile.roles =
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_IMPORT |
      AMDF_MEMORY_PROFILE_ROLE_EXPORT | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
  out_device->profile.guaranteed_flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  out_device->profile.supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                                        AMDF_MEMORY_FLAG_SHAREABLE |
                                        AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  out_device->profile.guaranteed_device_access = AMDF_MEMORY_ACCESS_READ;
  out_device->profile.supported_device_access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  out_device->profile.device_address.address_domain_ordinal = 0;
  out_device->profile.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  out_device->addresses[AMDF_MEMORY_ADDRESS_GPU] = UINT64_C(0x100000);
  out_device->profile.device_address.address_bit_count = 48;
  out_device->profile.device_address.maximum_address = (UINT64_C(1) << 48) - 1;
  out_device->profile.device_address.minimum_alignment = 4096;
  out_device->profile.allocation.maximum_byte_length = UINT64_C(1) << 32;
  out_device->profile.allocation.byte_length_granularity = 1;
  out_device->profile.allocation.minimum_alignment = 4096;
  out_device->profile.allocation.maximum_alignment = UINT64_C(1) << 30;
  out_device->profile.allocation.native_byte_length_granularity = 4096;
  out_device->profile.import = out_device->profile.allocation;
  out_device->profile.import.minimum_alignment = 1;
  out_device->profile.host_mapping.maximum_byte_length = UINT64_C(1) << 32;
  out_device->profile.host_mapping.byte_offset_granularity = 1;
  out_device->profile.host_mapping.byte_length_granularity = 1;
  out_device->profile.host_mapping.supported_access =
      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  out_device->profile.external_memory_support_count = 1;
  out_device->profile.external_memory_support[0].type =
      AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
  out_device->profile.external_memory_support[0].flags =
      AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
      AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
      AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET;
  out_device->profile.external_memory_support[0].source_offset_alignment = 4096;
  out_device->profile.external_memory_support[0].byte_length_alignment = 4096;
  out_device->profile_status = AMDF_STATUS_OK;
  out_device->create_status = AMDF_STATUS_OK;
  out_device->import_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  out_device->backing_id.words[0] = identity;
  out_device->backing_id.words[1] = identity ^ UINT64_C(0xA5A5A5A5);
}

amdf_memory_create_info_t MakeMemoryCreateInfo(FakeDevice& device) {
  amdf_memory_create_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  info.structure_size = sizeof(info);
  info.memory_profile_ordinal = 0;
  info.access_count = 1;
  info.accesses = &device.request;
  info.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  info.byte_length = 4096;
  info.minimum_alignment = 4096;
  return info;
}

amdf_memory_import_info_t MakeMemoryImportInfo(FakeDevice& device) {
  amdf_memory_import_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
  info.structure_size = sizeof(info);
  info.memory_profile_ordinal = 2;
  info.access_count = 1;
  info.accesses = &device.request;
  info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  info.minimum_alignment = 4096;
  return info;
}

amdf_memory_export_info_t MakeMemoryExportInfo() {
  amdf_memory_export_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO;
  info.structure_size = sizeof(info);
  info.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
  info.byte_length = 4096;
  return info;
}

amdf_memory_site_t MakeMemorySite(amdf_memory_t* memory,
                                  uint32_t queue_family_ordinal) {
  amdf_memory_site_t site = {};
  site.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
  site.structure_size = sizeof(site);
  site.value.device.memory = memory;
  site.value.device.queue_family_ordinal = queue_family_ordinal;
  return site;
}

}  // namespace amdf::testing
