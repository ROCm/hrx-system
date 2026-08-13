// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "iree/base/api.h"
#include "iree/hal/device_spec.h"

static void iree_hal_device_spec_fuzz_assert(bool condition) {
  if (!condition) abort();
}

static void iree_hal_device_spec_fuzz_parse(iree_const_byte_span_t bytes) {
  iree_hal_device_spec_t* spec = NULL;
  iree_status_t status =
      iree_hal_device_spec_parse(bytes, iree_allocator_system(), &spec);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return;
  }

  // Every accepted image must already be canonical. Reserializing may not
  // normalize, reorder, discard, or otherwise change any byte.
  iree_byte_span_t canonical_bytes = iree_byte_span_empty();
  status = iree_hal_device_spec_serialize(spec, iree_allocator_system(),
                                          &canonical_bytes);
  iree_hal_device_spec_fuzz_assert(iree_status_is_ok(status));
  iree_hal_device_spec_fuzz_assert(canonical_bytes.data_length ==
                                   bytes.data_length);
  iree_hal_device_spec_fuzz_assert(
      memcmp(canonical_bytes.data, bytes.data, bytes.data_length) == 0);

  // Canonical output must remain parseable and preserve its fingerprint.
  iree_hal_device_spec_t* reparsed_spec = NULL;
  status =
      iree_hal_device_spec_parse(iree_const_cast_byte_span(canonical_bytes),
                                 iree_allocator_system(), &reparsed_spec);
  iree_hal_device_spec_fuzz_assert(iree_status_is_ok(status));
  iree_hal_device_spec_fuzz_assert(
      iree_hal_device_spec_fingerprint(spec) ==
      iree_hal_device_spec_fingerprint(reparsed_spec));

  // Do not rely on the non-cryptographic fingerprint to establish equivalence.
  // Serializing the reparsed object must produce the same canonical bytes.
  iree_byte_span_t reparsed_bytes = iree_byte_span_empty();
  status = iree_hal_device_spec_serialize(
      reparsed_spec, iree_allocator_system(), &reparsed_bytes);
  iree_hal_device_spec_fuzz_assert(iree_status_is_ok(status));
  iree_hal_device_spec_fuzz_assert(reparsed_bytes.data_length ==
                                   canonical_bytes.data_length);
  iree_hal_device_spec_fuzz_assert(memcmp(reparsed_bytes.data,
                                          canonical_bytes.data,
                                          canonical_bytes.data_length) == 0);

  iree_allocator_free(iree_allocator_system(), reparsed_bytes.data);
  iree_hal_device_spec_release(reparsed_spec);
  iree_allocator_free(iree_allocator_system(), canonical_bytes.data);
  iree_hal_device_spec_release(spec);
}

static const std::vector<uint8_t>& iree_hal_device_spec_fuzz_seed(void) {
  static const std::vector<uint8_t> seed = [] {
    iree_hal_physical_device_spec_t physical_device = {};
    physical_device.identity.display_name =
        iree_make_cstring_view("fuzz physical device");
    physical_device.identity.backend_path =
        iree_make_cstring_view("pci:0000:01:00.0");
    physical_device.identity.vendor_id = 0x1002;
    physical_device.identity.device_id = 0x744c;
    physical_device.identity.revision_id = 1;
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(physical_device.identity.uuid.bytes); ++i) {
      physical_device.identity.uuid.bytes[i] = (uint8_t)i;
    }
    physical_device.identity.pci.domain = 0;
    physical_device.identity.pci.bus = 1;
    physical_device.identity.pci.device = 0;
    physical_device.identity.pci.function = 0;
    physical_device.identity.numa.node_id = 2;
    physical_device.identity.flags =
        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID |
        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS |
        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE;
    physical_device.partition_count = 1;
    physical_device.physical_device_affinity = 1;

    iree_hal_device_identity_spec_t identity = {};
    identity.logical_device_id = iree_make_cstring_view("fuzz-device-0");
    identity.display_name = iree_make_cstring_view("Fuzz Device");
    identity.driver_id = iree_make_cstring_view("fuzz-driver");
    identity.driver_version = iree_make_cstring_view("1.2.3");
    identity.backend_id = iree_make_cstring_view("fuzz-backend");
    identity.device_path = iree_make_cstring_view("fuzz://device/0");
    identity.vendor_name = iree_make_cstring_view("Fuzz Vendor");
    identity.vendor_id = 0x1002;
    identity.device_id = 0x744c;
    identity.revision_id = 1;
    identity.physical_device_count = 1;
    identity.physical_devices = &physical_device;

    iree_hal_memory_heap_spec_t memory_heap = {};
    memory_heap.name = iree_make_cstring_view("device-local");
    memory_heap.capacity_bytes = UINT64_C(8) * 1024 * 1024 * 1024;
    memory_heap.allocation_granularity = 4096;
    memory_heap.allocation_alignment = 256;
    memory_heap.maximum_allocation_size = UINT64_C(4) * 1024 * 1024 * 1024;
    memory_heap.physical_device_affinity = 1;

    iree_hal_memory_type_spec_t memory_type = {};
    memory_type.heap_index = 0;
    memory_type.memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    memory_type.allowed_buffer_usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    memory_type.allowed_memory_access = IREE_HAL_MEMORY_ACCESS_ALL;
    memory_type.minimum_alignment = 256;
    memory_type.optimal_transfer_granularity = 4096;

    iree_hal_external_buffer_handle_spec_t external_buffer_handle = {};
    external_buffer_handle.handle_type_mask =
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD |
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF;
    external_buffer_handle.direction_flags =
        IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
        IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT;
    external_buffer_handle.allowed_buffer_usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE_READ;
    external_buffer_handle.allowed_memory_access =
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
    external_buffer_handle.compatible_memory_type_mask = 1u;
    external_buffer_handle.flags =
        IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS |
        IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_OWNING;

    iree_hal_device_memory_spec_t memory = {};
    memory.heap_count = 1;
    memory.heaps = &memory_heap;
    memory.memory_type_count = 1;
    memory.memory_types = &memory_type;
    memory.external_buffer_handle_count = 1;
    memory.external_buffer_handles = &external_buffer_handle;

    iree_hal_virtual_memory_class_spec_t virtual_memory_class = {};
    virtual_memory_class.compatible_memory_type_mask = 1u;
    virtual_memory_class.allowed_buffer_usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    virtual_memory_class.allowed_memory_access = IREE_HAL_MEMORY_ACCESS_ALL;
    virtual_memory_class.minimum_page_size = 4096;
    virtual_memory_class.recommended_page_size = 65536;
    virtual_memory_class.maximum_reservation_size = UINT64_C(1) << 40;
    virtual_memory_class.maximum_physical_allocation_size = UINT64_C(1) << 32;
    virtual_memory_class.operation_flags =
        IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RESERVE |
        IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RELEASE |
        IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PHYSICAL_ALLOCATE |
        IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PHYSICAL_FREE |
        IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_MAP |
        IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_UNMAP |
        IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PROTECT |
        IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_ADVISE;
    virtual_memory_class.protection_flags =
        IREE_HAL_MEMORY_PROTECTION_READ_WRITE;
    virtual_memory_class.advice_flags = IREE_HAL_MEMORY_ADVICE_WILL_NEED;

    iree_hal_device_virtual_memory_spec_t virtual_memory = {};
    virtual_memory.class_count = 1;
    virtual_memory.classes = &virtual_memory_class;

    iree_hal_queue_family_spec_t queue_family = {};
    queue_family.name = iree_make_cstring_view("dispatch-and-transfer");
    queue_family.queue_count = 4;
    queue_family.priority_count = 2;
    queue_family.timestamp_valid_bits = 64;
    queue_family.timestamp_frequency_hz = UINT64_C(1000000000);
    queue_family.physical_device_affinity = 1;
    queue_family.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                              IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                              IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING;

    iree_hal_external_timepoint_handle_spec_t external_timepoint_handle = {};
    external_timepoint_handle.handle_type =
        IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_ASYNC_PRIMITIVE;
    external_timepoint_handle.direction_flags =
        IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
        IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT;
    external_timepoint_handle.compatibility =
        IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
        IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL;
    external_timepoint_handle.flags =
        IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS |
        IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_OWNING;

    iree_hal_device_queue_spec_t queues = {};
    queues.family_count = 1;
    queues.families = &queue_family;
    queues.external_timepoint_handle_count = 1;
    queues.external_timepoint_handles = &external_timepoint_handle;

    iree_hal_device_dispatch_spec_t dispatch = {};
    dispatch.launch.maximum_workgroup_invocations = 1024;
    dispatch.launch.maximum_workgroup_size[0] = 1024;
    dispatch.launch.maximum_workgroup_size[1] = 1024;
    dispatch.launch.maximum_workgroup_size[2] = 64;
    dispatch.launch.maximum_workgroup_count[0] = 65535;
    dispatch.launch.maximum_workgroup_count[1] = 65535;
    dispatch.launch.maximum_workgroup_count[2] = 65535;
    dispatch.subgroup.default_size = 64;
    dispatch.subgroup.minimum_size = 32;
    dispatch.subgroup.maximum_size = 64;
    dispatch.subgroup.supported_size_mask = UINT64_C(1) << 32;
    dispatch.execution.unit_count = 120;
    dispatch.execution.group_count = 8;
    dispatch.execution.maximum_resident_workgroup_count = 16;
    dispatch.execution.maximum_resident_invocation_count = 2048;
    dispatch.execution.maximum_resident_subgroup_count = 32;
    dispatch.execution.maximum_register_count = 65536;
    dispatch.execution.maximum_workgroup_register_count = 65536;
    dispatch.execution.maximum_local_memory_size = 64 * 1024;
    dispatch.execution.maximum_workgroup_local_memory_size = 64 * 1024;
    dispatch.execution.maximum_workgroup_local_memory_size_optin = 96 * 1024;
    dispatch.addressing.pointer_size_bits = 64;
    dispatch.addressing.address_space_bits = 64;
    dispatch.addressing.minimum_buffer_device_address_alignment = 256;

    iree_hal_device_timing_spec_t timing = {};
    timing.timestamp_valid_bits = 64;
    timing.timestamp_frequency_hz = UINT64_C(1000000000);
    timing.flags = IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS |
                   IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HOST_CORRELATION |
                   IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HARDWARE_COUNTERS;

    iree_hal_executable_target_t executable_target = {};
    executable_target.family = iree_make_cstring_view("amdgpu");
    executable_target.target_key = iree_make_cstring_view("gfx1100:xnack-");
    executable_target.kind = IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT;
    executable_target.priority = 100;
    executable_target.physical_device_affinity = 1;

    iree_hal_device_executable_spec_t executables = {};
    executables.target_count = 1;
    executables.targets = &executable_target;

    iree_hal_device_sanitizer_spec_t sanitizer = {};
    sanitizer.flags = IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN;
    sanitizer.asan.pool_options.mode = IREE_HAL_ASAN_POOL_MODE_SHADOW;
    sanitizer.asan.pool_options.shadow_granule_size = 8;
    sanitizer.asan.pool_options.redzone_size = 64;
    sanitizer.asan.pool_options.backing_alignment = 4096;
    sanitizer.asan.pool_options.quarantine_size = 1024 * 1024;

    static const uint8_t facet_payload[] = {0x00, 0x7f, 0x80, 0xff};
    iree_hal_device_spec_facet_t facet = {};
    facet.schema_id = iree_make_cstring_view("fuzz.example.facet");
    facet.schema_version = 7;
    facet.payload =
        iree_make_const_byte_span(facet_payload, sizeof(facet_payload));

    iree_hal_device_spec_params_t params = {};
    params.identity = &identity;
    params.memory = &memory;
    params.virtual_memory = &virtual_memory;
    params.queues = &queues;
    params.dispatch = &dispatch;
    params.timing = &timing;
    params.executables = &executables;
    params.sanitizer = &sanitizer;
    params.facet_count = 1;
    params.facets = &facet;

    iree_hal_device_spec_t* spec = NULL;
    iree_status_t status =
        iree_hal_device_spec_create(&params, iree_allocator_system(), &spec);
    iree_hal_device_spec_fuzz_assert(iree_status_is_ok(status));
    iree_byte_span_t bytes = iree_byte_span_empty();
    status =
        iree_hal_device_spec_serialize(spec, iree_allocator_system(), &bytes);
    iree_hal_device_spec_fuzz_assert(iree_status_is_ok(status));
    std::vector<uint8_t> result(bytes.data, bytes.data + bytes.data_length);
    iree_allocator_free(iree_allocator_system(), bytes.data);
    iree_hal_device_spec_release(spec);
    return result;
  }();
  return seed;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  constexpr size_t kMaximumInputSize = 1024 * 1024;
  if (size > kMaximumInputSize) return 0;

  // Exercise the complete hostile-input parser surface.
  iree_hal_device_spec_fuzz_parse(iree_make_const_byte_span(data, size));

  // Also mutate a valid, comprehensive seed so fuzzing reaches every record
  // decoder without requiring the engine to synthesize the full format. The
  // mutation program uses a 16-bit offset so every byte in the seed is
  // reachable; a single-byte offset would silently limit coverage to the first
  // 256 bytes.
  const std::vector<uint8_t>& seed = iree_hal_device_spec_fuzz_seed();
  std::vector<uint8_t> mutated(seed);
  const size_t maximum_mutation_program_size = seed.size() * 3;
  const size_t mutation_program_size = size < maximum_mutation_program_size
                                           ? size
                                           : maximum_mutation_program_size;
  for (size_t i = 0; i + 2 < mutation_program_size; i += 3) {
    const size_t requested_offset =
        (size_t)data[i] | ((size_t)data[i + 1] << 8);
    const size_t offset = requested_offset % mutated.size();
    mutated[offset] ^= data[i + 2];
  }
  iree_hal_device_spec_fuzz_parse(
      iree_make_const_byte_span(mutated.data(), mutated.size()));
  return 0;
}
