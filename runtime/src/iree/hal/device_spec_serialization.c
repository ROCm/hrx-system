// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_spec_serialization.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/alignment.h"

//===----------------------------------------------------------------------===//
// Canonical wire format
//===----------------------------------------------------------------------===//

// Device specs use a fixed-schema forward-only stream. All integers are little
// endian, all strings and payloads are byte-length-prefixed, and all arrays are
// emitted in source order. The format contains no native structs, alignment
// gaps, or padding bytes.
#define IREE_HAL_DEVICE_SPEC_MAGIC UINT32_C(0x43505344)  // DSPC
#define IREE_HAL_DEVICE_SPEC_VERSION UINT32_C(7)
#define IREE_HAL_DEVICE_SPEC_FNV1A64_OFFSET_BASIS UINT64_C(0xcbf29ce484222325)
#define IREE_HAL_DEVICE_SPEC_FNV1A64_PRIME UINT64_C(0x100000001b3)

typedef struct iree_hal_device_spec_writer_t {
  // Optional output storage, or NULL when measuring or hashing.
  uint8_t* data;
  // Maximum bytes the writer may produce.
  iree_host_size_t capacity;
  // Number of bytes produced so far.
  iree_host_size_t offset;
  // Rolling FNV-1a state when |is_fingerprinting| is true.
  uint64_t fingerprint;
  // Whether written bytes are incorporated into |fingerprint|.
  bool is_fingerprinting;
} iree_hal_device_spec_writer_t;

static void iree_hal_device_spec_writer_initialize(
    uint8_t* data, iree_host_size_t capacity, bool is_fingerprinting,
    iree_hal_device_spec_writer_t* out_writer) {
  *out_writer = (iree_hal_device_spec_writer_t){
      .data = data,
      .capacity = capacity,
      .offset = 0,
      .fingerprint = IREE_HAL_DEVICE_SPEC_FNV1A64_OFFSET_BASIS,
      .is_fingerprinting = is_fingerprinting,
  };
}

static iree_status_t iree_hal_device_spec_writer_write_bytes(
    iree_hal_device_spec_writer_t* writer, const void* source,
    iree_host_size_t length) {
  if (IREE_UNLIKELY(length && !source)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty device spec field has NULL storage");
  }
  if (IREE_UNLIKELY(length > writer->capacity - writer->offset)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "canonical device spec size overflow");
  }
  const uint8_t* source_bytes = (const uint8_t*)source;
  if (writer->data && length) {
    memcpy(writer->data + writer->offset, source_bytes, length);
  }
  if (writer->is_fingerprinting) {
    for (iree_host_size_t i = 0; i < length; ++i) {
      writer->fingerprint ^= source_bytes[i];
      writer->fingerprint *= IREE_HAL_DEVICE_SPEC_FNV1A64_PRIME;
    }
  }
  writer->offset += length;
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_writer_write_u8(
    iree_hal_device_spec_writer_t* writer, uint8_t value) {
  return iree_hal_device_spec_writer_write_bytes(writer, &value, sizeof(value));
}

static iree_status_t iree_hal_device_spec_writer_write_u16(
    iree_hal_device_spec_writer_t* writer, uint16_t value) {
  uint8_t storage[sizeof(value)];
  iree_unaligned_store_le_u16(storage, value);
  return iree_hal_device_spec_writer_write_bytes(writer, storage,
                                                 sizeof(storage));
}

static iree_status_t iree_hal_device_spec_writer_write_u32(
    iree_hal_device_spec_writer_t* writer, uint32_t value) {
  uint8_t storage[sizeof(value)];
  iree_unaligned_store_le_u32(storage, value);
  return iree_hal_device_spec_writer_write_bytes(writer, storage,
                                                 sizeof(storage));
}

static iree_status_t iree_hal_device_spec_writer_write_u64(
    iree_hal_device_spec_writer_t* writer, uint64_t value) {
  uint8_t storage[sizeof(value)];
  iree_unaligned_store_le_u64(storage, value);
  return iree_hal_device_spec_writer_write_bytes(writer, storage,
                                                 sizeof(storage));
}

static iree_status_t iree_hal_device_spec_writer_write_size(
    iree_hal_device_spec_writer_t* writer, iree_host_size_t value) {
  return iree_hal_device_spec_writer_write_u64(writer, (uint64_t)value);
}

static iree_status_t iree_hal_device_spec_writer_write_string(
    iree_hal_device_spec_writer_t* writer, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_size(writer, value.size));
  return iree_hal_device_spec_writer_write_bytes(writer, value.data,
                                                 value.size);
}

static iree_status_t iree_hal_device_spec_writer_write_payload(
    iree_hal_device_spec_writer_t* writer, iree_const_byte_span_t value) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_size(writer, value.data_length));
  return iree_hal_device_spec_writer_write_bytes(writer, value.data,
                                                 value.data_length);
}

static iree_status_t iree_hal_device_spec_encode_physical_device(
    const iree_hal_physical_device_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  const iree_hal_physical_device_identity_t* identity = &value->identity;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, identity->display_name));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, identity->backend_path));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->vendor_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->device_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->revision_id));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_bytes(
      writer, identity->uuid.bytes, sizeof(identity->uuid.bytes)));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->pci.domain));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->pci.bus));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->pci.device));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->pci.function));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->numa.node_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->flags));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->physical_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->partition_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->partition_count));
  return iree_hal_device_spec_writer_write_u64(writer,
                                               value->physical_device_affinity);
}

static iree_status_t iree_hal_device_spec_encode_memory_heap(
    const iree_hal_memory_heap_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, value->name));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u64(writer, value->capacity_bytes));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->allocation_granularity));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->allocation_alignment));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->maximum_allocation_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->physical_device_affinity));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_atomic_operations(
    const iree_hal_atomic_operation_capabilities_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->device_scope_32));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->device_scope_64));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->system_scope_32));
  return iree_hal_device_spec_writer_write_u32(writer, value->system_scope_64);
}

static iree_status_t iree_hal_device_spec_encode_atomic_wait_conditions(
    const iree_hal_atomic_wait_condition_capabilities_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->device_scope_32));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->device_scope_64));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->system_scope_32));
  return iree_hal_device_spec_writer_write_u32(writer, value->system_scope_64);
}

static iree_status_t iree_hal_device_spec_encode_atomic_capabilities(
    const iree_hal_atomic_capabilities_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_atomic_operations(
      &value->operations, writer));
  return iree_hal_device_spec_encode_atomic_wait_conditions(
      &value->wait_conditions, writer);
}

static iree_status_t iree_hal_device_spec_encode_memory_type(
    const iree_hal_memory_type_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->heap_index));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->memory_type));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->allowed_buffer_usage));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u16(
      writer, value->allowed_memory_access));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u64(writer, value->minimum_alignment));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->optimal_transfer_granularity));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_atomic_operations(
      &value->atomic_operations, writer));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_external_buffer_handle(
    const iree_hal_external_buffer_handle_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u8(writer, value->handle_type_mask));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->direction_flags));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->allowed_buffer_usage));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u16(
      writer, value->allowed_memory_access));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->compatible_memory_type_mask));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_virtual_memory_class(
    const iree_hal_virtual_memory_class_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->compatible_memory_type_mask));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->allowed_buffer_usage));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u16(
      writer, value->allowed_memory_access));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u64(writer, value->minimum_page_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->recommended_page_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->maximum_reservation_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->maximum_physical_allocation_size));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->operation_flags));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u64(writer, value->protection_flags));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u64(writer, value->advice_flags));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_queue_family(
    const iree_hal_queue_family_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, value->name));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->queue_count));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->priority_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->timestamp_valid_bits));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->timestamp_frequency_hz));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->physical_device_affinity));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u64(writer, value->queue_affinity));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->role_flags));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_atomic_capabilities(
      &value->atomic_capabilities, writer));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_atomic_capabilities(
      &value->zero_compute_atomic_capabilities, writer));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_external_timepoint_handle(
    const iree_hal_external_timepoint_handle_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, (uint32_t)value->handle_type));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->direction_flags));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->compatibility));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_dispatch(
    const iree_hal_device_dispatch_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->launch.maximum_workgroup_invocations));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(value->launch.maximum_workgroup_size); ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
        writer, value->launch.maximum_workgroup_size[i]));
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(value->launch.maximum_workgroup_count); ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
        writer, value->launch.maximum_workgroup_count[i]));
  }
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->subgroup.default_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->subgroup.minimum_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->subgroup.maximum_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->subgroup.supported_size_mask));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->execution.unit_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->execution.group_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->execution.maximum_resident_workgroup_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->execution.maximum_resident_invocation_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->execution.maximum_resident_subgroup_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->execution.maximum_register_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->execution.maximum_workgroup_register_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->execution.maximum_local_memory_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->execution.maximum_workgroup_local_memory_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->execution.maximum_workgroup_local_memory_size_optin));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->addressing.pointer_size_bits));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->addressing.address_space_bits));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->addressing.minimum_buffer_device_address_alignment));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_timing(
    const iree_hal_device_timing_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, value->timestamp_valid_bits));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->timestamp_frequency_hz));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_executable_target(
    const iree_hal_executable_target_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, value->family));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, value->target_key));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->kind));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->priority));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, value->physical_device_affinity));
  return iree_hal_device_spec_writer_write_u32(writer, value->flags);
}

static iree_status_t iree_hal_device_spec_encode_sanitizer(
    const iree_hal_device_sanitizer_spec_t* value,
    iree_hal_device_spec_writer_t* writer) {
  const iree_hal_asan_pool_options_t* pool_options = &value->asan.pool_options;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->flags));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, (uint32_t)pool_options->mode));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, pool_options->shadow_granule_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, pool_options->redzone_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u64(
      writer, pool_options->backing_alignment));
  return iree_hal_device_spec_writer_write_u64(writer,
                                               pool_options->quarantine_size);
}

static iree_status_t iree_hal_device_spec_encode_facet(
    const iree_hal_device_spec_facet_t* value,
    iree_hal_device_spec_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, value->schema_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, value->schema_version));
  return iree_hal_device_spec_writer_write_payload(writer, value->payload);
}

static iree_status_t iree_hal_device_spec_encode(
    const iree_hal_device_spec_t* spec, iree_host_size_t total_length,
    iree_hal_device_spec_writer_t* writer) {
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(spec);
  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(spec);
  const iree_hal_device_virtual_memory_spec_t* virtual_memory =
      iree_hal_device_spec_virtual_memory(spec);
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(spec);
  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(spec);
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(spec);
  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(spec);
  const iree_hal_device_sanitizer_spec_t* sanitizer =
      iree_hal_device_spec_sanitizer(spec);
  const iree_host_size_t facet_count = iree_hal_device_spec_facet_count(spec);

  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, IREE_HAL_DEVICE_SPEC_MAGIC));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_u32(
      writer, IREE_HAL_DEVICE_SPEC_VERSION));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_size(writer, total_length));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_size(
      writer, identity->physical_device_count));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_size(writer, memory->heap_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_size(
      writer, memory->memory_type_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_size(
      writer, memory->external_buffer_handle_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_size(
      writer, virtual_memory->class_count));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_size(writer, queues->family_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_size(
      writer, queues->external_timepoint_handle_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_size(
      writer, executables->target_count));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_size(writer, facet_count));

  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_string(
      writer, identity->logical_device_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, identity->display_name));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, identity->driver_id));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_writer_write_string(
      writer, identity->driver_version));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, identity->backend_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, identity->device_path));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_string(writer, identity->vendor_name));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->vendor_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->device_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->revision_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->logical_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, identity->flags));
  for (iree_host_size_t i = 0; i < identity->physical_device_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_physical_device(
        &identity->physical_devices[i], writer));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, memory->flags));
  for (iree_host_size_t i = 0; i < memory->heap_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_encode_memory_heap(&memory->heaps[i], writer));
  }
  for (iree_host_size_t i = 0; i < memory->memory_type_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_memory_type(
        &memory->memory_types[i], writer));
  }
  for (iree_host_size_t i = 0; i < memory->external_buffer_handle_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_external_buffer_handle(
        &memory->external_buffer_handles[i], writer));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, virtual_memory->flags));
  for (iree_host_size_t i = 0; i < virtual_memory->class_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_virtual_memory_class(
        &virtual_memory->classes[i], writer));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, queues->flags));
  for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_encode_queue_family(&queues->families[i], writer));
  }
  for (iree_host_size_t i = 0; i < queues->external_timepoint_handle_count;
       ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_external_timepoint_handle(
        &queues->external_timepoint_handles[i], writer));
  }

  IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_dispatch(dispatch, writer));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_timing(timing, writer));

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_writer_write_u32(writer, executables->flags));
  for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_executable_target(
        &executables->targets[i], writer));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_encode_sanitizer(sanitizer, writer));
  for (iree_host_size_t i = 0; i < facet_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode_facet(
        iree_hal_device_spec_facet_at(spec, i), writer));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_measure(
    const iree_hal_device_spec_t* spec, iree_host_size_t* out_length) {
  iree_hal_device_spec_writer_t writer;
  iree_hal_device_spec_writer_initialize(NULL, IREE_HOST_SIZE_MAX, false,
                                         &writer);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_encode(spec, 0, &writer));
  *out_length = writer.offset;
  return iree_ok_status();
}

iree_status_t iree_hal_device_spec_compute_fingerprint(
    const iree_hal_device_spec_t* spec, uint64_t* out_fingerprint) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(out_fingerprint);
  *out_fingerprint = 0;
  iree_host_size_t total_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_measure(spec, &total_length));
  iree_hal_device_spec_writer_t writer;
  iree_hal_device_spec_writer_initialize(NULL, total_length, true, &writer);
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_encode(spec, total_length, &writer));
  if (IREE_UNLIKELY(writer.offset != total_length)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "canonical device spec size changed while "
                            "fingerprinting");
  }
  *out_fingerprint = writer.fingerprint;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_device_spec_serialize(
    const iree_hal_device_spec_t* spec, iree_allocator_t host_allocator,
    iree_byte_span_t* out_bytes) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(out_bytes);
  *out_bytes = iree_byte_span_empty();

  iree_host_size_t total_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_measure(spec, &total_length));
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, total_length, (void**)&data));

  iree_hal_device_spec_writer_t writer;
  iree_hal_device_spec_writer_initialize(data, total_length, false, &writer);
  iree_status_t status =
      iree_hal_device_spec_encode(spec, total_length, &writer);
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(writer.offset != total_length)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "canonical device spec size changed while serializing");
  }
  if (iree_status_is_ok(status)) {
    *out_bytes = iree_make_byte_span(data, total_length);
  } else {
    iree_allocator_free(host_allocator, data);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Canonical wire format parser
//===----------------------------------------------------------------------===//

typedef struct iree_hal_device_spec_reader_t {
  // Complete serialized image being read.
  iree_const_byte_span_t bytes;
  // Current byte offset within |bytes|.
  iree_host_size_t offset;
} iree_hal_device_spec_reader_t;

typedef struct iree_hal_device_spec_counts_t {
  // Number of physical device records.
  iree_host_size_t physical_device_count;
  // Number of memory heap records.
  iree_host_size_t memory_heap_count;
  // Number of memory type records.
  iree_host_size_t memory_type_count;
  // Number of external buffer handle records.
  iree_host_size_t external_buffer_handle_count;
  // Number of virtual memory class records.
  iree_host_size_t virtual_memory_class_count;
  // Number of queue family records.
  iree_host_size_t queue_family_count;
  // Number of external timepoint handle records.
  iree_host_size_t external_timepoint_handle_count;
  // Number of executable target records.
  iree_host_size_t executable_target_count;
  // Number of driver-local facet records.
  iree_host_size_t facet_count;
} iree_hal_device_spec_counts_t;

static iree_status_t iree_hal_device_spec_reader_read_bytes(
    iree_hal_device_spec_reader_t* reader, iree_host_size_t length,
    iree_const_byte_span_t* out_value) {
  if (IREE_UNLIKELY(length > reader->bytes.data_length - reader->offset)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device spec image is truncated at byte %" PRIhsz "; need %" PRIhsz
        " bytes but only %" PRIhsz " remain",
        reader->offset, length, reader->bytes.data_length - reader->offset);
  }
  *out_value =
      iree_make_const_byte_span(reader->bytes.data + reader->offset, length);
  reader->offset += length;
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_reader_read_u8(
    iree_hal_device_spec_reader_t* reader, uint8_t* out_value) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_bytes(reader, 1, &bytes));
  *out_value = bytes.data[0];
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_reader_read_u16(
    iree_hal_device_spec_reader_t* reader, uint16_t* out_value) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_bytes(reader, 2, &bytes));
  *out_value = iree_unaligned_load_le_u16(bytes.data);
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_reader_read_u32(
    iree_hal_device_spec_reader_t* reader, uint32_t* out_value) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_bytes(reader, 4, &bytes));
  *out_value = iree_unaligned_load_le_u32(bytes.data);
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_reader_read_u64(
    iree_hal_device_spec_reader_t* reader, uint64_t* out_value) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_bytes(reader, 8, &bytes));
  *out_value = iree_unaligned_load_le_u64(bytes.data);
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_reader_read_device_size(
    iree_hal_device_spec_reader_t* reader, const char* field_name,
    iree_device_size_t* out_value) {
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(reader, &value));
  if (IREE_UNLIKELY(value > (uint64_t)IREE_DEVICE_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s exceeds the device size limit",
                            field_name);
  }
  *out_value = (iree_device_size_t)value;
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_reader_read_size(
    iree_hal_device_spec_reader_t* reader, const char* field_name,
    iree_host_size_t* out_value) {
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(reader, &value));
  if (IREE_UNLIKELY(value > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s exceeds the host size limit",
                            field_name);
  }
  *out_value = (iree_host_size_t)value;
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_reader_read_string(
    iree_hal_device_spec_reader_t* reader, iree_string_view_t* out_value) {
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_size(reader, "string length", &length));
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_bytes(reader, length, &bytes));
  *out_value = iree_make_string_view((const char*)bytes.data, length);
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_reader_read_payload(
    iree_hal_device_spec_reader_t* reader, iree_const_byte_span_t* out_value) {
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_size(reader, "payload length", &length));
  return iree_hal_device_spec_reader_read_bytes(reader, length, out_value);
}

static iree_status_t iree_hal_device_spec_decode_header(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_device_spec_counts_t* out_counts) {
  uint32_t magic = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(reader, &magic));
  if (IREE_UNLIKELY(magic != IREE_HAL_DEVICE_SPEC_MAGIC)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec image has invalid magic");
  }
  uint32_t version = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(reader, &version));
  if (IREE_UNLIKELY(version != IREE_HAL_DEVICE_SPEC_VERSION)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device spec image version %" PRIu32 " is unsupported", version);
  }
  iree_host_size_t total_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "total length", &total_length));
  if (IREE_UNLIKELY(total_length != reader->bytes.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec total length does not match image");
  }
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "physical device count", &out_counts->physical_device_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "memory heap count", &out_counts->memory_heap_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "memory type count", &out_counts->memory_type_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "external buffer handle count",
      &out_counts->external_buffer_handle_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "virtual memory class count",
      &out_counts->virtual_memory_class_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "queue family count", &out_counts->queue_family_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "external timepoint handle count",
      &out_counts->external_timepoint_handle_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_size(
      reader, "executable target count", &out_counts->executable_target_count));
  return iree_hal_device_spec_reader_read_size(reader, "facet count",
                                               &out_counts->facet_count);
}

static iree_status_t iree_hal_device_spec_decode_physical_device(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_physical_device_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  iree_hal_physical_device_identity_t* identity = &out_value->identity;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_string(reader, &identity->display_name));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_string(reader, &identity->backend_path));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->vendor_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->device_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->revision_id));
  iree_const_byte_span_t uuid_bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_bytes(
      reader, sizeof(identity->uuid.bytes), &uuid_bytes));
  memcpy(identity->uuid.bytes, uuid_bytes.data, uuid_bytes.data_length);
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->pci.domain));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->pci.bus));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->pci.device));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->pci.function));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->numa.node_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &identity->flags));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->physical_ordinal));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->partition_ordinal));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->partition_count));
  return iree_hal_device_spec_reader_read_u64(
      reader, &out_value->physical_device_affinity);
}

static iree_status_t iree_hal_device_spec_decode_memory_heap(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_memory_heap_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_string(reader, &out_value->name));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u64(reader, &out_value->capacity_bytes));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->allocation_granularity));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->allocation_alignment));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->maximum_allocation_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->physical_device_affinity));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_atomic_operations(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_atomic_operation_capabilities_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->device_scope_32));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->device_scope_64));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->system_scope_32));
  return iree_hal_device_spec_reader_read_u32(reader,
                                              &out_value->system_scope_64);
}

static iree_status_t iree_hal_device_spec_decode_atomic_wait_conditions(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_atomic_wait_condition_capabilities_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->device_scope_32));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->device_scope_64));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->system_scope_32));
  return iree_hal_device_spec_reader_read_u32(reader,
                                              &out_value->system_scope_64);
}

static iree_status_t iree_hal_device_spec_decode_atomic_capabilities(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_atomic_capabilities_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_decode_atomic_operations(
      reader, &out_value->operations));
  return iree_hal_device_spec_decode_atomic_wait_conditions(
      reader, &out_value->wait_conditions);
}

static iree_status_t iree_hal_device_spec_decode_memory_type(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_memory_type_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->heap_index));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->memory_type));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->allowed_buffer_usage));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u16(
      reader, &out_value->allowed_memory_access));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->minimum_alignment));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->optimal_transfer_granularity));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_decode_atomic_operations(
      reader, &out_value->atomic_operations));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_external_buffer_handle(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_external_buffer_handle_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u8(
      reader, &out_value->handle_type_mask));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->direction_flags));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->allowed_buffer_usage));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u16(
      reader, &out_value->allowed_memory_access));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->compatible_memory_type_mask));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_virtual_memory_class(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_virtual_memory_class_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->compatible_memory_type_mask));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->allowed_buffer_usage));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u16(
      reader, &out_value->allowed_memory_access));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->minimum_page_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->recommended_page_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->maximum_reservation_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->maximum_physical_allocation_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->operation_flags));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->protection_flags));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u64(reader, &out_value->advice_flags));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_queue_family(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_queue_family_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_string(reader, &out_value->name));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->queue_count));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->priority_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->timestamp_valid_bits));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->timestamp_frequency_hz));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->physical_device_affinity));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u64(reader, &out_value->queue_affinity));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->role_flags));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_decode_atomic_capabilities(
      reader, &out_value->atomic_capabilities));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_decode_atomic_capabilities(
      reader, &out_value->zero_compute_atomic_capabilities));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_external_timepoint_handle(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_external_timepoint_handle_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  uint32_t handle_type = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &handle_type));
  if (!iree_hal_external_timepoint_type_is_valid(
          (iree_hal_external_timepoint_type_t)handle_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device spec external timepoint handle has invalid type %" PRIu32,
        handle_type);
  }
  out_value->handle_type = (iree_hal_external_timepoint_type_t)handle_type;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->direction_flags));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->compatibility));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_dispatch(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_device_dispatch_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->launch.maximum_workgroup_invocations));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(out_value->launch.maximum_workgroup_size); ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
        reader, &out_value->launch.maximum_workgroup_size[i]));
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(out_value->launch.maximum_workgroup_count); ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
        reader, &out_value->launch.maximum_workgroup_count[i]));
  }
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->subgroup.default_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->subgroup.minimum_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->subgroup.maximum_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->subgroup.supported_size_mask));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->execution.unit_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->execution.group_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->execution.maximum_resident_workgroup_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->execution.maximum_resident_invocation_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->execution.maximum_resident_subgroup_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->execution.maximum_register_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->execution.maximum_workgroup_register_count));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->execution.maximum_local_memory_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->execution.maximum_workgroup_local_memory_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->execution.maximum_workgroup_local_memory_size_optin));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->addressing.pointer_size_bits));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->addressing.address_space_bits));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->addressing.minimum_buffer_device_address_alignment));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_timing(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_device_timing_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_value->timestamp_valid_bits));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->timestamp_frequency_hz));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_executable_target(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_executable_target_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_string(reader, &out_value->family));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_string(reader, &out_value->target_key));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->kind));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->priority));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u64(
      reader, &out_value->physical_device_affinity));
  return iree_hal_device_spec_reader_read_u32(reader, &out_value->flags);
}

static iree_status_t iree_hal_device_spec_decode_sanitizer(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_device_sanitizer_spec_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  iree_hal_asan_pool_options_t* pool_options = &out_value->asan.pool_options;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->flags));
  uint32_t mode = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(reader, &mode));
  switch (mode) {
    case IREE_HAL_ASAN_POOL_MODE_DISABLED:
    case IREE_HAL_ASAN_POOL_MODE_SHADOW:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device spec has invalid ASAN mode %" PRIu32,
                              mode);
  }
  pool_options->mode = (iree_hal_asan_pool_mode_t)mode;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_device_size(
      reader, "ASAN shadow granule size", &pool_options->shadow_granule_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_device_size(
      reader, "ASAN redzone size", &pool_options->redzone_size));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_device_size(
      reader, "ASAN backing alignment", &pool_options->backing_alignment));
  return iree_hal_device_spec_reader_read_device_size(
      reader, "ASAN quarantine size", &pool_options->quarantine_size);
}

static iree_status_t iree_hal_device_spec_decode_facet(
    iree_hal_device_spec_reader_t* reader,
    iree_hal_device_spec_facet_t* out_value) {
  memset(out_value, 0, sizeof(*out_value));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_string(reader, &out_value->schema_id));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_value->schema_version));
  return iree_hal_device_spec_reader_read_payload(reader, &out_value->payload);
}

typedef struct iree_hal_device_spec_parse_storage_t {
  // Base allocation containing all temporary record arrays.
  void* allocation;
  // Temporary physical device records.
  iree_hal_physical_device_spec_t* physical_devices;
  // Temporary memory heap records.
  iree_hal_memory_heap_spec_t* memory_heaps;
  // Temporary memory type records.
  iree_hal_memory_type_spec_t* memory_types;
  // Temporary external buffer handle records.
  iree_hal_external_buffer_handle_spec_t* external_buffer_handles;
  // Temporary virtual memory class records.
  iree_hal_virtual_memory_class_spec_t* virtual_memory_classes;
  // Temporary queue family records.
  iree_hal_queue_family_spec_t* queue_families;
  // Temporary external timepoint handle records.
  iree_hal_external_timepoint_handle_spec_t* external_timepoint_handles;
  // Temporary executable target records.
  iree_hal_executable_target_t* executable_targets;
  // Temporary driver-local facet records.
  iree_hal_device_spec_facet_t* facets;
} iree_hal_device_spec_parse_storage_t;

typedef struct iree_hal_device_spec_parse_layout_t {
  // Total native temporary storage size.
  iree_host_size_t total_length;
  // Byte offset of physical device records.
  iree_host_size_t physical_devices;
  // Byte offset of memory heap records.
  iree_host_size_t memory_heaps;
  // Byte offset of memory type records.
  iree_host_size_t memory_types;
  // Byte offset of external buffer handle records.
  iree_host_size_t external_buffer_handles;
  // Byte offset of virtual memory class records.
  iree_host_size_t virtual_memory_classes;
  // Byte offset of queue family records.
  iree_host_size_t queue_families;
  // Byte offset of external timepoint handle records.
  iree_host_size_t external_timepoint_handles;
  // Byte offset of executable target records.
  iree_host_size_t executable_targets;
  // Byte offset of driver-local facet records.
  iree_host_size_t facets;
} iree_hal_device_spec_parse_layout_t;

static iree_status_t iree_hal_device_spec_layout_parse_array(
    iree_host_size_t count, iree_host_size_t element_size,
    iree_host_size_t alignment, iree_host_size_t* inout_total_length,
    iree_host_size_t* out_offset) {
  *out_offset = 0;
  if (!count) return iree_ok_status();
  if (IREE_UNLIKELY(count > IREE_HOST_SIZE_MAX / element_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec parse array size overflow");
  }
  const iree_host_size_t aligned_offset =
      iree_host_align(*inout_total_length, alignment);
  if (IREE_UNLIKELY(aligned_offset < *inout_total_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec parse array offset overflow");
  }
  const iree_host_size_t length = count * element_size;
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX - aligned_offset)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec parse storage size overflow");
  }
  *out_offset = aligned_offset;
  *inout_total_length = aligned_offset + length;
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_parse_storage_initialize(
    const iree_hal_device_spec_counts_t* counts,
    iree_allocator_t host_allocator,
    iree_hal_device_spec_parse_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  iree_hal_device_spec_parse_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->physical_device_count, sizeof(*out_storage->physical_devices),
      iree_alignof(iree_hal_physical_device_spec_t), &layout.total_length,
      &layout.physical_devices));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->memory_heap_count, sizeof(*out_storage->memory_heaps),
      iree_alignof(iree_hal_memory_heap_spec_t), &layout.total_length,
      &layout.memory_heaps));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->memory_type_count, sizeof(*out_storage->memory_types),
      iree_alignof(iree_hal_memory_type_spec_t), &layout.total_length,
      &layout.memory_types));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->external_buffer_handle_count,
      sizeof(*out_storage->external_buffer_handles),
      iree_alignof(iree_hal_external_buffer_handle_spec_t),
      &layout.total_length, &layout.external_buffer_handles));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->virtual_memory_class_count,
      sizeof(*out_storage->virtual_memory_classes),
      iree_alignof(iree_hal_virtual_memory_class_spec_t), &layout.total_length,
      &layout.virtual_memory_classes));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->queue_family_count, sizeof(*out_storage->queue_families),
      iree_alignof(iree_hal_queue_family_spec_t), &layout.total_length,
      &layout.queue_families));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->external_timepoint_handle_count,
      sizeof(*out_storage->external_timepoint_handles),
      iree_alignof(iree_hal_external_timepoint_handle_spec_t),
      &layout.total_length, &layout.external_timepoint_handles));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->executable_target_count, sizeof(*out_storage->executable_targets),
      iree_alignof(iree_hal_executable_target_t), &layout.total_length,
      &layout.executable_targets));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_parse_array(
      counts->facet_count, sizeof(*out_storage->facets),
      iree_alignof(iree_hal_device_spec_facet_t), &layout.total_length,
      &layout.facets));

  if (!layout.total_length) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, layout.total_length, &out_storage->allocation));
  uint8_t* base = (uint8_t*)out_storage->allocation;
  if (counts->physical_device_count) {
    out_storage->physical_devices =
        (iree_hal_physical_device_spec_t*)(base + layout.physical_devices);
  }
  if (counts->memory_heap_count) {
    out_storage->memory_heaps =
        (iree_hal_memory_heap_spec_t*)(base + layout.memory_heaps);
  }
  if (counts->memory_type_count) {
    out_storage->memory_types =
        (iree_hal_memory_type_spec_t*)(base + layout.memory_types);
  }
  if (counts->external_buffer_handle_count) {
    out_storage->external_buffer_handles =
        (iree_hal_external_buffer_handle_spec_t*)(base +
                                                  layout
                                                      .external_buffer_handles);
  }
  if (counts->virtual_memory_class_count) {
    out_storage->virtual_memory_classes =
        (iree_hal_virtual_memory_class_spec_t*)(base +
                                                layout.virtual_memory_classes);
  }
  if (counts->queue_family_count) {
    out_storage->queue_families =
        (iree_hal_queue_family_spec_t*)(base + layout.queue_families);
  }
  if (counts->external_timepoint_handle_count) {
    out_storage->external_timepoint_handles =
        (iree_hal_external_timepoint_handle_spec_t*)(base +
                                                     layout
                                                         .external_timepoint_handles);
  }
  if (counts->executable_target_count) {
    out_storage->executable_targets =
        (iree_hal_executable_target_t*)(base + layout.executable_targets);
  }
  if (counts->facet_count) {
    out_storage->facets = (iree_hal_device_spec_facet_t*)(base + layout.facets);
  }
  return iree_ok_status();
}

typedef struct iree_hal_device_spec_decoded_t {
  // Decoded logical device identity.
  iree_hal_device_identity_spec_t identity;
  // Decoded memory capabilities.
  iree_hal_device_memory_spec_t memory;
  // Decoded virtual memory capabilities.
  iree_hal_device_virtual_memory_spec_t virtual_memory;
  // Decoded queue capabilities.
  iree_hal_device_queue_spec_t queues;
  // Decoded dispatch capabilities.
  iree_hal_device_dispatch_spec_t dispatch;
  // Decoded timing capabilities.
  iree_hal_device_timing_spec_t timing;
  // Decoded executable capabilities.
  iree_hal_device_executable_spec_t executables;
  // Decoded sanitizer configuration.
  iree_hal_device_sanitizer_spec_t sanitizer;
} iree_hal_device_spec_decoded_t;

static iree_status_t iree_hal_device_spec_decode_body(
    iree_hal_device_spec_reader_t* reader,
    const iree_hal_device_spec_counts_t* counts,
    const iree_hal_device_spec_parse_storage_t* storage,
    iree_hal_device_spec_decoded_t* out_decoded) {
  memset(out_decoded, 0, sizeof(*out_decoded));
  out_decoded->identity.physical_device_count = counts->physical_device_count;
  out_decoded->identity.physical_devices = storage->physical_devices;
  out_decoded->memory.heap_count = counts->memory_heap_count;
  out_decoded->memory.heaps = storage->memory_heaps;
  out_decoded->memory.memory_type_count = counts->memory_type_count;
  out_decoded->memory.memory_types = storage->memory_types;
  out_decoded->memory.external_buffer_handle_count =
      counts->external_buffer_handle_count;
  out_decoded->memory.external_buffer_handles =
      storage->external_buffer_handles;
  out_decoded->virtual_memory.class_count = counts->virtual_memory_class_count;
  out_decoded->virtual_memory.classes = storage->virtual_memory_classes;
  out_decoded->queues.family_count = counts->queue_family_count;
  out_decoded->queues.families = storage->queue_families;
  out_decoded->queues.external_timepoint_handle_count =
      counts->external_timepoint_handle_count;
  out_decoded->queues.external_timepoint_handles =
      storage->external_timepoint_handles;
  out_decoded->executables.target_count = counts->executable_target_count;
  out_decoded->executables.targets = storage->executable_targets;

  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_string(
      reader, &out_decoded->identity.logical_device_id));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_string(
      reader, &out_decoded->identity.display_name));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_string(
      reader, &out_decoded->identity.driver_id));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_string(
      reader, &out_decoded->identity.driver_version));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_string(
      reader, &out_decoded->identity.backend_id));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_string(
      reader, &out_decoded->identity.device_path));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_string(
      reader, &out_decoded->identity.vendor_name));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_decoded->identity.vendor_id));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_decoded->identity.device_id));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_decoded->identity.revision_id));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_decoded->identity.logical_ordinal));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_decoded->identity.flags));
  for (iree_host_size_t i = 0; i < counts->physical_device_count; ++i) {
    iree_hal_physical_device_spec_t temporary;
    iree_hal_physical_device_spec_t* value =
        storage->physical_devices ? &storage->physical_devices[i] : &temporary;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_decode_physical_device(reader, value));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_decoded->memory.flags));
  for (iree_host_size_t i = 0; i < counts->memory_heap_count; ++i) {
    iree_hal_memory_heap_spec_t temporary;
    iree_hal_memory_heap_spec_t* value =
        storage->memory_heaps ? &storage->memory_heaps[i] : &temporary;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_decode_memory_heap(reader, value));
  }
  for (iree_host_size_t i = 0; i < counts->memory_type_count; ++i) {
    iree_hal_memory_type_spec_t temporary;
    iree_hal_memory_type_spec_t* value =
        storage->memory_types ? &storage->memory_types[i] : &temporary;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_decode_memory_type(reader, value));
  }
  for (iree_host_size_t i = 0; i < counts->external_buffer_handle_count; ++i) {
    iree_hal_external_buffer_handle_spec_t temporary;
    iree_hal_external_buffer_handle_spec_t* value =
        storage->external_buffer_handles ? &storage->external_buffer_handles[i]
                                         : &temporary;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_decode_external_buffer_handle(reader, value));
  }

  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_decoded->virtual_memory.flags));
  for (iree_host_size_t i = 0; i < counts->virtual_memory_class_count; ++i) {
    iree_hal_virtual_memory_class_spec_t temporary;
    iree_hal_virtual_memory_class_spec_t* value =
        storage->virtual_memory_classes ? &storage->virtual_memory_classes[i]
                                        : &temporary;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_decode_virtual_memory_class(reader, value));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_reader_read_u32(reader, &out_decoded->queues.flags));
  for (iree_host_size_t i = 0; i < counts->queue_family_count; ++i) {
    iree_hal_queue_family_spec_t temporary;
    iree_hal_queue_family_spec_t* value =
        storage->queue_families ? &storage->queue_families[i] : &temporary;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_decode_queue_family(reader, value));
  }
  for (iree_host_size_t i = 0; i < counts->external_timepoint_handle_count;
       ++i) {
    iree_hal_external_timepoint_handle_spec_t temporary;
    iree_hal_external_timepoint_handle_spec_t* value =
        storage->external_timepoint_handles
            ? &storage->external_timepoint_handles[i]
            : &temporary;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_decode_external_timepoint_handle(reader, value));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_decode_dispatch(reader, &out_decoded->dispatch));
  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_decode_timing(reader, &out_decoded->timing));

  IREE_RETURN_IF_ERROR(iree_hal_device_spec_reader_read_u32(
      reader, &out_decoded->executables.flags));
  for (iree_host_size_t i = 0; i < counts->executable_target_count; ++i) {
    iree_hal_executable_target_t temporary;
    iree_hal_executable_target_t* value = storage->executable_targets
                                              ? &storage->executable_targets[i]
                                              : &temporary;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_spec_decode_executable_target(reader, value));
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_device_spec_decode_sanitizer(reader, &out_decoded->sanitizer));
  for (iree_host_size_t i = 0; i < counts->facet_count; ++i) {
    iree_hal_device_spec_facet_t temporary;
    iree_hal_device_spec_facet_t* value =
        storage->facets ? &storage->facets[i] : &temporary;
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_decode_facet(reader, value));
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_device_spec_parse(
    iree_const_byte_span_t bytes, iree_allocator_t host_allocator,
    iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  if (IREE_UNLIKELY(bytes.data_length && !bytes.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec image storage is NULL");
  }

  iree_hal_device_spec_reader_t reader = {
      .bytes = bytes,
      .offset = 0,
  };
  iree_hal_device_spec_counts_t counts = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_decode_header(&reader, &counts));
  const iree_host_size_t body_offset = reader.offset;

  // Validate every declared record before allowing counts to drive native
  // allocations. This bounds memory use by bytes proven to exist in the image.
  const iree_hal_device_spec_parse_storage_t empty_storage = {0};
  iree_hal_device_spec_decoded_t validated = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_decode_body(
      &reader, &counts, &empty_storage, &validated));
  if (IREE_UNLIKELY(reader.offset != bytes.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec image has %" PRIhsz " trailing bytes",
                            bytes.data_length - reader.offset);
  }

  iree_hal_device_spec_parse_storage_t storage;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_parse_storage_initialize(
      &counts, host_allocator, &storage));
  reader.offset = body_offset;
  iree_hal_device_spec_decoded_t decoded = {0};
  iree_status_t status =
      iree_hal_device_spec_decode_body(&reader, &counts, &storage, &decoded);
  if (iree_status_is_ok(status)) {
    iree_hal_device_spec_params_t params = {
        .identity = &decoded.identity,
        .memory = &decoded.memory,
        .virtual_memory = &decoded.virtual_memory,
        .queues = &decoded.queues,
        .dispatch = &decoded.dispatch,
        .timing = &decoded.timing,
        .executables = &decoded.executables,
        .sanitizer = &decoded.sanitizer,
        .facet_count = counts.facet_count,
        .facets = storage.facets,
    };
    status = iree_hal_device_spec_create(&params, host_allocator, out_spec);
  }
  iree_allocator_free(host_allocator, storage.allocation);
  return status;
}
