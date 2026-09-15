// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One cold XDNA instruction instance in caller-owned storage.

#include "experimental/xdna/prepared_command.h"

#include <string.h>

#include "experimental/xdna/executable.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/transaction.h"

struct iree_hal_amd_xdna_prepared_command_t {
  // Host allocator owning this prepared command.
  iree_allocator_t host_allocator;
  // Retained executable owning immutable source bytes and metadata.
  iree_hal_executable_t* executable;
  // Immutable initialization and first execution range.
  amdf_xdna_kernel_command_t initialization;
  // Immutable steady-state execution range.
  amdf_xdna_kernel_command_t execution;
  // Number of direct HAL buffers retained by a prepared invocation.
  iree_host_size_t retained_buffer_count;
  // Retained direct HAL buffers in executable binding order.
  iree_hal_buffer_t** retained_buffers;
};

static iree_hal_memory_access_t
iree_hal_amd_xdna_prepared_command_required_access(
    iree_hal_amd_xdna_elf_binding_access_t access) {
  iree_hal_memory_access_t required_access = IREE_HAL_MEMORY_ACCESS_NONE;
  if (iree_any_bit_set(access, IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ)) {
    required_access |= IREE_HAL_MEMORY_ACCESS_READ;
  }
  if (iree_any_bit_set(access, IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_WRITE)) {
    required_access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  }
  return required_access;
}

static iree_hal_buffer_usage_t
iree_hal_amd_xdna_prepared_command_required_usage(
    iree_hal_amd_xdna_elf_binding_access_t access) {
  iree_hal_buffer_usage_t required_usage = IREE_HAL_BUFFER_USAGE_NONE;
  if (iree_any_bit_set(access, IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ)) {
    required_usage |= IREE_HAL_BUFFER_USAGE_STORAGE_READ;
  }
  if (iree_any_bit_set(access, IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_WRITE)) {
    required_usage |= IREE_HAL_BUFFER_USAGE_STORAGE_WRITE;
  }
  return required_usage;
}

static iree_hal_memory_type_t
iree_hal_amd_xdna_prepared_command_required_memory_type(
    const iree_hal_amd_xdna_elf_binding_record_t* contract) {
  iree_hal_memory_type_t required_memory_type = IREE_HAL_MEMORY_TYPE_NONE;
  if (iree_any_bit_set(contract->usage,
                       IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE)) {
    required_memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  }
  const bool requires_host_visibility =
      contract->address_space ==
          IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST ||
      iree_any_bit_set(contract->usage,
                       IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE);
  if (requires_host_visibility) {
    required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    if (iree_any_bit_set(contract->usage,
                         IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_COHERENT)) {
      required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
    }
    if (iree_any_bit_set(contract->usage,
                         IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_CACHED)) {
      required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
    }
  }
  return required_memory_type;
}

static iree_status_t iree_hal_amd_xdna_prepared_command_validate_binding(
    const iree_hal_amd_xdna_elf_binding_record_t* contract,
    const iree_hal_amd_xdna_prepared_command_binding_t* binding) {
  if (binding->buffer_ref.buffer == NULL || binding->memory == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA prepared-command binding %u is not fully resolved",
        contract->binding_ordinal);
  }

  iree_device_size_t resource_byte_offset = 0;
  iree_device_size_t resource_byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0,
      iree_hal_buffer_byte_length(binding->buffer_ref.buffer),
      binding->buffer_ref.offset, binding->buffer_ref.length,
      &resource_byte_offset, &resource_byte_length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(binding->buffer_ref.buffer),
      iree_hal_amd_xdna_prepared_command_required_access(contract->access)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(binding->buffer_ref.buffer),
      iree_hal_amd_xdna_prepared_command_required_usage(contract->access)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding->buffer_ref.buffer),
      iree_hal_amd_xdna_prepared_command_required_memory_type(contract)));

  if (resource_byte_offset < contract->minimum_byte_offset ||
      resource_byte_offset > contract->maximum_byte_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "XDNA binding %u offset %" PRIu64 " is outside [%" PRIu64 ", %" PRIu64
        "]",
        contract->binding_ordinal, (uint64_t)resource_byte_offset,
        contract->minimum_byte_offset, contract->maximum_byte_offset);
  }
  if (resource_byte_length < contract->minimum_byte_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "XDNA binding %u length %" PRIu64 " is smaller than required %" PRIu64,
        contract->binding_ordinal, (uint64_t)resource_byte_length,
        contract->minimum_byte_length);
  }
  if ((binding->device_address & (contract->minimum_alignment - 1)) != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u device address 0x%" PRIx64
                            " does not satisfy %" PRIu64 "-byte alignment",
                            contract->binding_ordinal, binding->device_address,
                            contract->minimum_alignment);
  }
  if ((resource_byte_length != 0 &&
       binding->device_address > UINT64_MAX - (resource_byte_length - 1)) ||
      binding->memory_byte_offset > UINT64_MAX - resource_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u native range overflows",
                            contract->binding_ordinal);
  }
  return iree_ok_status();
}

// Cold layout has one combined initialization transaction and one reusable
// CONTROL transaction. This is executable instantiation, not an allocator.
static iree_status_t iree_hal_amd_xdna_prepared_command_layout(
    const iree_hal_amd_xdna_executable_entry_t* entry,
    iree_host_size_t alignment, iree_host_size_t* out_initialization_length,
    iree_host_size_t* out_execution_offset, iree_host_size_t* out_byte_length) {
  if (!iree_is_power_of_two_uint64(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "instruction alignment must be a power of two");
  }
  iree_host_size_t initialization_length = 0;
  iree_host_size_t execution_offset = 0;
  iree_host_size_t byte_length = 0;
  if (!iree_host_size_checked_add(
          entry->array.data_length,
          entry->native.control.data_length -
              IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE,
          &initialization_length) ||
      initialization_length > UINT32_MAX ||
      !iree_host_size_checked_align(initialization_length, alignment,
                                    &execution_offset) ||
      !iree_host_size_checked_add(
          execution_offset, entry->native.control.data_length, &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA instruction instance exceeds format limits");
  }
  *out_initialization_length = initialization_length;
  *out_execution_offset = execution_offset;
  *out_byte_length = byte_length;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_prepared_command_query_storage_size(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_host_size_t instruction_alignment, iree_host_size_t* out_byte_length) {
  iree_hal_amd_xdna_executable_entry_t entry;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_executable_query_entry(executable, function, &entry));
  iree_host_size_t initialization_length = 0;
  iree_host_size_t execution_offset = 0;
  return iree_hal_amd_xdna_prepared_command_layout(
      &entry, instruction_alignment, &initialization_length, &execution_offset,
      out_byte_length);
}

// Qualification has established the field and its binding. Instantiation checks
// only the externally supplied address against the preserved ELF constraints.
static iree_status_t iree_hal_amd_xdna_prepared_command_validate_relocation(
    const iree_hal_amd_xdna_aie2p_native_relocation_t* relocation,
    uint64_t base_address) {
  const uint64_t displacement = relocation->addend < 0
                                    ? UINT64_C(0) - (uint64_t)relocation->addend
                                    : (uint64_t)relocation->addend;
  if ((relocation->addend < 0 && base_address < displacement) ||
      (relocation->addend >= 0 && base_address > UINT64_MAX - displacement)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA relocated DMA address overflows");
  }
  const uint64_t address = base_address + (uint64_t)relocation->addend;
  if (address < relocation->minimum_value ||
      address > relocation->maximum_value ||
      (address & (relocation->required_alignment - 1)) != 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "XDNA DMA address violates the relocation contract");
  }
  return iree_ok_status();
}

static void iree_hal_amd_xdna_prepared_command_write_address(uint8_t* target,
                                                             uint64_t address) {
  const uint32_t low = iree_unaligned_load_le_u32(target);
  const uint32_t high = iree_unaligned_load_le_u32(target + 4);
  iree_unaligned_store_le_u32(target, (low & 3u) | (uint32_t)address);
  iree_unaligned_store_le_u32(
      target + 4, (high & UINT32_C(0xFFFF0000)) | (uint32_t)(address >> 32));
}

iree_status_t iree_hal_amd_xdna_prepared_command_create(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_host_size_t instruction_alignment,
    const amdf_xdna_kernel_command_t* storage_range, iree_byte_span_t storage,
    iree_host_size_t binding_count,
    const iree_hal_amd_xdna_prepared_command_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_prepared_command_t** out_prepared_command) {
  if (storage_range == NULL || storage_range->memory == NULL ||
      storage_range->reserved != 0 || storage.data == NULL ||
      storage_range->byte_length != storage.data_length ||
      storage_range->byte_offset > UINT64_MAX - storage_range->byte_length ||
      (binding_count != 0 && bindings == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA instruction storage or bindings are invalid");
  }
  iree_hal_amd_xdna_executable_entry_t entry;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_executable_query_entry(executable, function, &entry));
  if (binding_count != entry.binding_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA function requires %u bindings, received %" PRIhsz,
        entry.binding_count, binding_count);
  }
  iree_host_size_t initialization_length = 0;
  iree_host_size_t execution_offset = 0;
  iree_host_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_prepared_command_layout(
      &entry, instruction_alignment, &initialization_length, &execution_offset,
      &byte_length));
  if (storage.data_length < byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA instruction storage is too small");
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    iree_hal_amd_xdna_elf_binding_record_t contract;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_query_binding(
        executable, function, i, &contract));
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_prepared_command_validate_binding(
        &contract, &bindings[i]));
  }
  for (iree_host_size_t i = 0; i < entry.native.relocation_count; ++i) {
    const iree_hal_amd_xdna_aie2p_native_relocation_t* relocation =
        &entry.native.relocations[i];
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_prepared_command_validate_relocation(
        relocation, bindings[relocation->binding_ordinal].device_address));
  }

  iree_host_size_t retained_buffers_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_prepared_command_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(binding_count, iree_hal_buffer_t*,
                                iree_alignof(iree_hal_buffer_t*),
                                &retained_buffers_offset)));
  iree_hal_amd_xdna_prepared_command_t* command = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&command));
  command->host_allocator = host_allocator;
  command->executable = executable;
  iree_hal_executable_retain(executable);
  command->initialization = *storage_range;
  command->initialization.byte_length = initialization_length;
  command->execution = *storage_range;
  command->execution.byte_offset += execution_offset;
  command->execution.byte_length = entry.native.control.data_length;
  command->retained_buffers =
      (iree_hal_buffer_t**)((uint8_t*)command + retained_buffers_offset);
  command->retained_buffer_count = binding_count;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    command->retained_buffers[i] = bindings[i].buffer_ref.buffer;
    iree_hal_buffer_retain(command->retained_buffers[i]);
  }

  const iree_host_size_t header_size =
      IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE;
  const iree_host_size_t control_offset = entry.array.data_length - header_size;
  memcpy(storage.data, entry.array.data, entry.array.data_length);
  memcpy(storage.data + entry.array.data_length,
         entry.native.control.data + header_size,
         entry.native.control.data_length - header_size);
  iree_unaligned_store_le_u32(
      storage.data + 8,
      iree_unaligned_load_le_u32(entry.array.data + 8) +
          iree_unaligned_load_le_u32(entry.native.control.data + 8));
  iree_unaligned_store_le_u32(storage.data + 12, initialization_length);
  memcpy(storage.data + execution_offset, entry.native.control.data,
         entry.native.control.data_length);
  for (iree_host_size_t i = 0; i < entry.native.relocation_count; ++i) {
    const iree_hal_amd_xdna_aie2p_native_relocation_t* relocation =
        &entry.native.relocations[i];
    const uint64_t address =
        bindings[relocation->binding_ordinal].device_address +
        (uint64_t)relocation->addend;
    iree_hal_amd_xdna_prepared_command_write_address(
        storage.data + control_offset + relocation->byte_offset, address);
    iree_hal_amd_xdna_prepared_command_write_address(
        storage.data + execution_offset + relocation->byte_offset, address);
  }
  *out_prepared_command = command;
  return iree_ok_status();
}

void iree_hal_amd_xdna_prepared_command_destroy(
    iree_hal_amd_xdna_prepared_command_t* prepared_command) {
  if (prepared_command == NULL) return;
  for (iree_host_size_t i = prepared_command->retained_buffer_count; i > 0;
       --i) {
    iree_hal_buffer_release(prepared_command->retained_buffers[i - 1]);
  }
  iree_hal_executable_release(prepared_command->executable);
  iree_allocator_free(prepared_command->host_allocator, prepared_command);
}

const amdf_xdna_kernel_command_t*
iree_hal_amd_xdna_prepared_command_initialization(
    const iree_hal_amd_xdna_prepared_command_t* prepared_command) {
  return &prepared_command->initialization;
}

const amdf_xdna_kernel_command_t* iree_hal_amd_xdna_prepared_command_execution(
    const iree_hal_amd_xdna_prepared_command_t* prepared_command) {
  return &prepared_command->execution;
}
