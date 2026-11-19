// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/submission.h"

#include <stddef.h>
#include <string.h>

#include "libamdf/src/xdna/elf_packet.h"
#include "libamdf/src/xdna/transaction_interpreter.h"

enum {
  AMDF_WINDOWS_XDNA_SUBMISSION_COMMAND_COPY_SIZE = 512,
  // One ERT chain root followed by one 64-bit child reference per command.
  AMDF_WINDOWS_XDNA_COMMAND_LIST_HEADER_SIZE = 28,
  // Firmware header plus the bootstrap interpreter's 15 argument words.
  AMDF_WINDOWS_XDNA_INTERPRETER_ENTRY_SIZE = 112,
  // Firmware header plus the ELF transaction selector.
  AMDF_WINDOWS_XDNA_ELF_ENTRY_SIZE = 56,
};

uint32_t amdf_windows_xdna_submission_header_size(
    amdf_windows_xdna_protocol_t protocol) {
  if (protocol == AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT) {
    return 120;
  }
  if (protocol == AMDF_WINDOWS_XDNA_PROTOCOL_METADATA) {
    return 104;
  }
  return 88;
}

static void amdf_windows_xdna_submission_write_u32(uint8_t* bytes,
                                                   size_t offset,
                                                   uint32_t value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static void amdf_windows_xdna_submission_write_u64(uint8_t* bytes,
                                                   size_t offset,
                                                   uint64_t value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static void amdf_windows_xdna_submission_initialize(
    uint32_t byte_length, amdf_windows_xdna_submission_t* out_submission) {
  memset(out_submission, 0, sizeof(*out_submission));
  out_submission->byte_length = byte_length;
}

void amdf_windows_xdna_submission_build_aperture(
    amdf_windows_xdna_protocol_t protocol,
    const amdf_windows_xdna_private_allocation_t* instruction_allocation,
    amdf_windows_xdna_submission_t* out_submission) {
  amdf_windows_xdna_submission_initialize(
      amdf_windows_xdna_submission_header_size(protocol), out_submission);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x00, 2);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x08,
                                         instruction_allocation->allocation);
  amdf_windows_xdna_submission_write_u64(
      out_submission->bytes, 0x10,
      instruction_allocation->descriptor.allocation_byte_length);
}

static void amdf_windows_xdna_submission_write_response(
    amdf_windows_xdna_protocol_t protocol,
    const amdf_windows_xdna_private_allocation_t* command_allocation,
    uint32_t byte_offset, amdf_windows_xdna_submission_t* submission) {
  amdf_windows_xdna_submission_write_u64(submission->bytes, 0x28,
                                         command_allocation->allocation);
  const bool direct = protocol == AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT;
  if (direct) {
    amdf_windows_xdna_submission_write_u64(submission->bytes, 0x30,
                                           command_allocation->device_address);
  }
  const size_t response_byte_offset = direct ? 0x38 : 0x30;
  amdf_windows_xdna_submission_write_u32(submission->bytes,
                                         response_byte_offset, byte_offset);
  amdf_windows_xdna_submission_write_u32(submission->bytes,
                                         response_byte_offset + 4, 8);
  amdf_windows_xdna_submission_write_u64(
      submission->bytes, response_byte_offset + 8,
      (uint64_t)(uintptr_t)command_allocation->host_pointer + byte_offset);
}

void amdf_windows_xdna_submission_build_context_initialize(
    amdf_windows_xdna_protocol_t protocol,
    const amdf_windows_xdna_private_allocation_t* command_allocation,
    amdf_windows_xdna_submission_t* out_submission) {
  amdf_windows_xdna_submission_initialize(
      amdf_windows_xdna_submission_header_size(protocol) +
          AMDF_WINDOWS_XDNA_SUBMISSION_COMMAND_COPY_SIZE + 8,
      out_submission);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x00, 5);
  amdf_windows_xdna_submission_write_response(protocol, command_allocation, 0,
                                              out_submission);
  memcpy(out_submission->bytes +
             amdf_windows_xdna_submission_header_size(protocol),
         command_allocation->host_pointer,
         AMDF_WINDOWS_XDNA_SUBMISSION_COMMAND_COPY_SIZE);
}

void amdf_windows_xdna_submission_build_accounting(
    amdf_windows_xdna_protocol_t protocol,
    const amdf_windows_xdna_private_allocation_t* instruction_allocation,
    uint64_t live_byte_length, amdf_windows_xdna_submission_t* out_submission) {
  amdf_windows_xdna_submission_initialize(
      amdf_windows_xdna_submission_header_size(protocol), out_submission);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x00, 9);
  // Metadata admission accounts against the instruction allocation. Direct
  // admission accounts against the context with no allocation handle.
  if (protocol != AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT) {
    amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x08,
                                           instruction_allocation->allocation);
  }
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x10,
                                         live_byte_length);
}

uint32_t amdf_windows_xdna_submission_build_execute(
    amdf_windows_xdna_protocol_t protocol, bool uses_elf_packets,
    const amdf_windows_xdna_private_allocation_t* execution_allocation,
    const amdf_windows_xdna_private_allocation_t* response_allocation,
    uint32_t response_byte_offset,
    const amdf_xdna_umd_kernel_command_t* command, void* packet_pointer,
    amdf_windows_xdna_submission_t* out_submission) {
  uint32_t packet_byte_length;
  if (uses_elf_packets) {
    amdf_xdna_elf_packet_build(command->instruction_address,
                               command->instruction_byte_length,
                               packet_pointer);
    packet_byte_length = sizeof(amdf_xdna_elf_packet_t);
  } else {
    amdf_xdna_transaction_interpreter_packet_build(
        command->instruction_address, command->instruction_byte_length,
        packet_pointer);
    packet_byte_length = sizeof(amdf_xdna_transaction_interpreter_packet_t);
  }
  amdf_windows_xdna_submission_initialize(
      amdf_windows_xdna_submission_header_size(protocol) +
          AMDF_WINDOWS_XDNA_SUBMISSION_COMMAND_COPY_SIZE,
      out_submission);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x00, 3);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x08,
                                         execution_allocation->allocation);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x10,
                                         packet_byte_length);
  amdf_windows_xdna_submission_write_response(
      protocol, response_allocation, response_byte_offset, out_submission);
  memcpy(out_submission->bytes +
             amdf_windows_xdna_submission_header_size(protocol),
         packet_pointer, packet_byte_length);
  return packet_byte_length;
}

void amdf_windows_xdna_submission_build_result_buffer(
    uint32_t opcode, const amdf_windows_xdna_private_allocation_t* allocation,
    uint64_t native_byte_offset, uint64_t byte_length,
    amdf_windows_xdna_submission_t* out_submission) {
  amdf_windows_xdna_submission_initialize(120, out_submission);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0, opcode);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 8,
                                         allocation->allocation);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 16,
                                         byte_length);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 24,
                                         native_byte_offset);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 32,
                                         opcode == 1 ? 1 : 0);
}

static uint32_t amdf_windows_xdna_list_entry_size(bool uses_elf_packets) {
  return uses_elf_packets ? AMDF_WINDOWS_XDNA_ELF_ENTRY_SIZE
                          : AMDF_WINDOWS_XDNA_INTERPRETER_ENTRY_SIZE;
}

uint32_t amdf_windows_xdna_submission_list_capacity(bool uses_elf_packets) {
  const uint32_t list_capacity =
      AMDF_WINDOWS_XDNA_COMMAND_LIST_BYTE_LENGTH /
      amdf_windows_xdna_list_entry_size(uses_elf_packets);
  const uint32_t root_capacity =
      (AMDF_WINDOWS_XDNA_SUBMISSION_COMMAND_COPY_SIZE -
       AMDF_WINDOWS_XDNA_COMMAND_LIST_HEADER_SIZE) /
      sizeof(uint64_t);
  return list_capacity < root_capacity ? list_capacity : root_capacity;
}

uint32_t amdf_windows_xdna_submission_build_execute_list(
    bool uses_elf_packets,
    const amdf_windows_xdna_private_allocation_t* execution_allocation,
    const amdf_windows_xdna_private_allocation_t* response_allocation,
    uint32_t response_byte_offset, uint64_t list_address,
    uint32_t command_count, const amdf_xdna_umd_kernel_command_t* commands,
    void* list_pointer, void* packet_pointer, uint32_t* out_list_byte_length,
    amdf_windows_xdna_submission_t* out_submission) {
  const uint32_t entry_size =
      amdf_windows_xdna_list_entry_size(uses_elf_packets);
  const uint32_t list_byte_length = command_count * entry_size;
  uint8_t* entry = list_pointer;
  memset(entry, 0, list_byte_length);
  if (uses_elf_packets) {
    // Firmware cmd_chain_slot_npu, packed on four-byte boundaries. One
    // transaction argument follows the empty save, restore, and property
    // fields.
    for (uint32_t i = 0; i < command_count; ++i, entry += entry_size) {
      amdf_windows_xdna_submission_write_u32(entry, 0,
                                             4);  // EXEC_NPU_TYPE_ELF.
      amdf_windows_xdna_submission_write_u64(entry, 4,
                                             commands[i].instruction_address);
      amdf_windows_xdna_submission_write_u32(
          entry, 0x1C, commands[i].instruction_byte_length);
      amdf_windows_xdna_submission_write_u32(entry, 0x30, 1);
      amdf_windows_xdna_submission_write_u32(entry, 0x34,
                                             3);  // Execute transaction.
    }
  } else {
    // CU zero is the context's bootstrap interpreter. Its 60 argument bytes
    // follow the firmware's 52-byte START_CU header.
    for (uint32_t i = 0; i < command_count; ++i, entry += entry_size) {
      amdf_windows_xdna_submission_write_u32(entry, 0, 1);
      amdf_windows_xdna_submission_write_u32(entry, 0x30, 15);
      amdf_xdna_transaction_interpreter_packet_t packet;
      amdf_xdna_transaction_interpreter_packet_build(
          commands[i].instruction_address, commands[i].instruction_byte_length,
          &packet);
      memcpy(entry + 0x34, packet.bytes + 8, 60);
    }
  }

  const uint32_t packet_byte_length =
      AMDF_WINDOWS_XDNA_COMMAND_LIST_HEADER_SIZE +
      sizeof(uint64_t) * command_count;
  uint8_t* packet = packet_pointer;
  memset(packet, 0, AMDF_WINDOWS_XDNA_SUBMISSION_COMMAND_COPY_SIZE);
  amdf_windows_xdna_submission_write_u32(
      packet, 0,
      1u | (AMDF_WINDOWS_XDNA_COMMAND_LIST_OPCODE << 23) |
          ((packet_byte_length / 4 - 1) << 12));
  amdf_windows_xdna_submission_write_u32(packet, 4, command_count);

  amdf_windows_xdna_submission_initialize(
      120 + AMDF_WINDOWS_XDNA_SUBMISSION_COMMAND_COPY_SIZE, out_submission);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x00, 6);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x08,
                                         execution_allocation->allocation);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x10,
                                         packet_byte_length);
  amdf_windows_xdna_submission_write_response(
      AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT, response_allocation,
      response_byte_offset, out_submission);
  amdf_windows_xdna_submission_write_u64(out_submission->bytes, 0x58,
                                         list_address);
  amdf_windows_xdna_submission_write_u32(out_submission->bytes, 0x60,
                                         list_byte_length);
  amdf_windows_xdna_submission_write_u32(out_submission->bytes, 0x64,
                                         command_count);
  // The miniport selects the child decoder once for the complete list.
  amdf_windows_xdna_submission_write_u32(
      out_submission->bytes, 0x68,
      uses_elf_packets ? AMDF_XDNA_ELF_PACKET_OPCODE : 0);

  memcpy(out_submission->bytes + 120, packet,
         AMDF_WINDOWS_XDNA_SUBMISSION_COMMAND_COPY_SIZE);
  *out_list_byte_length = list_byte_length;
  return packet_byte_length;
}

amdf_status_t amdf_windows_xdna_submission_query_initialize_result(
    const amdf_windows_xdna_private_allocation_t* command_allocation) {
  const uint32_t result =
      *(const volatile uint32_t*)command_allocation->host_pointer;
  if (result == 1) {
    return AMDF_STATUS_OK;
  }
  // Native initialization reports only a Boolean, not a firmware error code.
  return amdf_make_api_status(result == 0 ? AMDF_STATUS_CODE_DEVICE_LOST
                                          : AMDF_STATUS_CODE_INTERNAL);
}

amdf_status_t amdf_windows_xdna_submission_query_execute_result(
    const amdf_windows_xdna_private_allocation_t* response_allocation,
    uint32_t response_byte_offset) {
  const volatile uint32_t* response =
      (const volatile uint32_t*)((const uint8_t*)
                                     response_allocation->host_pointer +
                                 response_byte_offset);
  const uint32_t state = *response & 0xf;
  if (state == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  return state == AMDF_XDNA_TRANSACTION_INTERPRETER_STATE_COMPLETED
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, state);
}
