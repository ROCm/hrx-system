// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/legacy_submission.h"

#include <stddef.h>
#include <string.h>

enum {
  AMDF_WINDOWS_XDNA_LEGACY_SUBMISSION_COMMAND_COPY_SIZE = 512,
};

static void amdf_windows_xdna_legacy_write_u32(uint8_t* bytes, size_t offset,
                                               uint32_t value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static void amdf_windows_xdna_legacy_write_u64(uint8_t* bytes, size_t offset,
                                               uint64_t value) {
  memcpy(bytes + offset, &value, sizeof(value));
}

static void amdf_windows_xdna_legacy_submission_initialize(
    uint32_t byte_length,
    amdf_windows_xdna_legacy_submission_t* out_submission) {
  memset(out_submission, 0, sizeof(*out_submission));
  out_submission->byte_length = byte_length;
}

void amdf_windows_xdna_legacy_submission_build_aperture(
    uint32_t header_byte_length,
    const amdf_windows_xdna_private_allocation_t* instruction_allocation,
    amdf_windows_xdna_legacy_submission_t* out_submission) {
  amdf_windows_xdna_legacy_submission_initialize(header_byte_length,
                                                 out_submission);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x00, 2);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x08,
                                     instruction_allocation->allocation);
  amdf_windows_xdna_legacy_write_u64(
      out_submission->bytes, 0x10,
      instruction_allocation->descriptor.allocation_byte_length);
}

void amdf_windows_xdna_legacy_submission_build_context_initialize(
    uint32_t header_byte_length,
    const amdf_windows_xdna_private_allocation_t* command_allocation,
    amdf_windows_xdna_legacy_submission_t* out_submission) {
  amdf_windows_xdna_legacy_submission_initialize(
      header_byte_length +
          AMDF_WINDOWS_XDNA_LEGACY_SUBMISSION_COMMAND_COPY_SIZE + 8,
      out_submission);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x00, 5);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x28,
                                     command_allocation->allocation);
  amdf_windows_xdna_legacy_write_u32(out_submission->bytes, 0x30, 0);
  amdf_windows_xdna_legacy_write_u32(out_submission->bytes, 0x34, 8);
  amdf_windows_xdna_legacy_write_u64(
      out_submission->bytes, 0x38,
      (uint64_t)(uintptr_t)command_allocation->host_pointer);
  memcpy(out_submission->bytes + header_byte_length,
         command_allocation->host_pointer,
         AMDF_WINDOWS_XDNA_LEGACY_SUBMISSION_COMMAND_COPY_SIZE);
}

void amdf_windows_xdna_legacy_submission_build_accounting(
    uint32_t header_byte_length,
    const amdf_windows_xdna_private_allocation_t* instruction_allocation,
    uint64_t live_byte_length,
    amdf_windows_xdna_legacy_submission_t* out_submission) {
  amdf_windows_xdna_legacy_submission_initialize(header_byte_length,
                                                 out_submission);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x00, 9);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x08,
                                     instruction_allocation->allocation);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x10,
                                     live_byte_length);
}

void amdf_windows_xdna_legacy_submission_build_execute(
    uint32_t header_byte_length,
    const amdf_windows_xdna_private_allocation_t* execution_allocation,
    const amdf_windows_xdna_private_allocation_t* command_allocation,
    const amdf_xdna_transaction_interpreter_packet_t* packet,
    amdf_windows_xdna_legacy_submission_t* out_submission) {
  amdf_windows_xdna_legacy_submission_initialize(
      header_byte_length +
          AMDF_WINDOWS_XDNA_LEGACY_SUBMISSION_COMMAND_COPY_SIZE,
      out_submission);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x00, 3);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x08,
                                     execution_allocation->allocation);
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x10,
                                     sizeof(*packet));
  amdf_windows_xdna_legacy_write_u64(out_submission->bytes, 0x28,
                                     command_allocation->allocation);
  amdf_windows_xdna_legacy_write_u32(out_submission->bytes, 0x30, 8);
  amdf_windows_xdna_legacy_write_u32(out_submission->bytes, 0x34, 8);
  amdf_windows_xdna_legacy_write_u64(
      out_submission->bytes, 0x38,
      (uint64_t)(uintptr_t)command_allocation->host_pointer + 8);
  memcpy(out_submission->bytes + header_byte_length, packet, sizeof(*packet));
}

amdf_status_t amdf_windows_xdna_legacy_submission_query_initialize_result(
    const amdf_windows_xdna_private_allocation_t* command_allocation) {
  const uint32_t result =
      *(const volatile uint32_t*)command_allocation->host_pointer;
  if (result == 1) return AMDF_STATUS_OK;
  // Native initialization reports only a Boolean, not a firmware error code.
  return amdf_make_api_status(result == 0 ? AMDF_STATUS_CODE_DEVICE_LOST
                                          : AMDF_STATUS_CODE_INTERNAL);
}

amdf_status_t amdf_windows_xdna_legacy_submission_query_execute_result(
    const amdf_windows_xdna_private_allocation_t* command_allocation) {
  const volatile uint32_t* response =
      (const volatile uint32_t*)((const uint8_t*)
                                     command_allocation->host_pointer +
                                 8);
  const uint32_t state = *response & 0xf;
  if (state == 0) return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  return state == AMDF_XDNA_TRANSACTION_INTERPRETER_STATE_COMPLETED
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, state);
}
