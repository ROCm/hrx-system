// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/result_buffer.h"

#include "libamdf/src/xdna/umd/mcdm/context.h"
#include "libamdf/src/xdna/umd/mcdm/kernel_execution.h"
#include "libamdf/src/xdna/umd/mcdm/memory.h"
#include "libamdf/src/xdna/umd/mcdm/memory_profile.h"

amdf_status_t amdf_xdna_umd_context_query_result_buffer_info(
    amdf_xdna_umd_context_t* context,
    amdf_xdna_result_buffer_info_t* out_info) {
  if (context->adapter_info.protocol != AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint64_t prefix = amdf_windows_xdna_instruction_prefix_size(
      context->device->profile->info->instruction.address_alignment);
  *out_info = (amdf_xdna_result_buffer_info_t){
      .binding_mode = AMDF_XDNA_RESULT_BUFFER_BINDING_RANGE,
      .minimum_byte_length = AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT,
      .maximum_byte_length = AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE - prefix,
      .byte_offset_alignment = AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT,
      .byte_length_granularity = AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_context_result_buffer(
    amdf_xdna_umd_context_t* context,
    amdf_xdna_result_buffer_operation_t operation,
    amdf_xdna_umd_memory_t* memory, uint64_t byte_offset,
    uint64_t byte_length) {
  const uint64_t prefix = amdf_windows_xdna_instruction_prefix_size(
      context->device->profile->info->instruction.address_alignment);
  return amdf_windows_xdna_kernel_execution_result_buffer(
      context->kernel_execution, operation,
      amdf_windows_xdna_memory_private_allocation(memory), prefix + byte_offset,
      byte_length);
}
