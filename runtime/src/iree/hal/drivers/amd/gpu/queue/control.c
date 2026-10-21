// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/control.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

uint32_t iree_hal_amd_gpu_control_requirement(
    iree_hal_amd_gpu_control_t command) {
  if (command.kind == IREE_HAL_AMD_GPU_CONTROL_TIMESTAMP) {
    return 0;
  }
  const uint32_t operation = command.kind == IREE_HAL_AMD_GPU_CONTROL_WAIT
                                 ? IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT
                             : command.kind == IREE_HAL_AMD_GPU_CONTROL_STORE
                                 ? IREE_HAL_ATOMIC_OPERATION_FLAG_STORE
                                 : IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD
                                       << command.operation;
  const uint32_t shift =
      (command.width == IREE_HAL_ATOMIC_WIDTH_64 ? 8 : 0) +
      (command.flags & IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE ? 16 : 0);
  return operation << shift;
}

iree_status_t iree_hal_amd_gpu_control_validate_target(
    const iree_hal_queue_family_t* family, const iree_hal_buffer_t* buffer,
    uint32_t required, uint32_t incompatible) {
  const uint32_t available =
      buffer->memory.contract->scopes[family->memory.queue_scope_id]
          .atomic_operations;
  if (((required | incompatible) & ~available) != 0) {
    return iree_make_status(
        (required & ~available) ? IREE_STATUS_PERMISSION_DENIED
                                : IREE_STATUS_INCOMPATIBLE,
        "GPU target lacks prepared queue atomic capabilities");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_control_resolve(
    const iree_hal_amd_gpu_buffer_reference_t* target,
    iree_hal_amd_gpu_control_t command, uint64_t* out_address) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_buffer_reference_resolve(target, out_address));
  if (*out_address & (command.width / 8u - 1)) {
    return iree_make_status(
        command.target_error_mode ==
                IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE
            ? IREE_STATUS_INCOMPATIBLE
            : IREE_STATUS_INVALID_ARGUMENT,
        "GPU control address must be %u-byte aligned", command.width / 8u);
  }
  return iree_ok_status();
}

uint32_t iree_hal_amd_gpu_control_word_count(
    iree_hal_amd_gpu_control_t command) {
  if (command.kind == IREE_HAL_AMD_GPU_CONTROL_WAIT) {
    return command.width == IREE_HAL_ATOMIC_WIDTH_32 ? 7 : 9;
  }
  return command.kind == IREE_HAL_AMD_GPU_CONTROL_RMW ? 9 : 6;
}

uint32_t iree_hal_amd_gpu_control_address_word(
    iree_hal_amd_gpu_control_t command) {
  return command.kind == IREE_HAL_AMD_GPU_CONTROL_TIMESTAMP ||
                 command.kind == IREE_HAL_AMD_GPU_CONTROL_STORE
             ? 4
             : 2;
}

uint32_t* iree_hal_amd_gpu_control_emit(uint32_t* words, uint64_t address,
                                        iree_hal_amd_gpu_control_t command) {
  switch (command.kind) {
    case IREE_HAL_AMD_GPU_CONTROL_TIMESTAMP:
      return iree_hal_amd_gpu_pm4_emit_clock(words, address);
    case IREE_HAL_AMD_GPU_CONTROL_WAIT: {
      const iree_hal_amd_gpu_pm4_comparison_t comparison =
          (iree_hal_amd_gpu_pm4_comparison_t)(IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL +
                                              command.operation);
      return command.width == IREE_HAL_ATOMIC_WIDTH_32
                 ? iree_hal_amd_gpu_pm4_emit_wait32(
                       words, address, (uint32_t)command.value,
                       (uint32_t)command.mask, comparison)
                 : iree_hal_amd_gpu_pm4_emit_wait64(
                       words, address, command.value, command.mask, comparison);
    }
    case IREE_HAL_AMD_GPU_CONTROL_STORE:
      return iree_hal_amd_gpu_pm4_emit_store(
          words, address, command.value,
          command.width == IREE_HAL_ATOMIC_WIDTH_64);
    default: {
      // SUBTRACT is modular ADD of the negated operand, including INT_MIN.
      // This preserves PCIe ADD support without requiring raw PCIe SUB.
      static const uint8_t operations[] = {0x0f, 0x0f, 0x15, 0x16, 0x17};
      const uint32_t opcode =
          operations[command.operation] +
          (command.width == IREE_HAL_ATOMIC_WIDTH_64 ? 0x20 : 0);
      const uint64_t operand =
          command.operation == IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT
              ? UINT64_C(0) - command.value
              : command.value;
      return iree_hal_amd_gpu_pm4_emit_atomic(words, address, operand, opcode);
    }
  }
}
