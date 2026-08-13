// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_RECORD_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_RECORD_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opcode identifying one compact PM4 command record.
typedef uint16_t iree_hal_amdgpu_pm4_command_record_opcode_t;
typedef enum iree_hal_amdgpu_pm4_command_record_opcode_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH = 1,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT = 2,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE = 3,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW = 4,
} iree_hal_amdgpu_pm4_command_record_opcode_e;

// Prefix shared by every compact PM4 command record.
typedef struct iree_hal_amdgpu_pm4_command_record_header_t {
  // Total byte length of this record including inline payload.
  uint32_t length;
  // iree_hal_amdgpu_pm4_command_record_opcode_t value.
  iree_hal_amdgpu_pm4_command_record_opcode_t opcode;
  // Reserved padding; must be zero.
  uint16_t reserved0;
} iree_hal_amdgpu_pm4_command_record_header_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_RECORD_H_
