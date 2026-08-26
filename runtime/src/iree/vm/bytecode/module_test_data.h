// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_TEST_DATA_H_
#define IREE_VM_BYTECODE_MODULE_TEST_DATA_H_

#include <cstdint>
#include <vector>

#include "iree/vm/bytecode/wire/module_format.h"

namespace iree::vm::bytecode::testing {

// Mutable view into one function in a known-good test image.
struct MutableFunctionImage {
  // Mutable function-table row.
  iree_vm_bytecode_v0_function_row_t* row;
  // First instruction byte for |row|.
  uint8_t* bytecode;
};

// Stable function ordinals in BuildBufferModuleImage.
enum BufferFunctionOrdinal : uint32_t {
  kBufferAllocateFunctionOrdinal = 0,
  kBufferLengthFunctionOrdinal = 1,
  kBufferLoadFunctionOrdinal = 2,
  // Sixteen functions in exact memory.format selector order.
  kBufferRoundtripFunctionBase = 3,
  kBufferStoreFunctionOrdinal = 19,
  kBufferSubspanFunctionOrdinal = 20,
  kBufferWrongLengthFunctionOrdinal = 21,
  kBufferStackCopyFunctionOrdinal = 22,
};

// Finds one section payload in |image| or returns null when absent.
uint8_t* FindSectionPayload(std::vector<uint8_t>* image, uint16_t section_type);

// Finds function |ordinal| in |image| or returns an empty view when absent.
MutableFunctionImage FindFunctionImage(std::vector<uint8_t>* image,
                                       uint32_t ordinal);

// Builds the exact Core 0.0 ownership and reflection fixture.
std::vector<uint8_t> BuildOwnershipModuleImage();

// Builds one typed-ref fixture covering register and local-slot ownership.
std::vector<uint8_t> BuildRefModuleImage();

// Builds one typed-ref fixture covering ABI overflow and global ownership.
std::vector<uint8_t> BuildRefStateModuleImage();

// Builds one first-class function-carrier fixture covering every family op.
std::vector<uint8_t> BuildFunctionModuleImage();

// Builds one function-global and overflow-ABI fixture.
std::vector<uint8_t> BuildFunctionStateModuleImage();

// Builds one scalar direct, indirect, and suspending call fixture.
std::vector<uint8_t> BuildCallModuleImage();

// Builds the exact 17-record launch-configuration fixture plus empty and
// full-signature no-op decomposition functions.
std::vector<uint8_t> BuildLaunchConfigModuleImage();

// Builds an exact value-overflow fixture with successful and failing exports.
std::vector<uint8_t> BuildValueOverflowModuleImage();

// Builds one scalar-state fixture containing every Core 0.0 value constant,
// selection, mutable value-global, and constant-pool operation.
std::vector<uint8_t> BuildScalarStateModuleImage();

// Builds one vm.buffer fixture covering construction, views, checked lane
// access, and an intentionally mismatched operand type.
std::vector<uint8_t> BuildBufferModuleImage();

// Builds one structurally valid HAL-page inspection fixture.
std::vector<uint8_t> BuildHALInspectionModuleImage();

// Builds one structurally valid switch-target inspection fixture.
std::vector<uint8_t> BuildSwitchInspectionModuleImage();

}  // namespace iree::vm::bytecode::testing

#endif  // IREE_VM_BYTECODE_MODULE_TEST_DATA_H_
