// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_TEST_DATA_H_
#define IREE_VM_BYTECODE_MODULE_TEST_DATA_H_

#include <cstdint>
#include <vector>

namespace iree::vm::bytecode::testing {

// Builds the exact B0 ownership and reflection fixture.
std::vector<uint8_t> BuildOwnershipModuleImage();

// Builds the exact 17-record launch-configuration fixture plus empty and
// full-signature no-op decomposition functions.
std::vector<uint8_t> BuildLaunchConfigModuleImage();

}  // namespace iree::vm::bytecode::testing

#endif  // IREE_VM_BYTECODE_MODULE_TEST_DATA_H_
