// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_JIT_MODULE_H_
#define LOOM_SERVE_TESTING_JIT_MODULE_H_

#include "loom/serve/runtime/program/module.h"
#include "loom/serve/testing/jit.h"

namespace loom_serve::testing {

// Extends source JIT execution with a VM program bound to the prepared stages.
class JitModuleTest : public JitTest {
 protected:
  iree_status_t PrepareVm();
  void ReleaseProgram() override;

  // VM reference type registry.
  iree_vm_environment_t* environment_ = nullptr;
  // HAL types borrowed from the environment.
  iree_hal_module_types_t types_ = {};
  // Runner module retaining the first variant.
  iree_vm_module_t* native_ = nullptr;
  // JIT bytecode module owning its image.
  iree_vm_module_t* bytecode_ = nullptr;
  // One linked model program.
  iree_vm_program_t* program_ = nullptr;
  // Aligned invocation backing.
  alignas(iree_max_align_t) std::array<uint8_t, 16384> invocation_storage_ = {};
  // Invocation borrowing its backing.
  iree_vm_invocation_t* invocation_ = nullptr;
  // One shared model process.
  iree_vm_process_t* process_ = nullptr;
  // Resolved control entry point.
  iree_vm_function_t step_ = {};
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_JIT_MODULE_H_
