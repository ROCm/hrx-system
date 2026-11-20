// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SYMBOL_KERNEL_LAUNCH_H_
#define LOOM_IMPORT_CXX_SYMBOL_KERNEL_LAUNCH_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace loom::cxx_import {

// Host launch facts of one [[loom::kernel_ref]] specialization, as the
// importer knows them: the host object layout of the kernel's arguments and
// how they flatten into the imported kernel's argument list. The native
// argument layout (kernel argument offsets and the argument segment size) is
// not here: the target lowering that compiles the kernel decides it, and the
// compiled artifact reports it. A driver combines both to pack each kernel's
// arguments from its host parameter objects.

// One kernel argument copied from a host parameter object.
struct KernelLaunchLeaf {
  // Kernel parameter whose host object holds the leaf.
  uint32_t parameter;
  // 0: by-value bytes, or 1: a device pointer passed as a buffer binding.
  uint32_t kind;
  // Byte offset of the leaf in the host parameter object.
  uint64_t source_offset;
  // Byte size of the leaf in the host parameter object.
  uint64_t size;
};

struct KernelLaunch {
  // Itanium symbol of the kernel_ref specialization the host references.
  std::string symbol;
  // Whether the kernel was imported.
  bool available = false;
  // Export name of the kernel in the compiled executable.
  std::string export_name;
  // Exact workgroup size annotation, or zeros when the launch configuration
  // computes it.
  std::array<uint32_t, 3> workgroup_size = {0, 0, 0};
  // Number of C++ kernel parameters.
  uint32_t parameter_count = 0;
  // One leaf per argument of the imported kernel, in argument order: leaf i
  // is the kernel's argument i. A host subobject that flattens to several
  // arguments yields consecutive leaves with the same parameter, source offset
  // and size; a driver that packs each argument from host bytes cannot pack
  // them.
  std::vector<KernelLaunchLeaf> leaves;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SYMBOL_KERNEL_LAUNCH_H_
