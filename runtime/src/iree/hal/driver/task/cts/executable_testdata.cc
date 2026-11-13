// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/driver/task/executable/elf/testdata/compat_data.h"

namespace iree::hal::cts {

#if defined(IREE_ARCH_ARM_32) || defined(IREE_ARCH_ARM_64) ||     \
    defined(IREE_ARCH_RISCV_32) || defined(IREE_ARCH_RISCV_64) || \
    defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)

static iree_const_byte_span_t GetTaskExecutableData(
    iree_string_view_t file_name) {
  if (!iree_string_view_equal(file_name,
                              IREE_SV("command_buffer_dispatch_test.bin")) &&
      !iree_string_view_equal(file_name, IREE_SV("elementwise_mul.bin"))) {
    return iree_const_byte_span_empty();
  }

  const iree_string_view_t artifact_name =
      IREE_SV("elementwise_mul_" IREE_ARCH ".so");
  const iree_file_toc_t* toc = elementwise_mul_compatibility_create();
  for (size_t i = 0; i < elementwise_mul_compatibility_size(); ++i) {
    if (iree_string_view_equal(iree_make_cstring_view(toc[i].name),
                               artifact_name)) {
      return iree_make_const_byte_span(
          reinterpret_cast<const uint8_t*>(toc[i].data), toc[i].size);
    }
  }
  return iree_const_byte_span_empty();
}

static bool task_native_executable_registered_ =
    (CtsRegistry::RegisterExecutableTarget(
         "task",
         {"native_" IREE_ARCH, "cpu", IREE_ARCH, GetTaskExecutableData}),
     true);

#endif  // supported executable fixture architecture

}  // namespace iree::hal::cts
