// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

namespace {

const char* HrxLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return nullptr;
#endif
}

using HipIpcGetMemHandleFn = hipError_t (*)(hipIpcMemHandle_t* handle,
                                            void* device_ptr);
using HipIpcOpenMemHandleFn = hipError_t (*)(void** device_ptr,
                                             hipIpcMemHandle_t handle,
                                             unsigned int flags);
using HipIpcCloseMemHandleFn = hipError_t (*)(void* device_ptr);

struct DynamicLibraryDeleter {
  void operator()(void* library) const {
    if (library) EXPECT_EQ(0, dlclose(library));
  }
};

struct IpcMemoryWire {
  uint8_t token[32];
  uint64_t allocation_size;
  uint64_t byte_offset;
  int32_t creator_process_id;
  int32_t device_ordinal;
  uint8_t reserved[8];
};
static_assert(sizeof(IpcMemoryWire) == sizeof(hipIpcMemHandle_t));

int32_t ForeignProcessId() {
  const int32_t current_process_id = static_cast<int32_t>(getpid());
  return current_process_id == 1 ? 2 : 1;
}

bool IsZeroed(const void* data, size_t data_length) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < data_length; ++i) {
    if (bytes[i] != 0) return false;
  }
  return true;
}

TEST(HipIpcMemoryUnsupportedApiTest,
     ValidatesArgumentsBeforeReportingUnsupported) {
  const char* library_path = HrxLibPath();
  ASSERT_NE(nullptr, library_path);
  void* library = dlopen(library_path, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(nullptr, library) << dlerror();
  std::unique_ptr<void, DynamicLibraryDeleter> library_guard(library);
  auto ipc_get_mem_handle = reinterpret_cast<HipIpcGetMemHandleFn>(
      dlsym(library, "hipIpcGetMemHandle"));
  ASSERT_NE(nullptr, ipc_get_mem_handle) << dlerror();
  auto ipc_open_mem_handle = reinterpret_cast<HipIpcOpenMemHandleFn>(
      dlsym(library, "hipIpcOpenMemHandle"));
  ASSERT_NE(nullptr, ipc_open_mem_handle) << dlerror();
  auto ipc_close_mem_handle = reinterpret_cast<HipIpcCloseMemHandleFn>(
      dlsym(library, "hipIpcCloseMemHandle"));
  ASSERT_NE(nullptr, ipc_close_mem_handle) << dlerror();
  int stack_sentinel = 0;
  EXPECT_EQ(hipErrorInvalidValue, ipc_get_mem_handle(nullptr, &stack_sentinel));

  hipIpcMemHandle_t handle;
  std::memset(&handle, 0xA5, sizeof(handle));
  EXPECT_EQ(hipErrorInvalidValue, ipc_get_mem_handle(&handle, nullptr));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));

  std::memset(&handle, 0xA5, sizeof(handle));
  EXPECT_EQ(hipErrorNotSupported, ipc_get_mem_handle(&handle, &stack_sentinel));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));

  IpcMemoryWire memory_wire = {};
  memory_wire.allocation_size = 1;
  memory_wire.creator_process_id = ForeignProcessId();
  hipIpcMemHandle_t foreign_memory_handle = {};
  std::memcpy(&foreign_memory_handle, &memory_wire, sizeof(memory_wire));

  EXPECT_EQ(hipErrorInvalidValue,
            ipc_open_mem_handle(nullptr, foreign_memory_handle,
                                hipIpcMemLazyEnablePeerAccess));
  void* imported_ptr = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            ipc_open_mem_handle(&imported_ptr, foreign_memory_handle,
                                /*flags=*/0));
  EXPECT_EQ(nullptr, imported_ptr);

  imported_ptr = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotSupported,
            ipc_open_mem_handle(&imported_ptr, foreign_memory_handle,
                                hipIpcMemLazyEnablePeerAccess));
  EXPECT_EQ(nullptr, imported_ptr);
  EXPECT_EQ(hipErrorInvalidValue, ipc_close_mem_handle(nullptr));
  EXPECT_EQ(hipErrorNotSupported,
            ipc_close_mem_handle(reinterpret_cast<void*>(uintptr_t{1})));
}

}  // namespace
