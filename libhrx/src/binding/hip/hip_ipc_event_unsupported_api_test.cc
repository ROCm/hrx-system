// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <unistd.h>

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

using HipIpcOpenEventHandleFn = hipError_t (*)(hipEvent_t* event,
                                               hipIpcEventHandle_t handle);
using HipEventCreateWithFlagsFn = hipError_t (*)(hipEvent_t* event,
                                                 unsigned int flags);

struct DynamicLibraryDeleter {
  void operator()(void* library) const {
    if (library) EXPECT_EQ(0, dlclose(library));
  }
};

struct IpcEventWire {
  uint32_t type;
  int32_t creator_process_id;
  uint8_t token[32];
  uint8_t reserved[24];
};
static_assert(sizeof(IpcEventWire) == sizeof(hipIpcEventHandle_t));

int32_t ForeignProcessId() {
  const int32_t current_process_id = static_cast<int32_t>(getpid());
  return current_process_id == 1 ? 2 : 1;
}

TEST(HipIpcEventUnsupportedApiTest, PreservesValidationAndOrdinaryEventPath) {
  const char* library_path = HrxLibPath();
  ASSERT_NE(nullptr, library_path);
  void* library = dlopen(library_path, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(nullptr, library) << dlerror();
  std::unique_ptr<void, DynamicLibraryDeleter> library_guard(library);
  auto ipc_open_event_handle = reinterpret_cast<HipIpcOpenEventHandleFn>(
      dlsym(library, "hipIpcOpenEventHandle"));
  ASSERT_NE(nullptr, ipc_open_event_handle) << dlerror();
  auto event_create_with_flags = reinterpret_cast<HipEventCreateWithFlagsFn>(
      dlsym(library, "hipEventCreateWithFlags"));
  ASSERT_NE(nullptr, event_create_with_flags) << dlerror();

  IpcEventWire event_wire = {};
  event_wire.type = 1;
  event_wire.creator_process_id = ForeignProcessId();
  hipIpcEventHandle_t foreign_event_handle = {};
  std::memcpy(&foreign_event_handle, &event_wire, sizeof(event_wire));

  EXPECT_EQ(hipErrorInvalidValue,
            ipc_open_event_handle(nullptr, foreign_event_handle));
  hipEvent_t imported_event = reinterpret_cast<hipEvent_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotSupported,
            ipc_open_event_handle(&imported_event, foreign_event_handle));
  EXPECT_EQ(nullptr, imported_event);

  EXPECT_EQ(hipErrorInvalidValue,
            event_create_with_flags(
                nullptr, hipEventInterprocess | hipEventDisableTiming));
  hipEvent_t created_event = reinterpret_cast<hipEvent_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            event_create_with_flags(&created_event, hipEventInterprocess));
  EXPECT_EQ(nullptr, created_event);

  created_event = reinterpret_cast<hipEvent_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotSupported,
            event_create_with_flags(
                &created_event, hipEventInterprocess | hipEventDisableTiming));
  EXPECT_EQ(nullptr, created_event);

  created_event = reinterpret_cast<hipEvent_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotInitialized,
            event_create_with_flags(&created_event, hipEventDisableTiming));
  EXPECT_EQ(nullptr, created_event);
}

}  // namespace
