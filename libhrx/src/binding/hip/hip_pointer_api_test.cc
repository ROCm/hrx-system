// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <cstdlib>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

namespace {

const char* CandidateLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return "libamdhip64.so";
#endif
}

using HipDrvPointerGetAttributesFn = hipError_t (*)(
    unsigned int attribute_count, hipPointer_attribute_t* attributes,
    void** data, const void* pointer);

TEST(HipPointerApiTest, BatchQueryRejectsInvalidPointerAndOutputSlots) {
  void* library = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
  if (!library) {
    GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();
  }
  auto get_attributes = reinterpret_cast<HipDrvPointerGetAttributesFn>(
      dlsym(library, "hipDrvPointerGetAttributes"));
  ASSERT_NE(nullptr, get_attributes);

  hipPointer_attribute_t attribute = HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
  int output = 0;
  void* outputs[] = {&output};
  EXPECT_EQ(hipErrorInvalidValue,
            get_attributes(1, &attribute, outputs, nullptr));

  int pointer_storage = 0;
  outputs[0] = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            get_attributes(1, &attribute, outputs, &pointer_storage));

  EXPECT_EQ(0, dlclose(library));
}

}  // namespace
