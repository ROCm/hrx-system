// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/temp_file.h"
#include "passthrough/hip_function_table.h"

extern "C" hip_function_table_t* hip_interceptor_init(hip_function_table_t*);
extern "C" void hip_interceptor_shutdown(void);

namespace {

TEST(HipBufferTracerTest, LaunchDumpsPreserveNamesAndResults) {
  const iree::testing::TempFilePath trace_path("hip-buffer-tracer", ".log");
  ASSERT_TRUE(trace_path);
  ASSERT_EQ(setenv("HIP_TRACE_FILE", trace_path.path().c_str(), 1), 0);
  ASSERT_EQ(setenv("HIP_TRACE_LEVEL", "3", 1), 0);
  ASSERT_EQ(setenv("HIP_TRACE_SYNC", "1", 1), 0);
  ASSERT_EQ(setenv("HIP_TRACE_DUMP", "2", 1), 0);

  // The real interceptor formats the trace. Only the downstream HIP functions
  // are substituted; no device execution is needed to exercise its log output.
  hip_function_table_t backend = {
      .hipDeviceSynchronize = +[]() { return 0; },
      .hipModuleGetFunction =
          +[](hipFunction_t* function, hipModule_t module, const char*) {
            *function = module;
            return 0;
          },
      .hipModuleLaunchKernel =
          +[](hipFunction_t, unsigned int, unsigned int, unsigned int,
              unsigned int, unsigned int, unsigned int, unsigned int,
              hipStream_t, void**, void**) { return 17; },
      .hipLaunchKernel = +[](const void*, dim3, dim3, void**, size_t,
                             hipStream_t) { return 23; },
      .hipExtModuleLaunchKernel =
          +[](hipFunction_t, unsigned int, unsigned int, unsigned int,
              unsigned int, unsigned int, unsigned int, size_t, hipStream_t,
              void**, void**, hipEvent_t, hipEvent_t,
              unsigned int) { return 31; },
  };

  hip_function_table_t* interceptor = hip_interceptor_init(&backend);
  const std::string long_name = std::string(500, 'k') + "_unique_end";
  int registered_identity = 0;
  int unknown_identity = 0;
  hipFunction_t registered_function = nullptr;
  EXPECT_EQ(interceptor->hipModuleGetFunction(
                &registered_function, &registered_identity, long_name.c_str()),
            0);
  const hipFunction_t functions[] = {registered_function, &unknown_identity};
  for (hipFunction_t function : functions) {
    EXPECT_EQ(interceptor->hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0,
                                                 nullptr, nullptr, nullptr),
              17);
    const dim3 size = {1, 1, 1};
    EXPECT_EQ(
        interceptor->hipLaunchKernel(function, size, size, nullptr, 0, nullptr),
        23);
    EXPECT_EQ(interceptor->hipExtModuleLaunchKernel(
                  function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                  nullptr, nullptr, 0),
              31);
  }
  hip_interceptor_shutdown();
  EXPECT_EQ(unsetenv("HIP_TRACE_FILE"), 0);
  EXPECT_EQ(unsetenv("HIP_TRACE_LEVEL"), 0);
  EXPECT_EQ(unsetenv("HIP_TRACE_SYNC"), 0);
  EXPECT_EQ(unsetenv("HIP_TRACE_DUMP"), 0);

  std::ifstream stream(trace_path.path());
  ASSERT_TRUE(stream.is_open());
  const std::string trace((std::istreambuf_iterator<char>(stream)),
                          std::istreambuf_iterator<char>());
  const int results[] = {17, 23, 31};
  for (int ordinal = 0; ordinal < 6; ++ordinal) {
    const std::string label = "kernel #" + std::to_string(ordinal) + ": " +
                              (ordinal < 3 ? long_name : "(unknown)");
    EXPECT_THAT(trace, testing::HasSubstr("=== Buffer Dump: BEFORE " + label +
                                          " (0 buffers) ===\n"));
    EXPECT_THAT(
        trace, testing::HasSubstr("=== Buffer Dump: AFTER " + label + " (ret=" +
                                  std::to_string(results[ordinal % 3]) +
                                  ") (0 buffers) ===\n"));
  }
  EXPECT_TRUE(trace_path.Remove());
}

}  // namespace
