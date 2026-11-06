// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/task/executable/environment.h"

#include <array>
#include <thread>

#include "iree/base/internal/cpu.h"
#include "iree/testing/gtest.h"

namespace {

void ExpectHostEnvironment(
    const iree_hal_executable_environment_v0_t& environment) {
  iree_cpu_data_t cpu_data;
  iree_cpu_query_data(iree_allocator_system(), &cpu_data);
  EXPECT_EQ(environment.constants, nullptr);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(environment.processor.data);
       ++i) {
    const uint64_t expected =
        i < IREE_ARRAYSIZE(cpu_data.fields) ? cpu_data.fields[i] : 0;
    EXPECT_EQ(environment.processor.data[i], expected) << "field " << i;
  }
}

TEST(ExecutableEnvironmentTest, QueriesHostWithoutChangingGlobalCpuData) {
  std::array<uint64_t, IREE_CPU_DATA_FIELD_COUNT> original_data;
  iree_cpu_read_data(original_data.size(), original_data.data());
  std::array<uint64_t, IREE_CPU_DATA_FIELD_COUNT> cached_data;
  cached_data.fill(UINT64_MAX);
  iree_cpu_initialize_with_data(cached_data.size(), cached_data.data());

  iree_hal_executable_environment_v0_t environment;
  memset(&environment, 0xCD, sizeof(environment));
  iree_hal_executable_environment_initialize(iree_allocator_system(),
                                             &environment);
  std::array<uint64_t, IREE_CPU_DATA_FIELD_COUNT> actual_data;
  iree_cpu_read_data(actual_data.size(), actual_data.data());
  iree_cpu_initialize_with_data(original_data.size(), original_data.data());

  EXPECT_EQ(actual_data, cached_data);
  ExpectHostEnvironment(environment);
}

TEST(ExecutableEnvironmentTest,
     InitializesIndependentEnvironmentsConcurrently) {
  iree_hal_executable_environment_v0_t environments[2];
  std::thread worker([&]() {
    iree_hal_executable_environment_initialize(iree_allocator_system(),
                                               &environments[0]);
  });
  iree_hal_executable_environment_initialize(iree_allocator_system(),
                                             &environments[1]);
  worker.join();
  ExpectHostEnvironment(environments[0]);
  ExpectHostEnvironment(environments[1]);
}

}  // namespace
