// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif

#include <windows.h>

#include <thread>
#include <tuple>

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/threading/latency.h"
#include "iree/base/threading/latency_test_library_embed.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr DWORD kExecutionSpeed = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;

struct RuntimeCopy {
  // Independently loaded runtime image, retained until its scopes end.
  iree_dynamic_library_t* library = nullptr;
  // Acquisition through this image's implementation.
  bool (*begin)(iree_latency_scope_t*) = nullptr;
  // Release through this image's implementation, consuming native status here.
  iree_status_code_t (*end)(iree_latency_scope_t*) = nullptr;
  // Scope owned by this image and cleaned up even after an assertion failure.
  iree_latency_scope_t scope = {};

  void Load() {
    const auto* file = latency_test_library_create();
    IREE_ASSERT_OK(iree_dynamic_library_load_from_memory(
        IREE_SV("latency_scope"),
        iree_make_const_byte_span(file->data, file->size),
        IREE_DYNAMIC_LIBRARY_FLAG_NONE, iree_allocator_system(), &library));
    IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(
        library, "iree_latency_test_begin", reinterpret_cast<void**>(&begin)));
    IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(
        library, "iree_latency_test_end", reinterpret_cast<void**>(&end)));
  }

  void Unload() {
    if (end) {
      EXPECT_EQ(IREE_STATUS_OK, end(&scope));
    }
    iree_dynamic_library_release(library);
    library = nullptr;
    begin = nullptr;
    end = nullptr;
  }
};

class WindowsLatencyEmbeddingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                               &original_, sizeof(original_))) {
      GTEST_SKIP() << "Process power policy is unavailable: " << GetLastError();
    }
    if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                               &original_, sizeof(original_))) {
      GTEST_SKIP() << "Process power policy is not writable: "
                   << GetLastError();
    }
    restore_policy_ = true;
    for (auto& copy : copies_) {
      copy.Load();
      if (HasFatalFailure()) {
        return;
      }
    }
    // The loader must produce separate static runtime state, not references
    // to a single image. Otherwise overlap would miss the embedding boundary.
    ASSERT_NE(copies_[0].begin, copies_[1].begin);
  }

  void TearDown() override {
    for (auto& copy : copies_) {
      copy.Unload();
    }
    if (restore_policy_) {
      EXPECT_TRUE(SetProcessInformation(GetCurrentProcess(),
                                        ProcessPowerThrottling, &original_,
                                        sizeof(original_)));
    }
  }

  void SetPolicy(DWORD control, DWORD state) {
    auto policy = original_;
    policy.ControlMask = (policy.ControlMask & ~kExecutionSpeed) | control;
    policy.StateMask = (policy.StateMask & ~kExecutionSpeed) | state;
    ASSERT_TRUE(SetProcessInformation(
        GetCurrentProcess(), ProcessPowerThrottling, &policy, sizeof(policy)));
  }

  void ExpectPolicy(DWORD control, DWORD state) {
    PROCESS_POWER_THROTTLING_STATE policy = {
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, 0, 0};
    ASSERT_TRUE(GetProcessInformation(
        GetCurrentProcess(), ProcessPowerThrottling, &policy, sizeof(policy)));
    EXPECT_EQ(control, policy.ControlMask & kExecutionSpeed);
    EXPECT_EQ(state, policy.StateMask & kExecutionSpeed);
  }

  // Native policy before this test; restored after all scopes have ended.
  PROCESS_POWER_THROTTLING_STATE original_ = {
      PROCESS_POWER_THROTTLING_CURRENT_VERSION, 0, 0};
  // Whether native policy was successfully read and can be restored.
  bool restore_policy_ = false;
  // Independent embeddings sharing only the native process policy.
  RuntimeCopy copies_[2];
};

class WindowsLatencyReleaseOrderTest
    : public WindowsLatencyEmbeddingTest,
      public ::testing::WithParamInterface<std::tuple<DWORD, bool>> {};

TEST_P(WindowsLatencyReleaseOrderTest, RestoresAfterLastOwner) {
  const DWORD initial = std::get<0>(GetParam());
  const DWORD control = initial ? kExecutionSpeed : 0;
  const DWORD state = initial == 2 ? kExecutionSpeed : 0;
  SetPolicy(control, state);
  if (HasFatalFailure()) {
    return;
  }

  for (auto& copy : copies_) {
    ASSERT_TRUE(copy.begin(&copy.scope));
    ExpectPolicy(kExecutionSpeed, 0);
  }
  const int first = std::get<1>(GetParam()) ? 1 : 0;
  ASSERT_EQ(IREE_STATUS_OK, copies_[first].end(&copies_[first].scope));
  ExpectPolicy(kExecutionSpeed, 0);
  ASSERT_EQ(IREE_STATUS_OK, copies_[1 - first].end(&copies_[1 - first].scope));
  ExpectPolicy(control, state);
}

INSTANTIATE_TEST_SUITE_P(SystemHighAndEcoQoS, WindowsLatencyReleaseOrderTest,
                         ::testing::Combine(::testing::Values(0u, 1u, 2u),
                                            ::testing::Bool()));

TEST_F(WindowsLatencyEmbeddingTest, OriginalOwnerMayUnloadAndRejoin) {
  SetPolicy(kExecutionSpeed, kExecutionSpeed);
  if (HasFatalFailure()) {
    return;
  }
  for (auto& copy : copies_) {
    ASSERT_TRUE(copy.begin(&copy.scope));
  }

  copies_[0].Unload();
  ExpectPolicy(kExecutionSpeed, 0);
  copies_[0].Load();
  if (HasFatalFailure()) {
    return;
  }
  ASSERT_TRUE(copies_[0].begin(&copies_[0].scope));
  ASSERT_EQ(IREE_STATUS_OK, copies_[1].end(&copies_[1].scope));
  ExpectPolicy(kExecutionSpeed, 0);
  ASSERT_EQ(IREE_STATUS_OK, copies_[0].end(&copies_[0].scope));
  ExpectPolicy(kExecutionSpeed, kExecutionSpeed);
}

TEST_F(WindowsLatencyEmbeddingTest, ConcurrentOwnershipEpochs) {
  SetPolicy(kExecutionSpeed, kExecutionSpeed);
  if (HasFatalFailure()) {
    return;
  }
  SYNCHRONIZATION_BARRIER barrier;
  ASSERT_TRUE(InitializeSynchronizationBarrier(&barrier, 3, -1));
  constexpr int kEpochCount = 16;
  auto run = [&](RuntimeCopy& copy) {
    for (int i = 0; i < kEpochCount; ++i) {
      EnterSynchronizationBarrier(&barrier, 0);
      EXPECT_TRUE(copy.begin(&copy.scope));
      iree_latency_scope_t nested = {};
      EXPECT_TRUE(copy.begin(&nested));
      EXPECT_EQ(IREE_STATUS_OK, copy.end(&nested));
      EnterSynchronizationBarrier(&barrier, 0);
      EnterSynchronizationBarrier(&barrier, 0);
      EXPECT_EQ(IREE_STATUS_OK, copy.end(&copy.scope));
      EnterSynchronizationBarrier(&barrier, 0);
    }
  };
  std::thread first([&] { run(copies_[0]); });
  std::thread second([&] { run(copies_[1]); });
  for (int i = 0; i < kEpochCount; ++i) {
    // Start acquisitions together and inspect only after both return. Likewise
    // race final releases and inspect after both native operations complete.
    EnterSynchronizationBarrier(&barrier, 0);
    EnterSynchronizationBarrier(&barrier, 0);
    ExpectPolicy(kExecutionSpeed, 0);
    EnterSynchronizationBarrier(&barrier, 0);
    EnterSynchronizationBarrier(&barrier, 0);
    ExpectPolicy(kExecutionSpeed, kExecutionSpeed);
  }
  first.join();
  second.join();
  DeleteSynchronizationBarrier(&barrier);
}

}  // namespace
