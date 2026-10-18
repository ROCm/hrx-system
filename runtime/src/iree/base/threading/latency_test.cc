// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif

#include "iree/base/threading/latency.h"

#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#endif  // IREE_PLATFORM_WINDOWS

namespace {

class LatencyTest : public ::testing::Test {
 protected:
  void TearDown() override {
    IREE_EXPECT_OK(iree_latency_scope_end(&first_));
    IREE_EXPECT_OK(iree_latency_scope_end(&second_));
  }

  void Begin(int32_t maximum_latency_us, iree_latency_scope_t* scope) {
    auto options = iree_latency_scope_options_default();
    options.maximum_cpu_idle_latency_us = maximum_latency_us;
    if (!iree_latency_scope_begin(options, scope)) {
      EXPECT_EQ(0u, scope->handle);
      GTEST_SKIP() << "Requested latency-sensitive power policy is unavailable";
    }
  }

  // First independently owned native request, released even on test failure.
  iree_latency_scope_t first_ = {};
  // Second independently owned request for overlap and release-order tests.
  iree_latency_scope_t second_ = {};
};

TEST_F(LatencyTest, InactiveScope) {
  IREE_EXPECT_OK(iree_latency_scope_end(&first_));
  IREE_EXPECT_OK(iree_latency_scope_end(&first_));
  EXPECT_EQ(0u, first_.handle);
}

TEST_F(LatencyTest, InvalidLatency) {
  EXPECT_FALSE(iree_latency_scope_begin({-2}, &first_));
  EXPECT_EQ(0u, first_.handle);
}

TEST_F(LatencyTest, OwnershipMayMoveBetweenThreads) {
  Begin(-1, &first_);
  if (IsSkipped() || HasFatalFailure()) {
    return;
  }
  EXPECT_NE(0u, first_.handle);
  std::thread releaser(
      [&] { IREE_EXPECT_OK(iree_latency_scope_end(&first_)); });
  releaser.join();
  EXPECT_EQ(0u, first_.handle);
}

TEST_F(LatencyTest, OverlappingActivities) {
  Begin(-1, &first_);
  if (IsSkipped() || HasFatalFailure()) {
    return;
  }
  ASSERT_TRUE(
      iree_latency_scope_begin(iree_latency_scope_options_default(), &second_));
  IREE_EXPECT_OK(iree_latency_scope_end(&first_));
  EXPECT_NE(0u, second_.handle);
  IREE_EXPECT_OK(iree_latency_scope_end(&second_));
}

#if defined(IREE_PLATFORM_WINDOWS) || defined(IREE_PLATFORM_MACOS)

TEST_F(LatencyTest, NumericIdleLimitsAreUnavailable) {
  EXPECT_FALSE(iree_latency_scope_begin({5}, &first_));
  EXPECT_EQ(0u, first_.handle);
}

#endif  // IREE_PLATFORM_WINDOWS || IREE_PLATFORM_MACOS

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

class LinuxLatencyTest : public LatencyTest {
 protected:
  void SetUp() override {
    // Reading effective policy is an additional permission beyond acquiring
    // a request. Restricted deployments may allow only write access.
    const int descriptor = open("/dev/cpu_dma_latency", O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      GTEST_SKIP() << "CPU idle latency policy cannot be observed: " << errno;
    }
    close(descriptor);
  }
};

// Read through the native interface independently of the IREE scope. Every
// open starts a new read position and contributes only the kernel default.
static int32_t ReadEffectiveLatency() {
  const int descriptor = open("/dev/cpu_dma_latency", O_RDONLY | O_CLOEXEC);
  EXPECT_GE(descriptor, 0);
  if (descriptor < 0) {
    return -1;
  }
  int32_t latency_us = -1;
  ssize_t length;
  do {
    length = read(descriptor, &latency_us, sizeof(latency_us));
  } while (length < 0 && errno == EINTR);
  close(descriptor);
  EXPECT_EQ(sizeof(latency_us), length);
  return latency_us;
}

TEST_F(LinuxLatencyTest, KernelAggregatesIndependentScopes) {
  Begin(30, &first_);
  if (IsSkipped() || HasFatalFailure()) {
    return;
  }
  EXPECT_LE(ReadEffectiveLatency(), 30);
  ASSERT_TRUE(iree_latency_scope_begin({5}, &second_));
  EXPECT_LE(ReadEffectiveLatency(), 5);

  const int first_descriptor = static_cast<int>(first_.handle - 1);
  const int second_descriptor = static_cast<int>(second_.handle - 1);
  EXPECT_NE(0, fcntl(first_descriptor, F_GETFD) & FD_CLOEXEC);
  EXPECT_NE(0, fcntl(second_descriptor, F_GETFD) & FD_CLOEXEC);

  // Ending the older request leaves the newer request intact. The effective
  // latency can be lower if another process already holds a tighter request.
  IREE_EXPECT_OK(iree_latency_scope_end(&first_));
  EXPECT_EQ(-1, fcntl(first_descriptor, F_GETFD));
  EXPECT_EQ(EBADF, errno);
  EXPECT_LE(ReadEffectiveLatency(), 5);
  IREE_EXPECT_OK(iree_latency_scope_end(&second_));
  EXPECT_EQ(-1, fcntl(second_descriptor, F_GETFD));
  EXPECT_EQ(EBADF, errno);
}

TEST_F(LinuxLatencyTest, ZeroLatency) {
  Begin(0, &first_);
  if (IsSkipped() || HasFatalFailure()) {
    return;
  }
  EXPECT_EQ(0, ReadEffectiveLatency());
}

TEST_F(LinuxLatencyTest, DefaultRequestsFiveMicroseconds) {
  Begin(-1, &first_);
  if (IsSkipped() || HasFatalFailure()) {
    return;
  }
  // Other processes may acquire or release requests while this test runs.
  // Our request bounds effective latency but does not own its exact value.
  EXPECT_LE(ReadEffectiveLatency(), 5);
}

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

#if defined(IREE_PLATFORM_WINDOWS)

class WindowsLatencyTest : public LatencyTest,
                           public ::testing::WithParamInterface<DWORD> {
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
  }

  void TearDown() override {
    LatencyTest::TearDown();
    if (restore_policy_) {
      EXPECT_TRUE(SetProcessInformation(GetCurrentProcess(),
                                        ProcessPowerThrottling, &original_,
                                        sizeof(original_)));
    }
  }

  // Original process policy restored even after an assertion fails.
  PROCESS_POWER_THROTTLING_STATE original_ = {
      PROCESS_POWER_THROTTLING_CURRENT_VERSION, 0, 0};
  // True once the original process policy was captured successfully.
  bool restore_policy_ = false;
};

TEST_P(WindowsLatencyTest, LastScopeRestoresPriorPolicy) {
  auto before = original_;
  const DWORD mask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  before.ControlMask &= ~mask;
  before.StateMask &= ~mask;
  if (GetParam() != 0) {
    before.ControlMask |= mask;
  }
  if (GetParam() == 2) {
    before.StateMask |= mask;
  }
  ASSERT_TRUE(SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                    &before, sizeof(before)));

  const auto options = iree_latency_scope_options_default();
  ASSERT_TRUE(iree_latency_scope_begin(options, &first_));
  ASSERT_TRUE(iree_latency_scope_begin(options, &second_));
  IREE_ASSERT_OK(iree_latency_scope_end(&first_));

  PROCESS_POWER_THROTTLING_STATE during = {
      PROCESS_POWER_THROTTLING_CURRENT_VERSION, 0, 0};
  ASSERT_TRUE(GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                    &during, sizeof(during)));
  EXPECT_EQ(mask, during.ControlMask & mask);
  EXPECT_EQ(0u, during.StateMask & mask);

  IREE_ASSERT_OK(iree_latency_scope_end(&second_));
  PROCESS_POWER_THROTTLING_STATE after = {
      PROCESS_POWER_THROTTLING_CURRENT_VERSION, 0, 0};
  ASSERT_TRUE(GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                    &after, sizeof(after)));
  EXPECT_EQ(before.ControlMask, after.ControlMask);
  EXPECT_EQ(before.StateMask, after.StateMask);
}

INSTANTIATE_TEST_SUITE_P(SystemHighAndEcoQoS, WindowsLatencyTest,
                         ::testing::Values(0u, 1u, 2u));

#endif  // IREE_PLATFORM_WINDOWS

}  // namespace
