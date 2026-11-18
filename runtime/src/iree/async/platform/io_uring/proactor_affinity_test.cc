// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dirent.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <tuple>

#include "iree/async/file.h"
#include "iree/async/operations/file.h"
#include "iree/async/platform/io_uring/api.h"
#include "iree/base/threading/affinity.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// io-wq sets affinity before the worker sets its name and starts executing.
// The blocked pipe read keeps that worker alive for independent native
// readback.
int FindNativeWorker() {
  DIR* tasks = opendir("/proc/self/task");
  if (!tasks) {
    return 0;
  }
  int worker = 0;
  for (dirent* entry; !worker && (entry = readdir(tasks));) {
    int tid = atoi(entry->d_name);
    if (!tid) {
      continue;
    }
    char path[128];
    snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
    FILE* file = fopen(path, "r");
    if (!file) {
      continue;
    }
    char name[64] = {};
    if (fgets(name, sizeof(name), file) && !strncmp(name, "iou-wrk-", 8)) {
      worker = tid;
    }
    fclose(file);
  }
  closedir(tasks);
  return worker;
}

class IoUringAffinityTest
    : public ::testing::TestWithParam<
          std::tuple<iree_async_proactor_threading_mode_t, int>> {};

TEST_P(IoUringAffinityTest, AppliesBeforeFirstQueuedRead) {
  const auto [mode, selection] = GetParam();
  iree_thread_affinity_set_t budget = {};
  IREE_ASSERT_OK(iree_thread_affinity_query_current(&budget));
  const auto cpu = iree_thread_affinity_set_find_first_set(&budget, 0);
  ASSERT_LT(cpu, IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
  // Query CPU and node together so migration cannot produce a mismatched pair.
  unsigned current_cpu = 0;
  unsigned current_node = 0;
  ASSERT_EQ(syscall(__NR_getcpu, &current_cpu, &current_node, nullptr), 0);
  const auto selected_cpu = selection == 1 ? current_cpu : cpu;
  budget = {};
  iree_bitmap_set(iree_thread_affinity_set_bitmap(&budget), selected_cpu);
  auto options = iree_async_proactor_options_default();
  options.threading_mode = mode;
  options.worker_processor_set = &budget;
  if (selection == 0) {
    options.worker_affinity.id_assigned = 1;
    options.worker_affinity.id = static_cast<uint32_t>(selected_cpu);
  } else if (selection == 1) {
    iree_thread_affinity_set_group_any(current_node, &options.worker_affinity);
  }

  iree_async_proactor_t* proactor = nullptr;
  // Probe availability before starting the owning thread or accepting I/O.
  // CROSS_THREAD also exercises real construction on a different task.
  iree_status_t status = iree_async_proactor_create_io_uring(
      options, iree_allocator_system(), &proactor);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "io_uring worker affinity is unavailable";
  }
  IREE_ASSERT_OK(status);
  if (mode == IREE_ASYNC_PROACTOR_THREADING_SAME_THREAD) {
    iree_async_proactor_release(proactor);
    proactor = nullptr;
  }

  int pipes[2] = {-1, -1};
  ASSERT_EQ(pipe(pipes), 0);
  std::atomic<bool> submitted{false};
  std::thread poller([&] {
    if (mode == IREE_ASYNC_PROACTOR_THREADING_SAME_THREAD) {
      IREE_CHECK_OK(iree_async_proactor_create_io_uring(
          options, iree_allocator_system(), &proactor));
    }
    iree_async_file_t* file = nullptr;
    IREE_CHECK_OK(iree_async_file_import(proactor,
                                         iree_async_primitive_from_fd(pipes[0]),
                                         IREE_ASYNC_FILE_FLAG_NONE, &file));
    bool complete = false;
    char output = 0;
    iree_async_file_read_operation_t read = {};
    iree_async_operation_initialize(
        &read.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        [](void* user_data, iree_async_operation_t* operation,
           iree_status_t status, iree_async_completion_flags_t flags) {
          IREE_CHECK_OK(status);
          *static_cast<bool*>(user_data) = true;
        },
        &complete);
    read.file = file;
    read.offset = UINT64_MAX;
    read.buffer = iree_async_span_from_ptr(&output, 1);
    IREE_CHECK_OK(iree_async_proactor_submit_one(proactor, &read.base));
    iree_status_t first_poll =
        iree_async_proactor_poll(proactor, iree_immediate_timeout(), nullptr);
    if (iree_status_is_deadline_exceeded(first_poll)) {
      iree_status_free(first_poll);
    } else {
      IREE_CHECK_OK(first_poll);
    }
    submitted.store(true, std::memory_order_release);
    while (!complete) {
      IREE_CHECK_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
    EXPECT_EQ(output, 'x');
    EXPECT_EQ(read.bytes_read, 1u);
    iree_async_file_release(file);
    iree_async_proactor_end_polling(proactor);
    iree_async_proactor_release(proactor);
  });
  while (!submitted.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  int worker = 0;
  while (!(worker = FindNativeWorker())) {
    std::this_thread::yield();
  }
  std::array<unsigned long,
             CPU_ALLOC_SIZE(IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT) /
                 sizeof(unsigned long)>
      observed = {};
  auto* native_mask = reinterpret_cast<cpu_set_t*>(observed.data());
  EXPECT_EQ(sched_getaffinity(worker, sizeof(observed), native_mask), 0);
  EXPECT_EQ(CPU_COUNT_S(sizeof(observed), native_mask), 1);
  EXPECT_TRUE(CPU_ISSET_S(selected_cpu, sizeof(observed), native_mask));
  // Always unblock and join before reporting a failed affinity expectation.
  if (write(pipes[1], "x", 1) != 1) {
    perror("release pipe reader");
    abort();
  }
  poller.join();
  EXPECT_EQ(close(pipes[1]), 0);
}

INSTANTIATE_TEST_SUITE_P(
    Ownership, IoUringAffinityTest,
    ::testing::Combine(
        ::testing::Values(IREE_ASYNC_PROACTOR_THREADING_SAME_THREAD,
                          IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD),
        ::testing::Values(0, 1, 2)));

TEST(IoUringAffinityValidationTest, EmptyBudgetRejectsBeforeCreation) {
  const iree_thread_affinity_set_t budget = {};
  auto options = iree_async_proactor_options_default();
  options.worker_processor_set = &budget;
  iree_async_proactor_t* proactor = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_proactor_create_io_uring(
                            options, iree_allocator_system(), &proactor));
  EXPECT_EQ(proactor, nullptr);
}

TEST(IoUringAffinityValidationTest, MissingNodeRejectsBeforeCreation) {
  auto options = iree_async_proactor_options_default();
  iree_thread_affinity_set_group_any(UINT32_MAX, &options.worker_affinity);
  iree_async_proactor_t* proactor = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_async_proactor_create_io_uring(
                            options, iree_allocator_system(), &proactor));
  EXPECT_EQ(proactor, nullptr);
}

}  // namespace
