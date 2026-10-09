// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/proactor.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <string>

#include "iree/async/operations/file.h"
#include "iree/async/platform/posix/api.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/thread.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_LINUX)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

TEST(PosixProactorDeathTest, EarlyCreationFailurePreservesStdin) {
#if defined(IREE_PLATFORM_LINUX)
  constexpr auto unavailable_backend = IREE_ASYNC_POSIX_EVENT_BACKEND_KQUEUE;
#else
  constexpr auto unavailable_backend = IREE_ASYNC_POSIX_EVENT_BACKEND_EPOLL;
#endif
  // Isolate descriptor 0 from the test harness. Failure before wake setup must
  // not mistake the zero-initialized field for an owned file descriptor.
  ASSERT_EXIT(
      {
        int descriptor = open("/dev/null", O_RDONLY);
        if (descriptor < 0 || dup2(descriptor, STDIN_FILENO) < 0) {
          _exit(1);
        }
        if (descriptor != STDIN_FILENO) {
          close(descriptor);
        }
        iree_async_proactor_t* proactor = nullptr;
        iree_status_t status = iree_async_proactor_create_posix_with_backend(
            iree_async_proactor_options_default(), unavailable_backend,
            iree_allocator_system(), &proactor);
        bool rejected = iree_status_is_unavailable(status);
        iree_status_free(status);
        if (!rejected || proactor) {
          _exit(2);
        }
        _exit(fcntl(STDIN_FILENO, F_GETFD) >= 0 ? 0 : 3);
      },
      ::testing::ExitedWithCode(0), "");
}

class PosixProactorTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_async_file_release(file_);
    iree_async_proactor_pool_release(pool_);
    iree_status_free(completion_status_);
  }

  iree_status_t CreatePool(uint32_t node_id,
                           iree_thread_affinity_t worker_affinity) {
    auto options = iree_async_proactor_pool_options_default();
    options.proactor_create = iree_async_proactor_create_posix;
    options.proactor_options.worker_affinity = worker_affinity;
    options.runner = {};  // The test drives poll() on its own thread.
    return iree_async_proactor_pool_create(1, &node_id, options,
                                           iree_allocator_system(), &pool_);
  }

  void SubmitAndAwait(iree_async_operation_t* operation) {
    completed_ = false;
    operation->completion_fn =
        +[](void* user_data, iree_async_operation_t* operation,
            iree_status_t status, iree_async_completion_flags_t flags) {
          auto* test = static_cast<PosixProactorTest*>(user_data);
          test->completion_status_ = status;
          test->completed_ = true;
        };
    operation->user_data = this;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, operation));
    while (!completed_) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
    iree_status_t status = completion_status_;
    completion_status_ = iree_ok_status();
    IREE_ASSERT_OK(status);
  }

  // Pool owning the real POSIX proactor and its workers.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Proactor borrowed from pool_.
  iree_async_proactor_t* proactor_ = nullptr;
  // File opened by a backend worker.
  iree_async_file_t* file_ = nullptr;
  // Callback state accessed only by the test's polling thread.
  bool completed_ = false;
  // Terminal status transferred by the most recent callback.
  iree_status_t completion_status_ = iree_ok_status();
};

TEST_F(PosixProactorTest, UnavailableWorkerNodeFailsCreation) {
  auto options = iree_async_proactor_options_default();
  iree_thread_affinity_set_group_any(UINT32_MAX - 1, &options.worker_affinity);
  iree_async_proactor_t* proactor = nullptr;
  IREE_EXPECT_NOT_OK(iree_async_proactor_create_posix(
      options, iree_allocator_system(), &proactor));
  EXPECT_EQ(proactor, nullptr);
  iree_async_proactor_release(proactor);
}

TEST_F(PosixProactorTest, PoolNodeConstrainsBackendWorkersWithoutRunner) {
  // With no poll runner, worker startup is what must reject the invalid node.
  // The entry's explicit node overrides unspecified worker placement.
  IREE_ASSERT_OK(CreatePool(UINT32_MAX - 1, {}));
  IREE_EXPECT_NOT_OK(iree_async_proactor_pool_get(pool_, 0, &proactor_));
  EXPECT_EQ(proactor_, nullptr);

  // Failed lazy creation must not publish an entry or strand startup resources.
  iree_async_proactor_pool_entry_t* entry = nullptr;
  IREE_EXPECT_NOT_OK(iree_async_proactor_pool_acquire(pool_, 0, &entry));
  EXPECT_EQ(entry, nullptr);
  iree_async_proactor_pool_entry_release(entry);
}

#if defined(IREE_PLATFORM_LINUX)
TEST_F(PosixProactorTest, WorkerAffinityAppliesToFileOperations) {
  int cpu = sched_getcpu();
  ASSERT_GE(cpu, 0);
  ASSERT_LT(cpu, CPU_SETSIZE);
  iree_thread_affinity_t affinity =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment conversion
           // differs from list initialization.
  affinity.id_assigned = 1;
  affinity.id = cpu;
  IREE_ASSERT_OK(CreatePool(UINT32_MAX, affinity));
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool_, 0, &proactor_));

  // Every worker has its affinity before construction publishes the proactor.
  auto* posix_proactor = iree_async_proactor_posix_cast(proactor_);
  for (iree_host_size_t i = 0; i < posix_proactor->worker_count; ++i) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    ASSERT_EQ(pthread_getaffinity_np(
                  (pthread_t)iree_thread_id(posix_proactor->workers[i].thread),
                  sizeof(mask), &mask),
              0);
    EXPECT_EQ(CPU_COUNT(&mask), 1);
    EXPECT_TRUE(CPU_ISSET(cpu, &mask));
  }

  // /proc/thread-self resolves on the worker executing FILE_OPEN. Reading the
  // resulting file observes that worker's identity and allowed CPUs through the
  // same asynchronous file path used by applications.
  iree_async_file_open_operation_t open_operation = {};
  open_operation.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
  open_operation.path = "/proc/thread-self/status";
  open_operation.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
  ASSERT_NO_FATAL_FAILURE(SubmitAndAwait(&open_operation.base));
  file_ = open_operation.opened_file;
  ASSERT_NE(file_, nullptr);

  std::array<char, 8192> buffer;
  iree_async_file_read_operation_t read_operation = {};
  read_operation.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_operation.file = file_;
  read_operation.buffer =
      iree_async_span_from_ptr(buffer.data(), buffer.size());
  ASSERT_NO_FATAL_FAILURE(SubmitAndAwait(&read_operation.base));
  ASSERT_LT(read_operation.bytes_read, buffer.size());
  std::string contents(buffer.data(), read_operation.bytes_read);
  EXPECT_NE(contents.find("Name:\tasync_worker\n"), std::string::npos);
  EXPECT_NE(contents.find("Cpus_allowed_list:\t" + std::to_string(cpu) + "\n"),
            std::string::npos);
}
#endif  // IREE_PLATFORM_LINUX

}  // namespace
