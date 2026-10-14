// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdio>

#include "iree/async/buffer_pool.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/platform/io_uring/api.h"
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Changes only the death-test child's syscall policy. Registration restrictions
// select one opcode; setup restrictions leave every registration untouched.
void RestrictIoUring(bool restrict_setup, uint32_t opcode, int error_number) {
  uint32_t syscall_number =
      restrict_setup ? __NR_io_uring_setup : __NR_io_uring_register;
  struct sock_filter instructions[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, syscall_number, 0, 3),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
               offsetof(struct seccomp_data, args[1])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, opcode, 0, 1),
      BPF_STMT(BPF_RET | BPF_K,
               SECCOMP_RET_ERRNO | static_cast<uint32_t>(error_number)),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  if (restrict_setup) {
    instructions[1].jt = 2;
  }
  struct sock_fprog program = {IREE_ARRAYSIZE(instructions), instructions};
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
      prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
    perror("install syscall restriction");
    _exit(2);
  }
}

struct Completion {
  // Number of terminal callbacks delivered on the polling task.
  int count = 0;
  // Result of the last completion, independent of operation storage.
  iree_status_code_t code = IREE_STATUS_UNKNOWN;

  static void Record(void* user_data, iree_async_operation_t* operation,
                     iree_status_t status,
                     iree_async_completion_flags_t flags) {
    auto* completion = static_cast<Completion*>(user_data);
    EXPECT_EQ(flags, IREE_ASYNC_COMPLETION_FLAG_NONE);
    completion->code = iree_status_code(status);
    ++completion->count;
    iree_status_free(status);
  }
};

// Owns real sockets, registrations and the platform-selected proactor. CHECKs
// abort the isolated child if native progress fails with borrowed operations
// still in flight; returning through their stack storage would be unsafe.
class ReceiveHarness {
 public:
  explicit ReceiveHarness(bool expect_io_uring = true) {
    auto options = iree_async_proactor_options_default();
    options.max_concurrent_operations = 8;
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor));
    auto slab_options = iree_async_slab_options_default();
    slab_options.buffer_size = 16384;
    slab_options.buffer_count = 4;
    IREE_CHECK_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));
    IREE_CHECK_OK(iree_async_proactor_register_slab(
        proactor, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region_));
    EXPECT_EQ(region_->type == IREE_ASYNC_REGION_TYPE_IOURING, expect_io_uring);
    if (expect_io_uring) {
      EXPECT_EQ(region_->handles.iouring.buffer_group_id, -1);
      EXPECT_FALSE(
          iree_any_bit_set(iree_async_proactor_query_capabilities(proactor),
                           IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT));
    }
    IREE_CHECK_OK(
        iree_async_buffer_pool_create(region_, iree_allocator_system(), &pool));
    int descriptors[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0,
                   descriptors) != 0) {
      perror("socketpair");
      _exit(2);
    }
    peer = descriptors[1];
    IREE_CHECK_OK(iree_async_socket_import(
        proactor, iree_async_primitive_from_fd(descriptors[0]),
        IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM, IREE_ASYNC_SOCKET_FLAG_NONE,
        &socket_));
  }

  ~ReceiveHarness() {
    EXPECT_EQ(iree_async_buffer_pool_available(pool), 4u);
    iree_async_socket_release(socket_);
    EXPECT_EQ(close(peer), 0);
    iree_async_buffer_pool_release(pool);
    iree_async_region_release(region_);
    iree_async_slab_release(slab_);
    iree_async_proactor_release(proactor);
  }

  void Initialize(iree_async_socket_recv_pool_operation_t* operation,
                  Completion* completion,
                  iree_async_operation_flags_t flags = 0) {
    *operation = {};
    iree_async_operation_initialize(&operation->base,
                                    IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
                                    flags, Completion::Record, completion);
    operation->socket = socket_;
    operation->pool = pool;
  }

  void PollUntil(Completion& completion, int count = 1) {
    while (completion.count < count) {
      IREE_CHECK_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
    EXPECT_EQ(completion.count, count);
  }

  void RoundTrip() {
    // More receives than pool slots proves that successful leases recycle.
    for (uint8_t expected = 1; expected <= 9; ++expected) {
      EXPECT_EQ(send(peer, &expected, 1, MSG_NOSIGNAL), 1);
      Completion completion;
      iree_async_socket_recv_pool_operation_t receive;
      Initialize(&receive, &completion);
      IREE_CHECK_OK(iree_async_proactor_submit_one(proactor, &receive.base));
      PollUntil(completion);
      EXPECT_EQ(completion.code, IREE_STATUS_OK);
      EXPECT_EQ(receive.bytes_received, 1u);
      if (receive.lease.release.fn && receive.bytes_received == 1) {
        EXPECT_EQ(*iree_async_span_ptr(receive.lease.span), expected);
      } else {
        ADD_FAILURE() << "successful receive did not transfer its lease";
      }
      iree_async_buffer_lease_release(&receive.lease);
    }
  }

  // Proactor owned until every completion has returned.
  iree_async_proactor_t* proactor = nullptr;
  // Pool retained until all operation and caller leases have returned.
  iree_async_buffer_pool_t* pool = nullptr;
  // Peer used to inject bytes and orderly shutdown without another backend.
  int peer = -1;

 private:
  // Slab storage backing the four receive slots.
  iree_async_slab_t* slab_ = nullptr;
  // Backend registration retained through pool retirement.
  iree_async_region_t* region_ = nullptr;
  // Imported receive descriptor owned through operation retirement.
  iree_async_socket_t* socket_ = nullptr;
};

class IoUringCapabilityDeathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_t* proactor = nullptr;
    iree_status_t status = iree_async_proactor_create_io_uring(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "native io_uring unavailable before restriction";
    }
    IREE_ASSERT_OK(status);
    bool has_provided_rings =
        iree_any_bit_set(iree_async_proactor_query_capabilities(proactor),
                         IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT);
    iree_async_proactor_release(proactor);
    if (!has_provided_rings) {
      GTEST_SKIP()
          << "native provided-buffer rings unavailable before restriction";
    }
  }

  template <typename Fn>
  void Run(Fn function, int error_number = EINVAL,
           bool restrict_setup = false) {
    ASSERT_EXIT(
        {
          RestrictIoUring(restrict_setup, IREE_IORING_REGISTER_PBUF_RING,
                          error_number);
          function();
          _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), "");
  }
};

TEST_F(IoUringCapabilityDeathTest, UnavailableProvidedRingsUseOneShotLeases) {
  for (int error_number : {EINVAL, ENOSYS, EOPNOTSUPP, EPERM, EACCES}) {
    Run([] { ReceiveHarness().RoundTrip(); }, error_number);
  }
}

TEST_F(IoUringCapabilityDeathTest, UnavailableSetupUsesPortableBackend) {
  for (int error_number : {ENOSYS, EPERM}) {
    Run([] { ReceiveHarness(false).RoundTrip(); }, error_number, true);
  }
}

TEST_F(IoUringCapabilityDeathTest, UnexpectedProbeFailureIsNotDowngraded) {
  Run(
      [] {
        iree_async_proactor_t* proactor = nullptr;
        IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                              iree_async_proactor_create_platform(
                                  iree_async_proactor_options_default(),
                                  iree_allocator_system(), &proactor));
        EXPECT_EQ(proactor, nullptr);
        iree_async_proactor_release(proactor);
      },
      ENOMEM);
}

TEST_F(IoUringCapabilityDeathTest,
       CancellationReturnsLeaseAfterNativeRetirement) {
  Run([] {
    ReceiveHarness harness;
    Completion completion;
    iree_async_socket_recv_pool_operation_t receive;
    harness.Initialize(&receive, &completion);
    IREE_CHECK_OK(
        iree_async_proactor_submit_one(harness.proactor, &receive.base));
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 3u);
    IREE_CHECK_OK(iree_async_proactor_cancel(harness.proactor, &receive.base));
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 3u);
    EXPECT_EQ(completion.count, 0);
    harness.PollUntil(completion);
    EXPECT_EQ(completion.code, IREE_STATUS_CANCELLED);
    EXPECT_EQ(receive.lease.release.fn, nullptr);
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 4u);
    harness.RoundTrip();
  });
}

TEST_F(IoUringCapabilityDeathTest,
       LinkedCancellationReturnsEveryReservedLease) {
  Run([] {
    ReceiveHarness harness;
    Completion completion;
    iree_async_socket_recv_pool_operation_t first;
    iree_async_socket_recv_pool_operation_t second;
    harness.Initialize(&first, &completion, IREE_ASYNC_OPERATION_FLAG_LINKED);
    harness.Initialize(&second, &completion);
    iree_async_operation_t* operations[] = {&first.base, &second.base};
    IREE_CHECK_OK(iree_async_proactor_submit(
        harness.proactor, iree_async_operation_list_make(operations, 2)));
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 2u);
    IREE_CHECK_OK(iree_async_proactor_cancel(harness.proactor, &first.base));
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 2u);
    harness.PollUntil(completion, 2);
    EXPECT_EQ(completion.code, IREE_STATUS_CANCELLED);
    EXPECT_EQ(first.lease.release.fn, nullptr);
    EXPECT_EQ(second.lease.release.fn, nullptr);
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 4u);
    harness.RoundTrip();
  });
}

TEST_F(IoUringCapabilityDeathTest, EofTransfersAnEmptyLease) {
  Run([] {
    ReceiveHarness harness;
    EXPECT_EQ(shutdown(harness.peer, SHUT_WR), 0);
    Completion completion;
    iree_async_socket_recv_pool_operation_t receive;
    harness.Initialize(&receive, &completion);
    IREE_CHECK_OK(
        iree_async_proactor_submit_one(harness.proactor, &receive.base));
    harness.PollUntil(completion);
    EXPECT_EQ(completion.code, IREE_STATUS_OK);
    EXPECT_EQ(receive.bytes_received, 0u);
    EXPECT_NE(receive.lease.release.fn, nullptr);
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 3u);
    iree_async_buffer_lease_release(&receive.lease);
  });
}

TEST_F(IoUringCapabilityDeathTest, ExhaustedPoolRejectsTheEntireBatch) {
  Run([] {
    ReceiveHarness harness;
    Completion completion;
    std::array<iree_async_socket_recv_pool_operation_t, 5> receives;
    std::array<iree_async_operation_t*, 5> operations;
    for (size_t i = 0; i < receives.size(); ++i) {
      harness.Initialize(&receives[i], &completion);
      operations[i] = &receives[i].base;
    }
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_async_proactor_submit(harness.proactor,
                                   iree_async_operation_list_make(
                                       operations.data(), operations.size())));
    EXPECT_EQ(completion.count, 0);
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 4u);
    for (auto& receive : receives) {
      EXPECT_EQ(receive.lease.release.fn, nullptr);
      EXPECT_FALSE(receive.base.resources_acquired);
    }
    harness.RoundTrip();
  });
}

TEST_F(IoUringCapabilityDeathTest, FullSubmissionQueueReturnsReservedLease) {
  Run([] {
    ReceiveHarness harness;
    Completion timers_completed;
    std::array<iree_async_timer_operation_t, 8> timers = {};
    std::array<iree_async_operation_t*, 8> operations;
    for (size_t i = 0; i < timers.size(); ++i) {
      iree_async_operation_initialize(&timers[i].base,
                                      IREE_ASYNC_OPERATION_TYPE_TIMER, 0,
                                      Completion::Record, &timers_completed);
      timers[i].deadline_ns = iree_time_now();
      operations[i] = &timers[i].base;
    }
    IREE_CHECK_OK(iree_async_proactor_submit(
        harness.proactor,
        iree_async_operation_list_make(operations.data(), operations.size())));
    Completion completion;
    iree_async_socket_recv_pool_operation_t receive;
    harness.Initialize(&receive, &completion);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_async_proactor_submit_one(harness.proactor, &receive.base));
    EXPECT_EQ(completion.count, 0);
    EXPECT_EQ(receive.lease.release.fn, nullptr);
    EXPECT_FALSE(receive.base.resources_acquired);
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 4u);
    harness.PollUntil(timers_completed, timers.size());
    harness.RoundTrip();
  });
}

TEST_F(IoUringCapabilityDeathTest,
       DeferredReceiveDoesNotReserveBeforeItsTrigger) {
  Run([] {
    ReceiveHarness harness;
    Completion trigger_completed;
    iree_async_timer_operation_t trigger = {};
    iree_async_operation_initialize(&trigger.base,
                                    IREE_ASYNC_OPERATION_TYPE_TIMER,
                                    IREE_ASYNC_OPERATION_FLAG_LINKED,
                                    Completion::Record, &trigger_completed);
    trigger.deadline_ns = iree_time_now();
    Completion receive_completed;
    iree_async_socket_recv_pool_operation_t receive;
    harness.Initialize(&receive, &receive_completed);
    iree_async_operation_t* operations[] = {&trigger.base, &receive.base};
    IREE_CHECK_OK(iree_async_proactor_submit(
        harness.proactor, iree_async_operation_list_make(operations, 2)));
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 4u);
    uint8_t value = 42;
    EXPECT_EQ(send(harness.peer, &value, 1, MSG_NOSIGNAL), 1);
    harness.PollUntil(receive_completed);
    EXPECT_EQ(trigger_completed.count, 1);
    EXPECT_EQ(trigger_completed.code, IREE_STATUS_OK);
    EXPECT_EQ(receive_completed.code, IREE_STATUS_OK);
    EXPECT_EQ(receive.bytes_received, 1u);
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 3u);
    iree_async_buffer_lease_release(&receive.lease);
  });
}

TEST_F(IoUringCapabilityDeathTest,
       UnsupportedMultishotRejectsBeforeReservation) {
  Run([] {
    ReceiveHarness harness;
    Completion completion;
    iree_async_socket_recv_pool_operation_t receive;
    harness.Initialize(&receive, &completion,
                       IREE_ASYNC_OPERATION_FLAG_MULTISHOT);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_async_proactor_submit_one(harness.proactor, &receive.base));
    EXPECT_EQ(completion.count, 0);
    EXPECT_EQ(iree_async_buffer_pool_available(harness.pool), 4u);
  });
}

TEST_F(IoUringCapabilityDeathTest, LaterRegistrationFailureIsNotDowngraded) {
  ASSERT_EXIT(
      {
        iree_async_proactor_t* proactor = nullptr;
        IREE_CHECK_OK(iree_async_proactor_create_io_uring(
            iree_async_proactor_options_default(), iree_allocator_system(),
            &proactor));
        RestrictIoUring(false, IREE_IORING_REGISTER_PBUF_RING, EINVAL);
        auto options = iree_async_slab_options_default();
        options.buffer_size = 16384;
        options.buffer_count = 4;
        iree_async_slab_t* slab = nullptr;
        IREE_CHECK_OK(
            iree_async_slab_create(options, iree_allocator_system(), &slab));
        iree_async_region_t* region = nullptr;
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_INVALID_ARGUMENT,
            iree_async_proactor_register_slab(
                proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));
        EXPECT_EQ(region, nullptr);
        iree_async_region_release(region);
        iree_async_slab_release(slab);
        iree_async_proactor_release(proactor);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(IoUringCapabilityDeathTest, ProbeRetirementFailurePreventsPublication) {
  ASSERT_EXIT(
      {
        RestrictIoUring(false, IREE_IORING_UNREGISTER_PBUF_RING, EPERM);
        iree_async_proactor_t* proactor = nullptr;
        IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                              iree_async_proactor_create_platform(
                                  iree_async_proactor_options_default(),
                                  iree_allocator_system(), &proactor));
        EXPECT_EQ(proactor, nullptr);
        iree_async_proactor_release(proactor);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

}  // namespace
