// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/local_stream.h"

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace iree::async::cts {
namespace {

struct TransferResult {
  // Number of terminal callbacks delivered.
  int calls = 0;
  // Owned terminal status consumed by the test.
  iree_status_t status = iree_ok_status();

  ~TransferResult() { iree_status_free(status); }

  iree_status_t TakeStatus() {
    iree_status_t result = status;
    status = iree_ok_status();
    return result;
  }

  iree_async_local_stream_callback_t callback() {
    return {+[](void* user_data, iree_status_t status) {
              auto* result = static_cast<TransferResult*>(user_data);
              EXPECT_EQ(result->calls++, 0);
              result->status = status;
            },
            this};
  }
};

class LocalStreamTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (!proactor_) {
      return;
    }
#if defined(IREE_PLATFORM_WINDOWS)
    static int ordinal = 0;
    name_ = "iree-local-stream-" + std::to_string(GetCurrentProcessId()) + "-" +
            std::to_string(++ordinal);
    IREE_ASSERT_OK(iree_async_local_stream_pipe_create(
        iree_make_string_view(name_.data(), name_.size()), 1,
        IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE, &channels_[0]));
    IREE_ASSERT_OK(iree_async_local_stream_pipe_open(
        iree_make_string_view(name_.data(), name_.size()), &channels_[1]));
#else
    int sockets[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
    for (int i = 0; i < 2; ++i) {
      channels_[i] = iree_async_primitive_from_fd(sockets[i]);
      ASSERT_EQ(fcntl(sockets[i], F_SETFL, O_NONBLOCK), 0);
      ASSERT_EQ(fcntl(sockets[i], F_SETFD, FD_CLOEXEC), 0);
    }
#endif
    for (int i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_async_local_stream_create(
          proactor_, channels_[i], 3, iree_allocator_system(), &streams_[i]));
    }
#if defined(IREE_PLATFORM_WINDOWS)
    TransferResult accepted;
    IREE_ASSERT_OK(
        iree_async_local_stream_pipe_accept(streams_[0], accepted.callback()));
    PollUntilCondition([&] { return accepted.calls; });
    IREE_ASSERT_OK(accepted.TakeStatus());
#endif
  }

  void TearDown() override {
    for (auto*& stream : streams_) {
      Close(stream);
    }
    for (auto& channel : channels_) {
      iree_async_primitive_close(&channel);
    }
    CtsTestBase<>::TearDown();
  }

  void Close(iree_async_local_stream_t*& stream) {
    if (!stream) {
      return;
    }
    struct Join {
      // Helper reference cleared after callback-owned destruction.
      iree_async_local_stream_t** stream;
      // Completion witness outside the destroyed helper.
      bool done = false;
    } join{&stream};
    IREE_ASSERT_OK(iree_async_local_stream_deactivate(
        stream, {+[](void* user_data) {
                   auto* join = static_cast<Join*>(user_data);
                   iree_async_local_stream_destroy(*join->stream);
                   *join->stream = nullptr;
                   join->done = true;
                 },
                 &join}));
    PollUntilCondition([&] { return join.done; });
  }

  void PollImmediate() {
    iree_status_t status =
        iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_free(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  // Two native endpoints retained independently of their helpers.
  iree_async_primitive_t channels_[2] = {};
  // Two sequential transfer owners on the same real poll loop.
  iree_async_local_stream_t* streams_[2] = {};
#if defined(IREE_PLATFORM_WINDOWS)
  // Unique local pipe name for this fixture.
  std::string name_;
#endif
};

TEST_P(LocalStreamTest, ExactTransferAndSlotReuse) {
  for (int iteration = 0; iteration < 32; ++iteration) {
    std::vector<uint8_t> source(128 * 1024 + iteration, (uint8_t)iteration);
    std::vector<uint8_t> target(source.size());
    TransferResult sent;
    TransferResult received;
    IREE_ASSERT_OK(iree_async_local_stream_receive(
        streams_[0], iree_make_byte_span(target.data(), target.size()), 0,
        nullptr, received.callback()));
    IREE_ASSERT_OK(iree_async_local_stream_send(
        streams_[1], iree_make_const_byte_span(source.data(), source.size()), 0,
        nullptr, sent.callback()));
    EXPECT_EQ(sent.calls, 0);
    EXPECT_EQ(received.calls, 0);
    PollUntilCondition([&] { return sent.calls && received.calls; });
    IREE_ASSERT_OK(sent.TakeStatus());
    IREE_ASSERT_OK(received.TakeStatus());
    EXPECT_EQ(target, source);
    EXPECT_EQ(proactor_->progress_list, nullptr);
  }
}

TEST_P(LocalStreamTest, ReceiveCancellationWithSilentPeer) {
  uint8_t target = 0;
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(&target, 1), 0, nullptr,
      received.callback()));
  PollImmediate();
  EXPECT_EQ(received.calls, 0);
  EXPECT_EQ(proactor_->progress_list, nullptr);
  Close(streams_[0]);
  EXPECT_EQ(received.calls, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, received.TakeStatus());
  EXPECT_EQ(proactor_->progress_list, nullptr);
}

TEST_P(LocalStreamTest, SendCancellationWithSilentPeer) {
  std::vector<uint8_t> source(4 * 1024 * 1024);
  TransferResult sent;
  IREE_ASSERT_OK(iree_async_local_stream_send(
      streams_[0], iree_make_const_byte_span(source.data(), source.size()), 0,
      nullptr, sent.callback()));
  PollImmediate();
  EXPECT_EQ(sent.calls, 0);
  EXPECT_EQ(proactor_->progress_list, nullptr);
  Close(streams_[0]);
  EXPECT_EQ(sent.calls, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, sent.TakeStatus());
}

TEST_P(LocalStreamTest, CancellationBeforeFirstPoll) {
  uint8_t target = 0;
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(&target, 1), 0, nullptr,
      received.callback()));
  Close(streams_[0]);
  EXPECT_EQ(received.calls, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, received.TakeStatus());
}

TEST_P(LocalStreamTest, RejectionDoesNotConsumeTransferSlot) {
  uint8_t target = 0;
  TransferResult received;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_local_stream_receive(
                            streams_[0], iree_make_byte_span(&target, 0), 0,
                            nullptr, received.callback()));
  iree_async_primitive_t handles[4] = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_async_local_stream_receive(
                            streams_[0], iree_make_byte_span(&target, 1), 4,
                            handles, received.callback()));
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(&target, 1), 0, nullptr,
      received.callback()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_async_local_stream_receive(
                            streams_[0], iree_make_byte_span(&target, 1), 0,
                            nullptr, received.callback()));
  EXPECT_EQ(received.calls, 0);
  Close(streams_[0]);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, received.TakeStatus());
}

TEST_P(LocalStreamTest, PeerEofPoisonsTransferAdmission) {
  Close(streams_[1]);
  iree_async_primitive_close(&channels_[1]);
  uint8_t target = 0;
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(&target, 1), 0, nullptr,
      received.callback()));
  PollUntilCondition([&] { return received.calls; });
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, received.TakeStatus());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_async_local_stream_receive(
                            streams_[0], iree_make_byte_span(&target, 1), 0,
                            nullptr, received.callback()));
  EXPECT_EQ(received.calls, 1);
}

TEST_P(LocalStreamTest, TerminalCallbackCanStartNextTransfer) {
  struct Chain {
    // Helper reused immediately after terminal delivery.
    iree_async_local_stream_t* stream;
    // First transfer destination.
    uint8_t first = 0;
    // Second transfer destination.
    uint8_t second = 0;
    // Terminal result for the second execution of the same slot.
    TransferResult result;
  } chain{streams_[0]};
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(&chain.first, 1), 0, nullptr,
      {+[](void* user_data, iree_status_t status) {
         IREE_ASSERT_OK(status);
         auto* chain = static_cast<Chain*>(user_data);
         IREE_ASSERT_OK(iree_async_local_stream_receive(
             chain->stream, iree_make_byte_span(&chain->second, 1), 0, nullptr,
             chain->result.callback()));
       },
       &chain}));
  const uint8_t source[2] = {42, 37};
  TransferResult sent;
  IREE_ASSERT_OK(iree_async_local_stream_send(
      streams_[1], iree_make_const_byte_span(source, sizeof(source)), 0,
      nullptr, sent.callback()));
  PollUntilCondition([&] { return sent.calls && chain.result.calls; });
  IREE_EXPECT_OK(sent.TakeStatus());
  IREE_EXPECT_OK(chain.result.TakeStatus());
  EXPECT_EQ(chain.first, 42);
  EXPECT_EQ(chain.second, 37);
}

#if !defined(IREE_PLATFORM_WINDOWS)

class LocalStreamRightsTest : public LocalStreamTest {
 protected:
  void SetUp() override {
    LocalStreamTest::SetUp();
    if (!proactor_) {
      return;
    }
    for (int i = 0; i < 8; ++i) {
      int descriptors[2];
      ASSERT_EQ(pipe(descriptors), 0);
      reads_[i] = descriptors[0];
      writes_[i] = iree_async_primitive_from_fd(descriptors[1]);
      ASSERT_EQ(fcntl(reads_[i], F_SETFL, O_NONBLOCK), 0);
      ASSERT_EQ(fcntl(reads_[i], F_SETFD, FD_CLOEXEC), 0);
      ASSERT_EQ(fcntl(descriptors[1], F_SETFD, FD_CLOEXEC), 0);
    }
  }

  void TearDown() override {
    LocalStreamTest::TearDown();
    for (auto& descriptor : writes_) {
      iree_async_primitive_close(&descriptor);
    }
    for (int descriptor : reads_) {
      if (descriptor >= 0) {
        close(descriptor);
      }
    }
  }

  void Fragment(const char* bytes, size_t length, size_t first, size_t count) {
    alignas(cmsghdr) char control[CMSG_SPACE(8 * sizeof(int))] = {};
    iovec buffer = {const_cast<char*>(bytes), length};
    msghdr message = {};  // NOLINT(iree-cpp-designated-initializer) --
                          // Assignment sequencing spans intervening work.
    message.msg_iov = &buffer;
    message.msg_iovlen = 1;
    if (count) {
      message.msg_control = control;
      message.msg_controllen = CMSG_SPACE(count * sizeof(int));
      auto* header = CMSG_FIRSTHDR(&message);
      header->cmsg_level = SOL_SOCKET;
      header->cmsg_type = SCM_RIGHTS;
      header->cmsg_len = CMSG_LEN(count * sizeof(int));
      for (size_t i = 0; i < count; ++i) {
        memcpy(CMSG_DATA(header) + i * sizeof(int),
               &writes_[first + i].value.fd, sizeof(int));
      }
    }
    ASSERT_EQ(sendmsg(channels_[1].value.fd, &message, 0), (ssize_t)length);
    PollImmediate();
  }

  void ExpectAllWritersRetired() {
    for (auto& descriptor : writes_) {
      iree_async_primitive_close(&descriptor);
    }
    for (int descriptor : reads_) {
      char byte = 0;
      // A leaked installed write descriptor leaves the read end at EAGAIN,
      // not EOF. No process-wide fd-count heuristic or timeout is involved.
      EXPECT_EQ(read(descriptor, &byte, 1), 0) << errno;
    }
  }

  // Read ends retained as exact ownership witnesses for every exported right.
  int reads_[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  // Locally owned resources exported through the stream.
  iree_async_primitive_t writes_[8] = {};
};

TEST_P(LocalStreamRightsTest, RightsMayArriveOnEveryFragment) {
  uint8_t target[8] = {};
  iree_async_primitive_t received_handles[3] = {};
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(target, sizeof(target)), 3,
      received_handles, received.callback()));
  Fragment("a", 1, 0, 1);
  Fragment("bcd", 3, 0, 0);
  Fragment("efgh", 4, 1, 2);
  PollUntilCondition([&] { return received.calls; });
  IREE_ASSERT_OK(received.TakeStatus());
  EXPECT_EQ(std::string((char*)target, sizeof(target)), "abcdefgh");
  for (int i = 0; i < 3; ++i) {
    EXPECT_NE(fcntl(received_handles[i].value.fd, F_GETFD) & FD_CLOEXEC, 0);
    char value = (char)(i + 1);
    ASSERT_EQ(write(received_handles[i].value.fd, &value, 1), 1);
    char observed = 0;
    ASSERT_EQ(read(reads_[i], &observed, 1), 1);
    EXPECT_EQ(observed, value);
    iree_async_primitive_close(&received_handles[i]);
  }
  ExpectAllWritersRetired();
}

TEST_P(LocalStreamRightsTest, PartialSendAttachesResourcesOnlyOnce) {
  int send_buffer_size = 4096;
  ASSERT_EQ(setsockopt(channels_[1].value.fd, SOL_SOCKET, SO_SNDBUF,
                       &send_buffer_size, sizeof(send_buffer_size)),
            0);
  std::vector<uint8_t> source(1024 * 1024, 0xAB);
  std::vector<uint8_t> target(source.size());
  TransferResult sent;
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_send(
      streams_[1], iree_make_const_byte_span(source.data(), source.size()), 3,
      writes_, sent.callback()));
  PollImmediate();
  EXPECT_EQ(sent.calls, 0);
  iree_async_primitive_t handles[3] = {};
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(target.data(), target.size()), 3,
      handles, received.callback()));
  PollUntilCondition([&] { return sent.calls && received.calls; });
  IREE_ASSERT_OK(sent.TakeStatus());
  IREE_ASSERT_OK(received.TakeStatus());
  EXPECT_EQ(target, source);
  for (auto& handle : handles) {
    iree_async_primitive_close(&handle);
  }
  ExpectAllWritersRetired();
}

TEST_P(LocalStreamRightsTest, OverflowAndTruncationRetireEveryInstalledRight) {
  for (size_t count : {4u, 8u}) {
    uint8_t target = 0;
    iree_async_primitive_t received_handles[3] = {};
    TransferResult received;
    IREE_ASSERT_OK(iree_async_local_stream_receive(
        streams_[0], iree_make_byte_span(&target, 1), 3, received_handles,
        received.callback()));
    Fragment("a", 1, 0, count);
    PollUntilCondition([&] { return received.calls; });
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, received.TakeStatus());
    for (auto handle : received_handles) {
      EXPECT_TRUE(iree_async_primitive_is_none(handle));
    }
    Close(streams_[0]);
    IREE_ASSERT_OK(iree_async_local_stream_create(
        proactor_, channels_[0], 3, iree_allocator_system(), &streams_[0]));
  }
  ExpectAllWritersRetired();
}

TEST_P(LocalStreamRightsTest, LateExcessRightRetiresEarlierBundle) {
  uint8_t target[2] = {};
  iree_async_primitive_t received_handles[3] = {};
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(target, 2), 3, received_handles,
      received.callback()));
  Fragment("a", 1, 0, 3);
  EXPECT_EQ(received.calls, 0);
  Fragment("b", 1, 3, 1);
  PollUntilCondition([&] { return received.calls; });
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, received.TakeStatus());
  for (auto handle : received_handles) {
    EXPECT_TRUE(iree_async_primitive_is_none(handle));
  }
  ExpectAllWritersRetired();
}

TEST_P(LocalStreamRightsTest, MissingResourceRetiresPartialBundle) {
  uint8_t target = 0;
  iree_async_primitive_t received_handles[3] = {};
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(&target, 1), 3, received_handles,
      received.callback()));
  Fragment("a", 1, 0, 2);
  PollUntilCondition([&] { return received.calls; });
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, received.TakeStatus());
  for (auto handle : received_handles) {
    EXPECT_TRUE(iree_async_primitive_is_none(handle));
  }
  ExpectAllWritersRetired();
}

TEST_P(LocalStreamRightsTest, CancelAfterPartialReceiptRetiresBundle) {
  uint8_t target[2] = {};
  iree_async_primitive_t received_handles[3] = {};
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(target, 2), 3, received_handles,
      received.callback()));
  Fragment("a", 1, 0, 3);
  EXPECT_EQ(received.calls, 0);
  Close(streams_[0]);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, received.TakeStatus());
  for (auto handle : received_handles) {
    EXPECT_TRUE(iree_async_primitive_is_none(handle));
  }
  ExpectAllWritersRetired();
}

TEST_P(LocalStreamRightsTest, EofAfterPartialReceiptRetiresBundle) {
  uint8_t target[2] = {};
  iree_async_primitive_t received_handles[3] = {};
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[0], iree_make_byte_span(target, 2), 3, received_handles,
      received.callback()));
  Fragment("a", 1, 0, 3);
  Close(streams_[1]);
  iree_async_primitive_close(&channels_[1]);
  PollUntilCondition([&] { return received.calls; });
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, received.TakeStatus());
  ExpectAllWritersRetired();
}

CTS_REGISTER_TEST_SUITE(LocalStreamRightsTest);

#else

// Raw producer for boundary tests. Its blocking completion wait is fixture
// synchronization; the helper under test still uses its native async observer.
void WritePipeBytes(iree_async_primitive_t channel, const void* data,
                    DWORD length) {
  OVERLAPPED operation = {};
  operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  ASSERT_NE(operation.hEvent, nullptr);
  DWORD written = 0;
  if (!WriteFile((HANDLE)channel.value.win32_handle, data, length, &written,
                 &operation)) {
    ASSERT_EQ(GetLastError(), ERROR_IO_PENDING);
    ASSERT_TRUE(GetOverlappedResult((HANDLE)channel.value.win32_handle,
                                    &operation, &written, TRUE));
  }
  EXPECT_EQ(written, length);
  EXPECT_TRUE(CloseHandle(operation.hEvent));
}

TEST_P(LocalStreamTest, CompletedNativeReadWinsDeactivation) {
  uint8_t target = 0;
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[1], iree_make_byte_span(&target, 1), 0, nullptr,
      received.callback()));
  PollImmediate();
  EXPECT_EQ(received.calls, 0);
  ASSERT_EQ(proactor_->progress_list, nullptr);
  const uint8_t source = 42;
  WritePipeBytes(channels_[0], &source, 1);
  // The pipe flush joins the pending native read, without dispatching the
  // helper's observer. Poll then collects its terminal result before the next
  // owner progress turn publishes it to the application.
  ASSERT_TRUE(FlushFileBuffers((HANDLE)channels_[0].value.win32_handle));
  PollUntilCondition([&] { return proactor_->progress_list != nullptr; });
  EXPECT_EQ(received.calls, 0);
  Close(streams_[1]);
  EXPECT_EQ(received.calls, 1);
  IREE_EXPECT_OK(received.TakeStatus());
  EXPECT_EQ(target, source);
}

TEST_P(LocalStreamTest, PartialImportFailureClosesEarlierResources) {
  HANDLE event = CreateEventW(nullptr, FALSE, TRUE, nullptr);
  ASSERT_NE(event, nullptr);
  // The second source handle is invalid, after a valid first duplication.
  uint64_t prefix[3] = {(uint64_t)(uintptr_t)event, 0,
                        (uint64_t)(uintptr_t)event};
  DWORD before = 0;
  ASSERT_TRUE(GetProcessHandleCount(GetCurrentProcess(), &before));
  uint8_t target = 0;
  iree_async_primitive_t handles[3] = {};
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[1], iree_make_byte_span(&target, 1), 3, handles,
      received.callback()));
  WritePipeBytes(channels_[0], prefix, sizeof(prefix));
  PollUntilCondition([&] { return received.calls; });
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, received.TakeStatus());
  for (auto handle : handles) {
    EXPECT_TRUE(iree_async_primitive_is_none(handle));
  }
  DWORD after = 0;
  ASSERT_TRUE(GetProcessHandleCount(GetCurrentProcess(), &after));
  EXPECT_EQ(after, before);
  EXPECT_TRUE(CloseHandle(event));
}

TEST_P(LocalStreamTest, CancelAfterImportWhileBodyIsPendingClosesResources) {
  HANDLE event = CreateEventW(nullptr, FALSE, TRUE, nullptr);
  ASSERT_NE(event, nullptr);
  uint64_t prefix[3] = {(uint64_t)(uintptr_t)event, (uint64_t)(uintptr_t)event,
                        (uint64_t)(uintptr_t)event};
  uint8_t target = 0;
  iree_async_primitive_t handles[3] = {};
  TransferResult received;
  DWORD before = 0;
  ASSERT_TRUE(GetProcessHandleCount(GetCurrentProcess(), &before));
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[1], iree_make_byte_span(&target, 1), 3, handles,
      received.callback()));
  WritePipeBytes(channels_[0], prefix, sizeof(prefix));
  // Observe ownership only, without using the tentative imports. No body byte
  // is sent, so the next native read remains pending until local cancellation.
  PollUntilCondition([&] {
    return !iree_async_primitive_is_none(handles[2]) &&
           proactor_->progress_list == nullptr;
  });
  EXPECT_EQ(received.calls, 0);
  bool deactivated = false;
  IREE_ASSERT_OK(iree_async_local_stream_deactivate(
      streams_[1],
      {+[](void* user_data) { *static_cast<bool*>(user_data) = true; },
       &deactivated}));
  PollUntilCondition([&] { return deactivated; });
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, received.TakeStatus());
  for (auto handle : handles) {
    EXPECT_TRUE(iree_async_primitive_is_none(handle));
  }
  // Deactivation removed the persistent wait packet but retained the helper's
  // event and peer process until destroy.
  DWORD after = 0;
  ASSERT_TRUE(GetProcessHandleCount(GetCurrentProcess(), &after));
  EXPECT_EQ(after + 1, before);
  iree_async_local_stream_destroy(streams_[1]);
  streams_[1] = nullptr;
  EXPECT_TRUE(CloseHandle(event));
}

TEST_P(LocalStreamTest, BusyPipeOpenIsSynchronousRejection) {
  iree_async_primitive_t channel = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_async_local_stream_pipe_open(
          iree_make_string_view(name_.data(), name_.size()), &channel));
  EXPECT_TRUE(iree_async_primitive_is_none(channel));
}

TEST_P(LocalStreamTest, PipeAcceptCancellationWithoutAClient) {
  std::string name = name_ + "-silent";
  iree_async_primitive_t channel = {};
  IREE_ASSERT_OK(iree_async_local_stream_pipe_create(
      iree_make_string_view(name.data(), name.size()), 1,
      IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE, &channel));
  iree_async_local_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_async_local_stream_create(
      proactor_, channel, 0, iree_allocator_system(), &stream));
  TransferResult accepted;
  IREE_ASSERT_OK(
      iree_async_local_stream_pipe_accept(stream, accepted.callback()));
  PollImmediate();
  EXPECT_EQ(accepted.calls, 0);
  EXPECT_EQ(proactor_->progress_list, nullptr);
  Close(stream);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, accepted.TakeStatus());
  iree_async_primitive_close(&channel);
}

TEST_P(LocalStreamTest, ImportedEventsRemainUsableAfterSourceClose) {
  iree_async_primitive_t source[3] = {};
  for (auto& handle : source) {
    HANDLE event = CreateEventW(nullptr, FALSE, TRUE, nullptr);
    ASSERT_NE(event, nullptr);
    handle = iree_async_primitive_from_win32_handle((uintptr_t)event);
  }
  iree_async_primitive_t target[3] = {};
  const uint8_t byte = 42;
  uint8_t received_byte = 0;
  TransferResult sent;
  TransferResult received;
  IREE_ASSERT_OK(iree_async_local_stream_send(
      streams_[0], iree_make_const_byte_span(&byte, 1), 3, source,
      sent.callback()));
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      streams_[1], iree_make_byte_span(&received_byte, 1), 3, target,
      received.callback()));
  PollUntilCondition([&] { return sent.calls && received.calls; });
  IREE_ASSERT_OK(sent.TakeStatus());
  IREE_ASSERT_OK(received.TakeStatus());
  EXPECT_EQ(received_byte, byte);
  for (int i = 0; i < 3; ++i) {
    iree_async_primitive_close(&source[i]);
    HANDLE event = (HANDLE)target[i].value.win32_handle;
    DWORD flags = 0;
    ASSERT_TRUE(GetHandleInformation(event, &flags));
    EXPECT_EQ(flags & HANDLE_FLAG_INHERIT, 0u);
    EXPECT_EQ(WaitForSingleObject(event, 0), WAIT_OBJECT_0);
    iree_async_primitive_close(&target[i]);
  }
}

#endif

#if defined(IREE_PLATFORM_WINDOWS)

class LocalStreamUnavailableTest : public CtsTestBase<> {};

TEST_P(LocalStreamUnavailableTest, RejectsSetupWithoutPersistentObservation) {
  std::string name =
      "iree-local-stream-unavailable-" + std::to_string(GetCurrentProcessId());
  iree_async_primitive_t channel = {};
  IREE_ASSERT_OK(iree_async_local_stream_pipe_create(
      iree_make_string_view(name.data(), name.size()), 1,
      IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE, &channel));
  DWORD before = 0;
  ASSERT_TRUE(GetProcessHandleCount(GetCurrentProcess(), &before));
  iree_async_local_stream_t* stream = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_async_local_stream_create(proactor_, channel, 3,
                                     iree_allocator_system(), &stream));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(proactor_->progress_list, nullptr);
  DWORD after = 0;
  ASSERT_TRUE(GetProcessHandleCount(GetCurrentProcess(), &after));
  EXPECT_EQ(after, before);
  iree_async_primitive_close(&channel);
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(LocalStreamTest, {"wait_completion_packet"},
                                  {});
CTS_REGISTER_TEST_SUITE_WITH_TAGS(LocalStreamUnavailableTest, {},
                                  {"wait_completion_packet"});

#else
CTS_REGISTER_TEST_SUITE(LocalStreamTest);
#endif

}  // namespace
}  // namespace iree::async::cts
