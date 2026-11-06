// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/handshake.h"

#include <array>
#include <functional>
#include <string>

#include "iree/async/proactor_platform.h"
#if !defined(IREE_PLATFORM_WINDOWS)
#include <sys/socket.h>

#include "iree/async/platform/posix/api.h"
#endif
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

class ShmHandshakeTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (!iree_async_notification_native_is_supported()) {
      GTEST_SKIP() << "native shared notification waits are unavailable";
    }
#if !defined(IREE_PLATFORM_WINDOWS)
    if (GetParam()) {
      IREE_ASSERT_OK(iree_async_proactor_create_posix(
          iree_async_proactor_options_default(), iree_allocator_system(),
          &proactor_));
    } else
#endif
    {
      IREE_ASSERT_OK(iree_async_proactor_create_platform(
          iree_async_proactor_options_default(), iree_allocator_system(),
          &proactor_));
    }
    IREE_ASSERT_OK(
        iree_net_shm_region_calculate_layout({1, 3, 4097}, &offered_));
    IREE_ASSERT_OK(
        iree_net_shm_region_calculate_layout({2, 4, 8192}, &limits_));
    IREE_ASSERT_OK(iree_net_shm_storage_create(
        &offered_, iree_allocator_system(), &storage_));
#if defined(IREE_PLATFORM_WINDOWS)
    std::string name = iree::testing::MakeTempFilePath("shm-handshake");
    name = name.substr(name.find_last_of("\\/") + 1);
    auto address = iree_make_string_view(name.data(), name.size());
    IREE_ASSERT_OK(iree_async_local_stream_pipe_create(
        address, 1, IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE,
        &channels_[0].pipe));
    IREE_ASSERT_OK(
        iree_async_local_stream_pipe_open(address, &channels_[1].pipe));
    for (auto& channel : channels_) {
      IREE_ASSERT_OK(iree_async_local_stream_allocate(
          proactor_, channel.pipe, IREE_NET_SHM_STORAGE_HANDLE_COUNT,
          iree_allocator_system(), &channel.stream));
    }
    IREE_ASSERT_OK(iree_async_local_stream_pipe_accept(channels_[0].stream,
                                                       {Transferred, this}));
    ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
#else
    int descriptors[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors), 0);
    for (int i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_async_socket_import(
          proactor_, iree_async_primitive_from_fd(descriptors[i]),
          IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM, IREE_ASYNC_SOCKET_FLAG_NONE,
          &channels_[i].socket));
      IREE_ASSERT_OK(iree_async_local_stream_allocate(
          proactor_, channels_[i].socket->primitive,
          IREE_NET_SHM_STORAGE_HANDLE_COUNT, iree_allocator_system(),
          &channels_[i].stream));
    }
#endif
  }

  void PollUntil(const std::function<bool()>& condition) {
    while (!condition()) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
  }

  static void Transferred(void* user_data, iree_status_t status) {
    auto& self = *static_cast<ShmHandshakeTest*>(user_data);
    EXPECT_FALSE(self.transfer_done_);
    self.transfer_done_ = true;
    self.transfer_code_ = iree_status_code(status);
    iree_status_free(status);
  }

  void AwaitTransfer() {
    ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return transfer_done_; }));
    transfer_done_ = false;
    ASSERT_EQ(transfer_code_, IREE_STATUS_OK);
  }

  void Begin(iree_allocator_t host_allocator = iree_allocator_system()) {
    pending_ = true;
    iree_net_shm_handshake_begin(
        &handshake_, IREE_NET_SHM_HANDSHAKE_ROLE_CLIENT, proactor_, &limits_,
        &options_, &channels_[1],
        {+[](void* user_data, iree_status_t status,
             iree_net_connection_t* connection) {
           auto& self = *static_cast<ShmHandshakeTest*>(user_data);
           EXPECT_TRUE(self.pending_);
           self.pending_ = false;
           ++self.completion_count_;
           self.completion_code_ = iree_status_code(status);
           self.connection_ = connection;
           iree_status_free(status);
         },
         this},
        host_allocator);
  }

  void OfferAndAwaitAccept() {
    std::array<uint8_t, IREE_NET_SHM_BOOTSTRAP_OFFER_SIZE> offer;
    iree_net_shm_bootstrap_encode_offer(&offered_, offer.data());
    iree_async_primitive_t handles[IREE_NET_SHM_STORAGE_HANDLE_COUNT];
    iree_net_shm_storage_export(storage_, handles);
    IREE_ASSERT_OK(iree_async_local_stream_send(
        channels_[0].stream,
        iree_make_const_byte_span(offer.data(), offer.size()),
        IREE_NET_SHM_STORAGE_HANDLE_COUNT, handles, {Transferred, this}));
    ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
    std::array<uint8_t, IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE> accept;
    IREE_ASSERT_OK(iree_async_local_stream_receive(
        channels_[0].stream, iree_make_byte_span(accept.data(), accept.size()),
        0, {Transferred, this}, nullptr));
    ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
    IREE_ASSERT_OK(iree_net_shm_bootstrap_decode_ack(
        iree_make_const_byte_span(accept.data(), accept.size()),
        IREE_NET_SHM_BOOTSTRAP_TYPE_ACCEPT));
    ASSERT_NO_FATAL_FAILURE(PollUntil([&] {
      return handshake_.phase ==
             IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_VALIDATE_READY;
    }));
  }

  void Confirm() {
    std::array<uint8_t, IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE> ready;
    iree_net_shm_bootstrap_encode_ack(IREE_NET_SHM_BOOTSTRAP_TYPE_READY,
                                      ready.data());
    IREE_ASSERT_OK(iree_async_local_stream_send(
        channels_[0].stream,
        iree_make_const_byte_span(ready.data(), ready.size()), 0, nullptr,
        {Transferred, this}));
    ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
    ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return !pending_; }));
    EXPECT_EQ(completion_count_, 1);
  }

  void ExpectTentativeImports() {
    EXPECT_TRUE(pending_);
    EXPECT_EQ(completion_count_, 0);
    EXPECT_EQ(handshake_.storage, nullptr);
    EXPECT_EQ(handshake_.connection, nullptr);
    for (auto handle : handshake_.handles) {
      EXPECT_FALSE(iree_async_primitive_is_none(handle));
    }
  }

  void ExpectReleasedImports() {
    EXPECT_EQ(handshake_.storage, nullptr);
    EXPECT_EQ(handshake_.connection, nullptr);
    for (auto handle : handshake_.handles) {
      EXPECT_TRUE(iree_async_primitive_is_none(handle));
    }
  }

  void TearDown() override {
    if (pending_) {
      iree_net_shm_handshake_cancel(&handshake_);
      PollUntil([&] { return !pending_; });
    }
    if (connection_) {
      bool drained = false;
      iree_net_connection_deactivate(
          connection_,
          {+[](void* data) { *static_cast<bool*>(data) = true; }, &drained});
      PollUntil([&] { return drained; });
      iree_net_connection_release(connection_);
    }
    for (auto& channel : channels_) {
      if (channel.stream) {
        bool drained = false;
        IREE_ASSERT_OK(iree_async_local_stream_deactivate(
            channel.stream,
            {+[](void* data) { *static_cast<bool*>(data) = true; }, &drained}));
        PollUntil([&] { return drained; });
      }
      iree_net_shm_connection_channel_deinitialize(&channel);
    }
    iree_net_shm_storage_release(storage_);
    iree_async_proactor_release(proactor_);
  }

  // Executor for native transfers and the production handshake state machine.
  iree_async_proactor_t* proactor_ = nullptr;
  // Raw server and client channels; the handshake consumes the client channel.
  iree_net_shm_connection_channel_t channels_[2] = {};
  // Actual server geometry, deliberately smaller than the client limits.
  iree_net_shm_region_layout_t offered_ = {};
  // Client resource policy, not the geometry to use for imported storage.
  iree_net_shm_region_layout_t limits_ = {};
  // Client endpoint admission configuration, live through the handshake.
  iree_net_shm_carrier_options_t options_ =
      iree_net_shm_carrier_options_default();
  // Server mapping and wakes retained through confirmation and test cleanup.
  iree_net_shm_storage_t* storage_ = nullptr;
  // Production state whose tentative-resource boundary is under test.
  iree_net_shm_handshake_t handshake_ = {};
  // Whether the handshake still owes terminal completion.
  bool pending_ = false;
  // Number of terminal handshake callbacks.
  int completion_count_ = 0;
  // Handshake result without status ownership.
  iree_status_code_t completion_code_ = IREE_STATUS_UNKNOWN;
  // Owned published connection, deactivated before fixture release.
  iree_net_connection_t* connection_ = nullptr;
  // Completion predicate for one raw-server transfer at a time.
  bool transfer_done_ = false;
  // Raw transfer result without status ownership.
  iree_status_code_t transfer_code_ = IREE_STATUS_UNKNOWN;
};

TEST_P(ShmHandshakeTest, ConstructsConnectionOnlyAfterConfirmation) {
  Begin();
  ASSERT_NO_FATAL_FAILURE(OfferAndAwaitAccept());
  ExpectTentativeImports();
  ASSERT_NO_FATAL_FAILURE(Confirm());
  ASSERT_EQ(completion_code_, IREE_STATUS_OK);
  ASSERT_NE(connection_, nullptr);
  ExpectReleasedImports();
  EXPECT_EQ(iree_net_connection_max_endpoint_count(connection_),
            offered_.options.endpoint_count);
  iree_net_message_endpoint_t endpoint = {};
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      connection_, {+[](void* data, iree_status_t status,
                        iree_net_message_endpoint_t value) {
                      IREE_EXPECT_OK(status);
                      *static_cast<iree_net_message_endpoint_t*>(data) = value;
                    },
                    &endpoint}));
  ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return endpoint.self != nullptr; }));
}

TEST_P(ShmHandshakeTest, CancellationDiscardsUnusedImportsWithoutConfirmation) {
  Begin();
  ASSERT_NO_FATAL_FAILURE(OfferAndAwaitAccept());
  ExpectTentativeImports();
  iree_net_shm_handshake_cancel(&handshake_);
  ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return !pending_; }));
  EXPECT_EQ(completion_count_, 1);
  EXPECT_EQ(completion_code_, IREE_STATUS_CANCELLED);
  EXPECT_EQ(connection_, nullptr);
  ExpectReleasedImports();
}

TEST_P(ShmHandshakeTest, ConstructionFailureAfterConfirmationClosesChannel) {
  int allocation_count = 0;
  Begin(
      {&allocation_count, +[](void* user_data, iree_allocator_command_t command,
                              const void*, void** inout_pointer) {
         if (command == IREE_ALLOCATOR_COMMAND_FREE) {
           return iree_ok_status();
         }
         ++*static_cast<int*>(user_data);
         *inout_pointer = nullptr;
         return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                 "handshake allocation failure");
       }});
  ASSERT_NO_FATAL_FAILURE(OfferAndAwaitAccept());
  ExpectTentativeImports();
  EXPECT_EQ(allocation_count, 0);
  ASSERT_NO_FATAL_FAILURE(Confirm());
  EXPECT_EQ(allocation_count, 1);
  EXPECT_EQ(completion_code_, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(connection_, nullptr);
  ExpectReleasedImports();
  uint8_t byte = 0;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      channels_[0].stream, iree_make_byte_span(&byte, sizeof(byte)), 0,
      {Transferred, this}, nullptr));
  ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return transfer_done_; }));
  EXPECT_NE(transfer_code_, IREE_STATUS_OK);
}

#if defined(IREE_PLATFORM_WINDOWS)
INSTANTIATE_TEST_SUITE_P(Platform, ShmHandshakeTest, ::testing::Values(false));
#else
INSTANTIATE_TEST_SUITE_P(PlatformAndPosix, ShmHandshakeTest,
                         ::testing::Values(false, true));
#endif

}  // namespace
