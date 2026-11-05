// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/factory.h"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "iree/async/operations/net.h"
#include "iree/async/proactor_platform.h"
#if !defined(IREE_PLATFORM_WINDOWS)
#include <unistd.h>

#include "iree/async/platform/posix/api.h"
#endif
#include "iree/net/carrier/shm/bootstrap.h"
#include "iree/net/carrier/shm/connection.h"
#include "iree/net/carrier/shm/storage.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

TEST(ShmFactoryAvailabilityTest, ConstructionMatchesNativeSupport) {
  iree_net_transport_factory_t* factory = nullptr;
  iree_status_t status =
      iree_net_shm_factory_create(nullptr, iree_allocator_system(), &factory);
  if (iree_async_notification_native_is_supported()) {
    IREE_EXPECT_OK(status);
  } else {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
    EXPECT_EQ(factory, nullptr);
  }
  iree_net_transport_factory_release(factory);
}

struct ConnectResult {
  // Caller-owned cancellation binding through terminal completion.
  iree_net_transport_connect_operation_t operation;
  // Whether an admitted attempt still owes its terminal callback.
  bool pending = false;
  // Number of terminal callbacks, including erroneous duplicate delivery.
  int count = 0;
  // Terminal result without status ownership.
  iree_status_code_t code = IREE_STATUS_UNKNOWN;
  // Owned result, drained by the fixture before release.
  iree_net_connection_t* connection = nullptr;

  ConnectResult() {
    iree_net_transport_connect_operation_initialize(&operation);
  }
  ~ConnectResult() {
    iree_net_transport_connect_operation_deinitialize(&operation);
  }
  static void Complete(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto& self = *static_cast<ConnectResult*>(user_data);
    EXPECT_TRUE(self.pending);
    self.pending = false;
    ++self.count;
    self.code = iree_status_code(status);
    self.connection = connection;
    iree_status_free(status);
  }
};

#if defined(IREE_PLATFORM_WINDOWS)
TEST(ShmFactoryAvailabilityTest, MissingPipeMonitoringRejectsAdmission) {
  auto options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_platform(
      options, iree_allocator_system(), &proactor));
  iree_net_transport_factory_t* factory = nullptr;
  IREE_ASSERT_OK(
      iree_net_shm_factory_create(nullptr, iree_allocator_system(), &factory));
  std::string name = iree::testing::MakeTempFilePath("shm-unavailable");
  name = name.substr(name.find_last_of("\\/") + 1);
  auto address = iree_make_string_view(name.data(), name.size());
  int accepts = 0;
  iree_net_listener_t* listener = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_transport_factory_create_listener(
                            factory, address, proactor, nullptr,
                            {+[](void* data, iree_status_t status,
                                 iree_net_connection_t* connection) {
                               ++*static_cast<int*>(data);
                               EXPECT_EQ(connection, nullptr);
                               iree_status_free(status);
                             },
                             &accepts},
                            iree_allocator_system(), &listener));
  EXPECT_EQ(listener, nullptr);
  if (listener) {
    bool stopped = false;
    IREE_ASSERT_OK(iree_net_listener_stop(
        listener,
        {+[](void* data) { *static_cast<bool*>(data) = true; }, &stopped}));
    while (!stopped) {
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
    iree_net_listener_free(listener);
  }
  ConnectResult result;
  iree_status_t status = iree_net_transport_factory_connect(
      factory, address, proactor, nullptr, {ConnectResult::Complete, &result},
      &result.operation);
  result.pending = iree_status_is_ok(status);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
  while (result.pending) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(result.count, 0);
  EXPECT_EQ(accepts, 0);
  iree_net_transport_factory_release(factory);
  iree_async_proactor_release(proactor);
}
#endif  // IREE_PLATFORM_WINDOWS

class ShmFactoryTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (!iree_async_notification_native_is_supported()) {
      GTEST_SKIP();
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
    options_.region = {2, 4, 4097};
    options_.max_pending_connections = 2;
    IREE_ASSERT_OK(iree_net_shm_factory_create(
        &options_, iree_allocator_system(), &factory_));
    address_ = iree::testing::MakeTempFilePath("shm-factory");
    address_ = address_.substr(address_.find_last_of("\\/") + 1);
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
    address_ = "@" + address_;
#endif
  }

  void PollUntil(const std::function<bool()>& condition) {
    while (!condition()) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
  }

  static void Stopped(void* user_data) {
    auto& self = *static_cast<ShmFactoryTest*>(user_data);
    EXPECT_FALSE(self.accept_callback_active_);
    iree_net_listener_free(self.listener_);
    self.listener_ = nullptr;
  }

  void Stop() {
    if (listener_ && !stop_requested_) {
      IREE_ASSERT_OK(iree_net_listener_stop(listener_, {Stopped, this}));
      stop_requested_ = true;
    }
  }

  static void Accepted(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto& self = *static_cast<ShmFactoryTest*>(user_data);
    self.accept_callback_active_ = true;
    if (iree_status_is_ok(status)) {
      EXPECT_NE(connection, nullptr);
      self.accepted_.push_back(connection);
      if (self.stop_on_accept_) {
        self.Stop();
      }
    } else {
      EXPECT_EQ(connection, nullptr);
      self.accept_errors_.push_back(iree_status_code(status));
    }
    iree_status_free(status);
    self.accept_callback_active_ = false;
  }

  void Listen() {
    stop_requested_ = false;
    IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
        factory_, Address(), proactor_, nullptr, {Accepted, this},
        iree_allocator_system(), &listener_));
  }

  ConnectResult* Connect(iree_net_transport_factory_t* factory = nullptr) {
    attempts_.push_back(std::make_unique<ConnectResult>());
    auto* result = attempts_.back().get();
    iree_status_t status = iree_net_transport_factory_connect(
        factory ? factory : factory_, Address(), proactor_, nullptr,
        {ConnectResult::Complete, result}, &result->operation);
    result->pending = iree_status_is_ok(status);
    IREE_EXPECT_OK(status);
    return result;
  }

  void DrainConnection(iree_net_connection_t*& connection) {
    if (!connection) {
      return;
    }
    bool drained = false;
    iree_net_connection_deactivate(
        connection,
        {+[](void* data) { *static_cast<bool*>(data) = true; }, &drained});
    ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return drained; }));
    iree_net_connection_release(connection);
    connection = nullptr;
  }

  void TearDown() override {
    for (auto& attempt : attempts_) {
      if (attempt->pending) {
        iree_net_transport_connect_operation_cancel(&attempt->operation);
      }
    }
    Stop();
    for (auto& attempt : attempts_) {
      PollUntil([&] { return !attempt->pending; });
      DrainConnection(attempt->connection);
    }
    PollUntil([&] { return !listener_; });
    for (auto*& connection : accepted_) {
      DrainConnection(connection);
    }
    if (raw_.stream) {
      bool drained = false;
      IREE_ASSERT_OK(iree_async_local_stream_deactivate(
          raw_.stream,
          {+[](void* data) { *static_cast<bool*>(data) = true; }, &drained}));
      ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return drained; }));
    }
    iree_net_shm_connection_channel_deinitialize(&raw_);
#if !defined(IREE_PLATFORM_WINDOWS)
    iree_async_socket_release(raw_listener_);
#endif
    for (auto& handle : received_handles_) {
      iree_async_primitive_close(&handle);
    }
    iree_net_shm_storage_release(raw_storage_);
    iree_net_transport_factory_release(factory_);
    iree_async_proactor_release(proactor_);
  }

  iree_string_view_t Address() {
    return iree_make_string_view(address_.data(), address_.size());
  }

  static void Transferred(void* user_data, iree_status_t status) {
    auto& self = *static_cast<ShmFactoryTest*>(user_data);
    self.transfer_code_ = iree_status_code(status);
    self.transfer_done_ = true;
    iree_status_free(status);
  }

  void AwaitTransfer() {
    ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return transfer_done_; }));
    ASSERT_EQ(transfer_code_, IREE_STATUS_OK);
    transfer_done_ = false;
  }

  void MakeRawStream() {
#if defined(IREE_PLATFORM_WINDOWS)
    auto primitive = raw_.pipe;
#else
    auto primitive = raw_.socket->primitive;
#endif
    IREE_ASSERT_OK(iree_async_local_stream_allocate(
        proactor_, primitive, IREE_NET_SHM_STORAGE_HANDLE_COUNT,
        iree_allocator_system(), &raw_.stream));
  }

  void ConnectRawClient() {
#if defined(IREE_PLATFORM_WINDOWS)
    IREE_ASSERT_OK(iree_async_local_stream_pipe_open(Address(), &raw_.pipe));
#else
    IREE_ASSERT_OK(
        iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                                 IREE_ASYNC_SOCKET_OPTION_NONE, &raw_.socket));
    iree_async_socket_connect_operation_t connect = {};
    iree_async_operation_initialize(
        &connect.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        +[](void* data, iree_async_operation_t*, iree_status_t status,
            iree_async_completion_flags_t) { Transferred(data, status); },
        this);
    connect.socket = raw_.socket;
    IREE_ASSERT_OK(iree_async_address_from_unix(Address(), &connect.address));
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect.base));
    ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
#endif
    ASSERT_NO_FATAL_FAILURE(MakeRawStream());
  }

  void AcceptRawServer(ConnectResult** out_result) {
#if defined(IREE_PLATFORM_WINDOWS)
    IREE_ASSERT_OK(iree_async_local_stream_pipe_create(
        Address(), 1, IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE,
        &raw_.pipe));
    ASSERT_NO_FATAL_FAILURE(MakeRawStream());
    IREE_ASSERT_OK(
        iree_async_local_stream_pipe_accept(raw_.stream, {Transferred, this}));
    *out_result = Connect();
    ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
#else
    iree_async_address_t native_address;
    IREE_ASSERT_OK(iree_async_address_from_unix(Address(), &native_address));
    IREE_ASSERT_OK(iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
        IREE_ASYNC_SOCKET_OPTION_NONE, &raw_listener_));
    IREE_ASSERT_OK(iree_async_socket_bind(raw_listener_, &native_address));
    IREE_ASSERT_OK(iree_async_socket_listen(raw_listener_, 1));
    iree_async_socket_accept_operation_t accept = {};
    iree_async_operation_initialize(
        &accept.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        +[](void* data, iree_async_operation_t* base, iree_status_t status,
            iree_async_completion_flags_t) {
          auto& self = *static_cast<ShmFactoryTest*>(data);
          auto* accept =
              reinterpret_cast<iree_async_socket_accept_operation_t*>(base);
          self.raw_.socket = accept->accepted_socket;
          accept->accepted_socket = nullptr;
          Transferred(data, status);
        },
        this);
    accept.listen_socket = raw_listener_;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept.base));
    *out_result = Connect();
    ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
    ASSERT_NO_FATAL_FAILURE(MakeRawStream());
#if !defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_ANDROID)
    ASSERT_EQ(unlink(address_.c_str()), 0);
#endif
#endif
  }

  // Fixture poll owner, exercised through native and portable implementations.
  iree_async_proactor_t* proactor_ = nullptr;
  // Small geometry keeps listener concurrency distinct from storage pressure.
  iree_net_shm_factory_options_t options_ =
      iree_net_shm_factory_options_default();
  // Owned generic factory; established work does not require its lifetime.
  iree_net_transport_factory_t* factory_ = nullptr;
  // Listener freed in its stop callback to verify callback lifetime fencing.
  iree_net_listener_t* listener_ = nullptr;
  // Unique short local address, independent of test output directory length.
  std::string address_;
  // Whether stop has been admitted for the current listener.
  bool stop_requested_ = false;
  // Whether successful acceptance should request reentrant stop.
  bool stop_on_accept_ = false;
  // Guards the callback-body lifetime against nested stop completion.
  bool accept_callback_active_ = false;
  // Stable caller-owned cancellation records through terminal callbacks.
  std::vector<std::unique_ptr<ConnectResult>> attempts_;
  // Connections owned by successful accept callbacks.
  std::vector<iree_net_connection_t*> accepted_;
  // Peer/setup failures delivered by the listener.
  std::vector<iree_status_code_t> accept_errors_;
  // Real protocol peer deliberately held at an import-acknowledgement boundary.
  iree_net_shm_connection_channel_t raw_ = {};
#if !defined(IREE_PLATFORM_WINDOWS)
  // Owned listening socket for the raw protocol server.
  iree_async_socket_t* raw_listener_ = nullptr;
#endif
  // Server resource owner kept alive until the raw peer sees ACCEPT.
  iree_net_shm_storage_t* raw_storage_ = nullptr;
  // Native imports owned by a raw client until fixture cleanup.
  iree_async_primitive_t received_handles_[IREE_NET_SHM_STORAGE_HANDLE_COUNT] =
      {};
  // Terminal callback predicate for sequential raw peer operations.
  bool transfer_done_ = false;
  // Terminal transfer result without status ownership.
  iree_status_code_t transfer_code_ = IREE_STATUS_UNKNOWN;
};

TEST_P(ShmFactoryTest, ListenerOwnsNameUntilStop) {
  ASSERT_NO_FATAL_FAILURE(Listen());
  iree_net_listener_t* duplicate = nullptr;
  IREE_EXPECT_NOT_OK(iree_net_transport_factory_create_listener(
      factory_, Address(), proactor_, nullptr, {Accepted, this},
      iree_allocator_system(), &duplicate));
  EXPECT_EQ(duplicate, nullptr);
  ASSERT_NO_FATAL_FAILURE(Stop());
  ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return !listener_; }));
  ASSERT_NO_FATAL_FAILURE(Listen());
  auto* result = Connect();
  ASSERT_TRUE(result->pending);
  ASSERT_NO_FATAL_FAILURE(
      PollUntil([&] { return !result->pending && accepted_.size() == 1; }));
  EXPECT_EQ(result->code, IREE_STATUS_OK);
}

TEST_P(ShmFactoryTest, SetupCapacityDoesNotLimitLiveConnections) {
  ASSERT_NO_FATAL_FAILURE(Listen());
  for (int batch = 0; batch < 4; ++batch) {
    auto* first = Connect();
    auto* second = Connect();
    ASSERT_TRUE(first->pending);
    ASSERT_TRUE(second->pending);
    ASSERT_NO_FATAL_FAILURE(PollUntil([&] {
      return !first->pending && !second->pending &&
             accepted_.size() == static_cast<size_t>(2 * (batch + 1));
    }));
    ASSERT_EQ(first->code, IREE_STATUS_OK);
    ASSERT_EQ(second->code, IREE_STATUS_OK);
  }
  EXPECT_TRUE(accept_errors_.empty());
  EXPECT_EQ(accepted_.size(), 8u);
}

TEST_P(ShmFactoryTest, StopFromAcceptAndReleaseFactoryBeforePublication) {
  ASSERT_NO_FATAL_FAILURE(Listen());
  stop_on_accept_ = true;
  auto* result = Connect();
  ASSERT_TRUE(result->pending);
  iree_net_transport_factory_release(factory_);
  factory_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      PollUntil([&] { return !listener_ && !result->pending; }));
  ASSERT_EQ(result->code, IREE_STATUS_OK);
  ASSERT_EQ(accepted_.size(), 1u);
  // Listener shutdown does not revoke a published connection's endpoint space.
  iree_net_message_endpoint_t endpoint = {};
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      result->connection, {+[](void* data, iree_status_t status,
                               iree_net_message_endpoint_t value) {
                             IREE_EXPECT_OK(status);
                             *static_cast<iree_net_message_endpoint_t*>(data) =
                                 value;
                           },
                           &endpoint}));
  ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return endpoint.self != nullptr; }));
}

TEST_P(ShmFactoryTest, RejectedImportDoesNotPoisonListener) {
  ASSERT_NO_FATAL_FAILURE(Listen());
  auto limits = options_;
  limits.region.slot_capacity /= 2;
  iree_net_transport_factory_t* small_factory = nullptr;
  IREE_ASSERT_OK(iree_net_shm_factory_create(&limits, iree_allocator_system(),
                                             &small_factory));
  auto* rejected = Connect(small_factory);
  iree_net_transport_factory_release(small_factory);
  ASSERT_TRUE(rejected->pending);
  ASSERT_NO_FATAL_FAILURE(
      PollUntil([&] { return !rejected->pending && !accept_errors_.empty(); }));
  EXPECT_EQ(rejected->code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(rejected->connection, nullptr);
  auto* accepted = Connect();
  ASSERT_TRUE(accepted->pending);
  ASSERT_NO_FATAL_FAILURE(
      PollUntil([&] { return !accepted->pending && accepted_.size() == 1; }));
  EXPECT_EQ(accepted->code, IREE_STATUS_OK);
}

TEST_P(ShmFactoryTest, StopDuringImportWaitNeedsNoPeerProgress) {
  ASSERT_NO_FATAL_FAILURE(Listen());
  ASSERT_NO_FATAL_FAILURE(ConnectRawClient());
  std::array<uint8_t, IREE_NET_SHM_BOOTSTRAP_OFFER_SIZE> offer;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      raw_.stream, iree_make_byte_span(offer.data(), offer.size()),
      IREE_NET_SHM_STORAGE_HANDLE_COUNT, {Transferred, this},
      received_handles_));
  ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
  iree_net_shm_region_layout_t layout;
  IREE_ASSERT_OK(iree_net_shm_bootstrap_decode_offer(
      iree_make_const_byte_span(offer.data(), offer.size()), &layout));
  ASSERT_NO_FATAL_FAILURE(Stop());
  ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return !listener_; }));
  EXPECT_TRUE(accepted_.empty());
  EXPECT_TRUE(accept_errors_.empty());
  // The receiver already owns this bundle even though ACCEPT was never sent.
  iree_net_shm_storage_t* imported = nullptr;
  IREE_ASSERT_OK(iree_net_shm_storage_import(
      &layout, received_handles_, iree_allocator_system(), &imported));
  iree_net_shm_storage_release(imported);
}

TEST_P(ShmFactoryTest, CancelAfterImportNeedsNoReadyOrPeerProgress) {
  ConnectResult* result = nullptr;
  ASSERT_NO_FATAL_FAILURE(AcceptRawServer(&result));
  iree_net_shm_region_layout_t layout;
  IREE_ASSERT_OK(
      iree_net_shm_region_calculate_layout(options_.region, &layout));
  IREE_ASSERT_OK(iree_net_shm_storage_create(&layout, iree_allocator_system(),
                                             &raw_storage_));
  std::array<uint8_t, IREE_NET_SHM_BOOTSTRAP_OFFER_SIZE> offer;
  iree_net_shm_bootstrap_encode_offer(&layout, offer.data());
  iree_async_primitive_t handles[IREE_NET_SHM_STORAGE_HANDLE_COUNT];
  iree_net_shm_storage_export(raw_storage_, handles);
  IREE_ASSERT_OK(iree_async_local_stream_send(
      raw_.stream, iree_make_const_byte_span(offer.data(), offer.size()),
      IREE_NET_SHM_STORAGE_HANDLE_COUNT, handles, {Transferred, this}));
  ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
  std::array<uint8_t, IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE> accept;
  IREE_ASSERT_OK(iree_async_local_stream_receive(
      raw_.stream, iree_make_byte_span(accept.data(), accept.size()), 0,
      {Transferred, this}, nullptr));
  ASSERT_NO_FATAL_FAILURE(AwaitTransfer());
  IREE_ASSERT_OK(iree_net_shm_bootstrap_decode_ack(
      iree_make_const_byte_span(accept.data(), accept.size()),
      IREE_NET_SHM_BOOTSTRAP_TYPE_ACCEPT));
  EXPECT_EQ(result->count, 0);
  std::thread cancel(
      [&] { iree_net_transport_connect_operation_cancel(&result->operation); });
  cancel.join();
  ASSERT_NO_FATAL_FAILURE(PollUntil([&] { return !result->pending; }));
  EXPECT_EQ(result->count, 1);
  EXPECT_EQ(result->code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(result->connection, nullptr);
}

#if defined(IREE_PLATFORM_WINDOWS)
INSTANTIATE_TEST_SUITE_P(Platform, ShmFactoryTest, ::testing::Values(false));
#else
INSTANTIATE_TEST_SUITE_P(PlatformAndPosix, ShmFactoryTest,
                         ::testing::Values(false, true));
#endif

}  // namespace
