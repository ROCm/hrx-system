// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/input.h"

#include <sys/socket.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/testing/worker_gate.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/serve/transport/websocket/handshake.h"

IREE_FLAG(string, model, "", "Recognition model document.");
IREE_FLAG(int32_t, test_workers, 1, "Shared task worker count.");

namespace {

class RealtimeInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    auto options = loom_serve_test_application_options_default();
    options.worker_count = FLAG_test_workers;
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &options, iree_allocator_system(), &application));
    host_ = loom_serve_test_application_host(application);
    registration_.device_group =
        loom_serve_test_application_device_group(application);
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, 1, loom_serve_model_registration_factory(&registration_),
        iree_allocator_system(), &registry_));
    loom_serve_model_configuration_t configuration;
    IREE_ASSERT_OK(loom_serve_model_configuration_load(
        iree_make_cstring_view(FLAG_model), iree_allocator_system(),
        &configuration));
    name_.assign(configuration.name.data, configuration.name.size);
    iree_host_size_t index = 0;
    auto status =
        loom_serve_registry_register(registry_, &configuration, &index);
    loom_serve_model_configuration_deinitialize(&configuration);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(Until([&] {
      return loom_serve_registry_at(registry_, 0)->phase ==
             LOOM_SERVE_REGISTRATION_READY;
    }));
    auto transport_options = loom_serve_http_server_options_default();
    transport_options.connection_capacity = 2;
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(host_), &transport_options,
        iree_allocator_system(), &server_));
    loom_serve_http_server_set_observer(server_, Wake());
    auto listen = loom_serve_http_listener_options_default();
    listen.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listen, iree_allocator_system(), &listener_));
    const auto* address = loom_serve_http_listener_address(listener_);
    const auto* native = reinterpret_cast<const sockaddr*>(address->storage);
    client_ = socket(native->sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(client_, 0);
    ASSERT_EQ(connect(client_, native, address->length), 0);
    Write(
        "GET /v1/realtime HTTP/1.1\r\nHost: local\r\n"
        "Connection: Upgrade\r\nUpgrade: websocket\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n");
    const loom_serve_http_request_t* request = nullptr;
    Await([&] {
      connection_ = loom_serve_http_server_take_request(server_, &request);
      return connection_ != nullptr;
    });
    char storage[512];
    iree_string_builder_t response;
    iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                                &response);
    IREE_ASSERT_OK(
        loom_serve_websocket_handshake(request, iree_string_view_list_empty(),
                                       iree_string_view_empty(), &response));
    IREE_ASSERT_OK(loom_serve_websocket_upgrade(
        connection_, loom_serve_websocket_limits_default(),
        iree_allocator_system(), &socket_));
    IREE_ASSERT_OK(loom_serve_http_connection_send(
        connection_, iree_string_builder_view(&response)));
    std::string received(response.size, '\0');
    ASSERT_EQ(recv(client_, received.data(), received.size(), MSG_WAITALL),
              received.size());
    ASSERT_EQ(received, std::string(response.buffer, response.size));
    loom_serve_realtime_binding_t binding;
    IREE_ASSERT_OK(loom_serve_realtime_binding_resolve(
        registry_, iree_make_string_view(name_.data(), name_.size()),
        LOOM_SERVE_MODEL_KIND_RECOGNITION, &binding));
    IREE_ASSERT_OK(loom_serve_realtime_input_allocate(
        loom_serve_host_executor(host_), registry_, binding, socket_, Wake(),
        iree_allocator_system(), &input_));
  }

  void TearDown() override {
    stopping_ = true;
    if (input_) {
      IREE_EXPECT_OK(
          Until([&] { return !loom_serve_realtime_input_pending(input_); }));
      loom_serve_realtime_input_free(input_);
      input_ = nullptr;
    }
    if (connection_) {
      loom_serve_http_connection_abort(connection_);
    }
    if (client_ >= 0) {
      EXPECT_EQ(close(client_), 0);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    if (registry_) {
      IREE_EXPECT_OK(loom_serve_registry_remove(
          loom_serve_registry_at(registry_, 0), LOOM_SERVE_REMOVAL_ABORT));
      IREE_EXPECT_OK(
          Until([&] { return loom_serve_registry_count(registry_) == 0; }));
      IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    }
    iree_notification_deinitialize(&notification_);
  }

  loom_serve_wake_callback_t Wake() {
    return {[](void* self) {
              iree_notification_post(static_cast<iree_notification_t*>(self),
                                     IREE_ALL_WAITERS);
            },
            &notification_};
  }

  template <typename Done>
  void Await(Done done) {
    while (true) {
      auto ticket = iree_notification_prepare_wait(&notification_);
      if (done()) {
        iree_notification_cancel_wait(&notification_);
        return;
      }
      iree_notification_commit_wait(&notification_, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }

  template <typename Done>
  iree_status_t Until(Done done) {
    auto status = iree_ok_status();
    while (iree_status_is_ok(status) && !done()) {
      auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      if (input_) {
        loom_serve_realtime_input_advance(input_, stopping_, &progress);
      }
      auto* model = loom_serve_registry_at(registry_, 0);
      if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
        bool advanced = false;
        status = model->service.advance(
            model->service.self, loom_serve_registry_service_flags(model),
            &advanced);
        progress |= advanced;
      }
      if (iree_status_is_ok(status) && model->service.name.size) {
        bool advanced = false;
        status = loom_serve_registry_advance(registry_, 0, Wake(), &advanced);
        progress |= advanced;
        status = iree_status_join(status, iree_status_clone(model->status));
      }
      if (!iree_status_is_ok(status) || progress || done()) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
    return status;
  }

  void Write(const std::string& bytes) {
    ASSERT_EQ(send(client_, bytes.data(), bytes.size(), MSG_NOSIGNAL),
              bytes.size());
  }

  loom_serve_websocket_message_t* Send(const std::string& text,
                                       iree_const_byte_span_t* payload) {
    // These bounded test messages use the short, client-masked frame form.
    const std::string frame = std::string(1, '\x81') +
                              static_cast<char>(0x80 | text.size()) +
                              std::string(4, '\0') + text;
    Write(frame);
    loom_serve_websocket_message_t* message = nullptr;
    loom_serve_websocket_message_type_t type;
    Await([&] {
      message = loom_serve_websocket_take(socket_, &type, payload);
      return message != nullptr;
    });
    EXPECT_EQ(type, LOOM_SERVE_WEBSOCKET_TEXT);
    return message;
  }

  // Shared application owner retained until all accepted model work retires.
  loom_serve_host_t* host_ = nullptr;
  // Registration factory borrows this immutable deployment group.
  loom_serve_model_registration_options_t registration_ = {};
  // Real source-backed recognition service, advanced on the test owner thread.
  loom_serve_registry_t* registry_ = nullptr;
  // Copied public deployment identity, stable for binding resolution.
  std::string name_;
  // Explicit transport/model/CPU completion notification; no timing polls.
  iree_notification_t notification_;
  // Transport retaining accepted WebSocket receipts through HTTP claim release.
  loom_serve_http_server_t* server_ = nullptr;
  // Ephemeral local TCP listener using the application's proactors.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Owned HTTP claim, released after every input worker and receipt.
  loom_serve_http_connection_t* connection_ = nullptr;
  // Framed protocol owned by the claimed connection.
  loom_serve_websocket_t* socket_ = nullptr;
  // Client descriptor; shutdown does not release the server's claim.
  int client_ = -1;
  // Production input owner under test, without a mock conversion path.
  loom_serve_realtime_input_t* input_ = nullptr;
  // Teardown closes items before joining their accepted borrowers.
  bool stopping_ = false;
};

TEST_F(RealtimeInputTest,
       ClearedInputRetainsReceiptThroughDisconnectAndWorkerRelease) {
  const loom_serve_recognition_request_t request = {
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
  };
  loom_serve_realtime_input_item_t* item = nullptr;
  IREE_ASSERT_OK(loom_serve_realtime_input_begin(input_, &request, &item));
  iree_const_byte_span_t payload;
  auto* message = Send("AAAAAA==", &payload);
  IREE_ASSERT_OK(loom_serve_realtime_input_append(
      input_, item, message,
      iree_make_string_view(reinterpret_cast<const char*>(payload.data),
                            payload.data_length)));
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> gates;
  for (int32_t i = 0; i < FLAG_test_workers; ++i) {
    gates.push_back(std::make_unique<loom_serve::testing::WorkerGate>(
        loom_serve_host_executor(host_)));
  }
  bool progress = false;
  loom_serve_realtime_input_advance(input_, false, &progress);
  IREE_ASSERT_OK(loom_serve_realtime_input_status(item));
  ASSERT_TRUE(progress);
  loom_serve_realtime_input_close(item);
  loom_serve_realtime_input_advance(input_, false, &progress);
  // If no conversion borrower was actually accepted, close would have retired
  // this item synchronously. Every worker is still held at the explicit gate.
  ASSERT_TRUE(loom_serve_realtime_input_pending(input_));
  auto* control = Send("later control", &payload);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(payload.data),
                        payload.data_length),
            "later control");
  loom_serve_websocket_message_release(socket_, control);
  ASSERT_EQ(shutdown(client_, SHUT_WR), 0);
  Await([&] { return loom_serve_http_connection_failed(connection_); });
  EXPECT_TRUE(loom_serve_realtime_input_pending(input_));
  gates.clear();
  IREE_ASSERT_OK(
      Until([&] { return !loom_serve_realtime_input_pending(input_); }));
}

}  // namespace
