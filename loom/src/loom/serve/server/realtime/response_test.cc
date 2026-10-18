// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/response.h"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <string>

#include "iree/base/internal/json.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/realtime/item.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/serve/transport/websocket/handshake.h"

IREE_FLAG(string, text_model, "", "Text model document.");
IREE_FLAG(string, speech_model, "", "Speech model document.");
IREE_FLAG(int32_t, test_workers, 1, "Shared task worker count.");

namespace {

iree_string_view_t View(const std::string& text) {
  return iree_make_string_view(text.data(), text.size());
}

std::string Field(const std::string& object, const char* key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(
      View(object), iree_make_cstring_view(key), &value));
  return std::string(value.data, value.size);
}

class RealtimeResponseTest : public ::testing::TestWithParam<int> {
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
        0, nullptr, 2, loom_serve_model_registration_factory(&registration_),
        iree_allocator_system(), &registry_));
    const std::string weights =
        iree::testing::MakeTempFilePath("realtime_text", ".safetensors");
    std::string header =
        R"({"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream file(weights, std::ios::binary);
    uint8_t bytes[8];
    iree_unaligned_store_le_u64(bytes, header.size());
    file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    file.write(header.data(), header.size());
    iree_unaligned_store_le_u32(bytes, 2);
    file.write(reinterpret_cast<const char*>(bytes), 4);
    file.close();
    ASSERT_TRUE(file.good());
    const std::array<const char*, 2> paths = {FLAG_text_model,
                                              FLAG_speech_model};
    for (size_t i = 0; i < paths.size(); ++i) {
      loom_serve_model_configuration_t configuration;
      IREE_ASSERT_OK(loom_serve_model_configuration_load(
          iree_make_cstring_view(paths[i]), iree_allocator_system(),
          &configuration));
      iree_host_size_t index = 0;
      auto status = iree_ok_status();
      if (configuration.kind == LOOM_SERVE_MODEL_KIND_TEXT) {
        configuration.text.session_cache_capacity = 4;
        configuration.text.checkpoint_capacity = 4;
        void* owned_weights = nullptr;
        status = iree_allocator_clone(
            configuration.host_allocator,
            iree_make_const_byte_span(weights.c_str(), weights.size() + 1),
            &owned_weights);
        if (iree_status_is_ok(status)) {
          iree_allocator_free(
              configuration.host_allocator,
              const_cast<char*>(configuration.text.weights.data));
          configuration.text.weights = iree_make_string_view(
              static_cast<const char*>(owned_weights), weights.size());
        }
      }
      if (iree_status_is_ok(status)) {
        status =
            loom_serve_registry_register(registry_, &configuration, &index);
      }
      loom_serve_model_configuration_deinitialize(&configuration);
      IREE_ASSERT_OK(status);
      ASSERT_EQ(index, i);
    }
    IREE_ASSERT_OK(Until([&] {
      return loom_serve_registry_at(registry_, 0)->phase ==
                 LOOM_SERVE_REGISTRATION_READY &&
             loom_serve_registry_at(registry_, 1)->phase ==
                 LOOM_SERVE_REGISTRATION_READY;
    }));

    const auto transport_options = loom_serve_http_server_options_default();
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
    const std::string upgrade =
        "GET /v1/realtime HTTP/1.1\r\nHost: local\r\n"
        "Connection: Upgrade\r\nUpgrade: websocket\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
    ASSERT_EQ(send(client_, upgrade.data(), upgrade.size(), MSG_NOSIGNAL),
              upgrade.size());
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(Until([&] {
      if (!connection_) {
        connection_ = loom_serve_http_server_take_request(server_, &request);
      }
      return connection_ != nullptr;
    }));
    char storage[512];
    iree_string_builder_t handshake;
    iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                                &handshake);
    IREE_ASSERT_OK(
        loom_serve_websocket_handshake(request, iree_string_view_list_empty(),
                                       iree_string_view_empty(), &handshake));
    IREE_ASSERT_OK(loom_serve_websocket_upgrade(
        connection_, loom_serve_websocket_limits_default(),
        iree_allocator_system(), &socket_));
    IREE_ASSERT_OK(loom_serve_http_connection_send(
        connection_, iree_string_builder_view(&handshake)));
    EXPECT_EQ(ReadBytes(handshake.size),
              std::string(handshake.buffer, handshake.size));
    loom_serve_realtime_wire_initialize(7, socket_, connection_,
                                        iree_allocator_system(), &wire_);
    loom_serve_realtime_conversation_initialize(7, iree_allocator_system(),
                                                &conversation_);
    loom_serve_realtime_binding_t text, speech;
    IREE_ASSERT_OK(loom_serve_realtime_binding_resolve(
        registry_, IREE_SV("fixture-text"), 1u << LOOM_SERVE_MODEL_KIND_TEXT,
        &text));
    IREE_ASSERT_OK(loom_serve_realtime_binding_resolve(
        registry_, IREE_SV("speech"), 1u << LOOM_SERVE_MODEL_KIND_SPEECH,
        &speech));
    loom_serve_realtime_history_initialize(registry_, text,
                                           iree_allocator_system(), &history_);
    profile_ = {
        .name = IREE_SV("assistant"),
        .text = text.name,
        .speech = speech.name,
        .voice = IREE_SV("positive"),
        .max_output_tokens = 12,
        .seed = 42,
    };
    IREE_ASSERT_OK(loom_serve_realtime_settings_initialize(
        &profile_, iree_allocator_system(), &settings_));
    IREE_ASSERT_OK(loom_serve_realtime_tools_parse(
        IREE_SV(
            R"([{"type":"function","name":"lookup","parameters":{"type":"object","properties":{}}},{"type":"function","name":"other","parameters":{"type":"object","properties":{}}}])"),
        iree_allocator_system(), &settings_.output.tools));
    IREE_ASSERT_OK(loom_serve_realtime_responses_allocate(
        7, &profile_, text, speech, loom_serve_host_executor(host_), registry_,
        server_, Wake(), &settings_, &conversation_, &history_, &wire_,
        iree_allocator_system(), &responses_));
    IREE_ASSERT_OK(Insert(IREE_SV(
        R"({"type":"message","role":"user","content":[{"type":"input_text","text":"a"}]})")));
  }

  void TearDown() override {
    if (responses_) {
      closing_ = true;
      loom_serve_realtime_responses_stop(responses_);
      loom_serve_realtime_history_invalidate(&history_, 0);
      loom_serve_realtime_wire_clear(&wire_);
      IREE_EXPECT_OK(Until([&] {
        return !loom_serve_realtime_responses_pending(responses_) &&
               !loom_serve_realtime_history_pending(&history_);
      }));
      loom_serve_realtime_responses_free(responses_);
      responses_ = nullptr;
    }
    loom_serve_realtime_history_deinitialize(&history_);
    loom_serve_realtime_settings_deinitialize(&settings_);
    loom_serve_realtime_conversation_deinitialize(&conversation_);
    loom_serve_realtime_wire_deinitialize(&wire_);
    if (connection_) {
      loom_serve_http_connection_abort(connection_);
    }
    if (client_ >= 0) {
      EXPECT_EQ(close(client_), 0);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    if (registry_) {
      for (iree_host_size_t i = 0; i < 2; ++i) {
        auto* model = loom_serve_registry_at(registry_, i);
        if (model->service.name.size) {
          IREE_EXPECT_OK(
              loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
        }
      }
      IREE_EXPECT_OK(
          Until([&] { return loom_serve_registry_count(registry_) == 0; }));
      IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    }
    iree_notification_deinitialize(&notification_);
  }

  iree_status_t Insert(iree_string_view_t object) {
    loom_serve_realtime_item_t* item = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_item_parse(&conversation_, object, &item));
    auto status = loom_serve_realtime_conversation_insert(
        &conversation_, conversation_.last, item);
    if (!iree_status_is_ok(status)) {
      loom_serve_realtime_conversation_remove(&conversation_, item);
    }
    return status;
  }

  std::string ReadBytes(size_t count) {
    std::string bytes(count, '\0');
    const auto received = recv(client_, bytes.data(), count, MSG_WAITALL);
    EXPECT_EQ(received, count);
    if (received < 0) {
      return {};
    }
    bytes.resize(received);
    return bytes;
  }

  std::string NextEvent() {
    IREE_EXPECT_OK(Until([&] {
      return wire_.first && loom_serve_http_connection_can_send(connection_);
    }));
    bool progress = false;
    IREE_EXPECT_OK(loom_serve_realtime_wire_flush(&wire_, &progress));
    EXPECT_TRUE(progress);
    const std::string header = ReadBytes(2);
    if (header.size() != 2) {
      return "{}";
    }
    EXPECT_EQ(static_cast<unsigned char>(header[0]), 0x81);
    uint64_t length = static_cast<unsigned char>(header[1]);
    if (length == 126 || length == 127) {
      const std::string extended = ReadBytes(length == 126 ? 2 : 8);
      length = 0;
      for (unsigned char byte : extended) {
        length = length * 256 + byte;
      }
    }
    return ReadBytes(length);
  }

  loom_serve_wake_callback_t Wake() {
    return {[](void* self) {
              iree_notification_post(static_cast<iree_notification_t*>(self),
                                     IREE_ALL_WAITERS);
            },
            &notification_};
  }

  template <typename Done>
  iree_status_t Until(Done done) {
    auto status = iree_ok_status();
    while (iree_status_is_ok(status) && !done()) {
      auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      if (responses_) {
        status = loom_serve_realtime_responses_advance(responses_, closing_,
                                                       UINT64_MAX, &progress);
        if (iree_status_is_ok(status)) {
          bool cleanup_failed = false;
          status = loom_serve_realtime_history_advance(
              &history_, closing_, &progress, &cleanup_failed);
        }
      }
      for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
        auto* model = loom_serve_registry_at(registry_, i);
        if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
          bool advanced = false;
          status = model->service.advance(
              model->service.self, loom_serve_registry_service_flags(model),
              &advanced);
          progress |= advanced;
        }
        if (iree_status_is_ok(status) && model->service.name.size) {
          bool advanced = false;
          status = loom_serve_registry_advance(registry_, i, Wake(), &advanced);
          progress |= advanced;
          status = iree_status_join(status, iree_status_clone(model->status));
        }
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

  // Process-owned application and real compiled text/speech registrations.
  loom_serve_host_t* host_ = nullptr;
  // Factory binding retained through native registration teardown.
  loom_serve_model_registration_options_t registration_ = {};
  // Serialized registration owner advanced alongside the actual response.
  loom_serve_registry_t* registry_ = nullptr;
  // Transport and model completion notification, without wall-clock polling.
  iree_notification_t notification_;
  // Actual ephemeral transport whose queue is flushed one event at a time.
  loom_serve_http_server_t* server_ = nullptr;
  // Listener retained through client and HTTP claim retirement.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Owned HTTP claim, transferred to WebSocket framing.
  loom_serve_http_connection_t* connection_ = nullptr;
  // Connection-owned RFC 6455 framing.
  loom_serve_websocket_t* socket_ = nullptr;
  // Real loopback reader, with no racing service-owner thread.
  int client_ = -1;
  // Shared event admission state, borrowed by the response owner.
  loom_serve_realtime_wire_t wire_ = {};
  // Canonical history retained independently of native caches.
  loom_serve_realtime_conversation_t conversation_ = {};
  // Actual native checkpoint controls and their retirement.
  loom_serve_realtime_history_t history_ = {};
  // Immutable assistant configuration retained through all work.
  loom_serve_realtime_profile_t profile_ = {};
  // Default policy including the two-function catalogue.
  loom_serve_realtime_settings_t settings_ = {};
  // Production response owner; no model output is mocked.
  loom_serve_realtime_responses_t* responses_ = nullptr;
  // Teardown withdraws accepted native and CPU work before freeing storage.
  bool closing_ = false;
};

TEST_P(RealtimeResponseTest, CancellationPreservesOnlyExecutableCalls) {
  IREE_ASSERT_OK(loom_serve_realtime_responses_create(
      responses_, iree_string_view_empty()));
  EXPECT_EQ(Field(NextEvent(), "type"), "response.created");
  // Six events per call. Parameterization cuts before and after each of the
  // first two calls' publication boundaries, with no timing or thread race.
  for (int i = 0; i < GetParam(); ++i) {
    const std::string event = NextEvent();
    ASSERT_NE(Field(event, "type"), "response.done") << event;
  }
  IREE_ASSERT_OK(loom_serve_realtime_responses_cancel(
      responses_, iree_string_view_empty()));
  std::string response;
  do {
    const std::string event = NextEvent();
    if (Field(event, "type") == "response.done") {
      response = Field(event, "response");
    }
  } while (response.empty());
  EXPECT_EQ(Field(response, "status"), "cancelled");
  const int announced = (GetParam() + 5) / 6;
  const int completed = (GetParam() + 1) / 6;
  const auto output = Field(response, "output");
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(iree_json_array_length(View(output), &count));
  ASSERT_EQ(count, announced) << response;
  for (int i = 0; i < announced; ++i) {
    iree_string_view_t value;
    IREE_ASSERT_OK(iree_json_array_get(View(output), i, &value));
    const std::string item(value.data, value.size);
    EXPECT_EQ(Field(item, "status"),
              i < completed ? "completed" : "incomplete");
    const std::string result =
        "{\"type\":\"function_call_output\",\"call_id\":\"" +
        Field(item, "call_id") + "\",\"output\":\"a\"}";
    if (i < completed) {
      IREE_ASSERT_OK(Insert(View(result)));
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Insert(View(result)));
    }
  }
  IREE_ASSERT_OK(loom_serve_realtime_responses_create(
      responses_,
      IREE_SV(
          R"({"tool_choice":"none","output_modalities":["text"],"max_output_tokens":2})")));
  std::string text;
  response.clear();
  do {
    const std::string event = NextEvent();
    const std::string type = Field(event, "type");
    if (type == "response.output_text.delta") {
      text += Field(event, "delta");
    }
    if (type == "response.done") {
      response = Field(event, "response");
    }
  } while (response.empty());
  // Independent recurrence results for a, ax#a, and axa#aa. Incomplete calls
  // contribute no raw letters or assistant delimiters to native replay.
  EXPECT_EQ(text, completed == 1 ? "xb" : "bb") << response;
  EXPECT_EQ(Field(Field(Field(response, "usage"), "input_token_details"),
                  "cached_tokens"),
            "0");
}

INSTANTIATE_TEST_SUITE_P(PublicationBoundary, RealtimeResponseTest,
                         ::testing::Range(0, 13));

}  // namespace
