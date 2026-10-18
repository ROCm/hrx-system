// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/tooling/flags.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/service/text/checkpoint_control.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/stream.h"
#include "loom/util/json.h"

IREE_FLAG(string, native_text_model, "",
          "Optional learned model document for native text qualification.");
IREE_FLAG(string, native_text_request, "",
          "Streaming chat JSON paired with native_text_model.");
IREE_FLAG(int32_t, native_text_workers, 1,
          "Shared worker count for filtered native text qualification.");

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::OwnerTurn;
using loom_serve::testing::text::RegisteredSourceTextServiceTest;

using NativeStream = std::unique_ptr<loom_serve_text_stream_t,
                                     decltype(&loom_serve_text_stream_close)>;
using NativeControl =
    std::unique_ptr<loom_serve_text_checkpoint_control_t,
                    decltype(&loom_serve_text_checkpoint_control_close)>;

std::string String(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

iree_string_view_t Field(iree_string_view_t object, const char* name) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(
      object, iree_make_cstring_view(name), &value));
  return value;
}

std::string Decode(iree_string_view_t value) {
  std::string decoded(value.size, '\0');
  iree_host_size_t length = 0;
  IREE_EXPECT_OK(iree_json_unescape_string(value, decoded.size(),
                                           decoded.data(), &length));
  decoded.resize(length);
  return decoded;
}

std::string Quote(iree_string_view_t value) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t output;
  loom_output_stream_for_builder(&builder, &output);
  IREE_EXPECT_OK(loom_json_write_escaped_string(&output, value));
  const std::string result = String(iree_string_builder_view(&builder));
  iree_string_builder_deinitialize(&builder);
  return result;
}

class NativeTextRegistryTest : public RegisteredSourceTextServiceTest {
 protected:
  iree_host_size_t WorkerCount() const override {
    return FLAG_native_text_workers;
  }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(RegisteredSourceTextServiceTest::SetUp());
    registration_options_.device_group = group_;
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, 1,
        loom_serve_model_registration_factory(&registration_options_),
        host_allocator_, &registry_));
    ASSERT_EQ(FLAG_native_text_model[0] != 0, FLAG_native_text_request[0] != 0);
    if (FLAG_native_text_request[0]) {
      iree_io_file_contents_t* contents = nullptr;
      IREE_ASSERT_OK(iree_io_file_contents_read(
          iree_make_cstring_view(FLAG_native_text_request), host_allocator_,
          &contents));
      body_.assign(reinterpret_cast<const char*>(contents->buffer.data),
                   contents->buffer.data_length);
      iree_io_file_contents_free(contents);
    } else {
      body_ = Request(Message("user", "abxa"), 4);
    }
    ASSERT_EQ(String(Field(iree_make_cstring_view(body_.c_str()), "stream")),
              "true");
    ASSERT_NO_FATAL_FAILURE(Register());
  }

  void TearDown() override {
    if (registry_ && loom_serve_registry_count(registry_)) {
      IREE_EXPECT_OK(
          loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_ABORT));
      AdvanceRegistryUntil(
          [&] { return loom_serve_registry_count(registry_) == 0; });
    }
    IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    loom_serve_http_server_request_stop(server_);
    if (reader_.joinable()) {
      reader_.join();
    }
    IREE_EXPECT_OK(read_status_);
    RegisteredSourceTextServiceTest::TearDown();
  }

  void Register() {
    loom_serve_model_configuration_t configuration;
    if (FLAG_native_text_model[0]) {
      IREE_ASSERT_OK(loom_serve_model_configuration_load(
          iree_make_cstring_view(FLAG_native_text_model), host_allocator_,
          &configuration));
    } else {
      const auto declaration =
          Declaration("fixture-text", directory_, "drained");
      IREE_ASSERT_OK(loom_serve_model_configuration_initialize(
          iree_make_cstring_view(declaration.c_str()), iree_string_view_empty(),
          nullptr, host_allocator_, &configuration));
    }
    name_ = String(configuration.name);
    configuration.start_drained = true;
    auto status =
        loom_serve_registry_register(registry_, &configuration, &index_);
    loom_serve_model_configuration_deinitialize(&configuration);
    IREE_ASSERT_OK(status);
  }

  loom_serve_registered_model_t* Model() {
    return loom_serve_registry_at(registry_, index_);
  }

  template <typename Predicate>
  void AdvanceRegistryUntil(Predicate ready) {
    auto* notification = loom_serve_http_server_notification(server_);
    const loom_serve_wake_callback_t wake = {
        [](void* self) {
          iree_notification_post(static_cast<iree_notification_t*>(self),
                                 IREE_ALL_WAITERS);
        },
        notification};
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      auto status = iree_ok_status();
      if (Model()->phase == LOOM_SERVE_REGISTRATION_READY) {
        auto& service = Model()->service;
        status = service.advance(service.self,
                                 loom_serve_registry_service_flags(Model()),
                                 &progress);
      }
      bool registry_progress = false;
      if (iree_status_is_ok(status)) {
        status = loom_serve_registry_advance(registry_, index_, wake,
                                             &registry_progress);
      }
      status = iree_status_join(status, iree_status_clone(Model()->status));
      if (ready() || progress || registry_progress ||
          !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
    }
  }

  void Prepare() {
    Model()->closed = false;
    ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil(
        [&] { return Model()->phase == LOOM_SERVE_REGISTRATION_READY; }));
  }

  bool Idle() {
    const auto& service = Model()->service;
    return !service.is_pending(service.self) &&
           !service.is_active(service.self) &&
           !service.snapshot(service.self).queued_requests;
  }

  iree_status_t Open(const std::string& body, const char* session,
                     NativeStream* stream, const char* checkpoint = "") {
    loom_serve_text_request_t request = {
        .session = iree_make_cstring_view(session),
        .checkpoint = iree_make_cstring_view(checkpoint),
    };
    iree_string_builder_initialize(host_allocator_, &request.body);
    auto status = iree_string_builder_append_string(
        &request.body, iree_make_string_view(body.data(), body.size()));
    const char* input = iree_string_builder_buffer(&request.body);
    loom_serve_text_stream_t* value = nullptr;
    if (iree_status_is_ok(status)) {
      status = loom_serve_registry_try_open_text(
          registry_, iree_make_cstring_view(name_.c_str()), &request, &value);
    }
    if (value) {
      EXPECT_EQ(request.body.buffer, nullptr);
      EXPECT_EQ(request.session.data, nullptr);
    } else {
      EXPECT_EQ(iree_string_builder_buffer(&request.body), input);
      EXPECT_EQ(String(iree_string_builder_view(&request.body)), body);
      EXPECT_EQ(String(request.session), session);
    }
    iree_string_builder_deinitialize(&request.body);
    stream->reset(value);
    return status;
  }

  void AwaitBlock(const NativeStream& stream) {
    ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
      return loom_serve_text_stream_peek(stream.get()) ||
             loom_serve_text_stream_is_complete(stream.get());
    }));
    IREE_ASSERT_OK(loom_serve_text_stream_status(stream.get()));
    ASSERT_NE(loom_serve_text_stream_peek(stream.get()), nullptr);
  }

  iree_status_t OpenCheckpoint(loom_serve_text_checkpoint_action_t action,
                               const char* name, const char* session,
                               NativeControl* control) {
    const loom_serve_text_checkpoint_request_t request = {
        .action = action,
        .name = iree_make_cstring_view(name),
        .session = iree_make_cstring_view(session),
    };
    loom_serve_text_checkpoint_control_t* value = nullptr;
    auto status = loom_serve_registry_try_open_text_checkpoint(
        registry_, iree_make_cstring_view(name_.c_str()), &request, &value);
    control->reset(value);
    return status;
  }

  void CompleteCheckpoint(const NativeControl& control) {
    ASSERT_NE(control, nullptr);
    ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
      return loom_serve_text_checkpoint_control_is_complete(control.get());
    }));
    IREE_ASSERT_OK(loom_serve_text_checkpoint_control_status(control.get()));
    ASSERT_NE(loom_serve_text_checkpoint_control_result(control.get()),
              nullptr);
  }

  void ReadFinal(const NativeStream& stream) {
    std::string content, reasoning;
    ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
      if (const auto* block = loom_serve_text_stream_peek(stream.get())) {
        for (iree_host_size_t i = 0;
             i < loom_serve_text_stream_block_count(block); ++i) {
          const auto span = loom_serve_text_stream_block_at(block, i);
          EXPECT_TRUE(span.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT ||
                      span.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING);
          (span.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT ? content
                                                            : reasoning)
              .append(span.text.data, span.text.size);
        }
        loom_serve_text_stream_consume(stream.get());
      }
      return loom_serve_text_stream_is_complete(stream.get());
    }));
    IREE_ASSERT_OK(loom_serve_text_stream_status(stream.get()));
    const auto* result = loom_serve_text_stream_result(stream.get());
    ASSERT_NE(result, nullptr);
    if (result->response.flags & LOOM_SERVE_TEXT_CHAT_FLAG_STREAM) {
      EXPECT_EQ(String(result->content), content);
      EXPECT_EQ(String(result->reasoning), reasoning);
    } else {
      EXPECT_TRUE(content.empty());
      EXPECT_TRUE(reasoning.empty());
    }
    EXPECT_EQ(String(result->response.model_name), name_);
    EXPECT_GT(result->response.usage.output_tokens, 0u);
    RecordProperty("content", String(result->content));
    RecordProperty("reasoning", String(result->reasoning));
    RecordProperty("finish_reason", String(result->finish_reason));
    RecordProperty("cached_tokens",
                   std::to_string(result->response.usage.cached_tokens));
    iree_string_builder_t calls;
    iree_string_builder_initialize(host_allocator_, &calls);
    IREE_EXPECT_OK(
        loom_serve_text_tool_calls_write_json(result->calls, false, &calls));
    RecordProperty("tool_calls", String(iree_string_builder_view(&calls)));
    iree_string_builder_deinitialize(&calls);
  }

  std::string CompleteRequest(std::string body) {
    iree_string_view_t options;
    IREE_EXPECT_OK(
        iree_json_try_lookup_object_value(iree_make_cstring_view(body.c_str()),
                                          IREE_SV("stream_options"), &options));
    if (options.size) {
      body.replace(options.data - body.data(), options.size, "null");
    }
    const auto streaming =
        Field(iree_make_cstring_view(body.c_str()), "stream");
    body.replace(streaming.data - body.data(), streaming.size, "false");
    return body;
  }

  void StartHttp(const std::string& request, const char* session = "http") {
    const auto body = CompleteRequest(request);
    const int client =
        Send(body, std::string("X-Loom-Session: ") + session + "\r\n");
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(client, Model()->service, &connection));
    reader_ = std::thread([this, client] {
      read_status_ = http_.Read(client, false, &response_);
    });
  }

  void CheckHttp(const loom_serve_text_stream_result_t& result) {
    reader_.join();
    ASSERT_EQ(response_.find("HTTP/1.1 200 OK\r\n"), 0u) << response_;
    const size_t offset = response_.find("\r\n\r\n");
    ASSERT_NE(offset, std::string::npos);
    const auto document = iree_make_string_view(response_.data() + offset + 4,
                                                response_.size() - offset - 4);
    iree_string_view_t choice;
    IREE_ASSERT_OK(iree_json_array_get(Field(document, "choices"), 0, &choice));
    const auto message = Field(choice, "message");
    if (result.calls->count && !result.content.size) {
      EXPECT_EQ(String(Field(message, "content")), "null");
    } else {
      EXPECT_EQ(Decode(Field(message, "content")), String(result.content));
    }
    if (result.reasoning.size) {
      EXPECT_EQ(Decode(Field(message, "reasoning_content")),
                String(result.reasoning));
    }
    EXPECT_EQ(Decode(Field(choice, "finish_reason")),
              String(result.finish_reason));
    const auto usage = Field(document, "usage");
    EXPECT_EQ(String(Field(usage, "prompt_tokens")),
              std::to_string(result.response.usage.input_tokens));
    EXPECT_EQ(String(Field(usage, "completion_tokens")),
              std::to_string(result.response.usage.output_tokens));
    EXPECT_EQ(
        String(Field(Field(usage, "prompt_tokens_details"), "cached_tokens")),
        std::to_string(result.response.usage.cached_tokens));
    if (result.calls->count) {
      const auto calls = Field(message, "tool_calls");
      iree_host_size_t count = 0;
      IREE_ASSERT_OK(iree_json_array_length(calls, &count));
      ASSERT_EQ(count, result.calls->count);
      for (iree_host_size_t i = 0; i < count; ++i) {
        iree_string_view_t call;
        IREE_ASSERT_OK(iree_json_array_get(calls, i, &call));
        const auto function = Field(call, "function");
        const auto native = loom_serve_text_tool_calls_at(result.calls, i);
        EXPECT_EQ(Decode(Field(function, "name")), String(native.name));
        EXPECT_EQ(Decode(Field(function, "arguments")),
                  String(native.arguments));
        EXPECT_NE(Decode(Field(call, "id")), String(native.id));
      }
    }
  }

  std::string ContinuedRequest(const loom_serve_text_stream_result_t& result,
                               const char* prompt = nullptr) {
    const auto messages =
        Field(iree_make_cstring_view(body_.c_str()), "messages");
    std::string continued(messages.data, messages.size - 1);
    continued +=
        ",{\"role\":\"assistant\",\"content\":" + Quote(result.content);
    if (result.reasoning.size) {
      continued += ",\"reasoning_content\":" + Quote(result.reasoning);
    }
    if (!prompt) {
      prompt = FLAG_native_text_model[0]
                   ? "Repeat your answer, with no extra words."
                   : "b";
    }
    continued += "}," + Message("user", prompt) + "]";
    std::string request = body_;
    request.replace(messages.data - body_.data(), messages.size, continued);
    return request;
  }

  // Factory borrows the process-wide physical group through model retirement.
  loom_serve_model_registration_options_t registration_options_ = {};
  // Actual managed service, with no duplicate executor or model loop.
  loom_serve_registry_t* registry_ = nullptr;
  // Stable registry slot, reused only after the prior incarnation retires.
  iree_host_size_t index_ = 0;
  // Copied deployment identity, retained across model destruction.
  std::string name_;
  // Complete supported request, reused across native and HTTP representations.
  std::string body_;
  // Outer socket reader, independent of the serialized serving owner.
  std::thread reader_;
  // Reader output, inspected only after joining the reader.
  std::string response_;
  // Reader terminal status, consumed by teardown after joining.
  iree_status_t read_status_ = iree_ok_status();
};

TEST_F(NativeTextRegistryTest, StalledNativeAndHttpPeerRetainDetachedResult) {
  NativeStream stream(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "native", &stream));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(Model()->phase, LOOM_SERVE_REGISTRATION_DECLARED);
  ASSERT_NO_FATAL_FAILURE(Prepare());
  IREE_ASSERT_OK(Open(body_, "native", &stream));
  ASSERT_NE(stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitBlock(stream));
  const auto* block = loom_serve_text_stream_peek(stream.get());
  ASSERT_GT(loom_serve_text_stream_block_count(block), 0u);
  const auto held = loom_serve_text_stream_block_at(block, 0);
  const std::string held_text = String(held.text);
  ASSERT_EQ(loom_serve_text_stream_result(stream.get()), nullptr);
  ASSERT_NO_FATAL_FAILURE(StartHttp(body_));
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
    const auto* counters = Model()->service.telemetry.counters;
    return counters->completed || counters->failed || counters->rejected;
  }));
  ASSERT_EQ(Model()->service.telemetry.counters->completed, 1u);
  EXPECT_EQ(String(held.text), held_text);
  EXPECT_EQ(loom_serve_text_stream_peek(stream.get()), block);
  EXPECT_EQ(loom_serve_text_stream_result(stream.get()), nullptr);
  const auto incarnation = Model()->incarnation;
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  NativeStream rejected(nullptr, loom_serve_text_stream_close);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        Open(body_, "later", &rejected));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(stream));
  const auto* final = loom_serve_text_stream_result(stream.get());
  ASSERT_NE(final, nullptr);
  ASSERT_NO_FATAL_FAILURE(CheckHttp(*final));
  const std::string content = String(final->content);
  if (!FLAG_native_text_model[0]) {
    EXPECT_EQ(content, Oracle::Complete("abxa", 4));
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil(
      [&] { return loom_serve_registry_count(registry_) == 0; }));
  loom_serve_telemetry_counters_t counters;
  ASSERT_TRUE(loom_serve_registry_counters(registry_, &counters));
  EXPECT_EQ(counters.requests, 2u);
  EXPECT_EQ(counters.completed, 2u);

  ASSERT_NO_FATAL_FAILURE(Register());
  EXPECT_GT(Model()->incarnation, incarnation);
  ASSERT_NO_FATAL_FAILURE(Prepare());
  NativeStream replacement(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "replacement", &replacement));
  ASSERT_NE(replacement, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(replacement));
  EXPECT_EQ(String(loom_serve_text_stream_result(replacement.get())->content),
            content);
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil(
      [&] { return loom_serve_registry_count(registry_) == 0; }));
  IREE_ASSERT_OK(loom_serve_registry_free(registry_));
  registry_ = nullptr;
  EXPECT_EQ(loom_serve_text_stream_result(stream.get()), final);
  EXPECT_EQ(String(final->content), content);
  EXPECT_EQ(String(final->response.model_name), name_);
  IREE_EXPECT_OK(loom_serve_text_stream_status(stream.get()));
}

TEST_F(NativeTextRegistryTest, AbortRetiresAcceptedWorkBeforeReleasingInput) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  NativeStream first(nullptr, loom_serve_text_stream_close);
  NativeStream second(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "first", &first));
  ASSERT_NE(first, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
    const auto phase = loom_serve_registry_snapshot(Model()).phase;
    return std::string(phase) == "execute";
  }));
  OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
    IREE_ASSERT_OK(Open(body_, "second", &second));
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(Model()->service.telemetry.counters->batches, 0u);
    // Admission has accepted the independent preparation loan. With one worker
    // it cannot start until this owner turn returns; additional workers may
    // already be reading the body. Neither case may release it at close.
    second.reset();
    IREE_ASSERT_OK(
        loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_ABORT));
  });
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil(
      [&] { return loom_serve_registry_count(registry_) == 0; }));
  loom_serve_telemetry_counters_t counters;
  ASSERT_TRUE(loom_serve_registry_counters(registry_, &counters));
  EXPECT_EQ(counters.requests, 2u);
  EXPECT_EQ(counters.cancelled, 2u);
  IREE_ASSERT_OK(loom_serve_registry_free(registry_));
  registry_ = nullptr;
  EXPECT_TRUE(loom_serve_text_stream_is_complete(first.get()));
  EXPECT_EQ(loom_serve_text_stream_result(first.get()), nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_text_stream_status(first.get()));
}

TEST_F(NativeTextRegistryTest,
       AdmissionPressureAndValidationPreserveOwnership) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  NativeStream rejected(nullptr, loom_serve_text_stream_close);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Open(body_, "invalid/session", &rejected));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Open(body_, "", &rejected, "missing"));
  EXPECT_EQ(Model()->service.telemetry.counters->requests, 0u);
  std::vector<NativeStream> waiting;
  const auto capacity = Model()->configuration.pending_requests;
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    NativeStream stream(nullptr, loom_serve_text_stream_close);
    IREE_ASSERT_OK(
        Open(body_, ("queued" + std::to_string(i)).c_str(), &stream));
    ASSERT_NE(stream, nullptr);
    waiting.push_back(std::move(stream));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        Open(body_, "queued0", &rejected));
  IREE_ASSERT_OK(Open(body_, "pressure", &rejected));
  EXPECT_EQ(rejected, nullptr);
  EXPECT_EQ(Model()->service.telemetry.counters->requests, capacity);
  waiting.clear();
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] { return Idle(); }));
  EXPECT_EQ(Model()->service.telemetry.counters->cancelled, capacity);

  NativeStream invalid(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open("{", "invalid-json", &invalid));
  ASSERT_NE(invalid, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil(
      [&] { return loom_serve_text_stream_is_complete(invalid.get()); }));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_stream_status(invalid.get()));
  EXPECT_EQ(Model()->service.telemetry.counters->rejected, 1u);
  NativeStream accepted(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "after-rejection", &accepted));
  ASSERT_NE(accepted, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(accepted));
}

TEST_F(NativeTextRegistryTest, CompleteResponseNeedsOnlyFinalDeliveryCredit) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  NativeStream stream(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(CompleteRequest(body_), "complete", &stream));
  ASSERT_NE(stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
    return loom_serve_text_stream_result(stream.get()) ||
           loom_serve_text_stream_is_complete(stream.get());
  }));
  IREE_ASSERT_OK(loom_serve_text_stream_status(stream.get()));
  const auto* result = loom_serve_text_stream_result(stream.get());
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(result->response.flags & LOOM_SERVE_TEXT_CHAT_FLAG_STREAM);
  const auto* block = loom_serve_text_stream_peek(stream.get());
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(loom_serve_text_stream_block_count(block), 0u);
  EXPECT_EQ(Model()->service.telemetry.counters->completed, 1u);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(stream));
  if (!FLAG_native_text_model[0]) {
    EXPECT_EQ(String(result->content), Oracle::Complete("abxa", 4));
  }
}

TEST_F(NativeTextRegistryTest, NativeFinalOutputMatchesHttp) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  NativeStream stream(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "native", &stream));
  ASSERT_NE(stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(StartHttp(body_));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(stream));
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] { return Idle(); }));
  const auto* result = loom_serve_text_stream_result(stream.get());
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->content.size || result->reasoning.size ||
              result->calls->count);
  ASSERT_NO_FATAL_FAILURE(CheckHttp(*result));
}

TEST_F(NativeTextRegistryTest, StoppingSealsReadyServiceAdmission) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  bool progress = false;
  auto& service = Model()->service;
  IREE_ASSERT_OK(service.advance(service.self, LOOM_SERVE_SERVICE_FLAG_STOPPING,
                                 &progress));
  NativeStream stream(nullptr, loom_serve_text_stream_close);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        Open(body_, "after-stop", &stream));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(service.telemetry.counters->requests, 0u);
  const int client = Send(body_);
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client, service, &connection));
  EXPECT_EQ(Read(client).find("HTTP/1.1 503"), 0u);
  EXPECT_EQ(service.telemetry.counters->rejected, 1u);
}

TEST_F(NativeTextRegistryTest, NativeCheckpointPressureAndRejectionStayLocal) {
  NativeControl waiting(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "retained", "conversation", &waiting));
  EXPECT_EQ(waiting, nullptr);
  ASSERT_NO_FATAL_FAILURE(Prepare());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                       "bad/name", "conversation", &waiting));
  EXPECT_EQ(waiting, nullptr);
  const auto* counters = Model()->service.telemetry.counters;
  EXPECT_EQ(counters->requests, 0u);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "retained", "conversation", &waiting));
  ASSERT_NE(waiting, nullptr);
  NativeControl pressured(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "retained", "", &pressured));
  EXPECT_EQ(pressured, nullptr);
  EXPECT_EQ(counters->requests, 1u);
  waiting.reset();
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] { return Idle(); }));
  EXPECT_EQ(counters->cancelled, 1u);

  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "retained", "", &pressured));
  ASSERT_NE(pressured, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
    return loom_serve_text_checkpoint_control_is_complete(pressured.get());
  }));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_text_checkpoint_control_status(pressured.get()));
  EXPECT_EQ(loom_serve_text_checkpoint_control_result(pressured.get()),
            nullptr);

  NativeStream first(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "conversation", &first));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(first));
  NativeStream queued(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "conversation", &queued));
  ASSERT_NE(queued, nullptr);
  NativeControl busy(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN, "busy",
                                "conversation", &busy));
  ASSERT_NE(busy, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
    return loom_serve_text_checkpoint_control_is_complete(busy.get());
  }));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_text_checkpoint_control_status(busy.get()));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(queued));
  std::string name = "retained", session = "conversation";
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                name.c_str(), session.c_str(), &waiting));
  ASSERT_NE(waiting, nullptr);
  name.assign(name.size(), 'x');
  session.assign(session.size(), 'x');
  const auto* owned = loom_serve_text_checkpoint_control_request(waiting.get());
  EXPECT_EQ(String(owned->name), "retained");
  EXPECT_EQ(String(owned->session), "conversation");
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(waiting));

  NativeControl duplicate(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "retained", "conversation", &duplicate));
  ASSERT_NE(duplicate, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
    return loom_serve_text_checkpoint_control_is_complete(duplicate.get());
  }));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_serve_text_checkpoint_control_status(duplicate.get()));
  EXPECT_EQ(counters->failed, 0u);
  EXPECT_EQ(counters->rejected, 3u);
  NativeControl released(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "retained", "", &released));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(released));
}

TEST_F(NativeTextRegistryTest, NativeCheckpointReleasePassesDrainedAdmission) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  NativeStream first(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "conversation", &first));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(first));
  NativeControl pinned(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "retained", "conversation", &pinned));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(pinned));

  Model()->closed = true;
  NativeStream waiting(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "waiting", &waiting));
  EXPECT_EQ(waiting, nullptr);
  NativeControl blocked(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "another", "conversation", &blocked));
  EXPECT_EQ(blocked, nullptr);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND,
                                "retained", "", &blocked));
  EXPECT_EQ(blocked, nullptr);

  NativeControl released(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "retained", "", &released));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(released));
  EXPECT_TRUE(Model()->closed);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "retained", "", &released));
  ASSERT_NE(released, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
    return loom_serve_text_checkpoint_control_is_complete(released.get());
  }));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_text_checkpoint_control_status(released.get()));
  EXPECT_TRUE(Model()->closed);
}

TEST_F(NativeTextRegistryTest, NativeCheckpointReleaseDoesNotPrepareModel) {
  NativeControl released(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE, "absent", "",
                     &released));
  EXPECT_EQ(released, nullptr);
  EXPECT_EQ(Model()->phase, LOOM_SERVE_REGISTRATION_DECLARED);
  EXPECT_TRUE(Model()->closed);
}

TEST_F(NativeTextRegistryTest,
       NativeCheckpointRetirementDoesNotRebindAfterReplacement) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  const auto incarnation = Model()->incarnation;
  NativeStream first(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "conversation", &first));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(first));
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] { return Idle(); }));
  NativeControl submitted(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "retained", "conversation", &submitted));
  ASSERT_NE(submitted, nullptr);
  auto& service = Model()->service;
  bool progress = false;
  IREE_ASSERT_OK(service.advance(service.self, 0, &progress));
  ASSERT_TRUE(progress);
  ASSERT_FALSE(loom_serve_text_checkpoint_control_is_complete(submitted.get()));
  ASSERT_EQ(service.snapshot(service.self).active_requests, 1u);
  submitted.reset();
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] { return Idle(); }));

  // Closing after submission suppresses delivery but preserves the accepted
  // pin. A real branch proves that the retained history was published.
  const auto continuation =
      ContinuedRequest(*loom_serve_text_stream_result(first.get()));
  NativeStream branch(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(continuation, "branch", &branch, "retained"));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(branch));
  EXPECT_GT(
      loom_serve_text_stream_result(branch.get())->response.usage.cached_tokens,
      0u);
  NativeControl surviving(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN, "copy",
                                "branch", &surviving));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(surviving));
  const auto retained =
      *loom_serve_text_checkpoint_control_result(surviving.get());
  NativeControl removed(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "retained", "", &removed));
  ASSERT_NE(removed, nullptr);
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_ABORT));
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil(
      [&] { return loom_serve_registry_count(registry_) == 0; }));
  EXPECT_TRUE(loom_serve_text_checkpoint_control_is_complete(removed.get()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      loom_serve_text_checkpoint_control_status(removed.get()));
  IREE_ASSERT_OK(loom_serve_text_checkpoint_control_status(surviving.get()));
  EXPECT_EQ(
      loom_serve_text_checkpoint_control_result(surviving.get())->position,
      retained.position);

  ASSERT_NO_FATAL_FAILURE(Register());
  ASSERT_NO_FATAL_FAILURE(Prepare());
  EXPECT_NE(Model()->incarnation, incarnation);
  NativeStream replacement(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "conversation", &replacement));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(replacement));
  NativeControl pinned(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "retained", "conversation", &pinned));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(pinned));
  removed.reset();
  surviving.reset();
  NativeControl released(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "retained", "", &released));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(released));
  EXPECT_EQ(Model()->service.telemetry.counters->failed, 0u);
}

TEST_F(NativeTextRegistryTest,
       NativeCheckpointExcludesInterruptedAndUnheardHistory) {
  if (!FLAG_native_text_model[0]) {
    body_ = Request(Message("user", "abxa"), 2);
  }
  ASSERT_NO_FATAL_FAILURE(Prepare());
  NativeStream base(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "base", &base));
  ASSERT_NE(base, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(base));
  NativeControl pin(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "pre-turn", "base", &pin));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(pin));
  const auto position =
      loom_serve_text_checkpoint_control_result(pin.get())->position;
  ASSERT_GT(position, 0u);
  pin.reset();

  const auto intervening = ContinuedRequest(
      *loom_serve_text_stream_result(base.get()),
      FLAG_native_text_model[0]
          ? "Explain how rain reaches rivers in exactly eight complete "
            "sentences. Use plain prose without headings or bullets."
          : "b");
  const auto messages =
      Field(iree_make_cstring_view(intervening.c_str()), "messages");
  std::string corrected = intervening;
  corrected.insert(
      messages.data - intervening.data() + messages.size - 1,
      "," +
          Message("user", FLAG_native_text_model[0]
                              ? "Ignore that last request. Reply with exactly "
                                "the two words blue lantern, without "
                                "punctuation or any extra words."
                              : "x"));
  NativeStream interrupted(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(intervening, "interrupted", &interrupted, "pre-turn"));
  ASSERT_NE(interrupted, nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitBlock(interrupted));
  ASSERT_FALSE(loom_serve_text_stream_is_complete(interrupted.get()));
  const auto* block = loom_serve_text_stream_peek(interrupted.get());
  std::string observed;
  for (iree_host_size_t i = 0; i < loom_serve_text_stream_block_count(block);
       ++i) {
    const auto span = loom_serve_text_stream_block_at(block, i);
    if (span.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT) {
      observed.append(span.text.data, span.text.size);
    }
  }
  ASSERT_FALSE(observed.empty());
  interrupted.reset();
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] { return Idle(); }));
  EXPECT_EQ(Model()->service.telemetry.counters->cancelled, 1u);

  NativeStream fresh(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(corrected, "fresh", &fresh));
  ASSERT_NE(fresh, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(fresh));
  const auto* expected = loom_serve_text_stream_result(fresh.get());
  ASSERT_EQ(expected->calls->count, 0u);
  EXPECT_EQ(expected->response.usage.cached_tokens, 0u);
  if (FLAG_native_text_model[0]) {
    EXPECT_EQ(String(expected->content), "blue lantern");
    EXPECT_EQ(String(expected->finish_reason), "stop");
  }
  const auto check_corrected = [&](const NativeStream& stream,
                                   iree_host_size_t cached_tokens) {
    const auto* actual = loom_serve_text_stream_result(stream.get());
    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(String(actual->content), String(expected->content));
    EXPECT_EQ(String(actual->reasoning), String(expected->reasoning));
    EXPECT_EQ(String(actual->finish_reason), String(expected->finish_reason));
    EXPECT_EQ(actual->calls->count, 0u);
    EXPECT_EQ(actual->response.usage.input_tokens,
              expected->response.usage.input_tokens);
    EXPECT_EQ(actual->response.usage.output_tokens,
              expected->response.usage.output_tokens);
    EXPECT_EQ(actual->response.usage.cached_tokens, cached_tokens);
  };
  NativeStream retained(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(corrected, "retained", &retained, "pre-turn"));
  ASSERT_NE(retained, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(retained));
  ASSERT_NO_FATAL_FAILURE(check_corrected(retained, position));

  // A fully generated answer can still be unheard. Excluding that assistant
  // item must not reuse its tail, even when the named session retains it.
  NativeStream unheard(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(intervening, "unheard", &unheard, "pre-turn"));
  ASSERT_NE(unheard, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(unheard));
  EXPECT_EQ(String(loom_serve_text_stream_result(unheard.get())->content)
                .find(observed),
            0u);
  NativeStream replaced(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(corrected, "unheard", &replaced));
  ASSERT_NE(replaced, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(replaced));
  ASSERT_NO_FATAL_FAILURE(check_corrected(replaced, 0));
  retained.reset();
  IREE_ASSERT_OK(Open(corrected, "retained-again", &retained, "pre-turn"));
  ASSERT_NE(retained, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(retained));
  ASSERT_NO_FATAL_FAILURE(check_corrected(retained, position));

  NativeStream incompatible(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "incompatible", &incompatible, "pre-turn"));
  ASSERT_NE(incompatible, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil(
      [&] { return loom_serve_text_stream_is_complete(incompatible.get()); }));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_stream_status(incompatible.get()));
  EXPECT_EQ(loom_serve_text_stream_result(incompatible.get()), nullptr);
  NativeControl release(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "pre-turn", "", &release));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(release));
  EXPECT_EQ(loom_serve_text_checkpoint_control_result(release.get())->position,
            position);
  EXPECT_EQ(Model()->service.telemetry.counters->failed, 0u);
  EXPECT_EQ(Model()->service.telemetry.counters->rejected, 1u);
  RecordProperty("interrupted_content", observed);
  RecordProperty("corrected_content", String(expected->content));
  RecordProperty("checkpoint_position", std::to_string(position));
}

TEST_F(NativeTextRegistryTest,
       NativeCompletionPreservesPinnedCanonicalHistory) {
  if (!FLAG_native_text_model[0]) {
    body_ = Request(Message("user", "abxa"), 2);
  }
  ASSERT_NO_FATAL_FAILURE(Prepare());
  NativeStream first(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body_, "conversation", &first));
  ASSERT_NE(first, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(first));
  const auto* final = loom_serve_text_stream_result(first.get());
  ASSERT_NE(final, nullptr);
  ASSERT_EQ(final->calls->count, 0u);
  const std::string answer = String(final->content);
  if (!FLAG_native_text_model[0]) {
    EXPECT_EQ(answer, Oracle::Complete("abxa", 2));
  }
  NativeControl pin(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
                                "retained", "conversation", &pin));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(pin));
  const auto* pinned = loom_serve_text_checkpoint_control_result(pin.get());
  EXPECT_GT(pinned->position, 0u);
  EXPECT_EQ(pinned->host_bytes, 0u);

  NativeControl suspended(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND,
                                "retained", "", &suspended));
  ASSERT_NE(suspended, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] {
    return loom_serve_text_checkpoint_control_is_complete(suspended.get());
  }));
  if (loom_serve_device_memory_pool(loom_serve_device_group_at(group_, 0))) {
    IREE_ASSERT_OK(loom_serve_text_checkpoint_control_status(suspended.get()));
    const auto* result =
        loom_serve_text_checkpoint_control_result(suspended.get());
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->position, pinned->position);
    EXPECT_GT(result->host_bytes, 0u);
  } else {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        loom_serve_text_checkpoint_control_status(suspended.get()));
    EXPECT_EQ(loom_serve_text_checkpoint_control_result(suspended.get()),
              nullptr);
  }
  const auto continuation = ContinuedRequest(*final);
  NativeStream branch(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(continuation, "branch", &branch, "retained"));
  ASSERT_NE(branch, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(branch));
  const auto* resumed = loom_serve_text_stream_result(branch.get());
  ASSERT_NE(resumed, nullptr);
  EXPECT_GT(resumed->response.usage.cached_tokens, 0u);
  if (!FLAG_native_text_model[0]) {
    EXPECT_EQ(resumed->response.usage.cached_tokens, 5u);
    EXPECT_EQ(String(resumed->content),
              Oracle::Complete("abxa" + answer + "#b", 2));
  }
  ASSERT_NO_FATAL_FAILURE(StartHttp(continuation, "conversation"));
  ASSERT_NO_FATAL_FAILURE(AdvanceRegistryUntil([&] { return Idle(); }));
  ASSERT_NO_FATAL_FAILURE(CheckHttp(*resumed));
  EXPECT_EQ(String(final->content), answer);
  NativeControl released(nullptr, loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(OpenCheckpoint(LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
                                "retained", "", &released));
  ASSERT_NO_FATAL_FAILURE(CompleteCheckpoint(released));
  EXPECT_EQ(loom_serve_text_checkpoint_control_result(released.get())->position,
            pinned->position);
  EXPECT_EQ(
      loom_serve_text_checkpoint_control_result(released.get())->host_bytes,
      0u);
}

}  // namespace
