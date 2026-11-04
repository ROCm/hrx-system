// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/json.h"
#include "loom/runtime/telemetry/field.h"
#include "loom/serve/server/http/logs.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/telemetry.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

const loom_runtime_telemetry_field_t* FindField(
    iree_host_size_t count, const loom_runtime_telemetry_field_t* fields,
    const char* name) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (std::strcmp(fields[i].name, name) == 0) {
      return &fields[i];
    }
  }
  return nullptr;
}

void ExpectEpochFields(const loom_serve_text_epoch_record_t& expected,
                       iree_host_size_t count,
                       const loom_runtime_telemetry_field_t* fields) {
  ASSERT_EQ(count, 16u);
  const auto* slots = FindField(count, fields, "token_slots");
  ASSERT_NE(slots, nullptr);
  EXPECT_EQ(slots->kind, expected.token_slots
                             ? LOOM_RUNTIME_TELEMETRY_FIELD_UINT64
                             : LOOM_RUNTIME_TELEMETRY_FIELD_NULL);
  if (expected.token_slots) {
    EXPECT_EQ(slots->value.uint64_value, expected.token_slots);
  }
  const auto* scheduler = FindField(count, fields, "scheduler");
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->kind, LOOM_RUNTIME_TELEMETRY_FIELD_STRING);
  const auto mode = expected.schedule_mode;
  EXPECT_TRUE(iree_string_view_equal(
      scheduler->value.string_value,
      mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED    ? IREE_SV("packed")
      : mode == LOOM_SERVE_TEXT_SCHEDULE_MATCHED ? IREE_SV("matched")
                                                 : IREE_SV("isolated")));
  const auto* draft = FindField(count, fields, "draft");
  ASSERT_NE(draft, nullptr);
  ASSERT_EQ(draft->kind, LOOM_RUNTIME_TELEMETRY_FIELD_OBJECT);
  const auto* proposed =
      FindField(draft->value.object_value.count,
                draft->value.object_value.fields, "proposed_tokens");
  ASSERT_NE(proposed, nullptr);
  EXPECT_EQ(proposed->kind, LOOM_RUNTIME_TELEMETRY_FIELD_UINT64);
  EXPECT_EQ(proposed->value.uint64_value,
            expected.verifications * expected.draft_depth);
}

TEST_F(SourceTextServiceTest, CopiedTelemetryOutlivesOwnersAndNameReuse) {
  struct Batch {
    // Original construction identity, independent of the reused display name.
    loom_serve_telemetry_identity_t identity;
    // Copied source association, never looked up after owner retirement.
    std::string name;
    // Three bounded polls of an eight-record stream, not borrowed ring slots.
    std::vector<loom_serve_telemetry_record_t> records;
  };
  std::vector<Batch> batches;
  for (const auto mode :
       {LOOM_SERVE_TEXT_SCHEDULE_PACKED, LOOM_SERVE_TEXT_SCHEDULE_ISOLATED,
        LOOM_SERVE_TEXT_SCHEDULE_MATCHED}) {
    SCOPED_TRACE(mode);
    loom_serve_telemetry_stream_t* stream_value = nullptr;
    IREE_ASSERT_OK(loom_serve_telemetry_stream_allocate(8, host_allocator_,
                                                        &stream_value));
    std::unique_ptr<loom_serve_telemetry_stream_t,
                    decltype(&loom_serve_telemetry_stream_free)>
        stream(stream_value, loom_serve_telemetry_stream_free);
    auto options = ServiceOptions();
    options.schedule_mode = mode;
    loom_serve_text_service_t* service_value = nullptr;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(
        model_, &options, stream.get(), host_allocator_, &service_value));
    auto destroy = [](loom_serve_text_service_t* service) {
      loom_serve_text_service_free(service, IREE_STATUS_OK);
    };
    std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
        service_value, destroy);
    const auto interface = loom_serve_text_service_interface(service.get());
    Observe(interface);
    Batch batch{loom_serve_telemetry_stream_identity(stream.get()),
                std::string(options.name.data, options.name.size),
                {}};
    batch.records.reserve(24);
    uint64_t cursor = 0;
    for (const std::string prompt : {"abxa", "xab", "baxab"}) {
      const int client = Send(Request(Message("user", prompt), 2));
      loom_serve_http_connection_t* connection = nullptr;
      ASSERT_NO_FATAL_FAILURE(Accept(client, interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
        return !interface.is_pending(interface.self) &&
               !interface.is_active(interface.self) &&
               !interface.snapshot(interface.self).queued_requests;
      }));
      EXPECT_EQ(Content(Read(client)), Oracle::Complete(prompt, 2));
      const auto head = loom_serve_telemetry_stream_head(stream.get());
      while (cursor < head) {
        loom_serve_telemetry_record_t record;
        const auto result =
            loom_serve_telemetry_stream_read(stream.get(), &cursor, &record);
        if (result == LOOM_SERVE_TELEMETRY_READ_GAP) {
          continue;
        }
        ASSERT_EQ(result, LOOM_SERVE_TELEMETRY_READ_RECORD);
        batch.records.push_back(record);
      }
    }
    EXPECT_EQ(interface.telemetry.counters->completed, 3u);
    uint64_t stalled_cursor = 0;
    loom_serve_telemetry_record_t unused;
    ASSERT_EQ(loom_serve_telemetry_stream_read(stream.get(), &stalled_cursor,
                                               &unused),
              LOOM_SERVE_TELEMETRY_READ_GAP);
    EXPECT_GT(stalled_cursor, 0u);
    iree_string_builder_t output;
    iree_string_builder_initialize(host_allocator_, &output);
    IREE_ASSERT_OK(loom_serve_http_logs(
        stream.get(), interface.telemetry.counters, options.name, {}, &output));
    const auto json = iree_string_builder_view(&output);
    const std::string response(json.data, json.size);
    EXPECT_NE(response.find(R"("event":"gap")"), std::string::npos);
    EXPECT_NE(response.find(R"("completed":3)"), std::string::npos);
    iree_string_builder_deinitialize(&output);
    interface.set_observer(interface.self, {});
    service.reset();
    stream.reset();
    batches.push_back(std::move(batch));
  }
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
  size_t epochs = 0, ends = 0;
  for (size_t i = 0; i < batches.size(); ++i) {
    EXPECT_EQ(batches[i].name, "fixture-text");
    if (i) {
      EXPECT_NE(batches[i].identity.ordinal, batches[i - 1].identity.ordinal);
    }
    for (const auto& record : batches[i].records) {
      if (record.event == &LOOM_SERVE_TEXT_EPOCH) {
        loom_serve_text_epoch_record_t expected;
        std::memcpy(&expected, record.data.words, sizeof(expected));
        IREE_ASSERT_OK(record.event->read_fields(
            &record,
            {&expected, [](void* self, iree_host_size_t count,
                           const loom_runtime_telemetry_field_t* fields) {
               ExpectEpochFields(
                   *static_cast<loom_serve_text_epoch_record_t*>(self), count,
                   fields);
               return iree_ok_status();
             }}));
        ++epochs;
      } else if (record.event == &LOOM_SERVE_TELEMETRY_REQUEST_END) {
        ++ends;
      } else {
        continue;
      }
      iree_string_builder_t output;
      iree_string_builder_initialize(host_allocator_, &output);
      IREE_ASSERT_OK(loom_serve_telemetry_record_format(0, &record, &output));
      iree_string_view_t event;
      IREE_ASSERT_OK(iree_json_lookup_object_value(
          iree_string_builder_view(&output), IREE_SV("event"), &event));
      EXPECT_TRUE(
          iree_string_view_equal(event, record.event == &LOOM_SERVE_TEXT_EPOCH
                                            ? IREE_SV("epoch")
                                            : IREE_SV("request_end")));
      iree_string_builder_deinitialize(&output);
    }
  }
  EXPECT_GE(epochs, 3u);
  EXPECT_EQ(ends, 9u);
}

}  // namespace
