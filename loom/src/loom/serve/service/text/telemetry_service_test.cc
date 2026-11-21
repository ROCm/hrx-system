// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/base/internal/json.h"
#include "loom/runtime/telemetry/field.h"
#include "loom/serve/server/http/logs.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/telemetry.h"
#include "loom/serve/telemetry/record.h"

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

void ExpectEpochFields(const loom_service_text_epoch_record_t& expected,
                       iree_host_size_t count,
                       const loom_runtime_telemetry_field_t* fields) {
  const auto* scheduled = FindField(count, fields, "scheduled_tokens");
  ASSERT_NE(scheduled, nullptr);
  ASSERT_EQ(scheduled->kind, LOOM_RUNTIME_TELEMETRY_FIELD_UINT64);
  EXPECT_EQ(scheduled->value.uint64_value, expected.scheduled_tokens);
  const auto* scheduler = FindField(count, fields, "scheduler");
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->kind, LOOM_RUNTIME_TELEMETRY_FIELD_STRING);
  const auto mode = expected.schedule_mode;
  EXPECT_TRUE(iree_string_view_equal(
      scheduler->value.string_value,
      mode == LOOM_SERVICE_TEXT_SCHEDULE_PACKED    ? IREE_SV("packed")
      : mode == LOOM_SERVICE_TEXT_SCHEDULE_MATCHED ? IREE_SV("matched")
                                                   : IREE_SV("isolated")));
  const auto* draft = FindField(count, fields, "draft");
  ASSERT_NE(draft, nullptr);
  ASSERT_EQ(draft->kind, LOOM_RUNTIME_TELEMETRY_FIELD_OBJECT);
  const auto* proposed =
      FindField(draft->value.object_value.count,
                draft->value.object_value.fields, "proposed_tokens");
  ASSERT_NE(proposed, nullptr);
  EXPECT_EQ(proposed->kind, LOOM_RUNTIME_TELEMETRY_FIELD_UINT64);
  EXPECT_EQ(proposed->value.uint64_value, expected.proposed_tokens);
}

TEST_F(SourceTextServiceTest, ActualDraftCountsReachSpansEpochAndHeartbeat) {
  loom_serve_text_telemetry_t telemetry = {};
  IREE_ASSERT_OK(loom_serve_text_telemetry_initialize(nullptr, host_allocator_,
                                                      &telemetry));
  loom_serve_text_heartbeat_t heartbeat;
  loom_serve_text_heartbeat_initialize(&heartbeat);
  loom_serve_text_session_t sessions[3] = {};
  loom_serve_text_session_t* session_table[3];
  for (size_t i = 0; i < IREE_ARRAYSIZE(sessions); ++i) {
    sessions[i].row = loom_serve_text_model_row(model_, i);
    sessions[i].lifetime.id = 100 + i;
    session_table[i] = &sessions[i];
  }
  // Completed source publication: unequal candidate counts, ordinary input,
  // and two verifications for one request. The ceiling cannot recover these.
  loom_serve_text_span_t spans[] = {
      {.row_index = 2, .token_count = 4},
      {.row_index = 0, .token_count = 4},
      {.row_index = 1, .token_count = 7},
  };
  loom_serve_text_result_t results[] = {
      {
          .consumed_count = 2,
          .output_count = 2,
          .verification_count = 1,
          .proposal_count = 3,
      },
      {.consumed_count = 4, .known_count = 4, .output_count = 1},
      {
          .consumed_count = 3,
          .output_count = 3,
          .verification_count = 2,
          .proposal_count = 8,
      },
  };
  iree_host_size_t positions[] = {10, 20, 30};
  uint32_t limits[] = {5, 0, 9};
  bool prefill[] = {false, true, false};
  const loom_serve_text_epoch_t cohort = {
      .count = IREE_ARRAYSIZE(spans),
      .start = {.serial = 7, .time = iree_time_now()},
      .device_epochs = 2,
      .continuation_token_count = 8,
      .spans = spans,
      .results = results,
      .positions = positions,
      .output_limits = limits,
      .prefill = prefill,
  };
  uint64_t cursor = loom_runtime_telemetry_stream_head(telemetry.stream);
  loom_serve_text_telemetry_epoch(
      &telemetry, &heartbeat, LOOM_SERVICE_TEXT_SCHEDULE_PACKED,
      LOOM_SERVICE_TEXT_PACKING_MIXED, &cohort, session_table);
  for (size_t i = 0; i < IREE_ARRAYSIZE(spans); ++i) {
    loom_runtime_telemetry_record_t record;
    ASSERT_EQ(
        loom_runtime_telemetry_stream_read(telemetry.stream, &cursor, &record),
        LOOM_RUNTIME_TELEMETRY_READ_RECORD);
    ASSERT_EQ(record.event, &LOOM_SERVICE_TEXT_SPAN);
    loom_service_text_span_record_t span;
    std::memcpy(&span, record.data.words, sizeof(span));
    EXPECT_EQ(span.session, sessions[spans[i].row_index].lifetime.id);
    EXPECT_EQ(span.proposed_tokens, results[i].proposal_count);
  }
  loom_runtime_telemetry_record_t record;
  ASSERT_EQ(
      loom_runtime_telemetry_stream_read(telemetry.stream, &cursor, &record),
      LOOM_RUNTIME_TELEMETRY_READ_RECORD);
  ASSERT_EQ(record.event, &LOOM_SERVICE_TEXT_EPOCH);
  loom_service_text_epoch_record_t epoch;
  std::memcpy(&epoch, record.data.words, sizeof(epoch));
  EXPECT_EQ(epoch.scheduled_tokens, 23u);
  EXPECT_EQ(epoch.proposed_tokens, 11u);
  EXPECT_EQ(epoch.verifications, 3u);
  EXPECT_EQ(epoch.accepted_draft_inputs, 2u);
  const auto snapshot = loom_serve_text_heartbeat_read(&heartbeat);
  EXPECT_EQ(snapshot.draft.proposed_tokens, 11u);
  EXPECT_EQ(snapshot.draft.accepted_inputs, 2u);

  // An older publisher supplies progress for a verification but no proposed
  // count. Accepted inputs remain exact even across the reporting boundary.
  results[0].proposal_count = UINT32_MAX;
  loom_serve_text_telemetry_epoch(
      &telemetry, &heartbeat, LOOM_SERVICE_TEXT_SCHEDULE_PACKED,
      LOOM_SERVICE_TEXT_PACKING_MIXED, &cohort, session_table);
  for (size_t i = 0; i <= IREE_ARRAYSIZE(spans); ++i) {
    ASSERT_EQ(
        loom_runtime_telemetry_stream_read(telemetry.stream, &cursor, &record),
        LOOM_RUNTIME_TELEMETRY_READ_RECORD);
  }
  ASSERT_EQ(record.event, &LOOM_SERVICE_TEXT_EPOCH);
  iree_string_builder_t output;
  iree_string_builder_initialize(host_allocator_, &output);
  IREE_EXPECT_OK(loom_serve_telemetry_record_format(0, &record, &output));
  const auto view = iree_string_builder_view(&output);
  const std::string json(view.data, view.size);
  EXPECT_NE(json.find(R"("proposed_tokens":null)"), std::string::npos);
  iree_string_builder_deinitialize(&output);

  results[0].proposal_count = 3;
  loom_serve_text_telemetry_epoch(
      &telemetry, &heartbeat, LOOM_SERVICE_TEXT_SCHEDULE_PACKED,
      LOOM_SERVICE_TEXT_PACKING_MIXED, &cohort, session_table);
  IREE_EXPECT_OK(loom_serve_text_heartbeat_sample(
      &heartbeat,
      {nullptr, [](void*, const loom_runtime_telemetry_topic_t* topic,
                   const uint64_t* values) {
         bool found_accepted = false;
         for (iree_host_size_t i = 0; i < topic->count; ++i) {
           EXPECT_STRNE(topic->fields[i].name, "draft_proposed_tokens");
           if (std::strcmp(topic->fields[i].name, "draft_accepted_inputs") ==
               0) {
             EXPECT_EQ(values[i], 6u);
             found_accepted = true;
           }
         }
         EXPECT_TRUE(found_accepted);
         return iree_ok_status();
       }}));
  loom_serve_text_heartbeat_deinitialize(&heartbeat);
  loom_serve_text_telemetry_deinitialize(&telemetry);
}

TEST_F(SourceTextServiceTest, CopiedTelemetryOutlivesOwnersAndNameReuse) {
  struct Batch {
    // Original construction identity, independent of the reused display name.
    loom_runtime_telemetry_identity_t identity;
    // Copied source association, never looked up after owner retirement.
    std::string name;
    // Three bounded polls of an eight-record stream, not borrowed ring slots.
    std::vector<loom_runtime_telemetry_record_t> records;
  };
  std::vector<Batch> batches;
  for (const auto mode :
       {LOOM_SERVICE_TEXT_SCHEDULE_PACKED, LOOM_SERVICE_TEXT_SCHEDULE_ISOLATED,
        LOOM_SERVICE_TEXT_SCHEDULE_MATCHED}) {
    SCOPED_TRACE(mode);
    loom_runtime_telemetry_stream_t* stream_value = nullptr;
    IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(8, host_allocator_,
                                                          &stream_value));
    std::unique_ptr<loom_runtime_telemetry_stream_t,
                    decltype(&loom_runtime_telemetry_stream_free)>
        stream(stream_value, loom_runtime_telemetry_stream_free);
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
    Batch batch{loom_runtime_telemetry_stream_identity(stream.get()),
                std::string(options.name.data, options.name.size),
                {}};
    batch.records.reserve(24);
    uint64_t cursor = 0;
    for (const std::string prompt : {"abxa", "xab", "baxab"}) {
      const int client = Send(Request(Message("user", prompt), 2));
      loom_serve_http_connection_t* connection = nullptr;
      ASSERT_NO_FATAL_FAILURE(Accept(client, interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
        return !interface.runtime.vtable->is_pending(interface.runtime.self) &&
               !interface.runtime.vtable->is_active(interface.runtime.self) &&
               !interface.snapshot(interface.runtime.self).queued_requests;
      }));
      EXPECT_EQ(Content(Read(client)), Oracle::Complete(prompt, 2));
      const auto head = loom_runtime_telemetry_stream_head(stream.get());
      while (cursor < head) {
        loom_runtime_telemetry_record_t record;
        const auto result =
            loom_runtime_telemetry_stream_read(stream.get(), &cursor, &record);
        if (result == LOOM_RUNTIME_TELEMETRY_READ_GAP) {
          continue;
        }
        ASSERT_EQ(result, LOOM_RUNTIME_TELEMETRY_READ_RECORD);
        batch.records.push_back(record);
      }
    }
    EXPECT_EQ(interface.telemetry.counters->completed, 3u);
    uint64_t stalled_cursor = 0;
    loom_runtime_telemetry_record_t unused;
    ASSERT_EQ(loom_runtime_telemetry_stream_read(stream.get(), &stalled_cursor,
                                                 &unused),
              LOOM_RUNTIME_TELEMETRY_READ_GAP);
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
    interface.runtime.vtable->set_observer(interface.runtime.self, {});
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
      if (record.event == &LOOM_SERVICE_TEXT_EPOCH) {
        loom_service_text_epoch_record_t expected;
        std::memcpy(&expected, record.data.words, sizeof(expected));
        IREE_ASSERT_OK(record.event->read_fields(
            &record,
            {&expected, [](void* self, iree_host_size_t count,
                           const loom_runtime_telemetry_field_t* fields) {
               ExpectEpochFields(
                   *static_cast<loom_service_text_epoch_record_t*>(self), count,
                   fields);
               return iree_ok_status();
             }}));
        ++epochs;
      } else if (record.event == &LOOM_SERVICE_TELEMETRY_REQUEST_END) {
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
          iree_string_view_equal(event, record.event == &LOOM_SERVICE_TEXT_EPOCH
                                            ? IREE_SV("epoch")
                                            : IREE_SV("request_end")));
      iree_string_builder_deinitialize(&output);
    }
  }
  EXPECT_GE(epochs, 3u);
  EXPECT_EQ(ends, 9u);
}

}  // namespace
