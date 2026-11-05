// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/image/telemetry.h"

#include <array>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/telemetry/field.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/serve/telemetry/record.h"

namespace {

TEST(ImageTelemetryTest, SamplesCompletedWorkWithoutAnEventStream) {
  const loom_service_image_totals_t totals = {
      1, UINT64_MAX, {3, 4, 5, 6, 7, 8, 9, 10}};
  IREE_ASSERT_OK(loom_service_image_sample_totals(
      &totals, {nullptr, [](void*, const loom_runtime_telemetry_topic_t* topic,
                            const uint64_t* values) {
                  EXPECT_STREQ(topic->name, "image");
                  EXPECT_EQ(topic->version, 1u);
                  EXPECT_EQ(topic->count, 10u);
                  for (size_t i = 0; i < topic->count; ++i) {
                    EXPECT_EQ(values[i], i == 1 ? UINT64_MAX : i + 1);
                    EXPECT_EQ(topic->fields[i].kind,
                              LOOM_RUNTIME_TELEMETRY_COUNTER);
                  }
                  EXPECT_STREQ(topic->fields[1].unit, "pixels");
                  EXPECT_STREQ(topic->fields[6].name, "execution");
                  EXPECT_STREQ(topic->fields[6].unit, "nanoseconds");
                  EXPECT_STREQ(topic->fields[9].unit, "bytes");
                  return iree_ok_status();
                }}));
}

TEST(ImageTelemetryTest, PreparedFieldsPreserveSignedTagsAfterRetirement) {
  loom_runtime_telemetry_stream_t* stream = nullptr;
  IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
      2, iree_allocator_system(), &stream));
  for (uint64_t request = 1; request <= 2; ++request) {
    const loom_service_image_prepared_t prepared = {
        1, request == 1 ? INT64_MIN : INT64_MAX, 48};
    loom_service_image_record_prepared(stream, request, &prepared);
  }
  std::array<loom_runtime_telemetry_record_t, 2> records;
  uint64_t cursor = 0;
  for (auto& record : records) {
    ASSERT_EQ(loom_runtime_telemetry_stream_read(stream, &cursor, &record),
              LOOM_RUNTIME_TELEMETRY_READ_RECORD);
  }
  loom_runtime_telemetry_stream_free(stream);
  for (const auto& record : records) {
    int64_t expected_tag = record.request == 1 ? INT64_MIN : INT64_MAX;
    IREE_ASSERT_OK(record.event->read_fields(
        &record,
        {&expected_tag, [](void* self, iree_host_size_t count,
                           const loom_runtime_telemetry_field_t* fields) {
           EXPECT_EQ(count, 3u);
           if (count == 3) {
             EXPECT_STREQ(fields[0].name, "stage");
             EXPECT_EQ(fields[0].value.uint64_value, 1u);
             EXPECT_STREQ(fields[1].name, "tag");
             EXPECT_EQ(fields[1].kind, LOOM_RUNTIME_TELEMETRY_FIELD_INT64);
             EXPECT_EQ(fields[1].value.int64_value,
                       *static_cast<int64_t*>(self));
             EXPECT_EQ(fields[2].value.uint64_value, 48u);
           }
           return iree_ok_status();
         }}));
    iree_status_t failure = record.event->read_fields(
        &record, {nullptr, [](void*, iree_host_size_t,
                              const loom_runtime_telemetry_field_t*) {
                    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "observer output is full");
                  }});
    EXPECT_EQ(iree_status_code(failure), IREE_STATUS_RESOURCE_EXHAUSTED);
    iree_status_free(failure);
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    IREE_ASSERT_OK(loom_serve_telemetry_record_format(0, &record, &output));
    iree_string_view_t tag;
    IREE_ASSERT_OK(iree_json_lookup_object_value(
        iree_string_builder_view(&output), IREE_SV("tag"), &tag));
    int64_t value = 0;
    IREE_ASSERT_OK(iree_json_parse_int64(tag, &value));
    EXPECT_EQ(value, expected_tag);
    iree_string_builder_deinitialize(&output);
  }
}

}  // namespace
