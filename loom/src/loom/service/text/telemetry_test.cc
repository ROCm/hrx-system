// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/text/telemetry.h"

#include <cstring>
#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/service/text/schedule.h"

namespace {

TEST(TextTelemetry, EpochPreservesModeAvailabilityAndDerivedDraftWork) {
  for (const auto mode :
       {LOOM_SERVICE_TEXT_SCHEDULE_PACKED, LOOM_SERVICE_TEXT_SCHEDULE_ISOLATED,
        LOOM_SERVICE_TEXT_SCHEDULE_MATCHED}) {
    for (const auto packing : {LOOM_SERVICE_TEXT_PACKING_MIXED,
                               LOOM_SERVICE_TEXT_PACKING_SEPARATE}) {
      loom_runtime_telemetry_stream_t* stream = nullptr;
      IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
          1, iree_allocator_system(), &stream));
      loom_service_text_epoch_record_t epoch = {
          .epoch = 7,
          .spans = 2,
          .token_capacity = 8,
          .token_slots = mode == LOOM_SERVICE_TEXT_SCHEDULE_PACKED ? 16u : 0u,
          .span_capacity = 2,
          .prefill_tokens = 6,
          .decode_tokens = 2,
          .output_tokens = 4,
          .traversals = 2,
          .device_epochs = 2,
          .verifications = 2,
          .accepted_draft_inputs = 4,
          .schedule_mode = static_cast<uint8_t>(mode),
          .packing_mode = static_cast<uint8_t>(packing),
          .draft_depth = 3,
      };
      loom_service_text_record(stream, &LOOM_SERVICE_TEXT_EPOCH, 0, &epoch,
                               sizeof(epoch));
      uint64_t cursor = 0;
      loom_runtime_telemetry_record_t record;
      ASSERT_EQ(loom_runtime_telemetry_stream_read(stream, &cursor, &record),
                LOOM_RUNTIME_TELEMETRY_READ_RECORD);
      loom_runtime_telemetry_stream_free(stream);
      epoch = {};
      struct Expected {
        // The same semantic enum as the source scheduler, not copied encodings.
        loom_service_text_schedule_mode_t mode;
        // Source packing policy, independent of execution mode.
        loom_service_text_packing_mode_t packing;
      } expected{mode, packing};
      IREE_ASSERT_OK(record.event->read_fields(
          &record,
          {&expected, [](void* self, iree_host_size_t count,
                         const loom_runtime_telemetry_field_t* fields) {
             const auto& expected = *static_cast<Expected*>(self);
             EXPECT_EQ(count, 16u);
             for (iree_host_size_t i = 0; i < count; ++i) {
               const auto& field = fields[i];
               if (std::strcmp(field.name, "token_slots") == 0) {
                 EXPECT_EQ(field.kind,
                           expected.mode == LOOM_SERVICE_TEXT_SCHEDULE_PACKED
                               ? LOOM_RUNTIME_TELEMETRY_FIELD_UINT64
                               : LOOM_RUNTIME_TELEMETRY_FIELD_NULL);
                 if (field.kind == LOOM_RUNTIME_TELEMETRY_FIELD_UINT64) {
                   EXPECT_EQ(field.value.uint64_value, 16u);
                 }
               } else if (std::strcmp(field.name, "scheduler") == 0) {
                 EXPECT_EQ(field.kind, LOOM_RUNTIME_TELEMETRY_FIELD_STRING);
                 EXPECT_TRUE(iree_string_view_equal(
                     field.value.string_value,
                     expected.mode == LOOM_SERVICE_TEXT_SCHEDULE_PACKED
                         ? IREE_SV("packed")
                     : expected.mode == LOOM_SERVICE_TEXT_SCHEDULE_MATCHED
                         ? IREE_SV("matched")
                         : IREE_SV("isolated")));
               } else if (std::strcmp(field.name, "packing") == 0) {
                 EXPECT_TRUE(iree_string_view_equal(
                     field.value.string_value,
                     expected.packing == LOOM_SERVICE_TEXT_PACKING_SEPARATE
                         ? IREE_SV("separate")
                         : IREE_SV("mixed")));
               } else if (std::strcmp(field.name, "draft") == 0) {
                 EXPECT_EQ(field.kind, LOOM_RUNTIME_TELEMETRY_FIELD_OBJECT);
                 EXPECT_EQ(field.value.object_value.count, 3u);
                 const auto* draft = field.value.object_value.fields;
                 EXPECT_STREQ(draft[1].name, "proposed_tokens");
                 EXPECT_EQ(draft[1].value.uint64_value, 6u);
                 EXPECT_EQ(draft[2].value.uint64_value, 4u);
               }
             }
             return iree_ok_status();
           }}));
    }
  }
}

}  // namespace
