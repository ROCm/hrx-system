// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/telemetry.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ParameterTelemetry, NativeActivityOutlivesItsOwnerAndStream) {
  loom_runtime_telemetry_stream_t* stream = nullptr;
  IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
      1, iree_allocator_system(), &stream));
  loom_runtime_parameter_activity_t activity = {3, 1, UINT64_MAX, 40, 5};
  loom_runtime_parameter_record_activity(stream, 17, &activity);
  activity = {};
  uint64_t cursor = 0;
  loom_runtime_telemetry_record_t record;
  ASSERT_EQ(loom_runtime_telemetry_stream_read(stream, &cursor, &record),
            LOOM_RUNTIME_TELEMETRY_READ_RECORD);
  loom_runtime_telemetry_stream_free(stream);
  EXPECT_EQ(record.time_ns, 17);
  EXPECT_STREQ(record.event->name, "parameter_activity");
  IREE_ASSERT_OK(record.event->read_fields(
      &record,
      {nullptr, [](void*, iree_host_size_t count,
                   const loom_runtime_telemetry_field_t* fields) {
         const std::array<const char*, 5> names = {
             "activations", "deactivations", "read_bytes", "prepared_bytes",
             "groups"};
         const std::array<uint64_t, 5> values = {3, 1, UINT64_MAX, 40, 5};
         EXPECT_EQ(count, names.size());
         for (size_t i = 0; i < count && i < names.size(); ++i) {
           EXPECT_STREQ(fields[i].name, names[i]);
           EXPECT_EQ(fields[i].kind, LOOM_RUNTIME_TELEMETRY_FIELD_UINT64);
           EXPECT_EQ(fields[i].value.uint64_value, values[i]);
         }
         return iree_ok_status();
       }}));
}

}  // namespace
