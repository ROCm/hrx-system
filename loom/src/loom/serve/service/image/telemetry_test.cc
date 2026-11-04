// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/image/telemetry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ImageTelemetryTest, SamplesCompletedWorkWithoutAnEventStream) {
  const loom_serve_image_totals_t totals = {
      1, UINT64_MAX, {3, 4, 5, 6, 7, 8, 9, 10}};
  IREE_ASSERT_OK(loom_serve_image_sample_totals(
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

}  // namespace
