// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/telemetry.h"

#include "gtest/gtest.h"
#include "util/provider.h"

namespace {

static_assert(sizeof(amdf_observation_value_format_t) == 32);
static_assert(sizeof(amdf_telemetry_value_t) == 32);
static_assert(sizeof(amdf_telemetry_sample_t) == 80);

TEST(TelemetryApiTest, NegotiationNeedsNoInstanceOrDevice) {
  const amdf_api_t* api = nullptr;
  ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_LATEST,
                                          AMDF_ABI_VERSION_LATEST, &api),
            AMDF_STATUS_OK);
  const void* extension = nullptr;
  ASSERT_EQ(api->query_extension(AMDF_EXTENSION_TELEMETRY, 1, 1, &extension),
            AMDF_STATUS_OK);
  const auto* telemetry = static_cast<const amdf_telemetry_api_t*>(extension);
  ASSERT_EQ(telemetry->structure_size, sizeof(*telemetry));
  EXPECT_EQ(telemetry->extension_version, 1u);
  ASSERT_NE(telemetry->device_enumerate_metrics, nullptr);
  ASSERT_NE(telemetry->device_read, nullptr);
  EXPECT_EQ(api->query_extension(AMDF_EXTENSION_TELEMETRY, 2, 2, &extension),
            amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH));
  EXPECT_EQ(extension, telemetry);
  uint32_t count = 123;
  EXPECT_EQ(telemetry->device_enumerate_metrics(nullptr, 0, nullptr, &count),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(count, 123u);
  EXPECT_EQ(telemetry->device_read(nullptr, 0, 0, nullptr, nullptr, nullptr),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
}

}  // namespace
