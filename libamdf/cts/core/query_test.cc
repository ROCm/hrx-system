// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "amdf/amdf.h"
#include "gtest/gtest.h"
#include "util/provider.h"

namespace {

TEST(QueryApiTest, NegotiatesSupportedVersion) {
  const amdf_api_t* api = nullptr;
  const amdf_status_t status = amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api);

  EXPECT_TRUE(amdf_status_is_ok(status));
  ASSERT_NE(api, nullptr);
  EXPECT_EQ(api->structure_size, sizeof(amdf_api_t));
  EXPECT_EQ(api->abi_version, AMDF_ABI_VERSION_1);
  EXPECT_NE(api->instance_create, nullptr);
  EXPECT_NE(api->instance_destroy, nullptr);
  EXPECT_NE(api->endpoint_enumerate, nullptr);
  EXPECT_NE(api->endpoint_open, nullptr);
  EXPECT_NE(api->endpoint_query_info, nullptr);
  EXPECT_NE(api->endpoint_close, nullptr);
  EXPECT_NE(api->query_extension, nullptr);
  EXPECT_NE(api->endpoint_query_queue_family_info, nullptr);
  EXPECT_NE(api->device_destroy, nullptr);
  EXPECT_NE(api->instance_enumerate_memory_scopes, nullptr);
  EXPECT_NE(api->endpoint_enumerate_memory_scopes, nullptr);
  EXPECT_NE(api->device_enumerate_memory_scopes, nullptr);
  EXPECT_NE(api->memory_scope_query_info, nullptr);
  EXPECT_NE(api->memory_scope_query_profile, nullptr);
  EXPECT_NE(api->memory_scope_query_device_profile, nullptr);
  EXPECT_NE(api->memory_create, nullptr);
  EXPECT_NE(api->memory_import, nullptr);
  EXPECT_NE(api->memory_query_info, nullptr);
  EXPECT_NE(api->memory_query_access_info, nullptr);
  EXPECT_NE(api->memory_query_address, nullptr);
  EXPECT_NE(api->memory_export, nullptr);
  EXPECT_NE(api->external_memory_release, nullptr);
  EXPECT_NE(api->memory_query_pair_info, nullptr);
  EXPECT_NE(api->memory_map, nullptr);
  EXPECT_NE(api->host_mapping_query_info, nullptr);
  EXPECT_NE(api->host_mapping_cache_control, nullptr);
  EXPECT_NE(api->host_mapping_destroy, nullptr);
  EXPECT_NE(api->memory_destroy, nullptr);
  EXPECT_NE(api->kernel_queue_query_info, nullptr);
  EXPECT_NE(api->kernel_queue_query_status, nullptr);
  EXPECT_NE(api->kernel_queue_wait, nullptr);
  EXPECT_NE(api->kernel_queue_destroy, nullptr);
  EXPECT_NE(api->user_queue_query_info, nullptr);
  EXPECT_NE(api->user_queue_map, nullptr);
  EXPECT_NE(api->user_queue_mapping_query_info, nullptr);
  EXPECT_NE(api->user_queue_mapping_destroy, nullptr);
  EXPECT_NE(api->user_queue_query_status, nullptr);
  EXPECT_NE(api->user_queue_wait_consumed, nullptr);
  EXPECT_NE(api->user_queue_destroy, nullptr);
}

TEST(QueryApiTest, ReturnsStableImmutableTable) {
  const amdf_api_t* first_api = nullptr;
  const amdf_api_t* second_api = nullptr;

  ASSERT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_1, &first_api)));
  ASSERT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_1, &second_api)));
  EXPECT_EQ(first_api, second_api);
}

TEST(QueryApiTest, RejectsNullOutput) {
  const amdf_status_t status = amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_1, nullptr);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_INVALID_ARGUMENT);
}

TEST(QueryApiTest, RejectsReversedVersionRangeWithoutPublishingOutput) {
  const auto* const sentinel =
      reinterpret_cast<const amdf_api_t*>(uintptr_t{1});
  const amdf_api_t* api = sentinel;
  const amdf_status_t status = amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1 + 1, AMDF_ABI_VERSION_1, &api);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(api, sentinel);
}

TEST(QueryApiTest, RejectsUnsupportedVersionWithoutPublishingOutput) {
  const auto* const sentinel =
      reinterpret_cast<const amdf_api_t*>(uintptr_t{1});
  const amdf_api_t* api = sentinel;
  const amdf_status_t status = amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_LATEST + 1, AMDF_ABI_VERSION_LATEST + 1, &api);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_VERSION_MISMATCH);
  EXPECT_EQ(api, sentinel);
}

TEST(QueryExtensionTest, RejectsNullOutput) {
  const amdf_api_t* api = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api)));

  const amdf_status_t status =
      api->query_extension(AMDF_EXTENSION_XDNA, 1, UINT32_MAX, nullptr);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_INVALID_ARGUMENT);
}

TEST(QueryExtensionTest, RejectsReversedVersionRangeWithoutPublishingOutput) {
  const amdf_api_t* api = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api)));
  const void* const sentinel = reinterpret_cast<const void*>(uintptr_t{1});
  const void* extension_api = sentinel;

  const amdf_status_t status =
      api->query_extension(AMDF_EXTENSION_XDNA, 2, 1, &extension_api);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(extension_api, sentinel);
}

TEST(QueryExtensionTest, RejectsUnknownExtensionWithoutPublishingOutput) {
  const amdf_api_t* api = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api)));
  const void* const sentinel = reinterpret_cast<const void*>(uintptr_t{1});
  const void* extension_api = sentinel;

  const amdf_status_t status =
      api->query_extension(UINT32_MAX, 1, UINT32_MAX, &extension_api);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(extension_api, sentinel);
}

TEST(StatusTest, PreservesDomainAndCode) {
  const amdf_status_t status =
      amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 0xFEDCBA98u);

  EXPECT_FALSE(amdf_status_is_ok(status));
  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_FIRMWARE);
  EXPECT_EQ(amdf_status_code(status), 0xFEDCBA98u);
  EXPECT_TRUE(amdf_status_is_ok(AMDF_STATUS_OK));
  EXPECT_TRUE(amdf_status_is_ok(amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, 0)));
}

}  // namespace
