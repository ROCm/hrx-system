// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "amdf/amdf.h"
#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

const amdf_api_t* QueryApi() {
  const amdf_api_t* api = nullptr;
  EXPECT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api)));
  return api;
}

amdf_instance_create_info_t MakeInstanceCreateInfo() {
  amdf_instance_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.native_lifetime = GetCtsDeviceCache().native_lifetime();
  return create_info;
}

amdf_endpoint_info_t MakeEndpointInfo() {
  amdf_endpoint_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
  info.structure_size = sizeof(info);
  return info;
}

amdf_queue_family_info_t MakeQueueFamilyInfo() {
  amdf_queue_family_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
  info.structure_size = sizeof(info);
  return info;
}

TEST(InstanceTest, ValidatesCreateInfoBeforePlatformInitialization) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);

  auto* const sentinel = reinterpret_cast<amdf_instance_t*>(uintptr_t{1});
  amdf_instance_t* instance = sentinel;
  EXPECT_EQ(amdf_status_code(api->instance_create(nullptr, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);

  amdf_instance_create_info_t create_info = {};
  create_info.structure_size = sizeof(create_info);
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);

  create_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.structure_size = sizeof(amdf_input_structure_t) - 1;
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);

  create_info.structure_size = sizeof(create_info);
  create_info.next = &create_info;
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(instance, sentinel);

  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(api->instance_destroy(nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  create_info = MakeInstanceCreateInfo();
  create_info.native_lifetime = UINT32_MAX;
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);
  create_info = MakeInstanceCreateInfo();
  create_info.reserved = 1;
  EXPECT_EQ(amdf_status_code(api->instance_create(&create_info, &instance)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(instance, sentinel);
}

TEST(InstanceTest, CreatesAndDestroys) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const amdf_instance_create_info_t create_info = MakeInstanceCreateInfo();
  amdf_instance_t* instance = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(api->instance_create(&create_info, &instance)));
  ASSERT_NE(instance, nullptr);
  EXPECT_TRUE(amdf_status_is_ok(api->instance_destroy(instance)));
}

TEST(InstanceTest, InterleavesIndependentInstances) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const amdf_instance_create_info_t create_info = MakeInstanceCreateInfo();
  amdf_instance_t* first_instance = nullptr;
  ASSERT_TRUE(
      amdf_status_is_ok(api->instance_create(&create_info, &first_instance)));

  amdf_instance_t* second_instance = nullptr;
  const amdf_status_t second_status =
      api->instance_create(&create_info, &second_instance);
  if (!amdf_status_is_ok(second_status)) {
    EXPECT_TRUE(amdf_status_is_ok(api->instance_destroy(first_instance)));
    FAIL() << "second instance creation failed: domain="
           << amdf_status_domain(second_status)
           << " code=" << amdf_status_code(second_status);
  }
  EXPECT_NE(first_instance, second_instance);
  EXPECT_TRUE(amdf_status_is_ok(api->instance_destroy(first_instance)));
  EXPECT_TRUE(amdf_status_is_ok(api->instance_destroy(second_instance)));
}

class DiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_ = QueryApi();
    ASSERT_NE(api_, nullptr);
    const amdf_instance_create_info_t create_info = MakeInstanceCreateInfo();
    ASSERT_TRUE(
        amdf_status_is_ok(api_->instance_create(&create_info, &instance_)));
    ASSERT_NE(instance_, nullptr);
  }

  void TearDown() override {
    if (endpoint_ != nullptr) {
      EXPECT_TRUE(amdf_status_is_ok(api_->endpoint_close(endpoint_)));
    }
    if (instance_ != nullptr) {
      EXPECT_TRUE(amdf_status_is_ok(api_->instance_destroy(instance_)));
    }
  }

  amdf_status_t Enumerate(std::vector<amdf_endpoint_summary_t>* summaries) {
    uint32_t endpoint_count = 0;
    amdf_status_t status =
        api_->endpoint_enumerate(instance_, 0, nullptr, &endpoint_count);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    summaries->resize(endpoint_count);
    if (endpoint_count != 0) {
      status = api_->endpoint_enumerate(instance_, endpoint_count,
                                        summaries->data(), &endpoint_count);
      if (amdf_status_is_ok(status)) {
        summaries->resize(endpoint_count);
      }
    }
    return status;
  }

  const amdf_api_t* api_ = nullptr;
  amdf_instance_t* instance_ = nullptr;
  amdf_endpoint_t* endpoint_ = nullptr;
};

TEST_F(DiscoveryTest, ValidatesEnumerationArguments) {
  amdf_endpoint_summary_t summary;
  std::memset(&summary, 0xA5, sizeof(summary));
  const amdf_endpoint_summary_t expected_summary = summary;
  uint32_t endpoint_count = 123;
  EXPECT_EQ(amdf_status_code(api_->endpoint_enumerate(nullptr, 1, &summary,
                                                      &endpoint_count)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&summary, &expected_summary, sizeof(summary)), 0);
  EXPECT_EQ(endpoint_count, 123u);
  EXPECT_EQ(amdf_status_code(api_->endpoint_enumerate(instance_, 1, nullptr,
                                                      &endpoint_count)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(endpoint_count, 123u);
  EXPECT_EQ(amdf_status_code(
                api_->endpoint_enumerate(instance_, 0, nullptr, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
}

TEST_F(DiscoveryTest, EnumeratesStableFixedStrideSummaries) {
  std::vector<amdf_endpoint_summary_t> summaries;
  ASSERT_TRUE(amdf_status_is_ok(Enumerate(&summaries)));
  const amdf_endpoint_type_flags_t known_type_flags =
      AMDF_ENDPOINT_TYPE_FLAG_DISPLAY_SUPPORTED |
      AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED |
      AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY |
      AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE;
  for (size_t i = 0; i < summaries.size(); ++i) {
    const amdf_endpoint_summary_t& summary = summaries[i];
    EXPECT_NE(std::memchr(summary.name, '\0', sizeof(summary.name)), nullptr);
    EXPECT_EQ(summary.type_flags & ~known_type_flags, 0u);
    EXPECT_LE(summary.engine_kind, AMDF_ENGINE_KIND_XDNA);
    for (size_t j = i + 1; j < summaries.size(); ++j) {
      EXPECT_FALSE(amdf_endpoint_id_is_equal(&summary.id, &summaries[j].id));
    }
  }

  if (summaries.size() > 1) {
    const uint32_t capacity = static_cast<uint32_t>(summaries.size() - 1);
    std::vector<amdf_endpoint_summary_t> prefix(capacity);
    uint32_t total_count = 0;
    const amdf_status_t status = api_->endpoint_enumerate(
        instance_, capacity, prefix.data(), &total_count);
    if (total_count > capacity) {
      EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
      EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    } else {
      EXPECT_TRUE(amdf_status_is_ok(status));
    }
  }
}

TEST_F(DiscoveryTest, OpensEndpointAndReturnsCachedProperties) {
  std::vector<amdf_endpoint_summary_t> summaries;
  ASSERT_TRUE(amdf_status_is_ok(Enumerate(&summaries)));
  if (summaries.empty()) {
    GTEST_SKIP() << "no AMD endpoints present";
  }
  ASSERT_TRUE(amdf_status_is_ok(
      api_->endpoint_open(instance_, &summaries[0].id, &endpoint_)));
  ASSERT_NE(endpoint_, nullptr);

  amdf_endpoint_info_t info = MakeEndpointInfo();
  ASSERT_TRUE(amdf_status_is_ok(api_->endpoint_query_info(endpoint_, &info)));
  EXPECT_TRUE(amdf_endpoint_id_is_equal(&info.id, &summaries[0].id));
  EXPECT_EQ(info.engine_kind, summaries[0].engine_kind);
  EXPECT_EQ(info.type_flags, summaries[0].type_flags);
  EXPECT_STREQ(info.name, summaries[0].name);
  EXPECT_TRUE(info.pci.vendor_id == 0x1002u || info.pci.vendor_id == 0x1022u);
}

TEST_F(DiscoveryTest, QueriesImmutableQueueFamilies) {
  std::vector<amdf_endpoint_summary_t> summaries;
  ASSERT_TRUE(amdf_status_is_ok(Enumerate(&summaries)));
  if (summaries.empty()) {
    GTEST_SKIP() << "no AMD endpoints present";
  }
  ASSERT_TRUE(amdf_status_is_ok(
      api_->endpoint_open(instance_, &summaries[0].id, &endpoint_)));

  amdf_endpoint_info_t endpoint_info = MakeEndpointInfo();
  ASSERT_TRUE(
      amdf_status_is_ok(api_->endpoint_query_info(endpoint_, &endpoint_info)));
  for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count;
       ++ordinal) {
    amdf_queue_family_info_t family_info = MakeQueueFamilyInfo();
    ASSERT_TRUE(amdf_status_is_ok(api_->endpoint_query_queue_family_info(
        endpoint_, ordinal, &family_info)));
    EXPECT_EQ(family_info.ordinal, ordinal);
    EXPECT_NE(family_info.command_type, AMDF_QUEUE_COMMAND_TYPE_UNKNOWN);
    EXPECT_NE(family_info.publication_modes, 0u);
    EXPECT_NE(family_info.format_version, 0u);
    EXPECT_NE(family_info.roles, 0u);
    EXPECT_EQ(
        family_info.publication_modes & ~(AMDF_QUEUE_PUBLICATION_MODE_USER |
                                          AMDF_QUEUE_PUBLICATION_MODE_KERNEL),
        0u);
  }

  amdf_queue_family_info_t family_info = MakeQueueFamilyInfo();
  family_info.ordinal = UINT32_MAX;
  family_info.command_type = UINT32_MAX;
  family_info.publication_modes = UINT32_MAX;
  const amdf_queue_family_info_t original_info = family_info;
  const amdf_status_t status = api_->endpoint_query_queue_family_info(
      endpoint_, endpoint_info.queue_family_count, &family_info);
  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(std::memcmp(&family_info, &original_info, sizeof(family_info)), 0);

  const amdf_status_t maximum_status = api_->endpoint_query_queue_family_info(
      endpoint_, UINT32_MAX, &family_info);
  EXPECT_EQ(amdf_status_domain(maximum_status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(maximum_status), AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(std::memcmp(&family_info, &original_info, sizeof(family_info)), 0);
}

TEST_F(DiscoveryTest, RejectsMalformedQueueFamilyQueryWithoutMutation) {
  std::vector<amdf_endpoint_summary_t> summaries;
  ASSERT_TRUE(amdf_status_is_ok(Enumerate(&summaries)));
  if (summaries.empty()) {
    GTEST_SKIP() << "no AMD endpoints present";
  }
  ASSERT_TRUE(amdf_status_is_ok(
      api_->endpoint_open(instance_, &summaries[0].id, &endpoint_)));

  EXPECT_EQ(amdf_status_code(
                api_->endpoint_query_queue_family_info(endpoint_, 0, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  amdf_queue_family_info_t family_info = {};
  family_info.structure_size = sizeof(family_info);
  family_info.ordinal = UINT32_MAX;
  EXPECT_EQ(amdf_status_code(api_->endpoint_query_queue_family_info(
                endpoint_, 0, &family_info)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(family_info.ordinal, UINT32_MAX);

  family_info.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
  family_info.next = &family_info;
  EXPECT_EQ(amdf_status_code(api_->endpoint_query_queue_family_info(
                endpoint_, 0, &family_info)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(family_info.ordinal, UINT32_MAX);
}

TEST_F(DiscoveryTest, RejectsMalformedInfoWithoutMutation) {
  std::vector<amdf_endpoint_summary_t> summaries;
  ASSERT_TRUE(amdf_status_is_ok(Enumerate(&summaries)));
  if (summaries.empty()) {
    GTEST_SKIP() << "no AMD endpoints present";
  }
  ASSERT_TRUE(amdf_status_is_ok(
      api_->endpoint_open(instance_, &summaries[0].id, &endpoint_)));

  EXPECT_EQ(amdf_status_code(api_->endpoint_query_info(endpoint_, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  amdf_endpoint_info_t info = {};
  info.structure_size = sizeof(info);
  info.engine_kind = UINT32_MAX;
  EXPECT_EQ(amdf_status_code(api_->endpoint_query_info(endpoint_, &info)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(info.engine_kind, UINT32_MAX);

  info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
  info.next = &info;
  EXPECT_EQ(amdf_status_code(api_->endpoint_query_info(endpoint_, &info)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(info.engine_kind, UINT32_MAX);
}

TEST_F(DiscoveryTest, OpenEndpointPreventsInstanceDestruction) {
  std::vector<amdf_endpoint_summary_t> summaries;
  ASSERT_TRUE(amdf_status_is_ok(Enumerate(&summaries)));
  if (summaries.empty()) {
    GTEST_SKIP() << "no AMD endpoints present";
  }
  ASSERT_TRUE(amdf_status_is_ok(
      api_->endpoint_open(instance_, &summaries[0].id, &endpoint_)));
  const amdf_status_t status = api_->instance_destroy(instance_);
  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_BUSY);
}

TEST_F(DiscoveryTest, RejectsStaleEndpointIdentityWithoutPublishingOutput) {
  std::vector<amdf_endpoint_summary_t> summaries;
  ASSERT_TRUE(amdf_status_is_ok(Enumerate(&summaries)));
  if (summaries.empty()) {
    GTEST_SKIP() << "no AMD endpoints present";
  }
  amdf_endpoint_id_t stale_id = summaries[0].id;
  stale_id.words[1] ^= UINT64_C(1) << 63;
  auto* const sentinel = reinterpret_cast<amdf_endpoint_t*>(uintptr_t{1});
  amdf_endpoint_t* endpoint = sentinel;
  const amdf_status_t status =
      api_->endpoint_open(instance_, &stale_id, &endpoint);
  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_NOT_FOUND);
  EXPECT_EQ(endpoint, sentinel);
}

TEST_F(DiscoveryTest, ValidatesEndpointArguments) {
  amdf_endpoint_id_t id = {};
  auto* const sentinel = reinterpret_cast<amdf_endpoint_t*>(uintptr_t{1});
  amdf_endpoint_t* endpoint = sentinel;
  EXPECT_EQ(amdf_status_code(api_->endpoint_open(nullptr, &id, &endpoint)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(endpoint, sentinel);
  EXPECT_EQ(
      amdf_status_code(api_->endpoint_open(instance_, nullptr, &endpoint)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(endpoint, sentinel);
  EXPECT_EQ(amdf_status_code(api_->endpoint_open(instance_, &id, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(api_->endpoint_query_info(nullptr, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(
                api_->endpoint_query_queue_family_info(nullptr, 0, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(api_->endpoint_close(nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
}

}  // namespace
