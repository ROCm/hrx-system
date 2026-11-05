// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/selection.h"

#include <array>
#include <cstring>

#include "gtest/gtest.h"

namespace {

amdf_info_endpoint_t Gpu(uint64_t identity, uint32_t major) {
  amdf_info_endpoint_t endpoint = {};
  endpoint.summary.id.words[0] = identity;
  endpoint.summary.engine_kind = AMDF_ENGINE_KIND_GPU;
  endpoint.info.engine_kind = AMDF_ENGINE_KIND_GPU;
  endpoint.target.gpu.gfx_ip.major = major;
  endpoint.stage = AMDF_INFO_ENDPOINT_TARGET;
  return endpoint;
}

TEST(SelectionTest, UsesFamilyOrdinalBeforeAdditionalFilters) {
  std::array endpoints = {Gpu(1, 11), Gpu(2, 9), Gpu(3, 12)};
  amdf_info_options_t options = {.device = "rdna:1"};
  amdf_info_selector_t selector;
  ASSERT_EQ(amdf_info_selector_parse(options.device, &selector), nullptr);
  EXPECT_EQ(amdf_info_selection_apply(&options, &selector, endpoints.size(),
                                      endpoints.data()),
            AMDF_INFO_EXIT_SUCCESS);
  EXPECT_FALSE(endpoints[0].selected);
  EXPECT_FALSE(endpoints[1].selected);
  EXPECT_TRUE(endpoints[2].selected);
  options.target = "gfx1100";
  EXPECT_EQ(amdf_info_selection_apply(&options, &selector, endpoints.size(),
                                      endpoints.data()),
            AMDF_INFO_EXIT_NOT_FOUND);
  for (const auto& endpoint : endpoints) {
    EXPECT_FALSE(endpoint.selected);
  }
}

TEST(SelectionTest, UnknownEarlierFamilyCannotShiftOrdinalSilently) {
  std::array endpoints = {Gpu(1, 11), Gpu(2, 11)};
  endpoints[0].stage = AMDF_INFO_ENDPOINT_INFO;
  endpoints[0].status =
      amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
  amdf_info_options_t options = {.device = "rdna:0"};
  amdf_info_selector_t selector;
  ASSERT_EQ(amdf_info_selector_parse(options.device, &selector), nullptr);
  EXPECT_EQ(amdf_info_selection_apply(&options, &selector, endpoints.size(),
                                      endpoints.data()),
            AMDF_INFO_EXIT_FAILURE);
  EXPECT_TRUE(endpoints[0].selected);
}

TEST(SelectionTest, UnknownLaterFamilyCannotInvalidateEarlierOrdinal) {
  std::array endpoints = {Gpu(1, 11), Gpu(2, 11)};
  endpoints[1].stage = AMDF_INFO_ENDPOINT_INFO;
  endpoints[1].status =
      amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
  amdf_info_options_t options = {.device = "rdna:0"};
  amdf_info_selector_t selector;
  ASSERT_EQ(amdf_info_selector_parse(options.device, &selector), nullptr);
  EXPECT_EQ(amdf_info_selection_apply(&options, &selector, endpoints.size(),
                                      endpoints.data()),
            AMDF_INFO_EXIT_SUCCESS);
  EXPECT_TRUE(endpoints[0].selected);
  EXPECT_FALSE(endpoints[1].selected);
}

TEST(SelectionTest, NativePathMatchesDeviceIdentityNotDisplayName) {
  auto endpoint = Gpu(1, 11);
  endpoint.info.native_identity.type =
      AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE;
  endpoint.info.native_identity.value.linux_device = {226, 129};
  amdf_info_selector_t selector;
  ASSERT_EQ(amdf_info_selector_parse("/dev/dri/renderD129", &selector),
            nullptr);
  ASSERT_EQ(selector.kind, AMDF_INFO_SELECTOR_NATIVE_PATH);
  // Identity resolution belongs to the OS adapter. Matching needs no I/O.
  selector.native_identity = endpoint.info.native_identity;
  amdf_info_options_t options = {.select = true};
  EXPECT_EQ(amdf_info_selection_apply(&options, &selector, 1, &endpoint),
            AMDF_INFO_EXIT_SUCCESS);
  ++selector.native_identity.value.linux_device.minor;
  EXPECT_EQ(amdf_info_selection_apply(&options, &selector, 1, &endpoint),
            AMDF_INFO_EXIT_NOT_FOUND);
}

TEST(SelectionTest, ExactUriRoundTripsFullWidthIdentity) {
  auto endpoint = Gpu(UINT64_MAX, 12);
  endpoint.summary.id.words[1] = UINT64_C(0x8000000000000001);
  char uri[48];
  amdf_info_selection_uri(&endpoint, uri);
  amdf_info_selector_t selector;
  ASSERT_EQ(amdf_info_selector_parse(uri, &selector), nullptr);
  EXPECT_TRUE(
      amdf_endpoint_id_is_equal(&selector.endpoint_id, &endpoint.summary.id));
  amdf_info_options_t options = {.select = true};
  EXPECT_EQ(amdf_info_selection_apply(&options, &selector, 1, &endpoint),
            AMDF_INFO_EXIT_SUCCESS);
}

TEST(SelectionTest, UnknownEngineHasNoInventedExecutableUri) {
  amdf_info_endpoint_t endpoint = {};
  char uri[48];
  amdf_info_selection_uri(&endpoint, uri);
  EXPECT_STREQ(uri, "");
}

TEST(SelectionTest, InvalidParseLeavesOutputUnchanged) {
  amdf_info_selector_t selector = {.ordinal = 123};
  const auto original = selector;
  for (const char* invalid : {"rdna://id/ffffffffffffffff-000000000000000",
                              "rdna://4294967296", "xdna://", "GPU-1"}) {
    ASSERT_NE(amdf_info_selector_parse(invalid, &selector), nullptr);
    EXPECT_EQ(std::memcmp(&selector, &original, sizeof(selector)), 0);
  }
}

TEST(SelectionTest, MaximumOrdinalIsValidSyntaxButMustStillExist) {
  amdf_info_selector_t selector;
  ASSERT_EQ(amdf_info_selector_parse("rdna://4294967295", &selector), nullptr);
  EXPECT_EQ(selector.ordinal, UINT32_MAX);
  amdf_info_options_t options = {.select = true};
  EXPECT_EQ(amdf_info_selection_apply(&options, &selector, 0, nullptr),
            AMDF_INFO_EXIT_NOT_FOUND);
}

}  // namespace
