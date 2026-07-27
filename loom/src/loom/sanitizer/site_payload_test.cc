// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/site_payload.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

loom_sanitizer_site_payload_t BaselinePayload() {
  loom_sanitizer_site_payload_t payload = {};
  payload.site_kind = LOOM_SANITIZER_SITE_KIND_ACCESS;
  payload.check_kind = LOOM_SANITIZER_CHECK_KIND_ACCESS_RANGE;
  payload.provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_ASSUME;
  payload.lane_policy = LOOM_SANITIZER_LANE_POLICY_PER_LANE;
  payload.lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL;
  payload.flags = 0;
  payload.extension_data = iree_const_byte_span_empty();
  return payload;
}

void ExpectSamePayload(const loom_sanitizer_site_payload_t& expected,
                       const loom_sanitizer_site_payload_t& actual) {
  EXPECT_EQ(actual.site_kind, expected.site_kind);
  EXPECT_EQ(actual.check_kind, expected.check_kind);
  EXPECT_EQ(actual.provenance_kind, expected.provenance_kind);
  EXPECT_EQ(actual.lane_policy, expected.lane_policy);
  EXPECT_EQ(actual.lineage_role, expected.lineage_role);
  EXPECT_EQ(actual.flags, expected.flags);
  ASSERT_EQ(actual.extension_data.data_length,
            expected.extension_data.data_length);
  if (expected.extension_data.data_length > 0) {
    ASSERT_NE(actual.extension_data.data, nullptr);
    ASSERT_NE(expected.extension_data.data, nullptr);
    EXPECT_EQ(memcmp(actual.extension_data.data, expected.extension_data.data,
                     expected.extension_data.data_length),
              0);
  }
}

void ExpectRoundTrip(const loom_sanitizer_site_payload_t& payload) {
  uint8_t storage[32] = {0};
  iree_host_size_t encoded_length = 0;
  IREE_ASSERT_OK(loom_sanitizer_site_payload_encode(
      &payload, iree_make_byte_span(storage, sizeof(storage)),
      &encoded_length));

  loom_sanitizer_site_payload_t decoded_payload = {};
  IREE_ASSERT_OK(loom_sanitizer_site_payload_decode(
      iree_make_const_byte_span(storage, encoded_length), &decoded_payload));
  ExpectSamePayload(payload, decoded_payload);
}

//===----------------------------------------------------------------------===//
// Encoding and decoding
//===----------------------------------------------------------------------===//

TEST(SanitizerSitePayloadTest, EncodesMinimumPayload) {
  loom_sanitizer_site_payload_t payload = BaselinePayload();
  payload.flags = 0xA55A;

  uint8_t storage[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH] = {0};
  iree_host_size_t encoded_length = 0;
  IREE_EXPECT_OK(loom_sanitizer_site_payload_encode(
      &payload, iree_make_byte_span(storage, sizeof(storage)),
      &encoded_length));

  EXPECT_EQ(encoded_length, LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH);
  const uint8_t expected_bytes[] = {
      LOOM_SANITIZER_SITE_PAYLOAD_CURRENT_VERSION,
      LOOM_SANITIZER_SITE_KIND_ACCESS,
      LOOM_SANITIZER_CHECK_KIND_ACCESS_RANGE,
      LOOM_SANITIZER_PROVENANCE_KIND_ASSUME,
      LOOM_SANITIZER_LANE_POLICY_PER_LANE,
      LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      0x5A,
      0xA5,
  };
  EXPECT_EQ(memcmp(storage, expected_bytes, sizeof(expected_bytes)), 0);
}

TEST(SanitizerSitePayloadTest, DecodesMinimumPayload) {
  const uint8_t bytes[] = {
      LOOM_SANITIZER_SITE_PAYLOAD_CURRENT_VERSION,
      LOOM_SANITIZER_SITE_KIND_VALUE,
      LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_NAN,
      LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      LOOM_SANITIZER_LANE_POLICY_SCALAR,
      LOOM_SANITIZER_LINEAGE_ROLE_TILE_LOWERING,
      0x34,
      0x12,
  };

  loom_sanitizer_site_payload_t decoded_payload = {};
  IREE_EXPECT_OK(loom_sanitizer_site_payload_decode(
      iree_make_const_byte_span(bytes, sizeof(bytes)), &decoded_payload));

  EXPECT_EQ(decoded_payload.site_kind, LOOM_SANITIZER_SITE_KIND_VALUE);
  EXPECT_EQ(decoded_payload.check_kind,
            LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_NAN);
  EXPECT_EQ(decoded_payload.provenance_kind,
            LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT);
  EXPECT_EQ(decoded_payload.lane_policy, LOOM_SANITIZER_LANE_POLICY_SCALAR);
  EXPECT_EQ(decoded_payload.lineage_role,
            LOOM_SANITIZER_LINEAGE_ROLE_TILE_LOWERING);
  EXPECT_EQ(decoded_payload.flags, 0x1234);
  EXPECT_TRUE(iree_const_byte_span_is_empty(decoded_payload.extension_data));
}

TEST(SanitizerSitePayloadTest, PreservesExtensionBytes) {
  const uint8_t extension_data[] = {0xCA, 0xFE, 0x12, 0x34};
  loom_sanitizer_site_payload_t payload = BaselinePayload();
  payload.extension_data =
      iree_make_const_byte_span(extension_data, sizeof(extension_data));

  uint8_t storage[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH +
                  sizeof(extension_data)] = {0};
  iree_host_size_t encoded_length = 0;
  IREE_ASSERT_OK(loom_sanitizer_site_payload_encode(
      &payload, iree_make_byte_span(storage, sizeof(storage)),
      &encoded_length));
  EXPECT_EQ(encoded_length, sizeof(storage));

  loom_sanitizer_site_payload_t decoded_payload = {};
  IREE_ASSERT_OK(loom_sanitizer_site_payload_decode(
      iree_make_const_byte_span(storage, encoded_length), &decoded_payload));
  ExpectSamePayload(payload, decoded_payload);
}

TEST(SanitizerSitePayloadTest, RoundTripsAllKnownEnumValues) {
  const loom_sanitizer_site_kind_t site_kinds[] = {
      LOOM_SANITIZER_SITE_KIND_UNKNOWN, LOOM_SANITIZER_SITE_KIND_ACCESS,
      LOOM_SANITIZER_SITE_KIND_VALUE,   LOOM_SANITIZER_SITE_KIND_OPERATION,
      LOOM_SANITIZER_SITE_KIND_LAYOUT,  LOOM_SANITIZER_SITE_KIND_RACE,
  };
  const loom_sanitizer_check_kind_t check_kinds[] = {
      LOOM_SANITIZER_CHECK_KIND_UNKNOWN,
      LOOM_SANITIZER_CHECK_KIND_ACCESS_RANGE,
      LOOM_SANITIZER_CHECK_KIND_ACCESS_ALIGNMENT,
      LOOM_SANITIZER_CHECK_KIND_VALUE_RANGE,
      LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_NAN,
      LOOM_SANITIZER_CHECK_KIND_VALUE_DIVISIBILITY,
      LOOM_SANITIZER_CHECK_KIND_INTEGER_OVERFLOW,
      LOOM_SANITIZER_CHECK_KIND_DIVIDE_BY_ZERO,
      LOOM_SANITIZER_CHECK_KIND_INVALID_SHIFT,
      LOOM_SANITIZER_CHECK_KIND_LAYOUT_REFINEMENT,
      LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_INF,
      LOOM_SANITIZER_CHECK_KIND_VALUE_FINITE,
      LOOM_SANITIZER_CHECK_KIND_VALUE_POWER_OF_TWO,
      LOOM_SANITIZER_CHECK_KIND_VALUE_RELATION,
      LOOM_SANITIZER_CHECK_KIND_VALUE_CONSTRAINTS,
      LOOM_SANITIZER_CHECK_KIND_DATA_RACE,
  };
  const loom_sanitizer_provenance_kind_t provenance_kinds[] = {
      LOOM_SANITIZER_PROVENANCE_KIND_UNKNOWN,
      LOOM_SANITIZER_PROVENANCE_KIND_USER_ASSERTION,
      LOOM_SANITIZER_PROVENANCE_KIND_ASSUME,
      LOOM_SANITIZER_PROVENANCE_KIND_ANALYSIS,
      LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      LOOM_SANITIZER_PROVENANCE_KIND_IMPORTER_PROMISE,
      LOOM_SANITIZER_PROVENANCE_KIND_FAST_MATH,
      LOOM_SANITIZER_PROVENANCE_KIND_ABI_PROMISE,
      LOOM_SANITIZER_PROVENANCE_KIND_OPTIMIZATION_OBLIGATION,
  };
  const loom_sanitizer_lane_policy_t lane_policies[] = {
      LOOM_SANITIZER_LANE_POLICY_UNKNOWN,   LOOM_SANITIZER_LANE_POLICY_SCALAR,
      LOOM_SANITIZER_LANE_POLICY_PER_LANE,  LOOM_SANITIZER_LANE_POLICY_ANY_LANE,
      LOOM_SANITIZER_LANE_POLICY_ALL_LANES,
  };
  const loom_sanitizer_lineage_role_t lineage_roles[] = {
      LOOM_SANITIZER_LINEAGE_ROLE_UNKNOWN,
      LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      LOOM_SANITIZER_LINEAGE_ROLE_CLONE,
      LOOM_SANITIZER_LINEAGE_ROLE_TEMPLATE_INSTANTIATION,
      LOOM_SANITIZER_LINEAGE_ROLE_TILE_LOWERING,
      LOOM_SANITIZER_LINEAGE_ROLE_UKERNEL_SELECTION,
  };

  for (loom_sanitizer_site_kind_t site_kind : site_kinds) {
    loom_sanitizer_site_payload_t payload = BaselinePayload();
    payload.site_kind = site_kind;
    ExpectRoundTrip(payload);
  }
  for (loom_sanitizer_check_kind_t check_kind : check_kinds) {
    loom_sanitizer_site_payload_t payload = BaselinePayload();
    payload.check_kind = check_kind;
    ExpectRoundTrip(payload);
  }
  for (loom_sanitizer_provenance_kind_t provenance_kind : provenance_kinds) {
    loom_sanitizer_site_payload_t payload = BaselinePayload();
    payload.provenance_kind = provenance_kind;
    ExpectRoundTrip(payload);
  }
  for (loom_sanitizer_lane_policy_t lane_policy : lane_policies) {
    loom_sanitizer_site_payload_t payload = BaselinePayload();
    payload.lane_policy = lane_policy;
    ExpectRoundTrip(payload);
  }
  for (loom_sanitizer_lineage_role_t lineage_role : lineage_roles) {
    loom_sanitizer_site_payload_t payload = BaselinePayload();
    payload.lineage_role = lineage_role;
    ExpectRoundTrip(payload);
  }
}

TEST(SanitizerSitePayloadTest, PreservesUnknownFieldValues) {
  loom_sanitizer_site_payload_t payload = BaselinePayload();
  payload.site_kind = (loom_sanitizer_site_kind_t)0xE1;
  payload.check_kind = (loom_sanitizer_check_kind_t)0xE2;
  payload.provenance_kind = (loom_sanitizer_provenance_kind_t)0xE3;
  payload.lane_policy = (loom_sanitizer_lane_policy_t)0xE4;
  payload.lineage_role = (loom_sanitizer_lineage_role_t)0xE5;
  payload.flags = 0xFEDC;

  ExpectRoundTrip(payload);
  EXPECT_TRUE(iree_string_view_is_empty(
      loom_sanitizer_site_kind_name(payload.site_kind)));
  EXPECT_TRUE(iree_string_view_is_empty(
      loom_sanitizer_check_kind_name(payload.check_kind)));
  EXPECT_TRUE(iree_string_view_is_empty(
      loom_sanitizer_provenance_kind_name(payload.provenance_kind)));
  EXPECT_TRUE(iree_string_view_is_empty(
      loom_sanitizer_lane_policy_name(payload.lane_policy)));
  EXPECT_TRUE(iree_string_view_is_empty(
      loom_sanitizer_lineage_role_name(payload.lineage_role)));
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

TEST(SanitizerSitePayloadTest, RejectsTruncatedPayload) {
  const uint8_t bytes[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH] = {0};
  for (iree_host_size_t byte_count = 0;
       byte_count < LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH; ++byte_count) {
    loom_sanitizer_site_payload_t decoded_payload = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        loom_sanitizer_site_payload_decode(
            iree_make_const_byte_span(bytes, byte_count), &decoded_payload));
  }
}

TEST(SanitizerSitePayloadTest, RejectsUnsupportedVersion) {
  uint8_t bytes[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH] = {0};
  bytes[0] = 1;

  loom_sanitizer_site_payload_t decoded_payload = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INCOMPATIBLE,
      loom_sanitizer_site_payload_decode(
          iree_make_const_byte_span(bytes, sizeof(bytes)), &decoded_payload));
}

TEST(SanitizerSitePayloadTest, RejectsNullExtensionData) {
  loom_sanitizer_site_payload_t payload = BaselinePayload();
  payload.extension_data.data = nullptr;
  payload.extension_data.data_length = 1;

  uint8_t storage[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH + 1] = {0};
  iree_host_size_t encoded_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_sanitizer_site_payload_encode(
          &payload, iree_make_byte_span(storage, sizeof(storage)),
          &encoded_length));
  EXPECT_EQ(encoded_length, 0);
}

TEST(SanitizerSitePayloadTest, RejectsMissingEncodingInputs) {
  loom_sanitizer_site_payload_t payload = BaselinePayload();
  uint8_t storage[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH] = {0};
  iree_host_size_t encoded_length = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_sanitizer_site_payload_encode(
          nullptr, iree_make_byte_span(storage, sizeof(storage)),
          &encoded_length));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_sanitizer_site_payload_encode(
                            &payload, iree_byte_span_empty(), &encoded_length));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_sanitizer_site_payload_encode(
          &payload, iree_make_byte_span(storage, sizeof(storage)), nullptr));
}

TEST(SanitizerSitePayloadTest, RejectsShortEncodingStorage) {
  loom_sanitizer_site_payload_t payload = BaselinePayload();
  uint8_t storage[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH - 1] = {0};
  iree_host_size_t encoded_length = 123;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_sanitizer_site_payload_encode(
          &payload, iree_make_byte_span(storage, sizeof(storage)),
          &encoded_length));
  EXPECT_EQ(encoded_length, 0);
}

TEST(SanitizerSitePayloadTest, RejectsOverflowingExtensionLength) {
  uint8_t extension_data = 0;
  loom_sanitizer_site_payload_t payload = BaselinePayload();
  payload.extension_data =
      iree_make_const_byte_span(&extension_data, IREE_HOST_SIZE_MAX);

  uint8_t storage[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH] = {0};
  iree_host_size_t encoded_length = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_sanitizer_site_payload_encode(
          &payload, iree_make_byte_span(storage, sizeof(storage)),
          &encoded_length));
  EXPECT_EQ(encoded_length, 0);
}

TEST(SanitizerSitePayloadTest, RejectsMissingDecodeInputs) {
  const uint8_t bytes[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH] = {0};
  loom_sanitizer_site_payload_t decoded_payload = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_sanitizer_site_payload_decode(
          iree_make_const_byte_span(bytes, sizeof(bytes)), nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_sanitizer_site_payload_decode(
          iree_make_const_byte_span(nullptr, sizeof(bytes)), &decoded_payload));
}

//===----------------------------------------------------------------------===//
// Names
//===----------------------------------------------------------------------===//

TEST(SanitizerSitePayloadTest, ReturnsKnownFieldNames) {
  EXPECT_EQ(StringViewToString(loom_sanitizer_site_kind_name(
                LOOM_SANITIZER_SITE_KIND_OPERATION)),
            "operation");
  EXPECT_EQ(StringViewToString(
                loom_sanitizer_site_kind_name(LOOM_SANITIZER_SITE_KIND_LAYOUT)),
            "layout");
  EXPECT_EQ(StringViewToString(
                loom_sanitizer_site_kind_name(LOOM_SANITIZER_SITE_KIND_RACE)),
            "race");
  EXPECT_EQ(StringViewToString(loom_sanitizer_check_kind_name(
                LOOM_SANITIZER_CHECK_KIND_VALUE_DIVISIBILITY)),
            "value_divisibility");
  EXPECT_EQ(StringViewToString(loom_sanitizer_check_kind_name(
                LOOM_SANITIZER_CHECK_KIND_LAYOUT_REFINEMENT)),
            "layout_refinement");
  EXPECT_EQ(StringViewToString(loom_sanitizer_check_kind_name(
                LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_INF)),
            "value_not_inf");
  EXPECT_EQ(StringViewToString(loom_sanitizer_check_kind_name(
                LOOM_SANITIZER_CHECK_KIND_VALUE_FINITE)),
            "value_finite");
  EXPECT_EQ(StringViewToString(loom_sanitizer_check_kind_name(
                LOOM_SANITIZER_CHECK_KIND_VALUE_POWER_OF_TWO)),
            "value_power_of_two");
  EXPECT_EQ(StringViewToString(loom_sanitizer_check_kind_name(
                LOOM_SANITIZER_CHECK_KIND_VALUE_RELATION)),
            "value_relation");
  EXPECT_EQ(StringViewToString(loom_sanitizer_check_kind_name(
                LOOM_SANITIZER_CHECK_KIND_VALUE_CONSTRAINTS)),
            "value_constraints");
  EXPECT_EQ(StringViewToString(loom_sanitizer_check_kind_name(
                LOOM_SANITIZER_CHECK_KIND_DATA_RACE)),
            "data_race");
  EXPECT_EQ(StringViewToString(loom_sanitizer_provenance_kind_name(
                LOOM_SANITIZER_PROVENANCE_KIND_USER_ASSERTION)),
            "user_assertion");
  EXPECT_EQ(StringViewToString(loom_sanitizer_provenance_kind_name(
                LOOM_SANITIZER_PROVENANCE_KIND_IMPORTER_PROMISE)),
            "importer_promise");
  EXPECT_EQ(StringViewToString(loom_sanitizer_provenance_kind_name(
                LOOM_SANITIZER_PROVENANCE_KIND_FAST_MATH)),
            "fast_math");
  EXPECT_EQ(StringViewToString(loom_sanitizer_provenance_kind_name(
                LOOM_SANITIZER_PROVENANCE_KIND_ABI_PROMISE)),
            "abi_promise");
  EXPECT_EQ(StringViewToString(loom_sanitizer_provenance_kind_name(
                LOOM_SANITIZER_PROVENANCE_KIND_OPTIMIZATION_OBLIGATION)),
            "optimization_obligation");
  EXPECT_EQ(StringViewToString(loom_sanitizer_lane_policy_name(
                LOOM_SANITIZER_LANE_POLICY_ALL_LANES)),
            "all_lanes");
  EXPECT_EQ(StringViewToString(loom_sanitizer_lineage_role_name(
                LOOM_SANITIZER_LINEAGE_ROLE_UKERNEL_SELECTION)),
            "ukernel_selection");
}

}  // namespace
}  // namespace loom
