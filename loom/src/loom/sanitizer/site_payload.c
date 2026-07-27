// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/site_payload.h"

#include <string.h>

iree_string_view_t loom_sanitizer_site_kind_name(
    loom_sanitizer_site_kind_t site_kind) {
  switch (site_kind) {
    case LOOM_SANITIZER_SITE_KIND_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_SANITIZER_SITE_KIND_ACCESS:
      return IREE_SV("access");
    case LOOM_SANITIZER_SITE_KIND_VALUE:
      return IREE_SV("value");
    case LOOM_SANITIZER_SITE_KIND_OPERATION:
      return IREE_SV("operation");
    case LOOM_SANITIZER_SITE_KIND_LAYOUT:
      return IREE_SV("layout");
    case LOOM_SANITIZER_SITE_KIND_RACE:
      return IREE_SV("race");
    default:
      return iree_string_view_empty();
  }
}

iree_string_view_t loom_sanitizer_check_kind_name(
    loom_sanitizer_check_kind_t check_kind) {
  switch (check_kind) {
    case LOOM_SANITIZER_CHECK_KIND_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_SANITIZER_CHECK_KIND_ACCESS_RANGE:
      return IREE_SV("access_range");
    case LOOM_SANITIZER_CHECK_KIND_ACCESS_ALIGNMENT:
      return IREE_SV("access_alignment");
    case LOOM_SANITIZER_CHECK_KIND_VALUE_RANGE:
      return IREE_SV("value_range");
    case LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_NAN:
      return IREE_SV("value_not_nan");
    case LOOM_SANITIZER_CHECK_KIND_VALUE_DIVISIBILITY:
      return IREE_SV("value_divisibility");
    case LOOM_SANITIZER_CHECK_KIND_INTEGER_OVERFLOW:
      return IREE_SV("integer_overflow");
    case LOOM_SANITIZER_CHECK_KIND_DIVIDE_BY_ZERO:
      return IREE_SV("divide_by_zero");
    case LOOM_SANITIZER_CHECK_KIND_INVALID_SHIFT:
      return IREE_SV("invalid_shift");
    case LOOM_SANITIZER_CHECK_KIND_LAYOUT_REFINEMENT:
      return IREE_SV("layout_refinement");
    case LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_INF:
      return IREE_SV("value_not_inf");
    case LOOM_SANITIZER_CHECK_KIND_VALUE_FINITE:
      return IREE_SV("value_finite");
    case LOOM_SANITIZER_CHECK_KIND_VALUE_POWER_OF_TWO:
      return IREE_SV("value_power_of_two");
    case LOOM_SANITIZER_CHECK_KIND_VALUE_RELATION:
      return IREE_SV("value_relation");
    case LOOM_SANITIZER_CHECK_KIND_VALUE_CONSTRAINTS:
      return IREE_SV("value_constraints");
    case LOOM_SANITIZER_CHECK_KIND_DATA_RACE:
      return IREE_SV("data_race");
    default:
      return iree_string_view_empty();
  }
}

iree_string_view_t loom_sanitizer_provenance_kind_name(
    loom_sanitizer_provenance_kind_t provenance_kind) {
  switch (provenance_kind) {
    case LOOM_SANITIZER_PROVENANCE_KIND_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_SANITIZER_PROVENANCE_KIND_USER_ASSERTION:
      return IREE_SV("user_assertion");
    case LOOM_SANITIZER_PROVENANCE_KIND_ASSUME:
      return IREE_SV("assume");
    case LOOM_SANITIZER_PROVENANCE_KIND_ANALYSIS:
      return IREE_SV("analysis");
    case LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT:
      return IREE_SV("compiler_contract");
    case LOOM_SANITIZER_PROVENANCE_KIND_IMPORTER_PROMISE:
      return IREE_SV("importer_promise");
    case LOOM_SANITIZER_PROVENANCE_KIND_FAST_MATH:
      return IREE_SV("fast_math");
    case LOOM_SANITIZER_PROVENANCE_KIND_ABI_PROMISE:
      return IREE_SV("abi_promise");
    case LOOM_SANITIZER_PROVENANCE_KIND_OPTIMIZATION_OBLIGATION:
      return IREE_SV("optimization_obligation");
    default:
      return iree_string_view_empty();
  }
}

iree_string_view_t loom_sanitizer_lane_policy_name(
    loom_sanitizer_lane_policy_t lane_policy) {
  switch (lane_policy) {
    case LOOM_SANITIZER_LANE_POLICY_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_SANITIZER_LANE_POLICY_SCALAR:
      return IREE_SV("scalar");
    case LOOM_SANITIZER_LANE_POLICY_PER_LANE:
      return IREE_SV("per_lane");
    case LOOM_SANITIZER_LANE_POLICY_ANY_LANE:
      return IREE_SV("any_lane");
    case LOOM_SANITIZER_LANE_POLICY_ALL_LANES:
      return IREE_SV("all_lanes");
    default:
      return iree_string_view_empty();
  }
}

iree_string_view_t loom_sanitizer_lineage_role_name(
    loom_sanitizer_lineage_role_t lineage_role) {
  switch (lineage_role) {
    case LOOM_SANITIZER_LINEAGE_ROLE_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL:
      return IREE_SV("original");
    case LOOM_SANITIZER_LINEAGE_ROLE_CLONE:
      return IREE_SV("clone");
    case LOOM_SANITIZER_LINEAGE_ROLE_TEMPLATE_INSTANTIATION:
      return IREE_SV("template_instantiation");
    case LOOM_SANITIZER_LINEAGE_ROLE_TILE_LOWERING:
      return IREE_SV("tile_lowering");
    case LOOM_SANITIZER_LINEAGE_ROLE_UKERNEL_SELECTION:
      return IREE_SV("ukernel_selection");
    default:
      return iree_string_view_empty();
  }
}

iree_status_t loom_sanitizer_site_payload_encode(
    const loom_sanitizer_site_payload_t* payload, iree_byte_span_t storage,
    iree_host_size_t* out_encoded_length) {
  if (out_encoded_length != NULL) {
    *out_encoded_length = 0;
  }
  if (payload == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sanitizer site payload is required");
  }
  if (out_encoded_length == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoded length output is required");
  }
  if (payload->extension_data.data_length > 0 &&
      payload->extension_data.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "extension data pointer is required");
  }
  iree_host_size_t required_length = 0;
  if (!iree_host_size_checked_add(LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH,
                                  payload->extension_data.data_length,
                                  &required_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site payload length overflow");
  }
  if (storage.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoding storage is required");
  }
  if (storage.data_length < required_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site payload requires %" PRIhsz
                            " bytes but only %" PRIhsz " bytes were provided",
                            required_length, storage.data_length);
  }

  storage.data[0] = LOOM_SANITIZER_SITE_PAYLOAD_CURRENT_VERSION;
  storage.data[1] = payload->site_kind;
  storage.data[2] = payload->check_kind;
  storage.data[3] = payload->provenance_kind;
  storage.data[4] = payload->lane_policy;
  storage.data[5] = payload->lineage_role;
  storage.data[6] = (uint8_t)(payload->flags & 0xFFu);
  storage.data[7] = (uint8_t)(payload->flags >> 8);
  if (payload->extension_data.data_length > 0) {
    memcpy(storage.data + LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH,
           payload->extension_data.data, payload->extension_data.data_length);
  }
  *out_encoded_length = required_length;
  return iree_ok_status();
}

iree_status_t loom_sanitizer_site_payload_decode(
    iree_const_byte_span_t data, loom_sanitizer_site_payload_t* out_payload) {
  if (out_payload == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sanitizer site payload output is required");
  }
  memset(out_payload, 0, sizeof(*out_payload));
  if (data.data_length < LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "sanitizer site payload requires at least %u bytes but only %" PRIhsz
        " bytes were provided",
        LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH, data.data_length);
  }
  if (data.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sanitizer site payload data is required");
  }

  const uint8_t version = data.data[0];
  if (version != LOOM_SANITIZER_SITE_PAYLOAD_CURRENT_VERSION) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported sanitizer site payload version %u",
                            version);
  }

  out_payload->site_kind = data.data[1];
  out_payload->check_kind = data.data[2];
  out_payload->provenance_kind = data.data[3];
  out_payload->lane_policy = data.data[4];
  out_payload->lineage_role = data.data[5];
  out_payload->flags =
      (loom_sanitizer_site_flags_t)data.data[6] |
      (loom_sanitizer_site_flags_t)((loom_sanitizer_site_flags_t)data.data[7]
                                    << 8);
  out_payload->extension_data = iree_make_const_byte_span(
      data.data + LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH,
      data.data_length - LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH);
  return iree_ok_status();
}
