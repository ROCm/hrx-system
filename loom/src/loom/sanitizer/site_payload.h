// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SANITIZER_SITE_PAYLOAD_H_
#define LOOM_SANITIZER_SITE_PAYLOAD_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compact metadata carried by LOOM_LOCATION_TAG_SANITIZER_SITE locations.
//
// The payload identifies what a sanitizer site means. It deliberately does
// not contain the final emitted site ID, an SSA value ID, a raw pointer, a
// string table reference, or target ABI data. Emission-time site collectors can
// assign dense target-local IDs from the surviving sanitizer operations and use
// this payload as optional diagnostic context when it is present.

// Format version byte written by the sanitizer site payload encoder.
#define LOOM_SANITIZER_SITE_PAYLOAD_CURRENT_VERSION 0u

// Number of fixed header bytes before optional extension data.
#define LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH 8u

enum loom_sanitizer_site_kind_e {
  // No known sanitizer site category.
  LOOM_SANITIZER_SITE_KIND_UNKNOWN = 0,
  // Memory access assertion.
  LOOM_SANITIZER_SITE_KIND_ACCESS = 1,
  // SSA value assertion.
  LOOM_SANITIZER_SITE_KIND_VALUE = 2,
  // Operation-level assertion.
  LOOM_SANITIZER_SITE_KIND_OPERATION = 3,
  // View layout, shape, or encoding assertion.
  LOOM_SANITIZER_SITE_KIND_LAYOUT = 4,
  // Race observation site.
  LOOM_SANITIZER_SITE_KIND_RACE = 5,
  // Number of known sanitizer site categories.
  LOOM_SANITIZER_SITE_KIND_COUNT_,
};
// One-byte sanitizer site category carried in a sanitizer site payload.
typedef uint8_t loom_sanitizer_site_kind_t;

enum loom_sanitizer_check_kind_e {
  // No known checked contract.
  LOOM_SANITIZER_CHECK_KIND_UNKNOWN = 0,
  // Access must stay within the allocation or view extent.
  LOOM_SANITIZER_CHECK_KIND_ACCESS_RANGE = 1,
  // Access address must satisfy a byte-alignment contract.
  LOOM_SANITIZER_CHECK_KIND_ACCESS_ALIGNMENT = 2,
  // Value must stay within a numeric range.
  LOOM_SANITIZER_CHECK_KIND_VALUE_RANGE = 3,
  // Floating-point value must not be NaN.
  LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_NAN = 4,
  // Value must be divisible by a numeric divisor.
  LOOM_SANITIZER_CHECK_KIND_VALUE_DIVISIBILITY = 5,
  // Integer arithmetic must not overflow.
  LOOM_SANITIZER_CHECK_KIND_INTEGER_OVERFLOW = 6,
  // Division divisor must be non-zero.
  LOOM_SANITIZER_CHECK_KIND_DIVIDE_BY_ZERO = 7,
  // Shift count must be in range for the shifted type.
  LOOM_SANITIZER_CHECK_KIND_INVALID_SHIFT = 8,
  // View must satisfy a refined layout, shape, or encoding contract.
  LOOM_SANITIZER_CHECK_KIND_LAYOUT_REFINEMENT = 9,
  // Floating-point value must not be infinity.
  LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_INF = 10,
  // Floating-point value must be finite.
  LOOM_SANITIZER_CHECK_KIND_VALUE_FINITE = 11,
  // Integer value must be a power of two.
  LOOM_SANITIZER_CHECK_KIND_VALUE_POWER_OF_TWO = 12,
  // Value must satisfy a relation against another value or constant.
  LOOM_SANITIZER_CHECK_KIND_VALUE_RELATION = 13,
  // Value must satisfy multiple or generic predicate constraints.
  LOOM_SANITIZER_CHECK_KIND_VALUE_CONSTRAINTS = 14,
  // Memory access participates in a data race.
  LOOM_SANITIZER_CHECK_KIND_DATA_RACE = 15,
  // Number of known checked contracts.
  LOOM_SANITIZER_CHECK_KIND_COUNT_,
};
// One-byte checked contract carried in a sanitizer site payload.
typedef uint8_t loom_sanitizer_check_kind_t;

enum loom_sanitizer_provenance_kind_e {
  // No known assertion provenance.
  LOOM_SANITIZER_PROVENANCE_KIND_UNKNOWN = 0,
  // Contract came from a user-authored assertion.
  LOOM_SANITIZER_PROVENANCE_KIND_USER_ASSERTION = 1,
  // Contract came from an assume-like source fact.
  LOOM_SANITIZER_PROVENANCE_KIND_ASSUME = 2,
  // Contract came from compiler analysis.
  LOOM_SANITIZER_PROVENANCE_KIND_ANALYSIS = 3,
  // Contract came from an internal compiler or target requirement.
  LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT = 4,
  // Contract came from an importer-owned source program promise.
  LOOM_SANITIZER_PROVENANCE_KIND_IMPORTER_PROMISE = 5,
  // Contract came from a floating-point fast-math flag.
  LOOM_SANITIZER_PROVENANCE_KIND_FAST_MATH = 6,
  // Contract came from a kernel or target ABI requirement.
  LOOM_SANITIZER_PROVENANCE_KIND_ABI_PROMISE = 7,
  // Contract came from a promise required by an optimization.
  LOOM_SANITIZER_PROVENANCE_KIND_OPTIMIZATION_OBLIGATION = 8,
  // Number of known provenance kinds.
  LOOM_SANITIZER_PROVENANCE_KIND_COUNT_,
};
// One-byte origin of the checked contract carried in a sanitizer site payload.
typedef uint8_t loom_sanitizer_provenance_kind_t;

enum loom_sanitizer_lane_policy_e {
  // No known lane interpretation.
  LOOM_SANITIZER_LANE_POLICY_UNKNOWN = 0,
  // Assertion is scalar and has no lane dimension.
  LOOM_SANITIZER_LANE_POLICY_SCALAR = 1,
  // Assertion applies independently to each lane.
  LOOM_SANITIZER_LANE_POLICY_PER_LANE = 2,
  // Assertion reports when any lane violates the contract.
  LOOM_SANITIZER_LANE_POLICY_ANY_LANE = 3,
  // Assertion reports when all lanes violate the contract.
  LOOM_SANITIZER_LANE_POLICY_ALL_LANES = 4,
  // Number of known lane policies.
  LOOM_SANITIZER_LANE_POLICY_COUNT_,
};
// One-byte lane interpretation carried in a sanitizer site payload.
typedef uint8_t loom_sanitizer_lane_policy_t;

enum loom_sanitizer_lineage_role_e {
  // No known transform lineage role.
  LOOM_SANITIZER_LINEAGE_ROLE_UNKNOWN = 0,
  // Original assertion site before cloning or specialization.
  LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL = 1,
  // Site cloned from another assertion site.
  LOOM_SANITIZER_LINEAGE_ROLE_CLONE = 2,
  // Site created while instantiating a template.
  LOOM_SANITIZER_LINEAGE_ROLE_TEMPLATE_INSTANTIATION = 3,
  // Site created while lowering tile-level IR.
  LOOM_SANITIZER_LINEAGE_ROLE_TILE_LOWERING = 4,
  // Site created while selecting or importing a ukernel path.
  LOOM_SANITIZER_LINEAGE_ROLE_UKERNEL_SELECTION = 5,
  // Number of known transform lineage roles.
  LOOM_SANITIZER_LINEAGE_ROLE_COUNT_,
};
// One-byte transform lineage summary carried in a sanitizer site payload.
typedef uint8_t loom_sanitizer_lineage_role_t;

// Two-byte flag set carried in a sanitizer site payload.
typedef uint16_t loom_sanitizer_site_flags_t;

typedef struct loom_sanitizer_site_payload_t {
  // Sanitizer site category.
  loom_sanitizer_site_kind_t site_kind;
  // Specific checked contract.
  loom_sanitizer_check_kind_t check_kind;
  // Origin of the checked contract.
  loom_sanitizer_provenance_kind_t provenance_kind;
  // Lane/reporting policy.
  loom_sanitizer_lane_policy_t lane_policy;
  // Transform lineage summary.
  loom_sanitizer_lineage_role_t lineage_role;
  // Opaque flag bits preserved by the codec.
  loom_sanitizer_site_flags_t flags;
  // Optional extension bytes following the fixed header.
  iree_const_byte_span_t extension_data;
} loom_sanitizer_site_payload_t;

// Returns the known name for |site_kind|, or an empty string for values
// unknown to this compiler.
iree_string_view_t loom_sanitizer_site_kind_name(
    loom_sanitizer_site_kind_t site_kind);

// Returns the known name for |check_kind|, or an empty string for values
// unknown to this compiler.
iree_string_view_t loom_sanitizer_check_kind_name(
    loom_sanitizer_check_kind_t check_kind);

// Returns the known name for |provenance_kind|, or an empty string for values
// unknown to this compiler.
iree_string_view_t loom_sanitizer_provenance_kind_name(
    loom_sanitizer_provenance_kind_t provenance_kind);

// Returns the known name for |lane_policy|, or an empty string for values
// unknown to this compiler.
iree_string_view_t loom_sanitizer_lane_policy_name(
    loom_sanitizer_lane_policy_t lane_policy);

// Returns the known name for |lineage_role|, or an empty string for values
// unknown to this compiler.
iree_string_view_t loom_sanitizer_lineage_role_name(
    loom_sanitizer_lineage_role_t lineage_role);

// Encodes a sanitizer site payload into |storage|.
//
// The first byte stores LOOM_SANITIZER_SITE_PAYLOAD_CURRENT_VERSION. The next
// five bytes store site kind, check kind, provenance kind, lane policy,
// and lineage role. Bytes six and seven store little-endian flag bits. Any
// remaining bytes are uninterpreted extension data owned by the caller.
iree_status_t loom_sanitizer_site_payload_encode(
    const loom_sanitizer_site_payload_t* payload, iree_byte_span_t storage,
    iree_host_size_t* out_encoded_length);

// Decodes a sanitizer site payload.
//
// Unknown field values and flag bits are preserved as raw bytes so newer
// assertion producers can still carry diagnostic metadata through older tools
// that do not need to interpret it.
iree_status_t loom_sanitizer_site_payload_decode(
    iree_const_byte_span_t data, loom_sanitizer_site_payload_t* out_payload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SANITIZER_SITE_PAYLOAD_H_
