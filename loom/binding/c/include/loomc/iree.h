// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_IREE_H_
#define LOOMC_IREE_H_

#include <stddef.h>

#include "iree/base/api.h"
#include "loomc/loomc.h"

/// @file
/// Optional header-only adapters between IREE base types and the Loom C API.
///
/// This header is for IREE-hosted tools and runtimes that already use IREE base
/// types. Core `loomc/*.h` headers remain independent from IREE. Including this
/// header verifies ABI compatibility and provides zero-copy conversions for
/// status, string view, byte span, and allocator values.
///
/// @par Example
/// Convert at the IREE/Loom boundary and preserve rich status payloads:
///
/// @code{.c}
/// iree_status_t create_source_from_iree(
///     iree_string_view_t identifier, iree_const_byte_span_t contents,
///     iree_allocator_t host_allocator, loomc_source_t** out_source) {
///   loomc_source_options_t options = {
///       .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
///       .structure_size = sizeof(loomc_source_options_t),
///       .format = LOOMC_SOURCE_FORMAT_TEXT,
///       .identifier = loomc_string_view_from_iree(identifier),
///       .contents = loomc_byte_span_from_iree(contents),
///       .storage = LOOMC_SOURCE_STORAGE_COPY,
///   };
///   loomc_status_t status = loomc_source_create(
///       &options, loomc_allocator_from_iree(host_allocator), out_source);
///   return iree_status_from_loomc(status);
/// }
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// @cond LOOMC_DOXYGEN_HIDDEN
#if defined(__cplusplus)
#define LOOMC_IREE_STATIC_ASSERT static_assert
#else
#define LOOMC_IREE_STATIC_ASSERT _Static_assert
#endif

LOOMC_IREE_STATIC_ASSERT(sizeof(loomc_string_view_t) ==
                             sizeof(iree_string_view_t),
                         "loomc_string_view_t must match iree_string_view_t");
LOOMC_IREE_STATIC_ASSERT(offsetof(loomc_string_view_t, data) ==
                             offsetof(iree_string_view_t, data),
                         "string view data fields must match");
LOOMC_IREE_STATIC_ASSERT(offsetof(loomc_string_view_t, size) ==
                             offsetof(iree_string_view_t, size),
                         "string view size fields must match");
LOOMC_IREE_STATIC_ASSERT(sizeof(loomc_byte_span_t) ==
                             sizeof(iree_const_byte_span_t),
                         "loomc_byte_span_t must match iree_const_byte_span_t");
LOOMC_IREE_STATIC_ASSERT(offsetof(loomc_byte_span_t, data) ==
                             offsetof(iree_const_byte_span_t, data),
                         "byte span data fields must match");
LOOMC_IREE_STATIC_ASSERT(offsetof(loomc_byte_span_t, data_length) ==
                             offsetof(iree_const_byte_span_t, data_length),
                         "byte span length fields must match");
LOOMC_IREE_STATIC_ASSERT((int)LOOMC_STATUS_OK == (int)IREE_STATUS_OK,
                         "status OK values must match");
LOOMC_IREE_STATIC_ASSERT((int)LOOMC_STATUS_INCOMPATIBLE ==
                             (int)IREE_STATUS_INCOMPATIBLE,
                         "status code values must match");
LOOMC_IREE_STATIC_ASSERT((int)LOOMC_STATUS_CODE_MASK ==
                             (int)IREE_STATUS_CODE_MASK,
                         "status code masks must match");
LOOMC_IREE_STATIC_ASSERT(LOOMC_STATUS_FEATURES == IREE_STATUS_FEATURES,
                         "status feature modes must match");
/// @endcond

/// Reinterprets an IREE status as a Loom status.
///
/// @param status IREE status to transfer.
/// @return Equivalent Loom status handle.
///
/// @ownership
/// Ownership transfers unchanged. The returned status is freed with
/// `loomc_status_free` or converted back to `iree_status_t` and handled by
/// IREE.
static inline loomc_status_t loomc_status_from_iree(iree_status_t status) {
  return (loomc_status_t)status;
}

/// Reinterprets a Loom status as an IREE status.
///
/// @param status Loom status to transfer.
/// @return Equivalent IREE status handle.
///
/// @ownership
/// Ownership transfers unchanged. The returned status is handled with IREE
/// status utilities or converted back to `loomc_status_t`.
static inline iree_status_t iree_status_from_loomc(loomc_status_t status) {
  return (iree_status_t)status;
}

/// Converts an IREE string view to a Loom string view.
///
/// @param value IREE string view.
/// @return Borrowed Loom view over the same bytes.
///
/// @lifetime
/// The returned view has the same lifetime as `value`.
static inline loomc_string_view_t loomc_string_view_from_iree(
    iree_string_view_t value) {
  return loomc_make_string_view(value.data, value.size);
}

/// Converts a Loom string view to an IREE string view.
///
/// @param value Loom string view.
/// @return Borrowed IREE view over the same bytes.
///
/// @lifetime
/// The returned view has the same lifetime as `value`.
static inline iree_string_view_t iree_string_view_from_loomc(
    loomc_string_view_t value) {
  return iree_make_string_view(value.data, value.size);
}

/// Converts an IREE immutable byte span to a Loom byte span.
///
/// @param value IREE byte span.
/// @return Borrowed Loom span over the same bytes.
///
/// @lifetime
/// The returned span has the same lifetime as `value`.
static inline loomc_byte_span_t loomc_byte_span_from_iree(
    iree_const_byte_span_t value) {
  return loomc_make_byte_span(value.data, value.data_length);
}

/// Converts a Loom byte span to an IREE immutable byte span.
///
/// @param value Loom byte span.
/// @return Borrowed IREE span over the same bytes.
///
/// @lifetime
/// The returned span has the same lifetime as `value`.
static inline iree_const_byte_span_t iree_const_byte_span_from_loomc(
    loomc_byte_span_t value) {
  return iree_make_const_byte_span(value.data, value.data_length);
}

/// Converts an IREE allocator to a Loom allocator.
///
/// @param allocator IREE allocator.
/// @return Loom allocator that forwards to the same callback and state.
///
/// @lifetime
/// The returned allocator is valid while the original allocator callback and
/// state remain valid.
static inline loomc_allocator_t loomc_allocator_from_iree(
    iree_allocator_t allocator) {
  return (loomc_allocator_t){
      .self = allocator.self,
      .ctl = (loomc_allocator_ctl_fn_t)allocator.ctl,
  };
}

/// Converts a Loom allocator to an IREE allocator.
///
/// @param allocator Valid Loom allocator.
/// @return IREE allocator that forwards to the same callback and state.
///
/// @lifetime
/// The returned allocator is valid while the original allocator callback and
/// state remain valid.
static inline iree_allocator_t iree_allocator_from_loomc(
    loomc_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(loomc_allocator_is_valid(allocator));
  return (iree_allocator_t){
      .self = allocator.self,
      .ctl = (iree_allocator_ctl_fn_t)allocator.ctl,
  };
}

#undef LOOMC_IREE_STATIC_ASSERT

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_IREE_H_
