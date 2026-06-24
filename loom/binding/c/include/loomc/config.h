// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_CONFIG_H_
#define LOOMC_CONFIG_H_

#include "loomc/base.h"

/// @file
/// Per-invocation configuration bindings.
///
/// Configuration is intentionally represented as plain borrowed key/value data
/// plus one optional JSON/JSONC object string. Operation descriptors decide how
/// strictly bindings are validated and whether unresolved configuration is
/// allowed to remain in the produced result.
///
/// @par Example
/// Use JSON for a framework-provided configuration file and explicit bindings
/// for programmatic overrides:
///
/// @code{.c}
/// const char* json_config = "{\"tile_m\":\"128\",\"tile_n\":\"64\"}";
/// loomc_config_binding_t overrides[] = {
///     {
///         .key = loomc_make_cstring_view("tile_m"),
///         .value = loomc_make_cstring_view("256"),
///     },
/// };
/// loomc_config_options_t config = {
///     .bindings = overrides,
///     .binding_count = 1,
///     .json_object = loomc_make_cstring_view(json_config),
///     .flags = LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
///              LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
/// };
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Single configuration key/value binding.
///
/// @lifetime
/// Binding strings are borrowed by the descriptor that contains them unless an
/// operation explicitly documents that it copies the bindings.
typedef struct loomc_config_binding_t {
  /// Config key, with a leading `@` accepted by normalization.
  loomc_string_view_t key;

  /// Config value spelling.
  loomc_string_view_t value;
} loomc_config_binding_t;

/// Configuration validation and resolution policy bit values.
typedef enum loomc_config_policy_flag_bits_e {
  /// Reject config bindings outside the current operation's known config
  /// declaration set. Link operations validate this before final pruning so
  /// declared-but-unused bindings may be accepted and ignored.
  LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN = 1u << 0,

  /// Require all final operation config values to be resolved.
  LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED = 1u << 1,
} loomc_config_policy_flag_bits_t;

/// Bitmask of `loomc_config_policy_flag_bits_t` values.
typedef uint32_t loomc_config_policy_flags_t;

/// Configuration options attached to a link or compile invocation.
///
/// The plain binding array and optional JSON/JSONC object string normalize into
/// the same key/value binding set. When both spellings define the same key, the
/// explicit `bindings` entry overrides the JSON object entry.
///
/// @lifetime
/// All views and arrays are borrowed for the duration of the API call unless
/// the consuming operation explicitly documents a longer copy.
typedef struct loomc_config_options_t {
  /// Plain key/value bindings.
  const loomc_config_binding_t* bindings;

  /// Number of entries in `bindings`.
  loomc_host_size_t binding_count;

  /// Optional JSON/JSONC object string normalized into the same binding set.
  loomc_string_view_t json_object;

  /// Config validation and resolution policy flags.
  loomc_config_policy_flags_t flags;
} loomc_config_options_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_CONFIG_H_
