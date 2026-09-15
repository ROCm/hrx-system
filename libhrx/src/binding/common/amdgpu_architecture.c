// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/amdgpu_architecture.h"

#include <stddef.h>
#include <string.h>

static bool iree_hal_streaming_parse_decimal_digit(char value,
                                                   uint32_t* out_digit) {
  if (value < '0' || value > '9') return false;
  *out_digit = (uint32_t)(value - '0');
  return true;
}

static bool iree_hal_streaming_parse_hex_digit(char value,
                                               uint32_t* out_digit) {
  if (iree_hal_streaming_parse_decimal_digit(value, out_digit)) return true;
  if (value >= 'a' && value <= 'f') {
    *out_digit = (uint32_t)(value - 'a' + 10);
    return true;
  }
  if (value >= 'A' && value <= 'F') {
    *out_digit = (uint32_t)(value - 'A' + 10);
    return true;
  }
  return false;
}

static bool iree_hal_streaming_is_supported_target_feature(const char* value,
                                                           size_t length) {
  return (length == strlen("sramecc") &&
          memcmp(value, "sramecc", length) == 0) ||
         (length == strlen("xnack") && memcmp(value, "xnack", length) == 0);
}

static bool iree_hal_streaming_validate_target_features(const char* value) {
  while (*value != '\0') {
    if (*value++ != ':') return false;
    const char* feature = value;
    while (*value != '\0' && *value != ':' && *value != '+' && *value != '-') {
      ++value;
    }
    const size_t feature_length = (size_t)(value - feature);
    if (!iree_hal_streaming_is_supported_target_feature(feature,
                                                        feature_length) ||
        (*value != '+' && *value != '-')) {
      return false;
    }
    ++value;
    if (*value != '\0' && *value != ':') return false;
  }
  return true;
}

bool iree_hal_streaming_parse_amdgpu_architecture(
    const char* value,
    iree_hal_streaming_amdgpu_architecture_t* out_architecture) {
  if (!value || !out_architecture || strncmp(value, "gfx", 3) != 0) {
    return false;
  }

  const char* processor = value + 3;
  const char* suffix = strchr(processor, ':');
  const size_t processor_length =
      suffix ? (size_t)(suffix - processor) : strlen(processor);
  iree_hal_streaming_amdgpu_architecture_t architecture = {0};
  uint32_t major_digit = 0;
  if (processor_length == 3 &&
      iree_hal_streaming_parse_decimal_digit(processor[0], &major_digit) &&
      iree_hal_streaming_parse_decimal_digit(processor[1],
                                             &architecture.minor) &&
      iree_hal_streaming_parse_hex_digit(processor[2],
                                         &architecture.stepping)) {
    architecture.major = major_digit;
  } else if (processor_length == 4 && processor[0] == '1' &&
             iree_hal_streaming_parse_decimal_digit(processor[1],
                                                    &major_digit) &&
             iree_hal_streaming_parse_decimal_digit(processor[2],
                                                    &architecture.minor) &&
             iree_hal_streaming_parse_hex_digit(processor[3],
                                                &architecture.stepping)) {
    architecture.major = 10 + major_digit;
  } else {
    return false;
  }
  if (suffix && !iree_hal_streaming_validate_target_features(suffix)) {
    return false;
  }

  *out_architecture = architecture;
  return true;
}
