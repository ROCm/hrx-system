// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_PRINTF_FORMAT_H_
#define IREE_HAL_STREAMING_PRINTF_FORMAT_H_

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum field width or precision accepted from a device printf record.
// Device messages are untrusted runtime input; bounding field expansion keeps
// a small corrupt record from causing effectively unbounded host output.
#define IREE_HAL_STREAMING_PRINTF_MAX_FIELD_LENGTH (1024 * 1024)

// Length modifier carried by one supported printf conversion.
typedef enum iree_hal_streaming_printf_length_modifier_e {
  IREE_HAL_STREAMING_PRINTF_LENGTH_DEFAULT = 0,
  IREE_HAL_STREAMING_PRINTF_LENGTH_HH,
  IREE_HAL_STREAMING_PRINTF_LENGTH_H,
  IREE_HAL_STREAMING_PRINTF_LENGTH_L,
  IREE_HAL_STREAMING_PRINTF_LENGTH_LL,
  IREE_HAL_STREAMING_PRINTF_LENGTH_J,
  IREE_HAL_STREAMING_PRINTF_LENGTH_Z,
  IREE_HAL_STREAMING_PRINTF_LENGTH_T,
} iree_hal_streaming_printf_length_modifier_t;

// Parsed printf conversion that can be safely materialized from a device ABI
// scalar slot and passed to vfprintf.
typedef struct iree_hal_streaming_printf_spec_t {
  // Terminal conversion character, such as d, s, or f.
  char conversion;
  // Parsed length modifier affecting the host vararg type.
  iree_hal_streaming_printf_length_modifier_t length_modifier;
  // Number of width and precision arguments supplied with *.
  uint8_t star_count;
  // True when the field width is supplied by an argument.
  bool has_width_star;
  // True when the field precision is supplied by an argument.
  bool has_precision_star;
  // True when the conversion consumes a signed integer argument.
  bool is_signed_integer;
  // True when the conversion consumes a wide character argument.
  bool is_wide_character;
} iree_hal_streaming_printf_spec_t;

static inline bool iree_hal_streaming_printf_is_flag(char c) {
  switch (c) {
    case '#':
    case '0':
    case '-':
    case ' ':
    case '+':
    case '\'':
      return true;
    default:
      return false;
  }
}

static inline bool iree_hal_streaming_printf_parse_field_length(
    const char* spec, iree_host_size_t conversion_index,
    iree_host_size_t* index) {
  uint32_t value = 0;
  while (*index < conversion_index && spec[*index] >= '0' &&
         spec[*index] <= '9') {
    const uint32_t digit = (uint32_t)(spec[*index] - '0');
    if (value > (IREE_HAL_STREAMING_PRINTF_MAX_FIELD_LENGTH - digit) / 10u) {
      return false;
    }
    value = value * 10u + digit;
    ++*index;
  }
  return true;
}

// Parses the supported C printf grammar. Device-side scalar values are carried
// in eight-byte slots. Long double and wide strings cannot be mapped to a safe
// host-side vararg representation; %n consumes its pointer slot without
// dereferencing it.
static inline bool iree_hal_streaming_printf_parse_spec(
    const char* spec, iree_host_size_t spec_length,
    iree_hal_streaming_printf_spec_t* out_spec) {
  if (!spec || !out_spec || spec_length < 2 || spec[0] != '%') return false;

  *out_spec = (iree_hal_streaming_printf_spec_t){0};
  const iree_host_size_t conversion_index = spec_length - 1;
  out_spec->conversion = spec[conversion_index];

  iree_host_size_t index = 1;
  while (index < conversion_index &&
         iree_hal_streaming_printf_is_flag(spec[index])) {
    ++index;
  }
  if (index < conversion_index && spec[index] == '*') {
    ++out_spec->star_count;
    out_spec->has_width_star = true;
    ++index;
  } else {
    if (!iree_hal_streaming_printf_parse_field_length(spec, conversion_index,
                                                      &index)) {
      return false;
    }
  }
  if (index < conversion_index && spec[index] == '.') {
    ++index;
    if (index < conversion_index && spec[index] == '*') {
      ++out_spec->star_count;
      out_spec->has_precision_star = true;
      ++index;
    } else {
      if (!iree_hal_streaming_printf_parse_field_length(spec, conversion_index,
                                                        &index)) {
        return false;
      }
    }
  }

  if (index < conversion_index) {
    switch (spec[index++]) {
      case 'h':
        if (index < conversion_index && spec[index] == 'h') {
          ++index;
          out_spec->length_modifier = IREE_HAL_STREAMING_PRINTF_LENGTH_HH;
        } else {
          out_spec->length_modifier = IREE_HAL_STREAMING_PRINTF_LENGTH_H;
        }
        break;
      case 'l':
        if (index < conversion_index && spec[index] == 'l') {
          ++index;
          out_spec->length_modifier = IREE_HAL_STREAMING_PRINTF_LENGTH_LL;
        } else {
          out_spec->length_modifier = IREE_HAL_STREAMING_PRINTF_LENGTH_L;
        }
        break;
      case 'j':
        out_spec->length_modifier = IREE_HAL_STREAMING_PRINTF_LENGTH_J;
        break;
      case 'z':
        out_spec->length_modifier = IREE_HAL_STREAMING_PRINTF_LENGTH_Z;
        break;
      case 't':
        out_spec->length_modifier = IREE_HAL_STREAMING_PRINTF_LENGTH_T;
        break;
      case 'L':
        return false;
      default:
        return false;
    }
  }
  if (index != conversion_index) return false;

  switch (out_spec->conversion) {
    case 'd':
    case 'i':
      out_spec->is_signed_integer = true;
      return true;
    case 'o':
    case 'u':
    case 'x':
    case 'X':
      return true;
    case 'c':
      if (out_spec->length_modifier == IREE_HAL_STREAMING_PRINTF_LENGTH_L) {
        out_spec->is_wide_character = true;
        return true;
      }
      return out_spec->length_modifier ==
             IREE_HAL_STREAMING_PRINTF_LENGTH_DEFAULT;
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A':
      return out_spec->length_modifier ==
                 IREE_HAL_STREAMING_PRINTF_LENGTH_DEFAULT ||
             out_spec->length_modifier == IREE_HAL_STREAMING_PRINTF_LENGTH_L;
    case 's':
    case 'p':
      return out_spec->length_modifier ==
             IREE_HAL_STREAMING_PRINTF_LENGTH_DEFAULT;
    case 'n':
      return true;
    default:
      return false;
  }
}

#if SIZE_MAX == UINT_MAX
typedef int iree_hal_streaming_printf_signed_size_t;
#elif SIZE_MAX == ULONG_MAX
typedef long iree_hal_streaming_printf_signed_size_t;
#elif SIZE_MAX == ULLONG_MAX
typedef long long iree_hal_streaming_printf_signed_size_t;
#else
#error "unsupported size_t representation for printf formatting"
#endif

#if PTRDIFF_MAX == INT_MAX
typedef unsigned int iree_hal_streaming_printf_unsigned_ptrdiff_t;
#elif PTRDIFF_MAX == LONG_MAX
typedef unsigned long iree_hal_streaming_printf_unsigned_ptrdiff_t;
#elif PTRDIFF_MAX == LLONG_MAX
typedef unsigned long long iree_hal_streaming_printf_unsigned_ptrdiff_t;
#else
#error "unsupported ptrdiff_t representation for printf formatting"
#endif

// Formats one device printf record. Scalar arguments use eight-byte slots and
// strings occupy the minimum whole number of slots including their terminator.
// The returned count follows printf semantics and excludes a trailing NUL.
iree_status_t iree_hal_streaming_printf_format(FILE* stream,
                                               iree_string_view_t format,
                                               const uint8_t* arguments,
                                               iree_host_size_t argument_length,
                                               int* out_count);

// Parses one amdhsa.printf metadata record into its hash and format string.
iree_status_t iree_hal_streaming_printf_parse_metadata_record(
    iree_string_view_t record, uint64_t* out_hash,
    iree_string_view_t* out_format);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STREAMING_PRINTF_FORMAT_H_
