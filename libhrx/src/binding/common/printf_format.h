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

#include "iree/base/api.h"

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

// Parses the supported C printf grammar. Device-side scalar values are carried
// in eight-byte slots, so long double, wide strings, and %n cannot be mapped to
// a safe host-side vararg representation.
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
    ++index;
  } else {
    while (index < conversion_index && spec[index] >= '0' &&
           spec[index] <= '9') {
      ++index;
    }
  }
  if (index < conversion_index && spec[index] == '.') {
    ++index;
    if (index < conversion_index && spec[index] == '*') {
      ++out_spec->star_count;
      ++index;
    } else {
      while (index < conversion_index && spec[index] >= '0' &&
             spec[index] <= '9') {
        ++index;
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

#endif  // IREE_HAL_STREAMING_PRINTF_FORMAT_H_
