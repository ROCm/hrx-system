// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/printf_format.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static iree_host_size_t iree_hal_streaming_printf_strnlen(
    const char* data, iree_host_size_t data_length) {
  iree_host_size_t length = 0;
  while (length < data_length && data[length] != '\0') ++length;
  return length;
}

static iree_host_size_t iree_hal_streaming_printf_align8(
    iree_host_size_t value) {
  return (value + 7u) & ~(iree_host_size_t)7u;
}

iree_status_t iree_hal_streaming_printf_parse_metadata_record(
    iree_string_view_t record, uint64_t* out_hash,
    iree_string_view_t* out_format) {
  *out_hash = 0;
  *out_format = iree_string_view_empty();

  if (!iree_string_view_consume_prefix(&record, IREE_SV("0:0:"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "printf metadata record must start with fixed `0:0:` fields");
  }
  iree_host_size_t comma = iree_string_view_find_char(record, ',', 0);
  if (comma == IREE_STRING_VIEW_NPOS || comma == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "printf metadata record is missing hash/format delimiter");
  }
  iree_string_view_t hash_text = iree_string_view_substr(record, 0, comma);
  if (hash_text.size > 16) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "printf metadata hash exceeds 64 bits");
  }
  uint64_t hash = 0;
  if (!iree_string_view_atoi_uint64_base(hash_text, 16, &hash)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "printf metadata hash is not hexadecimal");
  }

  *out_hash = hash;
  *out_format =
      iree_string_view_substr(record, comma + 1, IREE_STRING_VIEW_NPOS);
  return iree_ok_status();
}

static bool iree_hal_streaming_printf_write(FILE* stream, int* out_count,
                                            const char* format, ...) {
  va_list args;
  va_start(args, format);
  const int result = vfprintf(stream, format, args);
  va_end(args);
  if (result < 0) {
    *out_count = result;
    return false;
  }
  if (*out_count <= INT_MAX - result) {
    *out_count += result;
  } else {
    *out_count = INT_MAX;
  }
  return true;
}

static bool iree_hal_streaming_printf_write_bytes(FILE* stream, int* out_count,
                                                  const char* data,
                                                  iree_host_size_t length) {
  if (length == 0) return true;
  if (IREE_UNLIKELY(length > (iree_host_size_t)INT_MAX)) {
    *out_count = -1;
    return false;
  }
  if (fwrite(data, 1, length, stream) != length) {
    *out_count = -1;
    return false;
  }
  if (*out_count <= INT_MAX - (int)length) {
    *out_count += (int)length;
  } else {
    *out_count = INT_MAX;
  }
  return true;
}

static const uint8_t* iree_hal_streaming_printf_read_u64(
    const uint8_t* argument, const uint8_t* argument_end, uint64_t* out_value) {
  if (IREE_UNLIKELY((iree_host_size_t)(argument_end - argument) <
                    sizeof(*out_value))) {
    return NULL;
  }
  memcpy(out_value, argument, sizeof(*out_value));
  return argument + sizeof(*out_value);
}

static bool iree_hal_streaming_printf_field_lengths_are_valid(
    const iree_hal_streaming_printf_spec_t* spec, int width, int precision) {
  if (spec->has_width_star &&
      (width == INT_MIN ||
       abs(width) > IREE_HAL_STREAMING_PRINTF_MAX_FIELD_LENGTH)) {
    return false;
  }
  // A negative dynamic precision means that no precision was specified.
  if (spec->has_precision_star && precision >= 0 &&
      precision > IREE_HAL_STREAMING_PRINTF_MAX_FIELD_LENGTH) {
    return false;
  }
  return true;
}

static const uint8_t* iree_hal_streaming_printf_process_spec(
    FILE* stream, int* out_count, const char* spec_begin,
    iree_host_size_t spec_length, const uint8_t* argument,
    const uint8_t* argument_end) {
  if (IREE_UNLIKELY(spec_length >= 128)) return NULL;
  char spec[128];
  memcpy(spec, spec_begin, spec_length);
  spec[spec_length] = '\0';

  iree_hal_streaming_printf_spec_t parsed_spec;
  if (IREE_UNLIKELY(!iree_hal_streaming_printf_parse_spec(spec, spec_length,
                                                          &parsed_spec))) {
    return NULL;
  }

  const int star_count = parsed_spec.star_count;
  int star0 = 0;
  int star1 = 0;
  uint64_t raw_value = 0;
  if (star_count >= 1) {
    argument =
        iree_hal_streaming_printf_read_u64(argument, argument_end, &raw_value);
    if (!argument) return NULL;
    star0 = (int)raw_value;
  }
  if (star_count >= 2) {
    argument =
        iree_hal_streaming_printf_read_u64(argument, argument_end, &raw_value);
    if (!argument) return NULL;
    star1 = (int)raw_value;
  }
  const int width = parsed_spec.has_width_star ? star0 : 0;
  const int precision = parsed_spec.has_precision_star
                            ? (parsed_spec.has_width_star ? star1 : star0)
                            : 0;
  if (IREE_UNLIKELY(!iree_hal_streaming_printf_field_lengths_are_valid(
          &parsed_spec, width, precision))) {
    return NULL;
  }

  if (parsed_spec.conversion == 'n') {
    return iree_hal_streaming_printf_read_u64(argument, argument_end,
                                              &raw_value);
  }

#define IREE_HAL_STREAMING_PRINTF_CALL(value)                                 \
  do {                                                                        \
    if (star_count == 0) {                                                    \
      iree_hal_streaming_printf_write(stream, out_count, spec, value);        \
    } else if (star_count == 1) {                                             \
      iree_hal_streaming_printf_write(stream, out_count, spec, star0, value); \
    } else {                                                                  \
      iree_hal_streaming_printf_write(stream, out_count, spec, star0, star1,  \
                                      value);                                 \
    }                                                                         \
  } while (0)

  switch (parsed_spec.conversion) {
    case 'd':
    case 'i':
    case 'o':
    case 'u':
    case 'x':
    case 'X': {
      argument = iree_hal_streaming_printf_read_u64(argument, argument_end,
                                                    &raw_value);
      if (!argument) return NULL;
      if (parsed_spec.is_signed_integer) {
        switch (parsed_spec.length_modifier) {
          case IREE_HAL_STREAMING_PRINTF_LENGTH_DEFAULT:
          case IREE_HAL_STREAMING_PRINTF_LENGTH_H:
          case IREE_HAL_STREAMING_PRINTF_LENGTH_HH: {
            const int value = (int)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_L: {
            const long value = (long)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_LL: {
            const long long value = (long long)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_J: {
            const intmax_t value = (intmax_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_Z: {
            const iree_hal_streaming_printf_signed_size_t value =
                (iree_hal_streaming_printf_signed_size_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_T: {
            const ptrdiff_t value = (ptrdiff_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
        }
      } else {
        switch (parsed_spec.length_modifier) {
          case IREE_HAL_STREAMING_PRINTF_LENGTH_DEFAULT:
          case IREE_HAL_STREAMING_PRINTF_LENGTH_H:
          case IREE_HAL_STREAMING_PRINTF_LENGTH_HH: {
            const unsigned int value = (unsigned int)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_L: {
            const unsigned long value = (unsigned long)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_LL: {
            const unsigned long long value = (unsigned long long)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_J: {
            const uintmax_t value = (uintmax_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_Z: {
            const size_t value = (size_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_T: {
            const iree_hal_streaming_printf_unsigned_ptrdiff_t value =
                (iree_hal_streaming_printf_unsigned_ptrdiff_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
        }
      }
      break;
    }
    case 'c': {
      argument = iree_hal_streaming_printf_read_u64(argument, argument_end,
                                                    &raw_value);
      if (!argument) return NULL;
      if (parsed_spec.is_wide_character) {
        const wint_t value = (wint_t)raw_value;
        IREE_HAL_STREAMING_PRINTF_CALL(value);
      } else {
        const int value = (int)raw_value;
        IREE_HAL_STREAMING_PRINTF_CALL(value);
      }
      break;
    }
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A': {
      argument = iree_hal_streaming_printf_read_u64(argument, argument_end,
                                                    &raw_value);
      if (!argument) return NULL;
      double value = 0;
      memcpy(&value, &raw_value, sizeof(value));
      IREE_HAL_STREAMING_PRINTF_CALL(value);
      break;
    }
    case 's': {
      const iree_host_size_t available_bytes =
          (iree_host_size_t)(argument_end - argument);
      const iree_host_size_t string_length = iree_hal_streaming_printf_strnlen(
          (const char*)argument, available_bytes);
      if (IREE_UNLIKELY(string_length == available_bytes)) return NULL;
      const iree_host_size_t occupied_length =
          iree_hal_streaming_printf_align8(string_length + 1);
      if (IREE_UNLIKELY(occupied_length > available_bytes)) return NULL;
      const char* value = (const char*)argument;
      IREE_HAL_STREAMING_PRINTF_CALL(value);
      argument += occupied_length;
      break;
    }
    case 'p': {
      argument = iree_hal_streaming_printf_read_u64(argument, argument_end,
                                                    &raw_value);
      if (!argument) return NULL;
      void* value = (void*)(uintptr_t)raw_value;
      IREE_HAL_STREAMING_PRINTF_CALL(value);
      break;
    }
    default:
      return NULL;
  }

#undef IREE_HAL_STREAMING_PRINTF_CALL

  return argument;
}

iree_status_t iree_hal_streaming_printf_format(FILE* stream,
                                               iree_string_view_t format,
                                               const uint8_t* arguments,
                                               iree_host_size_t argument_length,
                                               int* out_count) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;
  uint8_t empty_arguments = 0;
  if (!arguments) arguments = &empty_arguments;
  const uint8_t* argument = arguments;
  const uint8_t* argument_end = arguments + argument_length;
  iree_host_size_t cursor = 0;
  while (cursor < format.size) {
    const char* percent =
        memchr(format.data + cursor, '%', format.size - cursor);
    if (!percent) {
      if (!iree_hal_streaming_printf_write_bytes(
              stream, out_count, format.data + cursor, format.size - cursor)) {
        return iree_make_status(IREE_STATUS_UNKNOWN,
                                "device printf output write failed");
      }
      return iree_ok_status();
    }

    const iree_host_size_t percent_offset =
        (iree_host_size_t)(percent - format.data);
    if (!iree_hal_streaming_printf_write_bytes(
            stream, out_count, format.data + cursor, percent_offset - cursor)) {
      return iree_make_status(IREE_STATUS_UNKNOWN,
                              "device printf output write failed");
    }

    iree_host_size_t spec_end = percent_offset + 1;
    if (spec_end < format.size && format.data[spec_end] == '%') {
      if (!iree_hal_streaming_printf_write_bytes(stream, out_count, "%", 1)) {
        return iree_make_status(IREE_STATUS_UNKNOWN,
                                "device printf output write failed");
      }
      cursor = spec_end + 1;
      continue;
    }

    static const char kConversionSpecifiers[] = "diouxXfFeEgGaAcspn";
    while (spec_end < format.size &&
           !strchr(kConversionSpecifiers, format.data[spec_end])) {
      ++spec_end;
    }
    if (IREE_UNLIKELY(spec_end == format.size)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device printf format is unterminated");
    }
    ++spec_end;

    argument = iree_hal_streaming_printf_process_spec(
        stream, out_count, percent, spec_end - percent_offset, argument,
        argument_end);
    if (IREE_UNLIKELY(*out_count < 0 || !argument)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device printf arguments are malformed");
    }
    cursor = spec_end;
  }
  return iree_ok_status();
}
